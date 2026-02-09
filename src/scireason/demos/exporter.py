from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


from ..temporal.schemas import TemporalTriplet
from .schemas import DemoExample, DemoSource
from .store import upsert_demos


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"demo_{h}"


def _to_triplet(assertion: Dict[str, Any]) -> Dict[str, Any]:
    subj = str(assertion.get("subject") or "").strip()
    pred = str(assertion.get("predicate") or "").strip()
    obj = str(assertion.get("object") or "").strip()
    verdict = str(assertion.get("verdict") or "accepted").lower()

    confidence = 0.9 if verdict == "accepted" else 0.2 if verdict == "rejected" else 0.6
    polarity = "unknown"  # graph review typically does not encode support/contradict

    ev = assertion.get("evidence") or {}
    quote = str(ev.get("snippet_or_summary") or "")[:200]

    # NOTE: time in current schema is calendar-like. Expert artifacts may store "conditions" instead.
    # We keep time=null in demos; downstream pipeline will fallback to paper_year.
    return TemporalTriplet(
        subject=subj,
        predicate=pred,
        object=obj,
        confidence=confidence,
        polarity=polarity,
        time=None,
        evidence_quote=quote,
    ).model_dump()


def iter_graph_review_demos(graph_reviews_dir: Path) -> Iterator[DemoExample]:
    """Convert expert graph reviews into temporal_triplets demos."""
    for p in sorted(graph_reviews_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        domain = str(obj.get("domain") or "science")
        paper_id = str(obj.get("paper_id") or "")
        reviewer_id = str(obj.get("reviewer_id") or "")
        timestamp = str(obj.get("timestamp") or "")
        assertions = obj.get("assertions") or []
        if not isinstance(assertions, list):
            continue

        for a in assertions:
            aid = str(a.get("assertion_id") or "")
            ev = a.get("evidence") or {}
            # Minimal input text: use the evidence snippet + locator.
            loc = []
            if ev.get("page") is not None:
                loc.append(f"page={ev.get('page')}")
            if ev.get("figure_or_table"):
                loc.append(str(ev.get("figure_or_table")))
            loc = ", ".join(loc)
            snippet = str(ev.get("snippet_or_summary") or "")
            chunk_text = f"[{paper_id}] {loc}\n{snippet}".strip()

            demo = DemoExample(
                id=_stable_id("temporal_triplets", domain, paper_id, aid, p.as_posix()),
                task="temporal_triplets",
                domain=domain,
                quality="gold" if str(a.get("verdict")).lower() == "accepted" else "silver",
                source=DemoSource(artifact_path=str(p), reviewer_id=reviewer_id, timestamp=timestamp),
                tags=[str(ev.get("figure_or_table") or "").strip()] if ev.get("figure_or_table") else [],
                input={"chunk_text": chunk_text, "paper_year": obj.get("paper_year")},
                output=[_to_triplet(a)],
            )
            yield demo


def iter_hypothesis_review_demos(hyp_reviews_dir: Path) -> Iterator[DemoExample]:
    """Convert expert hypothesis reviews into hypothesis_test demos."""
    for p in sorted(hyp_reviews_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        domain = str(obj.get("domain") or "science")
        reviewer_id = str(obj.get("reviewer_id") or "")
        timestamp = str(obj.get("timestamp") or "")
        hyp_id = str(obj.get("hypothesis_id") or p.stem)

        # Input: hypothesis text fields are not present in template; we keep a short 'hypothesis_text'
        # composed from suggested_revision + issues as a proxy. Teams can later add explicit fields.
        major = obj.get("major_issues") or []
        minor = obj.get("minor_issues") or []
        time_scope = str(obj.get("time_scope") or "")
        suggested = str(obj.get("suggested_revision") or "")
        hyp_text = "\n".join(
            [x for x in [f"Time scope: {time_scope}" if time_scope else "", "Major: " + "; ".join(major) if major else "", "Minor: " + "; ".join(minor) if minor else "", "Suggested revision: " + suggested if suggested else ""] if x]
        ).strip()

        accept = bool(obj.get("accept"))
        verdict = "supported" if accept else "needs_revision"
        summary = suggested or ("; ".join(major) if major else "Needs revision")
        rec_exp = str(obj.get("required_experiments") or "") or None

        output = {
            "verdict": verdict,
            "summary": summary,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "temporal_notes": None,
            "recommended_experiments": rec_exp,
            "confidence_score": 6 if accept else 5,
        }

        demo = DemoExample(
            id=_stable_id("hypothesis_test", domain, hyp_id, p.as_posix()),
            task="hypothesis_test",
            domain=domain,
            quality="gold" if accept else "silver",
            source=DemoSource(artifact_path=str(p), reviewer_id=reviewer_id, timestamp=timestamp),
            tags=[],
            input={"hypothesis_text": hyp_text, "ctx_head": "", "time_scope": time_scope},
            output=output,
        )
        yield demo


def build_demos_from_experts(experts_root: Path) -> Dict[str, List[DemoExample]]:
    """Scan data/experts and build demo examples for supported tasks."""
    out: Dict[str, List[DemoExample]] = {"temporal_triplets": [], "hypothesis_test": []}
    gr = experts_root / "graph_reviews"
    if gr.exists():
        out["temporal_triplets"] = list(iter_graph_review_demos(gr))
    hr = experts_root / "hypothesis_reviews"
    if hr.exists():
        out["hypothesis_test"] = list(iter_hypothesis_review_demos(hr))
    return out


def index_demos_from_experts(experts_root: Path) -> Dict[str, int]:
    """Build demos from expert artifacts and upsert into Qdrant. Returns counts."""
    demos = build_demos_from_experts(experts_root)
    counts: Dict[str, int] = {}
    if demos["temporal_triplets"]:
        counts["temporal_triplets"] = upsert_demos("temporal_triplets", demos["temporal_triplets"])
    else:
        counts["temporal_triplets"] = 0
    if demos["hypothesis_test"]:
        counts["hypothesis_test"] = upsert_demos("hypothesis_test", demos["hypothesis_test"])
    else:
        counts["hypothesis_test"] = 0
    return counts
