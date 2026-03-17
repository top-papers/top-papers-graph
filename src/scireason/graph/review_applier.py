from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..review_schema import NEG_INFINITY, POS_INFINITY, normalize_temporal_payload


@dataclass
class ReviewStats:
    accepted: int = 0
    rejected: int = 0
    needs_fix: int = 0
    added: int = 0


def _iter_review_files(graph_reviews_dir: Path) -> Iterable[Path]:
    yield from sorted(graph_reviews_dir.glob("**/*.json"))


def _weight_for(verdict: str) -> float:
    if verdict == "accepted":
        return 1.0
    if verdict == "rejected":
        return -1.0
    if verdict in {"needs_time_fix", "needs_evidence_fix"}:
        return -0.25
    if verdict == "added":
        return 0.75
    return 0.0


def _legacy_temporal(a: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_temporal_payload(
        a,
        start_date_hint=NEG_INFINITY,
        end_date_hint=POS_INFINITY,
        valid_from_hint=a.get("time_interval") or None,
        valid_to_hint=POS_INFINITY,
        temporal_basis="legacy_review",
    )


def _normalize_assertions(doc: Dict[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    """Support both new and legacy expert formats.

    New format (preferred):
      { assertions: [ {subject,predicate,object,start_date,end_date,valid_from,valid_to,evidence,verdict,...}, ... ] }

    Legacy format (kept for backward compatibility in early course phases):
      { accepted_edges: [...], rejected_edges: [...], added_edges: [...] }
    """
    assertions = doc.get("assertions")
    if isinstance(assertions, list) and assertions:
        out: List[Dict[str, Any]] = []
        for a in assertions:
            if not isinstance(a, dict):
                continue
            temporal = normalize_temporal_payload(a, temporal_basis=str(a.get("temporal_basis") or "review_assertion"))
            out.append(dict(a) | temporal)
        return out

    out: List[Dict[str, Any]] = []

    for a in doc.get("accepted_edges", []) or []:
        temporal = _legacy_temporal(a)
        out.append({
            "assertion_id": a.get("assertion_id") or "legacy",
            "subject": a.get("subject"),
            "predicate": a.get("predicate"),
            "object": a.get("object"),
            **temporal,
            "evidence": {
                "page": a.get("evidence", {}).get("page"),
                "figure_or_table": a.get("evidence", {}).get("figure_or_table"),
                "snippet_or_summary": a.get("evidence", {}).get("text_snippet") or a.get("evidence", {}).get("snippet_or_summary"),
            },
            "verdict": "accepted",
            "rationale": a.get("rationale") or "legacy accepted_edges",
        })

    for a in doc.get("rejected_edges", []) or []:
        temporal = _legacy_temporal(a)
        out.append({
            "assertion_id": a.get("assertion_id") or "legacy",
            "subject": a.get("subject"),
            "predicate": a.get("predicate"),
            "object": a.get("object"),
            **temporal,
            "evidence": {
                "page": None,
                "figure_or_table": None,
                "snippet_or_summary": None,
            },
            "verdict": "rejected",
            "rationale": a.get("reason") or "legacy rejected_edges",
        })

    for a in doc.get("added_edges", []) or []:
        temporal = _legacy_temporal(a)
        out.append({
            "assertion_id": a.get("assertion_id") or "legacy",
            "subject": a.get("subject"),
            "predicate": a.get("predicate"),
            "object": a.get("object"),
            **temporal,
            "evidence": {
                "page": None,
                "figure_or_table": None,
                "snippet_or_summary": a.get("evidence_hint") or None,
            },
            "verdict": "added",
            "rationale": a.get("reason") or "legacy added_edges",
        })

    return out


def compile_overrides(graph_reviews_dir: Path, out_path: Path) -> ReviewStats:
    """Compile expert graph reviews into a DB-agnostic overrides file (JSONL).

    Each line keeps both the backward-compatible `time_interval` and the richer temporal fields.
    """
    stats = ReviewStats()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for f in _iter_review_files(graph_reviews_dir):
            doc = json.loads(f.read_text(encoding="utf-8"))
            assertions = _normalize_assertions(doc, f)
            for a in assertions:
                verdict = str(a.get("verdict", "")).strip()
                temporal = normalize_temporal_payload(a, temporal_basis=str(a.get("temporal_basis") or "compiled_review"))
                rec = {
                    "assertion_id": a.get("assertion_id") or "new",
                    "verdict": verdict,
                    "weight": _weight_for(verdict),
                    "subject": a.get("subject"),
                    "predicate": a.get("predicate"),
                    "object": a.get("object"),
                    **temporal,
                    "source_review": str(f),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if verdict == "accepted":
                    stats.accepted += 1
                elif verdict == "rejected":
                    stats.rejected += 1
                elif verdict in {"needs_time_fix", "needs_evidence_fix"}:
                    stats.needs_fix += 1
                elif verdict == "added":
                    stats.added += 1

    return stats
