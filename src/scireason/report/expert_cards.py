from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..mm.structured_pdf import StructuredChunk, load_structured_chunks
from ..temporal.temporal_kg_builder import EdgeStats, TemporalKnowledgeGraph


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]{3,}")
_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}(?:\s*[-–/]\s*(?:18|19|20)\d{2})?\b")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%\b")
_TEMP_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:°C|C|K)\b", re.IGNORECASE)
_PH_RE = re.compile(r"\bpH\s*\d+(?:\.\d+)?\b", re.IGNORECASE)
_VOLT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:V|mV|kV)\b")
_RATE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:C-rate|C rate|A|mA|A g-1|A/g)\b", re.IGNORECASE)
_DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|g|kg|uM|µM|mM|M|nM|wt%)\b", re.IGNORECASE)
_CAUSAL_HINTS = {"cause", "causes", "improves", "increase", "increases", "decrease", "decreases", "induces", "promotes", "reduces", "predicts"}
_CORREL_HINTS = ("correlat", "associat", "linked", "related", "trend", "co-occurr", "cooccur")


def _slugify(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:80] or "item"


class _LegacyChunk:
    def __init__(self, paper_id: str, chunk_id: str, text: str, order: int) -> None:
        self.paper_id = paper_id
        self.chunk_id = chunk_id
        self.modality = "text"
        self.text = text
        self.page = None
        self.order = order
        self.image_path = None
        self.figure_or_table = None
        self.table_markdown = None
        self.section = None
        self.summary = text[:400]
        self.backend = "legacy"
        self.metadata: Dict[str, Any] = {}

    def searchable_text(self) -> str:
        return (self.text or "").strip()


ChunkLike = StructuredChunk | _LegacyChunk


def _iter_paper_dirs(processed_dir: Path) -> Iterable[Path]:
    if not processed_dir.exists():
        return []
    return sorted([p for p in processed_dir.iterdir() if p.is_dir() and (p / "meta.json").exists()])


def _load_chunks_for_paper_dir(paper_dir: Path) -> List[ChunkLike]:
    chunks = load_structured_chunks(paper_dir)
    if chunks:
        return sorted(chunks, key=lambda c: (int(c.order or 0), c.chunk_id))

    legacy = paper_dir / "chunks.jsonl"
    if not legacy.exists():
        return []
    out: list[ChunkLike] = []
    for idx, line in enumerate(legacy.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        text = str(rec.get("text") or "").strip()
        if not text:
            continue
        cid = str(rec.get("chunk_id") or f"{paper_dir.name}:{idx}")
        out.append(_LegacyChunk(paper_id=paper_dir.name, chunk_id=cid, text=text, order=idx))
    return out


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _preview(text: str, limit: int = 360) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def extract_condition_hints(text: str, *, limit: int = 8) -> List[str]:
    low = text or ""
    hints: list[str] = []
    for rx in (_YEAR_RE, _TEMP_RE, _PERCENT_RE, _PH_RE, _VOLT_RE, _RATE_RE, _DOSE_RE):
        for match in rx.findall(low):
            value = match if isinstance(match, str) else match[0]
            value = str(value).strip()
            if value and value not in hints:
                hints.append(value)
                if len(hints) >= limit:
                    return hints
    return hints


def generate_chunk_cards(processed_dir: Path, out_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for paper_dir in _iter_paper_dirs(processed_dir):
        try:
            meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        paper_id = str(meta.get("id") or paper_dir.name)
        title = str(meta.get("title") or "")
        for chunk in _load_chunks_for_paper_dir(paper_dir):
            text = chunk.searchable_text()
            card = {
                "paper_id": paper_id,
                "paper_title": title,
                "chunk_id": chunk.chunk_id,
                "modality": getattr(chunk, "modality", "text"),
                "page": getattr(chunk, "page", None),
                "figure_or_table": getattr(chunk, "figure_or_table", None),
                "section": getattr(chunk, "section", None),
                "summary": getattr(chunk, "summary", "") or _preview(text, 240),
                "image_path": getattr(chunk, "image_path", None),
                "table_markdown": getattr(chunk, "table_markdown", None),
                "backend": getattr(chunk, "backend", "legacy"),
                "condition_hints": extract_condition_hints(text),
                "text_preview": _preview(text, 600),
            }
            cards.append(card)
    cards.sort(key=lambda c: (str(c.get("paper_id") or ""), str(c.get("chunk_id") or "")))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in cards:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return cards


def _time_interval_for_edge(edge: EdgeStats) -> str:
    years = sorted(int(y) for y in (edge.yearly_count or {}).keys())
    if years:
        if len(years) == 1:
            return str(years[0])
        return f"{years[0]}-{years[-1]}"
    return ""


def _build_chunk_map(processed_dir: Path) -> Dict[str, List[ChunkLike]]:
    out: dict[str, list[ChunkLike]] = {}
    for paper_dir in _iter_paper_dirs(processed_dir):
        try:
            meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        paper_id = str(meta.get("id") or paper_dir.name)
        chunks = _load_chunks_for_paper_dir(paper_dir)
        if chunks:
            out[paper_id] = chunks
    return out


def _predicate_tokens(predicate: str) -> List[str]:
    toks = [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", predicate or "") if t]
    return [t for t in toks if len(t) >= 3]


def _score_chunk(edge: EdgeStats, chunk: ChunkLike, quote: str) -> float:
    text = chunk.searchable_text().lower()
    if not text:
        return 0.0

    source = edge.source.lower()
    target = edge.target.lower()
    score = 0.0

    if source in text:
        score += 3.0
    if target in text:
        score += 3.0
    if quote and quote.lower()[:120] in text:
        score += 6.0

    for tok in _predicate_tokens(edge.predicate):
        if tok in text:
            score += 0.5

    # Figures and tables are especially important for Task 2 evidence.
    modality = getattr(chunk, "modality", "text")
    if modality in {"figure", "table"}:
        score += 0.75
    elif modality == "text":
        score += 0.25

    # Prefer shorter, more precise evidence spans over long page dumps.
    length_penalty = min(len(text), 4000) / 4000.0
    score -= 0.2 * length_penalty
    return score


def _best_evidence(edge: EdgeStats, chunk_map: Dict[str, List[ChunkLike]]) -> Dict[str, Any]:
    quote = ""
    if edge.evidence_quotes:
        quote = str(edge.evidence_quotes[0].get("quote") or "").strip()

    candidate_papers = list(edge.papers)
    if not candidate_papers and edge.evidence_quotes:
        candidate_papers = [str(edge.evidence_quotes[0].get("paper_id") or "")]
    best: Dict[str, Any] = {
        "paper_id": candidate_papers[0] if candidate_papers else "",
        "chunk_id": None,
        "page": None,
        "figure_or_table": None,
        "snippet_or_summary": _preview(quote, 220) if quote else "",
        "modality": None,
        "score": 0.0,
    }

    for paper_id in candidate_papers:
        for chunk in chunk_map.get(str(paper_id), []):
            score = _score_chunk(edge, chunk, quote)
            if score <= best.get("score", 0.0):
                continue
            snippet = quote or chunk.searchable_text()
            best = {
                "paper_id": str(paper_id),
                "chunk_id": chunk.chunk_id,
                "page": getattr(chunk, "page", None),
                "figure_or_table": getattr(chunk, "figure_or_table", None),
                "snippet_or_summary": _preview(snippet, 260),
                "modality": getattr(chunk, "modality", None),
                "score": score,
            }

    return best


def _suggest_verdict(edge: EdgeStats, evidence: Dict[str, Any]) -> str:
    has_time = bool(_time_interval_for_edge(edge))
    has_evidence = bool((evidence.get("snippet_or_summary") or "").strip())
    if not has_evidence:
        return "needs_evidence_fix"
    if not has_time:
        return "needs_time_fix"
    return "accepted"


def _rationale(edge: EdgeStats, evidence: Dict[str, Any], verdict: str) -> str:
    base = []
    if verdict == "needs_evidence_fix":
        base.append("Автокарточка не нашла достаточно точного evidence-фрагмента в чанках статьи; нужно проверить страницу/рисунок/таблицу вручную.")
    elif verdict == "needs_time_fix":
        base.append("Связь подтверждается текстом/мультимодальным чанком, но в агрегированном графе не хватает явного временного окна применимости.")
    else:
        base.append("В графе нашлись и временная привязка, и конкретный evidence-фрагмент; карточка готова к быстрой экспертной верификации.")

    pred_toks = set(_predicate_tokens(edge.predicate))
    if pred_toks & _CAUSAL_HINTS:
        snippet = str(evidence.get("snippet_or_summary") or "").lower()
        if any(h in snippet for h in _CORREL_HINTS):
            base.append("Похоже, источник описывает скорее ассоциацию/корреляцию, чем строгую причинность; при ревью стоит проверить предикат.")

    if evidence.get("figure_or_table"):
        base.append("Evidence локализовано в мультимодальном объекте (figure/table), что соответствует требованиям второй экспертной задачи.")

    return " ".join(base)


def _assertion_id(edge: EdgeStats) -> str:
    interval = _time_interval_for_edge(edge)
    raw = f"{edge.source}|{edge.predicate}|{edge.target}|{interval}"
    return "A-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def generate_task2_review_cards(
    kg: TemporalKnowledgeGraph,
    *,
    processed_dir: Path,
    domain_id: str,
    out_dir: Path,
    max_assertions: int = 250,
) -> List[Dict[str, Any]]:
    """Generate machine-filled Task 2 review cards aligned to the course template.

    Output structure:
    - `out_dir/index.json` with all generated per-paper files
    - one JSON file per paper compatible with `data/experts/graph_reviews/_template.json`
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_map = _build_chunk_map(processed_dir)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    per_paper: Dict[str, List[Dict[str, Any]]] = {}
    edges = sorted(kg.edges, key=lambda e: float(e.score or 0.0), reverse=True)[: max_assertions]

    for edge in edges:
        evidence = _best_evidence(edge, chunk_map)
        paper_id = str(evidence.get("paper_id") or (sorted(edge.papers)[0] if edge.papers else "unknown"))
        verdict = _suggest_verdict(edge, evidence)
        card = {
            "assertion_id": _assertion_id(edge),
            "subject": edge.source,
            "predicate": edge.predicate,
            "object": edge.target,
            "time_interval": _time_interval_for_edge(edge) or "If applicable: year range / before-after / condition window",
            "evidence": {
                "page": evidence.get("page"),
                "figure_or_table": evidence.get("figure_or_table"),
                "snippet_or_summary": evidence.get("snippet_or_summary") or "",
            },
            "verdict": verdict,
            "rationale": _rationale(edge, evidence, verdict),
            "metadata": {
                "papers": sorted(list(edge.papers))[:10],
                "mean_confidence": round(float(edge.mean_confidence), 4),
                "total_count": int(edge.total_count or 0),
                "yearly_count": edge.yearly_count,
                "modality": evidence.get("modality"),
                "chunk_id": evidence.get("chunk_id"),
                "score": round(float(edge.score or 0.0), 6),
            },
        }
        per_paper.setdefault(paper_id, []).append(card)

    manifest: list[dict[str, Any]] = []
    for paper_id, assertions in sorted(per_paper.items(), key=lambda kv: kv[0]):
        assertions.sort(key=lambda a: float(a.get("metadata", {}).get("score") or 0.0), reverse=True)
        payload = {
            "domain": domain_id,
            "paper_id": paper_id,
            "reviewer_id": "auto_pipeline",
            "timestamp": now,
            "assertions": assertions,
        }
        fname = f"{_slugify(paper_id)}.json"
        (out_dir / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({
            "paper_id": paper_id,
            "path": str((out_dir / fname).as_posix()),
            "n_assertions": len(assertions),
        })

    (out_dir / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
