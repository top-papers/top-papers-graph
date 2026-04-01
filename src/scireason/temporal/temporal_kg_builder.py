from __future__ import annotations

"""Temporal knowledge graph builder (term-level).

This module focuses on the *course requirement*:
"from a user query -> automatically build a **temporal** knowledge graph
from multiple papers -> propose testable hypotheses".

We support two extraction backends:
1) `llm_triplets`  - uses the existing TemporalTriplet extractor (best quality)
2) `cooccurrence`  - builds edges from term co-occurrence (no LLM required)

Both backends produce a unified in-memory representation that can be exported to JSON.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from rich.console import Console

from ..domain import DomainConfig
from ..temporal.schemas import TemporalTriplet
from ..temporal.temporal_triplet_extractor import extract_temporal_triplets
from .term_extraction import TermCandidate, extract_terms_rake


console = Console()

EdgeMode = Literal["auto", "llm_triplets", "cooccurrence"]


def _granularity_rank(value: str | None) -> int:
    mapping = {"year": 1, "month": 2, "interval": 2, "day": 3}
    return mapping.get(str(value or "year").strip().lower(), 1)


@dataclass(frozen=True)
class PaperEvidenceUnit:
    unit_id: str
    text: str
    source_kind: str = "text_chunk"
    page: Optional[int] = None
    image_path: str = ""
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    year: Optional[int]
    text: str
    url: str = ""
    source: str = ""
    multimodal_text: str = ""
    evidence_units: List[PaperEvidenceUnit] = field(default_factory=list)


@dataclass
class NodeStats:
    term: str
    doc_freq: int = 0
    yearly_doc_freq: Dict[int, int] = field(default_factory=dict)


@dataclass
class EdgeStats:
    source: str
    target: str
    predicate: str
    directed: bool = True

    # Aggregates
    total_count: int = 0
    yearly_count: Dict[int, int] = field(default_factory=dict)
    confidence_sum: float = 0.0
    confidence_n: int = 0
    polarity_counts: Dict[str, int] = field(default_factory=lambda: {"supports": 0, "contradicts": 0, "unknown": 0})

    # Provenance
    papers: Set[str] = field(default_factory=set)
    evidence_quotes: List[Dict[str, Any]] = field(default_factory=list)  # {paper_id, quote, page, source_kind, ...}
    time_intervals: List[Dict[str, Any]] = field(default_factory=list)  # {start, end, granularity, source, paper_id, ...}

    # Derived features / final score
    features: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    @property
    def mean_confidence(self) -> float:
        if self.confidence_n <= 0:
            return 0.0
        return float(self.confidence_sum) / float(self.confidence_n)


@dataclass
class TemporalKnowledgeGraph:
    """A lightweight temporal KG representation exportable to JSON."""

    nodes: Dict[str, NodeStats] = field(default_factory=dict)  # key = canonical term
    edges: List[EdgeStats] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "nodes": [
                {
                    "term": n.term,
                    "doc_freq": n.doc_freq,
                    "yearly_doc_freq": n.yearly_doc_freq,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "predicate": e.predicate,
                    "directed": e.directed,
                    "total_count": e.total_count,
                    "yearly_count": e.yearly_count,
                    "mean_confidence": e.mean_confidence,
                    "polarity_counts": e.polarity_counts,
                    "papers": sorted(list(e.papers))[:50],
                    "evidence_quotes": e.evidence_quotes[:10],
                    "time_intervals": e.time_intervals[:20],
                    "features": e.features,
                    "score": e.score,
                }
                for e in self.edges
            ],
        }

    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_term(t: str) -> str:
    return " ".join(str(t or "").strip().lower().split())


def _year_from_triplet(tr: TemporalTriplet, paper_year: Optional[int]) -> Optional[int]:
    if tr.time and tr.time.start:
        s = str(tr.time.start)
        try:
            y = int(s[:4])
            if 1800 <= y <= 2100:
                return y
        except Exception:
            pass
    return paper_year


def _stream_chunk_units(chunks_path: Path, *, max_chars_per_paper: int) -> tuple[list[PaperEvidenceUnit], str]:
    units: List[PaperEvidenceUnit] = []
    texts: List[str] = []
    total = 0
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line_index, line in enumerate(fh, start=1):
            if total >= max_chars_per_paper:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            raw_text = str(rec.get("text") or "")
            if not raw_text:
                continue
            remaining = max_chars_per_paper - total
            piece = raw_text[:remaining]
            if not piece:
                break
            chunk_id = str(rec.get("chunk_id") or f"chunk-{line_index:05d}")
            page_value = rec.get("page")
            try:
                page = int(page_value) if page_value is not None else None
            except Exception:
                page = None
            unit = PaperEvidenceUnit(
                unit_id=chunk_id,
                chunk_id=chunk_id,
                text=piece,
                source_kind=str(rec.get("modality") or "text_chunk"),
                page=page,
                image_path=str(rec.get("image_path") or ""),
                metadata={
                    "source_backend": rec.get("source_backend"),
                    "bbox": rec.get("bbox"),
                    "reading_order": rec.get("reading_order"),
                },
            )
            units.append(unit)
            texts.append(piece)
            total += len(piece)
    return units, "\n\n".join(texts).strip()


def _compose_multimodal_unit_text(rec: Dict[str, Any]) -> str:
    parts: List[str] = []
    text = str(rec.get("text") or "").strip()
    caption = str(rec.get("vlm_caption") or "").strip()
    tables = str(rec.get("tables_md") or "").strip()
    equations = str(rec.get("equations_md") or "").strip()

    if text:
        parts.append(text)
    if caption:
        parts.append(f"Visual evidence: {caption}")
    if tables:
        parts.append(f"Tables: {tables}")
    if equations:
        parts.append(f"Equations: {equations}")
    return "\n".join(p for p in parts if p).strip()


def _stream_multimodal_units(mm_pages_path: Path, *, max_chars_per_paper: int) -> tuple[list[PaperEvidenceUnit], str]:
    units: List[PaperEvidenceUnit] = []
    texts: List[str] = []
    total = 0
    with mm_pages_path.open("r", encoding="utf-8") as fh:
        for line_index, line in enumerate(fh, start=1):
            if total >= max_chars_per_paper:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            raw_text = _compose_multimodal_unit_text(rec)
            if not raw_text:
                continue
            remaining = max_chars_per_paper - total
            piece = raw_text[:remaining]
            if not piece:
                break
            page_value = rec.get("page")
            try:
                page = int(page_value) if page_value is not None else None
            except Exception:
                page = None
            unit_id = f"mm-page-{page if page is not None else line_index:05d}"
            unit = PaperEvidenceUnit(
                unit_id=unit_id,
                text=piece,
                source_kind="multimodal_page",
                page=page,
                image_path=str(rec.get("image_path") or ""),
                metadata={
                    "vlm_caption": rec.get("vlm_caption"),
                    "tables_md": rec.get("tables_md"),
                    "equations_md": rec.get("equations_md"),
                },
            )
            units.append(unit)
            texts.append(piece)
            total += len(piece)
    return units, "\n\n".join(texts).strip()


def load_papers_from_processed(
    processed_dir: Path,
    *,
    max_papers: Optional[int] = None,
    max_chars_per_paper: int = 40_000,
) -> List[PaperRecord]:
    """Load PaperRecord list from `data/processed/papers/<paper_id>/...`.

    In addition to plain `chunks.jsonl`, this loader now also folds in multimodal page
    records from `mm/pages.jsonl` so that the automatic graph builder can consume page text,
    VLM captions, tables, and equations directly during triplet extraction.
    """
    out: List[PaperRecord] = []
    if not processed_dir.exists():
        return out

    for d in sorted([p for p in processed_dir.iterdir() if p.is_dir()]):
        meta_path = d / "meta.json"
        chunks_path = d / "chunks.jsonl"
        mm_pages_path = d / "mm" / "pages.jsonl"
        if not meta_path.exists() or not chunks_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        pid = str(meta.get("id") or d.name)
        title = str(meta.get("title") or "").strip()
        year = meta.get("year")
        try:
            year_int = int(year) if year is not None else None
        except Exception:
            year_int = None

        try:
            text_units, text = _stream_chunk_units(chunks_path, max_chars_per_paper=max_chars_per_paper)
        except OSError:
            continue

        multimodal_units: List[PaperEvidenceUnit] = []
        multimodal_text = ""
        if mm_pages_path.exists():
            try:
                multimodal_units, multimodal_text = _stream_multimodal_units(mm_pages_path, max_chars_per_paper=max_chars_per_paper)
            except OSError:
                multimodal_units, multimodal_text = [], ""

        if not text and meta.get("abstract"):
            text = str(meta.get("abstract") or "").strip()
        if not text and title:
            text = title

        evidence_units = [*text_units, *multimodal_units]
        if not evidence_units and text:
            evidence_units = [
                PaperEvidenceUnit(
                    unit_id=f"{pid}:fulltext",
                    chunk_id=f"{pid}:fulltext",
                    text=text,
                    source_kind="text_chunk",
                )
            ]

        out.append(
            PaperRecord(
                paper_id=pid,
                title=title,
                year=year_int,
                text=text,
                url=str(meta.get("url") or ""),
                source=str(meta.get("source") or ""),
                multimodal_text=multimodal_text,
                evidence_units=evidence_units,
            )
        )

        if max_papers and len(out) >= max_papers:
            break

    return out


def _sentences(text: str) -> List[str]:
    parts = []
    for p in (text or "").replace("\n", " ").split("."):
        p = p.strip()
        if len(p) >= 20:
            parts.append(p)
    return parts if parts else [text]


def _apply_expert_overrides(edges: Dict[Tuple[str, str, str], EdgeStats], overrides_path: Optional[Path]) -> None:
    """Adjust edge scores using expert graph reviews compiled into JSONL.

    The compiled overrides format is produced by `top-papers-graph refresh-feedback`.
    """

    if not overrides_path or not overrides_path.exists():
        return

    w: Dict[Tuple[str, str, str], float] = {}
    for line in overrides_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        s = _norm_term(str(rec.get("subject") or ""))
        p = _norm_term(str(rec.get("predicate") or ""))
        o = _norm_term(str(rec.get("object") or ""))
        if not s or not p or not o:
            continue
        try:
            weight = float(rec.get("weight") or 0.0)
        except Exception:
            weight = 0.0
        w[(s, p, o)] = w.get((s, p, o), 0.0) + weight

    if not w:
        return

    for k, edge in edges.items():
        ew = w.get(k)
        if ew is None:
            continue
        edge.features["expert_weight"] = float(ew)


def _iter_paper_units(pr: PaperRecord) -> List[PaperEvidenceUnit]:
    units = [u for u in (pr.evidence_units or []) if str(u.text or "").strip()]
    if units:
        return units
    fallback_text = f"{pr.title}\n\n{pr.text}".strip()
    if not fallback_text:
        return []
    return [
        PaperEvidenceUnit(
            unit_id=f"{pr.paper_id}:fulltext",
            chunk_id=f"{pr.paper_id}:fulltext",
            text=fallback_text,
            source_kind="text_chunk",
        )
    ]


def _time_payload_from_triplet(tr: TemporalTriplet, *, paper_id: str, unit: PaperEvidenceUnit, paper_year: Optional[int]) -> Optional[Dict[str, Any]]:
    if tr.time is None:
        return None
    start = str(tr.time.start or "").strip() or (str(paper_year) if paper_year is not None else "")
    end = str(tr.time.end or tr.time.start or "").strip() or start
    if not start:
        return None
    return {
        "start": start,
        "end": end,
        "granularity": str(tr.time.granularity or "year"),
        "source": str(getattr(tr, "time_source", "extracted") or "extracted"),
        "paper_id": paper_id,
        "unit_id": unit.unit_id,
        "source_kind": unit.source_kind,
        "page": unit.page,
        "image_path": unit.image_path,
    }


def _append_unique_interval(edge: EdgeStats, payload: Dict[str, Any]) -> None:
    key = (
        str(payload.get("start") or ""),
        str(payload.get("end") or ""),
        str(payload.get("granularity") or "year"),
        str(payload.get("source") or "extracted"),
        str(payload.get("paper_id") or ""),
        str(payload.get("unit_id") or ""),
    )
    existing = {
        (
            str(item.get("start") or ""),
            str(item.get("end") or ""),
            str(item.get("granularity") or "year"),
            str(item.get("source") or "extracted"),
            str(item.get("paper_id") or ""),
            str(item.get("unit_id") or ""),
        )
        for item in edge.time_intervals
    }
    if key not in existing:
        edge.time_intervals.append(payload)


def _append_evidence(edge: EdgeStats, *, pr: PaperRecord, unit: PaperEvidenceUnit, tr: TemporalTriplet) -> None:
    quote = str(tr.evidence_quote or "").strip()
    if not quote:
        return
    payload: Dict[str, Any] = {
        "paper_id": pr.paper_id,
        "quote": quote,
        "source_kind": unit.source_kind,
        "unit_id": unit.unit_id,
    }
    if unit.page is not None:
        payload["page"] = unit.page
    if unit.image_path:
        payload["image_path"] = unit.image_path
    if unit.chunk_id:
        payload["chunk_id"] = unit.chunk_id
    if tr.time is not None:
        payload["time"] = {
            "start": tr.time.start,
            "end": tr.time.end,
            "granularity": tr.time.granularity,
            "source": str(getattr(tr, "time_source", "extracted") or "extracted"),
        }
    edge.evidence_quotes.append(payload)


def _merge_triplets_into_graph(
    *,
    pr: PaperRecord,
    unit: PaperEvidenceUnit,
    triplets: Sequence[TemporalTriplet],
    edges: Dict[Tuple[str, str, str], EdgeStats],
    term_set: Set[str],
) -> None:
    year = pr.year
    for tr in triplets:
        s = _norm_term(tr.subject)
        o = _norm_term(tr.object)
        p = _norm_term(tr.predicate)
        if not s or not o or not p:
            continue
        term_set.add(s)
        term_set.add(o)

        y = _year_from_triplet(tr, year)
        if y is None:
            continue

        key = (s, p, o)
        if key not in edges:
            edges[key] = EdgeStats(source=s, target=o, predicate=p, directed=True)
        edge = edges[key]
        edge.total_count += 1
        edge.yearly_count[y] = edge.yearly_count.get(y, 0) + 1
        edge.papers.add(pr.paper_id)
        edge.confidence_sum += float(tr.confidence)
        edge.confidence_n += 1
        edge.polarity_counts[tr.polarity] = edge.polarity_counts.get(tr.polarity, 0) + 1
        _append_evidence(edge, pr=pr, unit=unit, tr=tr)
        time_payload = _time_payload_from_triplet(tr, paper_id=pr.paper_id, unit=unit, paper_year=year)
        if time_payload is not None:
            _append_unique_interval(edge, time_payload)


def _merge_cooccurrence_into_graph(
    *,
    pr: PaperRecord,
    unit: PaperEvidenceUnit,
    domain: DomainConfig,
    edges: Dict[Tuple[str, str, str], EdgeStats],
    term_set: Set[str],
    max_terms_per_paper: int,
    language: str,
) -> None:
    base_text = unit.text.strip() or f"{pr.title}\n\n{pr.text}".strip()
    if not base_text:
        return
    terms: List[TermCandidate] = extract_terms_rake(
        base_text,
        max_terms=max_terms_per_paper,
        language=language,
        boost_terms=domain.keywords,
    )
    term_list = [_norm_term(t.term) for t in terms if _norm_term(t.term)]
    term_set.update(term_list)

    if pr.year is None:
        return
    low_text = base_text.lower()
    sents = _sentences(low_text)
    for snt in sents:
        present = [t for t in term_list if t and (t in snt)]
        if len(present) < 2:
            continue
        uniq = sorted(set(present))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                src, tgt = (a, b) if a <= b else (b, a)
                key = (src, "cooccurs_with", tgt)
                if key not in edges:
                    edges[key] = EdgeStats(source=src, target=tgt, predicate="cooccurs_with", directed=False)
                edge = edges[key]
                edge.total_count += 1
                edge.yearly_count[pr.year] = edge.yearly_count.get(pr.year, 0) + 1
                edge.papers.add(pr.paper_id)
                payload = {
                    "paper_id": pr.paper_id,
                    "quote": snt[:200],
                    "source_kind": f"{unit.source_kind}:cooccurrence",
                    "unit_id": unit.unit_id,
                }
                if unit.page is not None:
                    payload["page"] = unit.page
                if unit.image_path:
                    payload["image_path"] = unit.image_path
                edge.evidence_quotes.append(payload)
                _append_unique_interval(
                    edge,
                    {
                        "start": str(pr.year),
                        "end": str(pr.year),
                        "granularity": "year",
                        "source": "paper_year_fallback",
                        "paper_id": pr.paper_id,
                        "unit_id": unit.unit_id,
                        "source_kind": unit.source_kind,
                        "page": unit.page,
                        "image_path": unit.image_path,
                    },
                )


def build_temporal_kg(
    papers: Sequence[PaperRecord],
    *,
    domain: DomainConfig,
    query: str = "",
    edge_mode: EdgeMode = "auto",
    max_terms_per_paper: int = 25,
    recent_window_years: int = 3,
    expert_overrides_path: Optional[Path] = Path("data/derived/expert_overrides.jsonl"),
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> TemporalKnowledgeGraph:
    """Build a temporal KG from a list of papers.

    When `edge_mode=auto`, the builder prefers LLM triplets on *each evidence unit* and only
    falls back to local co-occurrence for the failing unit. The next unit/paper again retries
    the primary LLM strategy.
    """

    if not papers:
        return TemporalKnowledgeGraph(meta={"domain": domain.domain_id, "query": query, "n_papers": 0})

    cfg = dict(domain.term_graph or {})  # type: ignore[attr-defined]
    max_terms_per_paper = int(cfg.get("max_terms_per_paper", max_terms_per_paper))
    recent_window_years = int(cfg.get("recent_window_years", recent_window_years))
    language = str(cfg.get("language", "en"))

    nodes: Dict[str, NodeStats] = {}
    edges: Dict[Tuple[str, str, str], EdgeStats] = {}
    paper_term_sets: Dict[str, Set[str]] = {}

    base_mode = "llm_triplets" if edge_mode == "auto" else edge_mode
    llm_failures = 0
    localized_fallbacks = 0

    for pr in papers:
        units = _iter_paper_units(pr)
        if not units:
            continue
        paper_term_set: Set[str] = set()

        for unit in units:
            unit_text = str(unit.text or "").strip()
            if not unit_text:
                continue
            unit_mode = base_mode

            if unit_mode == "llm_triplets":
                try:
                    triplets = extract_temporal_triplets(
                        domain=domain.title,
                        chunk_text=unit_text,
                        paper_year=pr.year,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                    )
                except Exception as e:
                    llm_failures += 1
                    localized_fallbacks += 1
                    console.print(
                        f"[yellow]Temporal triplets failed for {pr.paper_id}/{unit.unit_id}: {e}. "
                        "Using localized co-occurrence fallback for this evidence unit only.[/yellow]"
                    )
                    triplets = []
                    unit_mode = "cooccurrence"
                else:
                    if triplets:
                        _merge_triplets_into_graph(pr=pr, unit=unit, triplets=triplets, edges=edges, term_set=paper_term_set)
                    else:
                        localized_fallbacks += 1
                        unit_mode = "cooccurrence"

            if unit_mode == "cooccurrence":
                _merge_cooccurrence_into_graph(
                    pr=pr,
                    unit=unit,
                    domain=domain,
                    edges=edges,
                    term_set=paper_term_set,
                    max_terms_per_paper=max_terms_per_paper,
                    language=language,
                )

        if paper_term_set:
            paper_term_sets[pr.paper_id] = paper_term_set

    for pr in papers:
        ts = paper_term_sets.get(pr.paper_id) or set()
        if not ts:
            continue
        for term in ts:
            if term not in nodes:
                nodes[term] = NodeStats(term=term)
            n = nodes[term]
            n.doc_freq += 1
            if pr.year is not None:
                n.yearly_doc_freq[pr.year] = n.yearly_doc_freq.get(pr.year, 0) + 1

    n_docs = max(1, len(paper_term_sets))
    years_all = sorted({y for e in edges.values() for y in e.yearly_count.keys()})
    max_year = max(years_all) if years_all else None

    def pmi(edge: EdgeStats) -> float:
        du = nodes.get(edge.source).doc_freq if nodes.get(edge.source) else 0
        dv = nodes.get(edge.target).doc_freq if nodes.get(edge.target) else 0
        dij = len(edge.papers)
        if du <= 0 or dv <= 0 or dij <= 0:
            return 0.0
        return math.log((dij * n_docs) / float(du * dv) + 1e-9)

    def trend(edge: EdgeStats) -> float:
        if max_year is None:
            return 0.0
        rw = max(1, recent_window_years)
        recent_years = [y for y in range(max_year - rw + 1, max_year + 1)]
        recent = sum(edge.yearly_count.get(y, 0) for y in recent_years)
        total = sum(edge.yearly_count.values())
        recent_share = recent / float(max(1, total))
        expected = rw / float(max(1, len(years_all))) if years_all else 0.0
        return recent_share - expected

    _apply_expert_overrides(edges, expert_overrides_path)

    edge_list: List[EdgeStats] = []
    for e in edges.values():
        e.features.setdefault("pmi", pmi(e))
        e.features.setdefault("trend", trend(e))
        e.features.setdefault("log_count", math.log1p(float(e.total_count)))
        e.features.setdefault("mean_conf", e.mean_confidence)
        if e.time_intervals:
            extracted_count = sum(1 for item in e.time_intervals if str(item.get("source") or "") == "extracted")
            e.features.setdefault("time_signal_count", float(len(e.time_intervals)))
            e.features.setdefault("time_extracted_ratio", float(extracted_count) / float(len(e.time_intervals)))
            e.features.setdefault(
                "time_best_granularity",
                float(max(_granularity_rank(str(item.get("granularity") or "year")) for item in e.time_intervals)),
            )

        expert_w = float(e.features.get("expert_weight", 0.0) or 0.0)
        e.score = (
            1.0 * e.features["log_count"]
            + 0.75 * e.features["pmi"]
            + 1.25 * e.features["trend"]
            + 0.5 * e.features["mean_conf"]
            + 1.5 * expert_w
        )
        edge_list.append(e)

    edge_list.sort(key=lambda x: x.score, reverse=True)

    return TemporalKnowledgeGraph(
        nodes=nodes,
        edges=edge_list,
        meta={
            "domain": domain.domain_id,
            "domain_title": domain.title,
            "query": query,
            "n_papers": len(paper_term_sets),
            "edge_mode": base_mode,
            "years": years_all,
            "localized_fallbacks": localized_fallbacks,
            "llm_failures": llm_failures,
            "multimodal_enabled": any(bool(getattr(pr, "multimodal_text", "").strip()) for pr in papers),
        },
    )
