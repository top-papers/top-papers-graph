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

import asyncio
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from rich.console import Console

from ..domain import DomainConfig
from ..ingest.vlm_ocr import extract_pdf_page_chunks_vlm_ocr
from ..temporal.schemas import TemporalTriplet
from ..config import settings
from ..llm import _resolve_llm_selection
from ..temporal.temporal_triplet_extractor import (
    extract_temporal_triplets,
    extract_temporal_triplets_async,
    extract_temporal_triplets_localized_fallback,
)
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
    pdf_path: str = ""
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
                pdf_path=str((processed_dir.parent / "raw_pdfs" / f"{d.name}.pdf").resolve()),
                url=str(meta.get("url") or ""),
                source=str(meta.get("source") or ""),
                multimodal_text=multimodal_text,
                evidence_units=evidence_units,
            )
        )

        if max_papers and len(out) >= max_papers:
            break

    return out


def _repairable_with_vlm(pr: PaperRecord, unit: PaperEvidenceUnit) -> bool:
    if not bool(getattr(settings, "ocr_vlm_fallback_enabled", False)):
        return False
    if str(getattr(settings, "vlm_backend", "none") or "none").lower() == "none":
        return False
    if not pr.pdf_path or unit.page is None or not Path(pr.pdf_path).exists():
        return False
    return str(unit.metadata.get("source_backend") or "").lower() not in {"pymupdf_vlm_ocr", "multimodal"}


def _best_repaired_page_chunk_text(original_text: str, repaired_chunks: Sequence[Any]) -> str:
    probe = str(original_text or "").strip()[:600]
    if not probe:
        return ""
    best_text = ""
    best_score = -1.0
    for rec in repaired_chunks:
        candidate = str(getattr(rec, "text", "") or "").strip()
        if not candidate:
            continue
        score = SequenceMatcher(None, probe, candidate[:600]).ratio()
        if score > best_score:
            best_score = score
            best_text = candidate
    return best_text


def _retry_unit_with_vlm_page_repair(
    *,
    domain_title: str,
    pr: PaperRecord,
    unit: PaperEvidenceUnit,
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> list[TemporalTriplet] | None:
    if not _repairable_with_vlm(pr, unit):
        return None
    try:
        repaired_chunks = extract_pdf_page_chunks_vlm_ocr(
            pdf_path=Path(pr.pdf_path),
            paper_id=pr.paper_id,
            page_index=int(unit.page),
            backend=str(getattr(settings, "vlm_backend", "none") or "none"),
            model_id=str(getattr(settings, "vlm_model_id", "") or ""),
        )
    except Exception as e:
        console.print(f"[yellow]    VLM page repair unavailable for {pr.paper_id}/{unit.unit_id}: {e}[/yellow]")
        return None

    repaired_text = _best_repaired_page_chunk_text(unit.text, repaired_chunks)
    if not repaired_text or repaired_text == str(unit.text or "").strip():
        return None

    console.print(f"[cyan]    ↻ VLM page repair retry for page {unit.page}[/cyan]")
    try:
        return extract_temporal_triplets(
            domain=domain_title,
            chunk_text=repaired_text,
            paper_year=pr.year,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
    except Exception as e:
        console.print(f"[yellow]    VLM page repair retry failed for {pr.paper_id}/{unit.unit_id}: {e}[/yellow]")
        return None


def _sentences(text: str) -> List[str]:
    parts = []
    for p in (text or "").replace("\n", " ").split("."):
        p = p.strip()
        if len(p) >= 20:
            parts.append(p)
    return parts if parts else [text]


def _run_async_sync(awaitable):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - forwarded to caller
            result["error"] = exc

    import threading

    thread = threading.Thread(target=_worker, name="temporal-kg-async-runner", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _extract_triplets_batch_async(
    *,
    domain: DomainConfig,
    pr: PaperRecord,
    units: Sequence[PaperEvidenceUnit],
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> list[tuple[PaperEvidenceUnit, list[TemporalTriplet], Exception | None]]:
    concurrency = max(1, int(getattr(settings, "g4f_async_max_concurrency", 3) or 3))
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(unit: PaperEvidenceUnit):
        try:
            triplets = await extract_temporal_triplets_async(
                domain=domain.title,
                chunk_text=str(unit.text or ""),
                paper_year=pr.year,
                llm_provider=llm_provider,
                llm_model=llm_model,
                semaphore=semaphore,
            )
            return unit, triplets, None
        except Exception as exc:
            return unit, [], exc

    tasks = [asyncio.create_task(_worker(unit)) for unit in units if str(unit.text or "").strip()]
    if not tasks:
        return []
    return await asyncio.gather(*tasks)


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


def _build_canonical_vocabulary(
    papers: Sequence["PaperRecord"],
    domain: "DomainConfig",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> List[str]:
    """Улучшение D: формирование канонического словаря понятий по всему корпусу.

    Один LLM-вызов на весь корпус статей (заголовки + аннотации). На выходе —
    единый список ключевых научных понятий в канонической форме (2-4 слова).

    Этот словарь инжектируется в промпт каждого chunk'а, чтобы LLM использовала
    согласованные имена сущностей между статьями. Без этого одно и то же понятие
    может называться по-разному в разных PDF ("treatment effect" vs "causal effect"
    vs "heterogeneous treatment effect"), что приводит к фрагментации графа и
    отсутствию кросс-документных связей.

    Настройки:
        canonical_vocabulary_enabled (bool): включить/выключить.
        canonical_vocabulary_max_concepts (int): макс. число понятий в словаре.

    Затраты: +1 LLM-вызов на весь корпус (~10-15с).
    """
    from ..llm import chat_json, temporary_llm_selection, _resolve_auto_provider

    max_concepts = settings.canonical_vocabulary_max_concepts
    # Build input: title + first 500 chars of each paper
    paper_descriptions = []
    for pr in papers:
        desc = pr.title or ""
        text_preview = (pr.text or "")[:500]
        if text_preview:
            desc += f"\n{text_preview}"
        paper_descriptions.append(f"--- {pr.paper_id} ---\n{desc}")

    corpus_text = "\n\n".join(paper_descriptions)

    system = f"""Ты — помощник исследователя в области {domain.title}.
Тебе даны заголовки и аннотации нескольких научных статей из одной предметной области.
Твоя задача — составить единый канонический словарь ключевых научных понятий, которые встречаются или подразумеваются в этих статьях.

Правила:
- Используй короткие, канонические формы (2-4 слова): "causal inference", не "causal inference methods and approaches"
- Включи как общие понятия области (методы, фреймворки), так и специфичные для этих статей
- Если понятие встречается в нескольких статьях — оно обязательно должно быть в списке
- Не включай имена авторов, названия журналов, годы
- Верни JSON-массив строк, до {max_concepts} понятий
"""

    user = f"""Статьи:

{corpus_text}

Верни JSON-массив из {max_concepts} ключевых канонических понятий этой группы статей."""

    with temporary_llm_selection(llm_provider=llm_provider, llm_model=llm_model):
        provider = (settings.llm_provider or "").lower().strip() or "auto"
        if provider == "auto":
            provider = _resolve_auto_provider()
        if provider == "mock":
            return []

        timeout_seconds = float(getattr(settings, "llm_request_timeout_seconds", 25) or 25)
        try:
            from .temporal_triplet_extractor import _run_with_timeout
            data = _run_with_timeout(
                timeout_seconds,
                chat_json,
                system=system,
                user=user,
                schema_hint="Верни JSON-массив строк: [\"concept1\", \"concept2\", ...]",
                temperature=0.0,
            )
            if isinstance(data, list):
                vocab = [str(c).strip().lower() for c in data if isinstance(c, str) and str(c).strip()]
                return vocab[:max_concepts]
            return []
        except Exception as e:
            console.print(f"  [yellow]Canonical vocabulary extraction failed: {e}[/yellow]")
            return []


def _extract_paper_summary_triplets(
    pr: "PaperRecord",
    domain: "DomainConfig",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> List[TemporalTriplet]:
    """Улучшение C: извлечение якорных триплетов из abstract+intro статьи.

    Один LLM-вызов на статью — извлекает 3-7 ключевых каузальных утверждений,
    отражающих главный научный вклад. Якорные триплеты получают бонус +0.15 к
    confidence и маркируются source_kind="paper_summary".

    Цель: создать «скелет» графа из сильных утверждений. Без этого все chunk-level
    триплеты имеют одинаковый вес, и scoring не может отличить ключевые связи от
    деталей в разделе Related Work или Methodology.

    Настройки:
        paper_summary_triplets_enabled (bool): включить/выключить.
        paper_summary_max_input_chars (int): макс. длина входного текста.

    Затраты: +1 LLM-вызов на статью (~10-20с каждый).
    """
    from ..llm import chat_json, temporary_llm_selection, _resolve_auto_provider
    from .temporal_triplet_extractor import (
        _validate_triplet_payload,
        _finalize_triplets,
        TEMPORAL_TRIPLET_SCHEMA_HINT,
    )

    max_chars = settings.paper_summary_max_input_chars
    text = f"{pr.title}\n\n{(pr.text or '')[:max_chars]}"

    system = f"""Ты — помощник исследователя в области {domain.title}.
Тебе дан заголовок и начало научной статьи. Извлеки 3-7 КЛЮЧЕВЫХ каузальных утверждений статьи — её главные научные находки и вклад.
Фокусируйся на причинно-следственных связях, методологических вкладах и количественных результатах.
Не извлекай общеизвестные факты или определения — только то, что является вкладом ЭТОЙ статьи."""

    user = f"""{text}

{TEMPORAL_TRIPLET_SCHEMA_HINT}

Извлеки 3-7 ключевых каузальных утверждений этой статьи."""

    with temporary_llm_selection(llm_provider=llm_provider, llm_model=llm_model):
        provider = (settings.llm_provider or "").lower().strip() or "auto"
        if provider == "auto":
            provider = _resolve_auto_provider()
        if provider == "mock":
            return []

        timeout_seconds = float(getattr(settings, "llm_request_timeout_seconds", 25) or 25)
        try:
            from .temporal_triplet_extractor import _run_with_timeout
            data = _run_with_timeout(
                timeout_seconds,
                chat_json,
                system=system,
                user=user,
                schema_hint=TEMPORAL_TRIPLET_SCHEMA_HINT,
                temperature=0.0,
            )
            triplets = _finalize_triplets(_validate_triplet_payload(data), pr.year)
            # Boost confidence for anchor triplets
            for t in triplets:
                t.confidence = min(1.0, t.confidence + 0.15)
            return triplets
        except Exception as e:
            console.print(f"  [yellow]Paper summary extraction failed for {pr.paper_id}: {e}[/yellow]")
            return []


def _token_similarity(a: str, b: str) -> float:
    """Токенная схожесть для fuzzy-matching сущностей (Jaccard + containment).

    Возвращает max(Jaccard, containment * 0.9), где containment — доля
    общих токенов к минимальному из двух множеств. Это позволяет
    склеивать "credit limit" и "credit limit management" (containment=1.0),
    но не "credit limit" и "credit card" (Jaccard=0.33).
    """
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    # Jaccard
    jaccard = intersection / len(ta | tb)
    # Containment: if one is subset of another
    containment = intersection / min(len(ta), len(tb))
    return max(jaccard, containment * 0.9)


def _normalize_entities_fuzzy(
    edges: Dict[Tuple[str, str, str], EdgeStats],
    nodes: Dict[str, NodeStats],
    threshold: float = 0.85,
) -> Tuple[Dict[Tuple[str, str, str], EdgeStats], Dict[str, NodeStats]]:
    """Улучшение B: post-hoc fuzzy-нормализация сущностей в графе.

    После построения графа проходит по всем нодам и склеивает пары с
    токенной схожестью >= threshold. Каноническим становится термин с
    большей doc_freq (или более короткий при равной частоте).

    При склейке: рёбра объединяются (суммируются counts, confidence,
    papers, evidence_quotes), self-loops удаляются.

    Настройки:
        entity_normalization_enabled (bool): включить/выключить.
        entity_normalization_threshold (float): порог схожести 0.0-1.0.
            0.85 — хороший баланс (склеивает "credit limit" + "credit limit
            management", но не "credit limit" + "credit card").
            0.75 — агрессивнее, больше склеек, риск ложных.
            0.92 — консервативно, только почти идентичные.

    Сложность: O(n^2) по числу уникальных терминов. При ~1000 терминов <1с.
    Затраты: 0 LLM-вызовов.
    """
    terms = sorted(nodes.keys())
    if len(terms) < 2:
        return edges, nodes

    # Build merge map: term -> canonical
    merge_map: Dict[str, str] = {}
    canonical_set: Set[str] = set()

    for i in range(len(terms)):
        if terms[i] in merge_map:
            continue
        for j in range(i + 1, len(terms)):
            if terms[j] in merge_map:
                continue
            sim = _token_similarity(terms[i], terms[j])
            if sim >= threshold:
                # Keep the one with higher doc_freq, or shorter name
                ni = nodes.get(terms[i])
                nj = nodes.get(terms[j])
                freq_i = ni.doc_freq if ni else 0
                freq_j = nj.doc_freq if nj else 0
                if freq_i >= freq_j:
                    canonical, alias = terms[i], terms[j]
                else:
                    canonical, alias = terms[j], terms[i]
                merge_map[alias] = canonical
                canonical_set.add(canonical)

    if not merge_map:
        return edges, nodes

    console.print(f"  [cyan]Entity normalization: merging {len(merge_map)} aliases into {len(canonical_set)} canonical terms[/cyan]")

    def _resolve(term: str) -> str:
        seen: Set[str] = set()
        while term in merge_map and term not in seen:
            seen.add(term)
            term = merge_map[term]
        return term

    # Rebuild edges with merged terms
    new_edges: Dict[Tuple[str, str, str], EdgeStats] = {}
    for (s, p, o), edge in edges.items():
        ns = _resolve(s)
        no = _resolve(o)
        if ns == no:
            continue  # self-loop after merge
        key = (ns, p, no)
        if key not in new_edges:
            new_edges[key] = EdgeStats(source=ns, target=no, predicate=p, directed=edge.directed)
        target = new_edges[key]
        target.total_count += edge.total_count
        for y, c in edge.yearly_count.items():
            target.yearly_count[y] = target.yearly_count.get(y, 0) + c
        target.confidence_sum += edge.confidence_sum
        target.confidence_n += edge.confidence_n
        for pol, cnt in edge.polarity_counts.items():
            target.polarity_counts[pol] = target.polarity_counts.get(pol, 0) + cnt
        target.papers.update(edge.papers)
        target.evidence_quotes.extend(edge.evidence_quotes)
        target.time_intervals.extend(edge.time_intervals)
        for k, v in edge.features.items():
            if k not in target.features:
                target.features[k] = v

    # Rebuild nodes
    new_nodes: Dict[str, NodeStats] = {}
    for term, node in nodes.items():
        canonical = _resolve(term)
        if canonical not in new_nodes:
            new_nodes[canonical] = NodeStats(term=canonical)
        target = new_nodes[canonical]
        target.doc_freq += node.doc_freq
        for y, c in node.yearly_doc_freq.items():
            target.yearly_doc_freq[y] = target.yearly_doc_freq.get(y, 0) + c

    console.print(f"  [cyan]Graph reduced: {len(nodes)} → {len(new_nodes)} nodes, {len(edges)} → {len(new_edges)} edges[/cyan]")
    return new_edges, new_nodes


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
    heuristic_fallbacks = 0
    llm_disabled_after_timeout = False
    selected_provider, _selected_model = _resolve_llm_selection(llm_provider=llm_provider, llm_model=llm_model)
    use_g4f_async_batch = (
        base_mode == "llm_triplets"
        and selected_provider == "g4f"
        and bool(getattr(settings, "g4f_async_enabled", True))
    )

    total_units_all = sum(len(_iter_paper_units(pr)) for pr in papers)
    global_unit_idx = 0

    # Каноническая лексика: общий словарь терминов для кросс-документной связности
    _canonical_vocab: List[str] = []
    if settings.canonical_vocabulary_enabled and base_mode == "llm_triplets" and len(papers) > 1:
        import time as _time
        _t0_vocab = _time.monotonic()
        console.print("[bold cyan]Building canonical concept vocabulary across all papers...[/bold cyan]")
        _canonical_vocab = _build_canonical_vocabulary(
            papers, domain, llm_provider=llm_provider, llm_model=llm_model,
        )
        _elapsed_vocab = _time.monotonic() - _t0_vocab
        if _canonical_vocab:
            console.print(f"[green]✓ Canonical vocabulary: {len(_canonical_vocab)} concepts in {_elapsed_vocab:.1f}s[/green]")
            for i, c in enumerate(_canonical_vocab):
                console.print(f"  [dim]{i+1}. {c}[/dim]")
        else:
            console.print(f"[yellow]⚠ Canonical vocabulary: 0 concepts in {_elapsed_vocab:.1f}s[/yellow]")

    for paper_idx, pr in enumerate(papers, 1):
        units = _iter_paper_units(pr)
        if not units:
            continue
        paper_term_set: Set[str] = set()
        # Build paper context string for prompt enrichment (improvements A + D)
        _paper_context: Optional[str] = None
        if settings.triplet_paper_context_enabled or _canonical_vocab:
            _ctx_parts = []
            if pr.title:
                _ctx_parts.append(pr.title)
            _abstract_proxy = (pr.text or "")[:settings.triplet_paper_context_max_chars]
            if _abstract_proxy:
                _ctx_parts.append(_abstract_proxy)
            if _canonical_vocab:
                _ctx_parts.append(
                    "Канонический словарь понятий (используй эти термины как сущности, если они подходят): "
                    + ", ".join(_canonical_vocab)
                )
            _paper_context = "\n".join(_ctx_parts) if _ctx_parts else None
        console.print(f"[bold cyan]── Paper {paper_idx}/{len(papers)}: {pr.title[:80]} ({len(units)} units) ──[/bold cyan]")

        # Summary-триплеты: ключевые утверждения на уровне всей статьи
        if settings.paper_summary_triplets_enabled and base_mode == "llm_triplets":
            import time as _time
            _t0_summary = _time.monotonic()
            console.print(f"  [dim]Extracting paper summary triplets...[/dim]")
            summary_triplets = _extract_paper_summary_triplets(
                pr, domain, llm_provider=llm_provider, llm_model=llm_model,
            )
            _elapsed_summary = _time.monotonic() - _t0_summary
            if summary_triplets:
                anchor_unit = PaperEvidenceUnit(
                    unit_id=f"{pr.paper_id}:summary",
                    chunk_id=f"{pr.paper_id}:summary",
                    text=pr.title,
                    source_kind="paper_summary",
                )
                _merge_triplets_into_graph(
                    pr=pr, unit=anchor_unit, triplets=summary_triplets,
                    edges=edges, term_set=paper_term_set,
                )
                console.print(f"  [green]✓ Paper summary: {len(summary_triplets)} anchor triplets in {_elapsed_summary:.1f}s[/green]")
            else:
                console.print(f"  [yellow]⚠ Paper summary: 0 triplets in {_elapsed_summary:.1f}s[/yellow]")

        if use_g4f_async_batch:
            batch_results = _run_async_sync(
                _extract_triplets_batch_async(
                    domain=domain,
                    pr=pr,
                    units=units,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                )
            )
            for unit, triplets, error in batch_results:
                if error is None and triplets:
                    _merge_triplets_into_graph(pr=pr, unit=unit, triplets=triplets, edges=edges, term_set=paper_term_set)
                    continue

                localized_fallbacks += 1
                fallback_triplets = extract_temporal_triplets_localized_fallback(str(unit.text or ""), paper_year=pr.year)
                if fallback_triplets:
                    heuristic_fallbacks += 1
                    if error is not None:
                        llm_failures += 1
                        console.print(
                            f"[yellow]Temporal triplets failed for {pr.paper_id}/{unit.unit_id}: {error}. "
                            "Using localized heuristic triplet fallback for this evidence unit after async g4f retries.[/yellow]"
                        )
                    _merge_triplets_into_graph(pr=pr, unit=unit, triplets=fallback_triplets, edges=edges, term_set=paper_term_set)
                    continue

                if error is not None:
                    llm_failures += 1
                    console.print(
                        f"[yellow]Temporal triplets failed for {pr.paper_id}/{unit.unit_id}: {error}. "
                        "Using localized co-occurrence fallback for this evidence unit after async g4f retries.[/yellow]"
                    )
                _merge_cooccurrence_into_graph(
                    pr=pr,
                    unit=unit,
                    domain=domain,
                    edges=edges,
                    term_set=paper_term_set,
                    max_terms_per_paper=max_terms_per_paper,
                    language=language,
                )
        else:
            for unit_idx, unit in enumerate(units, 1):
                unit_text = str(unit.text or "").strip()
                if not unit_text:
                    continue
                global_unit_idx += 1
                unit_chars = len(unit_text)
                console.print(
                    f"  [dim]chunk {unit_idx}/{len(units)} "
                    f"(global {global_unit_idx}/{total_units_all}) "
                    f"| {unit_chars} chars | {unit.unit_id}[/dim]"
                )
                unit_mode = base_mode

                if unit_mode == "llm_triplets":
                    import time as _time
                    _t0 = _time.monotonic()
                    try:
                        triplets = extract_temporal_triplets(
                            domain=domain.title,
                            chunk_text=unit_text,
                            paper_year=pr.year,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            paper_context=_paper_context,
                        )
                    except TimeoutError as e:
                        _elapsed = _time.monotonic() - _t0
                        llm_failures += 1
                        localized_fallbacks += 1
                        repaired_triplets = _retry_unit_with_vlm_page_repair(
                            domain_title=domain.title,
                            pr=pr,
                            unit=unit,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                        )
                        if repaired_triplets:
                            console.print(
                                f"[green]    ✓ VLM page repair recovered {len(repaired_triplets)} triplets after timeout[/green]"
                            )
                            _merge_triplets_into_graph(pr=pr, unit=unit, triplets=repaired_triplets, edges=edges, term_set=paper_term_set)
                            continue
                        console.print(
                            f"[red]✗ TIMEOUT after {_elapsed:.1f}s for {pr.paper_id}/{unit.unit_id}: {e}. "
                            "Using heuristic fallback for this chunk only, continuing LLM for next chunks.[/red]"
                        )
                        triplets = extract_temporal_triplets_localized_fallback(unit_text, paper_year=pr.year)
                        if triplets:
                            heuristic_fallbacks += 1
                            _merge_triplets_into_graph(pr=pr, unit=unit, triplets=triplets, edges=edges, term_set=paper_term_set)
                            continue
                        unit_mode = "cooccurrence"
                    except Exception as e:
                        _elapsed = _time.monotonic() - _t0
                        llm_failures += 1
                        localized_fallbacks += 1
                        repaired_triplets = _retry_unit_with_vlm_page_repair(
                            domain_title=domain.title,
                            pr=pr,
                            unit=unit,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                        )
                        if repaired_triplets:
                            console.print(
                                f"[green]    ✓ VLM page repair recovered {len(repaired_triplets)} triplets after error[/green]"
                            )
                            _merge_triplets_into_graph(pr=pr, unit=unit, triplets=repaired_triplets, edges=edges, term_set=paper_term_set)
                            continue
                        console.print(f"    [red]✗ LLM error after {_elapsed:.1f}s: {e}[/red]")
                        triplets = extract_temporal_triplets_localized_fallback(unit_text, paper_year=pr.year)
                        if triplets:
                            heuristic_fallbacks += 1
                            console.print(
                                f"[yellow]Temporal triplets failed for {pr.paper_id}/{unit.unit_id}: {e}. "
                                "Using localized heuristic triplet fallback for this evidence unit only.[/yellow]"
                            )
                            _merge_triplets_into_graph(pr=pr, unit=unit, triplets=triplets, edges=edges, term_set=paper_term_set)
                            continue
                        console.print(
                            f"[yellow]Temporal triplets failed for {pr.paper_id}/{unit.unit_id}: {e}. "
                            "Using localized co-occurrence fallback for this evidence unit only.[/yellow]"
                        )
                        unit_mode = "cooccurrence"
                    else:
                        _elapsed = _time.monotonic() - _t0
                        if triplets:
                            console.print(
                                f"    [green]✓ LLM ok: {len(triplets)} triplets in {_elapsed:.1f}s[/green]"
                            )
                            _merge_triplets_into_graph(pr=pr, unit=unit, triplets=triplets, edges=edges, term_set=paper_term_set)
                            continue
                        console.print(
                            f"    [yellow]⚠ LLM returned 0 triplets in {_elapsed:.1f}s → heuristic fallback[/yellow]"
                        )
                        localized_fallbacks += 1
                        triplets = extract_temporal_triplets_localized_fallback(unit_text, paper_year=pr.year)
                        if triplets:
                            heuristic_fallbacks += 1
                            _merge_triplets_into_graph(pr=pr, unit=unit, triplets=triplets, edges=edges, term_set=paper_term_set)
                            continue
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

    # Fuzzy-нормализация сущностей: дедупликация похожих вершин
    if settings.entity_normalization_enabled:
        edges, nodes = _normalize_entities_fuzzy(edges, nodes, threshold=settings.entity_normalization_threshold)

    # Верификация рёбер: confidence gate + LLM-проверка
    _rejected_for_training: List[Dict[str, Any]] = []
    _verify_rejected: List[EdgeStats] = []
    if settings.triplet_verify_enabled and base_mode == "llm_triplets":
        import time as _time
        _t0_verify = _time.monotonic()
        console.print("[bold cyan]Верификация рёбер...[/bold cyan]")
        from .triplet_verifier import verify_edges
        edges, _verify_rejected, _vresult = verify_edges(
            edges,
            papers,
            confidence_threshold=settings.triplet_verify_confidence_threshold,
            batch_size=settings.triplet_verify_batch_size,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        _elapsed_verify = _time.monotonic() - _t0_verify
        console.print(f"  [green]✓ Verification in {_elapsed_verify:.1f}s: {_vresult.summary_line()}[/green]")
        # Rebuild nodes to remove orphans from rejected edges
        _active_terms: Set[str] = set()
        for e in edges.values():
            _active_terms.add(e.source)
            _active_terms.add(e.target)
        nodes = {t: n for t, n in nodes.items() if t in _active_terms}
        # Store rejected edges for scorer training (as negatives)
        _rejected_for_training: List[Dict[str, Any]] = []
        for re in _verify_rejected:
            _rejected_for_training.append({
                "source": re.source, "target": re.target, "predicate": re.predicate,
                "total_count": re.total_count, "mean_confidence": re.mean_confidence,
                "polarity_counts": re.polarity_counts,
                "papers": sorted(list(re.papers))[:10],
                "evidence_quotes": re.evidence_quotes[:5],
                "features": re.features, "score": re.score,
            })

    # Assertion quality scoring (logistic regression on observable features) or legacy linear scoring
    _use_scorer = getattr(settings, "assertion_scorer_enabled", False)
    _quality_scorer = None
    _scorer_stats = None
    if _use_scorer:
        try:
            from .assertion_scorer import AssertionScorer, compute_corpus_stats as _compute_corpus_stats
            _quality_scorer = AssertionScorer.load()
            _scorer_stats = _compute_corpus_stats(edges.values(), nodes, n_docs)
            console.print("[cyan]Using assertion quality scoring[/cyan]")
        except Exception as _scorer_err:
            console.print(f"[yellow]Assertion scoring unavailable ({_scorer_err}), falling back to legacy[/yellow]")
            _use_scorer = False

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

        if _use_scorer and _quality_scorer is not None and _scorer_stats is not None:
            e.score = _quality_scorer.score_edge(e, nodes, _scorer_stats)
            e.features["quality_score"] = e.score
            # Expert overrides as additive bonus
            if expert_w != 0.0:
                e.score = max(0.0, min(1.0, e.score + 0.15 * expert_w))
        else:
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
            "heuristic_fallbacks": heuristic_fallbacks,
            "llm_failures": llm_failures,
            "llm_disabled_after_timeout": llm_disabled_after_timeout,
            "g4f_async_batch": use_g4f_async_batch,
            "g4f_async_max_concurrency": int(getattr(settings, "g4f_async_max_concurrency", 3) or 3) if use_g4f_async_batch else 0,
            "multimodal_enabled": any(bool(getattr(pr, "multimodal_text", "").strip()) for pr in papers),
            "verify_enabled": bool(settings.triplet_verify_enabled),
            "verify_rejected_count": len(_verify_rejected),
            "verify_rejected_edges": _rejected_for_training if settings.triplet_verify_enabled else [],
        },
    )
