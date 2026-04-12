from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import yaml  # type: ignore
from rich.console import Console

from ..config import settings
from ..contracts import ChunkRecord
from ..domain import load_domain_config
from ..graph.review_applier import compile_overrides
from ..hypotheses.temporal_graph_hypotheses import HypothesisCandidate, generate_candidates
from ..index.annoy_store import AnnoyBundle, build_annoy_index, search_annoy_index
from ..ingest.acquire import AcquireResult, acquire_pdfs
from ..ingest.mm_pipeline import ingest_pdf_multimodal_auto
from ..ingest.pipeline import ingest_pdf_auto
from ..ingest.store import save_paper
from ..llm import chat_json, embed, temporary_llm_selection
from ..mm.multimodal_triplets import MultimodalTripletArtifact, dump_multimodal_triplets, extract_multimodal_triplets
from ..mm.vlm import describe_image, temporary_vlm_selection
from ..papers.resolver import resolve_ids
from ..papers.schema import ExternalIds, PaperMetadata, PaperSource
from ..papers.service import DEFAULT_SOURCES, get_paper_by_doi, search_papers
from ..pipeline.task2_validation import resolve_papers_from_trajectory
from ..reward.rule_based import RuleBasedReward
from ..schemas import Citation, HypothesisDraft
from ..temporal.temporal_kg_builder import EdgeStats, PaperRecord, TemporalKnowledgeGraph, build_temporal_kg, load_papers_from_processed
from ..tgnn.event_dataset import build_event_stream
from ..tgnn.pygt_temporal_link_prediction import PyGTemporalLinkPredConfig, PyGTemporalUnavailableError, pygt_temporal_link_prediction
from ..tgnn.tgn_link_prediction import TGNLinkPredConfig, tgn_link_prediction
from ..tgnn.prediction_types import LinkPredictionRecord, SemanticEdgeKey, same_predicate_family, normalize_predicate, normalize_term

try:  # pragma: no cover
    from ..mm.mm_embed import embed_images as mm_embed_images
except Exception:  # pragma: no cover
    mm_embed_images = None


console = Console()


TASK3_HYP_SCHEMA_HINT = """Ожидается JSON объект HypothesisDraft:
{
  "title": "...",
  "premise": "...",
  "mechanism": "...",
  "time_scope": "...",
  "proposed_experiment": "...",
  "supporting_evidence": [{"source_id":"...","text_snippet":"..."}],
  "confidence_score": 1-10
}
"""


@dataclass
class Task3BundleResult:
    bundle_dir: Path
    manifest_path: Path
    hypotheses_path: Path


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9\-\s_]+", "", (text or "").strip().lower())
    value = re.sub(r"\s+", "-", value).strip("-")
    return value[:80] or "task3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _looks_like_url(value: str) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
    except Exception:
        return False
    return bool(parsed.scheme and parsed.netloc)


def _rank_papers(papers: Sequence[PaperMetadata], query: str) -> List[PaperMetadata]:
    q = set(re.findall(r"[a-z0-9]+", (query or "").lower()))

    def score(p: PaperMetadata) -> float:
        text = f"{p.title} {p.abstract or ''}".lower()
        toks = set(re.findall(r"[a-z0-9]+", text))
        overlap = (len(q & toks) / float(max(1, len(q)))) if q else 0.0
        cites = float(p.citation_count or 0)
        year = float(p.year or 0)
        has_pdf = 1.0 if p.pdf_url else 0.0
        return 3.0 * overlap + 0.0008 * cites + 0.0005 * year + 0.6 * has_pdf

    return sorted(list(papers), key=score, reverse=True)


def _dedupe_papers(papers: Iterable[PaperMetadata]) -> List[PaperMetadata]:
    out: List[PaperMetadata] = []
    seen: set[str] = set()
    for paper in papers:
        if not paper.id or paper.id in seen:
            continue
        seen.add(paper.id)
        out.append(paper)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Task 3 input YAML must contain a top-level object")
    return payload


def _parse_identifier_blob(text: str) -> List[str]:
    parts: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for token in re.split(r"[,;]", line):
            token = token.strip()
            if token:
                parts.append(token)
    return parts


def _fallback_metadata_from_identifier(identifier: str) -> PaperMetadata:
    identifier = str(identifier or "").strip()
    ids = ExternalIds()
    url = identifier if _looks_like_url(identifier) else None
    pdf_url = identifier if str(identifier).lower().endswith(".pdf") else None
    rid = resolve_ids(identifier)
    if rid.doi:
        ids.doi = rid.doi
    if rid.pmid:
        ids.pmid = rid.pmid
    if rid.pmcid:
        ids.pmcid = rid.pmcid
    if rid.arxiv:
        ids.arxiv = rid.arxiv
    if rid.openalex:
        ids.openalex = rid.openalex
    canonical = ids.best_canonical() or (identifier if not _looks_like_url(identifier) else f"url:{_slugify(identifier)}")
    title = identifier if not _looks_like_url(identifier) else urlparse(identifier).path.split("/")[-1] or identifier
    return PaperMetadata(
        id=canonical,
        source=PaperSource.unknown,
        title=title,
        url=url,
        pdf_url=pdf_url,
        ids=ids,
        raw={"identifier_input": identifier},
    )


def _candidate_matches_identifier(identifier: str, paper: PaperMetadata) -> bool:
    raw = str(identifier or "").strip().lower()
    if not raw:
        return False
    ids = paper.ids
    values = [
        str(paper.id or "").lower(),
        str(ids.doi or "").lower(),
        str(ids.pmid or "").lower(),
        str(ids.pmcid or "").lower(),
        str(ids.arxiv or "").lower(),
        str(ids.openalex or "").lower(),
        str(paper.url or "").lower(),
        str(paper.pdf_url or "").lower(),
        str(paper.title or "").lower(),
    ]
    return any(raw == val or raw in val for val in values if val)


def _resolve_papers_from_identifiers(identifiers: Sequence[str], *, search_limit: int = 8) -> List[PaperMetadata]:
    resolved: List[PaperMetadata] = []
    for identifier in identifiers:
        raw = str(identifier or "").strip()
        if not raw:
            continue

        meta: Optional[PaperMetadata] = None
        rid = resolve_ids(raw)
        if rid.doi:
            try:
                meta = get_paper_by_doi(rid.doi)
            except Exception:
                meta = None

        if meta is None:
            try:
                candidates = search_papers(raw, limit=search_limit, sources=DEFAULT_SOURCES)
            except Exception:
                candidates = []
            for candidate in candidates:
                if _candidate_matches_identifier(raw, candidate):
                    meta = candidate
                    break
            if meta is None and candidates:
                meta = candidates[0]

        if meta is None:
            meta = _fallback_metadata_from_identifier(raw)
        resolved.append(meta)

    return _dedupe_papers(resolved)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _paperrecord_from_metadata(paper: PaperMetadata) -> PaperRecord:
    text = (paper.abstract or "").strip() or paper.title
    return PaperRecord(
        paper_id=paper.id,
        title=paper.title,
        year=paper.year,
        text=text,
        url=paper.url or "",
        source=str(paper.source),
    )


def _ingest_metadata_only(paper: PaperMetadata, processed_dir: Path) -> Path:
    pieces = [paper.title.strip()]
    if paper.abstract:
        pieces.append(str(paper.abstract).strip())
    if paper.url:
        pieces.append(f"Source URL: {paper.url}")
    fallback_text = "\n\n".join(piece for piece in pieces if piece).strip() or paper.title
    chunk = ChunkRecord(
        chunk_id=f"{paper.id}:metadata",
        paper_id=paper.id,
        text=fallback_text,
        modality="text",
        source_backend="metadata_fallback",
        reading_order=0,
    )
    return save_paper(processed_dir, meta=paper.model_dump(mode="json"), chunks=[chunk])


def _copy_processed_tree(src: Path, dst: Path) -> Path:
    if src.resolve() == dst.resolve():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    return dst


def _acquire_and_ingest_papers(
    papers: Sequence[PaperMetadata],
    *,
    bundle_dir: Path,
    include_multimodal: bool,
    run_vlm: bool,
) -> tuple[Path, List[AcquireResult]]:
    raw_dir = bundle_dir / "raw_pdfs"
    meta_dir = bundle_dir / "raw_meta"
    processed_dir = bundle_dir / "processed_papers"
    processed_dir.mkdir(parents=True, exist_ok=True)

    acquire_results = acquire_pdfs(papers, raw_dir=raw_dir, meta_dir=meta_dir)
    for paper, result in zip(papers, acquire_results):
        if result.pdf_path:
            try:
                if include_multimodal:
                    ingest_pdf_multimodal_auto(result.pdf_path, paper.model_dump(mode="json"), processed_dir, run_vlm=run_vlm)
                else:
                    ingest_pdf_auto(result.pdf_path, paper.model_dump(mode="json"), processed_dir)
                continue
            except Exception as exc:
                console.print(f"[yellow]Task3 ingest fallback for {paper.id}: {type(exc).__name__}: {exc}[/yellow]")
        _ingest_metadata_only(paper, processed_dir)

    return processed_dir, acquire_results


def _load_chunk_registry(processed_dir: Path) -> List[ChunkRecord]:
    out: List[ChunkRecord] = []
    for paper_dir in sorted(path for path in processed_dir.iterdir() if path.is_dir()):
        meta_path = paper_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        paper_id = str(meta.get("id") or paper_dir.name)
        paper_year = meta.get("year")
        paper_title = str(meta.get("title") or "").strip()

        chunks_path = paper_dir / "chunks.jsonl"
        if chunks_path.exists():
            for idx, line in enumerate(chunks_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    raw = {"text": line}
                if not isinstance(raw, dict):
                    raw = {"text": str(raw)}
                raw.setdefault("chunk_id", f"{paper_id}:{idx}")
                raw.setdefault("paper_id", paper_id)
                raw.setdefault("modality", "text")
                raw.setdefault("source_backend", "unknown")
                record = ChunkRecord.model_validate(raw)
                metadata = dict(record.metadata or {})
                metadata.setdefault("paper_year", paper_year)
                metadata.setdefault("paper_title", paper_title)
                metadata.setdefault("paper_dir", str(paper_dir))
                record.metadata = metadata
                out.append(record)

        mm_pages = paper_dir / "mm" / "pages.jsonl"
        if mm_pages.exists():
            for idx, line in enumerate(mm_pages.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                page_value = raw.get("page")
                try:
                    page = int(page_value) if page_value is not None else None
                except Exception:
                    page = None
                image_path = str(raw.get("image_path") or "")
                base_chunk_id = f"{paper_id}:mm:{page if page is not None else idx}"
                metadata = {
                    "paper_year": paper_year,
                    "paper_title": paper_title,
                    "paper_dir": str(paper_dir),
                    "vlm_caption": str(raw.get("vlm_caption") or "").strip(),
                    "tables_md": str(raw.get("tables_md") or "").strip(),
                    "equations_md": str(raw.get("equations_md") or "").strip(),
                }
                page_text = str(raw.get("text") or "").strip()
                out.append(
                    ChunkRecord(
                        chunk_id=base_chunk_id,
                        paper_id=paper_id,
                        page=page,
                        modality="page",
                        text=page_text,
                        image_path=image_path or None,
                        source_backend="mm_pdf",
                        metadata=metadata,
                    )
                )
                if metadata["tables_md"]:
                    out.append(
                        ChunkRecord(
                            chunk_id=f"{base_chunk_id}:table",
                            paper_id=paper_id,
                            page=page,
                            modality="table",
                            text=metadata["tables_md"],
                            image_path=image_path or None,
                            source_backend="mm_pdf",
                            metadata=metadata,
                        )
                    )
                if metadata["equations_md"]:
                    out.append(
                        ChunkRecord(
                            chunk_id=f"{base_chunk_id}:formula",
                            paper_id=paper_id,
                            page=page,
                            modality="formula",
                            text=metadata["equations_md"],
                            image_path=image_path or None,
                            source_backend="mm_pdf",
                            metadata=metadata,
                        )
                    )

    deduped: List[ChunkRecord] = []
    seen: set[str] = set()
    for record in out:
        if record.chunk_id in seen:
            continue
        seen.add(record.chunk_id)
        deduped.append(record)
    return deduped


def _chunk_retrieval_text(record: ChunkRecord) -> str:
    parts: List[str] = []
    if record.text:
        parts.append(str(record.text).strip())
    if record.table_md:
        parts.append(f"Table: {record.table_md}")
    for key, prefix in (("vlm_caption", "Visual"), ("tables_md", "Tables"), ("equations_md", "Equations")):
        val = str((record.metadata or {}).get(key) or "").strip()
        if val:
            parts.append(f"{prefix}: {val}")
    payload = "\n\n".join(part for part in parts if part).strip()
    if payload:
        return payload
    if record.image_path:
        return f"Image chunk from {record.paper_id} page {record.page or 'unknown'}"
    return record.chunk_id


def _pad_vector(values: Sequence[float], dim: int) -> List[float]:
    row = list(float(v) for v in values)
    if len(row) == dim:
        return row
    if len(row) > dim:
        return row[:dim]
    return row + [0.0] * (dim - len(row))


def _merge_dense_vectors(text_vectors: List[List[float]], image_vectors: Dict[int, List[float]]) -> List[List[float]]:
    if not text_vectors and not image_vectors:
        return []
    dim = 0
    for vec in text_vectors:
        dim = max(dim, len(vec))
    for vec in image_vectors.values():
        dim = max(dim, len(vec))
    if dim <= 0:
        return text_vectors

    merged: List[List[float]] = []
    for idx, text_vec in enumerate(text_vectors):
        t = _pad_vector(text_vec, dim)
        if idx not in image_vectors:
            merged.append(t)
            continue
        i = _pad_vector(image_vectors[idx], dim)
        merged.append([0.7 * tv + 0.3 * iv for tv, iv in zip(t, i)])
    return merged


def _build_chunk_embeddings(chunk_records: Sequence[ChunkRecord]) -> tuple[List[List[float]], List[Dict[str, Any]]]:
    texts = [_chunk_retrieval_text(record) for record in chunk_records]
    text_vectors = embed(texts) if texts else []

    image_vectors: Dict[int, List[float]] = {}
    use_mm_images = mm_embed_images is not None and str(getattr(settings, "mm_embed_backend", "none") or "none").lower() != "none"
    if use_mm_images:
        image_rows = [(idx, Path(str(record.image_path))) for idx, record in enumerate(chunk_records) if record.image_path]
        if image_rows:
            valid_rows = [(idx, path) for idx, path in image_rows if path.exists()]
            if valid_rows:
                try:
                    embeds = mm_embed_images([path for _, path in valid_rows])
                    for (idx, _), vec in zip(valid_rows, embeds):
                        image_vectors[idx] = list(vec)
                except Exception:
                    image_vectors = {}

    vectors = _merge_dense_vectors(text_vectors, image_vectors)
    payloads: List[Dict[str, Any]] = []
    for idx, record in enumerate(chunk_records):
        payloads.append(
            {
                "item_id": record.chunk_id,
                "chunk_id": record.chunk_id,
                "paper_id": record.paper_id,
                "page": record.page,
                "modality": record.modality,
                "image_path": record.image_path,
                "text": _chunk_retrieval_text(record)[:1200],
                "embedding_source": "text+image" if idx in image_vectors else "text",
            }
        )
    return vectors, payloads


def _edge_for_candidate(kg: TemporalKnowledgeGraph, candidate: HypothesisCandidate) -> Optional[EdgeStats]:
    for edge in kg.edges:
        if edge.source == candidate.source and edge.target == candidate.target and edge.predicate == candidate.predicate:
            return edge
    return None


def _candidate_temporal_context(kg: TemporalKnowledgeGraph, candidate: HypothesisCandidate) -> Dict[str, Any]:
    edge = _edge_for_candidate(kg, candidate)
    if edge is None:
        return {
            "time_scope": candidate.time_scope,
            "ordering": "predicted_or_missing",
            "yearly_count": {},
            "first_seen": None,
            "last_seen": None,
            "time_intervals": [],
        }

    yearly = dict(sorted(edge.yearly_count.items()))
    years = sorted(yearly)
    first_seen = years[0] if years else None
    last_seen = years[-1] if years else None
    ordering = "stable"
    if len(years) >= 2:
        first_count = yearly[years[0]]
        last_count = yearly[years[-1]]
        if last_count > first_count:
            ordering = "strengthening"
        elif last_count < first_count:
            ordering = "weakening"
        else:
            ordering = "persistent"
    if candidate.kind.endswith("missing_link"):
        ordering = "predicted_missing_link"
    return {
        "time_scope": candidate.time_scope,
        "ordering": ordering,
        "yearly_count": yearly,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "time_intervals": edge.time_intervals[:12],
    }


def _candidate_semantic_key(candidate: HypothesisCandidate) -> SemanticEdgeKey:
    return SemanticEdgeKey(
        source=candidate.source,
        predicate=candidate.predicate,
        target=candidate.target,
        direction="directed",
        time_bucket=(str(candidate.time_scope or "").strip() or None),
    ).normalized()


def _match_link_prediction(candidate: HypothesisCandidate, predictions: Sequence[LinkPredictionRecord]) -> Optional[LinkPredictionRecord]:
    ck = _candidate_semantic_key(candidate)

    # 1) exact semantic match
    for pred in predictions:
        if pred.edge_key() == ck:
            return pred

    # 2) exact nodes + predicate family match (temporal bucket may differ)
    for pred in predictions:
        pk = pred.edge_key()
        if pk.source == ck.source and pk.target == ck.target and same_predicate_family(pk.predicate, ck.predicate):
            return pred

    # 3) undirected relaxation only for undirected predicates
    if ck.direction == "undirected" or normalize_predicate(candidate.predicate) == "cooccurs_with":
        for pred in predictions:
            pk = pred.edge_key()
            if pk.direction != "undirected":
                continue
            if {pk.source, pk.target} == {ck.source, ck.target} and same_predicate_family(pk.predicate, ck.predicate):
                return pred

    return None


def _term_match(term: str, value: str) -> bool:
    t = str(term or "").strip().lower()
    v = str(value or "").strip().lower()
    return bool(t and v and (t == v or t in v or v in t))


def _match_multimodal_triplets(candidate: HypothesisCandidate, records: Sequence[MultimodalTripletArtifact], *, limit: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records:
        for triplet in record.triplets:
            if (
                (_term_match(candidate.source, triplet.subject) and _term_match(candidate.target, triplet.object))
                or (_term_match(candidate.source, triplet.object) and _term_match(candidate.target, triplet.subject))
                or (_term_match(candidate.source, triplet.subject) and _term_match(candidate.predicate, triplet.predicate))
            ):
                out.append(
                    {
                        "paper_id": record.paper_id,
                        "chunk_id": record.chunk_id,
                        "modality": record.modality,
                        "page": record.page,
                        "image_path": record.image_path,
                        "analysis_text": record.analysis_text[:1200],
                        "triplet": triplet.model_dump(mode="json"),
                    }
                )
                if len(out) >= limit:
                    return out
    return out


def _candidate_neighbor_evidence(
    candidate: HypothesisCandidate,
    *,
    annoy_bundle: AnnoyBundle,
    top_k: int,
) -> List[Dict[str, Any]]:
    query_text = f"{candidate.source} {candidate.predicate} {candidate.target} {candidate.time_scope}".strip()
    query_vec = embed([query_text])[0]
    return search_annoy_index(annoy_bundle, query_vec, top_k=top_k)


def _candidate_vlm_analyses(
    candidate: HypothesisCandidate,
    neighbors: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    run_vlm: bool,
    vlm_backend: Optional[str],
    vlm_model_id: Optional[str],
) -> List[Dict[str, Any]]:
    if not run_vlm:
        return []
    out: List[Dict[str, Any]] = []
    for row in neighbors:
        image_path = str(row.get("image_path") or "").strip()
        if not image_path:
            continue
        path = Path(image_path)
        if not path.exists():
            continue
        prompt = (
            "Ты проверяешь научную гипотезу на основе мультимодального свидетельства. "
            f"Оцени, содержит ли изображение/страница доказательство для связи '{candidate.source}' | '{candidate.predicate}' | '{candidate.target}'. "
            "Особенно отметь временные маркеры, последовательность событий, динамику во времени и количественные показатели. "
            "Не делай выводов, которых нет на изображении."
        )
        try:
            res = describe_image(path, prompt=prompt, backend=vlm_backend, model_id=vlm_model_id)
        except Exception:
            continue
        if not res.caption and not res.extracted_tables_md and not res.extracted_equations_md:
            continue
        out.append(
            {
                "chunk_id": row.get("chunk_id") or row.get("item_id"),
                "image_path": image_path,
                "caption": res.caption,
                "tables_md": res.extracted_tables_md,
                "equations_md": res.extracted_equations_md,
            }
        )
        if len(out) >= top_k:
            break
    return out


def _citations_from_candidate_context(
    candidate: HypothesisCandidate,
    *,
    neighbors: Sequence[Dict[str, Any]],
    matched_triplets: Sequence[Dict[str, Any]],
    limit: int = 6,
) -> List[Citation]:
    citations: List[Citation] = []
    seen: set[Tuple[str, str]] = set()

    for cite in candidate.evidence:
        key = (cite.source_id, cite.text_snippet)
        if key in seen:
            continue
        seen.add(key)
        citations.append(cite)
        if len(citations) >= limit:
            return citations

    for row in neighbors:
        source_id = str(row.get("paper_id") or row.get("chunk_id") or row.get("item_id") or "")
        snippet = str(row.get("text") or row.get("analysis_text") or "").strip()
        if not source_id or not snippet:
            continue
        key = (source_id, snippet)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(source_id=source_id, text_snippet=snippet[:220]))
        if len(citations) >= limit:
            return citations

    for row in matched_triplets:
        source_id = str(row.get("paper_id") or row.get("chunk_id") or "")
        triplet = row.get("triplet") if isinstance(row.get("triplet"), dict) else {}
        snippet = str(triplet.get("evidence_quote") or row.get("analysis_text") or "").strip()
        if not source_id or not snippet:
            continue
        key = (source_id, snippet)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(source_id=source_id, text_snippet=snippet[:220]))
        if len(citations) >= limit:
            return citations

    return citations


def _template_hypothesis_from_context(
    candidate: HypothesisCandidate,
    *,
    temporal_context: Dict[str, Any],
    link_prediction: Optional[LinkPredictionRecord],
    supporting_evidence: Sequence[Citation],
    matched_triplets: Sequence[Dict[str, Any]],
) -> HypothesisDraft:
    ordering = str(temporal_context.get("ordering") or candidate.time_scope or "recent evidence")
    pred_bits = ""
    if link_prediction is not None:
        pred_bits = f" Дополнительный сигнал link prediction={link_prediction.score:.3f}."
    title = f"{candidate.source} {candidate.predicate} {candidate.target}"
    premise = (
        f"Темпоральный граф знаний указывает на связь '{candidate.source}' -> '{candidate.target}' через '{candidate.predicate}'. "
        f"Упорядочивание во времени классифицировано как '{ordering}'.{pred_bits}"
    )
    mechanism = (
        "Гипотеза предполагает, что связь проявляется не изолированно, а через темпорально упорядоченный механизм: "
        "ранние события/сигналы закрепляют последующие наблюдения, а мультимодальные фрагменты (текст, таблицы, изображения) "
        "дают согласующиеся признаки этой динамики."
    )
    if matched_triplets:
        mechanism += " В мультимодальных триплетах уже есть частичные подтверждения направления эффекта."
    proposed_experiment = (
        "Соберите независимую выборку/экспериментальную серию, где можно измерить субъект, объект и временной лаг между ними; "
        "зафиксируйте временные метки, выполните сравнение по временным окнам, а затем проведите абляцию ключевого посредника и репликацию."
    )
    confidence = 5
    confidence += min(2, len(supporting_evidence) // 2)
    if link_prediction is not None and link_prediction.score >= 0.2:
        confidence += 1
    confidence = max(1, min(10, confidence))
    return HypothesisDraft(
        title=title,
        premise=premise,
        mechanism=mechanism,
        time_scope=str(temporal_context.get("time_scope") or candidate.time_scope or ""),
        proposed_experiment=proposed_experiment,
        supporting_evidence=list(supporting_evidence),
        confidence_score=confidence,
    )


def _llm_hypothesis_from_context(
    candidate: HypothesisCandidate,
    *,
    query: str,
    domain: str,
    temporal_context: Dict[str, Any],
    link_prediction: Optional[LinkPredictionRecord],
    neighbors: Sequence[Dict[str, Any]],
    matched_triplets: Sequence[Dict[str, Any]],
    vlm_analyses: Sequence[Dict[str, Any]],
    supporting_evidence: Sequence[Citation],
) -> HypothesisDraft:
    evidence_block = "\n".join(f"- ({cite.source_id}) {cite.text_snippet}" for cite in supporting_evidence[:8])
    triplet_block = "\n".join(
        "- " + json.dumps(row.get("triplet") or {}, ensure_ascii=False)
        for row in matched_triplets[:6]
    )
    neighbor_block = "\n".join(
        f"- {row.get('chunk_id') or row.get('item_id')} | modality={row.get('modality')} | score={row.get('score', row.get('distance', 0))} | text={str(row.get('text') or '')[:260]}"
        for row in neighbors[:6]
    )
    vlm_block = "\n".join(
        f"- {row.get('chunk_id')} | caption={str(row.get('caption') or '')[:260]}"
        for row in vlm_analyses[:4]
    )
    pred_payload = link_prediction.to_json_dict() if link_prediction is not None else None

    system = (
        f"Ты научный ассистент для Task 3. Домен: {domain}. "
        "Сформулируй только проверяемую, falsifiable гипотезу. "
        "Обязательно используй time_scope и временное упорядочивание событий. "
        "Не выдумывай факты сверх переданного контекста."
    )
    user = (
        f"Research topic/query: {query}\n\n"
        f"Candidate relation: {candidate.source} | {candidate.predicate} | {candidate.target}\n"
        f"Candidate kind: {candidate.kind}\n"
        f"Graph score: {candidate.score}\n"
        f"Graph signals: {json.dumps(candidate.graph_signals, ensure_ascii=False)}\n"
        f"Temporal context: {json.dumps(temporal_context, ensure_ascii=False)}\n"
        f"Predicted future/missing link: {json.dumps(pred_payload, ensure_ascii=False)}\n\n"
        f"Annoy neighbors:\n{neighbor_block or '-'}\n\n"
        f"Multimodal triplets:\n{triplet_block or '-'}\n\n"
        f"Candidate-specific VLM analyses:\n{vlm_block or '-'}\n\n"
        f"Supporting evidence:\n{evidence_block or '-'}\n\n"
        "Сформулируй одну гипотезу, укажи предполагаемый механизм, time_scope и экспериментальную проверку "
        "с измеримыми метриками и контролями."
    )
    data = chat_json(system=system, user=user, schema_hint=TASK3_HYP_SCHEMA_HINT, temperature=0.2)
    return HypothesisDraft.model_validate(data)


def _positive_squash(value: float, *, scale: float = 1.0) -> float:
    v = max(0.0, float(value or 0.0))
    s = max(1e-6, float(scale or 1.0))
    return 1.0 - (2.718281828459045 ** (-v / s))


def _std(values: Sequence[float]) -> float:
    xs = [float(v) for v in values]
    if not xs:
        return 0.0
    mean = sum(xs) / float(len(xs))
    return (sum((x - mean) ** 2 for x in xs) / float(len(xs))) ** 0.5


def _hypothesis_calibration_payload(
    *,
    candidate: HypothesisCandidate,
    draft: HypothesisDraft,
    temporal_context: Mapping[str, Any],
    prediction_score: float,
    retrieval_score: float,
    multimodal_score: float,
    reward_score: float,
    neighbors: Sequence[Dict[str, Any]],
    matched_triplets: Sequence[Dict[str, Any]],
    vlm_analyses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_components = [
        _positive_squash(float(candidate.score), scale=6.0),
        _positive_squash(float(prediction_score), scale=0.75),
        _positive_squash(float(multimodal_score), scale=0.8),
        _positive_squash(float(retrieval_score), scale=0.8),
        _positive_squash(float(reward_score), scale=3.0),
    ]
    agreement = max(0.0, 1.0 - min(1.0, _std(normalized_components)))
    support_density = min(1.0, (len(neighbors[:6]) + len(matched_triplets[:6]) + len(vlm_analyses[:4])) / 10.0)
    evidence_density = min(1.0, len(list(getattr(draft, 'supporting_evidence', []) or [])) / 4.0)
    ordering = str(temporal_context.get('ordering') or '')
    temporal_grounding = 1.0 if ordering in {'stable', 'strengthening', 'weakening', 'persistent', 'predicted_missing_link'} else 0.45
    calibration_score = 0.4 * agreement + 0.25 * support_density + 0.2 * evidence_density + 0.15 * temporal_grounding
    return {
        'calibration_score': float(calibration_score),
        'calibration_components': {
            'agreement': float(agreement),
            'support_density': float(support_density),
            'evidence_density': float(evidence_density),
            'temporal_grounding': float(temporal_grounding),
        },
    }


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    n = min(len(left), len(right))
    if n <= 0:
        return 0.0
    dot = sum(float(left[i]) * float(right[i]) for i in range(n))
    ln = sum(float(left[i]) ** 2 for i in range(n)) ** 0.5
    rn = sum(float(right[i]) ** 2 for i in range(n)) ** 0.5
    if ln <= 0.0 or rn <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (ln * rn)))


def _ranking_text(row: Mapping[str, Any]) -> str:
    hyp = row.get('hypothesis') or {}
    cand = row.get('candidate') or {}
    parts = [
        str(hyp.get('title') or ''),
        str(hyp.get('premise') or ''),
        str(hyp.get('mechanism') or ''),
        str(cand.get('source') or ''),
        str(cand.get('predicate') or ''),
        str(cand.get('target') or ''),
    ]
    return ' '.join(part for part in parts if part).strip()


def _candidate_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    lc = left.get('candidate') or {}
    rc = right.get('candidate') or {}
    edge_overlap = 0.0
    if str(lc.get('source') or '') == str(rc.get('source') or '') and str(lc.get('target') or '') == str(rc.get('target') or ''):
        edge_overlap = 1.0
    elif same_predicate_family(lc.get('predicate') or '', rc.get('predicate') or '') and ({str(lc.get('source') or ''), str(lc.get('target') or '')} & {str(rc.get('source') or ''), str(rc.get('target') or '')}):
        edge_overlap = 0.7
    lv = left.get('_rank_vec') or []
    rv = right.get('_rank_vec') or []
    return max(edge_overlap, _cosine_similarity(lv, rv))


def _apply_mmr_rerank(records: Sequence[Dict[str, Any]], *, top_k: int, lambda_mult: float = 0.72) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in records]
    if not rows:
        return []
    texts = [_ranking_text(row) for row in rows]
    vecs = embed(texts) if texts else []
    for row, vec in zip(rows, vecs):
        row['_rank_vec'] = vec
        row['_relevance'] = float(row.get('calibrated_final_score') or row.get('final_score') or 0.0)
    remaining = rows[:]
    selected: list[dict[str, Any]] = []
    lam = max(0.0, min(1.0, float(lambda_mult)))
    while remaining and len(selected) < int(top_k):
        best_row = None
        best_score = None
        for row in remaining:
            relevance = float(row.get('_relevance') or 0.0)
            redundancy = max((_candidate_similarity(row, picked) for picked in selected), default=0.0)
            mmr_score = lam * relevance - (1.0 - lam) * redundancy
            row['_mmr_score'] = float(mmr_score)
            row['_redundancy'] = float(redundancy)
            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_row = row
        assert best_row is not None
        remaining.remove(best_row)
        selected.append(best_row)
    for rank, row in enumerate(selected, start=1):
        row['diversity_rank'] = rank
        row['diversity_penalty'] = float(row.pop('_redundancy', 0.0))
        row['mmr_score'] = float(row.pop('_mmr_score', row.get('_relevance', 0.0)))
        row.pop('_rank_vec', None)
        row.pop('_relevance', None)
    return selected


def _score_hypothesis_record(
    *,
    candidate: HypothesisCandidate,
    draft: HypothesisDraft,
    reward: RuleBasedReward,
    temporal_context: Mapping[str, Any],
    link_prediction: Optional[LinkPredictionRecord],
    neighbors: Sequence[Dict[str, Any]],
    matched_triplets: Sequence[Dict[str, Any]],
    vlm_analyses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    neighbor_strengths: List[float] = []
    for row in neighbors[:6]:
        if 'score' in row:
            neighbor_strengths.append(float(row.get('score') or 0.0))
        elif 'distance' in row:
            neighbor_strengths.append(max(0.0, 1.0 / (1.0 + float(row.get('distance') or 0.0))))
    retrieval_score = sum(neighbor_strengths) / float(max(1, len(neighbor_strengths)))
    prediction_score = float(link_prediction.score) if link_prediction is not None else 0.0
    multimodal_score = min(1.0, 0.2 * len(matched_triplets) + 0.15 * len(vlm_analyses))
    reward_breakdown = reward.score(draft.model_dump(mode='json'))
    final_score = (
        1.6 * float(candidate.score)
        + 0.9 * prediction_score
        + 0.7 * multimodal_score
        + 0.6 * retrieval_score
        + float(reward_breakdown.score)
    )
    calibration_payload = _hypothesis_calibration_payload(
        candidate=candidate,
        draft=draft,
        temporal_context=temporal_context,
        prediction_score=prediction_score,
        retrieval_score=retrieval_score,
        multimodal_score=multimodal_score,
        reward_score=float(reward_breakdown.score),
        neighbors=neighbors,
        matched_triplets=matched_triplets,
        vlm_analyses=vlm_analyses,
    )
    calibration_score = float(calibration_payload['calibration_score']) if bool(getattr(settings, 'task3_calibration_enabled', True)) else 1.0
    calibrated_final_score = float(final_score) * (0.7 + 0.3 * calibration_score)
    return {
        'final_score': final_score,
        'calibrated_final_score': calibrated_final_score,
        'score_components': {
            'candidate_score': float(candidate.score),
            'prediction_score': prediction_score,
            'multimodal_score': multimodal_score,
            'retrieval_score': retrieval_score,
            'reward_score': float(reward_breakdown.score),
            'reward_reasons': reward_breakdown.reasons,
        },
        **calibration_payload,
    }


def _prediction_records(
    *,
    events: Sequence[Any],
    backend: str,
    top_k: int,
) -> tuple[List[LinkPredictionRecord], Dict[str, Any]]:
    requested = str(backend or "auto").strip().lower()
    used_backend = "tgn"
    error: Optional[str] = None
    predictions: List[LinkPredictionRecord] = []

    if requested in {"auto", "pygt", "pygt_temporal", "pyg_temporal"}:
        try:
            cfg = PyGTemporalLinkPredConfig(
                hidden_dim=int(getattr(settings, "hyp_tgnn_memory_dim", 64) or 64),
                relation_dim=int(getattr(settings, "hyp_tgnn_relation_dim", 24) or 24),
                time_dim=int(getattr(settings, "hyp_tgnn_time_dim", 16) or 16),
                epochs=int(getattr(settings, "hyp_tgnn_epochs", 25) or 25),
                recent_window_years=int(getattr(settings, "hyp_tgnn_recent_window_years", 3) or 3),
                min_candidate_score=float(getattr(settings, "hyp_tgnn_min_candidate_score", 0.05) or 0.05),
            )
            predictions = pygt_temporal_link_prediction(events, top_k=top_k, config=cfg)
            used_backend = "pygt_temporal"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            predictions = []

    if not predictions:
        tgn_cfg = TGNLinkPredConfig(
            backend="heuristic" if requested == "heuristic" else str(getattr(settings, "hyp_tgnn_backend", "auto") or "auto"),
            recent_window_years=int(getattr(settings, "hyp_tgnn_recent_window_years", 3) or 3),
            recency_half_life_years=float(getattr(settings, "hyp_tgnn_half_life_years", 2.0) or 2.0),
            min_candidate_score=float(getattr(settings, "hyp_tgnn_min_candidate_score", 0.05) or 0.05),
            memory_dim=int(getattr(settings, "hyp_tgnn_memory_dim", 64) or 64),
            time_dim=int(getattr(settings, "hyp_tgnn_time_dim", 16) or 16),
        )
        predictions = tgn_link_prediction(events, top_k=top_k, config=tgn_cfg)
        used_backend = "tgn" if requested != "heuristic" else "heuristic"

    rows = [
        LinkPredictionRecord(
            source=row.source,
            predicate=row.predicate,
            target=row.target,
            score=float(row.score),
            backend=used_backend or row.backend,
            polarity=row.polarity,
            direction=row.direction,
            ts_pred=row.ts_pred,
            relation_family=row.relation_family,
            evidence_path=list(row.evidence_path),
            aux=dict(row.aux),
        )
        for row in predictions
    ]
    meta = {"requested_backend": requested, "used_backend": used_backend, "fallback_error": error, "n_predictions": len(rows)}
    return rows, meta


def prepare_task3_hypothesis_bundle(
    *,
    trajectory: Optional[Path] = None,
    query: str = "",
    identifiers: Optional[Sequence[str]] = None,
    identifiers_file: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    out_dir: Path = Path("runs/task3_hypotheses"),
    domain_id: str = "science",
    search_limit: int = 25,
    top_papers: int = 12,
    top_hypotheses: int = 8,
    candidate_top_k: int = 16,
    include_multimodal: bool = True,
    run_vlm: bool = True,
    edge_mode: str = "auto",
    link_prediction_backend: str = "auto",
    link_prediction_top_k: int = 24,
    annoy_metric: str = "angular",
    annoy_n_trees: int = 32,
    annoy_top_k: int = 6,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    g4f_model: str | None = None,
    local_model: str | None = None,
    vlm_backend: str | None = None,
    vlm_model_id: str | None = None,
) -> Task3BundleResult:
    domain = load_domain_config(domain_id)
    out_dir = Path(out_dir)

    input_identifiers: List[str] = list(identifiers or [])
    if identifiers_file is not None and Path(identifiers_file).exists():
        input_identifiers.extend(_parse_identifier_blob(Path(identifiers_file).read_text(encoding="utf-8")))

    trajectory_doc: Dict[str, Any] = {}
    resolved_papers: List[PaperMetadata] = []
    effective_query = str(query or "").strip()
    run_name = "task3_run"

    if trajectory is not None:
        trajectory = Path(trajectory)
        trajectory_doc = _load_yaml(trajectory)
        effective_query = effective_query or str(trajectory_doc.get("topic") or trajectory.stem)
        run_name = str(trajectory_doc.get("submission_id") or trajectory.stem)
        resolved_papers.extend(resolve_papers_from_trajectory(trajectory_doc, enable_remote_lookup=True))

    if input_identifiers:
        if not effective_query:
            effective_query = " | ".join(input_identifiers[:3])
        if run_name == "task3_run":
            run_name = _slugify("-".join(input_identifiers[:2]))
        resolved_papers.extend(_resolve_papers_from_identifiers(input_identifiers, search_limit=search_limit))

    if not resolved_papers and effective_query and processed_dir is None:
        searched = search_papers(effective_query, limit=search_limit, sources=DEFAULT_SOURCES)
        resolved_papers.extend(_rank_papers(searched, effective_query)[:top_papers])
        if run_name == "task3_run":
            run_name = _slugify(effective_query)
    elif processed_dir is not None and run_name == "task3_run":
        run_name = _slugify(effective_query or Path(processed_dir).name)

    resolved_papers = _dedupe_papers(resolved_papers)
    if top_papers > 0:
        resolved_papers = resolved_papers[: int(top_papers)]

    bundle_dir = out_dir / run_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    overrides_path = bundle_dir / "expert_overrides.jsonl"
    try:
        compile_overrides(Path("data/experts/graph_reviews"), overrides_path)
    except Exception:
        overrides_path = Path("data/derived/expert_overrides.jsonl")

    runtime_info = {
        "requested_query": query,
        "effective_query": effective_query,
        "domain_id": domain.domain_id,
        "domain_title": domain.title,
        "generated_at": _utc_now(),
    }
    _write_json(bundle_dir / "query.json", runtime_info)
    if trajectory_doc:
        _write_json(bundle_dir / "trajectory_snapshot.json", trajectory_doc)

    with temporary_llm_selection(llm_provider=llm_provider, llm_model=llm_model, g4f_model=g4f_model, local_model=local_model):
        with temporary_vlm_selection(vlm_backend=vlm_backend, vlm_model_id=vlm_model_id):
            _write_json(bundle_dir / "papers_selected.json", [paper.model_dump(mode="json") for paper in resolved_papers])

            if processed_dir is not None:
                prepared_processed_dir = _copy_processed_tree(Path(processed_dir), bundle_dir / "processed_papers")
                acquire_results: List[AcquireResult] = []
            else:
                prepared_processed_dir, acquire_results = _acquire_and_ingest_papers(
                    resolved_papers,
                    bundle_dir=bundle_dir,
                    include_multimodal=include_multimodal,
                    run_vlm=run_vlm,
                )
            _write_json(
                bundle_dir / "acquire_results.json",
                [asdict(item) | {"pdf_path": str(item.pdf_path) if item.pdf_path else None, "meta_path": str(item.meta_path)} for item in acquire_results],
            )

            paper_records_processed = load_papers_from_processed(prepared_processed_dir)
            processed_lookup = {paper.paper_id: paper for paper in paper_records_processed}
            paper_records: List[PaperRecord] = []
            for paper in resolved_papers:
                paper_records.append(processed_lookup.get(paper.id) or _paperrecord_from_metadata(paper))
            if not paper_records:
                paper_records = paper_records_processed
            _write_json(bundle_dir / "paper_records.json", [asdict(item) for item in paper_records])

            chunk_records = _load_chunk_registry(prepared_processed_dir)
            _write_jsonl(bundle_dir / "chunk_registry.jsonl", [record.model_dump(mode="json") for record in chunk_records])

            vectors, annoy_payloads = _build_chunk_embeddings(chunk_records)
            annoy_bundle = build_annoy_index(
                vectors,
                [record.chunk_id for record in chunk_records],
                bundle_dir / "annoy",
                metric=annoy_metric,
                n_trees=annoy_n_trees,
                item_payloads=annoy_payloads,
            )

            kg = build_temporal_kg(
                paper_records,
                domain=domain,
                query=effective_query,
                edge_mode=edge_mode,  # type: ignore[arg-type]
                expert_overrides_path=overrides_path,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            kg_path = bundle_dir / "automatic_graph" / "temporal_kg.json"
            kg.dump_json(kg_path)

            events = build_event_stream(kg, papers=paper_records)
            _write_jsonl(bundle_dir / "automatic_graph" / "events.jsonl", [event.model_dump(mode="json") for event in events])

            paper_years = {paper.paper_id: paper.year for paper in paper_records}
            mm_triplets = extract_multimodal_triplets(
                chunk_records,
                paper_years=paper_years,
                domain=domain.title,
                llm_provider=llm_provider,
                llm_model=llm_model,
                run_vlm=run_vlm,
                vlm_backend=vlm_backend,
                vlm_model_id=vlm_model_id,
            )
            mm_triplets_path = bundle_dir / "automatic_graph" / "multimodal_triplets.jsonl"
            dump_multimodal_triplets(mm_triplets_path, mm_triplets)

            link_predictions, link_meta = _prediction_records(events=events, backend=link_prediction_backend, top_k=link_prediction_top_k)
            _write_json(
                bundle_dir / "automatic_graph" / "link_predictions.json",
                {"meta": link_meta, "predictions": [row.to_json_dict() for row in link_predictions]},
            )

            candidates = generate_candidates(
                kg,
                papers=paper_records,
                query=effective_query,
                domain=domain.title,
                top_k=max(candidate_top_k, top_hypotheses),
                prediction_records=link_predictions,
            )
            candidate_rows = []
            for cand in candidates:
                candidate_rows.append(
                    {
                        "kind": cand.kind,
                        "source": cand.source,
                        "target": cand.target,
                        "predicate": cand.predicate,
                        "score": cand.score,
                        "time_scope": cand.time_scope,
                        "graph_signals": cand.graph_signals,
                        "evidence": [cite.model_dump(mode="json") for cite in cand.evidence],
                    }
                )
            _write_json(bundle_dir / "hypotheses_candidates.json", candidate_rows)

            reward = RuleBasedReward(overrides_path=overrides_path)
            ranked_records: List[Dict[str, Any]] = []
            vlm_analysis_rows: List[Dict[str, Any]] = []
            for cand in candidates[: max(candidate_top_k, top_hypotheses)]:
                temporal_context = _candidate_temporal_context(kg, cand)
                prediction_match = _match_link_prediction(cand, link_predictions)
                neighbors = _candidate_neighbor_evidence(cand, annoy_bundle=annoy_bundle, top_k=annoy_top_k) if annoy_bundle.size > 0 else []
                matched_triplets = _match_multimodal_triplets(cand, mm_triplets, limit=6)
                vlm_analyses = _candidate_vlm_analyses(
                    cand,
                    neighbors,
                    top_k=2,
                    run_vlm=run_vlm,
                    vlm_backend=vlm_backend,
                    vlm_model_id=vlm_model_id,
                )
                if vlm_analyses:
                    vlm_analysis_rows.extend(
                        {
                            "candidate": {"source": cand.source, "predicate": cand.predicate, "target": cand.target},
                            **row,
                        }
                        for row in vlm_analyses
                    )
                supporting_evidence = _citations_from_candidate_context(
                    cand,
                    neighbors=neighbors,
                    matched_triplets=matched_triplets,
                )

                try:
                    draft = _llm_hypothesis_from_context(
                        cand,
                        query=effective_query,
                        domain=domain.title,
                        temporal_context=temporal_context,
                        link_prediction=prediction_match,
                        neighbors=neighbors,
                        matched_triplets=matched_triplets,
                        vlm_analyses=vlm_analyses,
                        supporting_evidence=supporting_evidence,
                    )
                except Exception as exc:
                    console.print(f"[yellow]Task3 LLM hypothesis fallback for {cand.source}|{cand.predicate}|{cand.target}: {type(exc).__name__}: {exc}[/yellow]")
                    draft = _template_hypothesis_from_context(
                        cand,
                        temporal_context=temporal_context,
                        link_prediction=prediction_match,
                        supporting_evidence=supporting_evidence,
                        matched_triplets=matched_triplets,
                    )

                score_payload = _score_hypothesis_record(
                    candidate=cand,
                    draft=draft,
                    reward=reward,
                    temporal_context=temporal_context,
                    link_prediction=prediction_match,
                    neighbors=neighbors,
                    matched_triplets=matched_triplets,
                    vlm_analyses=vlm_analyses,
                )
                ranked_records.append(
                    {
                        "candidate": {
                            "kind": cand.kind,
                            "source": cand.source,
                            "target": cand.target,
                            "predicate": cand.predicate,
                            "score": cand.score,
                            "time_scope": cand.time_scope,
                            "graph_signals": cand.graph_signals,
                        },
                        "temporal_context": temporal_context,
                        "prediction_support": prediction_match.to_json_dict() if prediction_match is not None else None,
                        "annoy_neighbors": neighbors,
                        "multimodal_support": matched_triplets,
                        "vlm_candidate_analysis": vlm_analyses,
                        "hypothesis": draft.model_dump(mode="json"),
                        **score_payload,
                    }
                )

            if vlm_analysis_rows:
                _write_jsonl(bundle_dir / "automatic_graph" / "vlm_candidate_analysis.jsonl", vlm_analysis_rows)

            ranked_records.sort(key=lambda row: float(row.get("calibrated_final_score") or row.get("final_score") or 0.0), reverse=True)
            if bool(getattr(settings, "task3_mmr_enabled", True)):
                ranked_records = _apply_mmr_rerank(
                    ranked_records,
                    top_k=max(top_hypotheses, len(ranked_records)),
                    lambda_mult=float(getattr(settings, "task3_mmr_lambda", 0.72) or 0.72),
                )
            for rank, row in enumerate(ranked_records[:top_hypotheses], start=1):
                row["rank"] = rank
            hypotheses_path = bundle_dir / "hypotheses_ranked.json"
            _write_json(hypotheses_path, ranked_records[:top_hypotheses])

            md_lines = [f"# Task 3 hypotheses for: {effective_query}", "", f"Domain: {domain.title} ({domain.domain_id})", ""]
            for row in ranked_records[:top_hypotheses]:
                hyp = row.get("hypothesis") or {}
                md_lines.extend(
                    [
                        f"## H-{int(row.get('rank') or 0):03d}: {hyp.get('title', '')}",
                        "",
                        f"**Final score:** {float(row.get('final_score') or 0.0):.3f}",
                        f"**Calibrated score:** {float(row.get('calibrated_final_score') or 0.0):.3f}",
                        f"**Calibration:** {float(row.get('calibration_score') or 0.0):.3f}",
                        "",
                        f"**Premise:** {hyp.get('premise', '')}",
                        "",
                        f"**Mechanism:** {hyp.get('mechanism', '')}",
                        "",
                        f"**Time scope:** {hyp.get('time_scope', '')}",
                        "",
                        f"**Proposed experiment:** {hyp.get('proposed_experiment', '')}",
                        "",
                    ]
                )
                evidence = hyp.get("supporting_evidence") if isinstance(hyp, dict) else []
                if isinstance(evidence, list) and evidence:
                    md_lines.append("**Supporting evidence:**")
                    for cite in evidence:
                        if isinstance(cite, dict):
                            md_lines.append(f"- {cite.get('source_id')}: {cite.get('text_snippet')}")
                    md_lines.append("")
            (bundle_dir / "hypotheses_ranked.md").write_text("\n".join(md_lines), encoding="utf-8")

            manifest = {
                "bundle_dir": str(bundle_dir),
                "generated_at": _utc_now(),
                "domain_id": domain.domain_id,
                "domain_title": domain.title,
                "query": effective_query,
                "trajectory": str(trajectory) if trajectory else None,
                "identifiers_file": str(identifiers_file) if identifiers_file else None,
                "n_selected_papers": len(resolved_papers),
                "n_paper_records": len(paper_records),
                "n_chunks": len(chunk_records),
                "n_mm_triplets": sum(len(record.triplets) for record in mm_triplets),
                "n_link_predictions": len(link_predictions),
                "n_candidates": len(candidates),
                "n_hypotheses": min(top_hypotheses, len(ranked_records)),
                "inputs": {
                    "include_multimodal": include_multimodal,
                    "run_vlm": run_vlm,
                    "edge_mode": edge_mode,
                    "link_prediction_backend": link_prediction_backend,
                    "annoy_metric": annoy_metric,
                    "annoy_n_trees": annoy_n_trees,
                },
                "artifacts": {
                    "papers_selected": str(bundle_dir / "papers_selected.json"),
                    "chunk_registry": str(bundle_dir / "chunk_registry.jsonl"),
                    "annoy_manifest": str(annoy_bundle.manifest_path),
                    "temporal_kg": str(kg_path),
                    "events": str(bundle_dir / "automatic_graph" / "events.jsonl"),
                    "multimodal_triplets": str(mm_triplets_path),
                    "link_predictions": str(bundle_dir / "automatic_graph" / "link_predictions.json"),
                    "hypotheses_candidates": str(bundle_dir / "hypotheses_candidates.json"),
                    "hypotheses_ranked": str(hypotheses_path),
                    "hypotheses_markdown": str(bundle_dir / "hypotheses_ranked.md"),
                },
                "runtime": {
                    "llm_provider": settings.llm_provider,
                    "llm_model": settings.llm_model,
                    "vlm_backend": settings.vlm_backend,
                    "vlm_model_id": settings.vlm_model_id,
                    "annoy_backend": annoy_bundle.backend,
                    "link_prediction_used_backend": link_meta.get("used_backend"),
                    "link_prediction_fallback_error": link_meta.get("fallback_error"),
                },
            }
            manifest_path = bundle_dir / "task3_manifest.json"
            _write_json(manifest_path, manifest)
            return Task3BundleResult(bundle_dir=bundle_dir, manifest_path=manifest_path, hypotheses_path=hypotheses_path)
