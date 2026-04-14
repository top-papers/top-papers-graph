from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from rich.console import Console

from ..contracts import ChunkRecord
from ..temporal.schemas import TemporalTriplet
from ..temporal.temporal_triplet_extractor import extract_temporal_triplets
from .vlm import VLMResult, describe_image


console = Console()


@dataclass(frozen=True)
class MultimodalTripletArtifact:
    paper_id: str
    chunk_id: str
    modality: str
    page: Optional[int]
    image_path: str
    extraction_backend: str
    vlm_caption: str
    tables_md: str
    equations_md: str
    analysis_text: str
    triplets: List[TemporalTriplet]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "chunk_id": self.chunk_id,
            "modality": self.modality,
            "page": self.page,
            "image_path": self.image_path,
            "extraction_backend": self.extraction_backend,
            "vlm_caption": self.vlm_caption,
            "tables_md": self.tables_md,
            "equations_md": self.equations_md,
            "analysis_text": self.analysis_text,
            "triplets": [t.model_dump(mode="json") for t in self.triplets],
        }


def _metadata_text(record: ChunkRecord, key: str) -> str:
    value = (record.metadata or {}).get(key)
    return str(value or "").strip()


def _compose_analysis_text(
    record: ChunkRecord,
    *,
    vlm_caption: str,
    tables_md: str,
    equations_md: str,
) -> str:
    parts: List[str] = []
    base_text = str(record.text or "").strip()
    if base_text:
        parts.append(base_text)

    table_md = str(record.table_md or "").strip()
    if table_md:
        parts.append(f"Table evidence:\n{table_md}")
    if tables_md and tables_md != table_md:
        parts.append(f"VLM table extraction:\n{tables_md}")

    if vlm_caption:
        parts.append(f"Visual evidence: {vlm_caption}")
    if equations_md:
        parts.append(f"Equations:\n{equations_md}")

    payload = "\n\n".join(part for part in parts if part).strip()
    if payload:
        return payload
    if record.image_path:
        return f"Image-backed chunk from paper {record.paper_id} page {record.page or 'unknown'}"
    return ""


def _has_cached_vlm_artifacts(record: ChunkRecord) -> bool:
    return any(
        bool(_metadata_text(record, key))
        for key in ("vlm_caption", "tables_md", "equations_md")
    )


def _should_request_vlm(record: ChunkRecord, *, run_vlm: bool) -> bool:
    if not run_vlm:
        return False
    if not record.image_path:
        return False
    if _has_cached_vlm_artifacts(record):
        return False
    modality = str(record.modality or "unknown").strip().lower()
    if modality in {"table", "formula"} and bool(str(record.text or "").strip()):
        return False
    return modality in {"page", "figure", "table", "formula", "unknown"} or not bool(str(record.text or "").strip())


def _vlm_prompt(record: ChunkRecord) -> str:
    page = f"page {record.page}" if record.page is not None else "unknown page"
    modality = str(record.modality or "unknown")
    return (
        "Ты подготавливаешь мультимодальное доказательство для построения темпорального графа знаний. "
        "Опиши только проверяемые наблюдения из изображения/страницы: объекты, процессы, таблицы, графики, "
        "направление эффекта, количественные значения, даты и временные интервалы, если они явно видны. "
        "Не выдумывай скрытые выводы. "
        f"Контекст: {modality} chunk, {page}, paper_id={record.paper_id}."
    )


def extract_multimodal_triplets(
    chunk_records: Sequence[ChunkRecord],
    *,
    paper_years: Mapping[str, Optional[int]],
    domain: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    run_vlm: bool = True,
    vlm_backend: Optional[str] = None,
    vlm_model_id: Optional[str] = None,
    max_chunks: Optional[int] = None,
) -> List[MultimodalTripletArtifact]:
    """Extract temporal triplets from text/image/table chunks.

    Strategy:
    - keep textual chunk content when it already exists;
    - augment image/table/page chunks with VLM captions and table/equation extraction;
    - convert the multimodal evidence into a shared textual representation;
    - run the existing temporal triplet extractor to keep the downstream KG contract stable.
    """

    out: List[MultimodalTripletArtifact] = []
    limit = int(max_chunks) if max_chunks not in (None, 0) else None
    vlm_cache: Dict[str, VLMResult] = {}

    for idx, record in enumerate(chunk_records):
        if limit is not None and idx >= limit:
            break

        vlm_caption = _metadata_text(record, "vlm_caption")
        tables_md = _metadata_text(record, "tables_md")
        equations_md = _metadata_text(record, "equations_md")
        extraction_backend = "text_only"

        if _should_request_vlm(record, run_vlm=run_vlm):
            image_key = str(record.image_path or "").strip()
            try:
                cached = vlm_cache.get(image_key) if image_key else None
                result = cached
                if result is None:
                    result = describe_image(
                        image_path=Path(str(record.image_path)),
                        prompt=_vlm_prompt(record),
                        backend=vlm_backend,  # type: ignore[arg-type]
                        model_id=vlm_model_id,
                    )
                    if image_key:
                        vlm_cache[image_key] = result
                if result.caption:
                    vlm_caption = str(result.caption).strip()
                if result.extracted_tables_md:
                    tables_md = str(result.extracted_tables_md).strip()
                if result.extracted_equations_md:
                    equations_md = str(result.extracted_equations_md).strip()
                extraction_backend = "vlm_augmented"
            except Exception as exc:
                console.print(
                    f"[yellow]Multimodal triplets fallback for {record.chunk_id}: {type(exc).__name__}: {exc}. "
                    "Продолжаю без дополнительного VLM-анализа.[/yellow]"
                )
                extraction_backend = "text_only"

        analysis_text = _compose_analysis_text(record, vlm_caption=vlm_caption, tables_md=tables_md, equations_md=equations_md)
        if not analysis_text:
            continue

        try:
            triplets = extract_temporal_triplets(
                domain=domain,
                chunk_text=analysis_text,
                paper_year=paper_years.get(record.paper_id),
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        except Exception:
            triplets = []

        out.append(
            MultimodalTripletArtifact(
                paper_id=str(record.paper_id or ""),
                chunk_id=str(record.chunk_id),
                modality=str(record.modality or "unknown"),
                page=record.page,
                image_path=str(record.image_path or ""),
                extraction_backend=extraction_backend,
                vlm_caption=vlm_caption,
                tables_md=tables_md,
                equations_md=equations_md,
                analysis_text=analysis_text,
                triplets=list(triplets),
            )
        )

    return out


def dump_multimodal_triplets(path: Path, records: Iterable[MultimodalTripletArtifact]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
    return path
