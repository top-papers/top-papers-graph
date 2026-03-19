from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from ..config import settings
from ..contracts import ChunkRecord
from .store import save_paper

console = Console()


class PaddleOCRUnavailableError(RuntimeError):
    pass


def paddleocr_available() -> bool:
    try:
        import paddleocr  # noqa: F401

        return True
    except Exception:
        return False


def _load_pipeline(lang: Optional[str] = None):
    try:
        from paddleocr import PPStructureV3  # type: ignore

        pipeline = PPStructureV3(lang=lang) if lang else PPStructureV3()
        return pipeline, "PPStructureV3"
    except Exception:
        pass

    try:
        from paddleocr import PPStructure  # type: ignore

        pipeline = PPStructure(lang=lang, show_log=False) if lang else PPStructure(show_log=False)
        return pipeline, "PPStructure"
    except Exception as e:  # pragma: no cover - import guard only
        raise PaddleOCRUnavailableError(
            "PaddleOCR is not installed. Install optional dependencies: pip install -e '.[ocr]'"
        ) from e


def _safe_bbox(obj: Any) -> Optional[List[float]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        obj = obj.get("bbox")
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        flat: List[float] = []
        for x in obj:
            if isinstance(x, (list, tuple)):
                flat.extend([float(v) for v in x[:2]])
            else:
                flat.append(float(x))
        return flat[:8] if flat else None
    return None


def _text_from_markdown(md: Any) -> str:
    if md is None:
        return ""
    if isinstance(md, str):
        return md.strip()
    if isinstance(md, dict):
        for key in ("markdown", "markdown_texts", "text", "content"):
            value = md.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _records_from_predict_result(result: Any, *, paper_id: str, page_index: int, source_backend: str) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []

    # Primary path for PP-StructureV3: markdown per page.
    markdown = getattr(result, "markdown", None)
    page_text = _text_from_markdown(markdown)
    if page_text:
        records.append(
            ChunkRecord(
                chunk_id=f"{paper_id}:page:{page_index}:markdown",
                paper_id=paper_id,
                page=page_index,
                text=page_text,
                modality="page",
                source_backend=source_backend,
                metadata={"markdown": markdown if isinstance(markdown, dict) else None},
            )
        )

    # Best-effort access to structured layout blocks.
    items = None
    for attr in ("json", "data", "result", "res"):
        value = getattr(result, attr, None)
        if isinstance(value, list):
            items = value
            break
        if isinstance(value, dict):
            items = value.get("res") or value.get("layout") or value.get("items")
            if isinstance(items, list):
                break
    if items is None and isinstance(result, list):
        items = result

    if not isinstance(items, list):
        return records

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or item.get("label") or "text").lower()
        text = ""
        res = item.get("res")
        if isinstance(res, dict):
            text = str(res.get("html") or res.get("text") or res.get("markdown") or "")
        elif isinstance(res, list):
            bits: List[str] = []
            for row in res:
                if isinstance(row, dict):
                    bits.append(str(row.get("text") or row.get("html") or ""))
                else:
                    bits.append(str(row))
            text = "\n".join(b for b in bits if b)
        elif res is not None:
            text = str(res)
        if not text:
            text = str(item.get("text") or item.get("html") or "")
        if not text.strip():
            continue

        modality = "text"
        if "table" in item_type:
            modality = "table"
        elif "formula" in item_type:
            modality = "formula"
        elif item_type in {"image", "figure", "chart"}:
            modality = "figure"

        table_html = None
        if modality == "table":
            table_html = text if "<table" in text.lower() else None

        records.append(
            ChunkRecord(
                chunk_id=f"{paper_id}:page:{page_index}:item:{idx}",
                paper_id=paper_id,
                page=page_index,
                text=text.strip(),
                bbox=_safe_bbox(item),
                modality=modality,
                source_backend=source_backend,
                reading_order=idx,
                table_html=table_html,
                metadata={"item_type": item_type},
            )
        )
    return records


def _fallback_pdf_records(pdf_path: Path, paper_id: str) -> List[ChunkRecord]:
    try:
        import fitz  # PyMuPDF  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for OCR fallback.") from e

    doc = fitz.open(str(pdf_path))
    records: List[ChunkRecord] = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = (page.get_text("text", sort=True) or "").strip()
        if not text:
            continue
        records.append(
            ChunkRecord(
                chunk_id=f"{paper_id}:page:{page_index}:fallback",
                paper_id=paper_id,
                page=page_index,
                text=text,
                modality="page",
                source_backend="pymupdf_fallback",
            )
        )
    return records


def extract_pdf_chunks_paddleocr(pdf_path: Path, *, paper_id: str, lang: Optional[str] = None) -> List[ChunkRecord]:
    pipeline, backend_name = _load_pipeline(lang=lang)
    try:
        output = pipeline.predict(input=str(pdf_path))
    except TypeError:
        output = pipeline.predict(str(pdf_path))

    records: List[ChunkRecord] = []
    for page_index, result in enumerate(output):
        page_records = _records_from_predict_result(
            result,
            paper_id=paper_id,
            page_index=page_index,
            source_backend=backend_name,
        )
        records.extend(page_records)

    if not records:
        console.print("[yellow]PaddleOCR returned no structured chunks; using PyMuPDF fallback.[/yellow]")
        return _fallback_pdf_records(pdf_path=pdf_path, paper_id=paper_id)
    return records


def ingest_pdf_paddleocr(pdf_path: Path, meta: Dict[str, Any], out_dir: Path, *, lang: Optional[str] = None) -> Path:
    pid = str(meta.get("id") or pdf_path.stem)
    records = extract_pdf_chunks_paddleocr(pdf_path=pdf_path, paper_id=pid, lang=lang or getattr(settings, "paddleocr_lang", None))
    console.print(f"[green]PaddleOCR chunks:[/green] {len(records)}")
    return save_paper(out_dir, meta=meta, chunks=records)
