from __future__ import annotations

"""Compatibility VLM-OCR helpers for Task 2 temporal KG repair.

Older Task 2 notebooks import ``scireason.ingest.vlm_ocr`` through
``temporal_kg_builder``.  Some repository snapshots accidentally missed this
module while keeping the import.  This file provides the narrow API expected by
``temporal_kg_builder`` and reuses the existing VLM/image stack from
``scireason.mm.vlm``.
"""

from dataclasses import dataclass, field
import gc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from rich.console import Console

from ..config import settings
from ..mm.vlm import VLMResult, describe_image, temporary_vlm_selection

console = Console()


@dataclass
class VLMOCRPageChunk:
    paper_id: str
    page: int
    text: str
    image_path: str = ""
    vlm_caption: str = ""
    tables_md: Optional[str] = None
    equations_md: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _as_positive_dpi(value: Optional[int]) -> int:
    try:
        dpi = int(value or getattr(settings, "pdf_render_dpi", 150) or 150)
    except Exception:
        dpi = 150
    return max(72, dpi)


def _page_indices(total_pages: int, page_index: Optional[int] = None, pages: Optional[Sequence[int]] = None) -> List[int]:
    if pages is not None:
        indices = [int(p) for p in pages]
    elif page_index is not None:
        indices = [int(page_index)]
    else:
        indices = list(range(total_pages))
    return [i for i in indices if 0 <= i < total_pages]


def _emit(progress_callback: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception:
        pass


def extract_pdf_page_chunks_vlm_ocr(
    pdf_path: str | Path,
    paper_id: str,
    *,
    page_index: Optional[int] = None,
    pages: Optional[Sequence[int]] = None,
    out_dir: str | Path | None = None,
    prompt_context: str = "",
    backend: Optional[str] = None,
    model_id: Optional[str] = None,
    dpi: Optional[int] = None,
    run_vlm: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **_: Any,
) -> List[VLMOCRPageChunk]:
    """Extract page-level text plus optional VLM OCR/captions from a PDF.

    The temporal KG builder only requires returned objects to expose a ``text``
    attribute.  We keep richer metadata as a best-effort compatibility layer.
    If VLM is unavailable, the function still returns normal PyMuPDF text so the
    Task 2 pipeline can continue.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        import fitz  # PyMuPDF  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("Для VLM-OCR fallback нужен PyMuPDF: pip install pymupdf") from exc

    if out_dir is None:
        out_dir = pdf_path.parent / "mm" / "vlm_ocr_pages"
    out_dir = Path(out_dir)
    img_dir = out_dir / str(paper_id) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    effective_backend = str(backend or getattr(settings, "vlm_backend", "none") or "none")
    effective_model_id = str(model_id or getattr(settings, "vlm_model_id", "") or "")
    should_run_vlm = bool(run_vlm) and effective_backend.lower() != "none"
    prompt = prompt_context or (
        "Извлеки текст, таблицы, формулы и ключевые факты со страницы научной статьи. "
        "Верни результат кратко, но с сохранением важных терминов и чисел."
    )

    records: List[VLMOCRPageChunk] = []
    dpi_value = _as_positive_dpi(dpi)
    zoom = dpi_value / 72.0

    with fitz.open(str(pdf_path)) as doc:
        indices = _page_indices(len(doc), page_index=page_index, pages=pages)
        total = len(indices)
        for ordinal, i in enumerate(indices, start=1):
            page = doc.load_page(i)
            text = (page.get_text("text") or "").strip()
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img_path = img_dir / f"page_{i:03d}.png"
            pix.save(str(img_path))

            caption = ""
            tables_md: Optional[str] = None
            equations_md: Optional[str] = None
            if should_run_vlm:
                try:
                    with temporary_vlm_selection(vlm_backend=effective_backend, vlm_model_id=effective_model_id):
                        res: VLMResult = describe_image(img_path, prompt=prompt, backend=effective_backend, model_id=effective_model_id)
                    caption = str(res.caption or "").strip()
                    tables_md = res.extracted_tables_md
                    equations_md = res.extracted_equations_md
                except Exception as exc:  # pragma: no cover - defensive fallback
                    console.print(
                        f"[yellow]VLM-OCR fallback unavailable for {pdf_path.name} page {i + 1}: "
                        f"{type(exc).__name__}: {exc}. Продолжаю с PyMuPDF text.[/yellow]"
                    )

            combined_parts = [text]
            if caption:
                combined_parts.append("VLM caption/OCR:\n" + caption)
            if tables_md:
                combined_parts.append("Tables:\n" + str(tables_md).strip())
            if equations_md:
                combined_parts.append("Equations:\n" + str(equations_md).strip())
            combined_text = "\n\n".join(p for p in combined_parts if p and str(p).strip()).strip()

            records.append(
                VLMOCRPageChunk(
                    paper_id=str(paper_id),
                    page=i,
                    text=combined_text,
                    image_path=str(img_path.as_posix()),
                    vlm_caption=caption,
                    tables_md=tables_md,
                    equations_md=equations_md,
                    metadata={
                        "source_backend": "pymupdf_vlm_ocr" if caption else "pymupdf_text",
                        "pdf_path": str(pdf_path),
                        "dpi": dpi_value,
                    },
                )
            )
            _emit(progress_callback, {
                "event": "vlm_ocr_page",
                "paper_id": str(paper_id),
                "pdf_path": str(pdf_path),
                "current": ordinal,
                "total": total,
                "page": i,
                "message": f"VLM-OCR page {ordinal}/{total}",
            })
            pix = None
            page = None
            if ordinal % 5 == 0:
                gc.collect()

    return records
