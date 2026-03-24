from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from rich.console import Console

from ..config import settings
from ..contracts import ChunkRecord
from .chunking import simple_chunks
from .grobid_client import GrobidUnavailableError, grobid_fulltext, grobid_status
from .paddleocr_pipeline import PaddleOCRUnavailableError, ingest_pdf_paddleocr
from .store import save_paper
from .tei_to_text import tei_to_plaintext

console = Console()


def _plain_text_chunks(text: str, *, paper_id: str, backend: str) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=f"{paper_id}:{idx}",
            paper_id=paper_id,
            text=chunk,
            source_backend=backend,
            modality="text",
            reading_order=idx,
        )
        for idx, chunk in enumerate(simple_chunks(text))
    ]


def ingest_pdf(pdf_path: Path, meta: Dict[str, Any], out_dir: Path) -> Path:
    ok, _ = grobid_status()
    if ok:
        console.print(f"[cyan]GROBID парсит:[/cyan] {pdf_path}")
    tei = grobid_fulltext(pdf_path)
    text = tei_to_plaintext(tei)
    pid = str(meta.get("id") or pdf_path.stem)
    chunks = _plain_text_chunks(text, paper_id=pid, backend="grobid")
    console.print(f"[green]Извлечено чанков:[/green] {len(chunks)}")
    return save_paper(out_dir, meta=meta, chunks=chunks)


def _extract_text_pymupdf(pdf_path: Path) -> str:
    """Fallback extractor using PyMuPDF (no external GROBID service)."""

    try:
        import fitz  # PyMuPDF  # type: ignore

        parts: list[str] = []
        with fitz.open(str(pdf_path)) as doc:
            for i in range(len(doc)):
                page = doc.load_page(i)
                t = (page.get_text("text", sort=True) or "").strip()
                if t:
                    parts.append(t)
        return "\n\n".join(parts)
    except Exception:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Для fallback PDF-парсинга нужен PyMuPDF (pip install pymupdf) или pypdf (pip install pypdf)."
            ) from e

        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            if t:
                parts.append(t)
        return "\n\n".join(parts)


def _ingest_pdf_pymupdf(pdf_path: Path, meta: Dict[str, Any], out_dir: Path) -> Path:
    text = _extract_text_pymupdf(pdf_path)
    pid = str(meta.get("id") or pdf_path.stem)
    chunks = _plain_text_chunks(text, paper_id=pid, backend="pymupdf")
    console.print(f"[green]Extracted chunks (fallback):[/green] {len(chunks)}")
    return save_paper(out_dir, meta=meta, chunks=chunks)


def ingest_pdf_auto(pdf_path: Path, meta: Dict[str, Any], out_dir: Path, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Path:
    """Ingest PDF using the best available backend.

    Priority in `OCR_BACKEND=auto`:
    1) PaddleOCR / PP-StructureV3 (preferred structured multimodal parser for text/tables/figures)
    2) PyMuPDF / pypdf fallback for text-only recovery

    GROBID is kept only as an explicit opt-in backend (`OCR_BACKEND=grobid`) for legacy setups.
    """

    backend = str(getattr(settings, "ocr_backend", "auto") or "auto").strip().lower()

    if backend in {"paddleocr", "paddle", "ppstructure", "pp_structure"}:
        return ingest_pdf_paddleocr(pdf_path=pdf_path, meta=meta, out_dir=out_dir, progress_callback=progress_callback)
    if backend in {"pymupdf", "fitz", "local"}:
        return _ingest_pdf_pymupdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    if backend == "grobid":
        return ingest_pdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)

    # Auto mode: prefer structured multimodal OCR/layout first, then local text fallback.
    try:
        return ingest_pdf_paddleocr(pdf_path=pdf_path, meta=meta, out_dir=out_dir, progress_callback=progress_callback)
    except PaddleOCRUnavailableError as e:
        if progress_callback is not None:
            progress_callback({
                "event": "ocr_fallback",
                "paper_id": str(meta.get("id") or pdf_path.stem),
                "pdf_path": str(pdf_path),
                "current": 1,
                "total": 1,
                "message": f"Fallback to PyMuPDF: {e}",
            })
        if not getattr(ingest_pdf_auto, "_paddle_warned", False):
            console.print(f"[yellow]{e} Буду использовать локальный парсер (PyMuPDF / pypdf).[/yellow]")
            setattr(ingest_pdf_auto, "_paddle_warned", True)
    except Exception as e:
        if progress_callback is not None:
            progress_callback({
                "event": "ocr_fallback",
                "paper_id": str(meta.get("id") or pdf_path.stem),
                "pdf_path": str(pdf_path),
                "current": 1,
                "total": 1,
                "message": f"PaddleOCR failed -> fallback to PyMuPDF ({type(e).__name__}: {e})",
            })
        console.print(f"[yellow]PaddleOCR failed ({type(e).__name__}: {e}). Falling back to local parsing...[/yellow]")

    return _ingest_pdf_pymupdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
