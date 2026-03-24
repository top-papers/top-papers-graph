from __future__ import annotations

import os
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




def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _pdf_looks_text_native(
    pdf_path: Path,
    *,
    probe_pages: int = 4,
    min_chars_per_page: int = 400,
    min_coverage: float = 0.75,
) -> bool:
    """Heuristic: prefer local text extraction for born-digital PDFs.

    Many journal PDFs already contain a dense text layer. Running PP-Structure on
    such files is unnecessarily slow and can trigger worker timeouts while adding
    little value over PyMuPDF for the Task 2 pipeline.
    """

    if not _bool_env("OCR_AUTO_SKIP_PADDLE_FOR_TEXT_NATIVE", True):
        return False

    try:
        import fitz  # type: ignore

        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            if total_pages <= 0:
                return False
            limit = min(total_pages, max(1, int(probe_pages)))
            rich_pages = 0
            for page_index in range(limit):
                page = doc.load_page(page_index)
                text = (page.get_text("text", sort=True) or "").strip()
                if len(text) >= int(min_chars_per_page):
                    rich_pages += 1
            return (rich_pages / float(limit)) >= float(min_coverage)
    except Exception:
        return False

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

    # Auto mode: prefer the lightweight local parser for born-digital PDFs with
    # an obvious text layer; otherwise try structured OCR/layout first.
    if _pdf_looks_text_native(pdf_path):
        if progress_callback is not None:
            progress_callback({
                "event": "ocr_short_circuit",
                "paper_id": str(meta.get("id") or pdf_path.stem),
                "pdf_path": str(pdf_path),
                "current": 1,
                "total": 1,
                "message": "PDF already contains a dense text layer -> using PyMuPDF directly",
            })
        console.print(f"[cyan]Text-native PDF detected:[/cyan] {pdf_path.name} -> using local parser directly.")
        return _ingest_pdf_pymupdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)

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
