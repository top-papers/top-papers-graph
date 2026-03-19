from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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

        doc = fitz.open(str(pdf_path))
        parts: list[str] = []
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


def ingest_pdf_auto(pdf_path: Path, meta: Dict[str, Any], out_dir: Path) -> Path:
    """Ingest PDF using the best available backend.

    Priority in `OCR_BACKEND=auto`:
    1) PaddleOCR / PP-StructureV3 (preferred structured OCR/layout pipeline)
    2) GROBID (best text parser for scientific PDFs when service is up)
    3) PyMuPDF / pypdf fallback
    """

    backend = str(getattr(settings, "ocr_backend", "auto") or "auto").strip().lower()

    if backend in {"paddleocr", "paddle", "ppstructure", "pp_structure"}:
        return ingest_pdf_paddleocr(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    if backend in {"pymupdf", "fitz", "local"}:
        return _ingest_pdf_pymupdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    if backend == "grobid":
        return ingest_pdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)

    # Auto mode: prefer structured OCR if available, then GROBID, then local fallback.
    try:
        return ingest_pdf_paddleocr(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    except PaddleOCRUnavailableError:
        pass
    except Exception as e:
        console.print(f"[yellow]PaddleOCR failed ({type(e).__name__}: {e}). Trying GROBID...[/yellow]")

    try:
        return ingest_pdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    except GrobidUnavailableError as e:
        if not getattr(ingest_pdf_auto, "_grobid_warned", False):
            console.print(f"[yellow]{e} Буду использовать локальный парсер (PyMuPDF / pypdf).[/yellow]")
            setattr(ingest_pdf_auto, "_grobid_warned", True)
    except Exception as e:
        console.print(f"[yellow]GROBID failed ({type(e).__name__}: {e}). Falling back to local parsing...[/yellow]")

    return _ingest_pdf_pymupdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
