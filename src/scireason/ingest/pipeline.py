from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console

from .grobid_client import GrobidUnavailableError, grobid_fulltext, grobid_status
from .tei_to_text import tei_to_plaintext
from .chunking import simple_chunks
from .store import save_paper

console = Console()


def ingest_pdf(pdf_path: Path, meta: Dict[str, Any], out_dir: Path) -> Path:
    ok, _ = grobid_status()
    if ok:
        console.print(f"[cyan]GROBID парсит:[/cyan] {pdf_path}")
    tei = grobid_fulltext(pdf_path)
    text = tei_to_plaintext(tei)
    chunks = simple_chunks(text)
    console.print(f"[green]Извлечено чанков:[/green] {len(chunks)}")
    return save_paper(out_dir, meta=meta, chunks=chunks)


def _extract_text_pymupdf(pdf_path: Path) -> str:
    """Fallback extractor using PyMuPDF (no external GROBID service).

    This is *less accurate* than GROBID for structured scientific PDFs,
    but it makes the end-to-end educational pipeline runnable out-of-the-box.
    """

    # Preferred fallback: PyMuPDF (fast, decent text extraction)
    try:
        import fitz  # PyMuPDF  # type: ignore

        doc = fitz.open(str(pdf_path))
        parts: list[str] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            # `sort=True` often yields a more natural reading order.
            t = (page.get_text("text", sort=True) or "").strip()
            if t:
                parts.append(t)
        return "\n\n".join(parts)
    except Exception:
        # Secondary fallback: pure-python reader (lower quality, but avoids hard-failing)
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Для fallback PDF-парсинга нужен PyMuPDF (pip install pymupdf) "
                "или pypdf (pip install pypdf)."
            ) from e

        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            if t:
                parts.append(t)
        return "\n\n".join(parts)


def ingest_pdf_auto(pdf_path: Path, meta: Dict[str, Any], out_dir: Path) -> Path:
    """Ingest PDF using the best available backend.

    Priority:
    1) GROBID (best for scientific PDFs)
    2) PyMuPDF text extraction (fallback)

    The function is designed for **automation**: it will not fail the whole pipeline
    if GROBID is not running.
    """
    # Avoid spamming connection errors: if GROBID is down, we learn it once (cached in grobid_client)
    # and switch to local extraction for all PDFs.
    try:
        return ingest_pdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    except GrobidUnavailableError as e:
        if not getattr(ingest_pdf_auto, "_grobid_warned", False):
            console.print(f"[yellow]{e} Буду использовать локальный парсер (PyMuPDF / pypdf).[/yellow]")
            setattr(ingest_pdf_auto, "_grobid_warned", True)
    except Exception as e:
        # Unexpected GROBID error on a specific PDF — still fall back.
        console.print(
            f"[yellow]GROBID failed on this PDF ({type(e).__name__}: {e}). Falling back to local parsing...[/yellow]"
        )

    text = _extract_text_pymupdf(pdf_path)
    chunks = simple_chunks(text)
    console.print(f"[green]Extracted chunks (fallback):[/green] {len(chunks)}")
    return save_paper(out_dir, meta=meta, chunks=chunks)
