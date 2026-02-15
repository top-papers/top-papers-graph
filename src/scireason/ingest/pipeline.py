from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console

from .grobid_client import grobid_fulltext
from .tei_to_text import tei_to_plaintext
from .chunking import simple_chunks
from .store import save_paper

console = Console()


def ingest_pdf(pdf_path: Path, meta: Dict[str, Any], out_dir: Path) -> Path:
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

    try:
        import fitz  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is required for fallback PDF parsing. Install: pip install -e '.[mm]'"
        ) from e

    doc = fitz.open(str(pdf_path))
    parts = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        t = (page.get_text("text") or "").strip()
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
    try:
        return ingest_pdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    except Exception as e:
        console.print(f"[yellow]GROBID failed ({type(e).__name__}: {e}). Falling back to PyMuPDF...[/yellow]")
        text = _extract_text_pymupdf(pdf_path)
        chunks = simple_chunks(text)
        console.print(f"[green]Extracted chunks (fallback):[/green] {len(chunks)}")
        return save_paper(out_dir, meta=meta, chunks=chunks)
