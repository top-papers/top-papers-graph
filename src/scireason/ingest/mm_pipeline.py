from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from rich.console import Console

from .pipeline import ingest_pdf, ingest_pdf_auto
from ..mm.pdf_mm_extract import extract_pages

console = Console()


def ingest_pdf_multimodal(pdf_path: Path, meta: Dict[str, Any], out_dir: Path, run_vlm: bool = True) -> Path:
    """1) Текст: GROBID -> chunks (как в базовом пайплайне)
    2) Мультимодальность: PyMuPDF -> страницы + картинки (+ опционально VLM)

    В результате paper_dir содержит:
      - meta.json
      - chunks.jsonl
      - mm/pages.jsonl + mm/images/page_XXX.png
    """
    paper_dir = ingest_pdf(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    paper_id = meta.get("id") or paper_dir.name
    prompt_context = meta.get("title", "") or meta.get("domain", "Science")

    console.print(f"[cyan]MM extract pages:[/cyan] {pdf_path}")
    extract_pages(
        pdf_path=pdf_path,
        paper_id=paper_id,
        out_dir=paper_dir,
        prompt_context=prompt_context,
        run_vlm=run_vlm,
    )
    console.print(f"[green]MM stored:[/green] {paper_dir / 'mm'}")
    return paper_dir


def ingest_pdf_multimodal_auto(pdf_path: Path, meta: Dict[str, Any], out_dir: Path, run_vlm: bool = True) -> Path:
    """Multimodal ingest with automatic fallback for PDF->text.

    Uses `ingest_pdf_auto()` for text (GROBID -> PyMuPDF fallback), then runs the multimodal stage.
    """
    paper_dir = ingest_pdf_auto(pdf_path=pdf_path, meta=meta, out_dir=out_dir)
    paper_id = meta.get("id") or paper_dir.name
    prompt_context = meta.get("title", "") or meta.get("domain", "Science")

    console.print(f"[cyan]MM extract pages:[/cyan] {pdf_path}")
    extract_pages(
        pdf_path=pdf_path,
        paper_id=paper_id,
        out_dir=paper_dir,
        prompt_context=prompt_context,
        run_vlm=run_vlm,
    )
    console.print(f"[green]MM stored:[/green] {paper_dir / 'mm'}")
    return paper_dir
