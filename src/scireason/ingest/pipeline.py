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
