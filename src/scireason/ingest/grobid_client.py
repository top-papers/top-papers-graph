from __future__ import annotations

from pathlib import Path
import httpx

from ..config import settings


def grobid_fulltext(pdf_path: Path) -> str:
    """Отправляет PDF в GROBID и возвращает TEI XML как строку."""
    url = f"{settings.grobid_url.rstrip('/')}/api/processFulltextDocument"
    with pdf_path.open("rb") as f:
        files = {"input": (pdf_path.name, f, "application/pdf")}
        r = httpx.post(url, files=files, timeout=120)
    r.raise_for_status()
    return r.text
