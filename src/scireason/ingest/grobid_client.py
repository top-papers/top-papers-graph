from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import httpx

from ..config import settings


class GrobidUnavailableError(RuntimeError):
    """Raised when GROBID is not reachable."""


@lru_cache(maxsize=1)
def grobid_status() -> tuple[bool, str]:
    """Best-effort availability check.

    GROBID exposes a lightweight health endpoint (`/api/isalive`). If the server
    is down (e.g., connection refused), we cache the result so we don't spam the
    console with the same connect error for every PDF.
    """

    base = settings.grobid_url.rstrip("/")
    url = f"{base}/api/isalive"
    try:
        r = httpx.get(url, timeout=2.0)
        if r.status_code == 200:
            # Most deployments return plain "true".
            txt = (r.text or "").strip().lower()
            return True, txt
        return False, f"HTTP {r.status_code}"
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"


def grobid_fulltext(pdf_path: Path) -> str:
    """Отправляет PDF в GROBID и возвращает TEI XML как строку."""
    ok, reason = grobid_status()
    if not ok:
        raise GrobidUnavailableError(
            f"GROBID недоступен по {settings.grobid_url!r} ({reason})."
        )
    url = f"{settings.grobid_url.rstrip('/')}/api/processFulltextDocument"
    with pdf_path.open("rb") as f:
        files = {"input": (pdf_path.name, f, "application/pdf")}
        r = httpx.post(url, files=files, timeout=120)
    r.raise_for_status()
    return r.text
