from __future__ import annotations

"""Utilities to acquire PDFs (best-effort) for a PaperMetadata.

Why this module exists
----------------------
In the educational version of the project, students often had to **download PDFs manually**.
That breaks the requirement of a fully automated pipeline "from query to temporal KG".

This module provides a best-effort downloader that tries multiple candidate URLs:
* `paper.pdf_url` (from OpenAlex/Semantic Scholar/arXiv normalizers)
* arXiv PDF endpoint (if arXiv id is known)
* landing page URL if it already looks like a PDF

Important
---------
* Academic PDFs can be behind paywalls or require cookies.
  We intentionally do not attempt to bypass paywalls.
* If no PDF is available, the pipeline can still proceed using the abstract-only mode.
"""

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..net.http import build_user_agent
from ..net.http_client import HostRateLimiter, default_policies
from ..papers.schema import PaperMetadata


ARXIV_PDF_BASE = "https://arxiv.org/pdf"


def _safe_stem(s: str, *, max_len: int = 160) -> str:
    """Filesystem-safe stem for paper ids like doi:10.1000/xyz."""
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = s.replace("/", "_").replace(":", "_").replace("\\", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in {"_", "-", "."})
    if len(s) <= max_len:
        return s
    # Keep deterministic suffix so two long ids don't collide.
    h = sha1(s.encode("utf-8")).hexdigest()[:10]
    return s[: max_len - 11] + "_" + h


def candidate_pdf_urls(paper: PaperMetadata) -> List[str]:
    """Return candidate PDF URLs in priority order."""
    urls: List[str] = []

    def add(u: Optional[str]) -> None:
        if not u:
            return
        u = str(u).strip()
        if not u:
            return
        if u not in urls:
            urls.append(u)

    # 1) Normalized field (OpenAlex best_oa_location / arXiv pdf_url / S2 openAccessPdf)
    add(paper.pdf_url)

    # 2) arXiv explicit PDF endpoint
    if paper.ids and paper.ids.arxiv:
        aid = str(paper.ids.arxiv).strip().replace(".pdf", "")
        if aid:
            add(f"{ARXIV_PDF_BASE}/{aid}.pdf")

    # 3) Landing page if it already looks like a PDF
    if paper.url and str(paper.url).lower().endswith(".pdf"):
        add(paper.url)

    return urls


@dataclass
class AcquireResult:
    paper_id: str
    pdf_path: Optional[Path]
    meta_path: Path
    used_pdf_url: Optional[str] = None
    error: Optional[str] = None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=12), reraise=True)
def _download_stream(url: str, out_path: Path, *, timeout_seconds: int = 180) -> None:
    policies = default_policies(ncbi_api_key_present=bool(settings.ncbi_api_key))
    limiter = HostRateLimiter(policies)

    headers = {
        "User-Agent": build_user_agent(),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }

    host = urlparse(url).netloc
    limiter.acquire(host)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    with httpx.stream("GET", url, timeout=timeout_seconds, follow_redirects=True, headers=headers) as r:
        # Some servers return HTML with 200; we still save but keep a lightweight check.
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    tmp.replace(out_path)


def acquire_pdf(
    paper: PaperMetadata,
    *,
    raw_dir: Path = Path("data/raw/papers"),
    meta_dir: Path = Path("data/raw/metadata"),
    prefer_cached: bool = True,
) -> AcquireResult:
    """Download paper PDF (best-effort) and store metadata JSON.

    The function never raises for "no PDF"; it only raises for unexpected IO errors.
    Download failures are returned in `AcquireResult.error`.
    """

    paper_id = paper.id or "unknown:unknown"
    stem = _safe_stem(paper_id)
    pdf_path = raw_dir / f"{stem}.pdf"
    meta_path = meta_dir / f"{stem}.json"

    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(paper.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

    if prefer_cached and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return AcquireResult(paper_id=paper_id, pdf_path=pdf_path, meta_path=meta_path, used_pdf_url="cached")

    urls = candidate_pdf_urls(paper)
    last_err: Optional[str] = None
    for u in urls:
        try:
            _download_stream(u, pdf_path)
            return AcquireResult(paper_id=paper_id, pdf_path=pdf_path, meta_path=meta_path, used_pdf_url=u)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue

    # No success; keep meta.json and return None pdf.
    return AcquireResult(paper_id=paper_id, pdf_path=None, meta_path=meta_path, error=last_err)


def acquire_pdfs(
    papers: Iterable[PaperMetadata],
    *,
    raw_dir: Path = Path("data/raw/papers"),
    meta_dir: Path = Path("data/raw/metadata"),
    prefer_cached: bool = True,
) -> List[AcquireResult]:
    return [acquire_pdf(p, raw_dir=raw_dir, meta_dir=meta_dir, prefer_cached=prefer_cached) for p in papers]
