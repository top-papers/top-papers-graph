from __future__ import annotations

"""Utilities to acquire PDFs (best-effort) for a PaperMetadata.

Why this module exists
----------------------
In the educational version of the project, students often had to **download PDFs manually**.
That breaks the requirement of a fully automated pipeline "from query to temporal KG".

This module provides a best-effort downloader that tries multiple candidate URLs:
* `paper.pdf_url` (from OpenAlex/Semantic Scholar/arXiv normalizers)
* known landing-page heuristics (for example ACL Anthology article page -> direct `.pdf`)
* arXiv PDF endpoint (if arXiv id is known)
* landing-page HTML discovery via `citation_pdf_url` / `.pdf` links

Important
---------
* Academic PDFs can be behind paywalls or require cookies.
  We intentionally do not attempt to bypass paywalls.
* If no PDF is available, the pipeline can still proceed using the abstract-only mode.
"""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha1
from pathlib import Path
import re
from typing import Iterable, List, Optional
from urllib.parse import quote, urljoin, urlparse

import httpx
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:  # pragma: no cover
    def retry(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

from ..config import settings
from ..net.http import build_user_agent
from ..net.http_client import HostRateLimiter, default_policies
from ..papers.schema import PaperMetadata


ARXIV_PDF_BASE = "https://arxiv.org/pdf"
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
PLOS_JOURNAL_BY_CODE = {
    "pbio": "plosbiology",
    "pcbi": "ploscompbiol",
    "pgen": "plosgenetics",
    "pmed": "plosmedicine",
    "pntd": "plosntds",
    "pone": "plosone",
    "ppat": "plospathogens",
}
_HTML_PDF_META_RE = re.compile(
    r'<meta[^>]+(?:name|property)=["\'](?:citation_pdf_url|dc\.identifier|og:pdf_url)["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_HTML_HREF_PDF_RE = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE)
_HTML_ANCHOR_RE = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)




def _extract_doi(text: Optional[str]) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    m = DOI_RE.search(raw)
    if not m:
        return None
    return m.group(0).rstrip(").,;]")


def _doi_resolver_url(doi: Optional[str]) -> Optional[str]:
    doi = str(doi or "").strip()
    if not doi:
        return None
    return f"https://doi.org/{doi}"


def _plos_candidate_urls_from_doi(doi: Optional[str]) -> List[str]:
    doi = str(doi or "").strip()
    if not doi or not doi.lower().startswith("10.1371/journal."):
        return []

    m = re.match(r"10\.1371/journal\.([a-z0-9]+)\.(.+)$", doi, flags=re.IGNORECASE)
    if not m:
        return []

    journal_code = m.group(1).lower()
    journal_slug = PLOS_JOURNAL_BY_CODE.get(journal_code)
    if not journal_slug:
        return []

    quoted = quote(doi, safe="")
    return [
        f"https://journals.plos.org/{journal_slug}/article?id={quoted}",
        f"https://journals.plos.org/{journal_slug}/article/file?id={quoted}&type=printable",
    ]


def _pmc_candidate_urls(pmcid: Optional[str]) -> List[str]:
    pmcid = str(pmcid or "").strip().upper()
    if not pmcid:
        return []
    return [
        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/",
    ]


def _pubmed_candidate_urls(pmid: Optional[str]) -> List[str]:
    pmid = str(pmid or "").strip()
    if not pmid:
        return []
    return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"]


def _iter_raw_acquire_hints(paper: PaperMetadata) -> Iterable[str]:
    raw = paper.raw if isinstance(paper.raw, dict) else {}
    hints = raw.get("acquire_hints") or []
    if isinstance(hints, (str, bytes)):
        hints = [hints]
    for hint in hints:
        if hint:
            yield str(hint).strip()


def _safe_stem(s: str, *, max_len: int = 160) -> str:
    """Filesystem-safe stem for paper ids like doi:10.1000/xyz."""
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = s.replace("/", "_").replace(":", "_").replace("\\", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in {"_", "-", "."})
    if len(s) <= max_len:
        return s
    h = sha1(s.encode("utf-8")).hexdigest()[:10]
    return s[: max_len - 11] + "_" + h


def _looks_like_pdf_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ".pdf" in str(url).lower()


@lru_cache(maxsize=256)
def _known_pdf_candidates_from_url(url: str) -> tuple[str, ...]:
    out: list[str] = []
    raw = str(url or "").strip()
    if not raw:
        return tuple()

    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path = parsed.path or ""

    if _looks_like_pdf_url(raw):
        out.append(raw)

    if host.endswith("aclanthology.org") and path and not path.lower().endswith(".pdf"):
        out.append(raw.rstrip("/") + ".pdf")

    if host.endswith("arxiv.org"):
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"abs", "pdf"}:
            aid = parts[1].replace(".pdf", "")
            if aid:
                out.append(f"{ARXIV_PDF_BASE}/{aid}.pdf")

    dedup: list[str] = []
    for item in out:
        if item and item not in dedup:
            dedup.append(item)
    return tuple(dedup)


def _extract_pdf_urls_from_html(html_text: str, *, base_url: str) -> List[str]:
    found: list[str] = []

    def add(url: str) -> None:
        url = str(url or "").strip()
        if not url:
            return
        full = urljoin(base_url, url)
        if full not in found:
            found.append(full)

    html_text = html_text or ""

    for match in _HTML_PDF_META_RE.findall(html_text):
        add(match)
    for match in _HTML_HREF_PDF_RE.findall(html_text):
        add(match)

    for href, anchor_html in _HTML_ANCHOR_RE.findall(html_text):
        summary = re.sub(r"<[^>]+>", " ", anchor_html or " ")
        blob = f"{href} {summary}".lower()
        if any(token in blob for token in ("pdf", "download", "printable", "full text", "article/file")):
            add(href)

    return found


@lru_cache(maxsize=128)
def _resolve_pdf_urls_from_landing_page(url: str) -> tuple[str, ...]:
    dedup: list[str] = list(_known_pdf_candidates_from_url(url))

    raw = str(url or "").strip()
    if not raw or _looks_like_pdf_url(raw):
        return tuple(dedup)

    headers = {
        "User-Agent": build_user_agent(),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    try:
        resp = httpx.get(raw, timeout=25, follow_redirects=True, headers=headers)
        resp.raise_for_status()
    except Exception:
        return tuple(dedup)

    ctype = str(resp.headers.get("content-type") or "").lower()
    if "html" not in ctype and "xml" not in ctype:
        return tuple(dedup)

    for candidate in _extract_pdf_urls_from_html(resp.text, base_url=str(resp.url)):
        if candidate not in dedup:
            dedup.append(candidate)
    return tuple(dedup)


def candidate_pdf_urls(paper: PaperMetadata) -> List[str]:
    """Return candidate article/PDF URLs in priority order.

    The list may contain direct PDF endpoints as well as landing pages that can redirect
    or expose a downloadable PDF/full-text representation. The downloader validates that
    the fetched payload is actually a PDF before accepting it.
    """
    urls: List[str] = []

    def add(u: Optional[str]) -> None:
        if not u:
            return
        u = str(u).strip()
        if not u:
            return
        if u not in urls:
            urls.append(u)

    def add_landing(u: Optional[str]) -> None:
        raw = str(u or "").strip()
        if not raw:
            return
        if _looks_like_pdf_url(raw):
            add(raw)
            return
        for candidate in _known_pdf_candidates_from_url(raw):
            add(candidate)
        for candidate in _resolve_pdf_urls_from_landing_page(raw):
            add(candidate)
        add(raw)

    add(paper.pdf_url)

    doi = None
    pmcid = None
    pmid = None
    if paper.ids:
        doi = paper.ids.doi
        pmcid = paper.ids.pmcid
        pmid = paper.ids.pmid
        if paper.ids.arxiv:
            aid = str(paper.ids.arxiv).strip().replace(".pdf", "")
            if aid:
                add(f"{ARXIV_PDF_BASE}/{aid}.pdf")

    doi = doi or _extract_doi(paper.id) or _extract_doi(paper.url)
    pmcid = pmcid or (paper.id.split(":", 1)[1] if str(paper.id or "").lower().startswith("pmc:") else None)
    pmid = pmid or (paper.id.split(":", 1)[1] if str(paper.id or "").lower().startswith("pmid:") else None)

    for maybe_url in (paper.url, paper.id if str(paper.id or "").startswith(("http://", "https://")) else None):
        if maybe_url:
            add_landing(maybe_url)

    doi_url = _doi_resolver_url(doi)
    if doi_url:
        add_landing(doi_url)
        for candidate in _plos_candidate_urls_from_doi(doi):
            add_landing(candidate)

    for candidate in _pmc_candidate_urls(pmcid):
        add_landing(candidate)
    for candidate in _pubmed_candidate_urls(pmid):
        add_landing(candidate)

    for hint in _iter_raw_acquire_hints(paper):
        doi_hint = _extract_doi(hint)
        if doi_hint and not hint.lower().startswith(("http://", "https://")):
            add_landing(_doi_resolver_url(doi_hint))
            for candidate in _plos_candidate_urls_from_doi(doi_hint):
                add_landing(candidate)
            continue
        add_landing(hint)

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

    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    try:
        with httpx.stream("GET", url, timeout=timeout_seconds, follow_redirects=True, headers=headers) as r:
            r.raise_for_status()
            content_type = str(r.headers.get("content-type") or "").lower()
            iterator = r.iter_bytes()
            try:
                first_chunk = next(iterator)
            except StopIteration as exc:  # pragma: no cover
                raise ValueError("empty response while downloading PDF") from exc

            sniff = bytes(first_chunk[:2048])
            sniff_lower = sniff.lstrip().lower()
            if "html" in content_type or sniff_lower.startswith(b"<!doctype html") or sniff_lower.startswith(b"<html"):
                raise ValueError(f"response is HTML, not PDF: {url}")
            if b"%PDF" not in sniff[:1024]:
                raise ValueError(f"response does not look like a PDF: {url}")

            with tmp.open("wb") as f:
                f.write(first_chunk)
                for chunk in iterator:
                    if chunk:
                        f.write(chunk)
        tmp.replace(out_path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


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

    return AcquireResult(paper_id=paper_id, pdf_path=None, meta_path=meta_path, error=last_err)


def acquire_pdfs(
    papers: Iterable[PaperMetadata],
    *,
    raw_dir: Path = Path("data/raw/papers"),
    meta_dir: Path = Path("data/raw/metadata"),
    prefer_cached: bool = True,
) -> List[AcquireResult]:
    return [acquire_pdf(p, raw_dir=raw_dir, meta_dir=meta_dir, prefer_cached=prefer_cached) for p in papers]
