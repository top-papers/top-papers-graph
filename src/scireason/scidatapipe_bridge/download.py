from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import urljoin, urlparse

import httpx
import yaml

from scireason.ingest.mm_pipeline import ingest_pdf_multimodal_auto
from scireason.ingest.pipeline import ingest_pdf_auto
from scireason.net.http import build_user_agent

from .discovery import is_task2_bundle_dir
from .vendor.common.utils.paper_ids import paper_slug, resolve

logger = logging.getLogger(__name__)

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
PMID_RE = re.compile(r"^(?:pmid\s*:\s*)?(\d{4,10})$", re.IGNORECASE)
PMCID_RE = re.compile(r"^(?:pmcid\s*:\s*)?(PMC\d+)$", re.IGNORECASE)
OPENALEX_RE = re.compile(r"^(?:openalex\s*:\s*)?(W\d+)$", re.IGNORECASE)
OPENALEX_URL_RE = re.compile(r"openalex\.org/(W\d+)", re.IGNORECASE)
ARXIV_RE = re.compile(r"^(?:arxiv\s*:\s*)?([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)
REQUEST_TIMEOUT = 45.0
RATE_LIMIT_SEC = 0.8


@dataclass(frozen=True)
class ExternalRef:
    id: str
    paper_type: str
    raw: str
    doi: str = ""
    url: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    pmcid: str = ""
    openalex_id: str = ""


@dataclass(frozen=True)
class DownloadedPaper:
    ref_id: str
    paper_type: str
    slug: str
    pdf_path: Optional[Path] = None
    html_path: Optional[Path] = None
    source: str = ""
    error: str = ""
    ingested_paper_dir: Optional[Path] = None


@dataclass(frozen=True)
class DownloadSummary:
    refs_total: int
    refs_supported: int
    pdf_downloaded: int
    html_downloaded: int
    ingested_processed_papers: int
    skipped_existing: int
    errors: int
    produced_processed_papers_dir: Optional[Path]
    download_root: Path
    records: list[DownloadedPaper]


class Downloader:
    def __init__(self, *, unpaywall_email: str | None = None) -> None:
        self.unpaywall_email = (unpaywall_email or "").strip() or None
        self.headers_json = {
            "User-Agent": build_user_agent(),
            "Accept": "application/json, text/html, application/xhtml+xml;q=0.9, */*;q=0.8",
        }
        self.headers_pdf = {
            "User-Agent": build_user_agent(),
            "Accept": "application/pdf, application/octet-stream;q=0.9, */*;q=0.8",
        }

    def get(self, url: str, *, accept_pdf: bool = False) -> Optional[httpx.Response]:
        headers = self.headers_pdf if accept_pdf else self.headers_json
        try:
            response = httpx.get(url, headers=headers, timeout=REQUEST_TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            time.sleep(RATE_LIMIT_SEC)
            return response
        except Exception as exc:
            logger.debug("GET %s failed: %s", url, exc)
            return None



def canonicalize_article_ref(raw: str) -> ExternalRef:
    original = str(raw or "").strip()
    if not original:
        return ExternalRef(id="", paper_type="unsupported", raw="")

    openalex_match = OPENALEX_RE.match(original) or OPENALEX_URL_RE.search(original)
    if openalex_match:
        work_id = openalex_match.group(1).upper()
        return ExternalRef(id=f"openalex:{work_id}", paper_type="openalex", raw=original, openalex_id=work_id)

    pmcid_match = PMCID_RE.match(original)
    if pmcid_match:
        pmcid = pmcid_match.group(1).upper()
        return ExternalRef(id=f"pmcid:{pmcid}", paper_type="pmcid", raw=original, pmcid=pmcid)

    pmid_match = PMID_RE.match(original)
    if pmid_match and not DOI_RE.search(original):
        pmid = pmid_match.group(1)
        return ExternalRef(id=f"pmid:{pmid}", paper_type="pmid", raw=original, pmid=pmid)

    arxiv_match = ARXIV_RE.match(original)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        return ExternalRef(id=f"arxiv:{arxiv_id}", paper_type="arxiv", raw=original, arxiv_id=arxiv_id)

    resolved = resolve(original)
    if resolved.paper_type == "arxiv":
        return ExternalRef(id=resolved.id, paper_type="arxiv", raw=original, arxiv_id=resolved.arxiv_id or "")
    if resolved.paper_type == "doi":
        doi = resolved.id.split(":", 1)[1]
        return ExternalRef(id=resolved.id, paper_type="doi", raw=original, doi=doi)
    if resolved.paper_type == "wiki":
        slug = resolved.id.split(":", 1)[1]
        return ExternalRef(id=resolved.id, paper_type="wiki", raw=original, url=f"https://en.wikipedia.org/wiki/{slug}")
    if resolved.paper_type == "url":
        body = resolved.id.split(":", 1)[1] if ":" in resolved.id else resolved.raw
        return ExternalRef(id=resolved.id, paper_type="url", raw=original, url=body)

    return ExternalRef(id=resolved.id or f"id:{original}", paper_type="unsupported", raw=original)



def collect_refs_from_task1_files(task1_files: Sequence[Path]) -> list[ExternalRef]:
    refs: list[str] = []
    for path in task1_files:
        try:
            doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        for paper in doc.get("papers") or []:
            if not isinstance(paper, dict):
                continue
            for key in ("raw", "id", "doi", "url", "paper_id", "source_id"):
                value = paper.get(key)
                if isinstance(value, str) and value.strip():
                    refs.append(value.strip())
        for step in doc.get("steps") or []:
            if not isinstance(step, dict):
                continue
            for src in step.get("sources") or []:
                if not isinstance(src, dict):
                    continue
                for key in ("paper_ref_id", "source", "url", "paper_id"):
                    value = src.get(key)
                    if isinstance(value, str) and value.strip():
                        refs.append(value.strip())
    return _dedupe_refs(canonicalize_article_ref(ref) for ref in refs)



def collect_refs_from_task2_inputs(task2_inputs: Sequence[Path]) -> list[ExternalRef]:
    refs: list[str] = []
    for path in task2_inputs:
        bundle_root = Path(path)
        if bundle_root.is_dir() and is_task2_bundle_dir(bundle_root):
            refs.extend(_collect_refs_from_bundle_dir(bundle_root))
            continue
        if bundle_root.is_dir():
            for candidate in sorted(bundle_root.rglob("edge_reviews.json")):
                refs.extend(_collect_refs_from_edge_reviews(candidate))
            continue
        if bundle_root.is_file() and bundle_root.name == "edge_reviews.json":
            refs.extend(_collect_refs_from_edge_reviews(bundle_root))
    return _dedupe_refs(canonicalize_article_ref(ref) for ref in refs)



def download_and_ingest_refs(
    refs: Sequence[ExternalRef],
    *,
    download_root: Path,
    processed_papers_dir: Optional[Path],
    existing_processed_papers_dirs: Sequence[Path] = (),
    unpaywall_email: str | None = None,
    ingest_downloaded: bool = True,
    multimodal: bool = True,
    run_vlm: bool = True,
    prefer_cached: bool = True,
) -> DownloadSummary:
    download_root = download_root.resolve()
    pdf_dir = download_root / "pdfs"
    html_dir = download_root / "html"
    meta_dir = download_root / "meta"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    produced_processed_dir = processed_papers_dir.resolve() if processed_papers_dir is not None else None
    if produced_processed_dir is not None:
        produced_processed_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = _existing_processed_ids(existing_processed_papers_dirs)
    downloader = Downloader(unpaywall_email=unpaywall_email)
    records: list[DownloadedPaper] = []
    refs_supported = 0
    skipped_existing = 0
    pdf_downloaded = 0
    html_downloaded = 0
    ingested_count = 0
    errors = 0

    for ref in refs:
        if not ref.id or ref.paper_type == "unsupported":
            records.append(DownloadedPaper(ref_id=ref.id or ref.raw, paper_type=ref.paper_type, slug=paper_slug(ref.raw), error="unsupported_identifier"))
            errors += 1
            continue
        refs_supported += 1
        slug = paper_slug(ref.id)
        if ref.id in existing_ids:
            records.append(DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=slug, source="existing_processed"))
            skipped_existing += 1
            continue

        downloaded = _download_ref(ref, pdf_dir=pdf_dir, html_dir=html_dir, meta_dir=meta_dir, downloader=downloader, prefer_cached=prefer_cached)
        ingested_paper_dir: Optional[Path] = None
        if downloaded.pdf_path is not None:
            pdf_downloaded += 1
        elif downloaded.html_path is not None:
            html_downloaded += 1
        else:
            errors += 1

        if ingest_downloaded and produced_processed_dir is not None and downloaded.pdf_path is not None:
            meta = _build_meta(ref, downloaded)
            try:
                if multimodal:
                    # Use the auto multimodal ingester instead of the legacy GROBID-only path.
                    # Colab does not start a local GROBID service by default; the auto path
                    # falls back to local PyMuPDF/pypdf text extraction and still renders
                    # page PNGs into processed_papers/<paper>/mm/images/.
                    ingested_paper_dir = ingest_pdf_multimodal_auto(
                        downloaded.pdf_path,
                        meta,
                        produced_processed_dir,
                        run_vlm=run_vlm,
                    )
                else:
                    ingested_paper_dir = ingest_pdf_auto(downloaded.pdf_path, meta, produced_processed_dir)
                existing_ids.add(ref.id)
                ingested_count += 1
            except Exception as exc:
                logger.warning("failed to ingest %s: %s", ref.id, exc)
                if not downloaded.error:
                    downloaded = DownloadedPaper(
                        ref_id=downloaded.ref_id,
                        paper_type=downloaded.paper_type,
                        slug=downloaded.slug,
                        pdf_path=downloaded.pdf_path,
                        html_path=downloaded.html_path,
                        source=downloaded.source,
                        error=f"ingest_failed: {exc}",
                    )
                    errors += 1
        records.append(
            DownloadedPaper(
                ref_id=downloaded.ref_id,
                paper_type=downloaded.paper_type,
                slug=downloaded.slug,
                pdf_path=downloaded.pdf_path,
                html_path=downloaded.html_path,
                source=downloaded.source,
                error=downloaded.error,
                ingested_paper_dir=ingested_paper_dir,
            )
        )

    return DownloadSummary(
        refs_total=len(refs),
        refs_supported=refs_supported,
        pdf_downloaded=pdf_downloaded,
        html_downloaded=html_downloaded,
        ingested_processed_papers=ingested_count,
        skipped_existing=skipped_existing,
        errors=errors,
        produced_processed_papers_dir=produced_processed_dir,
        download_root=download_root,
        records=records,
    )



def _collect_refs_from_bundle_dir(bundle_dir: Path) -> list[str]:
    refs: list[str] = []
    edge_reviews = bundle_dir / "edge_reviews.json"
    if edge_reviews.exists():
        refs.extend(_collect_refs_from_edge_reviews(edge_reviews))
    for candidate in ("gold.json", "auto.json"):
        path = bundle_dir / candidate
        if path.exists():
            refs.extend(_collect_refs_from_gold_auto(path))
    return refs



def _collect_refs_from_edge_reviews(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    refs: list[str] = []
    assertions = payload.get("assertions") if isinstance(payload, dict) else []
    for item in assertions or []:
        if not isinstance(item, dict):
            continue
        for paper_id in item.get("paper_ids") or []:
            if isinstance(paper_id, str) and paper_id.strip():
                refs.append(paper_id.strip())
        evidence = _safe_parse_mapping(item.get("evidence_payload_full") or item.get("evidence"))
        for key in ("paper_id", "source", "url", "doi"):
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(value.strip())
    return refs



def _collect_refs_from_gold_auto(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    refs: list[str] = []
    for assertion in payload.get("assertions") or []:
        if not isinstance(assertion, dict):
            continue
        for paper_id in assertion.get("paper_ids") or []:
            if isinstance(paper_id, str) and paper_id.strip():
                refs.append(paper_id.strip())
        evidence = assertion.get("evidence") if isinstance(assertion.get("evidence"), dict) else {}
        for key in ("paper_id", "source", "url"):
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(value.strip())
    return refs



def _dedupe_refs(refs: Iterable[ExternalRef]) -> list[ExternalRef]:
    seen: set[str] = set()
    out: list[ExternalRef] = []
    for ref in refs:
        key = ref.id or ref.raw
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out



def _existing_processed_ids(roots: Sequence[Path]) -> set[str]:
    ids: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.exists() or not root.is_dir():
            continue
        candidates = [root] if (root / "meta.json").exists() else [p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()]
        for paper_dir in candidates:
            try:
                meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            value = str(meta.get("id") or "").strip()
            if value:
                ids.add(value)
    return ids



def _safe_parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        for loader in (json.loads, _literal_eval_mapping):
            try:
                parsed = loader(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return {}



def _literal_eval_mapping(text: str) -> dict[str, Any]:
    import ast

    parsed = ast.literal_eval(text)
    return parsed if isinstance(parsed, dict) else {}



def _download_ref(ref: ExternalRef, *, pdf_dir: Path, html_dir: Path, meta_dir: Path, downloader: Downloader, prefer_cached: bool) -> DownloadedPaper:
    slug = paper_slug(ref.id)
    cached_pdf = pdf_dir / f"{slug}.pdf"
    cached_html = html_dir / f"{slug}.html"
    if prefer_cached and cached_pdf.exists() and cached_pdf.stat().st_size > 0:
        return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=slug, pdf_path=cached_pdf, source="cache")
    if prefer_cached and cached_html.exists() and cached_html.stat().st_size > 0:
        return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=slug, html_path=cached_html, source="cache")

    result = None
    if ref.paper_type == "arxiv":
        result = _download_from_candidates(ref, [("arxiv", f"https://arxiv.org/pdf/{ref.arxiv_id}.pdf")], downloader, cached_pdf, cached_html)
    elif ref.paper_type == "doi":
        result = _download_doi(ref, downloader, cached_pdf, cached_html)
    elif ref.paper_type == "wiki":
        result = _download_wiki(ref, downloader, cached_pdf, cached_html)
    elif ref.paper_type == "url":
        result = _download_url(ref, downloader, cached_pdf, cached_html)
    elif ref.paper_type == "pmcid":
        result = _download_pmcid(ref, downloader, cached_pdf, cached_html)
    elif ref.paper_type == "pmid":
        result = _download_pmid(ref, downloader, cached_pdf, cached_html)
    elif ref.paper_type == "openalex":
        result = _download_openalex(ref, downloader, cached_pdf, cached_html)

    meta_payload = {
        "id": ref.id,
        "paper_type": ref.paper_type,
        "raw": ref.raw,
        "doi": ref.doi,
        "url": ref.url,
        "arxiv_id": ref.arxiv_id,
        "pmid": ref.pmid,
        "pmcid": ref.pmcid,
        "openalex_id": ref.openalex_id,
        "download": {
            "pdf_path": str(result.pdf_path) if result and result.pdf_path else "",
            "html_path": str(result.html_path) if result and result.html_path else "",
            "source": result.source if result else "",
            "error": result.error if result else "unsupported_identifier",
        },
    }
    (meta_dir / f"{slug}.json").write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if result is not None:
        return result
    return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=slug, error="unsupported_identifier")



def _download_doi(ref: ExternalRef, downloader: Downloader, pdf_path: Path, html_path: Path) -> DownloadedPaper:
    doi = ref.doi
    candidates: list[tuple[str, str]] = []

    if downloader.unpaywall_email:
        meta_resp = downloader.get(f"https://api.unpaywall.org/v2/{doi}?email={downloader.unpaywall_email}")
        if meta_resp is not None:
            try:
                payload = meta_resp.json()
                best = payload.get("best_oa_location") or {}
                for url in (best.get("url_for_pdf"), best.get("url")):
                    if isinstance(url, str) and url.strip():
                        candidates.append(("unpaywall", url.strip()))
            except Exception:
                pass

    s2_resp = downloader.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf,title,year")
    if s2_resp is not None:
        try:
            payload = s2_resp.json()
            oa = payload.get("openAccessPdf") or {}
            url = oa.get("url")
            if isinstance(url, str) and url.strip():
                candidates.append(("semanticscholar", url.strip()))
        except Exception:
            pass

    candidates.append(("doi_landing", f"https://doi.org/{doi}"))
    return _download_from_candidates(ref, candidates, downloader, pdf_path, html_path)



def _download_wiki(ref: ExternalRef, downloader: Downloader, pdf_path: Path, html_path: Path) -> DownloadedPaper:
    slug = ref.id.split(":", 1)[1]
    lang = "en"
    body = slug
    if ":" in slug and len(slug.split(":", 1)[0]) <= 3:
        lang, body = slug.split(":", 1)
    candidates = [
        ("wiki_pdf", f"https://{lang}.wikipedia.org/api/rest_v1/page/pdf/{body}"),
        ("wiki_html", f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{body}"),
    ]
    return _download_from_candidates(ref, candidates, downloader, pdf_path, html_path)



def _download_url(ref: ExternalRef, downloader: Downloader, pdf_path: Path, html_path: Path) -> DownloadedPaper:
    return _download_from_candidates(ref, [("url", ref.url or ref.raw)], downloader, pdf_path, html_path)



def _download_pmcid(ref: ExternalRef, downloader: Downloader, pdf_path: Path, html_path: Path) -> DownloadedPaper:
    pmcid = ref.pmcid.upper()
    candidates = [
        ("pmc_pdf", f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"),
        ("pmc_html", f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"),
    ]
    return _download_from_candidates(ref, candidates, downloader, pdf_path, html_path)



def _download_pmid(ref: ExternalRef, downloader: Downloader, pdf_path: Path, html_path: Path) -> DownloadedPaper:
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{ref.pmid}/"
    response = downloader.get(pubmed_url)
    if response is None:
        return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=paper_slug(ref.id), error="pmid: no accessible HTML")
    html = response.text or ""
    pdf_candidates = _extract_pdf_candidates_from_html(html, base_url=str(response.url))
    if pdf_candidates:
        return _download_from_candidates(ref, [("pmid_pdf", url) for url in pdf_candidates] + [("pmid_html", pubmed_url)], downloader, pdf_path, html_path)
    return _persist_html(ref, html_path, html, source="pmid_html")



def _download_openalex(ref: ExternalRef, downloader: Downloader, pdf_path: Path, html_path: Path) -> DownloadedPaper:
    work_id = ref.openalex_id.upper()
    response = downloader.get(f"https://api.openalex.org/works/{work_id}")
    if response is None:
        return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=paper_slug(ref.id), error="openalex: no metadata")
    try:
        payload = response.json()
    except Exception:
        return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=paper_slug(ref.id), error="openalex: invalid metadata")

    candidates: list[tuple[str, str]] = []
    best = payload.get("best_oa_location") or {}
    primary = payload.get("primary_location") or {}
    open_access = payload.get("open_access") or {}
    for container in (best, primary, open_access):
        if not isinstance(container, dict):
            continue
        for key in ("pdf_url", "landing_page_url"):
            url = container.get(key)
            if isinstance(url, str) and url.strip():
                candidates.append(("openalex", url.strip()))
    doi = (((payload.get("ids") or {}) if isinstance(payload.get("ids"), dict) else {}).get("doi") or "").strip()
    if doi:
        candidates.append(("openalex_doi", doi))
    if not candidates:
        return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=paper_slug(ref.id), error="openalex: no candidate URLs")

    expanded: list[tuple[str, str]] = []
    for source, url in candidates:
        if DOI_RE.search(url) and not url.lower().startswith(("http://", "https://")):
            expanded.append((source, f"https://doi.org/{url}"))
        else:
            expanded.append((source, url))
    return _download_from_candidates(ref, expanded, downloader, pdf_path, html_path)



def _download_from_candidates(ref: ExternalRef, candidates: Sequence[tuple[str, str]], downloader: Downloader, pdf_path: Path, html_path: Path) -> DownloadedPaper:
    slug = paper_slug(ref.id)
    seen: set[str] = set()
    last_error = ""
    for source, url in candidates:
        url = str(url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        response = downloader.get(url, accept_pdf=True)
        if response is None:
            last_error = f"{source}: no response"
            continue
        ctype = str(response.headers.get("content-type") or "").lower()
        if _looks_like_pdf(response.content, ctype):
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(response.content)
            return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=slug, pdf_path=pdf_path, source=source)
        html = response.text or ""
        pdf_candidates = _extract_pdf_candidates_from_html(html, base_url=str(response.url))
        if pdf_candidates:
            for nested in pdf_candidates:
                nested_response = downloader.get(nested, accept_pdf=True)
                if nested_response is None:
                    last_error = f"{source}: nested no response"
                    continue
                nested_ctype = str(nested_response.headers.get("content-type") or "").lower()
                if _looks_like_pdf(nested_response.content, nested_ctype):
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    pdf_path.write_bytes(nested_response.content)
                    return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=slug, pdf_path=pdf_path, source=f"{source}_nested")
        if "html" in ctype or html:
            return _persist_html(ref, html_path, html, source=source)
        last_error = f"{source}: unsupported content-type {ctype}"
    return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=slug, error=last_error or "no accessible PDF/HTML")



def _persist_html(ref: ExternalRef, html_path: Path, html: str, *, source: str) -> DownloadedPaper:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    return DownloadedPaper(ref_id=ref.id, paper_type=ref.paper_type, slug=paper_slug(ref.id), html_path=html_path, source=source)



def _looks_like_pdf(content: bytes, content_type: str = "") -> bool:
    if content_type and "pdf" in content_type:
        return True
    return bytes(content[:5]) == b"%PDF-"



def _extract_pdf_candidates_from_html(html_text: str, *, base_url: str) -> list[str]:
    html_text = html_text or ""
    found: list[str] = []
    patterns = [
        re.compile(r'<meta[^>]+(?:name|property)=["\'](?:citation_pdf_url|dc\.identifier|og:pdf_url)["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE),
    ]
    for pattern in patterns:
        for match in pattern.findall(html_text):
            url = urljoin(base_url, match)
            if url not in found:
                found.append(url)
    anchor_re = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    for href, anchor_html in anchor_re.findall(html_text):
        summary = re.sub(r"<[^>]+>", " ", anchor_html or " ").lower()
        blob = f"{href} {summary}".lower()
        if any(token in blob for token in ("pdf", "download", "printable", "full text", "article/file")):
            url = urljoin(base_url, href)
            if url not in found:
                found.append(url)
    return found



def _build_meta(ref: ExternalRef, downloaded: DownloadedPaper) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "id": ref.id,
        "title": ref.id,
        "source": downloaded.source,
        "url": ref.url or (f"https://doi.org/{ref.doi}" if ref.doi else ""),
    }
    if ref.doi:
        meta["doi"] = ref.doi
    if ref.openalex_id:
        meta["openalex_id"] = ref.openalex_id
    if ref.pmid:
        meta["pmid"] = ref.pmid
    if ref.pmcid:
        meta["pmcid"] = ref.pmcid
    if ref.arxiv_id:
        meta["arxiv_id"] = ref.arxiv_id
    return meta
