from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import httpx

from ..connectors import arxiv as arxiv_conn
from ..connectors import openalex as openalex_conn
from ..connectors import semantic_scholar as s2_conn
from ..connectors import crossref as crossref_conn

ARXIV_PDF_BASE = "https://arxiv.org/pdf"


def normalize_arxiv_id(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip()
    arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "").replace("http://arxiv.org/abs/", "")
    arxiv_id = arxiv_id.replace("https://arxiv.org/pdf/", "").replace("http://arxiv.org/pdf/", "")
    arxiv_id = arxiv_id.replace(".pdf", "")
    return arxiv_id


def download_arxiv_pdf(arxiv_id: str, out_path: Path) -> Path:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    url = f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return out_path


def resolve_metadata(arxiv_id: str) -> Dict[str, Any]:
    """Resolve metadata from arXiv Atom, then enrich via OpenAlex/S2/Crossref when possible."""
    arxiv_id = normalize_arxiv_id(arxiv_id)

    results = arxiv_conn.search(f"id:{arxiv_id}", start=0, max_results=1)
    meta: Dict[str, Any] = {"arxiv_id": arxiv_id}
    if results:
        meta.update(results[0])

    title = meta.get("title") or ""

    # OpenAlex: best effort by title
    if title:
        try:
            oa = openalex_conn.search_works(title, per_page=5)
            if oa:
                meta["openalex"] = oa[0]
                doi = (oa[0].get("doi") or "").replace("https://doi.org/", "")
                if doi:
                    meta["doi"] = doi
        except Exception:
            pass

    # Semantic Scholar: try by id/title (best effort)
    try:
        s2 = s2_conn.search_papers(arxiv_id, limit=5)
        if s2:
            meta["semantic_scholar"] = s2[0]
            if not meta.get("doi") and s2[0].get("doi"):
                meta["doi"] = s2[0].get("doi")
    except Exception:
        pass

    # Crossref: DOI or title match
    try:
        doi = meta.get("doi")
        if doi:
            meta["crossref"] = crossref_conn.get_work_by_doi(doi)
        elif title:
            best = crossref_conn.search_best_match(title=title)
            if best and best.get("DOI"):
                meta["doi"] = best.get("DOI")
                meta["crossref"] = best
    except Exception:
        pass

    return meta


def one_click_ingest_arxiv(arxiv_id: str, raw_dir: Path, meta_dir: Path) -> Tuple[Path, Path]:
    """Download arXiv PDF + save metadata JSON."""
    arxiv_id = normalize_arxiv_id(arxiv_id)
    pdf_path = raw_dir / f"{arxiv_id}.pdf"
    meta_path = meta_dir / f"{arxiv_id}.json"

    download_arxiv_pdf(arxiv_id, pdf_path)
    meta = resolve_metadata(arxiv_id)

    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return pdf_path, meta_path
