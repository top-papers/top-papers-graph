from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from scireason.config import settings
from scireason.connectors import (
    arxiv as arxiv_connector,
    biorxiv as biorxiv_connector,
    crossref as crossref_connector,
    europe_pmc as europe_pmc_connector,
    openalex as openalex_connector,
    pubmed as pubmed_connector,
    semantic_scholar as s2_connector,
)

from .normalize import (
    normalize_arxiv,
    normalize_biorxiv,
    normalize_crossref,
    normalize_europe_pmc,
    normalize_openalex,
    normalize_pubmed,
    normalize_semantic_scholar,
)
from .schema import PaperMetadata, PaperSource


DEFAULT_SOURCES: List[PaperSource] = [
    PaperSource.semantic_scholar,
    PaperSource.openalex,
    PaperSource.crossref,
    PaperSource.pubmed,
    PaperSource.europe_pmc,
    PaperSource.arxiv,
    PaperSource.biorxiv,
]


def _ua() -> str | None:
    return settings.user_agent or None


def search_papers(
    query: str,
    *,
    limit: int = 25,
    sources: Optional[Sequence[PaperSource]] = None,
    with_abstracts: bool = False,
) -> List[PaperMetadata]:
    sources = list(sources or DEFAULT_SOURCES)
    out: List[PaperMetadata] = []

    for src in sources:
        try:
            if src == PaperSource.semantic_scholar:
                papers = s2_connector.search_papers(query, limit=limit, api_key=settings.s2_api_key)
                out.extend([normalize_semantic_scholar(p) for p in papers])
            elif src == PaperSource.openalex:
                works = openalex_connector.search_works(
                    query,
                    per_page=min(limit, 200),
                    mailto=settings.openalex_mailto or settings.contact_email,
                    api_key=settings.openalex_api_key,
                    user_agent=_ua(),
                )
                out.extend([normalize_openalex(w) for w in works])
            elif src == PaperSource.crossref:
                items = crossref_connector.search_works(query, rows=min(limit, 100), mailto=settings.crossref_mailto or settings.contact_email, user_agent=_ua())
                out.extend([normalize_crossref(i) for i in items])
            elif src == PaperSource.pubmed:
                recs = pubmed_connector.search(
                    query,
                    retmax=limit,
                    api_key=settings.ncbi_api_key,
                    tool=settings.ncbi_tool,
                    email=settings.ncbi_email or settings.contact_email,
                    user_agent=_ua(),
                    with_abstract=with_abstracts,
                )
                out.extend([normalize_pubmed(r) for r in recs])
            elif src == PaperSource.europe_pmc:
                recs = europe_pmc_connector.search(query, page_size=limit, page=1, user_agent=_ua())
                out.extend([normalize_europe_pmc(r) for r in recs])
            elif src == PaperSource.arxiv:
                recs = arxiv_connector.search(query, start=0, max_results=limit, user_agent=_ua())
                out.extend([normalize_arxiv(r) for r in recs])
            elif src == PaperSource.biorxiv:
                # bioRxiv doesn't support free-text search in this API; we use Europe PMC for that.
                # But we keep this here for DOI-based enrichment flows.
                continue
        except Exception:
            # Keep best-effort across sources; individual connector failures should not fail the whole call.
            continue

    # De-duplicate by canonical id, prefer earlier sources order.
    seen = set()
    uniq: List[PaperMetadata] = []
    for p in out:
        if p.id in seen:
            continue
        seen.add(p.id)
        uniq.append(p)

    return uniq[:limit]


def get_paper_by_doi(doi: str) -> Optional[PaperMetadata]:
    doi = doi.strip()
    if not doi:
        return None

    # Prefer OpenAlex (often has OA location), then Crossref, then bioRxiv details.
    try:
        from urllib.parse import quote
        doi_url = f"https://doi.org/{doi}"
        w = openalex_connector.get_work(
            f"https://api.openalex.org/works/{quote(doi_url, safe='')}",
            mailto=settings.openalex_mailto or settings.contact_email,
            api_key=settings.openalex_api_key,
            user_agent=_ua(),
        )
        return normalize_openalex(w)
    except Exception:
        pass

    try:
        item = crossref_connector.works_by_doi(doi, mailto=settings.crossref_mailto or settings.contact_email, user_agent=_ua())
        return normalize_crossref(item)
    except Exception:
        pass

    try:
        # bioRxiv returns list (versions), pick most recent
        records = biorxiv_connector.details_by_doi(doi, server="biorxiv", user_agent=_ua())
        if records:
            norm = biorxiv_connector.normalize_record(records[-1])
            return normalize_biorxiv(norm)
    except Exception:
        pass

    return None
