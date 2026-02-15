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
from .schema import ExternalIds, PaperMetadata, PaperSource


def _merge_ids(a, b):
    """Merge ExternalIds objects (best-effort)."""
    try:
        ad = a.model_dump() if a is not None else {}
    except Exception:
        ad = {}
    try:
        bd = b.model_dump() if b is not None else {}
    except Exception:
        bd = {}
    merged = dict(ad)
    for k, v in bd.items():
        if merged.get(k) in (None, "") and v not in (None, ""):
            merged[k] = v
    return ExternalIds(**merged)


def _merge_paper(a: PaperMetadata, b: PaperMetadata) -> PaperMetadata:
    """Merge two normalized records for the same canonical id.

    Goal: keep the record usable for downstream automatic ingestion:
    - prefer a record that has `pdf_url`
    - keep the longest abstract
    - keep max(citation_count)
    - merge external ids
    - keep `raw` as a per-source bundle for debugging
    """

    if a.id != b.id:
        return a

    # Prefer record that has a PDF URL
    preferred = b if (not a.pdf_url and b.pdf_url) else a
    other = a if preferred is b else b

    data = preferred.model_dump()

    # Fill blanks from the other record
    for field in ("title", "url", "pdf_url"):
        if not data.get(field) and getattr(other, field, None):
            data[field] = getattr(other, field)

    # Abstract: prefer longer
    abs_a = (preferred.abstract or "").strip()
    abs_b = (other.abstract or "").strip()
    if len(abs_b) > len(abs_a):
        data["abstract"] = other.abstract

    # Authors
    if (not preferred.authors) and other.authors:
        data["authors"] = other.authors

    # Venue
    if (preferred.venue is None) and (other.venue is not None):
        data["venue"] = other.venue

    # Year / published_date
    if data.get("year") is None and other.year is not None:
        data["year"] = other.year
    if data.get("published_date") is None and other.published_date is not None:
        data["published_date"] = other.published_date

    # Citation count: max
    ca = preferred.citation_count
    cb = other.citation_count
    if isinstance(cb, int) and (not isinstance(ca, int) or cb > ca):
        data["citation_count"] = cb

    # IDs
    data["ids"] = _merge_ids(preferred.ids, other.ids)

    # raw bundle
    raw_bundle = {}
    if isinstance(preferred.raw, dict):
        raw_bundle[getattr(preferred.source, "value", str(preferred.source))] = preferred.raw
    if isinstance(other.raw, dict):
        raw_bundle[getattr(other.source, "value", str(other.source))] = other.raw
    if raw_bundle:
        data["raw"] = raw_bundle

    # Keep canonical id stable; pick a best source label
    data["source"] = preferred.source
    return PaperMetadata(**data)


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

    # De-duplicate by canonical id **with field-level merging**.
    # This is important for a fully automatic ingestion pipeline because different APIs
    # have complementary strengths (e.g., OpenAlex often provides OA pdf_url, Semantic Scholar
    # provides strong abstracts/citations).
    order: List[str] = []
    by_id: dict[str, PaperMetadata] = {}
    for p in out:
        if not p.id:
            continue
        if p.id not in by_id:
            by_id[p.id] = p
            order.append(p.id)
        else:
            by_id[p.id] = _merge_paper(by_id[p.id], p)

    merged = [by_id[i] for i in order]
    return merged[:limit]


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
