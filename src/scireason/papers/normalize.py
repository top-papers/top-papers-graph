from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .schema import Author, ExternalIds, PaperMetadata, PaperSource, Venue


def _coalesce(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _authors_from_crossref(item: Dict[str, Any]) -> List[Author]:
    out: List[Author] = []
    for a in item.get("author") or []:
        if not isinstance(a, dict):
            continue
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        name = " ".join([p for p in [given, family] if p]).strip() or (a.get("name") or "").strip()
        if not name:
            continue
        out.append(Author(name=name, orcid=(a.get("ORCID") or None)))
    return out


def _date_from_crossref(item: Dict[str, Any]) -> Optional[str]:
    # pick best available
    for k in ("published-print", "published-online", "created", "issued"):
        v = item.get(k)
        if isinstance(v, dict):
            parts = (v.get("date-parts") or [])
            if parts and isinstance(parts, list) and parts[0]:
                dp = parts[0]
                if isinstance(dp, list) and len(dp) >= 1:
                    y = dp[0]
                    m = dp[1] if len(dp) > 1 else 1
                    d = dp[2] if len(dp) > 2 else 1
                    try:
                        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                    except Exception:
                        continue
    return None


def normalize_crossref(item: Dict[str, Any]) -> PaperMetadata:
    doi = (item.get("DOI") or "").strip() or None
    url = (item.get("URL") or "").strip() or None
    title = ""
    t = item.get("title")
    if isinstance(t, list) and t:
        title = str(t[0] or "").strip()
    elif isinstance(t, str):
        title = t.strip()

    venue_name = None
    ct = item.get("container-title")
    if isinstance(ct, list) and ct:
        venue_name = str(ct[0] or "").strip() or None
    elif isinstance(ct, str):
        venue_name = ct.strip() or None

    publisher = (item.get("publisher") or "").strip() or None
    issn_l = (item.get("ISSN") or None)
    if isinstance(issn_l, list):
        issn_l = issn_l[0] if issn_l else None

    ids = ExternalIds(doi=doi)
    rid = PaperMetadata.build_canonical_id(ids, fallback=f"crossref:{doi or (item.get('DOI') or '')}")

    published = _date_from_crossref(item)
    pd = PaperMetadata.parse_date(published)
    year = pd.year if pd else None

    return PaperMetadata(
        id=rid,
        source=PaperSource.crossref,
        title=title,
        abstract=(item.get("abstract") or None),
        authors=_authors_from_crossref(item),
        venue=Venue(name=venue_name, publisher=publisher, issn_l=issn_l),
        year=year,
        published_date=pd,
        url=url,
        citation_count=(item.get("is-referenced-by-count") if isinstance(item.get("is-referenced-by-count"), int) else None),
        ids=ids,
        raw=item,
    )


def _openalex_abstract_to_text(inv: Any) -> Optional[str]:
    # OpenAlex uses inverted index; reconstruct best-effort.
    # inv = {word: [positions...], ...}
    if not isinstance(inv, dict) or not inv:
        return None
    max_pos = -1
    for positions in inv.values():
        if isinstance(positions, list) and positions:
            try:
                max_pos = max(max_pos, max(int(p) for p in positions))
            except Exception:
                continue
    if max_pos < 0:
        return None
    words = [""] * (max_pos + 1)
    for w, positions in inv.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            try:
                idx = int(p)
                if 0 <= idx < len(words):
                    words[idx] = str(w)
            except Exception:
                continue
    text = " ".join([w for w in words if w])
    return text.strip() or None


def normalize_openalex(work: Dict[str, Any]) -> PaperMetadata:
    ids_dict = work.get("ids") if isinstance(work.get("ids"), dict) else {}
    doi = (ids_dict.get("doi") or work.get("doi") or "").replace("https://doi.org/", "").strip() or None
    openalex = (ids_dict.get("openalex") or work.get("id") or "").strip() or None
    if openalex and openalex.startswith("https://openalex.org/"):
        openalex_id = openalex.split("/")[-1]
    else:
        openalex_id = openalex

    pmid = ids_dict.get("pmid") or None
    if isinstance(pmid, str) and pmid.startswith("https://pubmed.ncbi.nlm.nih.gov/"):
        pmid = pmid.rstrip("/").split("/")[-1]

    arxiv = ids_dict.get("arxiv") or None
    if isinstance(arxiv, str) and "arxiv.org" in arxiv:
        arxiv = arxiv.rstrip("/").split("/")[-1]

    title = (work.get("title") or work.get("display_name") or "").strip()
    published_date = PaperMetadata.parse_date(work.get("publication_date"))
    year = work.get("publication_year")
    if isinstance(year, int):
        yr = year
    else:
        yr = published_date.year if published_date else None

    authors: List[Author] = []
    for a in work.get("authorships") or []:
        if not isinstance(a, dict):
            continue
        author = a.get("author") if isinstance(a.get("author"), dict) else {}
        name = (author.get("display_name") or "").strip()
        if name:
            authors.append(Author(name=name))

    venue_name = None
    host_venue = work.get("primary_location") or {}
    if isinstance(host_venue, dict):
        source = host_venue.get("source") if isinstance(host_venue.get("source"), dict) else {}
        venue_name = (source.get("display_name") or None)

    # Pick a best URL/PDF.
    # OpenAlex may provide multiple "locations" (version of record, preprint, repositories, ...).
    # For *automatic* ingestion we prefer an OA fulltext copy when available.
    url = None
    pdf_url = None

    best_oa_location = work.get("best_oa_location") if isinstance(work.get("best_oa_location"), dict) else {}
    primary_location = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    locations = work.get("locations") if isinstance(work.get("locations"), list) else []

    # URL preference: best OA landing page -> primary landing page -> work id.
    for loc in (best_oa_location, primary_location, *locations):
        if isinstance(loc, dict) and loc.get("landing_page_url"):
            url = str(loc.get("landing_page_url"))
            break
    if not url:
        url = work.get("id")

    # PDF preference: best OA pdf -> primary pdf -> any pdf in locations.
    for loc in (best_oa_location, primary_location, *locations):
        if isinstance(loc, dict) and loc.get("pdf_url"):
            pdf_url = str(loc.get("pdf_url"))
            break

    ids = ExternalIds(doi=doi, openalex=openalex_id, pmid=pmid, arxiv=arxiv)
    rid = PaperMetadata.build_canonical_id(ids, fallback=f"openalex:{openalex_id or ''}")

    abstract = _openalex_abstract_to_text(work.get("abstract_inverted_index"))

    return PaperMetadata(
        id=rid,
        source=PaperSource.openalex,
        title=title,
        abstract=abstract,
        authors=authors,
        venue=Venue(name=venue_name),
        year=yr if isinstance(yr, int) else None,
        published_date=published_date,
        url=url,
        pdf_url=pdf_url,
        citation_count=work.get("cited_by_count") if isinstance(work.get("cited_by_count"), int) else None,
        ids=ids,
        raw=work,
    )


def normalize_semantic_scholar(paper: Dict[str, Any]) -> PaperMetadata:
    ext = paper.get("externalIds") if isinstance(paper.get("externalIds"), dict) else {}
    doi = (ext.get("DOI") or ext.get("doi") or "").strip() or None
    pmid = (ext.get("PubMed") or ext.get("PMID") or "").strip() or None
    arxiv = (ext.get("ArXiv") or ext.get("arXiv") or "").strip() or None
    s2id = (paper.get("paperId") or "").strip() or None

    authors = []
    for a in paper.get("authors") or []:
        if isinstance(a, dict) and a.get("name"):
            authors.append(Author(name=str(a.get("name")).strip()))

    ids = ExternalIds(doi=doi, pmid=pmid, arxiv=arxiv, semantic_scholar=s2id)
    rid = PaperMetadata.build_canonical_id(ids, fallback=f"s2:{s2id or ''}")

    pdf_url = None
    oa = paper.get("openAccessPdf")
    if isinstance(oa, dict):
        pdf_url = (oa.get("url") or None)

    return PaperMetadata(
        id=rid,
        source=PaperSource.semantic_scholar,
        title=(paper.get("title") or "").strip(),
        abstract=(paper.get("abstract") or None),
        authors=authors,
        year=paper.get("year") if isinstance(paper.get("year"), int) else None,
        url=paper.get("url") or None,
        pdf_url=pdf_url,
        citation_count=paper.get("citationCount") if isinstance(paper.get("citationCount"), int) else None,
        ids=ids,
        raw=paper,
    )


def normalize_pubmed(rec: Dict[str, Any]) -> PaperMetadata:
    doi = rec.get("doi") or None
    pmid = rec.get("pmid") or None
    ids = ExternalIds(doi=doi, pmid=pmid)
    rid = PaperMetadata.build_canonical_id(ids, fallback=rec.get("id"))

    authors = []
    for name in rec.get("authors") or []:
        if isinstance(name, str) and name.strip():
            authors.append(Author(name=name.strip()))

    published = PaperMetadata.parse_date(rec.get("published"))
    year = rec.get("year") if isinstance(rec.get("year"), int) else (published.year if published else None)

    return PaperMetadata(
        id=rid,
        source=PaperSource.pubmed,
        title=(rec.get("title") or "").strip(),
        abstract=rec.get("abstract") or None,
        authors=authors,
        year=year,
        published_date=published,
        url=rec.get("url") or None,
        ids=ids,
        raw=rec.get("raw") or rec,
    )


def normalize_europe_pmc(rec: Dict[str, Any]) -> PaperMetadata:
    ids = ExternalIds(
        doi=rec.get("doi") or None,
        pmid=rec.get("pmid") or None,
        pmcid=rec.get("pmcid") or None,
    )
    rid = PaperMetadata.build_canonical_id(ids, fallback=rec.get("id"))
    authors = []
    auth_str = rec.get("authors") or ""
    if isinstance(auth_str, str) and auth_str.strip():
        for part in auth_str.split(","):
            name = part.strip()
            if name:
                authors.append(Author(name=name))

    year = rec.get("year") if isinstance(rec.get("year"), int) else None

    return PaperMetadata(
        id=rid,
        source=PaperSource.europe_pmc,
        title=(rec.get("title") or "").strip(),
        abstract=rec.get("abstract") or None,
        authors=authors,
        year=year,
        venue=Venue(name=rec.get("journal") or None),
        url=rec.get("url") or None,
        ids=ids,
        raw=rec.get("raw") or rec,
    )


def normalize_biorxiv(rec: Dict[str, Any]) -> PaperMetadata:
    doi = rec.get("doi") or None
    server = (rec.get("source") or "biorxiv").lower()
    source = PaperSource.medrxiv if server == "medrxiv" else PaperSource.biorxiv

    ids = ExternalIds(doi=doi)
    rid = PaperMetadata.build_canonical_id(ids, fallback=rec.get("id"))

    authors = []
    auth_str = rec.get("authors") or ""
    if isinstance(auth_str, str) and auth_str.strip():
        # bioRxiv uses "A; B; C"
        for part in auth_str.replace(";", ",").split(","):
            name = part.strip()
            if name:
                authors.append(Author(name=name))

    published = PaperMetadata.parse_date(rec.get("published"))
    year = rec.get("year") if isinstance(rec.get("year"), int) else (published.year if published else None)

    return PaperMetadata(
        id=rid,
        source=source,
        title=(rec.get("title") or "").strip(),
        abstract=rec.get("abstract") or None,
        authors=authors,
        year=year,
        published_date=published,
        url=rec.get("url") or None,
        ids=ids,
        raw=rec.get("raw") or rec,
    )


def normalize_arxiv(rec: Dict[str, Any]) -> PaperMetadata:
    # rec is produced by scireason.connectors.arxiv.search (enhanced)
    arxiv = (rec.get("arxiv_id") or rec.get("id") or "").strip()
    if arxiv.startswith("http"):
        arxiv = arxiv.rstrip("/").split("/")[-1]

    doi = (rec.get("doi") or "").strip() or None

    ids = ExternalIds(doi=doi, arxiv=arxiv)
    rid = PaperMetadata.build_canonical_id(ids, fallback=f"arxiv:{arxiv}")

    authors = []
    for a in rec.get("authors") or []:
        if isinstance(a, str) and a.strip():
            authors.append(Author(name=a.strip()))

    published = PaperMetadata.parse_date(rec.get("published"))
    year = rec.get("year") if isinstance(rec.get("year"), int) else (published.year if published else None)

    return PaperMetadata(
        id=rid,
        source=PaperSource.arxiv,
        title=(rec.get("title") or "").strip(),
        abstract=(rec.get("summary") or rec.get("abstract") or None),
        authors=authors,
        year=year,
        published_date=published,
        url=rec.get("url") or rec.get("id") or None,
        pdf_url=rec.get("pdf_url") or None,
        ids=ids,
        raw=rec.get("raw") or rec,
    )
