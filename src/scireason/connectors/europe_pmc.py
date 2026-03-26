from __future__ import annotations

from typing import Any, Dict, List, Optional

from scireason.net import get_json

EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _headers(user_agent: Optional[str] = None) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if user_agent:
        h["User-Agent"] = user_agent
    return h


def search(
    query: str,
    *,
    page_size: int = 25,
    page: int = 1,
    user_agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search Europe PMC via REST API.

    Europe PMC aggregates PubMed and many other sources (including preprints).

    Docs: https://europepmc.org/RestfulWebService
    """
    params = {
        "query": query,
        "pageSize": page_size,
        "page": page,
        "format": "json",
    }
    data = get_json(f"{EUROPE_PMC_API}/search", params=params, headers=_headers(user_agent), ttl_seconds=7 * 24 * 3600)

    results = (data.get("resultList") or {}).get("result") or [] if isinstance(data, dict) else []
    out: List[Dict[str, Any]] = []

    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict):
            continue

        doi = (item.get("doi") or "").strip() or None
        pmid = (item.get("pmid") or "").strip() or None
        pmcid = (item.get("pmcid") or "").strip() or None

        src = (item.get("source") or "").strip() or "EUROPEPMC"
        src_id = (item.get("id") or "").strip()

        # Prefer globally stable IDs.
        if doi:
            rid = f"doi:{doi}"
        elif pmid:
            rid = f"pmid:{pmid}"
        elif pmcid:
            rid = f"pmc:{pmcid}"
        elif src_id:
            rid = f"{src.lower()}:{src_id}"
        else:
            rid = ""

        title = (item.get("title") or "").strip()
        year = item.get("pubYear")
        abstract = (item.get("abstractText") or "").strip()

        url = ""
        if src and src_id:
            url = f"https://europepmc.org/article/{src}/{src_id}"

        rec: Dict[str, Any] = {
            "id": rid,
            "source": "europepmc",
            "title": title,
            "year": int(year) if isinstance(year, str) and year.isdigit() else year,
            "authors": (item.get("authorString") or "").strip(),
            "journal": (item.get("journalTitle") or "").strip(),
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "url": url,
            "abstract": abstract,
            "raw": item,
        }
        out.append(rec)

    return out
