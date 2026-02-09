from __future__ import annotations

from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from scireason.net import get_json

S2_API = "https://api.semanticscholar.org/graph/v1"

# A reasonable default subset for paper search.
DEFAULT_FIELDS = "title,authors,year,abstract,externalIds,citationCount,url,openAccessPdf"


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if api_key:
        h["x-api-key"] = api_key
    return h


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def search_papers(
    query: str,
    limit: int = 25,
    api_key: Optional[str] = None,
    fields: str = DEFAULT_FIELDS,
) -> List[Dict[str, Any]]:
    """Search papers via Semantic Scholar Graph API."""

    params = {"query": query, "limit": limit, "fields": fields}
    data = get_json(f"{S2_API}/paper/search", params=params, headers=_headers(api_key), ttl_seconds=24 * 3600)
    papers = data.get("data", []) if isinstance(data, dict) else []
    return papers if isinstance(papers, list) else []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def search_papers_raw(
    query: str,
    limit: int = 25,
    api_key: Optional[str] = None,
    fields: str = DEFAULT_FIELDS,
) -> Dict[str, Any]:
    params = {"query": query, "limit": limit, "fields": fields}
    data = get_json(f"{S2_API}/paper/search", params=params, headers=_headers(api_key), ttl_seconds=24 * 3600)
    return data if isinstance(data, dict) else {}
