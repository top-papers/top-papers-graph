from __future__ import annotations

from typing import Any, Dict, List, Optional

from scireason.net import get_json

CROSSREF_API = "https://api.crossref.org"


def _headers(user_agent: Optional[str] = None) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if user_agent:
        h["User-Agent"] = user_agent
    return h


def get_work_by_doi(doi: str, *, mailto: Optional[str] = None, user_agent: Optional[str] = None) -> Dict[str, Any]:
    """Fetch metadata for a DOI using Crossref REST API."""
    doi = doi.strip()
    params: Dict[str, Any] = {}
    if mailto:
        params["mailto"] = mailto

    data = get_json(f"{CROSSREF_API}/works/{doi}", params=params, headers=_headers(user_agent), ttl_seconds=30 * 24 * 3600)
    return data.get("message", {}) if isinstance(data, dict) else {}


def works_by_doi(doi: str, *, mailto: Optional[str] = None, user_agent: Optional[str] = None) -> Dict[str, Any]:
    """Alias kept for backward compatibility."""
    return get_work_by_doi(doi, mailto=mailto, user_agent=user_agent)


def search_best_match(
    title: str,
    first_author: Optional[str] = None,
    *,
    mailto: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Best-effort metadata resolution by title (and optional author)."""
    params: Dict[str, Any] = {"query.title": title, "rows": 5}
    if first_author:
        params["query.author"] = first_author
    if mailto:
        params["mailto"] = mailto

    data = get_json(f"{CROSSREF_API}/works", params=params, headers=_headers(user_agent), ttl_seconds=7 * 24 * 3600)
    items = (data.get("message", {}) if isinstance(data, dict) else {}).get("items", [])
    return items[0] if isinstance(items, list) and items else None


def search_works(
    query: str,
    rows: int = 25,
    *,
    mailto: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search Crossref works by a free-text query."""
    params: Dict[str, Any] = {"query": query, "rows": rows}
    if mailto:
        params["mailto"] = mailto

    data = get_json(f"{CROSSREF_API}/works", params=params, headers=_headers(user_agent), ttl_seconds=7 * 24 * 3600)
    items = (data.get("message", {}) if isinstance(data, dict) else {}).get("items", [])
    return items if isinstance(items, list) else []
