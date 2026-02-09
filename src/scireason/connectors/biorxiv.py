from __future__ import annotations

from typing import Any, Dict, List, Optional

from scireason.net import get_json, get_text

BIORXIV_API = "https://api.biorxiv.org"


def _headers(user_agent: Optional[str] = None) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if user_agent:
        h["User-Agent"] = user_agent
    return h


def details_by_doi(
    doi: str,
    *,
    server: str = "biorxiv",
    format: str = "json",
    user_agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch bioRxiv/medRxiv preprint metadata by DOI."""
    doi = doi.strip()
    url = f"{BIORXIV_API}/details/{server}/{doi}/na/{format}"

    if format != "json":
        return [{"raw": get_text(url, headers=_headers(user_agent), ttl_seconds=7 * 24 * 3600)}]

    data = get_json(url, headers=_headers(user_agent), ttl_seconds=7 * 24 * 3600)
    if not isinstance(data, dict):
        return []
    collection = data.get("collection") or []
    return collection if isinstance(collection, list) else []


def details_by_interval(
    *,
    server: str = "biorxiv",
    interval: str,
    cursor: int = 0,
    format: str = "json",
    category: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch bioRxiv/medRxiv preprint metadata for an interval."""
    interval = interval.strip()
    url = f"{BIORXIV_API}/details/{server}/{interval}/{cursor}/{format}"

    params: Dict[str, Any] = {}
    if category:
        params["category"] = category

    if format != "json":
        return [{"raw": get_text(url, params=params or None, headers=_headers(user_agent), ttl_seconds=24 * 3600)}]

    data = get_json(url, params=params or None, headers=_headers(user_agent), ttl_seconds=24 * 3600)
    if not isinstance(data, dict):
        return []
    collection = data.get("collection") or []
    return collection if isinstance(collection, list) else []


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert bioRxiv API record into a minimal normalized metadata dict."""
    doi = (rec.get("doi") or "").strip()
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    date = (rec.get("date") or "").strip()
    server = (rec.get("server") or "").strip() or "biorxiv"

    out: Dict[str, Any] = {
        "id": f"doi:{doi}" if doi else "",
        "doi": doi or None,
        "title": title,
        "abstract": abstract,
        "published": date,
        "source": server.lower(),
        "category": (rec.get("category") or "").strip() or None,
        "authors": (rec.get("authors") or "").strip(),
        "url": f"https://www.{server}.org/content/{doi}v{rec.get('version')}" if doi and rec.get("version") else "",
        "raw": rec,
    }

    if date and len(date) >= 4 and date[:4].isdigit():
        out["year"] = int(date[:4])

    return out
