from __future__ import annotations

from typing import Any, Dict, List, Optional

from scireason.net import get_json

OPENALEX_API = "https://api.openalex.org"


def search_works(
    query: str,
    per_page: int = 25,
    *,
    mailto: Optional[str] = None,
    api_key: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search works in OpenAlex by a free-text query.

    Returns raw OpenAlex `works` records.

    Notes:
    - OpenAlex supports a `mailto` query parameter (polite pool / contact).
    - Some plans require an API key; pass it as `api_key`.
    """
    params: Dict[str, Any] = {"search": query, "per-page": per_page}
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key

    headers: Dict[str, str] = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    data = get_json(f"{OPENALEX_API}/works", params=params, headers=headers, ttl_seconds=7 * 24 * 3600)
    results = data.get("results", []) if isinstance(data, dict) else []
    return results if isinstance(results, list) else []


def get_work(
    openalex_id_or_url: str,
    *,
    mailto: Optional[str] = None,
    api_key: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch a single work by OpenAlex ID or URL.

    `openalex_id_or_url` can be:
    - "W2741809807"
    - "https://openalex.org/W2741809807"
    - "https://api.openalex.org/works/W2741809807"
    - "https://api.openalex.org/works/<url-encoded-doi-url>"
    """
    if openalex_id_or_url.startswith("http"):
        url = openalex_id_or_url
    else:
        url = f"{OPENALEX_API}/works/{openalex_id_or_url}"

    headers: Dict[str, str] = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    params: Dict[str, Any] = {}
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key

    data = get_json(url, params=params or None, headers=headers, ttl_seconds=30 * 24 * 3600)
    return data if isinstance(data, dict) else {}
