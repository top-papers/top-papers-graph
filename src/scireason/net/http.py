from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from scireason.config import settings

from .http_client import HttpClient, HttpClientConfig, HostRateLimiter, default_policies


_DEFAULT_CLIENT: HttpClient | None = None


def build_user_agent() -> str:
    # Crossref/OpenAlex/etc. prefer a descriptive UA, often with contact email.
    base = settings.user_agent or "top-papers-graph"
    email = settings.contact_email or settings.crossref_mailto or settings.openalex_mailto or settings.ncbi_email
    if email and email not in base:
        return f"{base} (mailto:{email})"
    return base


def default_client() -> HttpClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        return _DEFAULT_CLIENT

    cache_dir = Path(settings.http_cache_dir).expanduser()
    cfg = HttpClientConfig(
        cache_dir=str(cache_dir),
        default_ttl_seconds=int(settings.http_cache_ttl_seconds),
        timeout_seconds=int(settings.http_timeout_seconds),
    )
    policies = default_policies(ncbi_api_key_present=bool(settings.ncbi_api_key))
    limiter = HostRateLimiter(policies)

    headers = {
        "User-Agent": build_user_agent(),
        "Accept": "application/json, text/xml;q=0.9, */*;q=0.8",
    }

    _DEFAULT_CLIENT = HttpClient(config=cfg, rate_limiter=limiter, default_headers=headers)
    return _DEFAULT_CLIENT


def get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    ttl_seconds: Optional[int] = None,
    use_cache: bool = True,
) -> Any:
    return default_client().get_json(url, params=params, headers=headers, ttl_seconds=ttl_seconds, use_cache=use_cache)


def get_text(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    ttl_seconds: Optional[int] = None,
    use_cache: bool = True,
) -> str:
    return default_client().get_text(url, params=params, headers=headers, ttl_seconds=ttl_seconds, use_cache=use_cache)
