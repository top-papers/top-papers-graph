from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .cache import DiskJSONCache
from .ratelimit import HostRateLimiter, RateLimitPolicy


@dataclass(frozen=True)
class HttpClientConfig:
    cache_dir: str
    default_ttl_seconds: int = 7 * 24 * 3600
    timeout_seconds: int = 30


class HttpClient:
    """HTTP client with:
    - disk cache for GET responses
    - per-host rate limiting (min interval)
    - retries for transient failures / 429/5xx
    """

    def __init__(
        self,
        *,
        config: HttpClientConfig,
        rate_limiter: HostRateLimiter,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.config = config
        self.cache = DiskJSONCache(config.cache_dir)
        self.rate_limiter = rate_limiter
        self.default_headers = default_headers or {}
        self._client = httpx.Client(timeout=config.timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def _cache_key(self, url: str, params: Optional[Dict[str, Any]]) -> str:
        # Stable key: URL + sorted querystring.
        qp = urlencode(sorted((params or {}).items()), doseq=True)
        raw = f"GET\n{url}\n{qp}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
        reraise=True,
    )
    def get_text(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        ttl_seconds: Optional[int] = None,
        use_cache: bool = True,
    ) -> str:
        ttl = int(ttl_seconds or self.config.default_ttl_seconds)
        key = self._cache_key(url, params)

        if use_cache:
            cached = self.cache.get(key)
            if isinstance(cached, dict) and cached.get("type") == "text":
                v = cached.get("value")
                if isinstance(v, str):
                    return v

        host = urlparse(url).netloc
        self.rate_limiter.acquire(host)

        h = dict(self.default_headers)
        if headers:
            h.update(headers)

        r = self._client.get(url, params=params, headers=h)
        # Handle rate limits: 429 → backoff and retry by raising TransportError-like.
        if r.status_code in (429, 500, 502, 503, 504):
            # raise to trigger retry
            raise httpx.TransportError(f"HTTP {r.status_code} from {host}")

        r.raise_for_status()
        text = r.text

        if use_cache:
            self.cache.set(key, {"type": "text", "value": text}, ttl)
        return text

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError)),
        reraise=True,
    )
    def get_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        ttl_seconds: Optional[int] = None,
        use_cache: bool = True,
    ) -> Any:
        ttl = int(ttl_seconds or self.config.default_ttl_seconds)
        key = self._cache_key(url, params)

        if use_cache:
            cached = self.cache.get(key)
            if isinstance(cached, dict) and cached.get("type") == "json":
                return cached.get("value")

        host = urlparse(url).netloc
        self.rate_limiter.acquire(host)

        h = dict(self.default_headers)
        if headers:
            h.update(headers)

        r = self._client.get(url, params=params, headers=h)
        if r.status_code in (429, 500, 502, 503, 504):
            raise httpx.TransportError(f"HTTP {r.status_code} from {host}")

        r.raise_for_status()
        data = r.json()

        if use_cache:
            self.cache.set(key, {"type": "json", "value": data}, ttl)
        return data


def default_policies(
    *,
    ncbi_api_key_present: bool,
) -> Dict[str, RateLimitPolicy]:
    """Reasonable defaults based on public guidance:

    - arXiv: 3 seconds between calls.
    - NCBI E-utilities: 3 req/s without key, higher with key. We enforce min interval.
    - Others: conservative min intervals to play nice.
    """

    # Convert RPS to min interval.
    def mi(rps: float) -> float:
        return 1.0 / rps if rps > 0 else 0.0

    ncbi_rps = 10.0 if ncbi_api_key_present else 3.0

    return {
        # NCBI
        "eutils.ncbi.nlm.nih.gov": RateLimitPolicy(min_interval_seconds=mi(ncbi_rps)),
        "www.ncbi.nlm.nih.gov": RateLimitPolicy(min_interval_seconds=mi(ncbi_rps)),
        "pmc.ncbi.nlm.nih.gov": RateLimitPolicy(min_interval_seconds=mi(ncbi_rps)),
        # Crossref
        "api.crossref.org": RateLimitPolicy(min_interval_seconds=mi(5.0)),
        # OpenAlex
        "api.openalex.org": RateLimitPolicy(min_interval_seconds=mi(10.0)),
        # Semantic Scholar
        "api.semanticscholar.org": RateLimitPolicy(min_interval_seconds=mi(5.0)),
        # Europe PMC
        "www.ebi.ac.uk": RateLimitPolicy(min_interval_seconds=mi(5.0)),
        # bioRxiv
        "api.biorxiv.org": RateLimitPolicy(min_interval_seconds=mi(5.0)),
        # arXiv (docs recommend 3 seconds between calls)
        "export.arxiv.org": RateLimitPolicy(min_interval_seconds=3.0),
    }
