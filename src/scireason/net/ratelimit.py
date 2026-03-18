from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RateLimitPolicy:
    """Simple per-host policy based on minimum interval between requests."""
    min_interval_seconds: float


class HostRateLimiter:
    """Thread-safe, process-local per-host rate limiter.

    This is intentionally simple. For production multi-process deployments
    you'd typically move to a shared store (Redis) or a gateway-level limiter.
    """

    def __init__(self, policies: Dict[str, RateLimitPolicy]) -> None:
        self._policies = dict(policies)
        self._lock = threading.Lock()
        self._last_call: Dict[str, float] = {}

    def acquire(self, host: str) -> None:
        policy = self._policies.get(host)
        if policy is None:
            return

        with self._lock:
            now = time.monotonic()
            last = self._last_call.get(host, 0.0)
            wait_for = policy.min_interval_seconds - (now - last)
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._last_call[host] = now
