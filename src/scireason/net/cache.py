from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class CacheItem:
    value: Any
    expires_at: float


class DiskJSONCache:
    """A tiny disk-backed JSON cache with TTL.

    - Keys are file-safe hex digests.
    - Values are stored as JSON.
    - Best-effort; cache failures never break the caller.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        try:
            if not p.exists():
                return None
            raw = json.loads(p.read_text(encoding="utf-8"))
            expires_at = float(raw.get("expires_at", 0))
            if expires_at and expires_at < time.time():
                # expired
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
                return None
            return raw.get("value")
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            expires_at = time.time() + max(int(ttl_seconds), 1)
            payload = {"expires_at": expires_at, "value": value}
            tmp = self._path(key + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path(key))
        except Exception:
            # best effort
            return
