from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("qdrant_client") is None:
    pytest.skip("qdrant-client is not installed (optional dependency)", allow_module_level=True)

from scireason.config import settings
from scireason.graph import qdrant_store


def test_remote_qdrant_client_disables_compatibility_check(monkeypatch) -> None:
    captured = {}

    class _Client:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_store, "QdrantClient", _Client)
    monkeypatch.setattr(settings, "qdrant_check_compatibility", False)

    store = qdrant_store.QdrantStore(url="http://localhost:6333")
    assert captured.get("check_compatibility") is False
    assert store._client is not None
