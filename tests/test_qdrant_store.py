from __future__ import annotations

import importlib.util

import pytest


if importlib.util.find_spec("qdrant_client") is None:
    pytest.skip("qdrant-client is not installed (optional dependency)", allow_module_level=True)

from scireason.graph.qdrant_store import QdrantStore


def test_qdrant_store_search_works_with_new_client_api() -> None:
    """qdrant-client>=1.16 removed QdrantClient.search in favor of query_points.

    Our wrapper must work with the currently installed client.
    """
    store = QdrantStore(url=":memory:")
    store.ensure_collection("test", vector_size=3)

    store.upsert(
        collection="test",
        ids=["a"],
        vectors=[[0.1, 0.2, 0.3]],
        payloads=[{"text": "hello"}],
    )

    res = store.search(collection="test", query_vector=[0.1, 0.2, 0.3], limit=1)
    assert res
    assert res[0]["id"] is not None
    assert res[0]["payload"]["text"] == "hello"
