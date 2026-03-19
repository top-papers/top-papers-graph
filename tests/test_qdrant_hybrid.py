from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec('qdrant_client') is None:
    pytest.skip('qdrant-client is not installed (optional dependency)', allow_module_level=True)

from scireason.graph.qdrant_store import QdrantStore


def test_qdrant_hybrid_search_works_in_memory() -> None:
    store = QdrantStore(url=':memory:')
    store.ensure_hybrid_collection('hybrid', dense_vector_size=4)
    store.upsert_hybrid(
        'hybrid',
        ids=['a', 'b'],
        dense_vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        sparse_vectors=[
            store.hashed_sparse_vector('alpha beta'),
            store.hashed_sparse_vector('beta gamma'),
        ],
        payloads=[
            {'text': 'alpha beta'},
            {'text': 'beta gamma'},
        ],
    )
    res = store.hybrid_search(
        'hybrid',
        dense_vector=[1.0, 0.0, 0.0, 0.0],
        sparse_vector=store.hashed_sparse_vector('alpha'),
        limit=2,
    )
    assert res
    assert res[0]['payload']['text'] == 'alpha beta'
