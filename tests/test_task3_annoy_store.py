from __future__ import annotations

import json

from scireason.index.annoy_store import annoy_available, build_annoy_index, search_annoy_index


def test_annoy_store_builds_manifest_and_searches(tmp_path) -> None:
    bundle = build_annoy_index(
        vectors=[
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        item_ids=["chunk-a", "chunk-b", "chunk-c"],
        out_dir=tmp_path / "annoy",
        metric="angular",
        n_trees=8,
        item_payloads=[
            {"paper_id": "p1", "text": "alpha"},
            {"paper_id": "p1", "text": "alpha beta"},
            {"paper_id": "p2", "text": "gamma"},
        ],
    )

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert isinstance(annoy_available(), bool)
    assert bundle.manifest_path.exists()
    assert bundle.ids_path.exists()
    assert bundle.metadata_path.exists()
    assert bundle.vectors_path.exists()
    assert manifest["backend"] in {"annoy", "numpy_fallback"}
    assert manifest["size"] == 3
    assert manifest["dim"] == 3

    rows = search_annoy_index(bundle, [1.0, 0.0, 0.0], top_k=2)

    assert len(rows) == 2
    assert rows[0]["item_id"] == "chunk-a"
    assert rows[0]["paper_id"] == "p1"
    assert any(key in rows[0] for key in {"score", "distance"})
