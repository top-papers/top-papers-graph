from __future__ import annotations

import pytest

import networkx as nx

from scireason.config import settings
from scireason.gnn.pyg_link_prediction import PyGUnavailableError, pyg_available, pyg_link_prediction
from scireason.hypotheses.temporal_graph_hypotheses import generate_candidates
from scireason.temporal.temporal_kg_builder import PaperRecord, build_temporal_kg
from scireason.domain import load_domain_config


def test_pyg_available_returns_bool() -> None:
    assert isinstance(pyg_available(), bool)


def test_pyg_link_prediction_raises_when_missing() -> None:
    if pyg_available():
        pytest.skip("PyG is installed in this environment")

    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")

    with pytest.raises(PyGUnavailableError):
        _ = pyg_link_prediction(G, top_k=3)


def test_pipeline_candidates_do_not_crash_when_gnn_enabled_without_pyg() -> None:
    """Even if HYP_GNN_ENABLED=1, base installation should not crash.

    The pipeline must fall back to heuristic link prediction when PyG is not installed.
    """

    prev = settings.hyp_gnn_enabled
    settings.hyp_gnn_enabled = True

    domain = load_domain_config("science")
    papers = [
        PaperRecord(
            paper_id="p1",
            title="A",
            year=2022,
            text="Graph neural networks learn node embeddings. Link prediction on graphs.",
        ),
        PaperRecord(
            paper_id="p2",
            title="B",
            year=2023,
            text="Temporal knowledge graphs can support hypothesis generation.",
        ),
    ]

    kg = build_temporal_kg(papers, domain=domain, query="demo", edge_mode="cooccurrence")

    # Should not raise, regardless of PyG availability.
    try:
        cands = generate_candidates(kg, papers=papers, query="demo", domain=domain.title, top_k=5)
        assert isinstance(cands, list)
        assert cands  # heuristic candidates should exist
    finally:
        settings.hyp_gnn_enabled = prev
