from __future__ import annotations

from scireason.config import settings
from scireason.domain import load_domain_config
from scireason.hypotheses.temporal_graph_hypotheses import generate_candidates
from scireason.temporal.schemas import TemporalEvent
from scireason.temporal.temporal_kg_builder import PaperRecord, build_temporal_kg
from scireason.tgnn.event_dataset import build_event_stream, chronological_split
from scireason.tgnn.tgn_link_prediction import tgn_link_prediction, tgnn_available


def test_temporal_event_stable_id_is_deterministic() -> None:
    event = TemporalEvent(
        paper_id="p1",
        chunk_id="c1",
        subject="graph neural networks",
        predicate="improves",
        object="link prediction",
        ts_start="2024",
        ts_end="2024",
    )
    assert event.stable_id() == event.stable_id()
    assert len(event.stable_id()) == 16


def test_build_event_stream_and_split() -> None:
    domain = load_domain_config("science")
    papers = [
        PaperRecord(paper_id="p1", title="A", year=2022, text="Graph neural networks improve link prediction in chemistry."),
        PaperRecord(paper_id="p2", title="B", year=2023, text="Temporal graph methods improve hypothesis generation in biology."),
        PaperRecord(paper_id="p3", title="C", year=2024, text="Graph neural networks and temporal graph methods both help prediction."),
    ]
    kg = build_temporal_kg(papers, domain=domain, query="demo", edge_mode="cooccurrence")
    events = build_event_stream(kg, papers=papers)
    train, valid, test = chronological_split(events)
    assert events
    assert len(train) >= 1
    assert len(train) + len(valid) + len(test) == len(events)


def test_tgn_link_prediction_returns_semantic_records() -> None:
    events = [
        TemporalEvent(paper_id="p1", subject="a", predicate="rel", object="b", ts_start="2022", ts_end="2022"),
        TemporalEvent(paper_id="p2", subject="b", predicate="rel", object="c", ts_start="2023", ts_end="2023"),
        TemporalEvent(paper_id="p3", subject="a", predicate="rel", object="d", ts_start="2024", ts_end="2024"),
        TemporalEvent(paper_id="p4", subject="d", predicate="rel", object="c", ts_start="2024", ts_end="2024"),
    ]
    preds = tgn_link_prediction(events, top_k=5)
    assert isinstance(tgnn_available(), bool)
    assert isinstance(preds, list)
    assert preds
    assert all(getattr(item, "predicate", None) for item in preds)
    assert all(getattr(item, "source", None) and getattr(item, "target", None) for item in preds)


def test_generate_candidates_prefers_tgnn_when_enabled() -> None:
    prev_tgnn = settings.hyp_tgnn_enabled
    prev_gnn = settings.hyp_gnn_enabled
    settings.hyp_tgnn_enabled = True
    settings.hyp_gnn_enabled = False
    try:
        domain = load_domain_config("science")
        papers = [
            PaperRecord(paper_id="p1", title="A", year=2022, text="Graph neural networks improve link prediction in chemistry."),
            PaperRecord(paper_id="p2", title="B", year=2023, text="Temporal graph methods improve hypothesis generation in biology."),
            PaperRecord(paper_id="p3", title="C", year=2024, text="Graph neural networks and temporal graph methods both help prediction."),
        ]
        kg = build_temporal_kg(papers, domain=domain, query="demo", edge_mode="cooccurrence")
        cands = generate_candidates(kg, papers=papers, query="demo", domain=domain.title, top_k=10)
        assert cands
        assert any(c.kind == "tgnn_missing_link" for c in cands)
    finally:
        settings.hyp_tgnn_enabled = prev_tgnn
        settings.hyp_gnn_enabled = prev_gnn
