from __future__ import annotations

from scireason.agentic.code_agent import run_in_sandbox
from scireason.agents import graph_candidate_agent as gca
from scireason.temporal.temporal_kg_builder import EdgeStats, NodeStats, PaperRecord, TemporalKnowledgeGraph


def _mini_kg() -> TemporalKnowledgeGraph:
    kg = TemporalKnowledgeGraph(
        nodes={
            "alpha": NodeStats(term="alpha", doc_freq=2),
            "beta": NodeStats(term="beta", doc_freq=2),
            "gamma": NodeStats(term="gamma", doc_freq=2),
            "delta": NodeStats(term="delta", doc_freq=2),
        },
        edges=[
            EdgeStats(source="alpha", target="beta", predicate="related_to", total_count=2, score=0.9),
            EdgeStats(source="beta", target="gamma", predicate="related_to", total_count=2, score=0.8),
            EdgeStats(source="gamma", target="delta", predicate="related_to", total_count=2, score=0.85),
        ],
        meta={"years": [2021, 2022, 2023]},
    )
    return kg


def _mini_papers() -> list[PaperRecord]:
    return [
        PaperRecord(
            paper_id="p1",
            title="alpha beta gamma",
            year=2023,
            text="Alpha interacts with beta and gamma in a shared mechanism.",
        ),
        PaperRecord(
            paper_id="p2",
            title="gamma delta",
            year=2022,
            text="Gamma is frequently discussed together with delta in experiments.",
        ),
    ]


def test_run_in_sandbox_normalizes_fullwidth_punctuation() -> None:
    env = run_in_sandbox(
        "```python\nfinal_answer = [1， 2， 3]\n```",
        tools={},
    )
    assert env["final_answer"] == [1, 2, 3]


def test_run_in_sandbox_wraps_tool_lists_with_safe_get() -> None:
    env = run_in_sandbox(
        "payload = link_prediction()\nfinal_answer = payload.get('predictions', [])",
        tools={"link_prediction": lambda: [["alpha", "beta", 0.7]]},
    )
    assert env["final_answer"] == [["alpha", "beta", 0.7]]


def test_agent_generate_candidates_uses_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(gca.settings, "hyp_agent_enabled", True)
    monkeypatch.setattr(gca.settings, "hyp_agent_backend", "internal")
    monkeypatch.setattr(gca.CodeAgent, "run", lambda self, task, context=None: (_ for _ in ()).throw(RuntimeError("boom")))

    candidates = gca.agent_generate_candidates(
        _mini_kg(),
        papers=_mini_papers(),
        query="test query",
        domain="science",
        top_k=4,
    )

    assert candidates, "expected deterministic fallback candidates"
    assert all(c.source and c.target for c in candidates)
    assert all(c.predicate == "may_relate_to" for c in candidates)
