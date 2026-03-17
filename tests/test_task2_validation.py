from __future__ import annotations

import json
from pathlib import Path

import yaml

from scireason.task2_validation import build_gold_graph_from_trajectory, build_task2_validation_bundle, load_task1_yaml


SAMPLE = {
    "artifact_version": 2,
    "topic": "Demo temporal graph topic",
    "domain": "science",
    "papers": [
        {"id": "doi:10.1000/demo", "year": 2024, "title": "Demo paper"},
    ],
    "steps": [
        {
            "step_id": 1,
            "claim": "A affects B",
            "sources": [
                {
                    "type": "text",
                    "source": "doi:10.1000/demo",
                    "snippet_or_summary": "A affects B under condition C.",
                    "locator": "p.1",
                }
            ],
            "conditions": {"system": "demo"},
            "inference": "There is a relation.",
            "next_question": "What mediates the effect?",
        },
        {
            "step_id": 2,
            "claim": "Mediator M links A and B",
            "sources": [],
            "conditions": {"system": "demo"},
            "inference": "Mediator hypothesis.",
            "next_question": "-",
        },
    ],
    "edges": [[1, 2]],
    "submission_id": "demo_submission",
}


def test_build_gold_graph_from_trajectory(tmp_path) -> None:
    out_dir = tmp_path / "gold"
    payload = build_gold_graph_from_trajectory(SAMPLE, out_dir)
    assert (out_dir / "gold_graph.json").exists()
    assert (out_dir / "gold_triplets.csv").exists()
    assert payload["graph_type"] == "gold_manual_trajectory"
    assert len(payload["triplets"]) >= 2
    assert any(t["predicate"] == "supports_claim" for t in payload["triplets"])
    assert any(t["predicate"] == "leads_to" for t in payload["triplets"])


def test_build_task2_validation_bundle_without_auto(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.yaml"
    trajectory.write_text(yaml.safe_dump(SAMPLE, allow_unicode=True, sort_keys=False), encoding="utf-8")
    bundle = build_task2_validation_bundle(
        trajectory,
        out_dir=tmp_path / "runs",
        include_auto_pipeline=False,
        enable_reference_scout=False,
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert Path(manifest["gold_graph"]).exists()
    assert Path(manifest["gold_triplets_csv"]).exists()
    assert manifest["auto_run_dir"] is None
    loaded = load_task1_yaml(trajectory)
    assert loaded["topic"] == SAMPLE["topic"]
