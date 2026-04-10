from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scireason.task3_offline_review import (
    build_task3_expert_artifact_bundle,
    build_task3_offline_review_package,
)


def test_task3_offline_review_and_expert_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "automatic_graph").mkdir(parents=True)
    (bundle_dir / "processed_papers" / "paper-1" / "mm").mkdir(parents=True)

    hypotheses = [
        {
            "rank": 1,
            "candidate": {
                "source": "Catalyst A",
                "predicate": "improves",
                "target": "forecast stability",
                "score": 0.73,
                "time_scope": "2023-2024",
                "graph_signals": {"support": 0.8, "temporal_support": 0.6},
            },
            "temporal_context": {"ordering": "strengthening", "years": [2023, 2024], "time_scope": "2023-2024"},
            "prediction_support": {"score": 0.55, "backend": "heuristic"},
            "multimodal_support": [{"chunk_id": "paper-1:c1", "snippet": "Figure shows improved stability", "page": 2}],
            "hypothesis": {
                "title": "Catalyst A strengthens forecast stability over time",
                "premise": "Observed in two time windows.",
                "mechanism": "Catalyst A reduces variance in later observations.",
                "time_scope": "2023-2024",
                "proposed_experiment": "Replicate on held-out monitoring data.",
                "supporting_evidence": [
                    {"source_id": "paper-1", "text_snippet": "2024 follow-up reports improved stability.", "page": 4}
                ],
            },
            "final_score": 0.81,
        }
    ]
    (bundle_dir / "hypotheses_ranked.json").write_text(json.dumps(hypotheses, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle_dir / "hypotheses_ranked.md").write_text("# demo", encoding="utf-8")
    (bundle_dir / "hypotheses_candidates.json").write_text("[]", encoding="utf-8")
    (bundle_dir / "query.json").write_text(json.dumps({"effective_query": "catalyst"}), encoding="utf-8")
    (bundle_dir / "papers_selected.json").write_text("[]", encoding="utf-8")
    (bundle_dir / "paper_records.json").write_text("[]", encoding="utf-8")
    (bundle_dir / "chunk_registry.jsonl").write_text("", encoding="utf-8")
    (bundle_dir / "automatic_graph" / "temporal_kg.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "automatic_graph" / "events.jsonl").write_text("", encoding="utf-8")
    (bundle_dir / "automatic_graph" / "multimodal_triplets.jsonl").write_text("", encoding="utf-8")
    (bundle_dir / "automatic_graph" / "link_predictions.json").write_text(json.dumps({"predictions": []}), encoding="utf-8")
    (bundle_dir / "processed_papers" / "paper-1" / "meta.json").write_text(json.dumps({"id": "paper-1"}), encoding="utf-8")
    (bundle_dir / "processed_papers" / "paper-1" / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (bundle_dir / "processed_papers" / "paper-1" / "mm" / "pages.jsonl").write_text("{}\n", encoding="utf-8")

    manifest = {
        "bundle_dir": str(bundle_dir),
        "query": "catalyst forecasting",
        "domain_id": "science",
        "artifacts": {
            "hypotheses_ranked": str(bundle_dir / "hypotheses_ranked.json"),
        },
    }
    manifest_path = bundle_dir / "task3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    task_meta = {
        "topic": "Catalyst task",
        "submission_id": "expert__abc123",
        "expert": {"latin_slug": "expert_slug"},
    }

    html_path = build_task3_offline_review_package(manifest, task_meta)
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Task 3" in html_text
    assert "A/B" in html_text
    assert "Catalyst A" in html_text

    zip_path = build_task3_expert_artifact_bundle(manifest_path, task_meta)
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    assert "hypotheses_ranked.json" in names
    assert "expert_review/offline_review/task3_hypothesis_review_offline_ab.html" in names
    assert "expert_review/task3_expert_review_manifest.json" in names
