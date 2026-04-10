from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scireason.task3_dual_model_review import (
    build_task3_dual_model_expert_bundle,
    build_task3_dual_model_offline_review_package,
)


def _write_bundle(bundle_dir: Path, hypotheses: list[dict]) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "hypotheses_ranked.json").write_text(json.dumps(hypotheses, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle_dir / "hypotheses_ranked.md").write_text("# demo", encoding="utf-8")
    manifest = {
        "bundle_dir": str(bundle_dir),
        "query": "shared topic",
        "runtime": {"vlm_backend": "qwen2_vl", "vlm_model_id": "local/model"},
        "artifacts": {
            "hypotheses_ranked": str(bundle_dir / "hypotheses_ranked.json"),
            "hypotheses_markdown": str(bundle_dir / "hypotheses_ranked.md"),
        },
    }
    path = bundle_dir / "task3_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_task3_dual_model_review_assets_and_bundle(tmp_path: Path) -> None:
    hypo_a = [
        {
            "rank": 1,
            "candidate": {"kind": "missing_link", "source": "Catalyst A", "predicate": "improves", "target": "stability", "score": 0.8, "time_scope": "2023-2024"},
            "temporal_context": {"ordering": "strengthening", "years": [2023, 2024], "time_scope": "2023-2024"},
            "hypothesis": {
                "title": "Catalyst A may improve stability over time",
                "premise": "Two time windows support the relation.",
                "mechanism": "Progressive reduction of variance.",
                "time_scope": "2023-2024",
                "proposed_experiment": "Replicate in held-out monitoring data.",
                "supporting_evidence": [{"source_id": "paper-a", "text_snippet": "Follow-up reports stronger stability in 2024."}],
            },
            "final_score": 0.91,
        }
    ]
    hypo_b = [
        {
            "rank": 1,
            "candidate": {"kind": "missing_link", "source": "Catalyst A", "predicate": "improves", "target": "stability", "score": 0.75, "time_scope": "2023-2024"},
            "temporal_context": {"ordering": "strengthening", "years": [2023, 2024], "time_scope": "2023-2024"},
            "hypothesis": {
                "title": "Catalyst A supports later-stage stability",
                "premise": "Signals align across two intervals.",
                "mechanism": "Delayed stabilization after intervention.",
                "time_scope": "2023-2024",
                "proposed_experiment": "Run a later-window replication study.",
                "supporting_evidence": [{"source_id": "paper-b", "text_snippet": "Later-stage measurements remain more stable."}],
            },
            "final_score": 0.88,
        }
    ]

    manifest_a = _write_bundle(tmp_path / "run_alpha", hypo_a)
    manifest_b = _write_bundle(tmp_path / "run_beta", hypo_b)

    assets = build_task3_dual_model_offline_review_package(
        manifest_a,
        manifest_b,
        task_meta={"topic": "Catalyst topic", "submission_id": "expert__dual"},
        model_a_descriptor={"vlm_model_id": "/models/base-local-vlm"},
        model_b_descriptor={"vlm_model_id": "/models/finetuned-local-vlm"},
    )
    assert assets.offline_html_path.exists()
    assert assets.owner_mapping_path.exists()
    assert assets.public_manifest_path.exists()

    html_text = assets.offline_html_path.read_text(encoding="utf-8")
    assert "Blind A/B review" in html_text
    assert "Hidden model α" in html_text
    assert "/models/base-local-vlm" not in html_text
    assert "/models/finetuned-local-vlm" not in html_text

    owner_text = assets.owner_mapping_path.read_text(encoding="utf-8")
    assert "/models/base-local-vlm" in owner_text
    assert "/models/finetuned-local-vlm" in owner_text

    zip_path = build_task3_dual_model_expert_bundle(
        manifest_a,
        manifest_b,
        task_meta={"topic": "Catalyst topic", "submission_id": "expert__dual"},
        model_a_descriptor={"vlm_model_id": "/models/base-local-vlm"},
        model_b_descriptor={"vlm_model_id": "/models/finetuned-local-vlm"},
    )
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "offline_review/task3_dual_local_model_review_offline_ab.html" in names
        assert "variant_alpha/hypotheses_ranked.json" in names
        assert "variant_beta/hypotheses_ranked.json" in names
        assert "expert_review/task3_dual_local_model_review_manifest.json" in names
        assert "owner_only/task3_dual_local_model_blind_key.json" not in names
