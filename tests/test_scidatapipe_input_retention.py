from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from scireason.scidatapipe_bridge.builder import export_dataset


def _write_task1_yaml(path: Path, *, submission_id: str, claim: str) -> None:
    doc = {
        "artifact_version": 4,
        "topic": "Collision test",
        "domain": "science",
        "submission_id": submission_id,
        "expert": {"latin_slug": "same_expert"},
        "papers": [{"id": "doi:10.1234/example", "year": 2020, "title": "Example paper"}],
        "steps": [
            {
                "step_id": 1,
                "claim": claim,
                "inference": f"inference for {claim}",
                "next_question": "",
                "sources": [
                    {
                        "type": "text",
                        "source": "doi:10.1234/example",
                        "locator": "p. 1",
                        "snippet_or_summary": "supporting evidence",
                    }
                ],
            }
        ],
        "edges": [],
    }
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_task2_bundle(path: Path, *, submission_id: str, assertion_id: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "submission_id": submission_id,
        "domain": "science",
        "topic": "Collision test",
        "reviewer_id": "reviewer",
        "assertions": [
            {
                "assertion_id": assertion_id,
                "graph_kind": "gold",
                "subject": "subject",
                "predicate": "predicate",
                "object": assertion_id,
                "start_date": "2020",
                "end_date": "2020",
                "evidence": {"text": "evidence", "paper_id": "doi:10.1234/example"},
                "paper_ids": ["doi:10.1234/example"],
                "expert": {"verdict": "accepted", "rationale": "ok"},
            }
        ],
    }
    (path / "edge_reviews.json").write_text(json.dumps(payload), encoding="utf-8")


def test_export_keeps_task1_files_with_same_submission_id(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_task1_yaml(input_dir / "trajectory.yaml", submission_id="same_submission", claim="claim A")
    _write_task1_yaml(input_dir / "trajectory_copy.yaml", submission_id="same_submission", claim="claim B")

    result = export_dataset(input_dirs=[input_dir], out_dir=tmp_path / "export")

    assert result.stats["discovered_task1_files"] == 2
    assert result.stats["normalized_task1_submissions"] == 2
    assert result.stats["trajectory_reasoning"] == 2
    rows = [json.loads(line) for line in result.sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len([row for row in rows if row["task_family"] == "trajectory_reasoning"]) == 2
    assert len({row["metadata"]["submission_id"] for row in rows}) == 2


def test_export_processes_all_task2_bundles_inside_one_zip(tmp_path: Path) -> None:
    zip_source = tmp_path / "zip_source"
    _write_task2_bundle(zip_source / "bundle_a", submission_id="same_bundle", assertion_id="a")
    _write_task2_bundle(zip_source / "bundle_b", submission_id="same_bundle", assertion_id="b")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    archive = input_dir / "task2_upload.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(zip_source.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(zip_source))

    result = export_dataset(input_dirs=[input_dir], out_dir=tmp_path / "export")

    assert result.stats["discovered_task2_inputs"] == 1
    assert result.stats["normalized_task2_bundles"] == 2
    assert result.stats["assertion_reconstruction"] == 2
    rows = [json.loads(line) for line in result.sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len([row for row in rows if row["task_family"] == "assertion_reconstruction"]) == 2
    assert len({row["metadata"]["submission_id"] for row in rows}) == 2


def test_export_adds_author_suffix_from_task1_filename(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_task1_yaml(
        input_dir / "reasoning_failures_in_large_language_models_Илья_Фёдоров.yaml",
        submission_id="reasoning_failures_in_large_language_models",
        claim="claim by Ilya",
    )

    result = export_dataset(input_dirs=[input_dir], out_dir=tmp_path / "export")

    expected_id = "reasoning_failures_in_large_language_models_Илья_Фёдоров"
    assert (result.normalized_task1_dir / expected_id / f"{expected_id}.yaml").exists()
    rows = [json.loads(line) for line in result.sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trajectory_rows = [row for row in rows if row["task_family"] == "trajectory_reasoning"]
    assert {row["metadata"]["submission_id"] for row in trajectory_rows} == {expected_id}
    assert {row["metadata"].get("original_submission_id") for row in trajectory_rows} == {"reasoning_failures_in_large_language_models"}


def test_export_adds_author_suffix_from_task2_zip_filename(tmp_path: Path) -> None:
    zip_source = tmp_path / "zip_source"
    _write_task2_bundle(
        zip_source / "reasoning_failures_in_large_language_models",
        submission_id="reasoning_failures_in_large_language_models",
        assertion_id="a",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    archive = input_dir / "expert_validation_bundle - Илья Фёдоров.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(zip_source.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(zip_source))

    result = export_dataset(input_dirs=[input_dir], out_dir=tmp_path / "export")

    expected_id = "reasoning_failures_in_large_language_models_Илья_Фёдоров"
    assert (result.normalized_task2_dir / expected_id / "gold.json").exists()
    rows = [json.loads(line) for line in result.sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assertion_rows = [row for row in rows if row["task_family"] == "assertion_reconstruction"]
    assert {row["metadata"]["submission_id"] for row in assertion_rows} == {expected_id}
