from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


def _load_build_script():
    path = Path("experiments/vlm_finetuning/scripts/build_vlm_sft_dataset.py")
    spec = importlib.util.spec_from_file_location("build_vlm_sft_dataset_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_vlm_sft_dataset_test"] = module
    spec.loader.exec_module(module)
    return module


def _write_generic_task1(path: Path, *, claim: str) -> None:
    payload = {
        "artifact_version": 3,
        "topic": "Input retention",
        "domain": "science",
        "submission_id": "trajectory_submission",
        "expert": {"latin_slug": "trajectory_submission"},
        "papers": [{"id": "doi:10.1234/example", "year": 2024, "title": "Example"}],
        "steps": [
            {
                "step_id": 1,
                "claim": claim,
                "inference": f"inference for {claim}",
                "next_question": "next",
                "sources": [{"type": "text", "source": "doi:10.1234/example", "snippet_or_summary": "evidence"}],
            }
        ],
        "edges": [],
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_build_vlm_sft_keeps_files_with_generic_submission_id(tmp_path: Path) -> None:
    mod = _load_build_script()
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    _write_generic_task1(first, claim="claim A")
    _write_generic_task1(second, claim="claim B")

    rows = [row for _, row in mod.trajectory_records(first)] + [row for _, row in mod.trajectory_records(second)]

    assert len(rows) == 2
    assert len({row["id"] for row in rows}) == 2
    assert len({row["metadata"]["submission_id"] for row in rows}) == 2
    assert {row["metadata"]["original_submission_id"] for row in rows} == {"trajectory_submission"}
    assert {row["metadata"]["input_file"] for row in rows} == {str(first), str(second)}


def test_build_vlm_sft_summary_counts_unique_record_ids(tmp_path: Path) -> None:
    mod = _load_build_script()
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_generic_task1(input_dir / "first.yaml", claim="claim A")
    _write_generic_task1(input_dir / "second.yaml", claim="claim B")

    rows = []
    for path in mod.iter_paths([input_dir], ("**/*.yaml", "**/*.yml")):
        rows.extend(mod.trajectory_records(path))
    mod.assert_unique_record_ids(rows)

    output = tmp_path / "all.jsonl"
    mod.write_jsonl(output, [row for _, row in rows])
    parsed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(parsed) == 2
    assert len({row["id"] for row in parsed}) == 2
