import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_full_data_audit_accepts_extra_dpo_hard_negative_pairs(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "summary.json").write_text(json.dumps({
        "raw_sft_rows_total": 2,
        "raw_grpo_rows_total": 1,
        "max_sft_samples": 0,
        "max_grpo_samples": 0,
        "counts": {"dpo_from_sft": 2, "dpo_from_grpo": 1},
    }), encoding="utf-8")
    (data / "image_resolution_report.json").write_text(json.dumps({
        "sft": {"image_refs_before_selection": 0, "image_refs_after_selection": 0, "rows_with_truncated_images": 0, "max_images_per_example": 0},
        "grpo": {"image_refs_before_selection": 0, "image_refs_after_selection": 0, "rows_with_truncated_images": 0, "max_images_per_example": 0},
    }), encoding="utf-8")
    (data / "export_summary.json").write_text(json.dumps({
        "trajectory_reasoning": 1,
        "assertion_reconstruction": 1,
        "assertion_review_rl": 1,
    }), encoding="utf-8")
    _write_jsonl(data / "sft_all.jsonl", [{"id": "sft-a"}, {"id": "sft-b"}])
    _write_jsonl(data / "dpo_all.jsonl", [
        {"id": "dpo:sft-a:0", "metadata": {"source_id": "sft-a"}},
        {"id": "dpo:sft-b:0", "metadata": {"source_id": "sft-b"}},
        {"id": "dpo-grpo:rl-a", "metadata": {"source_id": "rl-a"}},
    ])
    _write_jsonl(data / "grpo_train_all.jsonl", [{"id": "rl-a"}])
    _write_jsonl(data / "grpo_eval_all.jsonl", [])
    _write_jsonl(data / "grpo_all_verified.jsonl", [{"id": "rl-a"}])

    result = subprocess.run(
        [sys.executable, "experiments/vlm_finetuning/scripts/audit_full_data_usage.py", "--data-dir", str(data), "--strict", "--require-all-images"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "pass"
    check = next(c for c in report["checks"] if c["name"] == "dpo_all_covers_sft_sources")
    assert check["observed"]["dpo_all_pairs"] == 3
    assert check["observed"]["sft_sources_covered"] == 2


def test_full_data_audit_counts_merged_dpo_source_ids_from_deduped_pairs(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "summary.json").write_text(json.dumps({
        "raw_sft_rows_total": 3,
        "raw_grpo_rows_total": 0,
        "max_sft_samples": 0,
        "max_grpo_samples": 0,
        "counts": {"dpo_from_sft": 3, "dpo_from_grpo": 0},
    }), encoding="utf-8")
    (data / "image_resolution_report.json").write_text(json.dumps({
        "sft": {"image_refs_before_selection": 0, "image_refs_after_selection": 0, "rows_with_truncated_images": 0, "max_images_per_example": 0},
        "grpo": {"image_refs_before_selection": 0, "image_refs_after_selection": 0, "rows_with_truncated_images": 0, "max_images_per_example": 0},
    }), encoding="utf-8")
    (data / "export_summary.json").write_text(json.dumps({
        "trajectory_reasoning": 1,
        "assertion_reconstruction": 2,
        "assertion_review_rl": 0,
    }), encoding="utf-8")
    _write_jsonl(data / "sft_all.jsonl", [{"id": "sft-a"}, {"id": "sft-b"}, {"id": "sft-c"}])
    _write_jsonl(data / "dpo_all.jsonl", [
        {"id": "dpo:deduped:0", "metadata": {"source_id": "sft-a", "source_ids": ["sft-a", "sft-b"]}},
        {"id": "dpo:sft-c:0", "metadata": {"source_id": "sft-c"}},
    ])
    _write_jsonl(data / "grpo_train_all.jsonl", [])
    _write_jsonl(data / "grpo_eval_all.jsonl", [])
    _write_jsonl(data / "grpo_all_verified.jsonl", [])

    result = subprocess.run(
        [sys.executable, "experiments/vlm_finetuning/scripts/audit_full_data_usage.py", "--data-dir", str(data), "--strict", "--require-all-images"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    check = next(c for c in report["checks"] if c["name"] == "dpo_all_covers_sft_sources")
    assert check["observed"]["sft_sources_covered"] == 3
