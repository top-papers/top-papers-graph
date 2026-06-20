#!/usr/bin/env python3
"""Audit that the SciReason builder did not silently drop available data.

The training export contains instruction/alignment JSONL files plus assets.  The
normal training pipeline may intentionally create train/eval splits and may
filter GRPO to reward-ready examples, but this audit checks the input-preserving
parts of the pipeline:

* no max-sample subsampling was applied;
* sft_all/dpo_all/grpo_train_all+grpo_eval_all preserve the raw export rows;
* image references were not capped when full-image mode is requested;
* optional export_summary counts agree with the raw rows that were loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(data)!r}")
    return data


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return -1
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def add_check(checks: List[Dict[str, Any]], name: str, ok: bool, observed: Any, expected: Any, severity: str = "error", note: str = "") -> None:
    checks.append({
        "name": name,
        "ok": bool(ok),
        "severity": severity,
        "observed": observed,
        "expected": expected,
        "note": note,
    })


def nested(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_expected_from_export_summary(export_summary: Mapping[str, Any]) -> Dict[str, int]:
    sft_total = 0
    for key in ("trajectory_reasoning", "assertion_reconstruction"):
        value = export_summary.get(key)
        if isinstance(value, int):
            sft_total += value
    grpo_total = export_summary.get("assertion_review_rl")
    return {
        "sft": sft_total if sft_total > 0 else -1,
        "grpo": grpo_total if isinstance(grpo_total, int) else -1,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit full-data usage for SciReason alignment datasets.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--require-all-images", action="store_true", help="Fail if image refs were capped/truncated.")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    summary = read_json(data_dir / "summary.json")
    image_report = read_json(data_dir / "image_resolution_report.json")
    export_summary = read_json(data_dir / "export_summary.json")
    expected = infer_expected_from_export_summary(export_summary)

    counts = {
        "sft_all": count_jsonl(data_dir / "sft_all.jsonl"),
        "dpo_all": count_jsonl(data_dir / "dpo_all.jsonl"),
        "grpo_train_all": count_jsonl(data_dir / "grpo_train_all.jsonl"),
        "grpo_eval_all": count_jsonl(data_dir / "grpo_eval_all.jsonl"),
        "grpo_all_verified": count_jsonl(data_dir / "grpo_all_verified.jsonl"),
    }
    counts["grpo_all_split_sum"] = max(counts["grpo_train_all"], 0) + max(counts["grpo_eval_all"], 0)

    raw_sft = summary.get("raw_sft_rows_total")
    raw_grpo = summary.get("raw_grpo_rows_total")
    max_sft_samples = int(summary.get("max_sft_samples") or 0)
    max_grpo_samples = int(summary.get("max_grpo_samples") or 0)

    checks: List[Dict[str, Any]] = []
    add_check(checks, "no_sft_row_subsampling", max_sft_samples == 0, max_sft_samples, 0)
    add_check(checks, "no_grpo_row_subsampling", max_grpo_samples == 0, max_grpo_samples, 0)
    add_check(checks, "sft_all_preserves_raw_sft_rows", counts["sft_all"] == raw_sft, counts["sft_all"], raw_sft)
    add_check(checks, "dpo_all_covers_sft_rows", counts["dpo_all"] == raw_sft, counts["dpo_all"], raw_sft, note="Synthetic negatives should create one DPO pair per SFT row.")
    add_check(checks, "grpo_all_split_preserves_raw_grpo_rows", counts["grpo_all_split_sum"] == raw_grpo, counts["grpo_all_split_sum"], raw_grpo)

    if expected["sft"] > 0:
        add_check(checks, "raw_sft_matches_export_summary", raw_sft == expected["sft"], raw_sft, expected["sft"])
    if expected["grpo"] > 0:
        add_check(checks, "raw_grpo_matches_export_summary", raw_grpo == expected["grpo"], raw_grpo, expected["grpo"])

    for stage in ("sft", "grpo"):
        stage_report = image_report.get(stage, {}) if isinstance(image_report, Mapping) else {}
        before = nested(stage_report, "image_refs_before_selection", default=None)
        after = nested(stage_report, "image_refs_after_selection", default=None)
        truncated = nested(stage_report, "rows_with_truncated_images", default=None)
        max_images = nested(stage_report, "max_images_per_example", default=None)
        if args.require_all_images:
            add_check(checks, f"{stage}_all_image_refs_preserved", before == after, after, before)
            add_check(checks, f"{stage}_no_truncated_image_rows", truncated == 0, truncated, 0)
            add_check(checks, f"{stage}_image_cap_disabled", int(max_images or 0) <= 0, max_images, "<= 0")
        else:
            add_check(checks, f"{stage}_image_selection_audited", before is not None and after is not None, {"before": before, "after": after}, "present", severity="warning")

    errors = [c for c in checks if not c["ok"] and c["severity"] == "error"]
    warnings = [c for c in checks if not c["ok"] and c["severity"] == "warning"]
    report = {
        "status": "pass" if not errors else "fail",
        "strict": bool(args.strict),
        "require_all_images": bool(args.require_all_images),
        "data_dir": data_dir.as_posix(),
        "counts": counts,
        "expected_from_export_summary": expected,
        "checks": checks,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }
    out = args.out or (data_dir / "full_data_usage_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
