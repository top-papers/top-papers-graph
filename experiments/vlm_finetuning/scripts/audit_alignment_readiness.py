#!/usr/bin/env python3
"""Preflight quality gates for the SciReason SFT/DPO/GRPO pipeline.

The audit report for Qwen3-VL-8B-Instruct-scireason identified three costly
failure classes: train/eval leakage risk, aggressive image truncation, and weak
GRPO reward signal. This script turns those findings into machine-checkable
preflight gates before a DataSphere run spends GPU time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "path": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc), "path": str(path)}
    return value if isinstance(value, dict) else {"_value": value}


def gate(ok: bool, message: str, severity: str = "error") -> dict[str, Any]:
    return {"ok": bool(ok), "severity": severity, "message": message}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate that prepared SciReason alignment data is safe enough for training.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on error gates.")
    ap.add_argument("--min-dpo-train-rows", type=int, default=100)
    ap.add_argument("--min-sft-vlm-train-rows", type=int, default=100)
    ap.add_argument("--min-grpo-verified-rows", type=int, default=50)
    ap.add_argument("--max-grpo-image-truncation-rate", type=float, default=0.85)
    ap.add_argument("--min-hard-pair-ratio", type=float, default=0.35)
    args = ap.parse_args()

    data_dir = args.data_dir
    summary = read_json(data_dir / "summary.json")
    leakage = read_json(data_dir / "leakage_report.json")
    image_report = read_json(data_dir / "image_resolution_report.json")
    reward_audit = read_json(data_dir / "reward_audit_by_task_family.json")

    counts = summary.get("counts", {}) if isinstance(summary.get("counts"), dict) else {}
    gates: list[dict[str, Any]] = []

    for split_name in ("sft", "dpo", "grpo_all", "grpo_verified"):
        split = leakage.get(split_name, {}) if isinstance(leakage.get(split_name), dict) else {}
        overlap = int(split.get("group_overlap_count", 1 if split.get("_missing") else 0) or 0)
        gates.append(gate(overlap == 0, f"{split_name}: train/eval leakage group overlap = {overlap}"))

    gates.append(gate(int(counts.get("sft_vlm_train", 0) or 0) >= args.min_sft_vlm_train_rows,
                      f"sft_vlm_train rows = {counts.get('sft_vlm_train', 0)}; expected >= {args.min_sft_vlm_train_rows}"))
    gates.append(gate(int(counts.get("dpo_train", 0) or 0) >= args.min_dpo_train_rows,
                      f"dpo_train rows = {counts.get('dpo_train', 0)}; expected >= {args.min_dpo_train_rows}"))

    dpo_rows = read_jsonl(data_dir / "dpo_train.jsonl")
    hard_pair_rows = 0
    for row in dpo_rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        pair_type = str(meta.get("pair_type") or "")
        hardness = float(meta.get("pair_hardness", 0.0) or 0.0)
        if pair_type not in {"", "explicit"} or hardness >= 0.25:
            hard_pair_rows += 1
    hard_pair_ratio = hard_pair_rows / len(dpo_rows) if dpo_rows else 0.0
    gates.append(gate(hard_pair_ratio >= args.min_hard_pair_ratio,
                      f"DPO hard-pair ratio = {hard_pair_ratio:.3f}; expected >= {args.min_hard_pair_ratio:.3f}"))
    gates.append(gate(int(counts.get("grpo_train_verified", 0) or 0) >= args.min_grpo_verified_rows,
                      f"grpo_train_verified rows = {counts.get('grpo_train_verified', 0)}; expected >= {args.min_grpo_verified_rows}",
                      severity="warning"))

    grpo_img = image_report.get("grpo", {}) if isinstance(image_report.get("grpo"), dict) else {}
    total = int(grpo_img.get("rows_total", 0) or grpo_img.get("input_rows", 0) or 0)
    truncated = int(grpo_img.get("rows_truncated", 0) or grpo_img.get("truncated_rows", 0) or 0)
    trunc_rate = truncated / total if total else 0.0
    gates.append(gate(trunc_rate <= args.max_grpo_image_truncation_rate,
                      f"GRPO image truncation rate = {trunc_rate:.3f}; expected <= {args.max_grpo_image_truncation_rate:.3f}",
                      severity="warning"))

    kept_ratio = float(reward_audit.get("kept_ratio", 0.0) or 0.0)
    gates.append(gate(kept_ratio > 0.0, f"GRPO reward-ready kept_ratio = {kept_ratio:.4f}", severity="warning"))

    error_count = sum(1 for g in gates if not g["ok"] and g["severity"] == "error")
    warning_count = sum(1 for g in gates if not g["ok"] and g["severity"] == "warning")
    report = {
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "warning_count": warning_count,
        "data_dir": str(data_dir),
        "gates": gates,
        "summary_counts": counts,
        "dpo_hard_pair_ratio": hard_pair_ratio if 'hard_pair_ratio' in locals() else None,
        "note": "Warnings do not block SFT/DPO, but GRPO should stay disabled until reward/image warnings are understood.",
    }

    out = args.out or (data_dir / "alignment_readiness_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and error_count:
        sys.exit(2)


if __name__ == "__main__":
    main()
