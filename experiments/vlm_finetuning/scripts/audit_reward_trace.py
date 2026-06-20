#!/usr/bin/env python3
"""Post-run reward trace audit for GRPO.

Use this after train_vlm_grpo.py or as a standalone check on grpo_reward_trace.jsonl.
It detects the degenerate reward pattern highlighted in the audit report:
components with near-zero std and generation groups with zero reward variance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, var ** 0.5


def audit(path: Path, min_reward_std: float, max_zero_std_frac: float) -> dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj.get("reward"), (int, float)):
                rows.append(obj)
    by_component: dict[str, list[float]] = {}
    by_group: dict[tuple[Any, str, str], list[float]] = {}
    for obj in rows:
        component = str(obj.get("component") or "unknown")
        reward = float(obj["reward"])
        by_component.setdefault(component, []).append(reward)
        by_group.setdefault((obj.get("global_step"), component, str(obj.get("sample_id") or "")), []).append(reward)

    component_stats = {}
    weak_components = []
    for component, values in sorted(by_component.items()):
        mean, std = mean_std(values)
        component_stats[component] = {
            "count": len(values), "mean": round(mean, 6), "std": round(std, 6),
            "min": min(values), "max": max(values),
        }
        if len(values) >= 8 and std < min_reward_std and component not in {"label_exact_match"}:
            weak_components.append(component)

    group_stds = [mean_std(values)[1] for values in by_group.values() if len(values) > 1]
    zero_std_frac = sum(1 for x in group_stds if x < 1e-9) / len(group_stds) if group_stds else 1.0
    weak_reward = bool(weak_components) or zero_std_frac > max_zero_std_frac
    return {
        "status": "fail" if weak_reward else "pass",
        "rows": len(rows),
        "component_stats": component_stats,
        "weak_components": weak_components,
        "group_count": len(group_stds),
        "group_zero_std_fraction": round(zero_std_frac, 6),
        "thresholds": {"min_reward_std": min_reward_std, "max_zero_std_frac": max_zero_std_frac},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-reward-std", type=float, default=0.02)
    ap.add_argument("--max-zero-std-frac", type=float, default=0.7)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    report = audit(args.trace, args.min_reward_std, args.max_zero_std_frac)
    out = args.out or args.trace.with_name("post_run_reward_audit.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and report["status"] == "fail":
        sys.exit(2)


if __name__ == "__main__":
    main()
