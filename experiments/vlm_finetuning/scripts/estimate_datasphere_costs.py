#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Estimate Yandex DataSphere compute costs in RUB for VLM experiments.

The prices are intentionally stored in RUB/hour and match the public DataSphere
pricing table for the Russia region at the time this repository patch was made.
Storage, network traffic, and account-level discounts are not included.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

PRICES_RUB_PER_HOUR: Dict[str, float] = {
    "c1.4": 18.72,
    "c1.8": 37.44,
    "c1.32": 149.76,
    "c1.80": 374.40,
    "g1.1": 336.96,
    "g1.2": 673.92,
    "g1.4": 1347.84,
    "g2.1": 542.88,
    "g2.2": 1085.76,
    "g2.4": 2171.52,
    "g2.8": 4343.04,
    "gt4.1": 168.48,
    "gt4i.1": 234.00,
}

SCENARIOS: Dict[str, Tuple[str, float, float]] = {
    # name: (instance, compute_hours, default_reserve_rub)
    "qwen3vl_8b_sft_grpo_full_budget_guarded": ("g2.2", 87.0, 5000.0),
    "qwen3vl_8b_sft_only_pilot": ("g2.2", 24.0, 3000.0),
    "qwen3vl_8b_grpo_short_pilot": ("g2.2", 16.0, 3000.0),
    "qwen25vl_3b_smoke": ("g2.1", 8.0, 1500.0),
}


@dataclass
class Estimate:
    scenario: str
    instance: str
    hours: float
    price_rub_per_hour: float
    compute_cost_rub: float
    reserve_rub: float
    projected_total_with_reserve_rub: float
    budget_rub: float
    remaining_budget_after_projected_total_rub: float
    max_theoretical_compute_hours_without_reserve: float
    max_safe_compute_hours_after_reserve: float
    fits_budget_with_reserve: bool
    note: str


def build_estimate(
    scenario: str,
    instance: str,
    hours: float,
    budget_rub: float,
    reserve_rub: float,
) -> Estimate:
    if instance not in PRICES_RUB_PER_HOUR:
        raise KeyError(f"Unknown DataSphere instance {instance!r}. Known: {sorted(PRICES_RUB_PER_HOUR)}")
    price = PRICES_RUB_PER_HOUR[instance]
    compute_cost = hours * price
    projected_total = compute_cost + reserve_rub
    return Estimate(
        scenario=scenario,
        instance=instance,
        hours=round(hours, 4),
        price_rub_per_hour=price,
        compute_cost_rub=round(compute_cost, 2),
        reserve_rub=round(reserve_rub, 2),
        projected_total_with_reserve_rub=round(projected_total, 2),
        budget_rub=round(budget_rub, 2),
        remaining_budget_after_projected_total_rub=round(budget_rub - projected_total, 2),
        max_theoretical_compute_hours_without_reserve=round(budget_rub / price, 2),
        max_safe_compute_hours_after_reserve=round(max(0.0, budget_rub - reserve_rub) / price, 2),
        fits_budget_with_reserve=projected_total <= budget_rub,
        note=(
            "Estimate covers DataSphere compute only plus a user-selected reserve. "
            "Job data storage, network traffic, and account-level price changes are external to this file."
        ),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Estimate DataSphere compute cost for VLM fine-tuning runs.")
    ap.add_argument("--scenario", default="qwen3vl_8b_sft_grpo_full_budget_guarded", choices=sorted(SCENARIOS))
    ap.add_argument("--instance", default=None, choices=sorted(PRICES_RUB_PER_HOUR))
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--budget-rub", type=float, default=100000.0)
    ap.add_argument("--reserve-rub", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    return ap.parse_args()


def render_text(estimate: Estimate) -> str:
    return "\n".join(
        [
            f"scenario={estimate.scenario}",
            f"instance={estimate.instance}",
            f"hours={estimate.hours:.2f}",
            f"price_rub_per_hour={estimate.price_rub_per_hour:.2f}",
            f"compute_cost_rub={estimate.compute_cost_rub:.2f}",
            f"reserve_rub={estimate.reserve_rub:.2f}",
            f"projected_total_with_reserve_rub={estimate.projected_total_with_reserve_rub:.2f}",
            f"budget_rub={estimate.budget_rub:.2f}",
            f"remaining_budget_after_projected_total_rub={estimate.remaining_budget_after_projected_total_rub:.2f}",
            f"max_safe_compute_hours_after_reserve={estimate.max_safe_compute_hours_after_reserve:.2f}",
            f"fits_budget_with_reserve={str(estimate.fits_budget_with_reserve).lower()}",
        ]
    ) + "\n"


def render_markdown(estimate: Estimate) -> str:
    status = "OK" if estimate.fits_budget_with_reserve else "OVER BUDGET"
    return f"""# DataSphere cost estimate

| Field | Value |
|---|---:|
| Scenario | `{estimate.scenario}` |
| Instance | `{estimate.instance}` |
| Hours | {estimate.hours:.2f} |
| Price, RUB/hour | {estimate.price_rub_per_hour:.2f} |
| Compute cost, RUB | {estimate.compute_cost_rub:.2f} |
| Reserve, RUB | {estimate.reserve_rub:.2f} |
| Projected total with reserve, RUB | {estimate.projected_total_with_reserve_rub:.2f} |
| Budget, RUB | {estimate.budget_rub:.2f} |
| Remaining after projected total, RUB | {estimate.remaining_budget_after_projected_total_rub:.2f} |
| Max safe compute hours after reserve | {estimate.max_safe_compute_hours_after_reserve:.2f} |
| Status | **{status}** |

{estimate.note}
"""


def main() -> None:
    args = parse_args()
    default_instance, default_hours, default_reserve = SCENARIOS[args.scenario]
    instance = args.instance or default_instance
    hours = args.hours if args.hours is not None else default_hours
    reserve = args.reserve_rub if args.reserve_rub is not None else default_reserve
    estimate = build_estimate(args.scenario, instance, hours, args.budget_rub, reserve)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        text = json.dumps(asdict(estimate), ensure_ascii=False, indent=2) + "\n"
    elif args.format == "markdown":
        text = render_markdown(estimate)
    else:
        text = render_text(estimate)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
