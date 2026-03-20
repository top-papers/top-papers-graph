#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

PRICES = {'g2.1': 4.51, 'g2.2': 9.02, 'g2.4': 18.04, 'c1.4': 0.29, 'c1.8': 0.58}
SCENARIOS = {
    'sft_smoke': ('g2.1', 2.0),
    'sft_pilot': ('g2.2', 10.0),
    'dpo_pilot': ('g2.2', 8.0),
    'teacher_30b_a3b_sft': ('g2.4', 14.0),
    'student_distill_4b': ('g2.2', 12.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=sorted(SCENARIOS))
    ap.add_argument('--hours', type=float, default=None)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    instance, default_hours = SCENARIOS[args.scenario]
    hours = args.hours if args.hours is not None else default_hours
    text = (
        f'scenario={args.scenario}\n'
        f'instance={instance}\n'
        f'hours={hours:.2f}\n'
        f'price_per_hour_usd={PRICES[instance]:.2f}\n'
        f'estimated_total_usd={hours * PRICES[instance]:.2f}\n'
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
