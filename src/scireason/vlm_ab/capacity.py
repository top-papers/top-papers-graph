# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared constants and workflow detection for capacity-limited VLM studies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ARCHIVAL_BLOCKED_BENCHMARK_REVISION = "33ccc5ed08e314c6457dcaa23e7f7508406cb4f8"
CAPACITY_POWER: dict[str, int | float | bool] = {
    "n_items": 150,
    "require_exact_n_items": True,
    "reviews_per_item": 2,
    "evaluable_fraction": 0.9,
    "intracluster_correlation": 0.5,
    "alpha": 0.05,
    "target_power": 0.8,
    "score_sd": 0.5,
    "target_effect": 0.121,
}


def has_capacity_remediation_identity(value: Any) -> bool:
    """Return whether an explicit experiment/public ID identifies remediation work."""

    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and "cap150" in value.lower()
        and "remediation" in value.lower()
    )


def is_capacity_remediation_working_config(config: Mapping[str, Any]) -> bool:
    """Recognize the non-publication capacity contract without trusting cosmetic IDs."""

    experiment = config.get("experiment")
    power = config.get("power")
    if not all(isinstance(value, Mapping) for value in (experiment, power)):
        return False
    return bool(
        experiment.get("require_clean_code", False) is False
        and experiment.get("require_preregistered_plan", False) is False
        and all(power.get(key) == expected for key, expected in CAPACITY_POWER.items())
    )


__all__ = [
    "ARCHIVAL_BLOCKED_BENCHMARK_REVISION",
    "CAPACITY_POWER",
    "has_capacity_remediation_identity",
    "is_capacity_remediation_working_config",
]
