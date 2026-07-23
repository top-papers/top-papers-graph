# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Publication tables and a dependency-free effect plot for VLM A/B results."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def automatic_output_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize generation, JSON, and schema compliance by arm and condition."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("arm") or ""), str(row.get("condition") or ""))].append(row)
    result: dict[str, Any] = {}
    for (arm, condition), values in sorted(groups.items()):
        total = len(values)
        success = sum(row.get("status") == "success" for row in values)
        parsed = sum(row.get("parse_valid") is True for row in values)
        schema = sum(row.get("schema_valid") is True for row in values)
        result.setdefault(arm, {})[condition] = {
            "n": total,
            "generation_success": success,
            "generation_success_rate": success / total if total else None,
            "json_valid": parsed,
            "json_valid_rate": parsed / total if total else None,
            "schema_valid": schema,
            "schema_valid_rate": schema / total if total else None,
            "mean_runtime_seconds": (
                sum(float(row.get("runtime_seconds") or 0.0) for row in values) / total
                if total
                else None
            ),
        }
    return result


def error_tag_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count blinded-review error tags after deblinding."""

    result: dict[str, Any] = {}
    for arm in ("base", "tuned"):
        counter: Counter[str] = Counter()
        rows_with_any = 0
        for row in rows:
            tags = row.get(f"{arm}_error_tags") or []
            if tags:
                rows_with_any += 1
            counter.update(str(tag) for tag in tags)
        total = len(rows)
        result[arm] = {
            "n_reviews": total,
            "reviews_with_any_error": rows_with_any,
            "reviews_with_any_error_rate": rows_with_any / total if total else None,
            "tag_counts": dict(sorted(counter.items())),
            "tag_rates": {
                key: value / total if total else None for key, value in sorted(counter.items())
            },
        }
    return result


def review_missingness_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report skips and their reasons by reviewer, stratum, and criterion."""

    criteria = (
        "overall_preference",
        "evidence_preference",
        "visual_preference",
        "temporal_preference",
    )
    criterion_skips: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    by_reviewer: dict[str, Counter[str]] = defaultdict(Counter)
    by_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        reviewer = str(row.get("reviewer_id") or "unknown")
        stratum = str(row.get("stratum") or "unknown")
        skipped = [criterion for criterion in criteria if row.get(criterion) == "skip"]
        by_reviewer[reviewer]["assigned"] += 1
        by_stratum[stratum]["assigned"] += 1
        for criterion in skipped:
            criterion_skips[criterion] += 1
        if skipped:
            by_reviewer[reviewer]["rows_with_any_skip"] += 1
            by_stratum[stratum]["rows_with_any_skip"] += 1
            reason = str(row.get("comments") or "").strip()
            reasons[reason or "<missing reason>"] += 1
    return {
        "criterion_skip_counts": dict(sorted(criterion_skips.items())),
        "skip_reasons": dict(sorted(reasons.items())),
        "by_reviewer": {
            key: dict(sorted(value.items())) for key, value in sorted(by_reviewer.items())
        },
        "by_stratum": {
            key: dict(sorted(value.items())) for key, value in sorted(by_stratum.items())
        },
    }


def write_paper_scores(path: Path, summary: Mapping[str, Any]) -> None:
    """Write the prespecified paper-level analysis units to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    scores = summary.get("primary", {}).get("paper_scores", {})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "preference_score", "lift"])
        writer.writeheader()
        for paper_id, score in sorted(scores.items()):
            writer.writerow(
                {
                    "paper_id": paper_id,
                    "preference_score": score,
                    "lift": float(score) - 0.5,
                }
            )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_p(value: Any) -> str:
    if value is None:
        return "NA"
    probability = float(value)
    if probability < 0.001:
        return "<0.001"
    return f"{probability:.3f}"


def _effect_rows(results: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    rows = [("Primary", results["human_review"]["primary"])]
    for criterion, value in results["human_review"].get("secondary", {}).items():
        rows.append((criterion.replace("_", " ").title(), value))
    for stratum, value in results.get("strata", {}).items():
        rows.append((f"Exploratory stratum: {stratum}", value["primary"]))
    return rows


def write_effect_svg(path: Path, results: Mapping[str, Any]) -> None:
    """Write a simple vector forest plot without a plotting dependency."""

    rows = _effect_rows(results)
    width = 920
    left = 270
    right = 70
    top = 60
    row_height = 44
    height = top + row_height * len(rows) + 70
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + max(0.0, min(1.0, value)) * plot_width

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#17212b}.label{font-size:14px}"
        ".axis{font-size:12px;fill:#52616f}.ci{stroke:#075985;stroke-width:3}"
        ".point{fill:#075985;stroke:white;stroke-width:1.5}</style>",
        f'<text x="{left}" y="28" font-size="18" font-weight="700">Tuned preference score</text>',
        f'<line x1="{x(0.5):.1f}" y1="42" x2="{x(0.5):.1f}" y2="{height - 52}" '
        'stroke="#9aa9b7" stroke-dasharray="5 4"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        elements.append(
            f'<text class="axis" x="{x(tick):.1f}" y="{height - 25}" '
            f'text-anchor="middle">{tick:.2f}</text>'
        )
    for index, (label, value) in enumerate(rows):
        y = top + index * row_height
        estimate = value.get("estimate")
        ci = value.get("bootstrap_ci") or [None, None]
        elements.append(f'<text class="label" x="12" y="{y + 5}">{html.escape(label)}</text>')
        if estimate is None or ci[0] is None or ci[1] is None:
            elements.append(f'<text class="axis" x="{left}" y="{y + 5}">NA</text>')
            continue
        elements.extend(
            [
                f'<line class="ci" x1="{x(float(ci[0])):.1f}" y1="{y}" '
                f'x2="{x(float(ci[1])):.1f}" y2="{y}"/>',
                f'<circle class="point" cx="{x(float(estimate)):.1f}" cy="{y}" r="6"/>',
                f'<text class="axis" x="{width - 8}" y="{y + 5}" text-anchor="end">'
                f"{float(estimate):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]</text>",
            ]
        )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(elements) + "\n")


def build_markdown_report(results: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Build a manuscript-oriented Markdown results summary."""

    human = results["human_review"]
    primary = human["primary"]
    ci = primary.get("bootstrap_ci") or [None, None]
    missingness_bounds = primary.get("missingness_worst_best_case_bounds") or [None, None]
    missingness_worst_ci = primary.get("missingness_worst_case_bootstrap_ci") or [None, None]
    confidence_percent = 100.0 * float(primary.get("confidence_level", 0.95))
    publication_ready = results.get("publication_artifacts_ready") is True
    superiority = results.get("superiority_claim_supported") is True
    if publication_ready:
        status = (
            "PUBLICATION ANALYSIS READY; SUPERIORITY SUPPORTED"
            if superiority
            else "PUBLICATION ANALYSIS READY; SUPERIORITY NOT SUPPORTED"
        )
    else:
        status = "EXPLORATORY / NOT FOR PUBLICATION"
    lines = [
        "# Qwen3-VL SciReason Paired A/B Evaluation",
        "",
        f"**Status: {status}.**",
        "",
        "## Primary Result",
        "",
        (
            "The paper-macro tuned preference score was "
            f"**{_fmt(primary.get('estimate'))}** (paper-cluster bootstrap "
            f"{confidence_percent:g}% CI {_fmt(ci[0])} to {_fmt(ci[1])}; "
            "large-sample paper-cluster mean test "
            f"p={_fmt_p(primary.get('p_value'))})."
        ),
        (
            "The paired sign-flip sensitivity test targets the stronger sharp "
            "label-exchangeability null: "
            f"p={_fmt_p(primary.get('sharp_null_sign_flip_p_value'))}."
        ),
        (
            "Worst/best-case bounds after assigning every missing reviewer judgment a score "
            "of 0/1 were "
            f"{_fmt(missingness_bounds[0])} to {_fmt(missingness_bounds[1])}."
        ),
        (
            "For the worst-case imputation, the paper-cluster bootstrap CI was "
            f"{_fmt(missingness_worst_ci[0])} to {_fmt(missingness_worst_ci[1])} "
            "and the large-sample paper-cluster mean-test p-value was "
            f"{_fmt_p(primary.get('missingness_worst_case_p_value'))}."
        ),
        "",
        "| Analysis unit | Assigned reviews | Evaluable | Skips | Assigned samples | Evaluable samples | Papers | Tuned wins | Base wins | Ties |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Primary | {primary.get('n_reviews', 0)} | "
            f"{primary.get('n_evaluable_reviews', 0)} | {primary.get('n_skipped_reviews', 0)} | "
            f"{primary.get('n_assigned_samples', 0)} | {primary.get('n_samples', 0)} | "
            f"{primary.get('n_papers', 0)} | "
            f"{primary.get('tuned_wins', 0)} | {primary.get('base_wins', 0)} | "
            f"{primary.get('tie_count', 0)} |"
        ),
        "",
        "## Secondary Endpoints",
        "",
        f"| Endpoint | Preference | {confidence_percent:g}% CI | Raw p | Holm p |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for criterion, value in human.get("secondary", {}).items():
        secondary_ci = value.get("bootstrap_ci") or [None, None]
        lines.append(
            f"| {criterion} | {_fmt(value.get('estimate'))} | "
            f"{_fmt(secondary_ci[0])} to {_fmt(secondary_ci[1])} | "
            f"{_fmt_p(value.get('raw_p_value'))} | "
            f"{_fmt_p(value.get('holm_adjusted_p_value'))} |"
        )
    position = human.get("position_bias") or {}
    lines.extend(
        [
            "",
            "## Human Review Diagnostics",
            "",
            (f"- Nominal inter-rater Krippendorff alpha: {_fmt(primary.get('inter_rater_alpha'))}"),
            (
                "- Left-side preference rate: "
                f"{_fmt(position.get('left_preference_rate'))} "
                f"(descriptive exact-binomial p={_fmt_p(position.get('p_value'))})"
            ),
            "- Side assignment counts: `"
            + json.dumps(position.get("assignment_counts", {}), sort_keys=True)
            + "`",
            "- Displayed preference counts: `"
            + json.dumps(position.get("selected_side_counts", {}), sort_keys=True)
            + "`",
            "",
            "## Flow And Missingness",
            "",
            f"- Input benchmark rows: {results.get('flow', {}).get('input_rows', 0)}",
            f"- Frozen rows: {results.get('flow', {}).get('frozen_rows', 0)}",
            f"- Pre-inference exclusions: {results.get('flow', {}).get('excluded_rows', 0)}",
            (
                "- Primary paper evaluability: "
                f"{_fmt(results.get('review_completeness', {}).get('primary_paper_evaluable_rate'))}"
            ),
            (
                "- Primary review evaluability: "
                f"{_fmt(results.get('review_completeness', {}).get('primary_review_evaluable_rate'))}"
            ),
            "- Skip reasons: `"
            + json.dumps(results.get("missingness", {}).get("skip_reasons", {}), sort_keys=True)
            + "`",
            "- Skips by reviewer: `"
            + json.dumps(results.get("missingness", {}).get("by_reviewer", {}), sort_keys=True)
            + "`",
            "- Skips by stratum: `"
            + json.dumps(results.get("missingness", {}).get("by_stratum", {}), sort_keys=True)
            + "`",
            "",
            "## Reproducibility",
            "",
            f"- Experiment ID: `{config['experiment']['id']}`",
            f"- Benchmark revision: `{config['benchmark']['revision']}`",
            (
                "- Base model: "
                f"`{config['models']['base']['base_model']['id']}@"
                f"{config['models']['base']['base_model']['revision']}`"
            ),
            (
                "- Tuned adapter: "
                f"`{config['models']['tuned']['adapter']['id']}@"
                f"{config['models']['tuned']['adapter']['revision']}`"
            ),
            f"- Decoding: `{json.dumps(config['generation'], sort_keys=True)}`",
            "",
            "## Interpretation Guard",
            "",
        ]
    )
    if publication_ready:
        lines.append("All configured integrity, completeness, and benchmark gates passed.")
        if superiority:
            lines.append(
                "The prespecified statistical rule supports the tuned-model superiority claim."
            )
        else:
            lines.append("The prespecified statistical rule does not support a superiority claim.")
    else:
        lines.append(
            "At least one integrity, completeness, or benchmark gate failed. These numbers must "
            "not be presented as evidence that the tuned model outperforms the base model."
        )
    lines.extend(
        [
            "",
            "The primary endpoint is a paired human preference estimand. JSON/schema validity is "
            "reported as an operational secondary metric and is not treated as semantic quality.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "automatic_output_metrics",
    "build_markdown_report",
    "error_tag_metrics",
    "review_missingness_metrics",
    "write_effect_svg",
    "write_paper_scores",
]
