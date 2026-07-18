# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math

import pytest

from scireason.vlm_ab.stats import (
    cluster_sign_flip_pvalue,
    holm_correction,
    krippendorff_alpha_nominal,
    paper_cluster_bootstrap_ci,
    power_mde_plan,
    summarize_reviews,
)


def test_holm_correction_preserves_order_and_missing_values() -> None:
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    corrected = holm_correction({"first": 0.01, "missing": None, "last": 0.04})
    assert corrected == {"first": 0.03, "missing": None, "last": 0.08}

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_correction([1.01])


def test_nominal_krippendorff_alpha_handles_missing_ratings() -> None:
    ratings = {
        "item-1": ["tuned", "tuned"],
        "item-2": ["tuned", "base"],
        "item-3": ["base", None],
    }
    assert krippendorff_alpha_nominal(ratings) == pytest.approx(0.0)
    assert krippendorff_alpha_nominal([["tuned", "tuned"], ["base", "base"]]) == 1.0
    assert math.isnan(krippendorff_alpha_nominal([["tuned", None], ["tuned", "tuned"]]))


def test_paper_cluster_bootstrap_is_deterministic_and_aggregates_within_paper() -> None:
    first = paper_cluster_bootstrap_ci(
        [1.0, 1.0, 0.0],
        ["paper-a", "paper-a", "paper-b"],
        n_resamples=500,
        seed=19,
    )
    second = paper_cluster_bootstrap_ci(
        [0.0, 1.0, 1.0],
        ["paper-b", "paper-a", "paper-a"],
        n_resamples=500,
        seed=19,
    )
    assert first == second
    assert first == (0.0, 1.0)
    assert paper_cluster_bootstrap_ci({"only-paper": [1.0, 0.0]}) == (0.5, 0.5)


def test_cluster_sign_flip_uses_exact_test_for_small_samples() -> None:
    assert cluster_sign_flip_pvalue([1.0, 1.0, 1.0]) == pytest.approx(0.25)
    assert cluster_sign_flip_pvalue([0.5, 0.5]) == 1.0
    assert math.isnan(cluster_sign_flip_pvalue([]))

    first = cluster_sign_flip_pvalue(
        [0.9] * 17,
        n_resamples=300,
        seed=7,
        exact_max_clusters=16,
    )
    second = cluster_sign_flip_pvalue(
        [0.9] * 17,
        n_resamples=300,
        seed=7,
        exact_max_clusters=16,
    )
    assert first == second


def test_power_mde_plan_counts_only_independent_papers() -> None:
    assert power_mde_plan(10)["evaluable_fraction"] == 1.0
    plan = power_mde_plan(
        100,
        2,
        evaluable_fraction=0.8,
        intracluster_correlation=0.25,
        target_effect=0.15,
    )
    assert plan["design_effect"] == pytest.approx(1.25)
    assert plan["effective_sample_size"] == pytest.approx(80.0)
    assert plan["mde"] == pytest.approx(0.156613, abs=1e-6)
    assert plan["achieved_power"] is not None
    assert plan["required_items"] is not None

    with pytest.raises(ValueError, match="evaluable_fraction"):
        power_mde_plan(10, evaluable_fraction=0.0)


def _review_rows() -> list[dict[str, object]]:
    labels = [
        ("r1", "s1", "p1", "a", "a", "tuned", "tuned", "base"),
        ("r2", "s1", "p1", "b", "b", "tuned", "tuned", "tie"),
        ("r1", "s2", "p1", "a", "tie", "tie", "tuned", "tuned"),
        ("r2", "s2", "p1", "b", "a", "base", "base", "tuned"),
        ("r1", "s3", "p2", "a", "a", "tuned", "tuned", "base"),
        ("r2", "s3", "p2", "b", "a", "base", "tuned", "base"),
    ]
    rows: list[dict[str, object]] = []
    for reviewer, sample, paper, tuned_side, displayed, overall, evidence, temporal in labels:
        rows.append(
            {
                "reviewer_id": reviewer,
                "sample_id": sample,
                "paper_id": paper,
                "condition": "original",
                "stratum": "hard" if paper == "p1" else "control",
                "primary_endpoint": True,
                "tuned_side": tuned_side,
                "displayed_preference": displayed,
                "criterion_winners": {
                    "overall": overall,
                    "evidence": evidence,
                    "temporal": temporal,
                },
            }
        )
    rows.extend(
        [
            {
                **rows[0],
                "reviewer_id": "excluded-repeat",
                "sample_id": "excluded-repeat",
                "condition": "repeat",
            },
            {
                **rows[0],
                "reviewer_id": "excluded-control",
                "sample_id": "excluded-control",
                "primary_endpoint": False,
            },
        ]
    )
    return rows


def test_summarize_reviews_filters_and_aggregates_hierarchy() -> None:
    config = {
        "primary_criterion": "overall",
        "secondary_criteria": ["evidence", "temporal"],
        "bootstrap_resamples": 400,
        "randomization_resamples": 400,
        "seed": 23,
    }
    summary = summarize_reviews(_review_rows(), config)

    assert summary["n_input_rows"] == 8
    assert summary["n_primary_rows"] == 6
    assert summary["counts"] == {"tuned": 3, "base": 2, "tie": 1, "skip": 0}
    assert summary["primary"]["sample_scores"] == {
        "s1": 1.0,
        "s2": 0.25,
        "s3": 0.5,
    }
    assert summary["primary"]["paper_scores"] == {"p1": 0.625, "p2": 0.5}
    assert summary["estimate"] == pytest.approx(0.5625)
    assert summary["lift_from_half"] == pytest.approx(0.0625)
    assert summary["n_samples"] == 3
    assert summary["n_papers"] == 2
    assert summary["inter_rater_alpha"] is not None
    assert summary["position_bias"]["selected_side_counts"] == {
        "a": 4,
        "b": 1,
        "tie": 1,
        "skip": 0,
    }
    assert summary["position_bias"]["position_bias"] == pytest.approx(0.3)
    for result in summary["secondary"].values():
        assert result["holm_adjusted_p_value"] >= result["raw_p_value"]


def test_summarize_reviews_is_order_invariant_and_handles_empty_data() -> None:
    config = {
        "primary_criterion": "overall",
        "secondary_criteria": ["evidence"],
        "bootstrap_resamples": 300,
        "randomization_resamples": 300,
        "seed": 11,
    }
    rows = _review_rows()
    assert summarize_reviews(rows, config) == summarize_reviews(reversed(rows), config)

    empty = summarize_reviews([], {**config, "secondary_criteria": []})
    assert empty["estimate"] is None
    assert empty["bootstrap_ci"] == [None, None]
    assert empty["p_value"] is None
    assert empty["inter_rater_alpha"] is None
    assert empty["position_bias"] is None


def test_summarize_reviews_accepts_flat_deblinded_preference_fields() -> None:
    rows = [
        {
            "reviewer_id": reviewer,
            "sample_id": "sample",
            "paper_id": "paper",
            "condition": "original",
            "stratum": "hard",
            "primary_endpoint": True,
            "tuned_side": side,
            "displayed_preference": displayed,
            "preference": winner,
            "overall_preference": winner,
            "evidence_preference": "tie",
            "displayed_evidence_preference": "tie",
        }
        for reviewer, side, displayed, winner in (
            ("r1", "left", "left", "tuned"),
            ("r2", "right", "left", "base"),
        )
    ]
    summary = summarize_reviews(
        rows,
        {
            "secondary_criteria": [],
            "bootstrap_resamples": 100,
            "randomization_resamples": 100,
        },
    )
    assert summary["primary_criterion"] == "preference"
    assert summary["estimate"] == 0.5
    assert summary["secondary"] == {}


def test_missingness_sensitivity_imputes_each_skipped_judgment() -> None:
    rows = []
    for paper_id in ("paper-1", "paper-2"):
        sample_id = f"sample-{paper_id}"
        for reviewer_id, winner in (("r1", "tuned"), ("r2", "skip")):
            rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "sample_id": sample_id,
                    "paper_id": paper_id,
                    "condition": "original",
                    "stratum": "multimodal_hard",
                    "primary_endpoint": True,
                    "tuned_side": "a",
                    "overall_preference": winner,
                }
            )

    summary = summarize_reviews(
        rows,
        {
            "primary_criterion": "overall_preference",
            "secondary_criteria": [],
            "bootstrap_resamples": 100,
            "randomization_resamples": 100,
        },
    )["primary"]

    assert summary["estimate"] == 1.0
    assert summary["missingness_worst_best_case_bounds"] == [0.5, 1.0]
    assert summary["missingness_worst_case_bootstrap_ci"] == [0.5, 0.5]
    assert summary["missingness_worst_case_p_value"] == 1.0


def test_summarize_reviews_rejects_invalid_or_duplicate_normalized_rows() -> None:
    rows = _review_rows()[:1]
    rows[0]["criterion_winners"] = {"overall": "left"}
    with pytest.raises(ValueError, match="invalid winner"):
        summarize_reviews(rows, {"primary_criterion": "overall"})

    duplicate = _review_rows()[:1] * 2
    with pytest.raises(ValueError, match="duplicate review"):
        summarize_reviews(duplicate, {"primary_criterion": "overall"})
