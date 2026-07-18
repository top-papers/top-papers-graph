# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deterministic statistics for paired, paper-clustered VLM A/B reviews.

The public helpers deliberately depend only on the standard library and NumPy.  The
summary estimator gives equal weight at each prespecified level: reviewers are averaged
within samples, samples within papers, and papers in the final estimate.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np


_MISSING = object()
_WINNERS = ("tuned", "base", "tie", "skip")


@dataclass(frozen=True)
class ReviewStatsConfig:
    """Configuration accepted directly by :func:`summarize_reviews`.

    A mapping or another object with equivalent attributes is accepted as well.  If no
    primary criterion is supplied, it is inferred from the normalized rows.
    """

    primary_criterion: str | None = None
    secondary_criteria: tuple[str, ...] | None = None
    primary_condition: str | None = "original"
    primary_endpoint_only: bool = True
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    randomization_resamples: int = 10_000
    exact_max_clusters: int = 16
    seed: int = 0


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    return False


def _validate_probability(value: Any, *, name: str, open_interval: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return result


def _validate_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _holm_sequence(values: Sequence[float | None]) -> list[float | None]:
    adjusted: list[float | None] = list(values)
    valid: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        if _is_missing(value):
            continue
        probability = _validate_probability(value, name=f"p_values[{index}]")
        valid.append((index, probability))

    ordered = sorted(valid, key=lambda item: (item[1], item[0]))
    running_max = 0.0
    count = len(values)
    for rank, (index, probability) in enumerate(ordered):
        running_max = max(running_max, (count - rank) * probability)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def holm_correction(
    p_values: Sequence[float | None] | Mapping[Any, float | None],
) -> list[float | None] | dict[Any, float | None]:
    """Return family-wise-error-controlled p-values using Holm's step-down method.

    Missing values (``None`` or ``NaN``) are preserved but remain counted in the
    prespecified family as non-rejections. Mapping keys and sequence order are preserved.
    """

    if isinstance(p_values, Mapping):
        keys = list(p_values)
        corrected = _holm_sequence([p_values[key] for key in keys])
        return dict(zip(keys, corrected))
    return _holm_sequence(list(p_values))


holm_adjust = holm_correction


def krippendorff_alpha_nominal(
    ratings_by_item: Mapping[Any, Iterable[Any]] | Iterable[Iterable[Any]],
) -> float:
    """Compute nominal Krippendorff alpha with item-level missing ratings.

    Each inner iterable contains all ratings for one item.  ``None`` and ``NaN`` are
    missing.  Items with fewer than two observed ratings do not contribute coincidences.
    Alpha is ``NaN`` when it is unidentified, including data with no category variation.
    """

    units: Iterable[Iterable[Any]]
    units = ratings_by_item.values() if isinstance(ratings_by_item, Mapping) else ratings_by_item
    observed_disagreement_numerator = 0.0
    category_totals: Counter[Any] = Counter()
    coincidence_count = 0

    for unit in units:
        if isinstance(unit, Mapping):
            raw_values = list(unit.values())
        elif isinstance(unit, (str, bytes)):
            raw_values = [unit]
        else:
            raw_values = list(unit)
        values = [value for value in raw_values if not _is_missing(value)]
        if len(values) < 2:
            continue
        try:
            counts = Counter(values)
        except TypeError as exc:
            raise TypeError("nominal ratings must be hashable") from exc

        item_count = len(values)
        unequal_ordered_pairs = item_count**2 - sum(count**2 for count in counts.values())
        observed_disagreement_numerator += unequal_ordered_pairs / (item_count - 1)
        category_totals.update(counts)
        coincidence_count += item_count

    if coincidence_count < 2:
        return math.nan

    observed_disagreement = observed_disagreement_numerator / coincidence_count
    expected_numerator = coincidence_count**2 - sum(count**2 for count in category_totals.values())
    expected_disagreement = expected_numerator / (coincidence_count * (coincidence_count - 1))
    if expected_disagreement <= 0.0:
        return math.nan

    alpha = 1.0 - observed_disagreement / expected_disagreement
    if abs(alpha) < 1e-15:
        return 0.0
    if abs(alpha - 1.0) < 1e-15:
        return 1.0
    return float(alpha)


nominal_krippendorff_alpha = krippendorff_alpha_nominal
krippendorff_alpha = krippendorff_alpha_nominal


def _stable_key(value: Any) -> tuple[str, str, str]:
    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, repr(value))


def _finite_scores(values: Iterable[Any]) -> list[float]:
    scores: list[float] = []
    for value in values:
        if _is_missing(value):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("scores must be numeric") from exc
        if not math.isfinite(score):
            raise ValueError("scores must be finite")
        scores.append(score)
    return scores


def _paper_level_scores(
    scores: Sequence[float] | Mapping[Any, float | Sequence[float]],
    paper_ids: Sequence[Any] | None,
) -> np.ndarray:
    if isinstance(scores, Mapping):
        if paper_ids is not None:
            raise ValueError("paper_ids must be omitted when scores is a mapping")
        paper_scores: list[float] = []
        for paper_id in sorted(scores, key=_stable_key):
            raw = scores[paper_id]
            if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
                values = _finite_scores(raw)
            else:
                values = _finite_scores([raw])
            if values:
                paper_scores.append(float(np.mean(values)))
        return np.asarray(paper_scores, dtype=float)

    raw_scores = list(scores)
    if paper_ids is None:
        return np.asarray(sorted(_finite_scores(raw_scores)), dtype=float)

    raw_papers = list(paper_ids)
    if len(raw_scores) != len(raw_papers):
        raise ValueError("scores and paper_ids must have equal length")
    grouped: dict[Any, list[float]] = defaultdict(list)
    for score, paper_id in zip(raw_scores, raw_papers):
        values = _finite_scores([score])
        if not values:
            continue
        if _is_missing(paper_id):
            raise ValueError("paper_ids cannot be missing for observed scores")
        try:
            grouped[paper_id].append(values[0])
        except TypeError as exc:
            raise TypeError("paper_ids must be hashable") from exc
    return np.asarray(
        [float(np.mean(grouped[key])) for key in sorted(grouped, key=_stable_key)],
        dtype=float,
    )


def paper_cluster_bootstrap_ci(
    scores: Sequence[float] | Mapping[Any, float | Sequence[float]],
    paper_ids: Sequence[Any] | None = None,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI obtained by resampling papers with replacement.

    If ``paper_ids`` is supplied, raw scores are first averaged within papers.  A mapping
    may instead provide one score, or a sequence of scores, for each paper.
    """

    confidence = _validate_probability(
        confidence_level, name="confidence_level", open_interval=True
    )
    resamples = _validate_positive_int(n_resamples, name="n_resamples")
    paper_scores = _paper_level_scores(scores, paper_ids)
    if paper_scores.size == 0:
        return (math.nan, math.nan)
    if paper_scores.size == 1:
        value = float(paper_scores[0])
        return (value, value)

    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    cluster_count = paper_scores.size
    chunk_size = 2_048
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        sampled = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        estimates[start:stop] = paper_scores[sampled].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail], method="linear")
    return (float(lower), float(upper))


cluster_bootstrap_ci = paper_cluster_bootstrap_ci


def cluster_sign_flip_pvalue(
    scores: Sequence[float] | Mapping[Any, float | Sequence[float]],
    paper_ids: Sequence[Any] | None = None,
    *,
    null: float = 0.5,
    n_resamples: int = 10_000,
    seed: int = 0,
    exact_max_clusters: int = 16,
) -> float:
    """Two-sided paper-cluster sign-flip randomization p-value.

    Exact enumeration is used for at most ``exact_max_clusters`` nonzero paper effects
    (capped at 20 for tractability).  Larger tests use a seeded Monte Carlo estimate with
    the standard plus-one correction.
    """

    resamples = _validate_positive_int(n_resamples, name="n_resamples")
    if isinstance(exact_max_clusters, bool) or int(exact_max_clusters) != exact_max_clusters:
        raise TypeError("exact_max_clusters must be an integer")
    if exact_max_clusters < 0:
        raise ValueError("exact_max_clusters cannot be negative")
    try:
        null_value = float(null)
    except (TypeError, ValueError) as exc:
        raise TypeError("null must be numeric") from exc
    if not math.isfinite(null_value):
        raise ValueError("null must be finite")

    paper_scores = _paper_level_scores(scores, paper_ids)
    if paper_scores.size == 0:
        return math.nan
    effects = np.sort(paper_scores - null_value)
    effects = effects[effects != 0.0]
    if effects.size == 0:
        return 1.0

    observed = abs(float(effects.mean()))
    tolerance = max(1e-15, observed * 1e-12)
    cluster_count = effects.size
    exact_limit = min(int(exact_max_clusters), 20)
    if cluster_count <= exact_limit:
        total = 1 << cluster_count
        extreme = 0
        bit_positions = np.arange(cluster_count, dtype=np.uint64)
        chunk_size = 4_096
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where(((masks >> bit_positions) & 1) == 1, 1.0, -1.0)
            permuted = np.abs(signs @ effects / cluster_count)
            extreme += int(np.count_nonzero(permuted >= observed - tolerance))
        return extreme / total

    rng = np.random.default_rng(seed)
    extreme = 0
    chunk_size = 2_048
    for start in range(0, resamples, chunk_size):
        size = min(chunk_size, resamples - start)
        signs = rng.integers(0, 2, size=(size, cluster_count), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = np.abs(signs @ effects / cluster_count)
        extreme += int(np.count_nonzero(permuted >= observed - tolerance))
    return (extreme + 1.0) / (resamples + 1.0)


cluster_sign_flip_p_value = cluster_sign_flip_pvalue
cluster_randomization_pvalue = cluster_sign_flip_pvalue


def cluster_mean_normal_pvalue(
    scores: Sequence[float] | Mapping[Any, float | Sequence[float]],
    paper_ids: Sequence[Any] | None = None,
    *,
    null: float = 0.5,
) -> float:
    """Two-sided large-sample test of the paper-macro mean against ``null``.

    Papers are the independent units. This weak-null test uses the sample standard
    error across paper scores and a standard-normal reference distribution. Runs
    with few papers must remain exploratory.
    """

    paper_scores = _paper_level_scores(scores, paper_ids)
    if paper_scores.size < 2:
        return math.nan
    effect = float(paper_scores.mean() - float(null))
    standard_error = float(paper_scores.std(ddof=1) / math.sqrt(paper_scores.size))
    if standard_error == 0.0:
        return 1.0 if effect == 0.0 else 0.0
    statistic = abs(effect / standard_error)
    return float(2.0 * (1.0 - NormalDist().cdf(statistic)))


def power_mde_plan(
    n_items: int,
    reviews_per_item: float = 1.0,
    *,
    evaluable_fraction: float = 1.0,
    intracluster_correlation: float = 0.0,
    alpha: float = 0.05,
    target_power: float = 0.80,
    score_sd: float = 0.5,
    target_effect: float | None = None,
) -> dict[str, Any]:
    """Plan power and MDE with a conservative paper-cluster approximation.

    ``n_items`` is the number of independent papers. Repeated reviewers improve
    measurement reliability but are not counted as additional independent papers.
    ``score_sd=0.5`` is conservative for a paper score bounded by zero and one.
    """

    item_count = _validate_positive_int(n_items, name="n_items")
    try:
        reviews = float(reviews_per_item)
        standard_deviation = float(score_sd)
    except (TypeError, ValueError) as exc:
        raise TypeError("reviews_per_item and score_sd must be numeric") from exc
    if not math.isfinite(reviews) or reviews <= 0.0:
        raise ValueError("reviews_per_item must be positive and finite")
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("score_sd must be positive and finite")

    evaluable = _validate_probability(evaluable_fraction, name="evaluable_fraction")
    if evaluable == 0.0:
        raise ValueError("evaluable_fraction must be in (0, 1]")
    correlation = _validate_probability(intracluster_correlation, name="intracluster_correlation")
    alpha_value = _validate_probability(alpha, name="alpha", open_interval=True)
    power_value = _validate_probability(target_power, name="target_power", open_interval=True)

    design_effect = 1.0 + (reviews - 1.0) * correlation
    effective_sample_size = item_count * evaluable
    standard_error = standard_deviation / math.sqrt(effective_sample_size)
    normal = NormalDist()
    critical_value = normal.inv_cdf(1.0 - alpha_value / 2.0)
    power_quantile = normal.inv_cdf(power_value)
    mde = (critical_value + power_quantile) * standard_error

    achieved_power: float | None = None
    required_items: int | None = None
    effect_value: float | None = None
    if target_effect is not None:
        try:
            effect_value = abs(float(target_effect))
        except (TypeError, ValueError) as exc:
            raise TypeError("target_effect must be numeric") from exc
        if not math.isfinite(effect_value) or effect_value <= 0.0:
            raise ValueError("target_effect must be positive and finite")
        noncentrality = effect_value / standard_error
        achieved_power = (
            normal.cdf(-critical_value - noncentrality)
            + 1.0
            - normal.cdf(critical_value - noncentrality)
        )
        required_effective_n = (
            (critical_value + power_quantile) * standard_deviation / effect_value
        ) ** 2
        required_items = math.ceil(required_effective_n / evaluable)

    return {
        "independent_unit": "paper",
        "n_items": item_count,
        "reviews_per_item": reviews,
        "evaluable_fraction": evaluable,
        "expected_evaluable_reviews": item_count * reviews * evaluable,
        "intracluster_correlation": correlation,
        "design_effect": design_effect,
        "design_effect_is_informational": True,
        "effective_sample_size": effective_sample_size,
        "score_sd": standard_deviation,
        "standard_error": standard_error,
        "alpha": alpha_value,
        "target_power": power_value,
        "mde": mde,
        "target_effect": effect_value,
        "achieved_power": achieved_power,
        "required_items": required_items,
    }


plan_power_mde = power_mde_plan
power_mde_planner = power_mde_plan


def _config_sources(config: Any) -> list[Any]:
    if config is None:
        return []
    sources = [config]
    if isinstance(config, Mapping):
        for key in ("statistics", "stats", "analysis"):
            nested = config.get(key)
            if isinstance(nested, Mapping):
                sources.insert(0, nested)
    return sources


def _config_value(config: Any, names: Sequence[str], default: Any) -> Any:
    for source in _config_sources(config):
        for name in names:
            if isinstance(source, Mapping) and name in source:
                return source[name]
            value = getattr(source, name, _MISSING)
            if value is not _MISSING:
                return value
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    raise ValueError(f"cannot interpret {value!r} as boolean")


def _winner_value(row: Mapping[str, Any], criterion: str) -> Any:
    for container_name in ("criterion_winners", "winners", "criteria"):
        container = row.get(container_name)
        if isinstance(container, Mapping) and criterion in container:
            value = container[criterion]
            if isinstance(value, Mapping):
                return value.get("winner")
            return value
    for key in (criterion, f"{criterion}_winner"):
        if key in row:
            value = row[key]
            if isinstance(value, Mapping):
                return value.get("winner")
            return value
    if criterion in {"overall", "preference", "primary"} and "winner" in row:
        return row["winner"]
    return None


def _canonical_winner(value: Any, *, criterion: str) -> str:
    if _is_missing(value) or (isinstance(value, str) and not value.strip()):
        return "skip"
    normalized = str(value).strip().lower()
    if normalized not in _WINNERS:
        raise ValueError(
            f"invalid winner {value!r} for criterion {criterion!r}; expected one of {_WINNERS}"
        )
    return normalized


_NON_CRITERION_FIELDS = {
    "condition",
    "displayed_preference",
    "paper_id",
    "primary_endpoint",
    "reviewer_id",
    "sample_id",
    "stratum",
    "tuned_side",
}


def _discover_criteria(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for container_name in ("criterion_winners", "winners", "criteria"):
            container = row.get(container_name)
            if isinstance(container, Mapping):
                names.update(str(key) for key in container)
        for key, value in row.items():
            name = str(key)
            if (
                name.startswith("displayed_")
                or name in _NON_CRITERION_FIELDS
                or name
                in {
                    "criterion_winners",
                    "criteria",
                    "winners",
                }
            ):
                continue
            candidate = name[: -len("_winner")] if name.endswith("_winner") else name
            if isinstance(value, str) and value.strip().lower() in _WINNERS:
                names.add(candidate)
    return sorted(names)


def _normalise_criteria_config(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(name) for name in value]
    return [str(name) for name in value]


def _primary_rows(
    rows: Sequence[Mapping[str, Any]], config: Any
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    condition = _config_value(config, ("primary_condition", "condition"), "original")
    endpoint_only = _as_bool(
        _config_value(config, ("primary_endpoint_only", "require_primary_endpoint"), True)
    )
    endpoint_target: bool | None = True if endpoint_only else None
    strata = _config_value(config, ("primary_strata", "strata"), None)

    explicit_filter = _config_value(config, ("primary_filter",), None)
    if isinstance(explicit_filter, Mapping):
        if "condition" in explicit_filter:
            condition = explicit_filter["condition"]
        if "primary_endpoint" in explicit_filter:
            endpoint = explicit_filter["primary_endpoint"]
            endpoint_target = None if endpoint is None else _as_bool(endpoint)
        if "stratum" in explicit_filter:
            strata = explicit_filter["stratum"]

    if condition is None or condition == "*":
        conditions: set[str] | None = None
    elif isinstance(condition, str):
        conditions = {condition.strip().lower()}
    else:
        conditions = {str(item).strip().lower() for item in condition}

    if strata is None or strata == "*":
        allowed_strata: set[str] | None = None
    elif isinstance(strata, str):
        allowed_strata = {strata}
    else:
        allowed_strata = {str(item) for item in strata}

    selected: list[Mapping[str, Any]] = []
    for row in rows:
        row_condition = str(row.get("condition") or "").strip().lower()
        if conditions is not None and row_condition not in conditions:
            continue
        if endpoint_target is not None:
            try:
                is_primary = _as_bool(row.get("primary_endpoint", False))
            except ValueError:
                continue
            if is_primary != endpoint_target:
                continue
        if allowed_strata is not None and str(row.get("stratum") or "") not in allowed_strata:
            continue
        selected.append(row)

    description = {
        "condition": None if conditions is None else sorted(conditions),
        "primary_endpoint": endpoint_target,
        "strata": None if allowed_strata is None else sorted(allowed_strata),
    }
    return selected, description


def _required_id(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if _is_missing(value) or not str(value).strip():
        raise ValueError(f"filtered review row has missing {field}")
    return str(value).strip()


def _score_counts(scores: Iterable[float]) -> dict[str, int]:
    counts = {"tuned": 0, "base": 0, "tie": 0}
    for score in scores:
        if score > 0.5 + 1e-12:
            counts["tuned"] += 1
        elif score < 0.5 - 1e-12:
            counts["base"] += 1
        else:
            counts["tie"] += 1
    return counts


def _optional_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _criterion_statistics(
    rows: Sequence[Mapping[str, Any]],
    criterion: str,
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    randomization_resamples: int,
    exact_max_clusters: int,
    seed: int,
) -> dict[str, Any]:
    counts = {winner: 0 for winner in _WINNERS}
    sample_ratings: dict[str, list[str | None]] = defaultdict(list)
    sample_scores: dict[str, list[float]] = defaultdict(list)
    sample_papers: dict[str, str] = {}
    all_reviewers: set[str] = set()
    evaluable_reviewers: set[str] = set()
    seen_reviews: set[tuple[str, str]] = set()

    for row in rows:
        reviewer_id = _required_id(row, "reviewer_id")
        sample_id = _required_id(row, "sample_id")
        paper_id = _required_id(row, "paper_id")
        review_key = (reviewer_id, sample_id)
        if review_key in seen_reviews:
            raise ValueError(
                f"duplicate review for reviewer_id={reviewer_id!r}, sample_id={sample_id!r}"
            )
        seen_reviews.add(review_key)
        previous_paper = sample_papers.setdefault(sample_id, paper_id)
        if previous_paper != paper_id:
            raise ValueError(f"sample_id {sample_id!r} maps to multiple paper_ids")

        label = _canonical_winner(_winner_value(row, criterion), criterion=criterion)
        counts[label] += 1
        all_reviewers.add(reviewer_id)
        sample_ratings[sample_id].append(None if label == "skip" else label)
        if label == "skip":
            continue
        evaluable_reviewers.add(reviewer_id)
        sample_scores[sample_id].append({"tuned": 1.0, "tie": 0.5, "base": 0.0}[label])

    aggregated_samples = {
        sample_id: float(np.mean(scores))
        for sample_id, scores in sorted(sample_scores.items())
        if scores
    }
    paper_samples: dict[str, list[float]] = defaultdict(list)
    for sample_id, score in aggregated_samples.items():
        paper_samples[sample_papers[sample_id]].append(score)
    paper_scores = {
        paper_id: float(np.mean(scores)) for paper_id, scores in sorted(paper_samples.items())
    }
    sample_missingness_bounds: dict[str, tuple[float, float]] = {}
    for sample_id, ratings in sample_ratings.items():
        if not ratings:
            continue
        observed = sample_scores.get(sample_id, [])
        missing = len(ratings) - len(observed)
        sample_missingness_bounds[sample_id] = (
            float(sum(observed) / len(ratings)),
            float((sum(observed) + missing) / len(ratings)),
        )

    assigned_paper_samples: dict[str, list[str]] = defaultdict(list)
    for sample_id, paper_id in sample_papers.items():
        assigned_paper_samples[paper_id].append(sample_id)
    paper_missingness_lower: dict[str, float] = {}
    paper_missingness_upper: dict[str, float] = {}
    for paper_id, sample_ids in sorted(assigned_paper_samples.items()):
        sample_bounds = [sample_missingness_bounds[sample_id] for sample_id in sample_ids]
        paper_missingness_lower[paper_id] = float(np.mean([bounds[0] for bounds in sample_bounds]))
        paper_missingness_upper[paper_id] = float(np.mean([bounds[1] for bounds in sample_bounds]))
    missingness_bounds = [
        (
            float(np.mean(list(paper_missingness_lower.values())))
            if paper_missingness_lower
            else None
        ),
        (
            float(np.mean(list(paper_missingness_upper.values())))
            if paper_missingness_upper
            else None
        ),
    ]

    estimate = float(np.mean(list(paper_scores.values()))) if paper_scores else math.nan
    ci_low, ci_high = paper_cluster_bootstrap_ci(
        paper_scores,
        confidence_level=confidence_level,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    p_value = cluster_mean_normal_pvalue(paper_scores, null=0.5)
    sharp_null_p_value = cluster_sign_flip_pvalue(
        paper_scores,
        null=0.5,
        n_resamples=randomization_resamples,
        seed=seed,
        exact_max_clusters=exact_max_clusters,
    )
    worst_ci_low, worst_ci_high = paper_cluster_bootstrap_ci(
        paper_missingness_lower,
        confidence_level=confidence_level,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    worst_p_value = cluster_mean_normal_pvalue(paper_missingness_lower, null=0.5)
    alpha = krippendorff_alpha_nominal(sample_ratings)
    sample_counts = _score_counts(aggregated_samples.values())
    paper_counts = _score_counts(paper_scores.values())
    estimate_value = _optional_float(estimate)
    p_value_result = _optional_float(p_value)
    ci = [_optional_float(ci_low), _optional_float(ci_high)]

    return {
        "criterion": criterion,
        "n_reviews": len(rows),
        "n_evaluable_reviews": counts["tuned"] + counts["base"] + counts["tie"],
        "n_skipped_reviews": counts["skip"],
        "n_reviewers": len(evaluable_reviewers),
        "n_reviewers_total": len(all_reviewers),
        "n_samples": len(aggregated_samples),
        "n_assigned_samples": len(sample_papers),
        "n_unevaluable_samples": len(sample_papers) - len(aggregated_samples),
        "n_papers": len(paper_scores),
        "n_assigned_papers": len(assigned_paper_samples),
        "missingness_worst_best_case_bounds": missingness_bounds,
        "missingness_worst_case_bootstrap_ci": [
            _optional_float(worst_ci_low),
            _optional_float(worst_ci_high),
        ],
        "missingness_worst_case_p_value": _optional_float(worst_p_value),
        "estimate": estimate_value,
        "preference_score": estimate_value,
        "lift": None if estimate_value is None else estimate_value - 0.5,
        "lift_from_half": None if estimate_value is None else estimate_value - 0.5,
        "bootstrap_ci": ci,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "confidence_level": confidence_level,
        "p_value": p_value_result,
        "cluster_mean_normal_p_value": p_value_result,
        "randomization_p_value": _optional_float(sharp_null_p_value),
        "sharp_null_sign_flip_p_value": _optional_float(sharp_null_p_value),
        "counts": dict(counts),
        "review_counts": dict(counts),
        "wins": counts["tuned"],
        "losses": counts["base"],
        "ties": counts["tie"],
        "tuned_wins": counts["tuned"],
        "base_wins": counts["base"],
        "tie_count": counts["tie"],
        "skip_count": counts["skip"],
        "sample_counts": sample_counts,
        "sample_win_counts": sample_counts,
        "paper_counts": paper_counts,
        "paper_win_counts": paper_counts,
        "inter_rater_alpha": _optional_float(alpha),
        "sample_scores": dict(sorted(aggregated_samples.items())),
        "paper_scores": dict(sorted(paper_scores.items())),
        "paper_missingness_worst_case_scores": dict(sorted(paper_missingness_lower.items())),
        "paper_missingness_best_case_scores": dict(sorted(paper_missingness_upper.items())),
    }


def _canonical_side(value: Any, *, field: str) -> str:
    if _is_missing(value) or (isinstance(value, str) and not value.strip()):
        return "skip"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"a", "left", "first", "1", "side_a", "variant_a"}:
        return "a"
    if normalized in {"b", "right", "second", "2", "side_b", "variant_b"}:
        return "b"
    if normalized in {"tie", "equal", "same"}:
        return "tie"
    if normalized in {"skip", "none", "missing", "n/a", "na"}:
        return "skip"
    if field == "displayed_preference" and normalized in {"tuned", "base"}:
        return normalized
    raise ValueError(f"invalid {field} value {value!r}")


def _exact_binomial_two_sided(successes: int, total: int) -> float:
    if total <= 0:
        return math.nan
    tail_end = min(successes, total - successes)
    log_two = math.log(2.0)
    probabilities = (
        math.exp(
            math.lgamma(total + 1)
            - math.lgamma(k + 1)
            - math.lgamma(total - k + 1)
            - total * log_two
        )
        for k in range(tail_end + 1)
    )
    return min(1.0, 2.0 * math.fsum(probabilities))


def _position_bias(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    complete = [
        row
        for row in rows
        if not _is_missing(row.get("tuned_side"))
        and not _is_missing(row.get("displayed_preference"))
        and str(row.get("tuned_side")).strip()
        and str(row.get("displayed_preference")).strip()
    ]
    if not complete:
        return None

    selected_counts = {"a": 0, "b": 0, "tie": 0, "skip": 0}
    assignment_counts = {"a": 0, "b": 0}
    tuned_results = {
        "a": {"wins": 0, "evaluable": 0},
        "b": {"wins": 0, "evaluable": 0},
    }
    for row in complete:
        tuned_side = _canonical_side(row["tuned_side"], field="tuned_side")
        if tuned_side not in {"a", "b"}:
            raise ValueError("tuned_side must identify side A/left or B/right")
        displayed = _canonical_side(row["displayed_preference"], field="displayed_preference")
        if displayed == "tuned":
            displayed = tuned_side
        elif displayed == "base":
            displayed = "b" if tuned_side == "a" else "a"
        assignment_counts[tuned_side] += 1
        selected_counts[displayed] += 1
        if displayed in {"a", "b"}:
            tuned_results[tuned_side]["evaluable"] += 1
            if displayed == tuned_side:
                tuned_results[tuned_side]["wins"] += 1

    evaluable = selected_counts["a"] + selected_counts["b"]
    selected_a_rate = selected_counts["a"] / evaluable if evaluable else math.nan
    assignment_total = assignment_counts["a"] + assignment_counts["b"]
    assignment_a_rate = assignment_counts["a"] / assignment_total
    tuned_rate_a = (
        tuned_results["a"]["wins"] / tuned_results["a"]["evaluable"]
        if tuned_results["a"]["evaluable"]
        else math.nan
    )
    tuned_rate_b = (
        tuned_results["b"]["wins"] / tuned_results["b"]["evaluable"]
        if tuned_results["b"]["evaluable"]
        else math.nan
    )
    rate_difference = (
        tuned_rate_a - tuned_rate_b
        if math.isfinite(tuned_rate_a) and math.isfinite(tuned_rate_b)
        else math.nan
    )

    return {
        "n_complete": len(complete),
        "n_evaluable": evaluable,
        "selected_side_counts": selected_counts,
        "assignment_counts": assignment_counts,
        "side_a_preference_rate": _optional_float(selected_a_rate),
        "left_preference_rate": _optional_float(selected_a_rate),
        "position_bias": (None if not math.isfinite(selected_a_rate) else selected_a_rate - 0.5),
        "p_value": _optional_float(_exact_binomial_two_sided(selected_counts["a"], evaluable)),
        "assignment_side_a_rate": assignment_a_rate,
        "assignment_p_value": _exact_binomial_two_sided(assignment_counts["a"], assignment_total),
        "tuned_win_rate_when_side_a": _optional_float(tuned_rate_a),
        "tuned_win_rate_when_side_b": _optional_float(tuned_rate_b),
        "tuned_win_rate_difference_a_minus_b": _optional_float(rate_difference),
    }


def summarize_reviews(
    rows: Iterable[Mapping[str, Any]], config: ReviewStatsConfig | Mapping[str, Any] | Any
) -> dict[str, Any]:
    """Summarize normalized, deblinded paired A/B review rows.

    By default, only rows with ``condition == "original"`` and a true
    ``primary_endpoint`` are analyzed.  Winner values are ``tuned``, ``base``, ``tie``,
    or ``skip`` and can be stored in ``criterion_winners``/``winners``/``criteria`` or in
    flat criterion fields.
    """

    materialized = list(rows)
    if any(not isinstance(row, Mapping) for row in materialized):
        raise TypeError("rows must contain mappings")
    selected, filter_description = _primary_rows(materialized, config)
    discovered = _discover_criteria(selected or materialized)

    primary_criterion = _config_value(
        config, ("primary_criterion", "primary_outcome", "criterion"), None
    )
    if primary_criterion is None or not str(primary_criterion).strip():
        preference_order = (
            "preference",
            "overall",
            "overall_preference",
            "preferred_system",
            "preferred_variant",
            "primary",
        )
        primary_criterion = next(
            (name for name in preference_order if name in discovered),
            discovered[0] if discovered else "preference",
        )
    primary_criterion = str(primary_criterion)

    secondary_value = _config_value(config, ("secondary_criteria",), _MISSING)
    if secondary_value is not _MISSING and secondary_value is not None:
        secondary_criteria = _normalise_criteria_config(secondary_value)
    else:
        configured_criteria = _config_value(config, ("criteria",), None)
        candidates = _normalise_criteria_config(configured_criteria) or discovered
        secondary_criteria = [name for name in candidates if name != primary_criterion]
        if primary_criterion in {"preference", "overall_preference"}:
            secondary_criteria = [
                name
                for name in secondary_criteria
                if name not in {"preference", "overall_preference"}
            ]
    secondary_criteria = list(dict.fromkeys(secondary_criteria))

    confidence_level = float(_config_value(config, ("confidence_level", "ci_level"), 0.95))
    bootstrap_resamples = _config_value(
        config, ("bootstrap_resamples", "bootstrap_iterations", "n_bootstrap"), 10_000
    )
    randomization_resamples = _config_value(
        config,
        ("randomization_resamples", "permutation_resamples", "n_permutations"),
        10_000,
    )
    exact_max_clusters = _config_value(config, ("exact_max_clusters",), 16)
    seed = _config_value(config, ("seed", "random_seed"), 0)

    common = {
        "confidence_level": confidence_level,
        "bootstrap_resamples": bootstrap_resamples,
        "randomization_resamples": randomization_resamples,
        "exact_max_clusters": exact_max_clusters,
        "seed": seed,
    }
    primary = _criterion_statistics(selected, primary_criterion, **common)
    secondary: dict[str, dict[str, Any]] = {}
    for criterion in secondary_criteria:
        secondary[criterion] = _criterion_statistics(selected, criterion, **common)

    raw_secondary_p = {criterion: result["p_value"] for criterion, result in secondary.items()}
    adjusted_secondary_p = holm_correction(raw_secondary_p)
    assert isinstance(adjusted_secondary_p, dict)
    for criterion, result in secondary.items():
        adjusted = adjusted_secondary_p[criterion]
        result["raw_p_value"] = result["p_value"]
        result["holm_adjusted_p_value"] = adjusted
        result["p_value_holm"] = adjusted

    stratum_counts = Counter(str(row.get("stratum") or "") for row in selected)
    summary: dict[str, Any] = {
        "primary_criterion": primary_criterion,
        "primary_filter": filter_description,
        "n_input_rows": len(materialized),
        "n_primary_rows": len(selected),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "primary": primary,
        "position_bias": _position_bias(selected),
        "secondary": secondary,
        "secondary_criteria": secondary_criteria,
    }
    for key in (
        "estimate",
        "preference_score",
        "lift",
        "lift_from_half",
        "bootstrap_ci",
        "ci_low",
        "ci_high",
        "p_value",
        "cluster_mean_normal_p_value",
        "randomization_p_value",
        "sharp_null_sign_flip_p_value",
        "counts",
        "wins",
        "losses",
        "ties",
        "tuned_wins",
        "base_wins",
        "tie_count",
        "skip_count",
        "n_reviews",
        "n_evaluable_reviews",
        "n_reviewers",
        "n_samples",
        "n_assigned_samples",
        "n_unevaluable_samples",
        "n_papers",
        "n_assigned_papers",
        "missingness_worst_best_case_bounds",
        "missingness_worst_case_bootstrap_ci",
        "missingness_worst_case_p_value",
        "inter_rater_alpha",
    ):
        summary[key] = primary[key]
    return summary
