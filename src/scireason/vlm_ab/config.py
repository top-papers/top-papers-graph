# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Configuration loading and validation for the VLM A/B experiment."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .identities import normalize_identity


CONFIG_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or unsafe."""


def _strict_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable SHA256 fingerprint of a validated configuration."""

    return hashlib.sha256(_strict_json(config).encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"{label} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, label: str | None = None) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ExperimentConfigError(f"{label or key} must be a non-empty, trimmed string")
    return value


def _revision(mapping: Mapping[str, Any], key: str, label: str) -> str:
    revision = _text(mapping, key, label)
    if not _COMMIT_RE.fullmatch(revision):
        raise ExperimentConfigError(f"{label} must be an immutable 40-character commit SHA")
    return revision.lower()


def _load_document(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentConfigError(f"cannot read configuration {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExperimentConfigError(f"invalid JSON configuration: {exc}") from exc
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - base project depends on PyYAML
        raise ExperimentConfigError("PyYAML is required for YAML experiment configs") from exc
    try:
        return yaml.safe_load(text)
    except Exception as exc:
        raise ExperimentConfigError(f"invalid YAML configuration: {exc}") from exc


def _validate_model_arm(name: str, value: Any) -> dict[str, Any]:
    arm = _mapping(value, f"models.{name}")
    allowed_arm_fields = {"base_model", "model_kwargs"}
    if name == "tuned":
        allowed_arm_fields |= {"adapter", "adapter_kwargs"}
    unknown = sorted(set(arm) - allowed_arm_fields)
    if unknown:
        raise ExperimentConfigError(f"models.{name} contains unknown fields: {unknown}")
    base = _mapping(arm.get("base_model"), f"models.{name}.base_model")
    unknown_base = sorted(set(base) - {"id", "revision"})
    if unknown_base:
        raise ExperimentConfigError(
            f"models.{name}.base_model contains unknown fields: {unknown_base}"
        )
    base["id"] = _text(base, "id", f"models.{name}.base_model.id")
    base["revision"] = _revision(base, "revision", f"models.{name}.base_model.revision")
    arm["base_model"] = base
    if name == "base":
        if arm.get("adapter"):
            raise ExperimentConfigError("models.base must not contain an adapter")
    else:
        adapter = _mapping(arm.get("adapter"), "models.tuned.adapter")
        unknown_adapter = sorted(
            set(adapter) - {"id", "revision", "adapter_name", "adapter_kwargs"}
        )
        if unknown_adapter:
            raise ExperimentConfigError(
                f"models.tuned.adapter contains unknown fields: {unknown_adapter}"
            )
        adapter["id"] = _text(adapter, "id", "models.tuned.adapter.id")
        adapter["revision"] = _revision(adapter, "revision", "models.tuned.adapter.revision")
        arm["adapter"] = adapter
    return arm


def validate_experiment_config(value: Any) -> dict[str, Any]:
    """Validate and normalize a publication experiment configuration."""

    config = _mapping(value, "configuration")
    version = config.get("schema_version")
    if isinstance(version, bool) or version != CONFIG_VERSION:
        raise ExperimentConfigError(f"schema_version must be {CONFIG_VERSION}")

    experiment = _mapping(config.get("experiment"), "experiment")
    experiment["id"] = _text(experiment, "id", "experiment.id")
    if experiment.get("public_id") is not None:
        experiment["public_id"] = _text(experiment, "public_id", "experiment.public_id")
    seed = experiment.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExperimentConfigError("experiment.seed must be an integer")
    if not isinstance(experiment.get("require_clean_code", False), bool):
        raise ExperimentConfigError("experiment.require_clean_code must be boolean")
    if not isinstance(experiment.get("require_preregistered_plan", False), bool):
        raise ExperimentConfigError("experiment.require_preregistered_plan must be boolean")
    output_dir = _text(experiment, "output_dir", "experiment.output_dir")
    output_path = Path(output_dir)
    if (
        output_path.is_absolute()
        or output_path.drive
        or output_path.anchor
        or ".." in output_path.parts
    ):
        raise ExperimentConfigError("experiment.output_dir must be a safe relative path")
    config["experiment"] = experiment

    benchmark = _mapping(config.get("benchmark"), "benchmark")
    benchmark["repo_id"] = _text(benchmark, "repo_id", "benchmark.repo_id")
    benchmark["revision"] = _revision(benchmark, "revision", "benchmark.revision")
    benchmark["data_file"] = _text(benchmark, "data_file", "benchmark.data_file")
    provenance = benchmark.get("provenance_file")
    if provenance is not None and (not isinstance(provenance, str) or not provenance.strip()):
        raise ExperimentConfigError("benchmark.provenance_file must be a non-empty string")
    if not isinstance(benchmark.get("require_gold", False), bool):
        raise ExperimentConfigError("benchmark.require_gold must be boolean")
    require_complete_provenance = benchmark.get("require_complete_provenance", False)
    if not isinstance(require_complete_provenance, bool):
        raise ExperimentConfigError("benchmark.require_complete_provenance must be boolean")
    if require_complete_provenance and not provenance:
        raise ExperimentConfigError(
            "benchmark.provenance_file is required with complete provenance"
        )
    if not isinstance(benchmark.get("require_split_provenance", False), bool):
        raise ExperimentConfigError("benchmark.require_split_provenance must be boolean")
    config["benchmark"] = benchmark

    training = _mapping(config.get("training_audit", {}), "training_audit")
    sources = training.get("sources", [])
    if not isinstance(sources, list):
        raise ExperimentConfigError("training_audit.sources must be an array")
    normalized_sources: list[dict[str, Any]] = []
    for index, raw in enumerate(sources):
        source = _mapping(raw, f"training_audit.sources[{index}]")
        source["repo_id"] = _text(source, "repo_id", f"training_audit.sources[{index}].repo_id")
        source["repo_type"] = str(source.get("repo_type") or "model")
        if source["repo_type"] not in {"model", "dataset"}:
            raise ExperimentConfigError("training audit repo_type must be model or dataset")
        source["revision"] = _revision(
            source, "revision", f"training_audit.sources[{index}].revision"
        )
        files = source.get("files")
        if (
            not isinstance(files, list)
            or not files
            or not all(isinstance(item, str) and item.strip() for item in files)
        ):
            raise ExperimentConfigError(
                f"training_audit.sources[{index}].files must be a non-empty string array"
            )
        source["files"] = list(files)
        normalized_sources.append(source)
    training["sources"] = normalized_sources
    if not isinstance(training.get("require_lineage_manifest", False), bool):
        raise ExperimentConfigError("training_audit.require_lineage_manifest must be boolean")
    raw_lineage = training.get("lineage_manifest")
    if raw_lineage is not None:
        lineage = _mapping(raw_lineage, "training_audit.lineage_manifest")
        lineage["repo_id"] = _text(lineage, "repo_id", "training_audit.lineage_manifest.repo_id")
        lineage["repo_type"] = str(lineage.get("repo_type") or "model")
        if lineage["repo_type"] not in {"model", "dataset"}:
            raise ExperimentConfigError("training lineage repo_type must be model or dataset")
        lineage["revision"] = _revision(
            lineage, "revision", "training_audit.lineage_manifest.revision"
        )
        lineage["file"] = _text(lineage, "file", "training_audit.lineage_manifest.file")
        training["lineage_manifest"] = lineage
    config["training_audit"] = training

    models = _mapping(config.get("models"), "models")
    models["base"] = _validate_model_arm("base", models.get("base"))
    models["tuned"] = _validate_model_arm("tuned", models.get("tuned"))
    base_identity = (
        models["base"]["base_model"]["id"],
        models["base"]["base_model"]["revision"],
    )
    tuned_identity = (
        models["tuned"]["base_model"]["id"],
        models["tuned"]["base_model"]["revision"],
    )
    if base_identity != tuned_identity:
        raise ExperimentConfigError("both arms must use the same pinned base model")
    if models["base"].get("model_kwargs", {}) != models["tuned"].get("model_kwargs", {}):
        raise ExperimentConfigError("both arms must use identical model_kwargs")
    if training.get("require_lineage_manifest", False) or raw_lineage is not None:
        adapter = models["tuned"]["adapter"]
        adapter_identity = (adapter["id"], adapter["revision"])
        model_source_identities = {
            (source["repo_id"], source["revision"])
            for source in normalized_sources
            if source["repo_type"] == "model"
        }
        if adapter_identity not in model_source_identities:
            raise ExperimentConfigError(
                "required training lineage sources must include the evaluated adapter revision"
            )
        if raw_lineage is not None:
            lineage = training["lineage_manifest"]
            if lineage["repo_type"] != "model" or lineage["repo_id"] != adapter["id"]:
                raise ExperimentConfigError(
                    "training lineage manifest must be published in the evaluated adapter repository"
                )
            lineage_identity = (
                lineage["repo_id"],
                lineage["repo_type"],
                lineage["revision"],
            )
            source_identities = {
                (source["repo_id"], source["repo_type"], source["revision"])
                for source in normalized_sources
            }
            if lineage_identity in source_identities:
                raise ExperimentConfigError(
                    "training lineage manifest must use a distinct immutable attestation revision"
                )
    config["models"] = models

    processor = _mapping(config.get("processor"), "processor")
    processor["id"] = _text(processor, "id", "processor.id")
    processor["revision"] = _revision(processor, "revision", "processor.revision")
    config["processor"] = processor

    generation = _mapping(config.get("generation"), "generation")
    if generation.get("do_sample", False) is not False:
        raise ExperimentConfigError("the primary configuration must set generation.do_sample=false")
    max_tokens = generation.get("max_new_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ExperimentConfigError("generation.max_new_tokens must be a positive integer")
    config["generation"] = generation

    conditions = config.get("conditions", ["original"])
    allowed_conditions = {"original", "text_only", "shuffled_images"}
    if (
        not isinstance(conditions, list)
        or not conditions
        or not all(isinstance(item, str) and item in allowed_conditions for item in conditions)
    ):
        raise ExperimentConfigError(
            f"conditions must be selected from {sorted(allowed_conditions)}"
        )
    if len(conditions) != len(set(conditions)) or "original" not in conditions:
        raise ExperimentConfigError("conditions must be unique and include original")
    config["conditions"] = conditions

    review = _mapping(config.get("review"), "review")
    reviewers = review.get("reviewer_ids")
    if (
        not isinstance(reviewers, list)
        or not reviewers
        or not all(isinstance(item, str) and item.strip() for item in reviewers)
    ):
        raise ExperimentConfigError("review.reviewer_ids must be a non-empty string array")
    normalized_reviewers = [normalize_identity(item, ascii_reviewer=True) for item in reviewers]
    if any(not item for item in normalized_reviewers):
        raise ExperimentConfigError("review.reviewer_ids must be safe ASCII identifiers")
    if len(normalized_reviewers) != len(set(normalized_reviewers)):
        raise ExperimentConfigError("review.reviewer_ids must be normalized-distinct")
    reviews_per_item = review.get("reviews_per_item")
    if (
        isinstance(reviews_per_item, bool)
        or not isinstance(reviews_per_item, int)
        or not 1 <= reviews_per_item <= len(reviewers)
    ):
        raise ExperimentConfigError("review.reviews_per_item must be between 1 and reviewer count")
    config["review"] = review

    statistics = _mapping(config.get("statistics", {}), "statistics")
    primary_strata = statistics.get("primary_strata", ["multimodal_hard", "temporal_hard"])
    allowed_strata = {"multimodal_hard", "temporal_hard", "easy_control"}
    if (
        not isinstance(primary_strata, list)
        or not primary_strata
        or any(not isinstance(item, str) or item not in allowed_strata for item in primary_strata)
        or len(primary_strata) != len(set(primary_strata))
    ):
        raise ExperimentConfigError("statistics.primary_strata must list unique known strata")
    statistics["primary_strata"] = primary_strata
    primary_condition = statistics.get(
        "primary_condition", review.get("primary_condition", "original")
    )
    if primary_condition != review.get("primary_condition", "original"):
        raise ExperimentConfigError("statistics and review primary_condition must match")
    statistics["primary_condition"] = primary_condition
    config["statistics"] = statistics
    power = _mapping(config.get("power", {}), "power")
    n_items = power.get("n_items")
    if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items <= 0:
        raise ExperimentConfigError("power.n_items must be a positive integer")
    power["n_items"] = n_items
    config["power"] = power
    try:
        _strict_json(config)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"configuration is not strict JSON-compatible: {exc}") from exc
    return config


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load YAML/JSON and return a validated plain dictionary."""

    source = Path(path).resolve(strict=True)
    return validate_experiment_config(_load_document(source))


__all__ = [
    "CONFIG_VERSION",
    "ExperimentConfigError",
    "config_fingerprint",
    "load_experiment_config",
    "validate_experiment_config",
]
