#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Create a capacity-limited N=150 confirmatory config without preregistering it."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scireason.vlm_ab.config import (  # noqa: E402
    FP16_PRIMARY_MODEL_KWARGS,
    config_fingerprint,
    load_experiment_config,
    validate_kaggle_precision_contract,
    validate_experiment_config,
)
from scireason.vlm_ab.capacity import (  # noqa: E402
    ARCHIVAL_BLOCKED_BENCHMARK_REVISION,
    CAPACITY_POWER,
)
from scireason.vlm_ab.stats import power_mde_plan  # noqa: E402


POWER_TOLERANCE = 1e-9
_QUANTIZATION_MODEL_KWARG_KEYS = frozenset(
    {
        "quantization_config",
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit_quant_type",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_use_double_quant",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fresh N=150 capacity-limited confirmatory config without running plan or prepare."
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Corrected strict YAML/JSON config."
    )
    parser.add_argument("--output", type=Path, required=True, help="New YAML path; must not exist.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--public-id", required=True)
    parser.add_argument("--output-dir", required=True, help="Fresh repo-relative run directory.")
    return parser


def _require_capacity_identity(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty, trimmed string")
    lowered = value.lower()
    if "cap150" not in lowered or ("sensitivity" not in lowered and "capacity" not in lowered):
        raise ValueError(
            f"{label} must explicitly contain 'cap150' and either 'sensitivity' or 'capacity'"
        )


def _require_unquantized_model_kwargs(config: dict[str, Any]) -> None:
    for arm in ("base", "tuned"):
        model_kwargs = config["models"][arm].get("model_kwargs", {})
        quantization_keys = sorted(_QUANTIZATION_MODEL_KWARG_KEYS & set(model_kwargs))
        if quantization_keys:
            raise ValueError(
                f"source models.{arm}.model_kwargs must be unquantized; found {quantization_keys}"
            )


def _require_corrected_strict_source(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    if experiment.get("require_clean_code") is not True:
        raise ValueError("source strict config must set experiment.require_clean_code=true")
    if experiment.get("require_preregistered_plan") is not True:
        raise ValueError("source strict config must set experiment.require_preregistered_plan=true")

    if config["benchmark"]["revision"] == ARCHIVAL_BLOCKED_BENCHMARK_REVISION:
        raise ValueError("source benchmark revision is the blocked archival revision")
    if not config["benchmark"].get("assembly_manifest_file") or not config["benchmark"].get(
        "assembly_manifest_sha256"
    ):
        raise ValueError("source benchmark must pin the reviewed Phase 0 assembly manifest")

    training = config["training_audit"]
    if training.get("require_lineage_manifest") is not True:
        raise ValueError("source strict config must require a training lineage manifest")
    lineage = training.get("lineage_manifest")
    if not isinstance(lineage, dict):
        raise ValueError("source strict config must declare a training lineage manifest")

    adapter = config["models"]["tuned"]["adapter"]
    if lineage.get("repo_type") != "model":
        raise ValueError("training lineage manifest must be in a model repository")
    if lineage.get("repo_id") != adapter["id"]:
        raise ValueError("training lineage manifest must be in the evaluated adapter repository")
    if lineage.get("revision") == adapter["revision"]:
        raise ValueError(
            "training lineage manifest revision M must differ from evaluated adapter R"
        )

    has_adapter_source = any(
        source["repo_type"] == "model"
        and source["repo_id"] == adapter["id"]
        and source["revision"] == adapter["revision"]
        for source in training["sources"]
    )
    if not has_adapter_source:
        raise ValueError("training_audit.sources must include the evaluated adapter revision R")
    if (
        len(config["review"]["reviewer_ids"]) != 2
        or config["review"]["reviews_per_item"] != CAPACITY_POWER["reviews_per_item"]
    ):
        raise ValueError(
            "source review must contain exactly two reviewer IDs and two reviews per item"
        )
    _require_unquantized_model_kwargs(config)
    validate_kaggle_precision_contract(config, "fp16-primary", require_capacity=False)


def _validate_output(output: Path) -> None:
    if output.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("output must have a .yaml or .yml extension")


def _validate_output_dir(output_dir: str) -> None:
    if (
        not isinstance(output_dir, str)
        or not output_dir.strip()
        or output_dir != output_dir.strip()
    ):
        raise ValueError("output-dir must be a non-empty, trimmed string")
    relative = Path(output_dir)
    if (
        relative.is_absolute()
        or relative.drive
        or relative.anchor
        or ".." in relative.parts
        or len(relative.parts) < 2
        or relative.parts[0] != "runs"
    ):
        raise ValueError("output-dir must be a safe relative path under runs/ and not runs itself")
    repo_root = REPO_ROOT.resolve()
    target = repo_root / relative
    try:
        resolved = target.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve output-dir: {output_dir}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("output-dir must resolve under the repository") from exc
    for index in range(1, len(relative.parts)):
        if repo_root.joinpath(*relative.parts[:index]).is_symlink():
            raise FileExistsError(f"refusing existing or symlinked output-dir: {output_dir}")
    if target.exists() or target.is_symlink() or resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing existing or symlinked output-dir: {output_dir}")


def _power_preview(config: dict[str, Any]) -> dict[str, Any]:
    power = config["power"]
    preview = power_mde_plan(
        power["n_items"],
        power["reviews_per_item"],
        evaluable_fraction=power["evaluable_fraction"],
        intracluster_correlation=power["intracluster_correlation"],
        alpha=power["alpha"],
        target_power=power["target_power"],
        score_sd=power["score_sd"],
        target_effect=power["target_effect"],
    )
    achieved_power = preview["achieved_power"]
    if achieved_power is None or achieved_power + POWER_TOLERANCE < CAPACITY_POWER["target_power"]:
        raise RuntimeError("capacity-limited power preview is below the required target power")
    if preview["mde"] > CAPACITY_POWER["target_effect"] + POWER_TOLERANCE:
        raise RuntimeError("capacity-limited power preview exceeds the required target effect")
    return preview


def _write_yaml_exclusive(output: Path, config: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, line_break="\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    created = False
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            created = True
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if created and output.exists():
            output.unlink()
        raise


def make_config(
    source: Path,
    output: Path,
    experiment_id: str,
    public_id: str,
    output_dir: str,
) -> dict[str, Any]:
    """Build, validate, power-check, and exclusively write a capacity-150 config."""

    _require_capacity_identity("experiment-id", experiment_id)
    _require_capacity_identity("public-id", public_id)
    _validate_output(output)
    _validate_output_dir(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite config: {output}")

    config = copy.deepcopy(load_experiment_config(source))
    _require_corrected_strict_source(config)
    config["experiment"]["id"] = experiment_id
    config["experiment"]["public_id"] = public_id
    config["experiment"]["output_dir"] = output_dir
    config["power"] = copy.deepcopy(CAPACITY_POWER)
    config = validate_experiment_config(config)
    validate_kaggle_precision_contract(config, "fp16-primary")
    _power_preview(config)
    _write_yaml_exclusive(output, config)
    return config


def build_summary(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Return the non-preregistered local preview emitted by the command line interface."""

    return {
        "config": str(output),
        "config_fingerprint": config_fingerprint(config),
        "scope": "capacity-limited confirmatory protocol",
        "precision_contract": {
            "mode": "fp16-primary",
            "base_compute": FP16_PRIMARY_MODEL_KWARGS["torch_dtype"],
            "adapter": "native-fp32-peft",
        },
        "power_preview": {
            "status": "preview_not_preregistered",
            **_power_preview(config),
        },
        "warnings": [
            "Corrected immutable B/R/M revisions are required before plan.",
            "A clean committed evaluation source tree is required before plan.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = make_config(
            args.input,
            args.output,
            args.experiment_id,
            args.public_id,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            build_summary(config, args.output),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
