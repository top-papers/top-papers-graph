#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Create a fresh N=150 exploratory remediation working configuration."""

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

from scireason.vlm_ab.capacity import (  # noqa: E402
    ARCHIVAL_BLOCKED_BENCHMARK_REVISION,
    CAPACITY_POWER,
    has_capacity_remediation_identity,
    is_capacity_remediation_working_config,
)
from scireason.vlm_ab.config import (  # noqa: E402
    config_fingerprint,
    load_experiment_config,
    validate_experiment_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fresh N=150 exploratory remediation config without running plan or prepare."
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Archived remediation YAML/JSON config."
    )
    parser.add_argument("--output", type=Path, required=True, help="New YAML path; must not exist.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--public-id", required=True)
    parser.add_argument("--output-dir", required=True, help="Fresh repo-relative run directory.")
    return parser


def _require_remediation_identity(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty, trimmed string")
    if not has_capacity_remediation_identity(value):
        raise ValueError(f"{label} must explicitly contain both 'cap150' and 'remediation'")


def _require_archival_or_exploratory_remediation_source(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    if config["benchmark"]["revision"] == ARCHIVAL_BLOCKED_BENCHMARK_REVISION:
        return
    if (
        "remediation" in experiment["id"].lower()
        and experiment.get("require_clean_code") is False
        and experiment.get("require_preregistered_plan") is False
    ):
        return
    raise ValueError(
        "source must use the archival blocked benchmark revision or be an exploratory remediation "
        "config; corrected strict configs are not accepted"
    )


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
    """Build and exclusively write the non-publication remediation configuration."""

    _require_remediation_identity("experiment-id", experiment_id)
    _require_remediation_identity("public-id", public_id)
    _validate_output(output)
    _validate_output_dir(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite config: {output}")

    config = copy.deepcopy(load_experiment_config(source))
    _require_archival_or_exploratory_remediation_source(config)
    config["experiment"].update(
        {
            "id": experiment_id,
            "public_id": public_id,
            "output_dir": output_dir,
            "require_clean_code": False,
            "require_preregistered_plan": False,
        }
    )
    config["power"] = copy.deepcopy(CAPACITY_POWER)
    config = validate_experiment_config(config)
    if not is_capacity_remediation_working_config(config):
        raise ValueError("generated configuration is not a capacity remediation working config")
    _write_yaml_exclusive(output, config)
    return config


def build_summary(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Return the intentionally non-preregistered remediation preview."""

    return {
        "config": str(output),
        "config_fingerprint": config_fingerprint(config),
        "scope": "capacity-remediation-working-config",
        "publication_ready": False,
        "exact_primary_papers": CAPACITY_POWER["n_items"],
        "allowed_commands": [
            "prepare --exploratory",
            "curate-queue",
            "curate-forms",
            "curate-triage",
            "curate-capacity-plan",
            "curate-capacity-assist",
            "curate-capacity-enrichment",
            "curate-assemble",
        ],
        "warnings": [
            "This working config is not a preregistration and creates no power plan.",
            "Publish corrected B, then R/M, before generating the final strict capacity config.",
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
