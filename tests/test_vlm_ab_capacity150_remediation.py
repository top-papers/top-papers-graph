# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from scireason.vlm_ab.cli import (
    command_aggregate,
    command_blind,
    command_infer,
    command_plan,
    command_prepare,
)
from scireason.vlm_ab.config import (
    config_fingerprint,
    load_experiment_config,
    validate_experiment_config,
)
from scireason.vlm_ab.prepare import PublicationGateError
from scireason.vlm_ab.remediation import _policy_from_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "vlm_ab_evaluation" / "make_capacity150_remediation_config.py"
ARCHIVED_CONFIG = (
    ROOT
    / "experiments"
    / "vlm_ab_evaluation"
    / "configs"
    / "qwen3vl_scireason_remediation_audit_v2.yaml"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_make_capacity150_remediation_config", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archived_config() -> dict[str, Any]:
    return load_experiment_config(ARCHIVED_CONFIG)


def _write_source(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _working_config(module: ModuleType) -> dict[str, Any]:
    config = copy.deepcopy(_archived_config())
    config["experiment"].update(
        {
            "id": "qwen3vl-cap150-remediation-working-v1",
            "public_id": "study-2026-cap150-remediation-working-v1",
            "output_dir": "runs/vlm_ab/qwen3vl-cap150-remediation-working-v1",
            "require_clean_code": False,
            "require_preregistered_plan": False,
        }
    )
    config["power"] = copy.deepcopy(module.CAPACITY_POWER)
    return validate_experiment_config(config)


def test_remediation_config_preserves_archived_protocol_and_reports_working_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    source = tmp_path / "archived.yaml"
    output = tmp_path / "capacity-remediation.yaml"
    original = _archived_config()
    _write_source(source, original)
    output_dir = "runs/vlm_ab/qwen3vl-cap150-remediation-working-v1"

    assert (
        module.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--experiment-id",
                "qwen3vl-cap150-remediation-working-v1",
                "--public-id",
                "study-2026-cap150-remediation-working-v1",
                "--output-dir",
                output_dir,
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    generated = load_experiment_config(output)
    expected = copy.deepcopy(original)
    expected["experiment"].update(
        {
            "id": "qwen3vl-cap150-remediation-working-v1",
            "public_id": "study-2026-cap150-remediation-working-v1",
            "output_dir": output_dir,
            "require_clean_code": False,
            "require_preregistered_plan": False,
        }
    )
    expected["power"] = copy.deepcopy(module.CAPACITY_POWER)
    expected = validate_experiment_config(expected)

    assert output.read_bytes().endswith(b"\n")
    assert b"\r" not in output.read_bytes()
    assert generated == expected
    assert generated["benchmark"] == original["benchmark"]
    assert generated["models"] == original["models"]
    assert generated["training_audit"] == original["training_audit"]
    assert generated["experiment"]["require_clean_code"] is False
    assert generated["experiment"]["require_preregistered_plan"] is False
    assert generated["power"] == module.CAPACITY_POWER
    assert _policy_from_config(generated)["exact_primary_papers"] == 150
    assert summary["config_fingerprint"] == config_fingerprint(generated)
    assert summary["scope"] == "capacity-remediation-working-config"
    assert summary["publication_ready"] is False
    assert summary["exact_primary_papers"] == 150
    assert "power_plan" not in summary
    assert "prepare --exploratory" in summary["allowed_commands"]
    assert "curate-capacity-assist" in summary["allowed_commands"]
    assert "curate-capacity-enrichment" in summary["allowed_commands"]
    assert not (repo_root / output_dir).exists()


def test_remediation_config_rejects_unsafe_paths_overwrite_and_symlinked_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    repo_root = tmp_path / "repo"
    linked_target = repo_root / "linked-target"
    linked_target.mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    source = tmp_path / "archived.yaml"
    _write_source(source, _archived_config())

    with pytest.raises(ValueError, match=".yaml or .yml"):
        module.make_config(
            source,
            tmp_path / "capacity-remediation.json",
            "qwen3vl-cap150-remediation-working-v1",
            "study-2026-cap150-remediation-working-v1",
            "runs/vlm_ab/fresh",
        )
    with pytest.raises(ValueError, match="under runs"):
        module.make_config(
            source,
            tmp_path / "capacity-remediation.yaml",
            "qwen3vl-cap150-remediation-working-v1",
            "study-2026-cap150-remediation-working-v1",
            "output/capacity-remediation",
        )

    existing = tmp_path / "existing.yaml"
    existing.write_text("already exists\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.make_config(
            source,
            existing,
            "qwen3vl-cap150-remediation-working-v1",
            "study-2026-cap150-remediation-working-v1",
            "runs/vlm_ab/fresh",
        )
    assert existing.read_text(encoding="utf-8") == "already exists\n"

    try:
        (repo_root / "runs").symlink_to(linked_target, target_is_directory=True)
    except OSError:
        pytest.skip("creating directory symlinks is unavailable")
    with pytest.raises(FileExistsError, match="existing or symlinked output-dir"):
        module.make_config(
            source,
            tmp_path / "capacity-remediation.yaml",
            "qwen3vl-cap150-remediation-working-v1",
            "study-2026-cap150-remediation-working-v1",
            "runs/vlm_ab/fresh",
        )


def test_remediation_config_rejects_corrected_strict_source_without_blocked_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    source = tmp_path / "corrected-strict.yaml"
    strict = _archived_config()
    strict["benchmark"]["revision"] = "a" * 40
    _write_source(source, strict)

    with pytest.raises(
        ValueError, match="archival blocked benchmark revision|exploratory remediation"
    ):
        module.make_config(
            source,
            tmp_path / "capacity-remediation.yaml",
            "qwen3vl-cap150-remediation-working-v1",
            "study-2026-cap150-remediation-working-v1",
            "runs/vlm_ab/fresh",
        )


def test_remediation_config_accepts_an_existing_exploratory_remediation_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    source = tmp_path / "prior-remediation.yaml"
    prior = _working_config(module)
    prior["benchmark"]["revision"] = "a" * 40
    _write_source(source, prior)

    generated = module.make_config(
        source,
        tmp_path / "capacity-remediation.yaml",
        "qwen3vl-cap150-remediation-working-v2",
        "study-2026-cap150-remediation-working-v2",
        "runs/vlm_ab/fresh",
    )

    assert generated["benchmark"]["revision"] == "a" * 40


@pytest.mark.parametrize(
    ("command", "args"),
    [
        (command_plan, argparse.Namespace()),
        (
            command_prepare,
            argparse.Namespace(
                exploratory=False,
                benchmark_dir=None,
                training_file=[],
                cache_dir=None,
            ),
        ),
        (
            command_infer,
            argparse.Namespace(exploratory=True, arm="base", backend="mock", limit=None),
        ),
        (
            command_blind,
            argparse.Namespace(exploratory=True, reviewers=None, reviews_per_item=None),
        ),
        (
            command_aggregate,
            argparse.Namespace(
                exploratory=True,
                review_file=[],
                reviews_dir=None,
                allow_incomplete=False,
            ),
        ),
    ],
)
def test_remediation_working_config_cannot_start_planning_or_evaluation(
    tmp_path: Path, command, args: argparse.Namespace
) -> None:
    module = _load_script()

    with pytest.raises(PublicationGateError, match="capacity remediation working configs"):
        command(args, _working_config(module), tmp_path)


def test_remediation_command_guard_does_not_trust_cosmetic_ids(tmp_path: Path) -> None:
    config = _working_config(_load_script())
    config["experiment"]["id"] = "renamed-working-study"
    config["experiment"]["public_id"] = "renamed-public-study"
    config["experiment"].pop("require_clean_code")
    config["experiment"].pop("require_preregistered_plan")
    config["experiment"]["output_dir"] = "output/renamed-working-study"
    config["power"]["unused_extension"] = "does-not-disable-guard"

    with pytest.raises(PublicationGateError, match="capacity remediation working configs"):
        command_plan(argparse.Namespace(), config, tmp_path)


def test_remediation_config_script_help() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--experiment-id" in result.stdout
    assert "--public-id" in result.stdout
    assert "--output-dir" in result.stdout
