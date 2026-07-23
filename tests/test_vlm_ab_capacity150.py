# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

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

from scireason.vlm_ab.config import (
    FP16_PRIMARY_MODEL_KWARGS,
    FP32_LORA_ADAPTER_KWARGS,
    config_fingerprint,
    load_experiment_config,
    validate_experiment_config,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "vlm_ab_evaluation" / "make_capacity150_config.py"
BENCHMARK_REVISION = "a" * 40
ADAPTER_REVISION = "b" * 40
LINEAGE_REVISION = "c" * 40
BASE_REVISION = "d" * 40
ASSEMBLY_SHA256 = "e" * 64


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_make_capacity150_config", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corrected_config() -> dict[str, Any]:
    return validate_experiment_config(
        {
            "schema_version": 1,
            "experiment": {
                "id": "corrected-strict-v1",
                "public_id": "corrected-public-v1",
                "seed": 73,
                "require_clean_code": True,
                "require_preregistered_plan": True,
                "precision_mode": "fp16-primary",
                "output_dir": "runs/corrected-strict-v1",
            },
            "benchmark": {
                "repo_id": "example/benchmark",
                "revision": BENCHMARK_REVISION,
                "data_file": "benchmark.jsonl",
                "provenance_file": "provenance.jsonl",
                "assembly_manifest_file": "assembly_manifest.json",
                "assembly_manifest_sha256": ASSEMBLY_SHA256,
                "require_gold": False,
                "require_complete_provenance": True,
            },
            "training_audit": {
                "require_lineage_manifest": True,
                "lineage_manifest": {
                    "repo_id": "example/adapter",
                    "repo_type": "model",
                    "revision": LINEAGE_REVISION,
                    "file": "artifacts/training_lineage_manifest.json",
                },
                "sources": [
                    {
                        "repo_id": "example/adapter",
                        "repo_type": "model",
                        "revision": ADAPTER_REVISION,
                        "files": ["artifacts/data/training.jsonl"],
                    }
                ],
            },
            "models": {
                "base": {
                    "base_model": {"id": "example/base", "revision": BASE_REVISION},
                    "model_kwargs": copy.deepcopy(FP16_PRIMARY_MODEL_KWARGS),
                },
                "tuned": {
                    "base_model": {"id": "example/base", "revision": BASE_REVISION},
                    "adapter": {"id": "example/adapter", "revision": ADAPTER_REVISION},
                    "model_kwargs": copy.deepcopy(FP16_PRIMARY_MODEL_KWARGS),
                    "adapter_kwargs": copy.deepcopy(FP32_LORA_ADAPTER_KWARGS),
                },
            },
            "processor": {"id": "example/adapter", "revision": ADAPTER_REVISION},
            "generation": {"do_sample": False, "max_new_tokens": 64, "use_cache": True},
            "conditions": ["original", "text_only"],
            "review": {
                "reviewer_ids": ["reviewer-1", "reviewer-2"],
                "reviews_per_item": 2,
                "primary_condition": "original",
            },
            "statistics": {"primary_strata": ["multimodal_hard"], "primary_condition": "original"},
            "power": {
                "n_items": 240,
                "reviews_per_item": 2,
                "evaluable_fraction": 0.9,
                "intracluster_correlation": 0.5,
                "alpha": 0.05,
                "target_power": 0.8,
                "score_sd": 0.5,
                "target_effect": 0.1,
            },
        }
    )


def _write_source(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_capacity150_config_preserves_protocol_and_reports_preview(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    source = tmp_path / "corrected.yaml"
    output = tmp_path / "capacity150.yaml"
    original = _corrected_config()
    _write_source(source, original)
    output_dir = f"runs/vlm_ab/capacity-{tmp_path.name}"
    assert not (module.REPO_ROOT / output_dir).exists()

    assert (
        module.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--experiment-id",
                "qwen3vl-cap150-capacity-v1",
                "--public-id",
                "study-2026-cap150-sensitivity-v1",
                "--output-dir",
                output_dir,
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    generated = load_experiment_config(output)

    assert output.read_bytes().endswith(b"\n")
    assert b"\r" not in output.read_bytes()
    assert generated["power"] == module.CAPACITY_POWER
    assert generated["benchmark"]["revision"] == BENCHMARK_REVISION
    assert generated["benchmark"]["assembly_manifest_sha256"] == ASSEMBLY_SHA256
    assert generated["models"]["tuned"]["adapter"]["revision"] == ADAPTER_REVISION
    assert generated["training_audit"]["lineage_manifest"]["revision"] == LINEAGE_REVISION
    unchanged = copy.deepcopy(original)
    for config in (unchanged, generated):
        for field in ("id", "public_id", "output_dir"):
            config["experiment"].pop(field)
        config.pop("power")
    assert generated == unchanged

    assert summary["config_fingerprint"] == config_fingerprint(load_experiment_config(output))
    assert summary["scope"] == "capacity-limited confirmatory protocol"
    assert summary["precision_contract"] == {
        "mode": "fp16-primary",
        "base_compute": "float16",
        "adapter": "native-fp32-peft",
    }
    assert summary["power_preview"]["status"] == "preview_not_preregistered"
    assert summary["power_preview"]["n_items"] == 150
    assert summary["power_preview"]["reviews_per_item"] == 2.0
    assert summary["power_preview"]["evaluable_fraction"] == 0.9
    assert summary["power_preview"]["intracluster_correlation"] == 0.5
    assert summary["power_preview"]["target_effect"] == 0.121
    assert summary["power_preview"]["achieved_power"] == pytest.approx(0.8028444453537316)
    assert summary["power_preview"]["mde"] == pytest.approx(0.1205610321414711)
    assert summary["power_preview"]["achieved_power"] >= 0.8 - module.POWER_TOLERANCE
    assert summary["power_preview"]["mde"] <= 0.121 + module.POWER_TOLERANCE
    assert any("B/R/M" in warning for warning in summary["warnings"])
    assert any("clean" in warning.lower() for warning in summary["warnings"])
    assert not (tmp_path / "power_plan.json").exists()
    assert not (tmp_path / "runs").exists()
    assert not (module.REPO_ROOT / output_dir).exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config, module: config["benchmark"].update(
                revision=module.ARCHIVAL_BLOCKED_BENCHMARK_REVISION
            ),
            "blocked archival",
        ),
        (
            lambda config, module: config["training_audit"].pop("lineage_manifest"),
            "declare a training lineage manifest",
        ),
        (
            lambda config, module: config["benchmark"].pop("assembly_manifest_file"),
            "assembly_manifest_file.*set together",
        ),
        (
            lambda config, module: config["training_audit"]["lineage_manifest"].update(
                revision=ADAPTER_REVISION
            ),
            "distinct immutable attestation revision",
        ),
        (
            lambda config, module: config["training_audit"].update(sources=[]),
            "include the evaluated adapter revision",
        ),
        (
            lambda config, module: config["review"]["reviewer_ids"].append("reviewer-3"),
            "exactly two",
        ),
    ],
)
def test_capacity150_config_rejects_invalid_strict_sources(
    tmp_path: Path, mutate, message: str
) -> None:
    module = _load_script()
    source = tmp_path / "corrected.yaml"
    output = tmp_path / "capacity150.yaml"
    config = _corrected_config()
    mutate(config, module)
    _write_source(source, config)

    with pytest.raises(ValueError, match=message):
        module.make_config(
            source,
            output,
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/qwen3vl-cap150-capacity-v1",
        )
    assert not output.exists()
    assert not (tmp_path / "power_plan.json").exists()


def test_capacity150_config_rejects_invalid_ids_and_overwrite(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "corrected.yaml"
    _write_source(source, _corrected_config())

    with pytest.raises(ValueError, match="cap150"):
        module.make_config(
            source,
            tmp_path / "invalid-id.yaml",
            "qwen3vl-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/qwen3vl-cap150-capacity-v1",
        )
    with pytest.raises(ValueError, match="sensitivity.*capacity"):
        module.make_config(
            source,
            tmp_path / "invalid-public-id.yaml",
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-v1",
            "runs/vlm_ab/qwen3vl-cap150-capacity-v1",
        )

    output = tmp_path / "existing.yaml"
    output.write_text("already exists\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.make_config(
            source,
            output,
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/qwen3vl-cap150-capacity-v1",
        )
    assert output.read_text(encoding="utf-8") == "already exists\n"
    assert not (tmp_path / "power_plan.json").exists()


def test_capacity150_config_rejects_quantized_sources(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "nf4.yaml"
    output = tmp_path / "capacity150.yaml"
    config = _corrected_config()
    model_kwargs = {
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
        }
    }
    for arm in ("base", "tuned"):
        config["models"][arm]["model_kwargs"] = copy.deepcopy(model_kwargs)
    _write_source(source, config)

    with pytest.raises(ValueError, match="exact fp16-primary|unquantized.*quantization_config"):
        module.make_config(
            source,
            output,
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/capacity-quantized-source",
        )
    assert not output.exists()


def test_capacity150_config_rejects_non_fp16_primary_source(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "bf16.yaml"
    config = _corrected_config()
    for arm in ("base", "tuned"):
        config["models"][arm]["model_kwargs"] = {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        }
    _write_source(source, config)

    with pytest.raises(ValueError, match="exact fp16-primary contract"):
        module.make_config(
            source,
            tmp_path / "capacity150.yaml",
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/capacity-bf16-source",
        )


def test_capacity150_config_rejects_legacy_quantization_flags(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "legacy.yaml"
    config = _corrected_config()
    for arm in ("base", "tuned"):
        config["models"][arm]["model_kwargs"] = {"load_in_4bit": True}
    _write_source(source, config)

    with pytest.raises(ValueError, match="legacy top-level quantization"):
        module.make_config(
            source,
            tmp_path / "capacity150.yaml",
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/capacity-legacy-source",
        )


def test_capacity150_config_requires_yaml_and_fresh_run_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    source = tmp_path / "corrected.yaml"
    _write_source(source, _corrected_config())

    with pytest.raises(ValueError, match=".yaml or .yml"):
        module.make_config(
            source,
            tmp_path / "capacity150.json",
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/capacity-output-format",
        )
    with pytest.raises(ValueError, match="under runs"):
        module.make_config(
            source,
            tmp_path / "capacity150.yaml",
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "output/capacity-output-format",
        )

    repo_root = tmp_path / "repo"
    existing_run = repo_root / "runs" / "vlm_ab" / "existing"
    existing_run.mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    with pytest.raises(FileExistsError, match="existing or symlinked output-dir"):
        module.make_config(
            source,
            tmp_path / "existing.yaml",
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/existing",
        )
    assert not (repo_root / "runs" / "vlm_ab" / "new").exists()


def test_capacity150_config_rejects_symlinked_run_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    source = tmp_path / "corrected.yaml"
    _write_source(source, _corrected_config())
    repo_root = tmp_path / "repo"
    linked_target = repo_root / "linked-target"
    linked_target.mkdir(parents=True)
    try:
        (repo_root / "runs").symlink_to(linked_target, target_is_directory=True)
    except OSError:
        pytest.skip("creating directory symlinks is unavailable")
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)

    with pytest.raises(FileExistsError, match="existing or symlinked output-dir"):
        module.make_config(
            source,
            tmp_path / "capacity150.yaml",
            "qwen3vl-cap150-capacity-v1",
            "study-2026-cap150-capacity-v1",
            "runs/vlm_ab/fresh",
        )


def test_capacity150_config_script_help() -> None:
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
