# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import builtins
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

import scireason.vlm_ab.inference as inference_module
from scireason.vlm_ab.config import (
    FP16_PRIMARY_MODEL_KWARGS,
    FP32_LORA_ADAPTER_KWARGS,
    NF4_SENSITIVITY_MODEL_KWARGS,
    config_fingerprint,
    validate_experiment_config,
    validate_kaggle_precision_contract,
)
from scireason.vlm_ab.inference import _scireason_schema_errors


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "experiments" / "vlm_ab_evaluation" / "kaggle"
REVISION_A = "a" * 40
REVISION_B = "b" * 40
REVISION_C = "c" * 40


def _load_script(name: str) -> ModuleType:
    path = KAGGLE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return validate_experiment_config(
        {
            "schema_version": 1,
            "experiment": {
                "id": "corrected-strict",
                "public_id": "corrected-public",
                "seed": 73,
                "require_clean_code": True,
                "require_preregistered_plan": True,
                "precision_mode": "fp16-primary",
                "output_dir": "runs/corrected-strict",
            },
            "benchmark": {
                "repo_id": "example/benchmark",
                "revision": REVISION_A,
                "data_file": "benchmark.jsonl",
                "assembly_manifest_file": "assembly_manifest.json",
                "assembly_manifest_sha256": "d" * 64,
                "require_gold": False,
            },
            "training_audit": {
                "require_lineage_manifest": True,
                "lineage_manifest": {
                    "repo_id": "example/adapter",
                    "repo_type": "model",
                    "revision": REVISION_C,
                    "file": "artifacts/training_lineage_manifest.json",
                },
                "sources": [
                    {
                        "repo_id": "example/adapter",
                        "repo_type": "model",
                        "revision": REVISION_B,
                        "files": ["artifacts/data/training.jsonl"],
                    }
                ],
            },
            "models": {
                "base": {
                    "base_model": {"id": "example/base", "revision": REVISION_A},
                    "model_kwargs": copy.deepcopy(FP16_PRIMARY_MODEL_KWARGS),
                },
                "tuned": {
                    "base_model": {"id": "example/base", "revision": REVISION_A},
                    "adapter": {"id": "example/adapter", "revision": REVISION_B},
                    "model_kwargs": copy.deepcopy(FP16_PRIMARY_MODEL_KWARGS),
                    "adapter_kwargs": copy.deepcopy(FP32_LORA_ADAPTER_KWARGS),
                },
            },
            "processor": {"id": "example/processor", "revision": REVISION_B},
            "generation": {"do_sample": False, "max_new_tokens": 64},
            "conditions": ["original", "text_only"],
            "review": {
                "reviewer_ids": ["reviewer-1", "reviewer-2"],
                "reviews_per_item": 2,
                "primary_condition": "original",
            },
            "statistics": {"primary_strata": ["multimodal_hard"]},
            "power": {
                "n_items": 150,
                "require_exact_n_items": True,
                "reviews_per_item": 2,
            },
        }
    )


def test_nf4_config_is_fresh_identical_and_write_once(tmp_path: Path) -> None:
    module = _load_script("make_nf4_config")
    source = tmp_path / "corrected.yaml"
    output = tmp_path / "nf4.yaml"
    original = _config()
    source.write_text(yaml.safe_dump(original, sort_keys=False), encoding="utf-8")

    generated = module.make_config(
        source,
        output,
        "corrected-nf4-sensitivity-v1",
        "public-nf4-sensitivity-v1",
        "runs/corrected-nf4-sensitivity-v1",
    )
    assert output.read_bytes().endswith(b"\n")
    assert b"\r\n" not in output.read_bytes()
    assert (
        generated["models"]["base"]["model_kwargs"] == generated["models"]["tuned"]["model_kwargs"]
    )
    kwargs = generated["models"]["base"]["model_kwargs"]
    assert kwargs["torch_dtype"] == "float16"
    assert kwargs["device_map"] == "balanced"
    assert kwargs["quantization_config"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    }
    unchanged = copy.deepcopy(original)
    for config in (unchanged, generated):
        config["experiment"].pop("id")
        config["experiment"].pop("public_id")
        config["experiment"].pop("output_dir")
        config["experiment"].pop("precision_mode")
        config["models"]["base"].pop("model_kwargs")
        config["models"]["tuned"].pop("model_kwargs")
    assert generated == unchanged
    with pytest.raises(FileExistsError):
        module.make_config(
            source,
            output,
            "other-nf4-sensitivity",
            "other-public-nf4-sensitivity",
            "runs/other",
        )


def test_nf4_config_requires_sensitivity_identity(tmp_path: Path) -> None:
    module = _load_script("make_nf4_config")
    source = tmp_path / "corrected.yaml"
    source.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    with pytest.raises(ValueError, match="sensitivity"):
        module.make_config(source, tmp_path / "out.yaml", "nf4-only", "nf4-public", "runs/x")


def test_nf4_config_accepts_capacity_source(tmp_path: Path) -> None:
    module = _load_script("make_nf4_config")
    source = tmp_path / "capacity.yaml"
    output = tmp_path / "nf4.yaml"
    config = _config()
    config["power"].update({"n_items": 150, "require_exact_n_items": True})
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    generated = module.make_config(
        source,
        output,
        "capacity-nf4-sensitivity-v1",
        "capacity-public-nf4-sensitivity-v1",
        "runs/capacity-nf4-sensitivity-v1",
    )

    assert generated["power"]["require_exact_n_items"] is True
    assert generated["experiment"]["precision_mode"] == "nf4-sensitivity"
    assert "quantization_config" in generated["models"]["base"]["model_kwargs"]
    assert validate_kaggle_precision_contract(generated, "nf4-sensitivity") == "nf4-sensitivity"


@pytest.mark.parametrize("declared_mode", [None, "fp16-primary"])
def test_nf4_kwargs_cannot_omit_or_mislabel_precision_mode(declared_mode: str | None) -> None:
    config = _config()
    if declared_mode is None:
        config["experiment"].pop("precision_mode")
    else:
        config["experiment"]["precision_mode"] = declared_mode
    for arm_name in ("base", "tuned"):
        config["models"][arm_name]["model_kwargs"] = copy.deepcopy(
            NF4_SENSITIVITY_MODEL_KWARGS
        )

    with pytest.raises(ValueError, match="precision_mode|exact fp16-primary"):
        validate_experiment_config(config)


@pytest.mark.parametrize("flag", ["require_clean_code", "require_preregistered_plan"])
def test_precision_contract_requires_publication_provenance_flags(flag: str) -> None:
    config = _config()
    config["experiment"][flag] = False

    with pytest.raises(ValueError, match="require_clean_code=true"):
        validate_experiment_config(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config, _module: config["experiment"].update(require_clean_code=False),
            "require_clean_code",
        ),
        (
            lambda config, _module: config["experiment"].update(require_preregistered_plan=False),
            "require_preregistered_plan",
        ),
        (
            lambda config, module: config["benchmark"].update(
                revision=module.ARCHIVAL_BLOCKED_BENCHMARK_REVISION
            ),
            "blocked archival",
        ),
        (
            lambda config, _module: config["training_audit"].update(require_lineage_manifest=False),
            "require a training lineage manifest",
        ),
        (
            lambda config, _module: config["training_audit"]["lineage_manifest"].update(
                revision=REVISION_B
            ),
            "distinct immutable attestation revision",
        ),
        (
            lambda config, _module: config["training_audit"].update(sources=[]),
            "include the evaluated adapter revision",
        ),
        (
            lambda config, _module: config["review"]["reviewer_ids"].append("reviewer-3"),
            "exactly two",
        ),
    ],
)
def test_nf4_config_rejects_non_strict_sources(tmp_path: Path, mutate, message: str) -> None:
    module = _load_script("make_nf4_config")
    source = tmp_path / "source.yaml"
    config = _config()
    mutate(config, module)
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.make_config(
            source,
            tmp_path / "nf4.yaml",
            "corrected-nf4-sensitivity-v1",
            "public-nf4-sensitivity-v1",
            "runs/corrected-nf4-sensitivity-v1",
        )


def test_nf4_config_rejects_existing_quantization(tmp_path: Path) -> None:
    module = _load_script("make_nf4_config")
    source = tmp_path / "nf4-source.yaml"
    config = _config()
    for arm in ("base", "tuned"):
        config["models"][arm]["model_kwargs"] = copy.deepcopy(module.NF4_MODEL_KWARGS)
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="exact fp16-primary|unquantized.*quantization_config"):
        module.make_config(
            source,
            tmp_path / "nf4.yaml",
            "corrected-nf4-sensitivity-v1",
            "public-nf4-sensitivity-v1",
            "runs/corrected-nf4-sensitivity-v1",
        )


def test_nf4_config_requires_yaml_and_a_fresh_safe_run_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("make_nf4_config")
    source = tmp_path / "corrected.yaml"
    source.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=".yaml or .yml"):
        module.make_config(
            source,
            tmp_path / "nf4.json",
            "corrected-nf4-sensitivity-v1",
            "public-nf4-sensitivity-v1",
            "runs/nf4-json-output",
        )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    for output_dir in ("runs", "output/nf4", "runs/../outside"):
        with pytest.raises(ValueError, match="under runs"):
            module.make_config(
                source,
                tmp_path / "nf4.yaml",
                "corrected-nf4-sensitivity-v1",
                "public-nf4-sensitivity-v1",
                output_dir,
            )

    existing_run = repo_root / "runs" / "existing"
    existing_run.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="existing or symlinked output-dir"):
        module.make_config(
            source,
            tmp_path / "nf4.yaml",
            "corrected-nf4-sensitivity-v1",
            "public-nf4-sensitivity-v1",
            "runs/existing",
        )
    assert not (repo_root / "runs" / "fresh").exists()


def test_payload_is_minimal_private_hashed_and_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("build_payload")
    repo = tmp_path / "repo"
    config_path = repo / "experiments/vlm_ab_evaluation/configs/nf4.yaml"
    run = repo / "runs/nf4"
    config_path.parent.mkdir(parents=True)
    run.mkdir(parents=True)
    config = _config()
    config["experiment"].update(
        {
            "id": "nf4-sensitivity",
            "public_id": "public-nf4-sensitivity",
            "output_dir": "runs/nf4",
        }
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (repo / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (repo / "arbitrary-secret.txt").write_text("never copy", encoding="utf-8")
    (repo / "kaggle.json").write_text('{"key":"secret"}\n', encoding="utf-8")
    (run / "inputs/source").mkdir(parents=True)
    (run / "inputs/dataset/assets/images").mkdir(parents=True)
    (run / "audit").mkdir()
    benchmark = run / "inputs/source/benchmark.jsonl"
    frozen = run / "inputs/frozen.jsonl"
    image = run / "inputs/dataset/assets/images/page.png"
    audit_json = run / "audit/benchmark_audit.json"
    audit_markdown = run / "audit/benchmark_audit.md"
    adapter_attestation = run / "inputs/source/adapter_checkpoint_attestation.json"
    benchmark.write_text("{}\n", encoding="utf-8")
    frozen.write_text("{}\n", encoding="utf-8")
    image.write_bytes(b"audited-image")
    audit = {
        "publication_ready": True,
        "per_sample_findings": {
            "sample-1": {
                "image_hashes": [
                    {
                        "path": "assets/images/page.png",
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    }
                ]
            }
        },
    }
    audit_json.write_text(json.dumps(audit), encoding="utf-8")
    audit_markdown.write_text("# audit\n", encoding="utf-8")
    adapter_attestation.write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "repo_id": config["models"]["tuned"]["adapter"]["id"],
                "revision": config["models"]["tuned"]["adapter"]["revision"],
                "filename": "adapter_model.safetensors",
                "size": 8,
                "sha256": "d" * 64,
                "tensor_count": 2,
                "lora_module_count": 1,
                "tensor_dtype": "F32",
                "tensor_inventory_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )
    (run / "secret.txt").write_text("do not upload", encoding="utf-8")
    (run / "predictions").mkdir()
    (run / "predictions/base.jsonl").write_text("partial\n", encoding="utf-8")

    source_paths = [config_path, repo / "pyproject.toml"]
    files = [
        {
            "path": path.relative_to(repo).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in source_paths
    ]
    provenance = {"source_fingerprint": "c" * 64, "files": files}
    artifact_paths = {
        "dataset_root": "inputs/dataset",
        "benchmark_file": "inputs/source/benchmark.jsonl",
        "provenance_file": None,
        "frozen_benchmark": "inputs/frozen.jsonl",
        "audit_json": "audit/benchmark_audit.json",
        "audit_markdown": "audit/benchmark_audit.md",
        "power_plan": None,
        "training_lineage_manifest": None,
        "adapter_config": None,
        "adapter_checkpoint_attestation": "inputs/source/adapter_checkpoint_attestation.json",
    }
    manifest = {
        "config_fingerprint": config_fingerprint(validate_experiment_config(config)),
        "code_provenance": provenance,
        "publication_ready": True,
        "exploratory": False,
        "result_scope": "publication_ready",
        "artifact_paths": artifact_paths,
        "benchmark_file_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "provenance_file_sha256": None,
        "frozen_benchmark_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "audit_json_sha256": hashlib.sha256(audit_json.read_bytes()).hexdigest(),
        "audit_markdown_sha256": hashlib.sha256(audit_markdown.read_bytes()).hexdigest(),
        "power_plan_sha256": None,
        "training_lineage_manifest_sha256": None,
        "adapter_config_sha256": None,
        "adapter_checkpoint_attestation_sha256": hashlib.sha256(
            adapter_attestation.read_bytes()
        ).hexdigest(),
        "training_files": [],
    }
    manifest_path = run / "prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_bytes = manifest_path.read_bytes()
    monkeypatch.setattr(module, "code_provenance", lambda _root: provenance)
    monkeypatch.setattr(module, "verify_prepare_manifest", lambda *_args: None)

    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()
    raced_destination = staging_parent / "raced-input"

    def replace_manifest_after_validation(*_args):
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"publication_ready": True}

    monkeypatch.setattr(module, "verify_prepared_audit", replace_manifest_after_validation)
    with pytest.raises(ValueError, match="changed while it was copied"):
        module.build_payload(
            repo,
            config_path,
            run,
            raced_destination,
            "account",
            "nf4-sensitivity-input",
            "fp16-primary",
        )
    assert not raced_destination.exists()

    manifest_path.write_bytes(manifest_bytes)
    monkeypatch.setattr(module, "verify_prepared_audit", lambda *_args: {"publication_ready": True})
    destination = staging_parent / "input"
    module.build_payload(
        repo,
        config_path,
        run,
        destination,
        "account",
        "nf4-sensitivity-input",
        "fp16-primary",
    )
    metadata = json.loads((destination / "dataset-metadata.json").read_text(encoding="utf-8"))
    payload = json.loads((destination / "payload_manifest.json").read_text(encoding="utf-8"))
    assert metadata["isPrivate"] is True
    assert payload["dataset_slug"] == "account/nf4-sensitivity-input"
    assert payload["workflow_mode"] == "fp16-primary"
    assert payload["config_fingerprint"] == manifest["config_fingerprint"]
    assert not (destination / "payload/top-papers-graph/arbitrary-secret.txt").exists()
    assert not (destination / "payload/top-papers-graph/kaggle.json").exists()
    copied_run = destination / "payload/top-papers-graph/runs/nf4"
    assert (copied_run / "prepare_manifest.json").is_file()
    assert (copied_run / "inputs/dataset/assets/images/page.png").is_file()
    assert not (copied_run / "secret.txt").exists()
    assert not (copied_run / "predictions").exists()
    for entry in payload["files"]:
        path = destination / "payload" / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    with pytest.raises(FileExistsError):
        module.build_payload(
            repo,
            config_path,
            run,
            destination,
            "account",
            "nf4-sensitivity-input",
            "fp16-primary",
        )


def _render_template(
    tmp_path: Path,
    *,
    arm: str = "base",
    payload_slug: str = "owner/input",
    state_slug: str = "",
    workflow_mode: str = "fp16-primary",
    payload_manifest_sha256: str = "a" * 64,
    state_manifest_sha256: str | None = None,
) -> dict:
    text = (KAGGLE / "kernel_runner.py.template").read_text(encoding="utf-8")
    replacements = {
        "__ARM_JSON__": json.dumps(arm),
        "__WORKFLOW_MODE_JSON__": json.dumps(workflow_mode),
        "__PAYLOAD_DATASET_SLUG_JSON__": json.dumps(payload_slug),
        "__STATE_DATASET_SLUG_JSON__": json.dumps(state_slug),
        "__PAYLOAD_MANIFEST_SHA256_JSON__": json.dumps(payload_manifest_sha256),
        "__STATE_MANIFEST_SHA256_JSON__": json.dumps(
            state_manifest_sha256 if state_manifest_sha256 is not None else ("b" * 64 if state_slug else "")
        ),
        "__EXECUTION_LOCK_SHA256_JSON__": json.dumps("e" * 64),
        "__CONFIG_RELATIVE_JSON__": '"experiments/config.yaml"',
        "__OUTPUT_DATASET_SLUG__": "owner/state",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    compile(text, str(tmp_path / "runner.py"), "exec")
    namespace = {"__name__": "rendered_runner"}
    exec(text, namespace)
    namespace["MATERIALIZED_ROOT"] = tmp_path / "materialized"
    return namespace


def _entry(path: str, data: bytes) -> dict:
    return {
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _complete_prediction(
    arm: str = "base",
    *,
    protocol_fingerprint: str = "c" * 64,
    config_fingerprint: str = "d" * 64,
) -> tuple[dict, dict[tuple[str, str], dict]]:
    parsed_response: dict = {}
    schema_errors = _scireason_schema_errors(parsed_response)
    expected = {
        "sample_id": "sample-1",
        "paper_id": "paper-1",
        "condition": "original",
        "input_image_hashes": [],
        "shuffle_source_paper_id": None,
        "condition_input_fingerprint": "1" * 64,
    }
    row = {
        "record_version": 1,
        "sample_id": expected["sample_id"],
        "paper_id": expected["paper_id"],
        "condition": expected["condition"],
        "arm": arm,
        "backend": "transformers",
        "status": "success",
        "error": None,
        "runtime_seconds": 0.0,
        "raw_response": "{}",
        "parsed_response": parsed_response,
        "parse_valid": True,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "input_image_hashes": expected["input_image_hashes"],
        "input_image_count": 0,
        "shuffle_source_paper_id": expected["shuffle_source_paper_id"],
        "condition_input_fingerprint": expected["condition_input_fingerprint"],
        "protocol_fingerprint": protocol_fingerprint,
        "config_fingerprint": config_fingerprint,
    }
    return row, {("sample-1", "original"): expected}


def test_runner_safely_materializes_kaggle_zip_layouts(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    payload_dataset = tmp_path / "payload-dataset"
    payload_dataset.mkdir()
    payload_data = b"source"
    payload_manifest = {
        "dataset_slug": "owner/input",
        "files": [_entry("top-papers-graph/source.txt", payload_data)],
    }
    payload_manifest_path = payload_dataset / "payload_manifest.json"
    payload_manifest_path.write_text(json.dumps(payload_manifest), encoding="utf-8")
    with zipfile.ZipFile(payload_dataset / "payload.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("top-papers-graph/source.txt", payload_data)
    materialized = runner["materialize_payload"](payload_manifest_path, payload_manifest)
    assert (materialized / "top-papers-graph/source.txt").read_bytes() == payload_data

    state_dataset = tmp_path / "state-dataset"
    state_dataset.mkdir()
    prediction = b'{"answer":"ok"}\n'
    runtime_log = b"diagnostic\n"
    state_manifest = {
        "dataset_slug": "owner/base-state",
        "files": [
            _entry("predictions/base.jsonl", prediction),
            _entry("runtime/error.log", runtime_log),
        ],
    }
    state_manifest_path = state_dataset / "state_manifest.json"
    state_manifest_path.write_text(json.dumps(state_manifest), encoding="utf-8")
    with zipfile.ZipFile(state_dataset / "predictions.zip", "w") as archive:
        archive.writestr("base.jsonl", prediction)
    with zipfile.ZipFile(state_dataset / "runtime.zip", "w") as archive:
        archive.writestr("error.log", runtime_log)
    state_root = runner["materialize_state"](state_manifest_path, state_manifest)
    assert (state_root / "predictions/base.jsonl").read_bytes() == prediction
    assert (state_root / "runtime/error.log").read_bytes() == runtime_log


def test_runner_supports_unpacked_payload_layout(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    dataset = tmp_path / "dataset"
    source = dataset / "payload/top-papers-graph/source.txt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    manifest = {
        "dataset_slug": "owner/input",
        "files": [_entry("top-papers-graph/source.txt", b"source")],
    }
    manifest_path = dataset / "payload_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    materialized = runner["materialize_payload"](manifest_path, manifest)
    assert (materialized / "top-papers-graph/source.txt").read_bytes() == b"source"


def test_runner_rejects_unsafe_or_ambiguous_payload_archives(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = {"dataset_slug": "owner/input", "files": [_entry("safe.txt", b"safe")]}
    manifest_path = dataset / "payload_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with zipfile.ZipFile(dataset / "payload.zip", "w") as archive:
        archive.writestr("safe.txt", b"safe")
        archive.writestr("../safe.txt", b"safe")
    with pytest.raises(RuntimeError, match="unsafe"):
        runner["materialize_payload"](manifest_path, manifest)
    assert not (runner["MATERIALIZED_ROOT"] / "payload/safe.txt").exists()

    (dataset / "payload").mkdir()
    runner["MATERIALIZED_ROOT"] = tmp_path / "materialized-ambiguous"
    with pytest.raises(RuntimeError, match="ambiguous"):
        runner["materialize_payload"](manifest_path, manifest)


def test_runner_rejects_manifest_file_ancestor_collisions(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    manifest = {
        "files": [
            _entry("collision", b"file"),
            _entry("collision/child.txt", b"child"),
        ]
    }

    with pytest.raises(RuntimeError, match="also an ancestor"):
        runner["inventory_map"](manifest)


def test_runner_rejects_aggregate_zip_bomb_split_across_small_files(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data = b"0" * (700 * 1024)
    manifest = {
        "dataset_slug": "owner/input",
        "files": [_entry("first.bin", data), _entry("second.bin", data)],
    }
    manifest_path = dataset / "payload_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with zipfile.ZipFile(dataset / "payload.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("first.bin", data)
        archive.writestr("second.bin", data)

    with pytest.raises(RuntimeError, match="aggregate ZIP compression ratio"):
        runner["materialize_payload"](manifest_path, manifest)


@pytest.mark.parametrize(
    ("complete", "state_arm"),
    [(False, "base"), (True, "tuned")],
)
def test_tuned_rejects_incomplete_or_wrong_arm_state(
    tmp_path: Path, complete: bool, state_arm: str
) -> None:
    runner = _render_template(tmp_path, arm="tuned", state_slug="owner/base-state")
    input_root = tmp_path / "input"
    state_root = input_root / "mount"
    state_root.mkdir(parents=True)
    state_manifest_path = state_root / "state_manifest.json"
    state_manifest_path.write_text(
        json.dumps(
            {
                "dataset_slug": "owner/base-state",
                "config_fingerprint": "f" * 64,
                "complete": complete,
                "arm": state_arm,
                "workflow_mode": "fp16-primary",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    runner["INPUT_ROOT"] = input_root
    runner["STATE_MANIFEST_SHA256"] = hashlib.sha256(
        state_manifest_path.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="complete base state"):
        runner["overlay_state"](
            tmp_path / "output",
            "f" * 64,
            {},
            tmp_path,
            300,
            "r" * 64,
            {},
        )


def test_complete_sidecar_requires_zero_errors_and_all_successes(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    predictions = tmp_path / "output/predictions"
    predictions.mkdir(parents=True)
    output = predictions / "base.jsonl"
    row, expected_predictions = _complete_prediction()
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    sidecar_path = predictions / "base.jsonl.manifest.json"
    sidecar = {
        "status": "complete",
        "arm": "base",
        "backend": "transformers",
        "result_scope": "publication_candidate",
        "conditions": ["original"],
        "limit": None,
        "expected_rows": 1,
        "completed_rows": 1,
        "successful_rows": 0,
        "error_rows": 1,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "experiment_fingerprint": "a" * 64,
        "input_fingerprint": "b" * 64,
        "protocol_fingerprint": "c" * 64,
        "config_fingerprint": "d" * 64,
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(RuntimeError, match="error-free"):
        runner["validate_complete_sidecar"](
            tmp_path / "output", "base", expected_predictions=expected_predictions
        )
    sidecar.update(successful_rows=1, error_rows=0)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert (
        runner["validate_complete_sidecar"](
            tmp_path / "output", "base", expected_predictions=expected_predictions
        )
        == sidecar
    )
    row["arm"] = "tuned"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    sidecar["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(RuntimeError, match="strict output schema"):
        runner["validate_complete_sidecar"](
            tmp_path / "output", "base", expected_predictions=expected_predictions
        )
    row["arm"] = "base"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    sidecar["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar["result_scope"] = "automatic_sensitivity_only"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(RuntimeError, match="error-free"):
        runner["validate_complete_sidecar"](
            tmp_path / "output", "base", expected_predictions=expected_predictions
        )


def test_runner_selects_only_the_hash_locked_dataset_manifest(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    input_root = tmp_path / "input"
    mount = input_root / "mount"
    mount.mkdir(parents=True)
    manifest = mount / "payload_manifest.json"
    manifest.write_text(json.dumps({"dataset_slug": "owner/input"}), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    decoy = input_root / "decoy"
    decoy.mkdir()
    (decoy / "payload_manifest.json").write_text("not JSON", encoding="utf-8")
    runner["INPUT_ROOT"] = input_root

    assert runner["find_unique_manifest"]("payload_manifest.json", "owner/input", digest) == manifest
    with pytest.raises(RuntimeError, match="hash-locked"):
        runner["find_unique_manifest"]("payload_manifest.json", "owner/input", "0" * 64)


def _complete_base_export_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, str, str, dict[tuple[str, str], dict]]:
    output_root = tmp_path / "output"
    predictions = output_root / "predictions"
    predictions.mkdir(parents=True)
    prediction = predictions / "base.jsonl"
    run_fingerprint = "a" * 64
    protocol_fingerprint = "b" * 64
    input_fingerprint = "c" * 64
    arm_config_fingerprint = "d" * 64
    row, expected_predictions = _complete_prediction(
        protocol_fingerprint=protocol_fingerprint,
        config_fingerprint=arm_config_fingerprint,
    )
    prediction.write_text(json.dumps(row) + "\n", encoding="utf-8")
    sidecar = {
        "status": "complete",
        "arm": "base",
        "backend": "transformers",
        "result_scope": "publication_candidate",
        "conditions": ["original"],
        "limit": None,
        "expected_rows": 1,
        "completed_rows": 1,
        "successful_rows": 1,
        "error_rows": 0,
        "output_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        "experiment_fingerprint": run_fingerprint,
        "input_fingerprint": input_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "config_fingerprint": arm_config_fingerprint,
        "runtime": {
            "backend": "transformers",
            "packages": {"torch": "2.3.1+cu121"},
            "gpu_count": 2,
            "gpu_names": ["Tesla T4", "Tesla T4"],
            "cuda_version": "12.1",
        },
    }
    (predictions / "base.jsonl.manifest.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    arm_summary = {
        "config_fingerprint": arm_config_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "expected_rows": 1,
        "existing_rows": 0,
        "written_rows": 1,
        "successful_rows": 1,
        "error_rows": 0,
        "output_sha256": sidecar["output_sha256"],
    }
    (predictions / "base.summary.json").write_text(
        json.dumps(arm_summary) + "\n", encoding="utf-8"
    )
    (output_root / "inference_summary.json").write_text(
        json.dumps(
            {
                "experiment_config_fingerprint": "f" * 64,
                "run_fingerprint": run_fingerprint,
                "protocol_fingerprint": protocol_fingerprint,
                "precision_mode": "fp16-primary",
                "result_scope": "publication_candidate",
                "exploratory": False,
                "arms": {"base": arm_summary},
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in (
        "pip-freeze.txt",
        "install.log",
        "editable-install.log",
        "pip-check.log",
        "infer-base.log",
    ):
        (runtime / name).write_text("runtime\n", encoding="utf-8")
    (runtime / "environment.json").write_text(
        json.dumps(
            {
                "arm": "base",
                "workflow_mode": "fp16-primary",
                "payload_dataset_slug": "owner/input",
                "state_dataset_slug": "",
                "config_fingerprint": "f" * 64,
                "payload_manifest_sha256": "a" * 64,
                "state_manifest_sha256": None,
                "execution_lock_sha256": "e" * 64,
                "gpu_count": 2,
                "gpu_names": ["Tesla T4", "Tesla T4"],
                "torch": "2.3.1+cu121",
                "cuda": "12.1",
                "torch_after_install": "2.3.1+cu121",
                "bitsandbytes": None,
            }
        ),
        encoding="utf-8",
    )
    return output_root, runtime, run_fingerprint, protocol_fingerprint, expected_predictions


def test_successful_state_export_requires_exact_semantic_inventory(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    output_root, runtime, run_fingerprint, protocol_fingerprint, expected_predictions = (
        _complete_base_export_inputs(tmp_path)
    )
    runner["EXPORT_ROOT"] = tmp_path / "export"

    runner["export_state"](
        output_root,
        "f" * 64,
        None,
        runtime,
        True,
        1,
        run_fingerprint,
        expected_predictions,
    )

    manifest = json.loads(
        (runner["EXPORT_ROOT"] / "state_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["artifact_version"] == 4
    assert manifest["execution_lock_sha256"] == "e" * 64
    assert manifest["protocol_fingerprint"] == protocol_fingerprint
    assert {entry["path"] for entry in manifest["files"]} == runner[
        "required_complete_paths"
    ]("base")


def test_successful_state_export_rejects_extra_allowlisted_file(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    output_root, runtime, run_fingerprint, _, expected_predictions = (
        _complete_base_export_inputs(tmp_path)
    )
    (output_root / "predictions/extra.summary.json").write_text("{}\n", encoding="utf-8")
    runner["EXPORT_ROOT"] = tmp_path / "export"

    with pytest.raises(RuntimeError, match="exact required inventory"):
        runner["export_state"](
            output_root,
            "f" * 64,
            None,
            runtime,
            True,
            1,
            run_fingerprint,
            expected_predictions,
        )


def test_successful_state_export_rejects_tampered_arm_summary(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    output_root, runtime, run_fingerprint, _, expected_predictions = (
        _complete_base_export_inputs(tmp_path)
    )
    summary_path = output_root / "predictions/base.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["successful_rows"] = 0
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    runner["EXPORT_ROOT"] = tmp_path / "export"

    with pytest.raises(RuntimeError, match="arm summary"):
        runner["export_state"](
            output_root,
            "f" * 64,
            None,
            runtime,
            True,
            1,
            run_fingerprint,
            expected_predictions,
        )


def test_successful_state_export_rejects_unbound_runtime(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    output_root, runtime, run_fingerprint, _, expected_predictions = (
        _complete_base_export_inputs(tmp_path)
    )
    environment_path = runtime / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["execution_lock_sha256"] = "0" * 64
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    runner["EXPORT_ROOT"] = tmp_path / "export"

    with pytest.raises(RuntimeError, match="runtime environment"):
        runner["export_state"](
            output_root,
            "f" * 64,
            None,
            runtime,
            True,
            1,
            run_fingerprint,
            expected_predictions,
        )


def test_independent_state_validator_rejects_semantic_tampering_with_resealed_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _render_template(tmp_path)
    output_root, runtime, run_fingerprint, _, expected_predictions = (
        _complete_base_export_inputs(tmp_path)
    )
    export_root = tmp_path / "export"
    runner["EXPORT_ROOT"] = export_root
    runner["export_state"](
        output_root,
        "f" * 64,
        None,
        runtime,
        True,
        1,
        run_fingerprint,
        expected_predictions,
    )

    validator = _load_script("validate_state")
    runner_path = tmp_path / "runner.py"
    runner_path.write_text("# locked runner\n", encoding="utf-8")
    payload_manifest = tmp_path / "payload_manifest.json"
    payload_manifest.write_text("{}\n", encoding="utf-8")
    initial_sidecar = json.loads(
        (export_root / "predictions/base.jsonl.manifest.json").read_text(encoding="utf-8")
    )
    expected_sidecar = {
        field: initial_sidecar.get(field)
        for field in (
            "input_fingerprint",
            "protocol_fingerprint",
            "config_fingerprint",
            "conditions",
            "seed",
            "configuration",
            "fingerprints",
        )
    }
    monkeypatch.setattr(validator.runpy, "run_path", lambda *_args, **_kwargs: runner)
    monkeypatch.setattr(
        inference_module,
        "_validate_kaggle_runtime_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        validator,
        "_expected_protocol",
        lambda *_args, **_kwargs: (
            {"experiment": {"id": "test"}},
            run_fingerprint,
            expected_predictions,
            {"base": expected_sidecar},
        ),
    )
    arguments = SimpleNamespace(
        runner=runner_path,
        payload_manifest=payload_manifest,
        export_root=export_root,
        arm="base",
        expected_dataset="owner/state",
        expected_parent_sha256="",
        parent_state_root=None,
        expected_payload_sha256="a" * 64,
        expected_config_fingerprint="f" * 64,
        expected_execution_lock="e" * 64,
        expected_rows=1,
    )
    assert validator.validate_state(arguments)["valid"] is True

    sidecar_path = export_root / "predictions/base.jsonl.manifest.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    original_runtime = sidecar.pop("runtime")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    state_path = export_root / "state_manifest.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    sidecar_entry = next(
        entry
        for entry in state["files"]
        if entry["path"] == "predictions/base.jsonl.manifest.json"
    )
    sidecar_entry["size"] = sidecar_path.stat().st_size
    sidecar_entry["sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime provenance"):
        validator.validate_state(arguments)
    sidecar["runtime"] = original_runtime
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    sidecar_entry["size"] = sidecar_path.stat().st_size
    sidecar_entry["sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    prediction_path = export_root / "predictions/base.jsonl"
    row = json.loads(prediction_path.read_text(encoding="utf-8"))
    row["paper_id"] = "tampered-paper"
    prediction_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["output_sha256"] = output_hash
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    arm_summary_path = export_root / "predictions/base.summary.json"
    arm_summary = json.loads(arm_summary_path.read_text(encoding="utf-8"))
    arm_summary["output_sha256"] = output_hash
    arm_summary_path.write_text(json.dumps(arm_summary), encoding="utf-8")
    summary_path = export_root / "inference_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["arms"]["base"]["output_sha256"] = output_hash
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for entry in state["files"]:
        path = export_root / entry["path"]
        entry["size"] = path.stat().st_size
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="strict output schema"):
        validator.validate_state(arguments)


def test_final_state_must_preserve_parent_base_artifact_bytes(tmp_path: Path) -> None:
    base_runner = _render_template(tmp_path)
    output_root, runtime, run_fingerprint, _, expected_predictions = (
        _complete_base_export_inputs(tmp_path)
    )
    parent_root = tmp_path / "parent"
    base_runner["EXPORT_ROOT"] = parent_root
    base_runner["export_state"](
        output_root,
        "f" * 64,
        None,
        runtime,
        True,
        1,
        run_fingerprint,
        expected_predictions,
    )
    parent_manifest_path = parent_root / "state_manifest.json"
    parent_state = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_state["dataset_slug"] = "owner/base-state"
    parent_manifest_path.write_text(json.dumps(parent_state), encoding="utf-8")
    parent_hash = hashlib.sha256(parent_manifest_path.read_bytes()).hexdigest()
    tuned_runner = _render_template(
        tmp_path / "tuned",
        arm="tuned",
        state_slug="owner/base-state",
        state_manifest_sha256=parent_hash,
    )
    final_root = tmp_path / "final"
    for relative in (
        "predictions/base.jsonl",
        "predictions/base.jsonl.manifest.json",
        "predictions/base.summary.json",
    ):
        source = parent_root / relative
        destination = final_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    validator = _load_script("validate_state")

    validator._validate_parent_state(
        parent_root,
        final_root,
        parent_hash,
        parent_state,
        tuned_runner,
    )

    (final_root / "predictions/base.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed inherited parent artifact"):
        validator._validate_parent_state(
            parent_root,
            final_root,
            parent_hash,
            parent_state,
            tuned_runner,
        )


def test_early_failure_export_is_incomplete_and_keeps_traceback(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    export_root = tmp_path / "export"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "error.log").write_text("traceback\n", encoding="utf-8")
    (runtime / "credential.json").write_text('{"secret":"value"}\n', encoding="utf-8")
    runner["EXPORT_ROOT"] = export_root
    runner["fallback_export"](runtime, "unavailable", None)
    manifest = json.loads((export_root / "state_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["base_state_complete"] is False
    assert manifest["workflow_mode"] == "fp16-primary"
    assert (export_root / "runtime/error.log").read_text(encoding="utf-8") == "traceback\n"
    assert not (export_root / "runtime/credential.json").exists()


def test_failed_kernel_cannot_export_complete_state(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    output_root = tmp_path / "output"
    predictions = output_root / "predictions"
    predictions.mkdir(parents=True)
    output = predictions / "base.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    (predictions / "base.jsonl.manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "expected_rows": 1,
                "completed_rows": 1,
                "successful_rows": 1,
                "error_rows": 0,
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "credential.json").write_text('{"secret":"value"}\n', encoding="utf-8")
    runner["EXPORT_ROOT"] = tmp_path / "export"

    runner["export_state"](output_root, "f" * 64, None, runtime, False, None, None, {})

    manifest = json.loads(
        (runner["EXPORT_ROOT"] / "state_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["kernel_succeeded"] is False
    assert manifest["complete"] is False
    assert not (runner["EXPORT_ROOT"] / "runtime/credential.json").exists()


def test_runner_helpers_and_t4x2_metadata_contract(tmp_path: Path) -> None:
    runner = _render_template(tmp_path)
    assert runner["state_path_allowed"]("predictions/base.jsonl")
    assert runner["state_path_allowed"]("predictions/base.jsonl.manifest.json")
    assert runner["state_path_allowed"]("runtime/environment.json")
    assert not runner["state_path_allowed"]("../kaggle.json")
    assert not runner["state_path_allowed"]("src/scireason/__init__.py")

    powershell = (KAGGLE / "kaggle_api.ps1").read_text(encoding="utf-8")
    template = (KAGGLE / "kernel_runner.py.template").read_text(encoding="utf-8")
    state_validator = (KAGGLE / "validate_state.py").read_text(encoding="utf-8")
    compile(state_validator, str(KAGGLE / "validate_state.py"), "exec")
    assert 'machine_shape = "NvidiaTeslaT4"' in powershell
    assert '"--accelerator", "NvidiaTeslaT4", "--timeout", "43200"' in powershell
    assert "enable_gpu = $true" in powershell
    assert "is_private = $true" in powershell
    assert 'docker_image_pinning_type = "original"' in powershell
    assert "docker_image =" not in powershell
    assert "Invoke-NativeCapture" in powershell
    assert "Wait-DatasetReady" in powershell
    assert "Assert-KernelAccepted" in powershell
    assert "Assert-CompleteBaseState" in powershell
    assert "Assert-CompleteFinalState" in powershell
    assert "Assert-RenderedKernel" in powershell
    assert "Assert-PushReceipt" in powershell
    assert "kernel_version_io.py" in powershell
    assert "validate_state.py" in powershell
    assert "Assert-PythonState" in powershell
    assert "Assert-FileSha256" in powershell
    assert "Write-AtomicNewUtf8File" in powershell
    assert "push-attempt.json" in powershell
    assert '$ErrorActionPreference = "Continue"' in powershell
    assert 'runner["validate_inference_summary"]' in state_validator
    assert 'runner["validate_complete_sidecar"]' in state_validator
    assert "_validate_parent_state" in state_validator
    assert "Get-VerifiedPayloadBinding" in powershell
    assert "Get-ExecutionLockSha256" in powershell
    assert "payload_dataset_version" in powershell
    assert '"dataset-status", "--dataset"' in powershell
    assert "len(names) != 2" in template
    assert 're.fullmatch(r"(?:NVIDIA )?(?:TESLA )?T4"' in template
    assert '"--no-deps", "-e"' in template
    assert '"pip", "check"' in template
    assert "pip-freeze.txt" in template
    assert '"pip", "install", "-r"' in template
    assert '"pip", "install", "--no-deps", "-r"' not in template
    assert "state_manifest.json" in template
    assert "parent_state_manifest_sha256" in template
    assert "PAYLOAD_MANIFEST_SHA256" in template
    assert "STATE_MANIFEST_SHA256" in template
    assert "EXECUTION_LOCK_SHA256" in template
    assert '"artifact_version": 4' in template
    assert "required_complete_paths" in template
    assert 'source_root = REPO_ROOT / "src"' in template
    assert "sys.path.insert(0, str(source_root))" in template
    assert '"base_state_complete": base_complete' in template
    assert 'sidecar.get("error_rows") != 0' in template
    assert "validate_kaggle_precision_contract(config, WORKFLOW_MODE)" in template
    assert '"requirements-kaggle-nf4.txt"' in template
    assert "__WORKFLOW_MODE_JSON__" in template
    assert '[ValidateSet("fp16-primary", "nf4-sensitivity")]' in powershell
    assert "Assert-StateInventory" in powershell
    assert '"kernel_succeeded": run_succeeded' in template


def test_exact_kernel_version_bridge_sets_sdk_version_label(tmp_path: Path) -> None:
    module = _load_script("kernel_version_io")
    service = SimpleNamespace()
    requests_seen: list[object] = []

    def status(request):
        requests_seen.append(request)
        return SimpleNamespace(status="COMPLETE", failure_message=None)

    def describe(request):
        requests_seen.append(request)
        return SimpleNamespace(
            blob=SimpleNamespace(source="print('locked')\n"),
            metadata=SimpleNamespace(
                ref="owner/kernel",
                is_private=True,
                enable_gpu=True,
                enable_internet=True,
                machine_shape="NvidiaTeslaT4",
                dataset_data_sources=["owner/input/3"],
            ),
        )

    service.get_kernel_session_status = status
    service.get_kernel = describe

    def output(request):
        requests_seen.append(request)
        return SimpleNamespace(files=[], log=None, next_page_token=None)

    service.list_kernel_session_output = output

    class Client:
        kernels = SimpleNamespace(kernels_api_client=service)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    api = SimpleNamespace(build_kaggle_client=lambda: Client())
    assert module.kernel_status(api, "owner/kernel/7")["status"] == "complete"
    details = module.describe_kernel(api, "owner/kernel/7")
    assert details["source_sha256"] == hashlib.sha256(b"print('locked')\n").hexdigest()
    assert details["dataset_sources"] == ["owner/input/3"]
    target = tmp_path / "output"
    target.mkdir()
    assert module.download_output(api, "owner/kernel/7", target)["files"] == 0
    assert [request.version_label for request in requests_seen] == ["7", "7", "7"]

    assert module._output_path(tmp_path, "kaggle_export/state_manifest.json").is_relative_to(
        tmp_path
    )
    with pytest.raises(ValueError, match="unsafe"):
        module._output_path(tmp_path, "../credential.json")
    for unsafe in ("nested/result.json:secret", "nested/CON", "nested/trailing. "):
        with pytest.raises(ValueError, match="unsafe"):
            module._output_path(tmp_path, unsafe)


def test_payload_json_reader_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    module = _load_script("build_payload")
    document = tmp_path / "document.json"
    document.write_text('{"revision":"a","revision":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        module._read_json(document)
    document.write_text('{"revision":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        module._read_json(document)
    document.write_text('{"revision":1e999}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        module._read_json(document)


def test_payload_copy_verifies_the_bytes_after_copy(tmp_path: Path) -> None:
    module = _load_script("build_payload")
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed while it was copied"):
        module._copy_file(source, destination, hashlib.sha256(b"expected").hexdigest())
    assert not destination.exists()


def test_sdk_bridge_authenticates_identity_and_binds_dataset_status_version() -> None:
    module = _load_script("kernel_version_io")
    listed: list[dict] = []
    dataset_service = SimpleNamespace()
    versions = iter((3, 3))
    dataset_service.get_dataset = lambda _request: SimpleNamespace(
        current_version_number=next(versions)
    )
    dataset_service.get_dataset_status = lambda _request: SimpleNamespace(
        status=SimpleNamespace(name="READY")
    )

    class Client:
        datasets = SimpleNamespace(dataset_api_client=dataset_service)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    api = SimpleNamespace(
        config_values={"username": "owner"},
        dataset_list=lambda **kwargs: listed.append(kwargs),
        build_kaggle_client=lambda: Client(),
    )

    assert module.auth_info(api) == {"authenticated": True, "username": "owner"}
    assert listed == [{"mine": True, "page": 1}]
    assert module.dataset_status(api, "owner/input") == {
        "dataset": "owner/input",
        "status": "ready",
        "current_version_number": 3,
    }

    versions = iter((3, 4))
    with pytest.raises(RuntimeError, match="version changed"):
        module.dataset_status(api, "owner/input")


def test_exact_output_download_preflights_path_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("kernel_version_io")
    service = SimpleNamespace(
        list_kernel_session_output=lambda _request: SimpleNamespace(
            files=[
                SimpleNamespace(file_name="collision", url="https://example.test/one"),
                SimpleNamespace(file_name="collision/child", url="https://example.test/two"),
            ],
            log=None,
            next_page_token=None,
        )
    )

    class Client:
        kernels = SimpleNamespace(kernels_api_client=service)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    api = SimpleNamespace(build_kaggle_client=lambda: Client())
    target = tmp_path / "output"
    target.mkdir()
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *_args, **_kwargs: pytest.fail("unsafe topology must fail before download"),
    )

    with pytest.raises(RuntimeError, match="also an ancestor"):
        module.download_output(api, "owner/kernel/2", target)
    assert not any(target.iterdir())


def test_paginated_output_preserves_nonempty_kernel_log(tmp_path: Path) -> None:
    module = _load_script("kernel_version_io")
    calls = 0

    def output(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(files=[], log="diagnostic\n", next_page_token="next")
        return SimpleNamespace(files=[], log="", next_page_token=None)

    service = SimpleNamespace(list_kernel_session_output=output)

    class Client:
        kernels = SimpleNamespace(kernels_api_client=service)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    target = tmp_path / "output"
    target.mkdir()
    result = module.download_output(
        SimpleNamespace(build_kaggle_client=lambda: Client()),
        "owner/kernel/2",
        target,
    )

    assert result["bytes"] == len(b"diagnostic\n")
    assert (target / "kernel.log").read_text(encoding="utf-8") == "diagnostic\n"


def test_runner_imports_bitsandbytes_only_for_nf4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "2.3.1+cu121"
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    real_import = builtins.__import__

    def reject_bitsandbytes(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise AssertionError("FP16 mode must not import bitsandbytes")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_bitsandbytes)
    fp16_runner = _render_template(tmp_path, workflow_mode="fp16-primary")
    assert fp16_runner["check_installed_runtime"]() == {
        "torch_after_install": "2.3.1+cu121",
        "bitsandbytes": None,
    }

    monkeypatch.setattr(builtins, "__import__", real_import)
    fake_bitsandbytes = ModuleType("bitsandbytes")
    fake_bitsandbytes.__version__ = "0.48.1"
    monkeypatch.setitem(__import__("sys").modules, "bitsandbytes", fake_bitsandbytes)
    nf4_runner = _render_template(tmp_path, workflow_mode="nf4-sensitivity")
    assert nf4_runner["check_installed_runtime"]()["bitsandbytes"] == "0.48.1"


def test_precision_contract_rejects_mode_mismatch() -> None:
    config = _config()
    validate_kaggle_precision_contract(config, "fp16-primary")
    with pytest.raises(ValueError, match="precision_mode.*nf4-sensitivity"):
        validate_kaggle_precision_contract(config, "nf4-sensitivity")

    config["models"]["tuned"]["adapter_kwargs"] = {"autocast_adapter_dtype": False}
    with pytest.raises(ValueError, match="native FP32 LoRA"):
        validate_kaggle_precision_contract(config, "fp16-primary")

    config["models"]["tuned"]["adapter_kwargs"] = {"autocast_adapter_dtype": 1}
    with pytest.raises(ValueError, match="native FP32 LoRA"):
        validate_kaggle_precision_contract(config, "fp16-primary")

    config = _config()
    config["models"]["base"]["model_kwargs"]["low_cpu_mem_usage"] = 1
    with pytest.raises(ValueError, match="exact fp16-primary"):
        validate_kaggle_precision_contract(config, "fp16-primary")

    config = _config()
    config["power"]["n_items"] = 149
    with pytest.raises(ValueError, match="exact power.n_items=150"):
        validate_kaggle_precision_contract(config, "fp16-primary")


def test_kaggle_sources_do_not_embed_or_copy_credentials() -> None:
    sources = [
        path
        for path in KAGGLE.iterdir()
        if path.is_file() and path.suffix in {".py", ".ps1", ".template", ".md"}
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "C:\\\\Users\\\\chim.chi\\\\.kaggle" not in combined
    assert "shutil.copyfile(Credential" not in combined
    assert "Copy-Item $CredentialPath" not in combined
    assert '"key":"secret"' not in combined
    assert "%USERPROFILE%\\.kaggle\\kaggle.json" in combined


def test_kaggle_runtime_has_direct_compatibility_pins() -> None:
    requirements = (KAGGLE / "requirements-kaggle.txt").read_text(encoding="utf-8")
    nf4_requirements = (KAGGLE / "requirements-kaggle-nf4.txt").read_text(encoding="utf-8")
    assert "peft==0.19.1" in requirements
    assert "numpy==1.26.4" in requirements
    assert "hf-xet==1.1.10" in requirements
    assert "tokenizers==0.22.1" in requirements
    assert "bitsandbytes" not in requirements
    assert "-r requirements-kaggle.txt" in nf4_requirements
    assert "bitsandbytes==0.48.1" in nf4_requirements
    assert not any(line.lower().startswith("torch==") for line in requirements.splitlines())
