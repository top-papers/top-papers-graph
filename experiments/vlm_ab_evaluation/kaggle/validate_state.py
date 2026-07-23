#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independently validate a downloaded Kaggle state against local immutable inputs."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path, PurePosixPath
from typing import Any


def _inventory(root: Path, state: dict[str, Any], runner: dict[str, Any]) -> None:
    declared = runner["inventory_map"](state)
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("downloaded state contains a symlink")
    actual = {
        path.relative_to(root).as_posix()
        for path in paths
        if path.is_file() and path.relative_to(root).as_posix() != "state_manifest.json"
    }
    if actual != set(declared):
        raise RuntimeError("downloaded state differs from its exact manifest inventory")
    for relative, entry in declared.items():
        path = runner["safe_file"](root, relative)
        if path.stat().st_size != entry["size"] or runner["sha256_file"](path) != entry["sha256"]:
            raise RuntimeError(f"downloaded state file differs from its manifest: {relative}")


def _validate_parent_state(
    parent_root: Path,
    final_root: Path,
    expected_manifest_sha256: str,
    final_state: dict[str, Any],
    runner: dict[str, Any],
) -> None:
    parent = parent_root.resolve(strict=True)
    manifest_path = parent / "state_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("local parent state has no regular manifest")
    if runner["sha256_file"](manifest_path) != expected_manifest_sha256:
        raise RuntimeError("local parent state differs from the tuned render lock")
    state = runner["read_object"](manifest_path)
    if (
        state.get("artifact_version") != 4
        or isinstance(state.get("artifact_version"), bool)
        or state.get("complete") is not True
        or state.get("kernel_succeeded") is not True
        or state.get("base_state_complete") is not True
        or state.get("arm") != "base"
        or state.get("workflow_mode") != runner["WORKFLOW_MODE"]
        or state.get("dataset_slug") != runner["STATE_DATASET_SLUG"]
        or state.get("payload_manifest_sha256") != runner["PAYLOAD_MANIFEST_SHA256"]
        or state.get("parent_state_manifest_sha256") is not None
        or state.get("config_fingerprint") != final_state.get("config_fingerprint")
        or state.get("run_fingerprint") != final_state.get("run_fingerprint")
        or state.get("protocol_fingerprint") != final_state.get("protocol_fingerprint")
        or state.get("input_fingerprint") != final_state.get("input_fingerprint")
        or state.get("result_scope") != final_state.get("result_scope")
    ):
        raise RuntimeError("local parent state is not the exact complete inherited base state")
    if set(runner["inventory_map"](state)) != runner["required_complete_paths"]("base"):
        raise RuntimeError("local parent state has the wrong successful inventory")
    _inventory(parent, state, runner)
    for relative in (
        "predictions/base.jsonl",
        "predictions/base.jsonl.manifest.json",
        "predictions/base.summary.json",
    ):
        if runner["safe_file"](parent, relative).read_bytes() != runner["safe_file"](
            final_root, relative
        ).read_bytes():
            raise RuntimeError(f"final state changed inherited parent artifact: {relative}")


def _expected_protocol(
    payload_manifest_path: Path,
    expected_payload_sha256: str,
    expected_config_fingerprint: str,
    expected_rows: int,
    runner: dict[str, Any],
) -> tuple[
    dict[str, Any],
    str,
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if runner["sha256_file"](payload_manifest_path) != expected_payload_sha256:
        raise RuntimeError("local payload manifest differs from the rendered lock")
    payload_manifest = runner["read_object"](payload_manifest_path)
    if (
        payload_manifest.get("artifact_version") != 3
        or payload_manifest.get("workflow_mode") != runner["WORKFLOW_MODE"]
        or payload_manifest.get("dataset_slug") != runner["PAYLOAD_DATASET_SLUG"]
        or payload_manifest.get("config_relative") != runner["CONFIG_RELATIVE"]
    ):
        raise RuntimeError("local payload manifest differs from the rendered runner")
    payload_root = payload_manifest_path.parent / "payload"
    runner["verify_inventory"](payload_root, payload_manifest)
    source_repo = payload_root / "top-papers-graph"
    source_root = source_repo / "src"
    if not source_root.is_dir():
        raise RuntimeError("verified local payload has no source tree")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root))

    from scireason.vlm_ab.cli import _arm_config, _processor_config, _run_fingerprint
    from scireason.vlm_ab.config import (
        config_fingerprint,
        load_experiment_config,
        validate_kaggle_precision_contract,
    )
    from scireason.vlm_ab.inference import (
        ENGINE_VERSION,
        _fingerprint,
        _generation_settings,
        build_condition_rows,
    )
    from scireason.vlm_ab.prepare import resolve_prepare_manifest

    config_relative = payload_manifest.get("config_relative")
    if not isinstance(config_relative, str):
        raise RuntimeError("payload manifest has no config path")
    config_path = runner["safe_file"](
        payload_root,
        PurePosixPath("top-papers-graph", config_relative).as_posix(),
    )
    config = load_experiment_config(config_path)
    validate_kaggle_precision_contract(config, runner["WORKFLOW_MODE"])
    fingerprint = config_fingerprint(config)
    if fingerprint != expected_config_fingerprint or fingerprint != payload_manifest.get(
        "config_fingerprint"
    ):
        raise RuntimeError("verified payload config differs from the rendered lock")

    prepared_root = (source_repo / config["experiment"]["output_dir"]).resolve(strict=True)
    prepared_root.relative_to(source_repo.resolve(strict=True))
    prepare_manifest = runner["read_object"](prepared_root / "prepare_manifest.json")
    prepared = resolve_prepare_manifest(prepare_manifest, prepared_root)
    benchmark: list[dict[str, Any]] = []
    with Path(prepared["frozen_benchmark"]).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"frozen benchmark has a blank row at {line_number}")
            row = runner["strict_json_loads"](line)
            if not isinstance(row, dict):
                raise RuntimeError(f"frozen benchmark row {line_number} is not an object")
            benchmark.append(row)
    condition_rows = build_condition_rows(
        benchmark,
        prepared["dataset_root"],
        config["conditions"],
        config["experiment"]["seed"],
    )
    expected_predictions = {
        (row["sample_id"], row["condition"]): row for row in condition_rows
    }
    if (
        len(expected_predictions) != len(condition_rows)
        or len(expected_predictions) != expected_rows
        or payload_manifest.get("n_items") != 150
        or payload_manifest.get("conditions") != config["conditions"]
        or payload_manifest.get("expected_rows") != expected_rows
    ):
        raise RuntimeError("verified payload has a different sample/condition protocol")
    fingerprint_rows = sorted(
        condition_rows,
        key=lambda row: (row["sample_id"], row["condition"]),
    )
    input_fingerprint = _fingerprint(
        [
            {
                "sample_id": row["sample_id"],
                "paper_id": row["paper_id"],
                "condition": row["condition"],
                "condition_input_fingerprint": row["condition_input_fingerprint"],
            }
            for row in fingerprint_rows
        ]
    )
    processor = _processor_config(config)
    generation = _generation_settings(config["generation"])
    protocol_fingerprint = _fingerprint(
        {
            "engine_version": ENGINE_VERSION,
            "backend": "transformers",
            "processor_config": processor,
            "generation_config": generation,
            "conditions": config["conditions"],
            "seed": config["experiment"]["seed"],
            "limit": None,
            "input_fingerprint": input_fingerprint,
        }
    )
    expected_sidecars: dict[str, dict[str, Any]] = {}
    for arm in ("base", "tuned"):
        arm_config = _arm_config(config, arm, prepared)
        arm_fingerprint = _fingerprint(arm_config)
        arm_config_fingerprint = _fingerprint(
            {
                "protocol_fingerprint": protocol_fingerprint,
                "arm": arm,
                "arm_config": arm_config,
            }
        )
        expected_sidecars[arm] = {
            "input_fingerprint": input_fingerprint,
            "protocol_fingerprint": protocol_fingerprint,
            "config_fingerprint": arm_config_fingerprint,
            "conditions": config["conditions"],
            "seed": config["experiment"]["seed"],
            "configuration": {
                "arm": arm_config,
                "processor": processor,
                "generation": generation,
            },
            "fingerprints": {
                "input": input_fingerprint,
                "protocol": protocol_fingerprint,
                "config": arm_config_fingerprint,
                "processor": _fingerprint(processor),
                "generation": _fingerprint(generation),
                "arm": arm_fingerprint,
            },
        }
    return config, _run_fingerprint(config, prepared), expected_predictions, expected_sidecars


def validate_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.runner.is_symlink() or not args.runner.is_file():
        raise RuntimeError("rendered runner is not a regular file")
    runner_path = args.runner.resolve(strict=True)
    runner = runpy.run_path(str(runner_path), run_name="locked_kaggle_runner")
    expected_parent = args.expected_parent_sha256 if args.arm == "tuned" else None
    if (
        runner.get("ARM") != args.arm
        or runner.get("PAYLOAD_MANIFEST_SHA256") != args.expected_payload_sha256
        or runner.get("EXECUTION_LOCK_SHA256") != args.expected_execution_lock
        or (runner.get("STATE_MANIFEST_SHA256") or None) != expected_parent
    ):
        raise RuntimeError("rendered runner constants differ from the expected workflow")
    root = args.export_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("downloaded state root is not a directory")
    manifest_path = root / "state_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("downloaded state has no regular state manifest")

    config, run_fingerprint, expected_predictions, expected_sidecars = _expected_protocol(
        args.payload_manifest.resolve(strict=True),
        args.expected_payload_sha256,
        args.expected_config_fingerprint,
        args.expected_rows,
        runner,
    )
    state = runner["read_object"](manifest_path)
    if (
        state.get("artifact_version") != 4
        or isinstance(state.get("artifact_version"), bool)
        or state.get("complete") is not True
        or state.get("kernel_succeeded") is not True
        or state.get("base_state_complete") is not True
        or state.get("arm") != args.arm
        or state.get("workflow_mode") != runner["WORKFLOW_MODE"]
        or state.get("dataset_slug") != args.expected_dataset
        or state.get("result_scope") != runner["expected_result_scope"]()
        or state.get("parent_state_manifest_sha256") != expected_parent
        or state.get("config_fingerprint") != args.expected_config_fingerprint
        or state.get("payload_manifest_sha256") != args.expected_payload_sha256
        or state.get("execution_lock_sha256") != args.expected_execution_lock
        or state.get("run_fingerprint") != run_fingerprint
    ):
        raise RuntimeError("downloaded state header differs from the immutable workflow")
    if set(runner["inventory_map"](state)) != runner["required_complete_paths"](args.arm):
        raise RuntimeError("downloaded state does not declare the exact successful inventory")
    _inventory(root, state, runner)
    if args.arm == "tuned":
        if args.parent_state_root is None:
            raise RuntimeError("tuned state validation requires the local parent state")
        _validate_parent_state(
            args.parent_state_root,
            root,
            expected_parent or "",
            state,
            runner,
        )
    elif args.parent_state_root is not None:
        raise RuntimeError("base state validation must not configure a parent state")
    environment = runner["validate_runtime_environment"](
        root,
        args.arm,
        args.expected_config_fingerprint,
        args.expected_execution_lock,
        expected_parent,
    )

    arms = ("base",) if args.arm == "base" else ("base", "tuned")
    sidecars = {
        arm: runner["validate_complete_sidecar"](
            root,
            arm,
            expected_predictions=expected_predictions,
            expected_rows=args.expected_rows,
            expected_run_fingerprint=run_fingerprint,
        )
        for arm in arms
    }
    for arm, sidecar in sidecars.items():
        expected_sidecar = expected_sidecars[arm]
        for field in (
            "input_fingerprint",
            "protocol_fingerprint",
            "config_fingerprint",
            "conditions",
            "seed",
            "configuration",
            "fingerprints",
        ):
            if sidecar.get(field) != expected_sidecar[field]:
                raise RuntimeError(f"{arm} sidecar differs from the immutable protocol in {field}")
    from scireason.vlm_ab.inference import _validate_kaggle_runtime_environment

    for arm, sidecar in sidecars.items():
        runtime = sidecar.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError(f"{arm} sidecar has no runtime provenance")
        _validate_kaggle_runtime_environment(runtime, runner["expected_result_scope"]())
        packages = runtime.get("packages")
        if not isinstance(packages, dict) or (
            runtime.get("gpu_count") != environment.get("gpu_count")
            or runtime.get("cuda_version") != environment.get("cuda")
            or packages.get("torch") != environment.get("torch_after_install")
            or (
                runner["WORKFLOW_MODE"] == "nf4-sensitivity"
                and packages.get("bitsandbytes") != environment.get("bitsandbytes")
            )
        ):
            raise RuntimeError(f"{arm} sidecar runtime differs from the exported environment")
    current = sidecars[args.arm]
    for field in ("experiment_fingerprint", "protocol_fingerprint", "input_fingerprint"):
        if state.get(field if field != "experiment_fingerprint" else "run_fingerprint") != current.get(
            field
        ):
            raise RuntimeError(f"downloaded state differs from its sidecar in {field}")
    if args.arm == "tuned":
        if sidecars["base"]["runtime"] != sidecars["tuned"]["runtime"]:
            raise RuntimeError("base/tuned sidecar runtime provenance differs")
        for field in ("experiment_fingerprint", "protocol_fingerprint", "input_fingerprint"):
            if sidecars["base"].get(field) != sidecars["tuned"].get(field):
                raise RuntimeError(f"base/tuned sidecars differ in {field}")
    runner["validate_inference_summary"](
        root,
        args.arm,
        args.expected_config_fingerprint,
        run_fingerprint,
        str(current["protocol_fingerprint"]),
        sidecars,
    )
    return {
        "arm": args.arm,
        "config_fingerprint": args.expected_config_fingerprint,
        "experiment_id": config["experiment"]["id"],
        "rows": args.expected_rows,
        "valid": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "tuned"), required=True)
    parser.add_argument("--expected-dataset", required=True)
    parser.add_argument("--expected-parent-sha256", default="")
    parser.add_argument("--parent-state-root", type=Path)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--expected-config-fingerprint", required=True)
    parser.add_argument("--expected-execution-lock", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        result = validate_state(_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
