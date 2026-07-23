#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare only a verified synthetic source through the existing exploratory gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    RESULT_SCOPE,
    SyntheticHarnessError,
    json_sha256,
    read_json_object,
    repo_root,
    scope_metadata,
    sha256_file,
    verify_generated_run,
    write_json_new,
)


_TECHNICAL_BLOCKERS = {
    "duplicate_sample_id",
    "empty_benchmark",
    "image_placeholder_mismatch",
    "missing_image",
    "schema_error",
    "unsafe_image_path",
}


def _load_existing_prepare(root: Path, generated: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from scireason.vlm_ab.prepare import verify_prepare_manifest, verify_prepared_audit

    run = generated["run"]
    manifest_path = run / "prepare_manifest.json"
    manifest = read_json_object(manifest_path, "prepare manifest")
    config = generated["config"]
    verify_prepare_manifest(manifest, run)
    audit = verify_prepared_audit(config, manifest, run)
    if manifest.get("exploratory") is not True or manifest.get("result_scope") != RESULT_SCOPE:
        raise SyntheticHarnessError("prepare manifest is not explicitly exploratory-only")
    if manifest.get("publication_ready") is not False:
        raise SyntheticHarnessError("synthetic prepare manifest must not be publication-ready")
    sample_count = generated["manifest"]["sample_count"]
    if (
        manifest.get("input_rows") != sample_count
        or manifest.get("frozen_rows") != sample_count
        or manifest.get("excluded_rows") != 0
        or len(manifest.get("frozen_sample_ids", [])) != sample_count
    ):
        raise SyntheticHarnessError("prepare did not freeze every synthetic source row")
    critical_by_code = audit.get("summary", {}).get("critical_by_code", {})
    if not isinstance(critical_by_code, dict) or set(critical_by_code).intersection(_TECHNICAL_BLOCKERS):
        raise SyntheticHarnessError("synthetic prepare has a technical benchmark blocker")
    return manifest, audit


def _receipt(generated: dict[str, Any], manifest: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    run = generated["run"]
    return {
        **scope_metadata(),
        "artifact_version": 1,
        "run_dir": generated["run_dir"].as_posix(),
        "result_scope": RESULT_SCOPE,
        "exploratory": True,
        "generator_manifest_sha256": sha256_file(run / "generator_manifest.json"),
        "synthetic_config_sha256": sha256_file(run / "synthetic_config.yaml"),
        "prepare_manifest_sha256": sha256_file(run / "prepare_manifest.json"),
        "frozen_benchmark_sha256": manifest.get("frozen_benchmark_sha256"),
        "input_rows": manifest.get("input_rows"),
        "frozen_rows": manifest.get("frozen_rows"),
        "technical_blocker_codes": sorted(
            set(audit.get("summary", {}).get("critical_by_code", {})).intersection(_TECHNICAL_BLOCKERS)
        ),
    }


def _verify_receipt(path: Path, expected: dict[str, Any]) -> None:
    existing = read_json_object(path, "synthetic prepare receipt")
    if existing != expected:
        raise SyntheticHarnessError("existing synthetic prepare receipt differs from verified inputs")
    if json_sha256(existing) != json_sha256(expected):  # defensive strict-JSON assertion
        raise SyntheticHarnessError("synthetic prepare receipt has an invalid canonical digest")


def prepare_generated(repo_root_value: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Run only ``prepare --exploratory`` for an intact synthetic source directory."""

    root = repo_root(repo_root_value)
    generated = verify_generated_run(root, run_dir)
    run = generated["run"]
    receipt_path = run / "synthetic_prepare_receipt.json"
    manifest_path = run / "prepare_manifest.json"
    if manifest_path.exists():
        manifest, audit = _load_existing_prepare(root, generated)
        expected = _receipt(generated, manifest, audit)
        if receipt_path.exists():
            _verify_receipt(receipt_path, expected)
        else:
            write_json_new(receipt_path, expected)
        return {
            **scope_metadata(),
            "run_dir": generated["run_dir"].as_posix(),
            "prepare_manifest": str(manifest_path),
            "frozen_rows": manifest["frozen_rows"],
            "result_scope": RESULT_SCOPE,
        }
    if receipt_path.exists():
        raise SyntheticHarnessError("prepare receipt exists without a prepare manifest")
    for partial in (run / "inputs", run / "audit"):
        if partial.exists():
            raise SyntheticHarnessError("refusing to overwrite an incomplete synthetic prepare output")

    command = [
        sys.executable,
        str(root / "experiments" / "vlm_ab_evaluation" / "run_pipeline.py"),
        "--config",
        str(run / "synthetic_config.yaml"),
        "--repo-root",
        str(root),
        "prepare",
        "--benchmark-dir",
        str(run / "synthetic_source"),
        "--exploratory",
    ]
    try:
        subprocess.run(command, cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        raise SyntheticHarnessError("existing exploratory prepare command failed") from exc

    # Re-read source/key integrity after the external process before trusting outputs.
    generated = verify_generated_run(root, run_dir)
    manifest, audit = _load_existing_prepare(root, generated)
    expected = _receipt(generated, manifest, audit)
    write_json_new(receipt_path, expected)
    return {
        **scope_metadata(),
        "run_dir": generated["run_dir"].as_posix(),
        "prepare_manifest": str(run / "prepare_manifest.json"),
        "frozen_rows": manifest["frozen_rows"],
        "result_scope": RESULT_SCOPE,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare_generated(args.repo_root, args.run_dir)
    except (OSError, RuntimeError, ValueError, SyntheticHarnessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
