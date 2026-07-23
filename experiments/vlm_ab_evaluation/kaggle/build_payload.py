#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build a private, immutable Kaggle input dataset staging directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_REPO_ROOT / "src"))

from scireason.vlm_ab.config import (  # noqa: E402
    config_fingerprint,
    load_experiment_config,
    validate_kaggle_precision_contract,
)
from scireason.vlm_ab.prepare import (  # noqa: E402
    code_provenance,
    source_path_is_evaluation_source,
    source_path_is_sensitive,
    verify_prepare_manifest,
    verify_prepared_audit,
)


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,49}$")
_GENERATED_PARTS = {"build", "staging", "downloads", "__pycache__", ".cache"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    data = path.read_bytes()
    value = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    pending: list[Any] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            raise ValueError(f"non-finite JSON number in {path}")
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return value, hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_snapshot(path)[0]


def _safe_relative(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside repo root: {path}") from exc
    if path.is_symlink() or any(part == ".." for part in relative.parts):
        raise ValueError(f"unsafe {label}: {path}")
    return relative


def _copy_file(source: Path, destination: Path, expected_sha256: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {source}")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError(f"source has no valid expected SHA256: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if _sha256(destination) != expected_sha256:
            raise ValueError(f"source changed while it was copied: {source}")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _safe_run_file(run: Path, relative_text: str, label: str) -> tuple[Path, Path]:
    relative = Path(relative_text)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError(f"unsafe {label} path: {relative_text}")
    unresolved = run / relative
    current = run
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path traverses a symlink: {relative_text}")
    resolved = unresolved.resolve(strict=True)
    try:
        resolved.relative_to(run.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} path escapes prepared run: {relative_text}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {relative_text}")
    return resolved, relative


def _add_run_file(
    selected: dict[Path, tuple[Path, str]],
    run: Path,
    relative_text: str,
    expected_sha256: Any,
    label: str,
) -> None:
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"{label} has no valid SHA256")
    source, relative = _safe_run_file(run, relative_text, label)
    actual = _sha256(source)
    if actual != expected_sha256:
        raise ValueError(f"{label} hash mismatch: {relative_text}")
    previous = selected.get(relative)
    if previous is not None and previous != (source, actual):
        raise ValueError(f"conflicting prepared artifact path: {relative_text}")
    selected[relative] = (source, actual)


def _prepared_files(
    manifest: dict[str, Any],
    prepared_run: Path,
    prepare_manifest_sha256: str,
) -> dict[Path, tuple[Path, str]]:
    selected: dict[Path, tuple[Path, str]] = {}
    manifest_path = prepared_run / "prepare_manifest.json"
    selected[Path("prepare_manifest.json")] = (manifest_path, prepare_manifest_sha256)
    artifact_paths = manifest.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise ValueError("prepare manifest artifact_paths is malformed")
    for key, relative in artifact_paths.items():
        if key == "dataset_root":
            continue
        expected = manifest.get(f"{key}_sha256")
        if relative is None and expected is None:
            continue
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"prepare manifest has an incomplete {key} path")
        _add_run_file(selected, prepared_run, relative, expected, key)

    training_files = manifest.get("training_files")
    if not isinstance(training_files, list):
        raise ValueError("prepare manifest training_files is malformed")
    for index, entry in enumerate(training_files):
        if not isinstance(entry, dict) or not isinstance(entry.get("relative_path"), str):
            raise ValueError(f"training_files[{index}] has no relative_path")
        _add_run_file(
            selected,
            prepared_run,
            entry["relative_path"],
            entry.get("sha256"),
            f"training_files[{index}]",
        )

    dataset_relative = artifact_paths.get("dataset_root")
    audit_relative = artifact_paths.get("audit_json")
    if not isinstance(dataset_relative, str) or not isinstance(audit_relative, str):
        raise ValueError("prepare manifest lacks dataset_root or audit_json")
    audit_path, _ = _safe_run_file(prepared_run, audit_relative, "audit_json")
    audit, audit_sha256 = _read_json_snapshot(audit_path)
    if audit_sha256 != manifest.get("audit_json_sha256"):
        raise ValueError("audit_json changed before payload selection")
    attestation_relative = artifact_paths.get("adapter_checkpoint_attestation")
    if not isinstance(attestation_relative, str):
        raise ValueError("strict prepare manifest lacks adapter checkpoint attestation")
    attestation_path, _ = _safe_run_file(
        prepared_run,
        attestation_relative,
        "adapter_checkpoint_attestation",
    )
    attestation, attestation_sha256 = _read_json_snapshot(attestation_path)
    if attestation_sha256 != manifest.get("adapter_checkpoint_attestation_sha256"):
        raise ValueError("adapter checkpoint attestation changed before payload selection")
    if (
        attestation.get("filename") != "adapter_model.safetensors"
        or attestation.get("tensor_dtype") != "F32"
        or not isinstance(attestation.get("tensor_count"), int)
        or attestation.get("tensor_count", 0) <= 0
    ):
        raise ValueError("prepared adapter checkpoint attestation is not native FP32 LoRA")
    findings = audit.get("per_sample_findings")
    if not isinstance(findings, dict):
        raise ValueError("audit per_sample_findings is malformed")
    image_records = 0
    for sample_id, details in findings.items():
        if not isinstance(details, dict):
            raise ValueError(f"audit sample is malformed: {sample_id}")
        hashes = details.get("image_hashes", [])
        if not isinstance(hashes, list):
            raise ValueError(f"audit image_hashes is malformed: {sample_id}")
        for index, record in enumerate(hashes):
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError(f"audit image record is malformed: {sample_id}[{index}]")
            image_relative = (Path(dataset_relative) / Path(record["path"])).as_posix()
            _add_run_file(
                selected,
                prepared_run,
                image_relative,
                record.get("sha256"),
                f"audited image {sample_id}[{index}]",
            )
            image_records += 1
    if not image_records:
        raise ValueError("strict prepared audit contains no audited images")
    return selected


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def build_payload(
    repo_root: Path,
    config_path: Path,
    prepared_run: Path,
    staging_dir: Path,
    owner: str,
    slug: str,
    workflow_mode: str,
) -> Path:
    """Verify strict inputs and atomically create a Kaggle dataset directory."""

    root = repo_root.resolve(strict=True)
    config_relative = _safe_relative(config_path, root, "config")
    run_relative = _safe_relative(prepared_run, root, "prepared run")
    config = load_experiment_config(root / config_relative)
    validate_kaggle_precision_contract(config, workflow_mode)
    expected_run = (root / config["experiment"]["output_dir"]).resolve()
    if prepared_run.resolve(strict=True) != expected_run:
        raise ValueError("prepared run does not match experiment.output_dir")

    manifest_path = prepared_run / "prepare_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("prepared run has no regular prepare_manifest.json")
    manifest, prepare_manifest_sha256 = _read_json_snapshot(manifest_path)
    fingerprint = config_fingerprint(config)
    if manifest.get("config_fingerprint") != fingerprint:
        raise ValueError("prepare manifest config fingerprint mismatch")
    verify_prepare_manifest(manifest, prepared_run)
    audit = verify_prepared_audit(config, manifest, prepared_run)
    if (
        manifest.get("publication_ready") is not True
        or manifest.get("exploratory") is not False
        or manifest.get("result_scope") != "publication_ready"
        or audit.get("publication_ready") is not True
    ):
        raise ValueError("payload requires a strict, nonexploratory publication-ready prepare")
    artifact_paths = manifest.get("artifact_paths")
    attestation_relative = (
        artifact_paths.get("adapter_checkpoint_attestation")
        if isinstance(artifact_paths, dict)
        else None
    )
    if not isinstance(attestation_relative, str):
        raise ValueError("prepare manifest lacks adapter checkpoint attestation")
    attestation_path, _ = _safe_run_file(
        prepared_run,
        attestation_relative,
        "adapter_checkpoint_attestation",
    )
    attestation, attestation_sha256 = _read_json_snapshot(attestation_path)
    if attestation_sha256 != manifest.get("adapter_checkpoint_attestation_sha256"):
        raise ValueError("adapter checkpoint attestation changed after prepare")
    adapter = config["models"]["tuned"]["adapter"]
    if (
        attestation.get("repo_id") != adapter["id"]
        or attestation.get("revision") != adapter["revision"]
    ):
        raise ValueError("adapter checkpoint attestation identity differs from the tuned arm")

    current_code = code_provenance(root)
    stored_code = manifest.get("code_provenance") or {}
    if stored_code.get("source_fingerprint") != current_code.get("source_fingerprint"):
        raise ValueError("evaluation source fingerprint changed after prepare")
    if stored_code.get("files") != current_code.get("files"):
        raise ValueError("evaluation source inventory changed after prepare")
    source_entries = current_code.get("files")
    if not isinstance(source_entries, list) or not source_entries:
        raise ValueError("code provenance has no source inventory")

    destination = staging_dir.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite staging directory: {destination}")
    if not destination.parent.exists():
        raise FileNotFoundError(f"staging parent does not exist: {destination.parent}")
    if not owner.strip() or "/" in owner or not _SLUG_RE.fullmatch(slug):
        raise ValueError("invalid Kaggle owner or dataset slug")

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    payload_root = temporary / "payload" / "top-papers-graph"
    try:
        temporary.mkdir()
        for entry in source_entries:
            relative_text = entry.get("path") if isinstance(entry, dict) else None
            if not isinstance(relative_text, str):
                raise ValueError("malformed code provenance file entry")
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe code provenance path: {relative_text}")
            if source_path_is_sensitive(relative):
                raise ValueError(f"credential-like path is forbidden in code provenance: {relative_text}")
            if not source_path_is_evaluation_source(relative):
                raise ValueError(f"unreviewed source path is forbidden in code provenance: {relative_text}")
            if "kaggle" in relative.parts and _GENERATED_PARTS.intersection(relative.parts):
                raise ValueError(f"generated Kaggle file contaminated prepare: {relative_text}")
            source = root / relative
            if _sha256(source) != entry.get("sha256"):
                raise ValueError(f"source hash changed during payload build: {relative_text}")
            _copy_file(source, payload_root / relative, str(entry["sha256"]))

        run_destination = payload_root / run_relative
        if run_destination.exists():
            raise ValueError("prepared run overlaps code provenance inventory")
        for relative, (source, expected) in sorted(
            _prepared_files(manifest, prepared_run, prepare_manifest_sha256).items(),
            key=lambda item: item[0].as_posix(),
        ):
            _copy_file(source, run_destination / relative, expected)
        if not (payload_root / config_relative).is_file():
            raise ValueError("config is absent from the exact code provenance inventory")

        metadata = {
            "title": slug,
            "id": f"{owner}/{slug}",
            "licenses": [{"name": "GPL-3.0"}],
            "isPrivate": True,
        }
        _write_json(temporary / "dataset-metadata.json", metadata)
        payload_manifest = {
            "artifact_version": 3,
            "dataset_slug": f"{owner}/{slug}",
            "workflow_mode": workflow_mode,
            "config_relative": config_relative.as_posix(),
            "config_fingerprint": fingerprint,
            "experiment_id": config["experiment"]["id"],
            "n_items": config["power"]["n_items"],
            "conditions": config["conditions"],
            "expected_rows": config["power"]["n_items"] * len(config["conditions"]),
            "prepared_run_relative": run_relative.as_posix(),
            "source_fingerprint": current_code["source_fingerprint"],
            "files": _inventory(temporary / "payload"),
        }
        _write_json(temporary / "payload_manifest.json", payload_manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("fp16-primary", "nf4-sensitivity"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_payload(
            args.repo_root,
            args.config,
            args.prepared_run,
            args.staging_dir,
            args.owner,
            args.slug,
            args.mode,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
