# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Download, audit, and freeze inputs for a VLM A/B experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import (
    audit_benchmark,
    canonical_paper_id,
    load_jsonl,
    paper_identity_errors,
    write_audit_report,
)
from .config import config_fingerprint
from .contracts import PUBLICATION_SCHEMA_SHA256
from .identities import normalize_identity
from .paths import resolve_dataset_file


_TECHNICAL_BLOCKERS = {
    "duplicate_sample_id",
    "empty_benchmark",
    "image_placeholder_mismatch",
    "missing_image",
    "schema_error",
    "unsafe_image_path",
}
_STRICT_WARNING_BLOCKERS = (
    "duplicate_normalized_prompt",
    "within_row_duplicate_image_bytes",
)
_MODEL_ROW_FIELDS = (
    "sample_id",
    "benchmark_version",
    "task_family",
    "language",
    "split",
    "topic",
    "case_id",
    "stratum",
    "primary_endpoint",
    "paper_title",
    "paper_id",
    "year",
    "evidence_kind",
    "page_hint",
    "model_task_prompt",
    "messages",
    "images",
    "generation_target_schema",
    "split_provenance",
)


class PublicationGateError(RuntimeError):
    """Raised after artifacts are written when the strict publication gate fails."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_strict_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKeyError as exc:
        raise PublicationGateError(f"{label} contains duplicate JSON key {exc.args[0]!r}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PublicationGateError(f"cannot read strict JSON {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicationGateError(f"{label} must contain a JSON object")
    pending: list[Any] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            raise PublicationGateError(f"{label} contains a non-finite JSON number")
        if isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise PublicationGateError(f"{label} must be a non-empty ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicationGateError(f"{label} is not a valid ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PublicationGateError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_source_records(
    config: Mapping[str, Any],
    paths: list[Path],
    rows_by_file: Mapping[Path, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    sources = config.get("training_audit", {}).get("sources", [])
    expected_file_count = sum(len(source["files"]) for source in sources)
    if len(paths) < expected_file_count:
        raise PublicationGateError(
            "every configured training source file must be present for lineage verification"
        )
    records: list[dict[str, Any]] = []
    file_index = 0
    for source in sources:
        files: list[dict[str, Any]] = []
        for reference in source["files"]:
            path = paths[file_index]
            file_index += 1
            files.append(
                {
                    "path": reference,
                    "sha256": _sha256(path),
                    "row_count": len(rows_by_file[path]),
                }
            )
        records.append(
            {
                "repo_id": source["repo_id"],
                "repo_type": source["repo_type"],
                "revision": source["revision"],
                "files": files,
            }
        )
    return records


def _validate_publication_schemas(
    rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]] | None,
    repo_root: Path,
) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except (ImportError, AttributeError) as exc:
        raise PublicationGateError(
            "jsonschema with Draft 2020-12 support is required for strict preparation"
        ) from exc

    schema_root = repo_root / "experiments" / "vlm_ab_evaluation" / "schemas"
    paths = {
        "benchmark_schema_sha256": schema_root / "publication_benchmark_row.schema.json",
        "provenance_schema_sha256": schema_root / "publication_provenance_row.schema.json",
    }
    validators: dict[str, Any] = {}
    checker = FormatChecker()
    for hash_name, path in paths.items():
        try:
            raw = path.read_bytes()
            schema = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PublicationGateError(
                f"cannot read canonical publication schema: {path.name}"
            ) from exc
        if hashlib.sha256(raw).hexdigest() != PUBLICATION_SCHEMA_SHA256[hash_name]:
            raise PublicationGateError(f"canonical publication schema hash changed: {path.name}")
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:
            raise PublicationGateError(
                f"canonical publication schema is invalid: {path.name}"
            ) from exc
        validators[hash_name] = Draft202012Validator(schema, format_checker=checker)

    if provenance_rows is None:
        raise PublicationGateError("strict preparation requires canonical provenance rows")
    for label, values, validator_name in (
        ("benchmark", rows, "benchmark_schema_sha256"),
        ("provenance", provenance_rows, "provenance_schema_sha256"),
    ):
        validator = validators[validator_name]
        for index, value in enumerate(values):
            errors = sorted(
                validator.iter_errors(value),
                key=lambda error: (
                    tuple(str(part) for part in error.absolute_path),
                    tuple(str(part) for part in error.absolute_schema_path),
                    error.message,
                ),
            )
            if errors:
                error = errors[0]
                field = "/".join(str(part) for part in error.absolute_path) or "<root>"
                raise PublicationGateError(
                    f"strict {label} row {index} violates the canonical publication schema "
                    f"at {field}: {error.message}"
                )
            if paper_identity_errors(value):
                raise PublicationGateError(
                    f"strict {label} row {index} has an invalid or conflicting paper identity"
                )
            if value.get("paper_id") != canonical_paper_id(value.get("paper_id")):
                raise PublicationGateError(
                    f"strict {label} row {index} paper_id is not in canonical form"
                )
            if label == "benchmark" and any(
                not reference.startswith("assets/images/") for reference in value["images"]
            ):
                raise PublicationGateError(
                    f"strict benchmark row {index} image paths must start with assets/images/"
                )
            if label == "provenance":
                for image_index, image in enumerate(value["images"]):
                    normalized_verifiers = [
                        normalize_identity(verifier, ascii_reviewer=True)
                        for verifier in image["verified_by"]
                    ]
                    if any(not verifier for verifier in normalized_verifiers) or len(
                        set(normalized_verifiers)
                    ) != len(normalized_verifiers):
                        raise PublicationGateError(
                            f"strict provenance row {index} image {image_index} requires "
                            "normalized-distinct safe verifier identifiers"
                        )


def _validate_training_lineage_schema(value: Mapping[str, Any], repo_root: Path) -> None:
    try:
        from jsonschema import Draft202012Validator
    except (ImportError, AttributeError) as exc:
        raise PublicationGateError(
            "jsonschema with Draft 2020-12 support is required for strict preparation"
        ) from exc

    path = (
        repo_root
        / "experiments"
        / "vlm_ab_evaluation"
        / "schemas"
        / "training_lineage_manifest.schema.json"
    )
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationGateError("cannot load the training lineage JSON Schema") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise PublicationGateError("the training lineage JSON Schema is invalid") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
            error.message,
        ),
    )
    if errors:
        error = errors[0]
        field = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise PublicationGateError(
            f"training lineage manifest violates its canonical schema at {field}: {error.message}"
        )


def _validate_adapter_base_metadata(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected = config["models"]["tuned"]["base_model"]
    if value.get("base_model_name_or_path") != expected["id"]:
        raise PublicationGateError(
            "adapter_config.json base_model_name_or_path differs from the configured base model"
        )
    if value.get("revision") != expected["revision"]:
        raise PublicationGateError(
            "adapter_config.json revision must equal the immutable configured base revision"
        )


def _apply_gate_failure(report: dict[str, Any], *, code: str, description: str, error: str) -> None:
    sample_ids = sorted(report.get("per_sample_findings", {}))
    finding = {
        "code": code,
        "message": description,
        "sample_ids": sample_ids,
        "details": {"error": error},
    }
    report["critical_findings"].append(finding)
    report["critical_findings"].sort(
        key=lambda value: (
            str(value.get("code", "")),
            json.dumps(value.get("details", {}), ensure_ascii=False, sort_keys=True),
        )
    )
    summary = report["summary"]
    summary["critical_findings"] += 1
    summary["critical_finding_count"] += 1
    critical_by_code = Counter(summary["critical_by_code"])
    critical_by_code[code] += 1
    summary["critical_by_code"] = dict(sorted(critical_by_code.items()))
    summary["eligible_samples"] = 0
    summary["eligible_sample_count"] = 0
    report["eligible_sample_ids"] = []
    for sample_report in report.get("per_sample_findings", {}).values():
        sample_report["eligible"] = False
        sample_report["critical_findings"].append(dict(finding))
        sample_report["critical_findings"].sort(
            key=lambda value: (
                str(value.get("code", "")),
                json.dumps(value.get("details", {}), ensure_ascii=False, sort_keys=True),
            )
        )
    report["status"] = "fail"
    report["publication_ready"] = False


def _apply_publication_schema_failure(report: dict[str, Any], message: str) -> None:
    _apply_gate_failure(
        report,
        code="canonical_publication_schema_error",
        description="The strict publication JSON Schema contract was not satisfied.",
        error=message,
    )


def _apply_lineage_schema_failure(report: dict[str, Any], message: str) -> None:
    _apply_gate_failure(
        report,
        code="canonical_training_lineage_schema_error",
        description="The strict training lineage JSON Schema contract was not satisfied.",
        error=message,
    )


def _apply_adapter_metadata_failure(report: dict[str, Any], message: str) -> None:
    _apply_gate_failure(
        report,
        code="adapter_base_metadata_error",
        description="The evaluated adapter does not bind the configured immutable base revision.",
        error=message,
    )


def code_provenance(repo_root: Path) -> dict[str, Any]:
    files: list[dict[str, str]] = []
    candidates = [repo_root / "src" / "scireason" / "vlm_ab"]
    experiment_root = repo_root / "experiments" / "vlm_ab_evaluation"
    candidates.append(experiment_root)
    package_init = repo_root / "src" / "scireason" / "__init__.py"
    pyproject = repo_root / "pyproject.toml"
    source_files: set[Path] = set()
    for candidate in candidates:
        if candidate.is_dir():
            source_files.update(
                path
                for path in candidate.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    if package_init.is_file():
        source_files.add(package_init)
    if pyproject.is_file():
        source_files.add(pyproject)
    for path in sorted(source_files, key=lambda item: item.as_posix()):
        files.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(path),
            }
        )
    fingerprint = hashlib.sha256(
        json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    git_head: str | None = None
    git_dirty: bool | None = None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src/scireason/__init__.py",
                "src/scireason/vlm_ab",
                "experiments/vlm_ab_evaluation",
                "pyproject.toml",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        git_head = head.stdout.strip()
        git_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "source_fingerprint": fingerprint,
        "git_head": git_head,
        "git_dirty": git_dirty,
        "files": files,
    }


def _verified_power_plan(
    output_dir: Path,
    config: Mapping[str, Any],
    current_code: Mapping[str, Any],
    *,
    required: bool,
) -> tuple[Path | None, str | None]:
    path = output_dir / "design" / "power_plan.json"
    if not path.is_file():
        if required:
            raise PublicationGateError(
                "strict preparation requires the immutable power plan to be created first"
            )
        return None, None
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationGateError(f"cannot read preregistered power plan: {path}") from exc
    if not isinstance(plan, Mapping):
        raise PublicationGateError("preregistered power plan must be a JSON object")
    if plan.get("config_fingerprint") != config_fingerprint(config):
        raise PublicationGateError("power plan belongs to another experiment configuration")
    if plan.get("code_fingerprint") != current_code.get("source_fingerprint"):
        raise PublicationGateError("evaluation source changed after the power plan")
    created_at = _parse_timestamp(plan.get("created_at"), "power plan created_at")
    if created_at > datetime.now(timezone.utc):
        raise PublicationGateError("power plan preregistration timestamp is in the future")
    return path.resolve(strict=True), _sha256(path)


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError(f"refusing to overwrite symlink: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary copy path already exists: {temporary}")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.resolve(strict=True)


def _bundle_destination(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise PublicationGateError(f"unsafe bundled image path: {reference!r}")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = destination.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise PublicationGateError(f"bundled image path escapes inputs: {reference!r}") from exc
    return resolved_parent / destination.name


def resolve_prepare_manifest(
    manifest: Mapping[str, Any], artifact_root: str | Path | None = None
) -> dict[str, Any]:
    """Resolve relocatable prepare paths against the run directory."""

    result = dict(manifest)
    base_value = artifact_root if artifact_root is not None else manifest.get("manifest_root")
    if base_value is None:
        raise PublicationGateError("prepare manifest has no artifact root")
    base = Path(base_value).resolve(strict=True)
    relative_paths = manifest.get("artifact_paths", {})
    if not isinstance(relative_paths, Mapping):
        raise PublicationGateError("prepare manifest artifact_paths is malformed")
    required_paths = {
        "dataset_root",
        "benchmark_file",
        "frozen_benchmark",
        "audit_json",
        "audit_markdown",
    }
    if any(
        not isinstance(relative_paths.get(key), str) or not relative_paths[key]
        for key in required_paths
    ):
        raise PublicationGateError("prepare manifest is missing a required artifact path")

    def resolve_relative(value: Any, label: str) -> Path:
        relative = Path(str(value))
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise PublicationGateError(f"prepare manifest has an unsafe {label} path")
        resolved = (base / relative).resolve(strict=True)
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise PublicationGateError(f"prepare manifest {label} path escapes the run") from exc
        return resolved

    for key in (
        "dataset_root",
        "benchmark_file",
        "provenance_file",
        "frozen_benchmark",
        "audit_json",
        "audit_markdown",
        "power_plan",
        "training_lineage_manifest",
        "adapter_config",
    ):
        relative = relative_paths.get(key)
        if relative is not None:
            result[key] = str(resolve_relative(relative, key))
    training_entries: list[dict[str, Any]] = []
    for entry in manifest.get("training_files", []):
        if not isinstance(entry, Mapping):
            raise PublicationGateError("prepare manifest contains a malformed training file entry")
        resolved_entry = dict(entry)
        if entry.get("relative_path") is not None:
            resolved_entry["path"] = str(resolve_relative(entry["relative_path"], "training file"))
        training_entries.append(resolved_entry)
    result["training_files"] = training_entries
    result["prepare_manifest"] = str(base / "prepare_manifest.json")
    return result


def verify_prepare_manifest(
    manifest: Mapping[str, Any], artifact_root: str | Path | None = None
) -> None:
    """Re-hash every frozen/audited file and image before a downstream stage."""

    resolved_manifest = resolve_prepare_manifest(manifest, artifact_root)
    declared_schema_hashes = manifest.get("publication_schema_hashes")
    if declared_schema_hashes != PUBLICATION_SCHEMA_SHA256:
        raise PublicationGateError("prepare manifest publication schema hashes are invalid")
    checks = [
        (
            resolved_manifest.get("benchmark_file"),
            resolved_manifest.get("benchmark_file_sha256"),
        ),
        (
            resolved_manifest.get("provenance_file"),
            resolved_manifest.get("provenance_file_sha256"),
        ),
        (
            resolved_manifest.get("frozen_benchmark"),
            resolved_manifest.get("frozen_benchmark_sha256"),
        ),
        (resolved_manifest.get("audit_json"), resolved_manifest.get("audit_json_sha256")),
        (
            resolved_manifest.get("audit_markdown"),
            resolved_manifest.get("audit_markdown_sha256"),
        ),
        (
            resolved_manifest.get("power_plan"),
            resolved_manifest.get("power_plan_sha256"),
        ),
        (
            resolved_manifest.get("training_lineage_manifest"),
            resolved_manifest.get("training_lineage_manifest_sha256"),
        ),
        (
            resolved_manifest.get("adapter_config"),
            resolved_manifest.get("adapter_config_sha256"),
        ),
    ]
    for entry in resolved_manifest.get("training_files", []):
        if not isinstance(entry, Mapping):
            raise PublicationGateError("prepare manifest contains a malformed training file entry")
        checks.append((entry.get("path"), entry.get("sha256")))
    for raw_path, expected in checks:
        if raw_path is None and expected is None:
            continue
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise PublicationGateError("prepare manifest has an incomplete file hash entry")
        try:
            actual = _sha256(Path(raw_path).resolve(strict=True))
        except OSError as exc:
            raise PublicationGateError(f"prepared artifact is missing: {raw_path}") from exc
        if actual != expected:
            raise PublicationGateError(f"prepared artifact hash changed: {raw_path}")

    audit_path = Path(str(resolved_manifest.get("audit_json") or ""))
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationGateError(f"cannot re-read benchmark audit: {audit_path}") from exc
    dataset_root = Path(str(resolved_manifest.get("dataset_root") or "")).resolve(strict=True)
    frozen_sample_ids = set(str(value) for value in manifest.get("frozen_sample_ids", []))
    image_cache: dict[Path, str] = {}
    for sample_id, sample in (audit.get("per_sample_findings") or {}).items():
        if frozen_sample_ids and str(sample_id) not in frozen_sample_ids:
            continue
        if not isinstance(sample, Mapping):
            continue
        for record in sample.get("image_hashes", []):
            if not isinstance(record, Mapping):
                raise PublicationGateError("audit contains a malformed image hash entry")
            relative = record.get("path")
            expected = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise PublicationGateError("audit contains an incomplete image hash entry")
            try:
                image_path = resolve_dataset_file(dataset_root, Path(relative))
            except (FileNotFoundError, ValueError) as exc:
                raise PublicationGateError(
                    f"audited image is missing or unsafe: {relative}"
                ) from exc
            if image_path not in image_cache:
                image_cache[image_path] = _sha256(image_path)
            if image_cache[image_path] != expected:
                raise PublicationGateError(f"audited image hash changed: {relative}")


def _safe_repo_file(root: Path, relative: str) -> Path:
    try:
        candidate = resolve_dataset_file(root, Path(relative))
    except ValueError as exc:
        raise RuntimeError(f"configured repository file escapes snapshot root: {relative}") from exc
    return candidate


def _snapshot(
    repo_id: str,
    repo_type: str,
    revision: str,
    allow_patterns: list[str],
    cache_dir: Path | None,
) -> tuple[Path, str]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - base project dependency
        raise RuntimeError("huggingface_hub is required to prepare remote assets") from exc
    try:
        info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
    except Exception as exc:
        raise RuntimeError(
            f"cannot resolve pinned Hugging Face {repo_type} {repo_id}@{revision}; "
            "set HF_TOKEN when the Hub rate-limits anonymous requests"
        ) from exc
    resolved = str(info.sha)
    if resolved.lower() != revision.lower():
        raise RuntimeError(
            f"{repo_type} {repo_id} resolved to {resolved}, expected pinned {revision}"
        )
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_dir) if cache_dir else None,
            max_workers=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"cannot download pinned Hugging Face {repo_type} {repo_id}@{revision}; "
            "check HF_TOKEN, network access, and cache integrity, then resume"
        ) from exc
    return Path(path).resolve(strict=True), resolved


def _download_inputs(
    config: Mapping[str, Any], cache_dir: Path | None
) -> tuple[Path, list[Path], Path | None, Path, list[dict[str, Any]]]:
    benchmark = config["benchmark"]
    patterns = [benchmark["data_file"], "assets/images/**"]
    if benchmark.get("provenance_file"):
        patterns.append(benchmark["provenance_file"])
    root, resolved = _snapshot(
        benchmark["repo_id"],
        "dataset",
        benchmark["revision"],
        patterns,
        cache_dir,
    )
    sources = [
        {
            "repo_id": benchmark["repo_id"],
            "repo_type": "dataset",
            "configured_revision": benchmark["revision"],
            "resolved_revision": resolved,
            "snapshot_root": str(root),
        }
    ]
    training_files: list[Path] = []
    for source in config.get("training_audit", {}).get("sources", []):
        source_root, source_revision = _snapshot(
            source["repo_id"],
            source["repo_type"],
            source["revision"],
            list(source["files"]),
            cache_dir,
        )
        training_files.extend(_safe_repo_file(source_root, item) for item in source["files"])
        sources.append(
            {
                "repo_id": source["repo_id"],
                "repo_type": source["repo_type"],
                "configured_revision": source["revision"],
                "resolved_revision": source_revision,
                "files": list(source["files"]),
                "snapshot_root": str(source_root),
            }
        )
    lineage_path: Path | None = None
    lineage = config.get("training_audit", {}).get("lineage_manifest")
    if isinstance(lineage, Mapping):
        lineage_root, lineage_revision = _snapshot(
            lineage["repo_id"],
            lineage["repo_type"],
            lineage["revision"],
            [lineage["file"]],
            cache_dir,
        )
        lineage_path = _safe_repo_file(lineage_root, lineage["file"])
        sources.append(
            {
                "repo_id": lineage["repo_id"],
                "repo_type": lineage["repo_type"],
                "configured_revision": lineage["revision"],
                "resolved_revision": lineage_revision,
                "files": [lineage["file"]],
                "role": "training_lineage_manifest",
                "snapshot_root": str(lineage_root),
            }
        )
    adapter = config["models"]["tuned"]["adapter"]
    adapter_root, adapter_revision = _snapshot(
        adapter["id"],
        "model",
        adapter["revision"],
        ["adapter_config.json"],
        cache_dir,
    )
    adapter_config_path = _safe_repo_file(adapter_root, "adapter_config.json")
    sources.append(
        {
            "repo_id": adapter["id"],
            "repo_type": "model",
            "configured_revision": adapter["revision"],
            "resolved_revision": adapter_revision,
            "files": ["adapter_config.json"],
            "role": "evaluated_adapter_metadata",
            "snapshot_root": str(adapter_root),
        }
    )
    return root, training_files, lineage_path, adapter_config_path, sources


def _technical_sample_ids(report: Mapping[str, Any]) -> set[str]:
    selected: set[str] = set()
    per_sample = report.get("per_sample_findings", {})
    if not isinstance(per_sample, Mapping):
        return selected
    for sample_id, details in per_sample.items():
        if not isinstance(details, Mapping) or str(sample_id).startswith("<row:"):
            continue
        findings = details.get("critical_findings", [])
        codes = {str(finding.get("code")) for finding in findings if isinstance(finding, Mapping)}
        if not codes.intersection(_TECHNICAL_BLOCKERS):
            selected.add(str(sample_id))
    return selected


def _freeze_rows(
    rows: list[dict[str, Any]], allowed_ids: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    counts = Counter(str(row.get("sample_id") or "") for row in rows)
    frozen: list[dict[str, Any]] = []
    excluded: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or counts[sample_id] != 1 or sample_id not in allowed_ids:
            excluded.append(sample_id or "<missing>")
            continue
        frozen.append({field: row[field] for field in _MODEL_ROW_FIELDS if field in row})
    frozen.sort(key=lambda row: row["sample_id"])
    return frozen, sorted(excluded)


def verify_prepared_audit(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the audit and frozen subset from the bundled immutable inputs."""

    prepared = resolve_prepare_manifest(manifest, artifact_root)
    try:
        stored_report = json.loads(Path(prepared["audit_json"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationGateError("cannot re-read benchmark audit") from exc
    if not isinstance(stored_report, Mapping):
        raise PublicationGateError("benchmark audit must contain a JSON object")
    rows = load_jsonl(prepared["benchmark_file"])
    provenance_rows = (
        load_jsonl(prepared["provenance_file"]) if prepared.get("provenance_file") else None
    )
    training_paths = [Path(entry["path"]) for entry in prepared.get("training_files", [])]
    training_rows_by_file = {path: load_jsonl(path) for path in training_paths}
    training_rows = [row for path in training_paths for row in training_rows_by_file[path]]
    exploratory = manifest.get("exploratory")
    if not isinstance(exploratory, bool):
        raise PublicationGateError("prepare manifest has no valid exploratory mode")
    strict_audit = not exploratory
    publication_schema_failure = None
    if strict_audit:
        try:
            _validate_publication_schemas(
                rows,
                provenance_rows,
                Path(__file__).resolve().parents[3],
            )
        except PublicationGateError as exc:
            publication_schema_failure = str(exc)
    expected_sources = _training_source_records(
        config,
        training_paths,
        training_rows_by_file,
    )
    training_lineage = None
    lineage_schema_failure = None
    if prepared.get("training_lineage_manifest"):
        training_lineage = _load_strict_json_object(
            Path(prepared["training_lineage_manifest"]),
            "training lineage manifest",
        )
        if strict_audit:
            try:
                _validate_training_lineage_schema(
                    training_lineage,
                    Path(__file__).resolve().parents[3],
                )
            except PublicationGateError as exc:
                lineage_schema_failure = str(exc)
    adapter_metadata_failure = None
    if strict_audit:
        adapter_path = prepared.get("adapter_config")
        if not adapter_path:
            adapter_metadata_failure = "strict preparation has no bundled adapter_config.json"
        else:
            try:
                adapter_config = _load_strict_json_object(
                    Path(adapter_path),
                    "adapter_config.json",
                )
                _validate_adapter_base_metadata(adapter_config, config)
            except PublicationGateError as exc:
                adapter_metadata_failure = str(exc)

    report = audit_benchmark(
        rows,
        prepared["dataset_root"],
        provenance_rows=provenance_rows,
        training_rows=training_rows if training_paths else None,
        training_lineage=training_lineage,
        expected_training_sources=expected_sources,
        require_training_lineage=(
            strict_audit
            or bool(config.get("training_audit", {}).get("require_lineage_manifest", False))
        ),
        require_provenance_order=strict_audit,
        require_citation=strict_audit,
        blocked_warning_codes=_STRICT_WARNING_BLOCKERS if strict_audit else (),
        minimum_primary_papers=(int(config["power"]["n_items"]) if strict_audit else 0),
        require_gold=bool(config["benchmark"].get("require_gold", False)),
        require_complete_provenance=bool(
            config["benchmark"].get("require_complete_provenance", False)
        ),
        require_split_provenance=bool(config["benchmark"].get("require_split_provenance", False)),
        primary_strata=config["statistics"]["primary_strata"],
        audit_version=stored_report.get("audit_version"),
    )
    if publication_schema_failure is not None:
        _apply_publication_schema_failure(report, publication_schema_failure)
    if lineage_schema_failure is not None:
        _apply_lineage_schema_failure(report, lineage_schema_failure)
    if adapter_metadata_failure is not None:
        _apply_adapter_metadata_failure(report, adapter_metadata_failure)
    if stored_report != report:
        raise PublicationGateError("stored benchmark audit differs from a fresh recomputation")

    allowed_ids = (
        _technical_sample_ids(report) if exploratory else set(report["eligible_sample_ids"])
    )
    expected_frozen, expected_excluded = _freeze_rows(rows, allowed_ids)
    actual_frozen = load_jsonl(prepared["frozen_benchmark"])
    if actual_frozen != expected_frozen:
        raise PublicationGateError("frozen benchmark differs from the recomputed audit subset")
    expected_manifest_values = {
        "input_rows": len(rows),
        "frozen_rows": len(expected_frozen),
        "excluded_rows": len(expected_excluded),
        "excluded_sample_ids": expected_excluded,
        "frozen_sample_ids": [row["sample_id"] for row in expected_frozen],
        "benchmark_audit_status": report["status"],
    }
    if any(manifest.get(key) != value for key, value in expected_manifest_values.items()):
        raise PublicationGateError("prepare manifest counts differ from the recomputed audit")
    return report


def prepare_experiment(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    benchmark_dir: str | Path | None = None,
    training_files: Iterable[str | Path] | None = None,
    cache_dir: str | Path | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    """Audit and freeze the exact model-facing rows for a future inference run."""

    root = Path(repo_root).resolve(strict=True)
    output_dir = (root / config["experiment"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_code = code_provenance(root)
    power_plan_path, power_plan_sha256 = _verified_power_plan(
        output_dir,
        config,
        current_code,
        required=bool(
            not exploratory and config["experiment"].get("require_preregistered_plan", False)
        ),
    )
    if benchmark_dir is None:
        dataset_root, remote_training, lineage_path, adapter_config_path, sources = (
            _download_inputs(config, Path(cache_dir).resolve() if cache_dir else None)
        )
    else:
        dataset_root = Path(benchmark_dir).resolve(strict=True)
        remote_training = []
        lineage_path = None
        adapter_config_path = None
        sources = [
            {
                "repo_id": config["benchmark"]["repo_id"],
                "repo_type": "dataset",
                "configured_revision": config["benchmark"]["revision"],
                "resolved_revision": "local-override-not-remotely-verified",
                "snapshot_root": str(dataset_root),
            }
        ]
    configured_training = [Path(path).resolve(strict=True) for path in (training_files or [])]
    if configured_training and not exploratory:
        raise PublicationGateError(
            "local training-file overrides are allowed only in explicitly exploratory runs"
        )
    all_training_files = remote_training + configured_training
    if benchmark_dir is not None and not exploratory:
        raise PublicationGateError(
            "local benchmark overrides are allowed only in explicitly exploratory runs"
        )
    if config.get("training_audit", {}).get("sources") and not all_training_files:
        raise PublicationGateError("configured training contamination sources were not audited")

    benchmark_path = _safe_repo_file(dataset_root, config["benchmark"]["data_file"])
    rows = load_jsonl(benchmark_path)
    provenance_rows = None
    provenance_path = None
    if config["benchmark"].get("provenance_file"):
        provenance_path = _safe_repo_file(dataset_root, config["benchmark"]["provenance_file"])
        provenance_rows = load_jsonl(provenance_path)
    train_rows: list[dict[str, Any]] = []
    training_rows_by_file: dict[Path, list[dict[str, Any]]] = {}
    for path in all_training_files:
        file_rows = load_jsonl(path)
        training_rows_by_file[path] = file_rows
        train_rows.extend(file_rows)
    configured_sources = config.get("training_audit", {}).get("sources", [])
    lineage_input_files = remote_training if remote_training else configured_training
    expected_file_count = sum(len(source["files"]) for source in configured_sources)
    if configured_sources and len(lineage_input_files) != expected_file_count:
        raise PublicationGateError(
            "every configured training source file must be present for lineage verification"
        )
    expected_training_sources = _training_source_records(
        config,
        lineage_input_files,
        training_rows_by_file,
    )
    training_lineage = None
    lineage_schema_failure = None
    if lineage_path is not None:
        training_lineage = _load_strict_json_object(lineage_path, "training lineage manifest")
        if not exploratory:
            try:
                _validate_training_lineage_schema(training_lineage, root)
            except PublicationGateError as exc:
                lineage_schema_failure = str(exc)

    adapter_metadata_failure = None
    if not exploratory:
        if adapter_config_path is None:
            adapter_metadata_failure = (
                "strict preparation requires adapter_config.json at revision R"
            )
        else:
            try:
                adapter_config = _load_strict_json_object(
                    adapter_config_path,
                    "adapter_config.json",
                )
                _validate_adapter_base_metadata(adapter_config, config)
            except PublicationGateError as exc:
                adapter_metadata_failure = str(exc)

    report = audit_benchmark(
        rows,
        dataset_root,
        provenance_rows=provenance_rows,
        training_rows=train_rows if all_training_files else None,
        training_lineage=training_lineage,
        expected_training_sources=expected_training_sources,
        require_training_lineage=(
            not exploratory
            or bool(config.get("training_audit", {}).get("require_lineage_manifest", False))
        ),
        require_provenance_order=not exploratory,
        require_citation=not exploratory,
        blocked_warning_codes=_STRICT_WARNING_BLOCKERS if not exploratory else (),
        minimum_primary_papers=(int(config["power"]["n_items"]) if not exploratory else 0),
        require_gold=bool(config["benchmark"].get("require_gold", False)),
        require_complete_provenance=bool(
            config["benchmark"].get("require_complete_provenance", False)
        ),
        require_split_provenance=bool(config["benchmark"].get("require_split_provenance", False)),
        primary_strata=config["statistics"]["primary_strata"],
    )
    if not exploratory:
        try:
            _validate_publication_schemas(rows, provenance_rows, root)
        except PublicationGateError as exc:
            _apply_publication_schema_failure(report, str(exc))
        if lineage_schema_failure is not None:
            _apply_lineage_schema_failure(report, lineage_schema_failure)
        if adapter_metadata_failure is not None:
            _apply_adapter_metadata_failure(report, adapter_metadata_failure)
    audit_json = output_dir / "audit" / "benchmark_audit.json"
    audit_markdown = output_dir / "audit" / "benchmark_audit.md"
    write_audit_report(report, audit_json, audit_markdown)
    require_clean_code = bool(config["experiment"].get("require_clean_code", False))
    code_gate_passed = not require_clean_code or (
        current_code["git_dirty"] is False
        and isinstance(current_code.get("git_head"), str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", current_code["git_head"]))
    )

    if exploratory:
        allowed_ids = _technical_sample_ids(report)
        scope = "exploratory_not_for_publication"
    else:
        allowed_ids = set(report["eligible_sample_ids"])
        scope = (
            "publication_ready" if report["publication_ready"] and code_gate_passed else "blocked"
        )
    frozen, excluded = _freeze_rows(rows, allowed_ids)
    bundled_dataset_root = output_dir / "inputs" / "dataset"
    bundled_dataset_root.mkdir(parents=True, exist_ok=True)
    copied_images: set[str] = set()
    for row in rows:
        for reference in row.get("images", []):
            if not isinstance(reference, str) or reference in copied_images:
                continue
            try:
                source_image = resolve_dataset_file(dataset_root, Path(reference))
                destination_image = _bundle_destination(bundled_dataset_root, reference)
            except (FileNotFoundError, ValueError, PublicationGateError):
                continue
            _copy_file(source_image, destination_image)
            copied_images.add(reference)

    source_dir = output_dir / "inputs" / "source"
    bundled_benchmark = _copy_file(
        benchmark_path, source_dir / f"benchmark{benchmark_path.suffix or '.jsonl'}"
    )
    bundled_provenance = (
        _copy_file(
            provenance_path,
            source_dir / f"provenance{provenance_path.suffix or '.jsonl'}",
        )
        if provenance_path is not None
        else None
    )
    bundled_training = [
        _copy_file(path, source_dir / f"training_{index:02d}{path.suffix or '.jsonl'}")
        for index, path in enumerate(all_training_files)
    ]
    bundled_lineage = (
        _copy_file(lineage_path, source_dir / "training_lineage_manifest.json")
        if lineage_path is not None
        else None
    )
    bundled_adapter_config = (
        _copy_file(adapter_config_path, source_dir / "adapter_config.json")
        if adapter_config_path is not None
        else None
    )
    frozen_path = output_dir / "inputs" / "frozen_benchmark.jsonl"
    _write_jsonl(frozen_path, frozen)

    def relative(path: Path | None) -> str | None:
        return path.relative_to(output_dir).as_posix() if path is not None else None

    manifest = {
        "artifact_version": 1,
        "created_at": _utc_now(),
        "experiment_id": config["experiment"]["id"],
        "config_fingerprint": config_fingerprint(config),
        "code_provenance": current_code,
        "code_gate_passed": code_gate_passed,
        "exploratory": bool(exploratory),
        "result_scope": scope,
        "publication_ready": bool(
            report["publication_ready"] and code_gate_passed and not exploratory
        ),
        "benchmark_audit_status": report["status"],
        "publication_schema_hashes": dict(PUBLICATION_SCHEMA_SHA256),
        "manifest_root": str(output_dir),
        "artifact_paths": {
            "dataset_root": relative(bundled_dataset_root),
            "benchmark_file": relative(bundled_benchmark),
            "provenance_file": relative(bundled_provenance),
            "frozen_benchmark": relative(frozen_path),
            "audit_json": relative(audit_json),
            "audit_markdown": relative(audit_markdown),
            "power_plan": relative(power_plan_path),
            "training_lineage_manifest": relative(bundled_lineage),
            "adapter_config": relative(bundled_adapter_config),
        },
        "dataset_root": str(bundled_dataset_root),
        "benchmark_file": str(bundled_benchmark),
        "benchmark_file_sha256": _sha256(bundled_benchmark),
        "provenance_file": str(bundled_provenance) if bundled_provenance else None,
        "provenance_file_sha256": (_sha256(bundled_provenance) if bundled_provenance else None),
        "training_files": [
            {
                "path": str(path),
                "relative_path": relative(path),
                "sha256": _sha256(path),
            }
            for path in bundled_training
        ],
        "training_lineage_manifest": str(bundled_lineage) if bundled_lineage else None,
        "training_lineage_manifest_sha256": (_sha256(bundled_lineage) if bundled_lineage else None),
        "adapter_config": str(bundled_adapter_config) if bundled_adapter_config else None,
        "adapter_config_sha256": (
            _sha256(bundled_adapter_config) if bundled_adapter_config else None
        ),
        "sources": sources,
        "input_rows": len(rows),
        "frozen_rows": len(frozen),
        "excluded_rows": len(excluded),
        "excluded_sample_ids": excluded,
        "frozen_sample_ids": [row["sample_id"] for row in frozen],
        "frozen_benchmark": str(frozen_path),
        "frozen_benchmark_sha256": _sha256(frozen_path),
        "audit_json": str(audit_json),
        "audit_json_sha256": _sha256(audit_json),
        "audit_markdown": str(audit_markdown),
        "audit_markdown_sha256": _sha256(audit_markdown),
        "power_plan": str(power_plan_path) if power_plan_path else None,
        "power_plan_sha256": power_plan_sha256,
    }
    manifest_path = output_dir / "prepare_manifest.json"
    _write_json(manifest_path, manifest)
    runtime_manifest = resolve_prepare_manifest(manifest, output_dir)
    if not exploratory and not runtime_manifest["publication_ready"]:
        raise PublicationGateError(
            f"publication gate failed; see {audit_markdown} and {manifest_path}"
        )
    if not frozen:
        raise PublicationGateError("no technically runnable rows remain after the benchmark audit")
    return runtime_manifest


__all__ = [
    "PublicationGateError",
    "code_provenance",
    "prepare_experiment",
    "resolve_prepare_manifest",
    "verify_prepared_audit",
    "verify_prepare_manifest",
]
