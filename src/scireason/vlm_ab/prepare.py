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
import stat
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapter_checkpoint import inspect_plain_fp32_lora_safetensors
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
_SENSITIVE_SOURCE_NAMES = {
    ".envrc",
    ".git-credentials",
    ".netrc",
    "credentials",
    "credentials.json",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "kaggle.json",
    "service-account.json",
    "service_account.json",
}
_SENSITIVE_SOURCE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
_EVALUATION_TOP_LEVEL_SOURCES = {
    "CAPACITY_150_PROTOCOL_RU.md",
    "DESIGN_RU.md",
    "EXPERT_REVIEW_RU.md",
    "NEXT_STEPS_RU.md",
    "PROJECT_STATUS_RU.md",
    "README.md",
    "REMEDIATION.md",
    "make_capacity150_config.py",
    "make_capacity150_remediation_config.py",
    "remediation_queue_v2_baseline_20260717.json",
    "remote_audit_baseline_20260717.json",
    "requirements.txt",
    "run_pipeline.py",
}
_KAGGLE_SOURCES = {
    "README_RU.md",
    "build_payload.py",
    "kaggle_api.ps1",
    "kernel_runner.py.template",
    "kernel_version_io.py",
    "make_nf4_config.py",
    "requirements-kaggle-nf4.txt",
    "requirements-kaggle.txt",
    "requirements-local.txt",
    "validate_state.py",
}
_DATASPHERE_SOURCES = {"job_config.yaml", "requirements.txt", "run_evaluation.sh"}
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


def _read_stable_bytes(path: Path, label: str) -> bytes:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise PublicationGateError(f"{label} must be a regular non-symlink file")
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            raw = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = os.lstat(path)
    except OSError as exc:
        raise PublicationGateError(f"cannot read {label}: {exc}") from exc

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
        or len(raw) != opened_after.st_size
    ):
        raise PublicationGateError(f"{label} changed while it was being read")
    return raw


def _load_strict_json_object_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKeyError as exc:
        raise PublicationGateError(f"{label} contains duplicate JSON key {exc.args[0]!r}") from exc
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
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


def _load_strict_json_object(path: Path, label: str) -> dict[str, Any]:
    return _load_strict_json_object_bytes(_read_stable_bytes(path, label), label)


def _load_strict_jsonl_bytes(raw: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise PublicationGateError(f"cannot decode strict JSONL {label}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        rows.append(
            _load_strict_json_object_bytes(
                line.encode("utf-8"),
                f"{label} line {line_number}",
            )
        )
    return rows


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
    if value.get("peft_type") != "LORA":
        raise PublicationGateError("adapter_config.json must declare peft_type=LORA")
    if value.get("bias") != "none":
        raise PublicationGateError("adapter_config.json must use bias=none")
    if value.get("modules_to_save") not in (None, []):
        raise PublicationGateError(
            "adapter_config.json modules_to_save is unsupported by the FP32 LoRA contract"
        )
    if value.get("trainable_token_indices") is not None:
        raise PublicationGateError(
            "adapter_config.json trainable_token_indices is unsupported by the FP32 LoRA contract"
        )
    exact_fields = {
        "peft_version": "0.19.1",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "init_lora_weights": True,
        "use_dora": False,
        "use_rslora": False,
        "use_qalora": False,
        "lora_bias": False,
        "ensure_weight_tying": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "loftq_config": {},
    }
    for field, expected_value in exact_fields.items():
        if value.get(field) != expected_value:
            raise PublicationGateError(
                f"adapter_config.json {field} must equal {expected_value!r} for plain LoRA"
            )
    for field in (
        "alora_invocation_tokens",
        "arrow_config",
        "corda_config",
        "eva_config",
        "layer_replication",
        "lora_ga_config",
        "megatron_config",
        "target_parameters",
        "use_bdlora",
    ):
        if value.get(field) is not None:
            raise PublicationGateError(f"adapter_config.json {field} is unsupported by plain LoRA")
    target_modules = value.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or not all(isinstance(item, str) and item.strip() for item in target_modules)
        or len(target_modules) != len(set(target_modules))
    ):
        raise PublicationGateError(
            "adapter_config.json target_modules must be a non-empty unique string array"
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


def source_path_is_sensitive(relative: Path) -> bool:
    """Return whether a repository-relative path may contain credentials."""

    lowered = tuple(part.casefold() for part in relative.parts)
    name = lowered[-1] if lowered else ""
    credential_words = ("credential", "password", "private_key", "private-key", "secret", "token")
    return bool(
        any(part.startswith(".") for part in lowered)
        or any(part in {"credentials", "secrets"} for part in lowered)
        or name in _SENSITIVE_SOURCE_NAMES
        or name == ".env"
        or name.startswith(".env.")
        or name.startswith("kaggle.json.")
        or any(word in name for word in credential_words)
        or Path(name).suffix in _SENSITIVE_SOURCE_SUFFIXES
    )


def source_path_is_evaluation_source(relative: Path) -> bool:
    """Allow only reviewed source locations required by the evaluation payload."""

    if source_path_is_sensitive(relative):
        return False
    parts = relative.parts
    if len(parts) == 1:
        return parts[0] == "pyproject.toml"
    if parts == ("src", "scireason", "__init__.py"):
        return True
    if parts[:3] == ("src", "scireason", "vlm_ab"):
        return len(parts) == 4 and relative.suffix == ".py"
    prefix = ("experiments", "vlm_ab_evaluation")
    if parts[:2] != prefix:
        return False
    nested = parts[2:]
    if len(nested) == 1:
        return nested[0] in _EVALUATION_TOP_LEVEL_SOURCES
    if len(nested) == 2 and nested[0] == "configs":
        return relative.suffix in {".yaml", ".yml"}
    if len(nested) == 2 and nested[0] == "schemas":
        return relative.suffix == ".json"
    if len(nested) == 2 and nested[0] == "datasphere":
        return nested[1] in _DATASPHERE_SOURCES
    if len(nested) == 2 and nested[0] == "kaggle":
        return nested[1] in _KAGGLE_SOURCES
    return False


def _source_fingerprint_candidate(repo_root: Path, path: Path) -> bool:
    relative = path.relative_to(repo_root)
    if any(
        part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "caches"}
        for part in relative.parts
    ):
        return False
    generated_kaggle_roots = {
        Path("experiments/vlm_ab_evaluation/kaggle") / name
        for name in ("build", "staging", "downloads")
    }
    return source_path_is_evaluation_source(relative) and not any(
        relative.is_relative_to(root) for root in generated_kaggle_roots
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
                if path.is_file() and _source_fingerprint_candidate(repo_root, path)
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
        "adapter_checkpoint_attestation",
        "assembly_manifest",
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
        (
            resolved_manifest.get("adapter_checkpoint_attestation"),
            resolved_manifest.get("adapter_checkpoint_attestation_sha256"),
        ),
        (
            resolved_manifest.get("assembly_manifest"),
            resolved_manifest.get("assembly_manifest_sha256"),
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


def _validate_archived_human_review(
    assembly: Mapping[str, Any],
    output_records: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    binding = assembly.get("human_review_evidence")
    expected_fields = {
        "artifact_version",
        "archive_root",
        "queue_manifest_path",
        "queue_manifest_sha256",
        "tasks_path",
        "tasks_sha256",
        "decision_template_path",
        "decision_template_sha256",
        "decisions_path",
        "decisions_sha256",
        "independent_attestation_required",
    }
    archive_root = "audit/human_review"
    paths = {
        "queue_manifest": f"{archive_root}/queue/queue_manifest.json",
        "tasks": f"{archive_root}/queue/tasks.jsonl",
        "decision_template": f"{archive_root}/queue/decision_template.jsonl",
        "decisions": f"{archive_root}/completed_decisions.jsonl",
    }
    if (
        not isinstance(binding, Mapping)
        or set(binding) != expected_fields
        or binding.get("artifact_version") != 3
        or binding.get("archive_root") != archive_root
        or binding.get("queue_manifest_path") != paths["queue_manifest"]
        or binding.get("tasks_path") != paths["tasks"]
        or binding.get("decision_template_path") != paths["decision_template"]
        or binding.get("decisions_path") != paths["decisions"]
        or binding.get("independent_attestation_required") is not True
    ):
        raise PublicationGateError("benchmark assembly human review binding is malformed")
    digest_fields = {
        "queue_manifest": "queue_manifest_sha256",
        "tasks": "tasks_sha256",
        "decision_template": "decision_template_sha256",
        "decisions": "decisions_sha256",
    }
    raw_files: dict[str, bytes] = {}
    for name, field in digest_fields.items():
        expected = binding.get(field)
        path = paths[name]
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or path not in output_records
            or output_records[path].get("sha256") != expected
        ):
            raise PublicationGateError("benchmark assembly human review bytes are not bound")
        raw = _read_stable_bytes(
            _safe_repo_file(dataset_root, path),
            f"archived human review {name}",
        )
        if hashlib.sha256(raw).hexdigest() != expected:
            raise PublicationGateError("archived human review bytes changed")
        raw_files[name] = raw
    actual_archive = {
        path for path in output_records if path.startswith(f"{archive_root}/")
    }
    if actual_archive != set(paths.values()):
        raise PublicationGateError("benchmark assembly human review archive is incomplete")
    queue = _load_strict_json_object_bytes(raw_files["queue_manifest"], "archived queue manifest")
    if (
        queue.get("artifact_version") != binding["artifact_version"]
        or queue.get("queue_fingerprint") != assembly.get("queue_fingerprint")
        or queue.get("config_fingerprint") != assembly.get("config_fingerprint")
        or queue.get("tasks_file") != "tasks.jsonl"
        or queue.get("tasks_sha256") != binding["tasks_sha256"]
        or queue.get("decision_template_file") != "decision_template.jsonl"
        or queue.get("decision_template_sha256") != binding["decision_template_sha256"]
    ):
        raise PublicationGateError("archived queue manifest differs from assembly bindings")
    fingerprint_payload = {
        "artifact_version": queue.get("artifact_version"),
        "config_fingerprint": queue.get("config_fingerprint"),
        "source_hashes": queue.get("source_hashes"),
        "schema_hashes": queue.get("schema_hashes"),
        "policy": queue.get("policy"),
        "task_count": queue.get("task_count"),
        "tasks_sha256": queue.get("tasks_sha256"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint != queue.get("queue_fingerprint"):
        raise PublicationGateError("archived queue fingerprint is invalid")
    tasks = _load_strict_jsonl_bytes(raw_files["tasks"], "archived queue tasks")
    templates = _load_strict_jsonl_bytes(
        raw_files["decision_template"],
        "archived decision template",
    )
    decisions = _load_strict_jsonl_bytes(raw_files["decisions"], "archived curator decisions")
    task_count = queue.get("task_count")
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count <= 0
        or len(tasks) != task_count
        or len(templates) != task_count
        or len(decisions) != task_count
        or assembly["counts"].get("source_tasks") != task_count
    ):
        raise PublicationGateError("archived human review row counts are invalid")
    decision_fields = {
        "artifact_version",
        "queue_fingerprint",
        "task_id",
        "prepare_manifest_sha256",
        "source_benchmark_sha256",
        "source_provenance_sha256",
        "source_audit_sha256",
        "source_row_index",
        "source_row_sha256",
        "status",
        "disposition",
        "exclusion_reason",
        "benchmark_row",
        "provenance_row",
        "reviewed_by",
        "independent_attestation",
        "notes",
    }
    binding_fields = {
        "artifact_version",
        "task_id",
        "prepare_manifest_sha256",
        "source_benchmark_sha256",
        "source_provenance_sha256",
        "source_audit_sha256",
        "source_row_index",
        "source_row_sha256",
    }
    tasks_by_id: dict[str, Mapping[str, Any]] = {}
    task_fields = {
        "artifact_version",
        "task_id",
        "prepare_manifest_sha256",
        "source_benchmark_sha256",
        "source_provenance_sha256",
        "source_audit_sha256",
        "source_row_index",
        "source_row_sha256",
        "original_row",
        "legacy_provenance_rows",
        "audited_image_hashes",
        "critical_codes",
        "warning_codes",
        "training_overlap_paper_ids",
    }
    queue_source_hashes = queue.get("source_hashes")
    if not isinstance(queue_source_hashes, Mapping):
        raise PublicationGateError("archived queue source hashes are malformed")
    for task in tasks:
        task_id = task.get("task_id")
        source_row_index = task.get("source_row_index")
        original_row = task.get("original_row")
        if (
            set(task) != task_fields
            or task.get("artifact_version") != 3
            or not isinstance(task_id, str)
            or not task_id
            or task_id in tasks_by_id
            or isinstance(source_row_index, bool)
            or not isinstance(source_row_index, int)
            or source_row_index < 0
            or not isinstance(original_row, Mapping)
            or task.get("prepare_manifest_sha256")
            != queue_source_hashes.get("prepare_manifest_sha256")
            or task.get("source_benchmark_sha256")
            != queue_source_hashes.get("benchmark_file_sha256")
            or task.get("source_provenance_sha256")
            != queue_source_hashes.get("provenance_file_sha256")
            or task.get("source_audit_sha256") != queue_source_hashes.get("audit_json_sha256")
            or not isinstance(task.get("legacy_provenance_rows"), list)
            or not isinstance(task.get("audited_image_hashes"), list)
            or not isinstance(task.get("critical_codes"), list)
            or not isinstance(task.get("warning_codes"), list)
            or not isinstance(task.get("training_overlap_paper_ids"), list)
        ):
            raise PublicationGateError("archived queue has invalid or duplicate task IDs")
        source_row_bytes = json.dumps(
            original_row,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        source_row_sha256 = hashlib.sha256(source_row_bytes).hexdigest()
        expected_task_id = "task_" + hashlib.sha256(
            (
                f"{task['prepare_manifest_sha256']}\x00{source_row_index}\x00"
                f"{source_row_sha256}"
            ).encode("ascii")
        ).hexdigest()
        if (
            task.get("source_row_sha256") != source_row_sha256
            or task_id != expected_task_id
        ):
            raise PublicationGateError("archived queue task source binding is invalid")
        tasks_by_id[task_id] = task
    templates_by_id: dict[str, Mapping[str, Any]] = {}
    decisions_by_id: dict[str, Mapping[str, Any]] = {}
    for rows, indexed in (
        (templates, templates_by_id),
        (decisions, decisions_by_id),
    ):
        for row in rows:
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in indexed:
                raise PublicationGateError("archived review has invalid or duplicate task IDs")
            indexed[task_id] = row
    if (
        set(templates_by_id) != set(tasks_by_id)
        or set(decisions_by_id) != set(tasks_by_id)
        or len(templates_by_id) != task_count
        or len(decisions_by_id) != task_count
    ):
        raise PublicationGateError("archived review rows do not cover the queue exactly once")
    retained_benchmark: list[dict[str, Any]] = []
    retained_provenance: list[dict[str, Any]] = []
    exclusion_count = 0
    normalized_reviewers: set[str] = set()
    for task_id, task in tasks_by_id.items():
        template = templates_by_id[task_id]
        decision = decisions_by_id[task_id]
        if not isinstance(template, Mapping) or not isinstance(decision, Mapping):
            raise PublicationGateError("archived review row is malformed")
        expected_binding = {field: task.get(field) for field in binding_fields}
        expected_binding["queue_fingerprint"] = assembly["queue_fingerprint"]
        if (
            set(template) != decision_fields
            or set(decision) != decision_fields
            or any(template.get(field) != value for field, value in expected_binding.items())
            or any(decision.get(field) != value for field, value in expected_binding.items())
            or template.get("status") != "pending"
            or template.get("disposition") is not None
            or template.get("exclusion_reason") is not None
            or template.get("benchmark_row") is not None
            or template.get("provenance_row") is not None
            or template.get("reviewed_by") != []
            or template.get("independent_attestation") is not False
            or template.get("notes") != ""
            or decision.get("status") != "complete"
            or decision.get("independent_attestation") is not True
            or not isinstance(decision.get("notes"), str)
        ):
            raise PublicationGateError("archived curator decision bindings are invalid")
        reviewers = decision.get("reviewed_by")
        if not isinstance(reviewers, list) or len(reviewers) < 2:
            raise PublicationGateError("archived curator decision lacks two reviewers")
        normalized = [normalize_identity(value, ascii_reviewer=True) for value in reviewers]
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise PublicationGateError("archived curator reviewer identities are invalid")
        normalized_reviewers.update(normalized)
        disposition = decision.get("disposition")
        if disposition == "retain":
            benchmark_row = decision.get("benchmark_row")
            provenance_row = decision.get("provenance_row")
            if (
                decision.get("exclusion_reason") is not None
                or not isinstance(benchmark_row, dict)
                or not isinstance(provenance_row, dict)
            ):
                raise PublicationGateError("archived retained decision is incomplete")
            retained_benchmark.append(benchmark_row)
            retained_provenance.append(provenance_row)
        elif disposition == "exclude":
            if (
                not isinstance(decision.get("exclusion_reason"), str)
                or not decision["exclusion_reason"].strip()
                or decision.get("benchmark_row") is not None
                or decision.get("provenance_row") is not None
            ):
                raise PublicationGateError("archived excluded decision is malformed")
            exclusion_count += 1
        else:
            raise PublicationGateError("archived curator decision disposition is invalid")
    declared_reviewers = {
        normalize_identity(value, ascii_reviewer=True) for value in assembly.get("reviewer_ids", [])
    }
    if (
        normalized_reviewers != declared_reviewers
        or binding["decisions_sha256"] != assembly.get("decisions_sha256")
        or len(retained_benchmark) != assembly["counts"].get("retained_rows")
        or len(retained_provenance) != assembly["counts"].get("provenance_rows")
        or exclusion_count != assembly["counts"].get("excluded_rows")
    ):
        raise PublicationGateError("archived human review summary differs from assembly")
    policy = queue.get("policy")
    if not isinstance(policy, dict):
        raise PublicationGateError("archived queue policy is malformed")
    return retained_benchmark, retained_provenance, tasks, policy


def _validate_archived_machine_enrichment(
    assembly: Mapping[str, Any],
    output_records: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    *,
    required: bool,
    queue_tasks: Sequence[Mapping[str, Any]],
    queue_policy: Mapping[str, Any],
) -> None:
    binding = assembly.get("machine_enrichment_evidence")
    if binding is None:
        if required:
            raise PublicationGateError(
                "capacity benchmark assembly has no archived machine enrichment evidence"
            )
        return
    expected_fields = {
        "artifact_version",
        "archive_root",
        "source_machine_enrichment_path",
        "source_machine_enrichment_sha256",
        "package_manifest_path",
        "package_manifest_sha256",
        "capacity_plan_manifest_sha256",
        "capacity_assist_manifest_sha256",
        "external_evidence_content_verified",
        "human_verification_required",
    }
    archive_root = "audit/machine_enrichment"
    package_root = f"{archive_root}/package"
    package_manifest_path = f"{package_root}/machine_enrichment_manifest.json"
    source_path = f"{archive_root}/source_machine_enrichment.jsonl"
    digest_fields = (
        "source_machine_enrichment_sha256",
        "package_manifest_sha256",
        "capacity_plan_manifest_sha256",
        "capacity_assist_manifest_sha256",
    )
    if (
        not isinstance(binding, Mapping)
        or set(binding) != expected_fields
        or binding.get("artifact_version") != 2
        or binding.get("archive_root") != archive_root
        or binding.get("source_machine_enrichment_path") != source_path
        or binding.get("package_manifest_path") != package_manifest_path
        or binding.get("external_evidence_content_verified") is not False
        or binding.get("human_verification_required") is not True
        or any(
            not isinstance(binding.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", binding[field]) is None
            for field in digest_fields
        )
    ):
        raise PublicationGateError("benchmark assembly machine enrichment binding is malformed")
    if (
        source_path not in output_records
        or output_records[source_path].get("sha256")
        != binding["source_machine_enrichment_sha256"]
        or package_manifest_path not in output_records
        or output_records[package_manifest_path].get("sha256")
        != binding["package_manifest_sha256"]
    ):
        raise PublicationGateError("benchmark assembly does not bind archived machine evidence bytes")
    package_manifest_bytes = _read_stable_bytes(
        _safe_repo_file(dataset_root, package_manifest_path),
        "archived machine enrichment manifest",
    )
    if hashlib.sha256(package_manifest_bytes).hexdigest() != binding["package_manifest_sha256"]:
        raise PublicationGateError("archived machine enrichment manifest bytes changed")
    package_manifest = _load_strict_json_object_bytes(
        package_manifest_bytes,
        "archived machine enrichment manifest",
    )
    policy = package_manifest.get("policy")
    if (
        package_manifest.get("artifact_version") != binding["artifact_version"]
        or package_manifest.get("queue_fingerprint") != assembly.get("queue_fingerprint")
        or package_manifest.get("source_machine_enrichment_sha256")
        != binding["source_machine_enrichment_sha256"]
        or package_manifest.get("capacity_plan_manifest_sha256")
        != binding["capacity_plan_manifest_sha256"]
        or package_manifest.get("capacity_assist_manifest_sha256")
        != binding["capacity_assist_manifest_sha256"]
        or not isinstance(policy, Mapping)
        or policy.get("machine_enrichment_schema_validated") is not True
        or policy.get("external_evidence_content_verified") is not False
        or policy.get("automatic_retain") is not False
        or policy.get("automatic_human_identity") is not False
        or policy.get("automatic_attestation") is not False
        or policy.get("quarantined_cross_paper_image_reuse_allowed") is not False
        or policy.get("human_verification_required") is not True
    ):
        raise PublicationGateError("archived machine enrichment manifest has unsafe bindings")
    candidate_count = package_manifest.get("candidate_group_count")
    prefilled_task_count = package_manifest.get("prefilled_task_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
        or isinstance(prefilled_task_count, bool)
        or not isinstance(prefilled_task_count, int)
        or prefilled_task_count < candidate_count
        or candidate_count != assembly["counts"].get("retained_rows")
    ):
        raise PublicationGateError("archived machine enrichment counts are invalid")
    inventory = package_manifest.get("files")
    expected_payloads = {
        "machine_assisted_draft.json",
        "machine_enrichment.jsonl",
        "machine_enrichment_summary.md",
    }
    if not isinstance(inventory, list) or len(inventory) != len(expected_payloads):
        raise PublicationGateError("archived machine enrichment inventory is malformed")
    seen: set[str] = set()
    for record in inventory:
        if not isinstance(record, Mapping):
            raise PublicationGateError("archived machine enrichment inventory is malformed")
        name = record.get("path")
        digest = record.get("sha256")
        size = record.get("size_bytes")
        archived_path = f"{package_root}/{name}" if isinstance(name, str) else ""
        if (
            not isinstance(name, str)
            or name not in expected_payloads
            or name in seen
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or archived_path not in output_records
            or output_records[archived_path].get("sha256") != digest
            or output_records[archived_path].get("size_bytes") != size
        ):
            raise PublicationGateError("archived machine enrichment inventory is malformed")
        seen.add(name)
    expected_archive = {
        source_path,
        package_manifest_path,
        *(f"{package_root}/{name}" for name in expected_payloads),
    }
    actual_archive = {
        path for path in output_records if path.startswith(f"{archive_root}/")
    }
    if actual_archive != expected_archive:
        raise PublicationGateError("benchmark assembly machine enrichment archive is incomplete")
    source_bytes = _read_stable_bytes(
        _safe_repo_file(dataset_root, source_path),
        "archived source machine enrichment",
    )
    if hashlib.sha256(source_bytes).hexdigest() != binding["source_machine_enrichment_sha256"]:
        raise PublicationGateError("archived source machine enrichment bytes changed")
    canonical_path = f"{package_root}/machine_enrichment.jsonl"
    canonical_bytes = _read_stable_bytes(
        _safe_repo_file(dataset_root, canonical_path),
        "archived canonical machine enrichment",
    )
    source_rows = _load_strict_jsonl_bytes(source_bytes, "archived source machine enrichment")
    canonical_rows = _load_strict_jsonl_bytes(
        canonical_bytes,
        "archived canonical machine enrichment",
    )
    if source_rows != canonical_rows or len(canonical_rows) != candidate_count:
        raise PublicationGateError("archived machine enrichment source and canonical rows differ")
    enrichment_fields = {
        "artifact_version",
        "queue_fingerprint",
        "capacity_plan_manifest_sha256",
        "group_id",
        "canonical_paper_id",
        "task_ids",
        "selected_task_ids",
        "benchmark_rows",
        "provenance_rows",
        "external_evidence",
        "claim_evidence",
        "human_review",
        "release_eligibility",
    }
    evidence_fields = {
        "evidence_id",
        "kind",
        "source_url",
        "content_sha256",
        "asserted_license",
        "retrieved_at",
        "notes",
    }
    claim_names = {
        "paper_identity",
        "scientific_prompt",
        "image_provenance",
        "usage_rights",
        "paper_holdout",
        "source_holdout",
        "creator_holdout",
        "training_overlap",
    }
    selected_count = 0
    seen_groups: set[str] = set()
    for row in canonical_rows:
        group_id = row.get("group_id")
        selected = row.get("selected_task_ids")
        evidence = row.get("external_evidence")
        claims = row.get("claim_evidence")
        if (
            set(row) != enrichment_fields
            or row.get("artifact_version") != binding["artifact_version"]
            or row.get("queue_fingerprint") != assembly.get("queue_fingerprint")
            or row.get("capacity_plan_manifest_sha256")
            != binding["capacity_plan_manifest_sha256"]
            or not isinstance(group_id, str)
            or not group_id
            or group_id in seen_groups
            or not isinstance(selected, list)
            or not selected
            or any(not isinstance(task_id, str) or not task_id for task_id in selected)
            or len(selected) != len(set(selected))
            or not isinstance(row.get("benchmark_rows"), Mapping)
            or set(row["benchmark_rows"]) != set(selected)
            or not isinstance(row.get("provenance_rows"), Mapping)
            or set(row["provenance_rows"]) != set(selected)
            or row.get("human_review")
            != {"curator_ids": [], "independent_attestation": False}
            or row.get("release_eligibility") != "blocked"
            or not isinstance(evidence, list)
            or not evidence
            or not isinstance(claims, Mapping)
            or set(claims) != claim_names
        ):
            raise PublicationGateError("archived machine enrichment row bindings are invalid")
        evidence_ids: set[str] = set()
        for record in evidence:
            if not isinstance(record, Mapping) or set(record) != evidence_fields:
                raise PublicationGateError("archived machine evidence record is malformed")
            evidence_id = record.get("evidence_id")
            kind = record.get("kind")
            asserted_license = record.get("asserted_license")
            if (
                not isinstance(evidence_id, str)
                or not evidence_id
                or evidence_id in evidence_ids
                or kind not in {"article", "figure", "license", "metadata", "training_lineage"}
                or not isinstance(record.get("source_url"), str)
                or not record["source_url"].startswith(("http://", "https://"))
                or not isinstance(record.get("content_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["content_sha256"]) is None
                or not isinstance(record.get("retrieved_at"), str)
                or not record["retrieved_at"]
                or not isinstance(record.get("notes"), str)
                or (
                    kind == "license"
                    and (not isinstance(asserted_license, str) or not asserted_license.strip())
                )
                or (kind != "license" and asserted_license is not None)
            ):
                raise PublicationGateError("archived machine evidence record is malformed")
            evidence_ids.add(evidence_id)
        for claim in claims.values():
            if (
                not isinstance(claim, Mapping)
                or set(claim) != {"status", "evidence_ids"}
                or claim.get("status") != "machine_supported"
                or not isinstance(claim.get("evidence_ids"), list)
                or not claim["evidence_ids"]
                or any(not isinstance(value, str) for value in claim["evidence_ids"])
                or len(claim["evidence_ids"]) != len(set(claim["evidence_ids"]))
                or not set(claim["evidence_ids"]).issubset(evidence_ids)
            ):
                raise PublicationGateError("archived machine claim evidence is malformed")
        seen_groups.add(group_id)
        selected_count += len(selected)
    if selected_count != prefilled_task_count:
        raise PublicationGateError("archived machine enrichment task count is invalid")
    draft = _load_strict_json_object(
        _safe_repo_file(dataset_root, f"{package_root}/machine_assisted_draft.json"),
        "archived machine-assisted draft",
    )
    edits = draft.get("edits")
    if (
        set(draft) != {"artifact_version", "queue_fingerprint", "edits"}
        or draft.get("artifact_version") != 1
        or draft.get("queue_fingerprint") != assembly.get("queue_fingerprint")
        or not isinstance(edits, Mapping)
    ):
        raise PublicationGateError("archived machine-assisted draft binding is invalid")
    for edit in edits.values():
        images = edit.get("images") if isinstance(edit, Mapping) else None
        if (
            not isinstance(edit, Mapping)
            or edit.get("disposition") != ""
            or edit.get("reviewed_by") != ""
            or edit.get("independent_attestation") is not False
            or not isinstance(images, list)
            or any(
                not isinstance(image, Mapping) or image.get("verified_by") != ""
                for image in images
            )
        ):
            raise PublicationGateError("archived machine-assisted draft contains human state")
    try:
        from .capacity_assist import (
            CAPACITY_ASSIST_MANIFEST_FILENAME,
            _capacity_assist_artifacts,
            _capacity_assist_candidates,
        )
        from .capacity_enrichment import _validated_enrichment
        from .capacity_plan import (
            CAPACITY_PLAN_MANIFEST_FILENAME,
            _capacity_plan_artifacts,
            _task_details,
        )
        from .remediation import _load_schemas

        primary_strata = queue_policy["primary_strata"]
        task_details = _task_details(queue_tasks, primary_strata)
        plan_files, _, groups = _capacity_plan_artifacts(
            task_details,
            assembly["queue_fingerprint"],
            candidate_count,
            primary_strata,
        )
        plan_sha256 = hashlib.sha256(
            plan_files[CAPACITY_PLAN_MANIFEST_FILENAME]
        ).hexdigest()
        candidates = _capacity_assist_candidates(groups)
        assist_files, _ = _capacity_assist_artifacts(
            candidates,
            queue_tasks,
            task_details,
            assembly["queue_fingerprint"],
            plan_sha256,
            candidate_count,
        )
        assist_sha256 = hashlib.sha256(
            assist_files[CAPACITY_ASSIST_MANIFEST_FILENAME]
        ).hexdigest()
        repository_root = Path(__file__).resolve().parents[3]
        schema_root = repository_root / "experiments" / "vlm_ab_evaluation" / "schemas"
        schemas = _load_schemas(
            schema_root / "publication_benchmark_row.schema.json",
            schema_root / "publication_provenance_row.schema.json",
        )
        reconstructed_rows, reconstructed_draft = _validated_enrichment(
            canonical_rows,
            candidates,
            queue_tasks,
            schemas,
            assembly["queue_fingerprint"],
            plan_sha256,
            primary_strata,
        )
    except PublicationGateError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PublicationGateError(
            f"cannot reconstruct archived machine enrichment: {exc}"
        ) from exc
    if (
        len(candidates) != candidate_count
        or plan_sha256 != binding["capacity_plan_manifest_sha256"]
        or assist_sha256 != binding["capacity_assist_manifest_sha256"]
        or reconstructed_rows != canonical_rows
        or reconstructed_draft != draft
    ):
        raise PublicationGateError(
            "archived machine enrichment differs from deterministic reconstruction"
        )


def _validated_benchmark_assembly(
    config: Mapping[str, Any], dataset_root: Path
) -> Path | None:
    benchmark = config["benchmark"]
    relative = benchmark.get("assembly_manifest_file")
    expected_hash = benchmark.get("assembly_manifest_sha256")
    if relative is None and expected_hash is None:
        return None
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise PublicationGateError("benchmark assembly manifest binding is incomplete")
    try:
        path = _safe_repo_file(dataset_root, relative)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise PublicationGateError("benchmark assembly manifest is missing or unsafe") from exc
    manifest_bytes = _read_stable_bytes(path, "benchmark assembly manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
        raise PublicationGateError("benchmark assembly manifest SHA256 differs from config")
    manifest = _load_strict_json_object_bytes(manifest_bytes, "benchmark assembly manifest")
    counts = manifest.get("counts")
    reviewer_ids = manifest.get("reviewer_ids")
    normalized_reviewers = (
        [normalize_identity(value, ascii_reviewer=True) for value in reviewer_ids]
        if isinstance(reviewer_ids, list)
        else []
    )
    if (
        manifest.get("artifact_version") != 3
        or manifest.get("scope") != "validated_benchmark_release_candidate"
        or manifest.get("technical_audit_passed") is not True
        or manifest.get("publication_ready") is not False
        or not isinstance(counts, Mapping)
        or not isinstance(reviewer_ids, list)
        or len(reviewer_ids) < 2
        or any(not isinstance(value, str) or not value.strip() for value in reviewer_ids)
        or any(not value for value in normalized_reviewers)
        or len(set(normalized_reviewers)) != len(normalized_reviewers)
        or not isinstance(manifest.get("queue_fingerprint"), str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest["queue_fingerprint"]) is None
        or not isinstance(manifest.get("decisions_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest["decisions_sha256"]) is None
    ):
        raise PublicationGateError("benchmark assembly manifest has invalid review bindings")
    expected_n = config["power"]["n_items"]
    if config["power"].get("require_exact_n_items") is True and any(
        counts.get(field) != expected_n
        for field in ("retained_rows", "provenance_rows", "unique_primary_papers")
    ):
        raise PublicationGateError("benchmark assembly manifest does not bind exact retained capacity")
    output_files = manifest.get("output_files")
    if not isinstance(output_files, list) or not output_files:
        raise PublicationGateError("benchmark assembly manifest has no output file inventory")
    records: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(output_files):
        if not isinstance(record, Mapping):
            raise PublicationGateError(
                f"benchmark assembly output_files[{index}] must be an object"
            )
        record_path = record.get("path")
        digest = record.get("sha256")
        size = record.get("size_bytes")
        if (
            not isinstance(record_path, str)
            or record_path in records
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise PublicationGateError("benchmark assembly output file inventory is malformed")
        try:
            source = _safe_repo_file(dataset_root, record_path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise PublicationGateError(
                f"benchmark assembly output file is missing or unsafe: {record_path}"
            ) from exc
        source_bytes = _read_stable_bytes(source, f"benchmark assembly output {record_path}")
        if len(source_bytes) != size or hashlib.sha256(source_bytes).hexdigest() != digest:
            raise PublicationGateError(
                f"benchmark assembly output file differs from manifest: {record_path}"
            )
        records[record_path] = record
    reviewed_benchmark, reviewed_provenance, queue_tasks, queue_policy = (
        _validate_archived_human_review(
            manifest,
            records,
            dataset_root,
        )
    )
    required = {
        benchmark["data_file"],
        "audit/benchmark_audit.json",
        "audit/benchmark_audit.md",
    }
    if benchmark.get("provenance_file"):
        required.add(benchmark["provenance_file"])
    try:
        assembled_rows = load_jsonl(_safe_repo_file(dataset_root, benchmark["data_file"]))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise PublicationGateError("assembled benchmark data file is missing or unsafe") from exc
    if counts.get("retained_rows") != len(assembled_rows):
        raise PublicationGateError("benchmark assembly retained-row count differs from release bytes")
    if sorted(reviewed_benchmark, key=lambda row: row.get("sample_id", "")) != assembled_rows:
        raise PublicationGateError("archived retained decisions differ from benchmark release bytes")
    if benchmark.get("provenance_file"):
        try:
            assembled_provenance = load_jsonl(
                _safe_repo_file(dataset_root, benchmark["provenance_file"])
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise PublicationGateError(
                "assembled benchmark provenance file is missing or unsafe"
            ) from exc
        if counts.get("provenance_rows") != len(assembled_provenance):
            raise PublicationGateError(
                "benchmark assembly provenance-row count differs from release bytes"
            )
        if (
            sorted(reviewed_provenance, key=lambda row: row.get("sample_id", ""))
            != assembled_provenance
        ):
            raise PublicationGateError(
                "archived retained decisions differ from provenance release bytes"
            )
    primary_strata = frozenset(config["statistics"]["primary_strata"])
    primary_papers = {
        canonical_paper_id(row.get("paper_id"))
        for row in assembled_rows
        if row.get("stratum") in primary_strata
    }
    primary_papers.discard("")
    if counts.get("unique_primary_papers") != len(primary_papers):
        raise PublicationGateError(
            "benchmark assembly primary-paper count differs from release bytes"
        )
    for row in assembled_rows:
        images = row.get("images", [])
        if not isinstance(images, list):
            raise PublicationGateError("assembled benchmark row has malformed images")
        required.update(reference for reference in images if isinstance(reference, str))
    if not required.issubset(records):
        raise PublicationGateError("benchmark assembly does not bind configured benchmark files")
    if relative in records:
        raise PublicationGateError("benchmark assembly manifest cannot inventory itself")
    require_machine_enrichment = bool(
        config["power"].get("require_exact_n_items") is True
        and config["power"]["n_items"] == 150
    )
    _validate_archived_machine_enrichment(
        manifest,
        records,
        dataset_root,
        required=require_machine_enrichment,
        queue_tasks=queue_tasks,
        queue_policy=queue_policy,
    )
    return path


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
) -> tuple[Path, list[Path], Path | None, Path, Path, list[dict[str, Any]]]:
    benchmark = config["benchmark"]
    patterns = [benchmark["data_file"], "assets/images/**"]
    if benchmark.get("provenance_file"):
        patterns.append(benchmark["provenance_file"])
    if benchmark.get("assembly_manifest_file"):
        patterns.extend([benchmark["assembly_manifest_file"], "audit/**"])
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
        ["adapter_config.json", "adapter_model.safetensors"],
        cache_dir,
    )
    adapter_config_path = _safe_repo_file(adapter_root, "adapter_config.json")
    adapter_checkpoint_path = _safe_repo_file(adapter_root, "adapter_model.safetensors")
    sources.append(
        {
            "repo_id": adapter["id"],
            "repo_type": "model",
            "configured_revision": adapter["revision"],
            "resolved_revision": adapter_revision,
            "files": ["adapter_config.json", "adapter_model.safetensors"],
            "role": "evaluated_adapter_checkpoint",
            "snapshot_root": str(adapter_root),
        }
    )
    return (
        root,
        training_files,
        lineage_path,
        adapter_config_path,
        adapter_checkpoint_path,
        sources,
    )


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
    expected_assembly_hash = config["benchmark"].get("assembly_manifest_sha256")
    if expected_assembly_hash is not None:
        assembly_path = prepared.get("assembly_manifest")
        if (
            manifest.get("assembly_manifest_sha256") != expected_assembly_hash
            or not isinstance(assembly_path, str)
            or _sha256(Path(assembly_path)) != expected_assembly_hash
        ):
            raise PublicationGateError(
                "prepare manifest does not preserve the configured benchmark assembly"
            )
        _validated_benchmark_assembly(
            config,
            Path(str(prepared["dataset_root"])).resolve(strict=True),
        )
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
        exact_primary_papers=(
            int(config["power"]["n_items"])
            if strict_audit and config["power"].get("require_exact_n_items") is True
            else None
        ),
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
        (
            dataset_root,
            remote_training,
            lineage_path,
            adapter_config_path,
            adapter_checkpoint_path,
            sources,
        ) = _download_inputs(config, Path(cache_dir).resolve() if cache_dir else None)
    else:
        dataset_root = Path(benchmark_dir).resolve(strict=True)
        remote_training = []
        lineage_path = None
        adapter_config_path = None
        adapter_checkpoint_path = None
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

    assembly_manifest_path = _validated_benchmark_assembly(config, dataset_root)
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
    adapter_checkpoint_attestation = None
    if not exploratory:
        if adapter_config_path is None or adapter_checkpoint_path is None:
            adapter_metadata_failure = (
                "strict preparation requires adapter_config.json and adapter_model.safetensors at revision R"
            )
        else:
            try:
                adapter_config = _load_strict_json_object(
                    adapter_config_path,
                    "adapter_config.json",
                )
                _validate_adapter_base_metadata(adapter_config, config)
                adapter = config["models"]["tuned"]["adapter"]
                adapter_checkpoint_attestation = inspect_plain_fp32_lora_safetensors(
                    adapter_checkpoint_path,
                    repo_id=adapter["id"],
                    revision=adapter["revision"],
                    error_type=PublicationGateError,
                )
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
        exact_primary_papers=(
            int(config["power"]["n_items"])
            if not exploratory and config["power"].get("require_exact_n_items") is True
            else None
        ),
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
    source_dir = output_dir / "inputs" / "source"
    if assembly_manifest_path is not None:
        assembly = _load_strict_json_object(assembly_manifest_path, "benchmark assembly manifest")
        for record in sorted(assembly["output_files"], key=lambda value: value["path"]):
            relative_output = record["path"]
            _copy_file(
                _safe_repo_file(dataset_root, relative_output),
                _bundle_destination(bundled_dataset_root, relative_output),
            )
        bundled_benchmark = _safe_repo_file(
            bundled_dataset_root,
            config["benchmark"]["data_file"],
        )
        bundled_provenance = (
            _safe_repo_file(bundled_dataset_root, config["benchmark"]["provenance_file"])
            if provenance_path is not None
            else None
        )
        bundled_assembly_manifest = _copy_file(
            assembly_manifest_path,
            _bundle_destination(
                bundled_dataset_root,
                config["benchmark"]["assembly_manifest_file"],
            ),
        )
        _validated_benchmark_assembly(config, bundled_dataset_root)
    else:
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
        bundled_assembly_manifest = None
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
    bundled_adapter_attestation = None
    if adapter_checkpoint_attestation is not None:
        bundled_adapter_attestation = source_dir / "adapter_checkpoint_attestation.json"
        _write_json(bundled_adapter_attestation, adapter_checkpoint_attestation)
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
            "adapter_checkpoint_attestation": relative(bundled_adapter_attestation),
            "assembly_manifest": relative(bundled_assembly_manifest),
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
        "adapter_checkpoint_attestation": (
            str(bundled_adapter_attestation) if bundled_adapter_attestation else None
        ),
        "adapter_checkpoint_attestation_sha256": (
            _sha256(bundled_adapter_attestation) if bundled_adapter_attestation else None
        ),
        "assembly_manifest": (
            str(bundled_assembly_manifest) if bundled_assembly_manifest else None
        ),
        "assembly_manifest_sha256": (
            _sha256(bundled_assembly_manifest) if bundled_assembly_manifest else None
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
    verify_prepare_manifest(manifest, output_dir)
    if verify_prepared_audit(config, manifest, output_dir) != report:
        raise PublicationGateError("fresh bundled audit differs from the preparation audit")
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
