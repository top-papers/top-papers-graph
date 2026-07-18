# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deterministic human remediation for failed VLM benchmark preparation runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .audit import (
    BenchmarkAuditError,
    audit_benchmark,
    canonical_paper_id,
    paper_identity_errors,
    write_audit_report,
)
from .config import config_fingerprint
from .contracts import PUBLICATION_SCHEMA_SHA256
from .identities import normalize_identity
from .paths import resolve_dataset_file
from .prepare import (
    PublicationGateError,
    resolve_prepare_manifest,
    verify_prepare_manifest,
    verify_prepared_audit,
)


ARTIFACT_VERSION = 2
TASKS_FILENAME = "tasks.jsonl"
DECISION_TEMPLATE_FILENAME = "decision_template.jsonl"
QUEUE_MANIFEST_FILENAME = "queue_manifest.json"
ASSEMBLY_MANIFEST_FILENAME = "assembly_manifest.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_FROZEN_WARNING_CODES = (
    "duplicate_normalized_prompt",
    "within_row_duplicate_image_bytes",
)
_DECISION_FIELDS = frozenset(
    {
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
        "notes",
    }
)
_RESERVED_RELEASE_PATHS = frozenset(
    {
        "article_image_sources.jsonl",
        "assembly_manifest.json",
        "audit/benchmark_audit.json",
        "audit/benchmark_audit.md",
        "data/task3_vlm_generation.jsonl",
    }
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {"aux", "clock$", "con", "conin$", "conout$", "nul", "prn"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_WINDOWS_SUPERSCRIPT_DIGITS = str.maketrans({"¹": "1", "²": "2", "³": "3"})
_REMAINING_REQUIREMENTS = (
    "publish immutable benchmark revision",
    "publish adapter lineage/revision",
    "create new experiment IDs and plan",
    "pass strict prepare",
)


class RemediationError(ValueError):
    """Raised when remediation inputs or artifacts cannot be handled safely."""


class DecisionValidationError(RemediationError):
    """Raised when curator decisions do not satisfy the remediation contract."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _read_stable_bytes(path: Path, label: str) -> bytes:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RemediationError(f"{label} must be a regular non-symlink file: {path}")
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            raw = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = os.lstat(path)
    except OSError as exc:
        raise RemediationError(f"cannot read {label} {path}: {exc}") from exc

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
        raise RemediationError(f"{label} changed while it was being read: {path}")
    return raw


def _strict_json(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    try:
        raw = _read_stable_bytes(source, label)
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise RemediationError(f"cannot read {label} {source}: {exc}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKeyError as exc:
        raise RemediationError(f"duplicate JSON key {exc.args[0]!r} in {label}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise RemediationError(f"invalid strict JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RemediationError(f"{label} must contain a JSON object")
    return value, raw


def _strict_jsonl(path: str | Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    source = Path(path)
    try:
        raw = _read_stable_bytes(source, label)
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise RemediationError(f"cannot read strict {label}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except _DuplicateJsonKeyError as exc:
            raise RemediationError(
                f"duplicate JSON key {exc.args[0]!r} in {label} line {line_number}"
            ) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise RemediationError(f"invalid strict {label} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RemediationError(f"{label} line {line_number} must contain a JSON object")
        rows.append(value)
    return rows, raw


def _json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RemediationError(f"value is not strict JSON data: {exc}") from exc
    return (text + ("\n" if newline else "")).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row, newline=True) for row in rows)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise RemediationError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest(), size


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise RemediationError(f"{label} must be a lowercase SHA256")
    return value


def _policy_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = config.get("benchmark")
    statistics = config.get("statistics")
    power = config.get("power")
    if not all(isinstance(value, Mapping) for value in (benchmark, statistics, power)):
        raise RemediationError("config must contain benchmark, statistics, and power objects")

    primary_strata = statistics.get("primary_strata")
    allowed_strata = {"multimodal_hard", "temporal_hard", "easy_control"}
    if (
        not isinstance(primary_strata, list)
        or not primary_strata
        or any(
            not isinstance(value, str) or value not in allowed_strata for value in primary_strata
        )
        or len(primary_strata) != len(set(primary_strata))
    ):
        raise RemediationError("config statistics.primary_strata must list unique known strata")
    require_gold = benchmark.get("require_gold", False)
    if not isinstance(require_gold, bool):
        raise RemediationError("config benchmark.require_gold must be boolean")
    minimum = power.get("n_items")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise RemediationError("config power.n_items must be a positive integer")
    return {
        "primary_strata": list(primary_strata),
        "require_gold": require_gold,
        "minimum_primary_papers": minimum,
        "warning_codes_must_be_zero": list(_FROZEN_WARNING_CODES),
    }


def _validate_prepare_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if manifest.get("artifact_version") != 1:
        raise RemediationError("prepare manifest artifact_version must be 1")
    _require_sha256(manifest.get("config_fingerprint"), "prepare manifest config_fingerprint")
    if not isinstance(manifest.get("exploratory"), bool):
        raise RemediationError("prepare manifest exploratory must be boolean")
    paths = manifest.get("artifact_paths")
    if not isinstance(paths, Mapping):
        raise RemediationError("prepare manifest artifact_paths must be an object")
    for key in (
        "dataset_root",
        "benchmark_file",
        "provenance_file",
        "frozen_benchmark",
        "audit_json",
    ):
        value = paths.get(key)
        if not isinstance(value, str) or not value:
            raise RemediationError(f"prepare manifest requires artifact_paths.{key}")
    for key in (
        "benchmark_file_sha256",
        "provenance_file_sha256",
        "frozen_benchmark_sha256",
        "audit_json_sha256",
    ):
        _require_sha256(manifest.get(key), f"prepare manifest {key}")
    training_files = manifest.get("training_files")
    if not isinstance(training_files, list):
        raise RemediationError("prepare manifest training_files must be an array")
    for index, entry in enumerate(training_files):
        if not isinstance(entry, Mapping):
            raise RemediationError(f"prepare manifest training_files[{index}] must be an object")
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise RemediationError(
                f"prepare manifest training_files[{index}].relative_path is required"
            )
        _require_sha256(entry.get("sha256"), f"prepare manifest training_files[{index}].sha256")
    lineage_path = paths.get("training_lineage_manifest")
    lineage_hash = manifest.get("training_lineage_manifest_sha256")
    if (lineage_path is None) != (lineage_hash is None):
        raise RemediationError("prepare manifest has an incomplete training lineage entry")
    if lineage_path is not None:
        if not isinstance(lineage_path, str) or not lineage_path:
            raise RemediationError("prepare manifest training lineage path is malformed")
        _require_sha256(lineage_hash, "prepare manifest training lineage hash")


def _validate_audit_images(
    audit: Mapping[str, Any], dataset_root: Path, source_row_count: int
) -> dict[int, list[dict[str, Any]]]:
    per_sample = audit.get("per_sample_findings")
    if not isinstance(per_sample, Mapping):
        raise RemediationError("source audit per_sample_findings must be an object")
    by_row: dict[int, list[dict[str, Any]]] = {index: [] for index in range(source_row_count)}
    reported_rows: list[int] = []
    for sample_key, details in per_sample.items():
        if not isinstance(sample_key, str) or not isinstance(details, Mapping):
            raise RemediationError("source audit contains a malformed per-sample entry")
        row_indices = details.get("row_indices")
        if not isinstance(row_indices, list):
            raise RemediationError(f"source audit row_indices is malformed for {sample_key!r}")
        for row_index in row_indices:
            if (
                isinstance(row_index, bool)
                or not isinstance(row_index, int)
                or not 0 <= row_index < source_row_count
            ):
                raise RemediationError(f"source audit has an invalid row index for {sample_key!r}")
            reported_rows.append(row_index)
        records = details.get("image_hashes")
        if not isinstance(records, list):
            raise RemediationError(f"source audit image_hashes is malformed for {sample_key!r}")
        for record in records:
            if not isinstance(record, Mapping):
                raise RemediationError("source audit contains a malformed image hash record")
            row_index = record.get("row_index")
            if (
                isinstance(row_index, bool)
                or not isinstance(row_index, int)
                or not 0 <= row_index < source_row_count
            ):
                raise RemediationError("source audit image hash has an invalid row_index")
            relative = record.get("path")
            if not isinstance(relative, str) or not relative:
                raise RemediationError("source audit image hash has no path")
            windows = PureWindowsPath(relative)
            posix = PurePosixPath(relative.replace("\\", "/"))
            parts = tuple(part for part in posix.parts if part not in {"", "."})
            normalized = "/".join(parts)
            if (
                _URI_SCHEME_RE.match(relative)
                or relative.startswith(("//", "\\\\"))
                or windows.is_absolute()
                or windows.drive
                or posix.is_absolute()
                or ".." in posix.parts
                or normalized != relative
                or any(":" in part for part in parts)
            ):
                raise RemediationError(f"source audit image path is unsafe: {relative!r}")
            expected = _require_sha256(record.get("sha256"), "source audit image hash")
            expected_size = record.get("size_bytes")
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
            ):
                raise RemediationError("source audit image hash has an invalid size_bytes")
            try:
                image_path = resolve_dataset_file(dataset_root, Path(*parts))
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise RemediationError(f"audited image is missing or unsafe: {relative}") from exc
            actual, actual_size = _sha256_file(image_path)
            if actual != expected or actual_size != expected_size:
                raise RemediationError(f"audited image hash changed: {relative}")
            copied = dict(record)
            by_row[row_index].append(copied)
    if Counter(reported_rows) != Counter(range(source_row_count)):
        raise RemediationError("source audit row indices do not exactly cover the benchmark rows")
    for records in by_row.values():
        records.sort(
            key=lambda record: (
                str(record.get("path")),
                str(record.get("sha256")),
                int(record.get("size_bytes", 0)),
            )
        )
    return by_row


def _load_source_bundle(prepare_manifest: str | Path) -> dict[str, Any]:
    manifest_path = Path(prepare_manifest)
    if manifest_path.is_symlink():
        raise RemediationError("prepare_manifest must not be a symlink")
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RemediationError(f"prepare_manifest does not exist: {prepare_manifest}") from exc
    manifest, manifest_bytes = _strict_json(manifest_path, "prepare manifest")
    _validate_prepare_manifest_shape(manifest)
    try:
        verify_prepare_manifest(manifest, manifest_path.parent)
        resolved = resolve_prepare_manifest(manifest, manifest_path.parent)
    except (OSError, PublicationGateError, RuntimeError, ValueError) as exc:
        raise RemediationError(f"prepare manifest verification failed: {exc}") from exc

    benchmark_rows, benchmark_bytes = _strict_jsonl(
        resolved["benchmark_file"], "source benchmark JSONL"
    )
    provenance_rows, provenance_bytes = _strict_jsonl(
        resolved["provenance_file"], "source provenance JSONL"
    )
    audit, audit_bytes = _strict_json(resolved["audit_json"], "source benchmark audit")
    source_byte_hashes = {
        "benchmark_file_sha256": _sha256_bytes(benchmark_bytes),
        "provenance_file_sha256": _sha256_bytes(provenance_bytes),
        "audit_json_sha256": _sha256_bytes(audit_bytes),
    }
    for field, actual in source_byte_hashes.items():
        if actual != manifest.get(field):
            raise RemediationError(f"source {field} changed while it was being read")
    if manifest.get("input_rows") != len(benchmark_rows):
        raise RemediationError("prepare manifest input_rows differs from the source benchmark")
    summary = audit.get("summary")
    if not isinstance(summary, Mapping) or summary.get("total_rows") != len(benchmark_rows):
        raise RemediationError("source audit row count differs from the source benchmark")
    if not isinstance(audit.get("critical_findings"), list) or not isinstance(
        audit.get("warnings"), list
    ):
        raise RemediationError("source audit findings are malformed")

    dataset_root = Path(resolved["dataset_root"])
    audit_images = _validate_audit_images(audit, dataset_root, len(benchmark_rows))
    training_rows: list[dict[str, Any]] = []
    training_hashes: list[str] = []
    training_file_records: list[dict[str, Any]] = []
    for index, entry in enumerate(resolved.get("training_files", [])):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise RemediationError(f"resolved training file entry {index} is malformed")
        rows, training_bytes = _strict_jsonl(entry["path"], f"source training JSONL {index}")
        training_rows.extend(rows)
        expected_training_hash = _require_sha256(entry.get("sha256"), "training file hash")
        if _sha256_bytes(training_bytes) != expected_training_hash:
            raise RemediationError(f"source training file {index} changed while it was being read")
        training_hashes.append(expected_training_hash)
        training_file_records.append(
            {
                "sha256": expected_training_hash,
                "row_count": len(rows),
            }
        )

    training_lineage = None
    lineage_hash = manifest.get("training_lineage_manifest_sha256")
    if resolved.get("training_lineage_manifest"):
        training_lineage, lineage_bytes = _strict_json(
            resolved["training_lineage_manifest"], "source training lineage manifest"
        )
        expected_lineage_hash = _require_sha256(lineage_hash, "training lineage hash")
        if _sha256_bytes(lineage_bytes) != expected_lineage_hash:
            raise RemediationError("source training lineage changed while it was being read")

    source_hashes = {
        "prepare_manifest_sha256": _sha256_bytes(manifest_bytes),
        "benchmark_file_sha256": _require_sha256(
            manifest.get("benchmark_file_sha256"), "benchmark file hash"
        ),
        "provenance_file_sha256": _require_sha256(
            manifest.get("provenance_file_sha256"), "provenance file hash"
        ),
        "audit_json_sha256": _require_sha256(manifest.get("audit_json_sha256"), "audit JSON hash"),
        "frozen_benchmark_sha256": _require_sha256(
            manifest.get("frozen_benchmark_sha256"), "frozen benchmark hash"
        ),
        "training_file_sha256s": training_hashes,
        "training_lineage_manifest_sha256": lineage_hash,
    }
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "resolved": resolved,
        "benchmark_rows": benchmark_rows,
        "provenance_rows": provenance_rows,
        "audit": audit,
        "audit_images": audit_images,
        "training_rows": training_rows,
        "training_file_records": training_file_records,
        "training_lineage": training_lineage,
        "source_hashes": source_hashes,
    }


def _reject_remote_schema_references(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"} and (
                not isinstance(child, str) or not child.startswith("#")
            ):
                raise RemediationError(
                    f"{label} contains a non-local {key}; network access is disabled"
                )
            _reject_remote_schema_references(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_remote_schema_references(child, label)


def _load_schemas(benchmark_schema: str | Path, provenance_schema: str | Path) -> dict[str, Any]:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except (ImportError, AttributeError) as exc:
        raise RemediationError(
            "jsonschema with Draft 2020-12 support is required for remediation"
        ) from exc

    benchmark, benchmark_bytes = _strict_json(benchmark_schema, "benchmark schema")
    provenance, provenance_bytes = _strict_json(provenance_schema, "provenance schema")
    actual_hashes = {
        "benchmark_schema_sha256": _sha256_bytes(benchmark_bytes),
        "provenance_schema_sha256": _sha256_bytes(provenance_bytes),
    }
    if actual_hashes != PUBLICATION_SCHEMA_SHA256:
        raise RemediationError("remediation requires byte-identical canonical publication schemas")
    for value, label in (
        (benchmark, "benchmark schema"),
        (provenance, "provenance schema"),
    ):
        _reject_remote_schema_references(value, label)
        try:
            Draft202012Validator.check_schema(value)
        except Exception as exc:
            raise RemediationError(f"invalid Draft 2020-12 {label}: {exc}") from exc
    checker = FormatChecker()
    return {
        "benchmark": benchmark,
        "provenance": provenance,
        "benchmark_validator": Draft202012Validator(benchmark, format_checker=checker),
        "provenance_validator": Draft202012Validator(provenance, format_checker=checker),
        "hashes": actual_hashes,
    }


def _sample_key(row: Mapping[str, Any], row_index: int) -> str:
    sample_id = row.get("sample_id")
    if (
        isinstance(sample_id, str)
        and sample_id
        and sample_id == sample_id.strip()
        and "\x00" not in sample_id
    ):
        return sample_id
    return f"<row:{row_index:06d}>"


def _finding_codes(details: Mapping[str, Any], field: str) -> list[str]:
    findings = details.get(field)
    if not isinstance(findings, list):
        raise RemediationError(f"source audit per-sample {field} is malformed")
    codes: set[str] = set()
    for finding in findings:
        if not isinstance(finding, Mapping):
            raise RemediationError(f"source audit per-sample {field} contains a malformed finding")
        code = finding.get("code")
        if not isinstance(code, str) or not code:
            raise RemediationError("source audit finding has no code")
        codes.add(code)
    return sorted(codes)


def _training_overlap_ids(
    audit: Mapping[str, Any], sample_key: str, canonical_ids: set[str]
) -> list[str]:
    contamination = audit.get("contamination")
    if not isinstance(contamination, Mapping):
        raise RemediationError("source audit contamination section is malformed")
    overlaps = contamination.get("training_paper_overlaps")
    if not isinstance(overlaps, list):
        raise RemediationError("source audit training_paper_overlaps is malformed")
    selected: set[str] = set()
    for overlap in overlaps:
        if not isinstance(overlap, Mapping) or not isinstance(overlap.get("sample_ids"), list):
            raise RemediationError("source audit contains a malformed training paper overlap")
        paper_id = overlap.get("paper_id")
        if sample_key in overlap["sample_ids"]:
            if not isinstance(paper_id, str) or not paper_id:
                raise RemediationError("source audit training paper overlap has no paper_id")
            if paper_id in canonical_ids:
                selected.add(paper_id)
    lineage = contamination.get("training_lineage")
    if isinstance(lineage, Mapping) and isinstance(lineage.get("overlaps"), Mapping):
        lineage_papers = lineage["overlaps"].get("paper_ids", [])
        if not isinstance(lineage_papers, list):
            raise RemediationError("source audit lineage paper overlaps are malformed")
        selected.update(
            value for value in lineage_papers if isinstance(value, str) and value in canonical_ids
        )
    return sorted(selected)


def _build_tasks(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = source["benchmark_rows"]
    provenance_rows = source["provenance_rows"]
    audit = source["audit"]
    per_sample = audit["per_sample_findings"]
    source_hashes = source["source_hashes"]
    tasks: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_sha256 = _sha256_bytes(_json_bytes(row))
        task_digest = hashlib.sha256(
            (f"{source_hashes['prepare_manifest_sha256']}\x00{row_index}\x00{row_sha256}").encode(
                "ascii"
            )
        ).hexdigest()
        sample_key = _sample_key(row, row_index)
        details = per_sample.get(sample_key)
        if not isinstance(details, Mapping):
            raise RemediationError(f"source audit has no per-sample entry for row {row_index}")
        sample_id = row.get("sample_id")
        matching_provenance = []
        for provenance_index, provenance_row in enumerate(provenance_rows):
            if (
                "sample_id" not in row
                or "sample_id" not in provenance_row
                or provenance_row["sample_id"] != sample_id
            ):
                continue
            matching_provenance.append(
                {
                    "source_provenance_row_index": provenance_index,
                    "source_provenance_row_sha256": _sha256_bytes(_json_bytes(provenance_row)),
                    "row": provenance_row,
                }
            )
        canonical_id = canonical_paper_id(row.get("paper_id"))
        canonical_ids = {canonical_id} if canonical_id else set()
        tasks.append(
            {
                "artifact_version": ARTIFACT_VERSION,
                "task_id": f"task_{task_digest}",
                "prepare_manifest_sha256": source_hashes["prepare_manifest_sha256"],
                "source_benchmark_sha256": source_hashes["benchmark_file_sha256"],
                "source_provenance_sha256": source_hashes["provenance_file_sha256"],
                "source_audit_sha256": source_hashes["audit_json_sha256"],
                "source_row_index": row_index,
                "source_row_sha256": row_sha256,
                "original_row": row,
                "legacy_provenance_rows": matching_provenance,
                "audited_image_hashes": source["audit_images"][row_index],
                "critical_codes": _finding_codes(details, "critical_findings"),
                "warning_codes": _finding_codes(details, "warnings"),
                "training_overlap_paper_ids": _training_overlap_ids(
                    audit, sample_key, canonical_ids
                ),
            }
        )
    return tasks


def _decision_binding(task: Mapping[str, Any], queue_fingerprint: str) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "task_id": task["task_id"],
        "prepare_manifest_sha256": task["prepare_manifest_sha256"],
        "source_benchmark_sha256": task["source_benchmark_sha256"],
        "source_provenance_sha256": task["source_provenance_sha256"],
        "source_audit_sha256": task["source_audit_sha256"],
        "source_row_index": task["source_row_index"],
        "source_row_sha256": task["source_row_sha256"],
    }


def _expected_training_sources(
    config: Mapping[str, Any], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    training_audit = config.get("training_audit", {})
    configured = training_audit.get("sources", []) if isinstance(training_audit, Mapping) else []
    records = source["training_file_records"]
    required_count = sum(len(entry["files"]) for entry in configured)
    if len(records) < required_count:
        raise RemediationError("prepare bundle omits configured training source files")
    expected: list[dict[str, Any]] = []
    file_index = 0
    for configured_source in configured:
        files: list[dict[str, Any]] = []
        for reference in configured_source["files"]:
            record = records[file_index]
            file_index += 1
            files.append(
                {
                    "path": reference,
                    "sha256": record["sha256"],
                    "row_count": record["row_count"],
                }
            )
        expected.append(
            {
                "repo_id": configured_source["repo_id"],
                "repo_type": configured_source["repo_type"],
                "revision": configured_source["revision"],
                "files": files,
            }
        )
    return expected


def _queue_material(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
) -> dict[str, Any]:
    policy = _policy_from_config(config)
    source = _load_source_bundle(prepare_manifest)
    current_config_fingerprint = config_fingerprint(config)
    if source["manifest"].get("config_fingerprint") != current_config_fingerprint:
        raise RemediationError("prepare manifest was created from a different configuration")
    try:
        verified_audit = verify_prepared_audit(
            config,
            source["manifest"],
            source["manifest_path"].parent,
        )
    except (BenchmarkAuditError, OSError, PublicationGateError, RuntimeError, ValueError) as exc:
        raise RemediationError(f"fresh prepare audit verification failed: {exc}") from exc
    if verified_audit != source["audit"]:
        raise RemediationError("fresh prepare audit differs from the bound source audit")
    source["expected_training_sources"] = _expected_training_sources(config, source)
    schemas = _load_schemas(benchmark_schema, provenance_schema)
    tasks = _build_tasks(source)
    tasks_bytes = _jsonl_bytes(tasks)
    tasks_sha256 = _sha256_bytes(tasks_bytes)
    fingerprint_payload = {
        "artifact_version": ARTIFACT_VERSION,
        "config_fingerprint": current_config_fingerprint,
        "source_hashes": source["source_hashes"],
        "schema_hashes": schemas["hashes"],
        "policy": policy,
        "task_count": len(tasks),
        "tasks_sha256": tasks_sha256,
    }
    queue_fingerprint = _sha256_bytes(_json_bytes(fingerprint_payload))
    templates = [
        {
            **_decision_binding(task, queue_fingerprint),
            "status": "pending",
            "disposition": None,
            "exclusion_reason": None,
            "benchmark_row": None,
            "provenance_row": None,
            "reviewed_by": [],
            "notes": "",
        }
        for task in tasks
    ]
    templates_bytes = _jsonl_bytes(templates)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "config_fingerprint": current_config_fingerprint,
        "source_hashes": source["source_hashes"],
        "schema_hashes": schemas["hashes"],
        "policy": policy,
        "task_count": len(tasks),
        "tasks_file": TASKS_FILENAME,
        "tasks_sha256": tasks_sha256,
        "decision_template_file": DECISION_TEMPLATE_FILENAME,
        "decision_template_sha256": _sha256_bytes(templates_bytes),
        "queue_fingerprint": queue_fingerprint,
    }
    return {
        "source": source,
        "schemas": schemas,
        "tasks": tasks,
        "templates": templates,
        "manifest": manifest,
        "files": {
            TASKS_FILENAME: tasks_bytes,
            DECISION_TEMPLATE_FILENAME: templates_bytes,
            QUEUE_MANIFEST_FILENAME: _json_bytes(manifest, newline=True),
        },
    }


def _output_target(output_dir: str | Path) -> Path:
    raw = Path(output_dir)
    if raw.name in {"", ".", ".."}:
        raise RemediationError("output_dir must name a new directory")
    try:
        raw.parent.mkdir(parents=True, exist_ok=True)
        parent = raw.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RemediationError(f"cannot create output_dir parent: {exc}") from exc
    target = parent / raw.name
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise RemediationError(f"output_dir is not a safe directory: {target}")
    return target


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RemediationError(f"refusing to overwrite staged file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RemediationError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RemediationError(f"refusing to overwrite staged asset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RemediationError(f"temporary asset path already exists: {temporary}")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise RemediationError(f"cannot copy curated asset {source}: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _existing_queue_is_identical(target: Path, files: Mapping[str, bytes]) -> bool:
    try:
        entries = list(target.iterdir())
    except OSError as exc:
        raise RemediationError(f"cannot inspect existing output_dir: {exc}") from exc
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        return False
    if {entry.name for entry in entries} != set(files):
        return False
    try:
        return all((target / name).read_bytes() == data for name, data in files.items())
    except OSError as exc:
        raise RemediationError(f"cannot inspect existing queue workspace: {exc}") from exc


def _publish_queue(output_dir: str | Path, files: Mapping[str, bytes]) -> None:
    target = _output_target(output_dir)
    if target.exists():
        if _existing_queue_is_identical(target, files):
            return
        raise RemediationError("output_dir already contains a non-identical remediation workspace")
    staging = target.with_name(f".{target.name}.queue.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise RemediationError(f"queue staging path already exists: {staging}")
    try:
        staging.mkdir()
        for name, data in files.items():
            _atomic_write(staging / name, data)
        if target.exists() or target.is_symlink():
            if target.is_dir() and _existing_queue_is_identical(target, files):
                return
            raise RemediationError("output_dir appeared while the queue was being staged")
        os.replace(staging, target)
    finally:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)


def generate_curator_queues(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    output_dir: str | Path,
    *,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
) -> dict[str, Any]:
    """Generate deterministic remediation tasks and blank curator decisions."""

    material = _queue_material(
        config,
        prepare_manifest,
        benchmark_schema,
        provenance_schema,
    )
    _publish_queue(output_dir, material["files"])
    return dict(material["manifest"])


def _verify_queue_workspace(
    material: Mapping[str, Any], queue_manifest: str | Path
) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(queue_manifest)
    if manifest_path.is_symlink():
        raise RemediationError("queue_manifest must not be a symlink")
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RemediationError(f"queue_manifest does not exist: {queue_manifest}") from exc
    actual_manifest, actual_bytes = _strict_json(manifest_path, "queue manifest")
    expected_manifest = material["manifest"]
    expected_bytes = material["files"][QUEUE_MANIFEST_FILENAME]
    if actual_bytes != expected_bytes or actual_manifest != expected_manifest:
        raise RemediationError("queue manifest or fingerprint does not match the prepare bundle")
    root = manifest_path.parent
    if manifest_path.name != QUEUE_MANIFEST_FILENAME:
        raise RemediationError(f"queue manifest must be named {QUEUE_MANIFEST_FILENAME}")
    expected_names = {
        QUEUE_MANIFEST_FILENAME,
        TASKS_FILENAME,
        DECISION_TEMPLATE_FILENAME,
    }
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise RemediationError(f"cannot inspect queue workspace: {exc}") from exc
    if {entry.name for entry in entries} != expected_names or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise RemediationError("queue workspace contains unlisted, missing, or unsafe artifacts")
    for name, expected_rows_key in (
        (TASKS_FILENAME, "tasks"),
        (DECISION_TEMPLATE_FILENAME, "templates"),
    ):
        rows, raw = _strict_jsonl(root / name, f"queue {name}")
        if raw != material["files"][name] or rows != material[expected_rows_key]:
            raise RemediationError(f"queue artifact was tampered: {name}")
    return root, actual_manifest


def _decision_error(message: str) -> DecisionValidationError:
    return DecisionValidationError(message)


def _validate_reviewers(value: Any, task_id: str) -> list[str]:
    if not isinstance(value, list):
        raise _decision_error(f"decision {task_id} reviewed_by must be an array")
    reviewers: list[str] = []
    for reviewer in value:
        if (
            not isinstance(reviewer, str)
            or not reviewer
            or reviewer != reviewer.strip()
            or "\x00" in reviewer
        ):
            raise _decision_error(
                f"decision {task_id} reviewed_by identifiers must be non-empty and trimmed"
            )
        reviewers.append(reviewer)
    normalized = [
        _normalized_identifier(
            reviewer,
            f"decision {task_id} reviewed_by",
            ascii_reviewer=True,
        )
        for reviewer in reviewers
    ]
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise _decision_error(f"decision {task_id} requires two distinct reviewed_by identifiers")
    return normalized


def _validate_decisions(
    decisions: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[str]]:
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    decisions_by_id: dict[str, Mapping[str, Any]] = {}
    for decision_index, decision in enumerate(decisions):
        if set(decision) != _DECISION_FIELDS:
            missing = sorted(_DECISION_FIELDS - set(decision))
            extra = sorted(set(decision) - _DECISION_FIELDS)
            raise _decision_error(
                f"decision row {decision_index} fields differ: missing={missing}, extra={extra}"
            )
        task_id = decision.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise _decision_error(f"decision row {decision_index} has no task_id")
        if task_id in decisions_by_id:
            raise _decision_error(f"duplicate decision for task {task_id}")
        if task_id not in tasks_by_id:
            raise _decision_error(f"decision references unknown task {task_id}")
        decisions_by_id[task_id] = decision
    missing_ids = sorted(set(tasks_by_id) - set(decisions_by_id))
    if missing_ids:
        raise _decision_error(f"missing decisions for tasks: {missing_ids}")

    retained: list[tuple[dict[str, Any], dict[str, Any]]] = []
    exclusions: list[dict[str, Any]] = []
    all_reviewers: set[str] = set()
    for task in tasks:
        task_id = str(task["task_id"])
        decision = decisions_by_id[task_id]
        expected_binding = _decision_binding(task, queue_fingerprint)
        for field, expected in expected_binding.items():
            if decision.get(field) != expected:
                raise _decision_error(f"decision {task_id} has a tampered {field} binding")
        status = decision.get("status")
        if status == "pending":
            raise _decision_error(f"decision {task_id} is still pending")
        if status != "complete":
            raise _decision_error(f"decision {task_id} status must be complete")
        reviewers = _validate_reviewers(decision.get("reviewed_by"), task_id)
        all_reviewers.update(reviewers)
        notes = decision.get("notes")
        if not isinstance(notes, str):
            raise _decision_error(f"decision {task_id} notes must be a string")
        disposition = decision.get("disposition")
        if disposition == "exclude":
            reason = decision.get("exclusion_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise _decision_error(f"excluded decision {task_id} requires a non-empty reason")
            if (
                decision.get("benchmark_row") is not None
                or decision.get("provenance_row") is not None
            ):
                raise _decision_error(
                    f"excluded decision {task_id} must not contain replacement objects"
                )
            exclusions.append(
                {
                    "task_id": task_id,
                    "source_row_index": task["source_row_index"],
                    "reason": reason,
                }
            )
        elif disposition == "retain":
            if decision.get("exclusion_reason") is not None:
                raise _decision_error(
                    f"retained decision {task_id} must have a null exclusion_reason"
                )
            benchmark_row = decision.get("benchmark_row")
            provenance_row = decision.get("provenance_row")
            if not isinstance(benchmark_row, dict) or not isinstance(provenance_row, dict):
                raise _decision_error(
                    f"retained decision {task_id} requires complete replacement objects"
                )
            retained.append((benchmark_row, provenance_row))
        else:
            raise _decision_error(f"decision {task_id} disposition must be exclude or retain")
    return retained, exclusions, sorted(all_reviewers)


def _schema_error(validator: Any, value: Mapping[str, Any], label: str) -> None:
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
            error.message,
        ),
    )
    if not errors:
        return
    error = errors[0]
    path = "/".join(str(part) for part in error.absolute_path) or "<root>"
    raise _decision_error(f"{label} schema validation failed at {path}: {error.message}")


def _normalized_identifier(value: Any, label: str, *, ascii_reviewer: bool = False) -> str:
    result = normalize_identity(value, ascii_reviewer=ascii_reviewer)
    if not result:
        raise _decision_error(
            f"{label} is empty, non-canonical, or contains unsafe, invisible or control characters"
        )
    return result


def _validate_distinct_identifiers(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise _decision_error(f"{label} must be an array")
    normalized = [
        _normalized_identifier(identifier, label, ascii_reviewer=True) for identifier in value
    ]
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise _decision_error(f"{label} requires two distinct identifiers")
    return normalized


def _validated_paper_id(row: Mapping[str, Any], label: str) -> str:
    raw = row.get("paper_id")
    _normalized_identifier(raw, f"{label} paper_id")
    canonical = canonical_paper_id(raw)
    identity_errors = paper_identity_errors(row)
    if identity_errors:
        unsafe_errors = [error for error in identity_errors if "unsafe" in error]
        if unsafe_errors:
            raise _decision_error(f"{label} paper identity is invalid: {unsafe_errors[0]}")
        if "paper identifier aliases conflict with top-level paper_id" in identity_errors:
            raise _decision_error(
                f"{label} contains identifiers that conflict with canonical paper_id"
            )
        raise _decision_error(f"{label} paper identity is invalid: {identity_errors[0]}")
    if raw != canonical:
        raise _decision_error(f"{label} paper_id must use its canonical representation")
    return canonical


def _windows_path_key(parts: Sequence[str], label: str) -> tuple[str, ...]:
    key: list[str] = []
    for part in parts:
        if part != part.rstrip(" ."):
            raise _decision_error(f"{label} contains a Windows trailing-dot/space alias")
        if any(ord(character) < 32 or character in '<>"|?*' for character in part):
            raise _decision_error(f"{label} contains a Windows-unsafe path character")
        device_name = (
            part.split(".", 1)[0].rstrip(" ").translate(_WINDOWS_SUPERSCRIPT_DIGITS).casefold()
        )
        if device_name in _WINDOWS_RESERVED_NAMES:
            raise _decision_error(f"{label} contains a reserved Windows device name")
        key.append(part.casefold())
    return tuple(key)


def _safe_asset_reference(value: Any) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise _decision_error("image path must be a non-empty string without NUL")
    if _URI_SCHEME_RE.match(value) or value.startswith(("//", "\\\\")):
        raise _decision_error(f"image path must be a local relative path: {value!r}")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if windows.is_absolute() or windows.drive or posix.is_absolute() or ".." in posix.parts:
        raise _decision_error(f"unsafe image path: {value!r}")
    parts = tuple(part for part in posix.parts if part not in {"", "."})
    normalized = "/".join(parts)
    if not parts or normalized != value or any(":" in part for part in parts):
        raise _decision_error(f"image path is not a normalized safe relative path: {value!r}")
    key = _windows_path_key(parts, f"image path {value!r}")
    for reserved in _RESERVED_RELEASE_PATHS:
        reserved_key = tuple(part.casefold() for part in reserved.split("/"))
        shared = min(len(key), len(reserved_key))
        if key[:shared] == reserved_key[:shared] and (
            len(key) == shared or len(reserved_key) == shared
        ):
            raise _decision_error(f"image path collides with a release artifact: {value!r}")
    if len(parts) < 3 or parts[:2] != ("assets", "images"):
        raise _decision_error("image path must start with the canonical assets/images/ prefix")
    return normalized, parts


def _validate_candidate_rows(
    retained: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    schemas: Mapping[str, Any],
    curated_dataset_root: str | Path,
    *,
    require_gold: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[Path, str]]]:
    try:
        curated_root = Path(curated_dataset_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _decision_error(
            f"curated_dataset_root does not exist: {curated_dataset_root}"
        ) from exc
    if not curated_root.is_dir():
        raise _decision_error("curated_dataset_root must be a directory")

    benchmark_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for index, (benchmark_row, provenance_row) in enumerate(retained):
        _schema_error(schemas["benchmark_validator"], benchmark_row, f"benchmark row {index}")
        _schema_error(schemas["provenance_validator"], provenance_row, f"provenance row {index}")
        benchmark_paper_id = _validated_paper_id(benchmark_row, f"benchmark row {index}")
        provenance_paper_id = _validated_paper_id(provenance_row, f"provenance row {index}")
        if benchmark_paper_id != provenance_paper_id:
            raise _decision_error(
                f"benchmark and provenance row {index} canonical paper_id values differ"
            )
        if require_gold:
            if not isinstance(benchmark_row.get("gold_answer"), Mapping) or not isinstance(
                benchmark_row.get("rubric"), Mapping
            ):
                raise _decision_error(
                    f"benchmark row {index} requires adjudicated gold_answer and rubric"
                )
            _validate_distinct_identifiers(
                benchmark_row["rubric"].get("adjudicators"),
                f"benchmark row {index} rubric adjudicators",
            )
        benchmark_rows.append(benchmark_row)
        provenance_rows.append(provenance_row)

    benchmark_ids = [row.get("sample_id") for row in benchmark_rows]
    provenance_ids = [row.get("sample_id") for row in provenance_rows]
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise _decision_error("retained benchmark sample_ids must be unique")
    if len(provenance_ids) != len(set(provenance_ids)):
        raise _decision_error("exactly one provenance row is required per retained sample")
    if set(benchmark_ids) != set(provenance_ids):
        raise _decision_error("benchmark and provenance sample_id sets must be exactly equal")
    provenance_by_id = {row["sample_id"]: row for row in provenance_rows}

    assets: dict[str, tuple[Path, str]] = {}
    windows_paths: dict[tuple[str, ...], str] = {}
    hash_cache: dict[Path, str] = {}
    for benchmark_row in benchmark_rows:
        sample_id = benchmark_row["sample_id"]
        provenance_row = provenance_by_id[sample_id]
        benchmark_images = benchmark_row.get("images")
        provenance_images = provenance_row.get("images")
        if not isinstance(benchmark_images, list) or not isinstance(provenance_images, list):
            raise _decision_error(f"sample {sample_id} image arrays are malformed")
        provenance_paths = [
            entry.get("image_path") if isinstance(entry, Mapping) else None
            for entry in provenance_images
        ]
        if provenance_paths != benchmark_images:
            raise _decision_error(
                f"sample {sample_id} provenance image paths must match benchmark images in order"
            )
        for image_index, (reference, provenance_image) in enumerate(
            zip(benchmark_images, provenance_images)
        ):
            normalized, parts = _safe_asset_reference(reference)
            _validate_distinct_identifiers(
                provenance_image.get("verified_by"),
                f"sample {sample_id} image {image_index} verified_by",
            )
            declared = provenance_image.get("sha256")
            if not isinstance(declared, str) or not _SHA256_RE.fullmatch(declared):
                raise _decision_error(
                    f"sample {sample_id} image {image_index} requires a lowercase SHA256"
                )
            try:
                source = resolve_dataset_file(curated_root, Path(*parts))
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise _decision_error(
                    f"sample {sample_id} image is missing or unsafe: {reference}"
                ) from exc
            if source not in hash_cache:
                hash_cache[source] = _sha256_file(source)[0]
            if hash_cache[source] != declared:
                raise _decision_error(
                    f"sample {sample_id} declared image SHA256 does not match bytes: {reference}"
                )
            path_key = _windows_path_key(parts, f"image path {reference!r}")
            previous = windows_paths.get(path_key)
            if previous is not None and previous != normalized:
                raise _decision_error(
                    f"case-insensitive image path collision: {previous!r} and {normalized!r}"
                )
            for existing_key, existing_path in windows_paths.items():
                shared = min(len(path_key), len(existing_key))
                if path_key[:shared] == existing_key[:shared] and len(path_key) != len(
                    existing_key
                ):
                    raise _decision_error(
                        f"file/directory image path collision: {existing_path!r} and {normalized!r}"
                    )
            windows_paths[path_key] = normalized
            existing = assets.get(normalized)
            if existing is not None and existing != (source, declared):
                raise _decision_error(f"conflicting curated assets for path {normalized!r}")
            assets[normalized] = (source, declared)
    benchmark_rows.sort(key=lambda row: row["sample_id"])
    provenance_rows.sort(key=lambda row: row["sample_id"])
    return benchmark_rows, provenance_rows, assets


def _write_audit_atomic(staging: Path, report: Mapping[str, Any]) -> None:
    audit_dir = staging / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    json_temp = audit_dir / ".benchmark_audit.json.write"
    markdown_temp = audit_dir / ".benchmark_audit.md.write"
    json_target = audit_dir / "benchmark_audit.json"
    markdown_target = audit_dir / "benchmark_audit.md"
    try:
        write_audit_report(report, json_temp, markdown_temp)
        for path in (json_temp, markdown_temp):
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        os.replace(json_temp, json_target)
        os.replace(markdown_temp, markdown_target)
    except (OSError, BenchmarkAuditError) as exc:
        raise RemediationError(f"cannot write candidate audit: {exc}") from exc
    finally:
        for path in (json_temp, markdown_temp):
            if path.exists():
                path.unlink()


def _verify_staged_assets(staging: Path, assets: Mapping[str, tuple[Path, str]]) -> None:
    for relative, (_, expected_hash) in assets.items():
        destination = staging.joinpath(*relative.split("/"))
        if destination.is_symlink() or not destination.is_file():
            raise RemediationError(f"staged asset is missing or unsafe: {relative}")
        actual_hash, _ = _sha256_file(destination)
        if actual_hash != expected_hash:
            raise RemediationError(f"staged asset hash changed unexpectedly: {relative}")


def _tree_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            if (directory_path / dirname).is_symlink():
                raise RemediationError("release tree contains a symlinked directory")
        for filename in filenames:
            path = directory_path / filename
            if path.is_symlink() or not path.is_file():
                raise RemediationError("release tree contains an unsafe file")
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _output_file_records(staging: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _tree_files(staging):
        digest, size = _sha256_file(path)
        records.append(
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return records


def _files_equal(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_handle, right.open("rb") as right_handle:
            while True:
                left_block = left_handle.read(1024 * 1024)
                right_block = right_handle.read(1024 * 1024)
                if left_block != right_block:
                    return False
                if not left_block:
                    return True
    except OSError as exc:
        raise RemediationError(f"cannot compare release candidates: {exc}") from exc


def _trees_identical(left: Path, right: Path) -> bool:
    left_files = _tree_files(left)
    right_files = _tree_files(right)
    left_by_name = {path.relative_to(left).as_posix(): path for path in left_files}
    right_by_name = {path.relative_to(right).as_posix(): path for path in right_files}
    if set(left_by_name) != set(right_by_name):
        return False
    return all(_files_equal(left_by_name[name], right_by_name[name]) for name in left_by_name)


def _publish_release(staging: Path, target: Path) -> dict[str, Any]:
    if target.exists():
        if _trees_identical(staging, target):
            existing, _ = _strict_json(target / ASSEMBLY_MANIFEST_FILENAME, "assembly manifest")
            return existing
        raise RemediationError("output_dir already contains a different release candidate")
    if target.is_symlink():
        raise RemediationError("output_dir is a symlink")
    try:
        os.replace(staging, target)
    except OSError as exc:
        if target.is_dir() and _trees_identical(staging, target):
            existing, _ = _strict_json(target / ASSEMBLY_MANIFEST_FILENAME, "assembly manifest")
            return existing
        raise RemediationError(f"cannot publish release candidate atomically: {exc}") from exc
    manifest, _ = _strict_json(target / ASSEMBLY_MANIFEST_FILENAME, "assembly manifest")
    return manifest


def assemble_corrected_release(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    queue_manifest: str | Path,
    decisions_jsonl: str | Path,
    curated_dataset_root: str | Path,
    output_dir: str | Path,
    *,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
) -> dict[str, Any]:
    """Validate curator decisions and stage a benchmark-only release candidate."""

    material = _queue_material(
        config,
        prepare_manifest,
        benchmark_schema,
        provenance_schema,
    )
    _, verified_queue = _verify_queue_workspace(material, queue_manifest)
    decisions, decisions_bytes = _strict_jsonl(decisions_jsonl, "curator decisions JSONL")
    retained, exclusions, reviewer_ids = _validate_decisions(
        decisions,
        material["tasks"],
        verified_queue["queue_fingerprint"],
    )
    policy = verified_queue["policy"]
    benchmark_rows, provenance_rows, assets = _validate_candidate_rows(
        retained,
        material["schemas"],
        curated_dataset_root,
        require_gold=policy["require_gold"],
    )

    target = _output_target(output_dir)
    staging = target.with_name(f".{target.name}.assembly.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise RemediationError(f"release staging path already exists: {staging}")
    try:
        staging.mkdir()
        _atomic_write(
            staging / "data" / "task3_vlm_generation.jsonl",
            _jsonl_bytes(benchmark_rows),
        )
        _atomic_write(
            staging / "article_image_sources.jsonl",
            _jsonl_bytes(provenance_rows),
        )
        for relative, (source, expected_hash) in sorted(assets.items()):
            destination = staging.joinpath(*relative.split("/"))
            _atomic_copy(source, destination)
            copied_hash, _ = _sha256_file(destination)
            if copied_hash != expected_hash:
                raise RemediationError(f"copied asset hash changed unexpectedly: {relative}")

        report = audit_benchmark(
            benchmark_rows,
            staging,
            provenance_rows=provenance_rows,
            training_rows=material["source"]["training_rows"],
            training_lineage=material["source"]["training_lineage"],
            expected_training_sources=material["source"]["expected_training_sources"],
            require_gold=policy["require_gold"],
            require_complete_provenance=True,
            require_split_provenance=True,
            require_training_lineage=False,
            require_provenance_order=True,
            require_citation=True,
            blocked_warning_codes=_FROZEN_WARNING_CODES,
            primary_strata=policy["primary_strata"],
        )
        if report["critical_findings"]:
            codes = sorted({finding["code"] for finding in report["critical_findings"]})
            raise _decision_error(f"corrected benchmark audit has critical findings: {codes}")
        frozen_warning_codes = set(policy["warning_codes_must_be_zero"])
        warning_codes = {finding["code"] for finding in report["warnings"]}
        blocked_warnings = sorted(frozen_warning_codes & warning_codes)
        if blocked_warnings:
            raise _decision_error(
                f"corrected benchmark audit has frozen warning findings: {blocked_warnings}"
            )
        primary_papers = {
            canonical_paper_id(row.get("paper_id"))
            for row in benchmark_rows
            if row.get("primary_endpoint") is True
        }
        primary_papers.discard("")
        minimum = policy["minimum_primary_papers"]
        if len(primary_papers) < minimum:
            raise _decision_error(
                "corrected benchmark has fewer unique primary paper IDs than required: "
                f"{len(primary_papers)} < {minimum}"
            )
        report["technical_audit_status"] = report["status"]
        report["technical_audit_passed"] = True
        report["scope"] = "validated_benchmark_release_candidate"
        report["publication_ready"] = False
        report["publication_readiness_blockers"] = list(_REMAINING_REQUIREMENTS)
        _write_audit_atomic(staging, report)
        _verify_staged_assets(staging, assets)

        output_files = _output_file_records(staging)
        lineage_provided = material["source"]["training_lineage"] is not None
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "scope": "validated_benchmark_release_candidate",
            "publication_ready": False,
            "technical_audit_passed": True,
            "config_fingerprint": verified_queue["config_fingerprint"],
            "source_hashes": verified_queue["source_hashes"],
            "schema_hashes": verified_queue["schema_hashes"],
            "queue_fingerprint": verified_queue["queue_fingerprint"],
            "decisions_sha256": _sha256_bytes(decisions_bytes),
            "counts": {
                "source_tasks": verified_queue["task_count"],
                "retained_rows": len(benchmark_rows),
                "excluded_rows": len(exclusions),
                "provenance_rows": len(provenance_rows),
                "copied_assets": len(assets),
                "unique_primary_papers": len(primary_papers),
            },
            "exclusions": exclusions,
            "reviewer_ids": reviewer_ids,
            "training_lineage": {
                "provided": lineage_provided,
                "validated_in_candidate_audit": lineage_provided,
                "required_for_strict_prepare": True,
                "unresolved_domains": (
                    []
                    if lineage_provided
                    else [
                        "paper_ids",
                        "source_documents",
                        "creator_groups",
                        "image_bytes",
                        "prompts",
                    ]
                ),
            },
            "output_files": output_files,
            "remaining_requirements": list(_REMAINING_REQUIREMENTS),
        }
        _atomic_write(
            staging / ASSEMBLY_MANIFEST_FILENAME,
            _json_bytes(manifest, newline=True),
        )
        _verify_staged_assets(staging, assets)
        return _publish_release(staging, target)
    except (BenchmarkAuditError, OSError) as exc:
        raise RemediationError(f"cannot assemble corrected release: {exc}") from exc
    finally:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)


__all__ = [
    "DecisionValidationError",
    "RemediationError",
    "assemble_corrected_release",
    "generate_curator_queues",
]
