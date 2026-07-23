#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build a deterministic, non-importable capacity-enrichment handoff workspace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ARTIFACT_VERSION = 1
TOTAL_ROWS = 150
QUEUE_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TASK_ID_RE = re.compile(r"^task_[0-9a-f]{64}$")
CANONICAL_PAPER_ID_RE = re.compile(
    r"^(?:doi:10\.\d{4,9}/[^\s]+|"
    r"arxiv:(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})|"
    r"paper:\S(?:[\x20-\x7e]*\S)?)$"
)
PARTITION_COUNTS = {
    "source_ready_lower_bound": 48,
    "metadata_only_or_unlicensed": 91,
    "unresolved_identity": 5,
    "duplicate_candidate": 1,
    "no_retrieval_hint": 5,
}
CORRECTION_ROWS = frozenset({26, 52, 78, 114})
IDENTITY_EXCEPTION_ROWS = frozenset({25, 27, 58, 109, 111, 112, 113, 115, 116})
STRATA = frozenset({"multimodal_hard", "temporal_hard", "easy_control"})
HTML_NONCE = "capacity-handoff-v1"
PAYLOAD_FILENAMES = frozenset(
    {
        "enrichment_requirements.jsonl",
        "source_submission_template.csv",
        "identity_resolution_template.csv",
        "source_submission_form.html",
        "identity_resolution_form.html",
        "training_lineage_form.html",
        "README_RU.md",
    }
)
MANIFEST_FILENAME = "handoff_manifest.json"
SOURCE_PROBE_FIELDS = frozenset(
    {
        "row",
        "group_id",
        "dossier_canonical_paper_id",
        "verified_canonical_paper_id",
        "metadata_url",
        "direct_image_url",
        "direct_image_sha256",
        "license",
        "license_url",
        "citation_locator",
        "verification_status",
        "human_verified",
        "publication_ready",
    }
)
IDENTITY_EXCEPTION_FIELDS = frozenset(
    {
        "row",
        "group_id",
        "current_canonical_paper_id",
        "candidate_canonical_paper_id",
        "disposition",
        "reason",
        "verification_status",
        "human_verified",
        "publication_ready",
    }
)
AUDIT_POLICY = {
    "network_probe_performed": True,
    "direct_image_bytes_sha256_checked": True,
    "source_probes_machine_only": True,
    "source_probes_human_verified": False,
    "license_human_verified": False,
    "external_verification_claimed": False,
    "automatic_retain": False,
    "publication_ready": False,
}


class HandoffError(ValueError):
    """Raised when a handoff input or output violates the fail-closed contract."""


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HandoffError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise HandoffError(f"non-finite JSON value is forbidden: {value}")


def _strict_loads(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_nonfinite,
        )
    except HandoffError:
        raise
    except (TypeError, ValueError) as exc:
        raise HandoffError(f"invalid {label}: {exc}") from exc


def _stable_read(path: str | Path, label: str) -> tuple[Path, bytes]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise HandoffError(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        before = resolved.stat()
        if not resolved.is_file():
            raise HandoffError(f"{label} must be a regular file")
        data = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise HandoffError(f"cannot read {label}: {exc}") from exc
    before_state = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_state = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_state != after_state or len(data) != after.st_size:
        raise HandoffError(f"{label} changed while it was being read")
    return resolved, data


def _decode_utf8(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HandoffError(f"{label} must be UTF-8") from exc


def _strict_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any], bytes]:
    resolved, data = _stable_read(path, label)
    value = _strict_loads(_decode_utf8(data, label), label)
    if not isinstance(value, dict):
        raise HandoffError(f"{label} must contain one JSON object")
    return resolved, value, data


def _strict_jsonl(path: str | Path, label: str) -> tuple[Path, list[dict[str, Any]], bytes]:
    resolved, data = _stable_read(path, label)
    text = _decode_utf8(data, label)
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise HandoffError(f"{label} must contain non-empty JSON objects on every line")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        value = _strict_loads(line, f"{label} line {index}")
        if not isinstance(value, dict):
            raise HandoffError(f"{label} line {index} must be a JSON object")
        rows.append(value)
    return resolved, rows, data


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise HandoffError(f"{label} fields are invalid")


def _row_number(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= TOTAL_ROWS:
        raise HandoffError(f"{label} must be an integer from 1 through {TOTAL_ROWS}")
    return value


def _trimmed_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip() or (not value and not allow_empty):
        raise HandoffError(f"{label} must be a trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise HandoffError(f"{label} contains control characters")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise HandoffError(f"{label} must be a lowercase SHA256")
    return value


def _is_canonical_paper_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and all("\x20" <= character <= "\x7e" for character in value)
        and value == value.lower()
        and CANONICAL_PAPER_ID_RE.fullmatch(value) is not None
    )


def _canonical_paper_id(value: Any, label: str) -> str:
    result = _trimmed_string(value, label)
    if not _is_canonical_paper_id(result):
        raise HandoffError(f"{label} must be a lowercase canonical paper ID")
    return result


def _safe_url(value: Any, label: str) -> str:
    url = _trimmed_string(value, label)
    if not url.isascii() or any(character.isspace() for character in url):
        raise HandoffError(f"{label} must be an ASCII HTTP(S) URL")
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as exc:
        raise HandoffError(f"{label} is malformed") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise HandoffError(f"{label} must be HTTP(S) without credentials, query, or fragment")
    return url


def _string_list(value: Any, label: str, *, task_ids: bool = False) -> list[str]:
    if not isinstance(value, list) or not value:
        raise HandoffError(f"{label} must be a non-empty array")
    result: list[str] = []
    for index, item in enumerate(value):
        item = _trimmed_string(item, f"{label}[{index}]")
        if task_ids and not TASK_ID_RE.fullmatch(item):
            raise HandoffError(f"{label}[{index}] is not a task ID")
        result.append(item)
    if len(result) != len(set(result)):
        raise HandoffError(f"{label} contains duplicates")
    return result


def _possibly_empty_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise HandoffError(f"{label} must be an array")
    result = [_trimmed_string(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise HandoffError(f"{label} contains duplicates")
    return result


def _task_summaries(
    value: Any,
    task_ids: Sequence[str],
    primary_task_ids: Sequence[str],
    row_number: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    label = f"dossier row {row_number} tasks"
    if not isinstance(value, list) or len(value) != len(task_ids):
        raise HandoffError(f"{label} must exactly match task_ids")
    summaries: list[dict[str, Any]] = []
    quarantined: set[str] = set()
    seen: set[str] = set()
    for index, task in enumerate(value):
        task_label = f"{label}[{index}]"
        if not isinstance(task, dict):
            raise HandoffError(f"{task_label} must be an object")
        required = {
            "binding",
            "machine_retrieval_url",
            "source_row_hint",
            "audited_image_hints",
            "critical_codes",
            "warning_codes",
            "required_actions",
        }
        if not required.issubset(task):
            raise HandoffError(f"{task_label} is missing safe summary fields")
        binding = task["binding"]
        if not isinstance(binding, dict):
            raise HandoffError(f"{task_label} binding must be an object")
        task_id = binding.get("task_id")
        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            raise HandoffError(f"{task_label} binding has an invalid task_id")
        source_row_index = binding.get("source_row_index")
        if (
            isinstance(source_row_index, bool)
            or not isinstance(source_row_index, int)
            or source_row_index < 0
        ):
            raise HandoffError(f"{task_label} binding has an invalid source_row_index")
        if task_id in seen:
            raise HandoffError(f"{label} repeats task_id {task_id}")
        seen.add(task_id)
        hint = task["source_row_hint"]
        if not isinstance(hint, dict):
            raise HandoffError(f"{task_label} source_row_hint must be an object")
        stratum = hint.get("stratum")
        if stratum not in STRATA | {""}:
            raise HandoffError(f"{task_label} source_row_hint has an invalid stratum")
        if not isinstance(hint.get("primary_endpoint"), bool):
            raise HandoffError(f"{task_label} source_row_hint has no primary_endpoint")
        if hint.get("legacy_prompt_text_included") is not False:
            raise HandoffError(f"{task_label} exposes legacy prompt text")
        retrieval_url = task["machine_retrieval_url"]
        if retrieval_url is not None:
            retrieval_url = _safe_url(retrieval_url, f"{task_label} machine_retrieval_url")
        critical_codes = _possibly_empty_string_list(
            task["critical_codes"], f"{task_label} critical_codes"
        )
        images = task["audited_image_hints"]
        if not isinstance(images, list):
            raise HandoffError(f"{task_label} audited_image_hints must be an array")
        if "cross_paper_image_reuse" in critical_codes and not images:
            raise HandoffError(f"{task_label} cross-paper reuse has no image hints")
        for image_index, image in enumerate(images):
            image_label = f"{task_label} audited_image_hints[{image_index}]"
            if not isinstance(image, dict):
                raise HandoffError(f"{image_label} must be an object")
            digest = _sha256(image.get("sha256"), f"{image_label} sha256")
            status = _trimmed_string(image.get("evidence_status"), f"{image_label} status")
            if status not in {
                "quarantined_cross_paper_reuse",
                "pending_external_source_verification",
            }:
                raise HandoffError(f"{image_label} has an unknown evidence status")
            if image.get("release_usable") is not False:
                raise HandoffError(f"{image_label} is incorrectly marked release-usable")
            quarantined.add(digest)
            if "cross_paper_image_reuse" in critical_codes and status != (
                "quarantined_cross_paper_reuse"
            ):
                raise HandoffError(f"{image_label} fails to quarantine reused bytes")
        summaries.append(
            {
                "task_id": task_id,
                "source_row_index": source_row_index,
                "stratum": stratum,
                "primary_endpoint": hint["primary_endpoint"],
                "machine_retrieval_url": retrieval_url,
                "critical_codes": critical_codes,
                "warning_codes": _possibly_empty_string_list(
                    task["warning_codes"], f"{task_label} warning_codes"
                ),
                "required_action_codes": _possibly_empty_string_list(
                    task["required_actions"], f"{task_label} required_actions"
                ),
                "is_primary": task_id in primary_task_ids,
                "machine_only": True,
            }
        )
    if [summary["task_id"] for summary in summaries] != list(task_ids):
        raise HandoffError(f"{label} order differs from task_ids")
    return summaries, quarantined


def _validate_partition(value: Any) -> tuple[dict[str, list[int]], dict[int, str]]:
    if not isinstance(value, dict) or set(value) != {"total_rows", "categories"}:
        raise HandoffError("source audit partition is malformed")
    if value["total_rows"] != TOTAL_ROWS:
        raise HandoffError(f"source audit partition must contain exactly {TOTAL_ROWS} rows")
    categories = value["categories"]
    if not isinstance(categories, dict) or set(categories) != set(PARTITION_COUNTS):
        raise HandoffError("source audit partition categories are invalid")
    normalized: dict[str, list[int]] = {}
    row_category: dict[int, str] = {}
    for name, expected_count in PARTITION_COUNTS.items():
        rows = categories[name]
        if not isinstance(rows, list):
            raise HandoffError(f"partition category {name} must be an array")
        checked = [_row_number(row, f"partition category {name}") for row in rows]
        if checked != sorted(checked) or len(checked) != len(set(checked)):
            raise HandoffError(f"partition category {name} must be sorted and unique")
        if len(checked) != expected_count:
            raise HandoffError(
                f"partition category {name} requires {expected_count} rows; found {len(checked)}"
            )
        for row in checked:
            if row in row_category:
                raise HandoffError(f"partition row {row} appears in multiple categories")
            row_category[row] = name
        normalized[name] = checked
    if set(row_category) != set(range(1, TOTAL_ROWS + 1)):
        raise HandoffError("partition must cover rows 1 through 150 exactly once")
    return normalized, row_category


def _validate_dossiers(
    rows: Sequence[Mapping[str, Any]], queue_fingerprint: str
) -> list[dict[str, Any]]:
    if len(rows) != TOTAL_ROWS:
        raise HandoffError(f"dossiers must contain exactly {TOTAL_ROWS} rows")
    result: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_task_ids: set[str] = set()
    seen_source_rows: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        required = {
            "artifact_version",
            "queue_fingerprint",
            "group_id",
            "canonical_paper_id",
            "capacity_role",
            "identity_status",
            "task_ids",
            "primary_task_ids",
            "tasks",
            "machine_selected_task_ids",
            "candidate_status",
            "human_review",
        }
        if not required.issubset(row):
            raise HandoffError(f"dossier row {row_number} is missing required fields")
        group_id = _sha256(row["group_id"], f"dossier row {row_number} group_id")
        if group_id in seen_groups:
            raise HandoffError(f"dossier row {row_number} repeats a group_id")
        seen_groups.add(group_id)
        current_id = _trimmed_string(
            row["canonical_paper_id"],
            f"dossier row {row_number} canonical_paper_id",
            allow_empty=True,
        )
        task_ids = _string_list(
            row["task_ids"], f"dossier row {row_number} task_ids", task_ids=True
        )
        primary_ids = _string_list(
            row["primary_task_ids"],
            f"dossier row {row_number} primary_task_ids",
            task_ids=True,
        )
        if not set(primary_ids).issubset(task_ids):
            raise HandoffError(f"dossier row {row_number} primary tasks are not group tasks")
        if row["queue_fingerprint"] != queue_fingerprint:
            raise HandoffError(f"dossier row {row_number} queue fingerprint differs from audit")
        if row["machine_selected_task_ids"] != []:
            raise HandoffError(f"dossier row {row_number} preselects machine tasks")
        if row["human_review"] != {"curator_ids": [], "independent_attestation": False}:
            raise HandoffError(f"dossier row {row_number} contains human state")
        if row["candidate_status"] != "blocked_pending_enrichment_and_human_attestation":
            raise HandoffError(f"dossier row {row_number} is not blocked")
        task_summaries, quarantined = _task_summaries(
            row["tasks"], task_ids, primary_ids, row_number
        )
        for summary in task_summaries:
            if summary["task_id"] in seen_task_ids:
                raise HandoffError(f"dossier row {row_number} repeats a task across groups")
            if summary["source_row_index"] in seen_source_rows:
                raise HandoffError(f"dossier row {row_number} repeats a source row")
            seen_task_ids.add(summary["task_id"])
            seen_source_rows.add(summary["source_row_index"])
        result.append(
            {
                "row": row_number,
                "group_id": group_id,
                "canonical_paper_id": current_id,
                "capacity_role": _trimmed_string(
                    row["capacity_role"], f"dossier row {row_number} capacity_role"
                ),
                "identity_status": _trimmed_string(
                    row["identity_status"], f"dossier row {row_number} identity_status"
                ),
                "task_ids": task_ids,
                "primary_task_ids": primary_ids,
                "task_summaries": task_summaries,
                "quarantined_legacy_sha256s": sorted(quarantined),
            }
        )
    return result


def _validate_source_probes(
    value: Any,
    dossiers: Sequence[Mapping[str, Any]],
    source_rows: set[int],
) -> dict[int, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != PARTITION_COUNTS["source_ready_lower_bound"]:
        raise HandoffError("source audit must contain exactly 48 source probes")
    result: dict[int, dict[str, Any]] = {}
    for index, probe in enumerate(value):
        if not isinstance(probe, dict):
            raise HandoffError(f"source probe {index} must be an object")
        _exact_keys(probe, SOURCE_PROBE_FIELDS, f"source probe {index}")
        row = _row_number(probe["row"], f"source probe {index} row")
        if row in result:
            raise HandoffError(f"source probe row {row} is duplicated")
        dossier = dossiers[row - 1]
        if (
            row not in source_rows
            or probe["group_id"] != dossier["group_id"]
            or probe["dossier_canonical_paper_id"] != dossier["canonical_paper_id"]
        ):
            raise HandoffError(f"source probe row {row} does not bind its dossier")
        _canonical_paper_id(probe["verified_canonical_paper_id"], f"source probe row {row} ID")
        _safe_url(probe["metadata_url"], f"source probe row {row} metadata_url")
        _safe_url(probe["direct_image_url"], f"source probe row {row} direct_image_url")
        _sha256(probe["direct_image_sha256"], f"source probe row {row} image hash")
        _safe_url(probe["license_url"], f"source probe row {row} license_url")
        _trimmed_string(probe["license"], f"source probe row {row} license")
        locator = probe["citation_locator"]
        if locator is not None:
            _trimmed_string(locator, f"source probe row {row} citation_locator")
        if (
            probe["verification_status"] != "machine_only"
            or probe["human_verified"] is not False
            or probe["publication_ready"] is not False
        ):
            raise HandoffError(f"source probe row {row} makes an unsafe verification claim")
        result[row] = dict(probe)
    if set(result) != source_rows:
        raise HandoffError("source probes do not exactly match the source-ready partition")
    corrections = {
        row
        for row, probe in result.items()
        if probe["dossier_canonical_paper_id"] != probe["verified_canonical_paper_id"]
    }
    if corrections != CORRECTION_ROWS:
        raise HandoffError("source probes must contain exactly the four known identity corrections")
    return result


def _validate_identity_exceptions(
    value: Any, dossiers: Sequence[Mapping[str, Any]], probe_rows: set[int]
) -> dict[int, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(IDENTITY_EXCEPTION_ROWS):
        raise HandoffError("source audit must contain exactly nine identity exceptions")
    result: dict[int, dict[str, Any]] = {}
    for index, exception in enumerate(value):
        if not isinstance(exception, dict):
            raise HandoffError(f"identity exception {index} must be an object")
        _exact_keys(exception, IDENTITY_EXCEPTION_FIELDS, f"identity exception {index}")
        row = _row_number(exception["row"], f"identity exception {index} row")
        if row in result or row in probe_rows:
            raise HandoffError(f"identity exception row {row} overlaps another audit record")
        dossier = dossiers[row - 1]
        if (
            exception["group_id"] != dossier["group_id"]
            or exception["current_canonical_paper_id"] != dossier["canonical_paper_id"]
        ):
            raise HandoffError(f"identity exception row {row} does not bind its dossier")
        candidate = exception["candidate_canonical_paper_id"]
        if candidate is not None:
            _canonical_paper_id(candidate, f"identity exception row {row} candidate ID")
        _trimmed_string(exception["disposition"], f"identity exception row {row} disposition")
        _trimmed_string(exception["reason"], f"identity exception row {row} reason")
        if (
            exception["verification_status"] != "machine_only"
            or exception["human_verified"] is not False
            or exception["publication_ready"] is not False
        ):
            raise HandoffError(f"identity exception row {row} makes an unsafe claim")
        result[row] = dict(exception)
    if set(result) != IDENTITY_EXCEPTION_ROWS:
        raise HandoffError("identity exception rows differ from the audited exception set")
    return result


def _validated_inputs(
    dossiers_path: str | Path, source_audit_path: str | Path
) -> tuple[
    Path,
    bytes,
    Path,
    bytes,
    list[dict[str, Any]],
    dict[str, list[int]],
    dict[int, str],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    str,
]:
    audit_path, audit, audit_bytes = _strict_json(source_audit_path, "source audit")
    if any(byte > 127 for byte in audit_bytes):
        raise HandoffError("source audit must be ASCII-only")
    _exact_keys(
        audit,
        frozenset(
            {
                "artifact_version",
                "queue_fingerprint",
                "partition",
                "source_probes",
                "identity_exceptions",
                "policy",
            }
        ),
        "source audit",
    )
    if audit["artifact_version"] != ARTIFACT_VERSION:
        raise HandoffError("source audit artifact_version must be 1")
    queue_fingerprint = audit["queue_fingerprint"]
    if not isinstance(queue_fingerprint, str) or not QUEUE_FINGERPRINT_RE.fullmatch(
        queue_fingerprint
    ):
        raise HandoffError("source audit queue_fingerprint must be a lowercase SHA256")
    if audit["policy"] != AUDIT_POLICY:
        raise HandoffError("source audit policy is not fail-closed")
    partition, row_category = _validate_partition(audit["partition"])
    dossier_path, raw_dossiers, dossier_bytes = _strict_jsonl(dossiers_path, "dossiers")
    dossiers = _validate_dossiers(raw_dossiers, queue_fingerprint)
    probes = _validate_source_probes(
        audit["source_probes"], dossiers, set(partition["source_ready_lower_bound"])
    )
    exceptions = _validate_identity_exceptions(audit["identity_exceptions"], dossiers, set(probes))
    return (
        dossier_path,
        dossier_bytes,
        audit_path,
        audit_bytes,
        dossiers,
        partition,
        row_category,
        probes,
        exceptions,
        queue_fingerprint,
    )


def _identity_routes(
    dossiers: Sequence[Mapping[str, Any]],
    probes: Mapping[int, Mapping[str, Any]],
    exceptions: Mapping[int, Mapping[str, Any]],
) -> dict[int, str]:
    routes = {row: "identity_exception" for row in exceptions}
    routes.update(
        {
            row: "identity_correction_probe"
            for row, probe in probes.items()
            if probe["dossier_canonical_paper_id"] != probe["verified_canonical_paper_id"]
        }
    )
    canonical_id_rows: dict[str, list[int]] = {}
    for dossier in dossiers:
        row = dossier["row"]
        current_id = dossier["canonical_paper_id"]
        if not current_id:
            routes.setdefault(row, "unresolved_empty_dossier_id")
        elif not _is_canonical_paper_id(current_id):
            routes.setdefault(row, "noncanonical_dossier_id")
        else:
            canonical_id_rows.setdefault(current_id, []).append(row)
    for duplicate_rows in canonical_id_rows.values():
        if len(duplicate_rows) > 1:
            for row in duplicate_rows:
                routes.setdefault(row, "duplicate_dossier_id")
    return dict(sorted(routes.items()))


def _status(
    row: int,
    category: str,
    probe: Mapping[str, Any] | None,
    exception: Mapping[str, Any] | None,
) -> str:
    if exception is not None:
        return "blocked_identity_exception"
    if probe is not None:
        if probe["dossier_canonical_paper_id"] != probe["verified_canonical_paper_id"]:
            return "identity_correction_requires_regeneration"
        return "machine_source_probe_pending_enrichment"
    return {
        "metadata_only_or_unlicensed": "blocked_metadata_or_license_only",
        "unresolved_identity": "blocked_unresolved_identity",
        "duplicate_candidate": "blocked_duplicate_candidate",
        "no_retrieval_hint": "blocked_no_retrieval_hint",
    }[category]


def _missing_inputs(status: str, requires_identity_resolution: bool) -> list[str]:
    missing = []
    if status == "blocked_no_retrieval_hint":
        missing.append("retrieval_hint")
    if status in {
        "blocked_metadata_or_license_only",
        "blocked_unresolved_identity",
        "blocked_duplicate_candidate",
        "blocked_no_retrieval_hint",
        "blocked_identity_exception",
    }:
        missing.append("complete_source_probe")
    if requires_identity_resolution:
        missing.extend(["identity_resolution", "capacity_plan_regeneration"])
    missing.extend(
        [
            "selected_task_ids",
            "benchmark_rows",
            "provenance_rows",
            "article_content_sha256",
            "license_content_sha256",
            "retrieved_at",
            "local_image_asset",
            "scientific_prompt_grounding",
            "training_lineage_evidence",
            "human_curator_ids",
            "independent_human_attestation",
            "full_enrichment_validation",
        ]
    )
    return missing


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    handle = io.StringIO(newline="")
    output = csv.writer(handle, lineterminator="\n")
    output.writerow(header)
    output.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _requirements(
    dossiers: Sequence[Mapping[str, Any]],
    row_category: Mapping[int, str],
    probes: Mapping[int, Mapping[str, Any]],
    exceptions: Mapping[int, Mapping[str, Any]],
    identity_routes: Mapping[int, str],
    queue_fingerprint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dossier in dossiers:
        row = dossier["row"]
        probe = probes.get(row)
        exception = exceptions.get(row)
        status = _status(row, row_category[row], probe, exception)
        requires_identity_resolution = row in identity_routes
        rows.append(
            {
                "artifact_version": ARTIFACT_VERSION,
                "queue_fingerprint": queue_fingerprint,
                "row": row,
                "group_id": dossier["group_id"],
                "dossier_canonical_paper_id": dossier["canonical_paper_id"],
                "capacity_role": dossier["capacity_role"],
                "identity_status": dossier["identity_status"],
                "task_ids": dossier["task_ids"],
                "primary_task_ids": dossier["primary_task_ids"],
                "task_summaries": dossier["task_summaries"],
                "partition_category": row_category[row],
                "status": status,
                "requires_identity_resolution": requires_identity_resolution,
                "source_probe": probe,
                "identity_exception": exception,
                "missing_inputs": _missing_inputs(status, requires_identity_resolution),
                "machine_only": True,
                "human_verified": False,
                "human_review_required": True,
                "automatic_retain": False,
                "publication_ready": False,
            }
        )
    return rows


def _identity_actions(
    dossiers: Sequence[Mapping[str, Any]],
    probes: Mapping[int, Mapping[str, Any]],
    exceptions: Mapping[int, Mapping[str, Any]],
    row_category: Mapping[int, str],
    identity_routes: Mapping[int, str],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    dossiers_by_row = {dossier["row"]: dossier for dossier in dossiers}
    for row, action_type in identity_routes.items():
        if action_type == "identity_exception":
            exception = exceptions[row]
            actions.append(
                {
                    "row": row,
                    "group_id": exception["group_id"],
                    "action_type": "identity_exception",
                    "current_canonical_paper_id": exception["current_canonical_paper_id"],
                    "machine_candidate_canonical_paper_id": exception[
                        "candidate_canonical_paper_id"
                    ],
                    "partition_category": row_category[row],
                    "machine_status": exception["disposition"],
                    "machine_reason": exception["reason"],
                    "machine_only": True,
                }
            )
        elif action_type == "identity_correction_probe":
            probe = probes[row]
            actions.append(
                {
                    "row": row,
                    "group_id": probe["group_id"],
                    "action_type": "identity_correction_probe",
                    "current_canonical_paper_id": probe["dossier_canonical_paper_id"],
                    "machine_candidate_canonical_paper_id": probe["verified_canonical_paper_id"],
                    "partition_category": row_category[row],
                    "machine_status": "machine_correction_requires_resolution",
                    "machine_reason": (
                        "Machine source probing found a different canonical paper ID; "
                        "the proposal remains unverified and requires capacity-plan regeneration."
                    ),
                    "machine_only": True,
                }
            )
        else:
            dossier = dossiers_by_row[row]
            dynamic_details = {
                "unresolved_empty_dossier_id": (
                    "The validated dossier canonical paper ID is empty and no "
                    "machine candidate is available; explicit human identity "
                    "resolution is required before source submission."
                ),
                "noncanonical_dossier_id": (
                    "The validated dossier paper ID does not satisfy the lowercase "
                    "DOI/arXiv/paper canonical-ID contract; explicit human identity "
                    "resolution is required before source submission."
                ),
                "duplicate_dossier_id": (
                    "The validated canonical dossier paper ID is shared by multiple "
                    "dossier groups; all affected rows require explicit human identity "
                    "resolution so at most one group retains this ID."
                ),
            }
            if action_type not in dynamic_details:
                raise HandoffError(f"unknown dynamic identity action type: {action_type}")
            actions.append(
                {
                    "row": row,
                    "group_id": dossier["group_id"],
                    "action_type": action_type,
                    "current_canonical_paper_id": dossier["canonical_paper_id"],
                    "machine_candidate_canonical_paper_id": None,
                    "partition_category": row_category[row],
                    "machine_status": action_type,
                    "machine_reason": dynamic_details[action_type],
                    "machine_only": True,
                }
            )
    return actions


def _source_template(
    requirements: Sequence[Mapping[str, Any]],
) -> bytes:
    header = [
        "row",
        "group_id",
        "dossier_canonical_paper_id",
        "partition_category",
        "status",
        "task_ids",
        "primary_task_ids",
        "selected_task_ids",
        "machine_candidate_canonical_paper_id",
        "metadata_url",
        "direct_image_url",
        "direct_image_sha256",
        "image_license",
        "license_url",
        "citation_locator",
        "article_content_sha256",
        "license_content_sha256",
        "retrieved_at",
        "local_image_path",
        "local_image_sha256",
        "scientific_prompt_grounding",
        "training_lineage_evidence",
        "notes",
    ]
    rows: list[list[Any]] = []
    for requirement in requirements:
        probe = requirement["source_probe"] or {}
        exception = requirement["identity_exception"] or {}
        candidate_id = probe.get("verified_canonical_paper_id") or exception.get(
            "candidate_canonical_paper_id"
        )
        rows.append(
            [
                requirement["row"],
                requirement["group_id"],
                requirement["dossier_canonical_paper_id"],
                requirement["partition_category"],
                requirement["status"],
                ";".join(requirement["task_ids"]),
                ";".join(requirement["primary_task_ids"]),
                "",
                candidate_id or "",
                probe.get("metadata_url", ""),
                probe.get("direct_image_url", ""),
                probe.get("direct_image_sha256", ""),
                probe.get("license", ""),
                probe.get("license_url", ""),
                probe.get("citation_locator") or "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    return _csv_bytes(header, rows)


def _identity_template(actions: Sequence[Mapping[str, Any]]) -> bytes:
    header = [
        "row",
        "group_id",
        "action_type",
        "current_canonical_paper_id",
        "machine_candidate_canonical_paper_id",
        "partition_category",
        "status",
        "machine_reason",
        "resolution_type",
        "resolved_canonical_paper_id",
        "resolution_evidence_url",
        "resolution_evidence_sha256",
        "resolution_evidence_retrieved_at",
        "capacity_plan_regenerated",
        "resolution_notes",
    ]
    rows = [
        [
            action["row"],
            action["group_id"],
            action["action_type"],
            action["current_canonical_paper_id"],
            action["machine_candidate_canonical_paper_id"] or "",
            action["partition_category"],
            action["machine_status"],
            action["machine_reason"],
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
        for action in actions
    ]
    return _csv_bytes(header, rows)


def _readme(
    action_count: int,
    empty_dossier_action_count: int,
    noncanonical_dossier_action_count: int,
    duplicate_dossier_action_count: int,
) -> bytes:
    text = f"""# Безопасная передача enrichment-данных

Этот workspace НЕ импортируется в curator и не является результатом human review.
Прямые image URL и SHA256 проверены машинным сетевым probe. Лицензия, identity и научное
содержание не подтверждены человеком, поэтому пакет не пригоден для публикации.

## Порядок работы владельца данных

1. Локально откройте `training_lineage_form.html`, заполните источники и coverage, затем
   экспортируйте финальный `training_lineage_manifest.json`.
2. Откройте `identity_resolution_form.html`, документируйте все {action_count} identity actions
   (9 audit exceptions, 4 audit corrections, {empty_dossier_action_count} validated dossiers с
   пустым canonical ID, {noncanonical_dossier_action_count} с неканоническим ID и
   {duplicate_dossier_action_count} с дублирующимся canonical ID), затем экспортируйте
   `identity_resolution_submission.json`. Машинные кандидаты являются только непроверенными
   подсказками. Строки 26, 52, 78 и 114 входят в тот же список действий; динамические действия
   без машинного кандидата требуют явного разрешения identity.
3. Откройте `source_submission_form.html`, импортируйте оба финальных файла, заполните ровно 150
   предложений и экспортируйте `capacity_source_submission.json`.
4. Передайте три финальных файла отдельному downstream validator. Ни одна HTML-форма не выполняет
   сетевых запросов и не подтверждает внешние источники.

Каждая форма сохраняет локальный черновик в браузере. Draft export создает отдельный неполный
envelope, который импортируется обратно только в форму того же вида, версии и queue fingerprint.
Final export работает fail-closed и показывает ошибки. Это submissions владельца данных, а не
решения кураторов; пакет не содержит curator IDs, verified_by, независимых attestations или
retain/exclude decisions.

CSV-файлы оставлены как вспомогательные пустые шаблоны. Сверьте
`enrichment_requirements.jsonl` и `handoff_manifest.json`; все строки остаются blocked до полного
downstream validation.

`machine_assisted_draft.json` можно создавать только отдельным валидатором после полного
enrichment validation, регенерации затронутого плана и проверки human evidence. Не переименовывайте
этот workspace в curator workspace и не создавайте здесь `machine_enrichment_v2.jsonl` или
`machine_assisted_draft.json`.
"""
    return text.encode("utf-8")


_FORM_CSS = r"""
:root{color-scheme:light;--ink:#17211b;--muted:#59645d;--paper:#f7f3e8;--card:#fffdf7;--line:#c8c1b0;--accent:#0b6b53;--accent2:#d9eee5;--danger:#9a2d2d;--warn:#7b5910;font:16px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#e8eee7,var(--paper) 42%);color:var(--ink)}header{padding:1.2rem clamp(1rem,4vw,3rem);background:#17372e;color:#fff;border-bottom:5px solid #d5a84f}header h1{margin:0 0 .3rem;font:700 clamp(1.5rem,4vw,2.5rem)/1.1 Georgia,serif}header p{max-width:80rem;margin:.3rem 0;color:#e0eee8}.shell{max-width:1500px;margin:auto;padding:1rem}.notice,.card,.errors{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 5px 18px #17211b12;padding:1rem;margin-bottom:1rem}.notice{border-left:6px solid #d5a84f}.layout{display:grid;grid-template-columns:minmax(245px,320px) 1fr;gap:1rem;align-items:start}.sidebar{position:sticky;top:.5rem;max-height:calc(100vh - 1rem);overflow:auto}.toolbar,.row{display:flex;gap:.65rem;align-items:center;flex-wrap:wrap}.toolbar{margin:.7rem 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem}.wide{grid-column:1/-1}label{display:grid;gap:.3rem;font-weight:650}label small,.help,.meta{font-weight:400;color:var(--muted);font-size:.87rem}input,select,textarea,button{font:inherit}input,select,textarea{width:100%;padding:.62rem;border:1px solid #8d958e;border-radius:7px;background:#fff;color:var(--ink)}textarea{min-height:6rem;resize:vertical}input:focus-visible,select:focus-visible,textarea:focus-visible,button:focus-visible{outline:3px solid #e3b34e;outline-offset:2px}button{border:1px solid #0a5744;border-radius:8px;padding:.58rem .8rem;background:var(--accent);color:#fff;font-weight:700;cursor:pointer}button.secondary{background:#fff;color:var(--accent)}button.danger{background:#fff;color:var(--danger);border-color:var(--danger)}button:disabled{opacity:.5;cursor:not-allowed}.task,.image,.source{border:1px solid var(--line);border-radius:9px;padding:.8rem;margin:.7rem 0;background:#fff}.machine{border-left:5px solid #c78d21;background:#fff8e7}.machine strong{color:var(--warn)}.codes{display:flex;gap:.35rem;flex-wrap:wrap}.code{background:#ece8dc;border-radius:999px;padding:.15rem .5rem;font:12px ui-monospace,monospace}.nav-list{display:grid;gap:.3rem;margin-top:.6rem}.nav-list button{display:block;text-align:left;background:#fff;color:var(--ink);border-color:var(--line);font-weight:500}.nav-list button.current{background:var(--accent2);border-color:var(--accent);font-weight:750}.nav-list button.incomplete::after{content:" • incomplete";color:var(--danger)}progress{width:100%;height:1.1rem}.errors{border:2px solid var(--danger);color:var(--danger)}.errors button{background:none;color:var(--danger);border:0;padding:.15rem;text-align:left}.errors ul{margin:.5rem 0}.status{min-height:1.4rem;color:var(--muted)}.ok{color:var(--accent)}.bad{color:var(--danger)}img.preview{display:block;max-width:min(100%,480px);max-height:280px;object-fit:contain;margin:.5rem 0;border:1px solid var(--line)}.hidden,[hidden]{display:none!important}.check{display:flex;align-items:flex-start;gap:.5rem;font-weight:500}.check input{width:auto;margin-top:.25rem}.actions{display:flex;gap:.5rem;flex-wrap:wrap;padding:1rem 0;border-top:1px solid var(--line);margin-top:1rem}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:800px){.layout,.grid{grid-template-columns:1fr}.sidebar{position:static;max-height:none}.nav-list{grid-template-columns:repeat(3,1fr)}.nav-list button{font-size:.8rem}.shell{padding:.65rem}}
html,body{min-width:0;max-width:100%}
.shell{width:100%;min-width:0}
.layout,main,aside,.notice,.card,.errors,.grid,label,.toolbar,.row,.actions,.task,.image,.source{min-width:0;max-width:100%}
.layout>*,.grid>*,.toolbar>*,.row>*,.actions>*,.task>*,.image>*,.source>*{min-width:0;max-width:100%}
input,select,textarea,button,progress{min-width:0;max-width:100%}
input[type="file"]{width:100%}
input[type="file"]::file-selector-button{max-width:100%;white-space:normal}
h1,h2,h3,p,li,label,small,strong,.help,.meta,.mono,.nav-list button{overflow-wrap:anywhere}
.mono,.nav-list button,.task strong{word-break:break-word}
button{white-space:normal}
@media(max-width:800px){
header{padding:.85rem .75rem}
.shell{padding:.65rem}
.notice,.card,.errors{padding:.75rem;margin-bottom:.75rem}
.task,.image,.source{padding:.65rem}
.layout,.grid{grid-template-columns:minmax(0,1fr)}
.layout{gap:.75rem}
.sidebar{position:static;top:auto;max-height:min(55vh,28rem);overflow-y:auto}
.nav-list{display:grid;grid-template-columns:minmax(0,1fr)}
.actions,.toolbar,.row{align-items:stretch}
.actions{display:flex}
.actions,.row{flex-direction:column}
.actions>*,.toolbar>*,.row>*{width:100%}
label.check{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:start}
label.check>input{width:auto}
}
"""


def _embedded_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _html_document(title: str, body: str, data: Mapping[str, Any], script: str) -> bytes:
    csp = (
        "default-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; "
        "form-action 'none'; img-src blob:; "
        f"style-src 'nonce-{HTML_NONCE}'; script-src 'nonce-{HTML_NONCE}'"
    )
    document = (
        '<!doctype html>\n<html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        f'<title>{title}</title><style nonce="{HTML_NONCE}">{_FORM_CSS}</style>'
        f"</head><body>{body}"
        f'<script nonce="{HTML_NONCE}" type="application/json" id="form-data">'
        f"{_embedded_json(data)}</script>"
        f'<script nonce="{HTML_NONCE}">{_FORM_SHARED_JS}\n{script}</script>'
        "</body></html>\n"
    )
    return document.encode("utf-8")


def _form_requirement(requirement: Mapping[str, Any]) -> dict[str, Any]:
    probe = requirement["source_probe"]
    exception = requirement["identity_exception"]
    machine_candidate: dict[str, Any] | None = None
    if probe is not None:
        machine_candidate = {
            "canonical_paper_id": probe["verified_canonical_paper_id"],
            "article_url": probe["metadata_url"],
            "image_url": probe["direct_image_url"],
            "image_sha256": probe["direct_image_sha256"],
            "license": probe["license"],
            "license_url": probe["license_url"],
            "citation_locator": probe["citation_locator"],
            "machine_status": probe["verification_status"],
        }
    elif exception is not None:
        machine_candidate = {
            "canonical_paper_id": exception["candidate_canonical_paper_id"],
            "machine_status": exception["disposition"],
            "machine_reason": exception["reason"],
        }
    machine_prefilled_fields: list[str] = []
    if machine_candidate is not None and "article_url" in machine_candidate:
        machine_prefilled_fields = [
            *([] if requirement["requires_identity_resolution"] else ["canonical_paper_id"]),
            "images[0].sha256",
            "images[0].locator",
            "images[0].source_url",
            "images[0].license",
            "images[0].license_url",
            "article_evidence.source_url",
        ]
    return {
        "artifact_version": requirement["artifact_version"],
        "queue_fingerprint": requirement["queue_fingerprint"],
        "row": requirement["row"],
        "group_id": requirement["group_id"],
        "dossier_canonical_paper_id": requirement["dossier_canonical_paper_id"],
        "capacity_role": requirement["capacity_role"],
        "identity_status": requirement["identity_status"],
        "task_ids": requirement["task_ids"],
        "primary_task_ids": requirement["primary_task_ids"],
        "task_summaries": [
            summary for summary in requirement["task_summaries"] if summary["is_primary"]
        ],
        "partition_category": requirement["partition_category"],
        "status": requirement["status"],
        "requires_identity_resolution": requirement["requires_identity_resolution"],
        "machine_candidate": machine_candidate,
        "machine_prefilled_fields": machine_prefilled_fields,
        "required_inputs": [
            item
            for item in requirement["missing_inputs"]
            if item
            not in {
                "human_curator_ids",
                "independent_human_attestation",
            }
        ],
        "machine_only": True,
        "publication_ready": False,
    }


def _blank_source_submission(requirement: Mapping[str, Any]) -> dict[str, Any]:
    candidate = requirement["machine_candidate"]
    probe = candidate if candidate is not None and "article_url" in candidate else {}
    return {
        "row": requirement["row"],
        "group_id": requirement["group_id"],
        "selected_task_ids": [],
        "sample_id": "",
        "canonical_paper_id": (
            ""
            if requirement["requires_identity_resolution"]
            else probe.get("canonical_paper_id", "")
        ),
        "stratum": "",
        "model_task_prompt": "",
        "system_instruction": "",
        "source_document_id": "",
        "creator_group_id": "",
        "scientific_prompt_grounding": "",
        "notes": "",
        "images": [
            {
                "image_path": "",
                "sha256": probe.get("image_sha256", ""),
                "page": "",
                "locator": probe.get("citation_locator") or "",
                "source_url": probe.get("image_url", ""),
                "license": probe.get("license", ""),
                "license_url": probe.get("license_url", ""),
                "citation": "",
                "retrieved_at": "",
            }
        ],
        "article_evidence": {
            "source_url": probe.get("article_url", ""),
            "sha256": "",
            "retrieved_at": "",
        },
        "metadata_evidence": {"source_url": "", "sha256": "", "retrieved_at": ""},
    }


def _source_form(
    requirements: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
    quarantined_hashes: Sequence[str],
) -> bytes:
    groups = [_form_requirement(requirement) for requirement in requirements]
    policy = {
        "data_owner_proposal_only": True,
        "curator_importable": False,
        "automatic_selection": False,
        "external_verification_claimed": False,
        "human_review_included": False,
        "independent_attestation_included": False,
        "retain_exclude_decisions_included": False,
        "full_downstream_validation_required": True,
        "publication_ready": False,
    }
    data = {
        "form_kind": "capacity_source_submission_form",
        "artifact_version": ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "groups": groups,
        "identity_actions": list(actions),
        "quarantined_legacy_sha256s": sorted(quarantined_hashes),
        "initial_submissions": [_blank_source_submission(group) for group in groups],
        "safe_submission_policy": policy,
        "export_contract": {
            "kind": "capacity_source_submission",
            "artifact_version": ARTIFACT_VERSION,
            "ordered_submission_count": TOTAL_ROWS,
            "top_level_fields": [
                "kind",
                "artifact_version",
                "queue_fingerprint",
                "submissions",
                "identity_resolution_binding",
                "training_lineage_manifest_binding",
                "policy",
                "publication_ready",
            ],
            "submission_fields": list(_blank_source_submission(groups[0])),
            "image_fields": list(_blank_source_submission(groups[0])["images"][0]),
            "evidence_fields": ["source_url", "sha256", "retrieved_at"],
        },
    }
    return _html_document("Передача исходных данных", _SOURCE_BODY, data, _SOURCE_JS)


def _blank_identity_resolution(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row": action["row"],
        "group_id": action["group_id"],
        "resolution_type": "",
        "resolved_canonical_paper_id": "",
        "evidence": {
            "source_url": "",
            "sha256": "",
            "retrieved_at": "",
            "notes": "",
        },
        "capacity_plan_regenerated": False,
    }


def _identity_form(
    actions: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
    occupied_paper_ids: Sequence[str],
) -> bytes:
    policy = {
        "data_owner_proposal_only": True,
        "curator_importable": False,
        "automatic_selection": False,
        "external_verification_claimed": False,
        "human_review_included": False,
        "independent_attestation_included": False,
        "retain_exclude_decisions_included": False,
        "full_downstream_validation_required": True,
        "publication_ready": False,
    }
    first = _blank_identity_resolution(actions[0])
    data = {
        "form_kind": "capacity_identity_resolution_form",
        "artifact_version": ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "actions": list(actions),
        "occupied_non_action_paper_ids": list(occupied_paper_ids),
        "initial_resolutions": [_blank_identity_resolution(action) for action in actions],
        "safe_submission_policy": policy,
        "export_contract": {
            "kind": "capacity_identity_resolution_submission",
            "artifact_version": ARTIFACT_VERSION,
            "action_count": len(actions),
            "top_level_fields": [
                "kind",
                "artifact_version",
                "queue_fingerprint",
                "actions",
                "policy",
                "publication_ready",
            ],
            "resolution_fields": list(first),
            "evidence_fields": list(first["evidence"]),
            "resolution_types": [
                "confirm_current",
                "replace_with_candidate",
                "replace_with_other",
                "unresolved",
            ],
        },
    }
    body = _IDENTITY_BODY.replace("__ACTION_COUNT__", str(len(actions)))
    return _html_document("Разрешение идентичности статей", body, data, _IDENTITY_JS)


# The hardened browser logic is line-oriented so the fail-closed checks remain auditable.
_FORM_CSS += r"""
.machine-prefill{border-color:#b27a13;background:#fff9e8}.storage-warning{white-space:pre-wrap}
"""

_FORM_SHARED_JS = r"""
"use strict";
const BOOT=JSON.parse(document.getElementById("form-data").textContent);
const MAX_JSON_BYTES=32*1024*1024;
const $=id=>document.getElementById(id);
function el(tag,text,className){
  const value=document.createElement(tag);
  if(text!==undefined)value.textContent=text;
  if(className)value.className=className;
  return value;
}
function exactKeys(value,keys){
  return value&&typeof value==="object"&&!Array.isArray(value)&&
    Object.keys(value).sort().join("\0")===keys.slice().sort().join("\0");
}
function trimmed(value,optional=false){
  return typeof value==="string"&&value===value.trim()&&
    (optional||value.length>0)&&!/[\x00-\x1f\x7f]/.test(value);
}
function sha256(value){return typeof value==="string"&&/^[0-9a-f]{64}$/.test(value)}
function canonicalPaperId(value){
  return trimmed(value)&&/^[\x20-\x7e]+$/.test(value)&&value===value.toLowerCase()&&
    /^(?:doi:10\.\d{4,9}\/[^\s]+|arxiv:(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?\/\d{7})|paper:\S(?:[\x20-\x7e]*\S)?)$/.test(value);
}
function safeUrl(value){
  if(!trimmed(value)||/[^\x21-\x7e]/.test(value)||value.includes("\\")||
      !/^https?:\/\/[^/?#]+(?:\/[^?#]*)?$/.test(value))return false;
  try{
    const url=new URL(value);
    return ["http:","https:"].includes(url.protocol)&&!url.username&&!url.password&&
      !url.search&&!url.hash&&Boolean(url.hostname);
  }catch(_error){return false}
}
function immutableUrl(value){
  return safeUrl(value)&&/(?:^|\/)[0-9a-f]{40,64}(?:\/|$)/.test(new URL(value).pathname);
}
function safeAssetPath(value){
  if(typeof value!=="string"||!value.startsWith("assets/images/")||
      /[^\x20-\x7e]/.test(value)||value.includes("\\")||value.includes(":"))return false;
  const parts=value.split("/");
  if(parts.length<3||parts.some(part=>!part||part==="."||part===".."||
      part!==part.replace(/[ .]+$/, "")))return false;
  for(const part of parts){
    if(/[<>"|?*\x00-\x1f]/.test(part))return false;
    const base=part.split(".",1)[0].trimEnd()
      .replace(/[¹²³]/g,d=>({"¹":"1","²":"2","³":"3"})[d]).toLowerCase();
    if(/^(aux|clock\$|con|conin\$|conout\$|nul|prn|com[1-9]|lpt[1-9])$/.test(base))return false;
  }
  return true;
}
function windowsPathKey(value){return value.normalize("NFKC").toLowerCase()}
function collidingAssetPathRow(paths,pathKey){
  for(const [existingPath,row] of paths){
    if(existingPath===pathKey||existingPath.startsWith(`${pathKey}/`)||
        pathKey.startsWith(`${existingPath}/`))return row;
  }
  return null;
}
function strictJsonParse(text){
  if(typeof text!=="string")throw new SyntaxError("JSON должен быть текстом");
  let at=0;
  const fail=message=>{throw new SyntaxError(`${message}, позиция ${at}`)};
  const space=()=>{while(at<text.length&&/[\x20\t\r\n]/.test(text[at]))at+=1};
  function string(){
    if(text[at]!=="\"")fail("Ожидалась JSON-строка");
    const start=at++;
    while(at<text.length){
      const code=text.charCodeAt(at++);
      if(code===34)return JSON.parse(text.slice(start,at));
      if(code<32)fail("Управляющий символ в JSON-строке");
      if(code===92){
        if(at>=text.length)fail("Незавершенная escape-последовательность");
        const escape=text[at++];
        if(escape==="u"){
          if(!/^[0-9a-fA-F]{4}$/.test(text.slice(at,at+4)))fail("Некорректный Unicode escape");
          at+=4;
        }else if(!'"\\/bfnrt'.includes(escape))fail("Некорректный JSON escape");
      }
    }
    fail("Незавершенная JSON-строка");
  }
  function value(depth){
    if(depth>256)fail("Слишком глубокая вложенность JSON");
    space();
    const current=text[at];
    if(current==='"')return string();
    if(current==="{"){
      at+=1;space();
      const result={},seen=new Set();
      if(text[at]==="}"){at+=1;return result}
      while(true){
        space();const key=string();
        if(seen.has(key))fail(`Повторяющийся ключ JSON: ${key}`);
        seen.add(key);space();
        if(text[at++]!==":")fail("После ключа ожидалось двоеточие");
        Object.defineProperty(result,key,{value:value(depth+1),enumerable:true,writable:true,configurable:true});
        space();const delimiter=text[at++];
        if(delimiter==="}")return result;
        if(delimiter!==",")fail("В объекте ожидалась запятая или закрывающая скобка");
      }
    }
    if(current==="["){
      at+=1;space();const result=[];
      if(text[at]==="]"){at+=1;return result}
      while(true){
        result.push(value(depth+1));space();const delimiter=text[at++];
        if(delimiter==="]")return result;
        if(delimiter!==",")fail("В массиве ожидалась запятая или закрывающая скобка");
      }
    }
    for(const [literal,result] of [["true",true],["false",false],["null",null]]){
      if(text.startsWith(literal,at)){at+=literal.length;return result}
    }
    const match=text.slice(at).match(/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/);
    if(!match)fail("Ожидалось JSON-значение");
    at+=match[0].length;const result=Number(match[0]);
    if(!Number.isFinite(result))fail("Неконечное число запрещено");
    return result;
  }
  const result=value(0);space();
  if(at!==text.length)fail("Лишние данные после JSON");
  return result;
}
function clone(value){return strictJsonParse(JSON.stringify(value))}
async function jsonBytes(file){
  if(file.size>MAX_JSON_BYTES)throw new Error("размер JSON превышает 32 МиБ");
  const bytes=await file.arrayBuffer();
  if(bytes.byteLength>MAX_JSON_BYTES)throw new Error("размер JSON превышает 32 МиБ");
  return bytes;
}
function decodeJson(bytes){
  const text=new TextDecoder("utf-8",{fatal:true}).decode(bytes);
  return strictJsonParse(text);
}
async function readJson(file){return decodeJson(await jsonBytes(file))}
async function readJsonWithHash(file){
  const bytes=await jsonBytes(file);
  const value=decodeJson(bytes);
  const digest=await crypto.subtle.digest("SHA-256",bytes);
  const hash=Array.from(new Uint8Array(digest),byte=>byte.toString(16).padStart(2,"0")).join("");
  return {value,sha256:hash};
}
async function fileSha(file){
  const bytes=await file.arrayBuffer();
  const digest=await crypto.subtle.digest("SHA-256",bytes);
  return Array.from(new Uint8Array(digest),byte=>byte.toString(16).padStart(2,"0")).join("");
}
function downloadJson(name,value){
  const blob=new Blob([JSON.stringify(value,null,2)+"\n"],{type:"application/json"});
  const url=URL.createObjectURL(blob),link=document.createElement("a");
  link.href=url;link.download=name;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),0);
}
function setMessage(text,bad=false){
  const target=$("status");
  if(target){target.textContent=text;target.className=bad?"status bad":"status ok"}
}
function storageGet(key){
  try{return localStorage.getItem(key)}
  catch(_error){setMessage("localStorage недоступен: черновик не прочитан. Экспортируйте его вручную.",true);return null}
}
function storageSet(key,value){
  try{localStorage.setItem(key,value);return true}
  catch(_error){setMessage("localStorage недоступен: автосохранение не выполнено. Экспортируйте черновик вручную.",true);return false}
}
function storageRemove(key){
  try{localStorage.removeItem(key);return true}
  catch(_error){setMessage("localStorage недоступен: сохраненный черновик не удален.",true);return false}
}
const fileHasherRuntimes=new Set();
let pendingHashCount=0;
function revokeHasherPreview(runtime){
  if(runtime.url){URL.revokeObjectURL(runtime.url);runtime.url=null}
  runtime.preview.removeAttribute("src");runtime.preview.classList.add("hidden");
}
function invalidateHasher(runtime,remove=false){
  runtime.token+=1;
  if(runtime.pending){runtime.pending=false;pendingHashCount-=1}
  revokeHasherPreview(runtime);runtime.active=!remove;
}
function cleanupFileHashers(target=null){
  for(const runtime of Array.from(fileHasherRuntimes)){
    if(target!==null&&runtime.target!==target)continue;
    invalidateHasher(runtime,true);fileHasherRuntimes.delete(runtime);
  }
}
function cancelFieldHashers(target,key){
  for(const runtime of fileHasherRuntimes){
    if(runtime.target===target&&runtime.key===key){
      invalidateHasher(runtime,false);
      runtime.status.textContent="SHA256 введен вручную; выбранный файл сброшен.";
      runtime.input.value="";
    }
  }
}
function addField(parent,labelText,object,key,options={}){
  const label=el("label"),caption=el("span",labelText);
  label.append(caption);
  if(options.machineHint){label.classList.add("machine-prefill");caption.append(" (непроверенная машинная подсказка)")}
  const input=options.textarea?el("textarea"):el("input");
  if(options.type)input.type=options.type;
  if(options.placeholder)input.placeholder=options.placeholder;
  input.value=object[key]??"";
  if(options.readOnly)input.readOnly=true;
  input.addEventListener("input",()=>{
    cancelFieldHashers(object,key);object[key]=input.value;
    if(options.onInput)options.onInput(input);save();
  });
  label.append(input);
  if(options.help)label.append(el("small",options.help));
  parent.append(label);return input;
}
function addFileHasher(parent,target,key,labelText,accept="",linkedInput=null){
  const label=el("label",labelText),input=el("input");
  input.type="file";if(accept)input.accept=accept;
  const status=el("span","Локальный файл не выбран.","help");
  const preview=el("img",undefined,"preview hidden");preview.alt="Предпросмотр локального файла";
  const runtime={target,key,input,status,preview,token:0,pending:false,url:null,active:true};
  fileHasherRuntimes.add(runtime);
  input.addEventListener("change",async()=>{
    invalidateHasher(runtime,false);target[key]="";if(linkedInput)linkedInput.value="";status.className="help";save();
    const file=input.files&&input.files[0],token=runtime.token;
    if(!file){status.textContent="Локальный файл не выбран.";return}
    runtime.pending=true;pendingHashCount+=1;status.textContent="Вычисляется локальный SHA256…";
    try{
      const hash=await fileSha(file);
      if(!runtime.active||runtime.token!==token||!input.files||input.files[0]!==file)return;
      target[key]=hash;if(linkedInput)linkedInput.value=hash;status.textContent=`${file.name} (${file.size} байт): ${hash}`;
      if(file.type.startsWith("image/")){
        runtime.url=URL.createObjectURL(file);preview.src=runtime.url;preview.classList.remove("hidden");
      }
    }catch(error){
      if(runtime.active&&runtime.token===token){
        target[key]="";if(linkedInput)linkedInput.value="";status.textContent=`Не удалось вычислить SHA256: ${String(error)}`;
        status.className="bad";
      }
    }finally{
      if(runtime.active&&runtime.token===token&&runtime.pending){
        runtime.pending=false;pendingHashCount-=1;save();
      }
    }
  });
  label.append(input,status,preview);parent.append(label);return {input,status,preview};
}
function renderErrorSummary(errors,navigate){
  const box=$("errors");box.replaceChildren();
  if(!errors.length){box.hidden=true;return}
  box.hidden=false;box.append(el("h2",`Финальный экспорт заблокирован: ошибок ${errors.length}`));
  const list=el("ul");
  for(const error of errors){
    const item=el("li");
    if(error.row&&navigate){
      const button=el("button",`Группа ${error.row}: ${error.message}`);button.type="button";
      button.addEventListener("click",()=>navigate(error.row));item.append(button);
    }else item.textContent=error.message;
    list.append(item);
  }
  box.append(list);box.tabIndex=-1;box.focus();
}
function listValues(text){return String(text||"").split(/\r?\n/).map(value=>value.trim()).filter(Boolean)}
function manifestErrors(manifest){
  const errors=[],top=["schema_version","training_sources","coverage","paper_ids","source_document_ids","creator_group_ids","image_sha256s","prompt_sha256s"];
  if(!exactKeys(manifest,top))return ["манифест должен содержать только официальные поля верхнего уровня"];
  if(manifest.schema_version!==1)errors.push("schema_version должен быть равен 1");
  if(!Array.isArray(manifest.training_sources)||!manifest.training_sources.length)errors.push("нужен хотя бы один источник обучения");
  else manifest.training_sources.forEach((source,index)=>{
    if(!exactKeys(source,["repo_id","repo_type","revision","files"]))return errors.push(`источник обучения ${index+1}: неверные поля`);
    if(!trimmed(source.repo_id))errors.push(`источник ${index+1}: repo_id обязателен`);
    if(!["model","dataset"].includes(source.repo_type))errors.push(`источник ${index+1}: неверный repo_type`);
    if(typeof source.revision!=="string"||!/^[0-9a-f]{40}$/.test(source.revision))errors.push(`источник ${index+1}: revision должен быть lowercase 40-hex`);
    if(!Array.isArray(source.files)||!source.files.length)errors.push(`источник ${index+1}: нужен хотя бы один файл`);
    else source.files.forEach((file,fileIndex)=>{
      if(!exactKeys(file,["path","sha256","row_count"]))return errors.push(`источник ${index+1}, файл ${fileIndex+1}: неверные поля`);
      if(!trimmed(file.path))errors.push(`источник ${index+1}, файл ${fileIndex+1}: path обязателен`);
      if(!sha256(file.sha256))errors.push(`источник ${index+1}, файл ${fileIndex+1}: неверный SHA256`);
      if(!Number.isInteger(file.row_count)||file.row_count<0)errors.push(`источник ${index+1}, файл ${fileIndex+1}: row_count должен быть неотрицательным целым`);
    });
  });
  const coverage=["paper_ids","source_documents","creator_groups","image_bytes","prompts"];
  if(!exactKeys(manifest.coverage,coverage)||coverage.some(key=>manifest.coverage[key]!==true))errors.push("все пять признаков coverage должны быть явно подтверждены");
  for(const key of ["paper_ids","source_document_ids","creator_group_ids","image_sha256s","prompt_sha256s"]){
    const values=manifest[key];
    if(!Array.isArray(values)||!values.length){errors.push(`${key}: нужен непустой массив`);continue}
    if(values.some(value=>!trimmed(value)))errors.push(`${key}: значения должны быть непустыми строками без пробелов по краям`);
    if(new Set(values).size!==values.length)errors.push(`${key}: значения должны быть уникальными`);
    if(key==="paper_ids"&&values.some(value=>!canonicalPaperId(value)))errors.push("paper_ids: каждый ID должен быть каноническим lowercase DOI/arXiv/paper ID");
    if(key.endsWith("sha256s")&&values.some(value=>!sha256(value)))errors.push(`${key}: каждый SHA256 должен быть lowercase 64-hex`);
  }
  return errors;
}
"""

_SOURCE_BODY = r"""
<header><h1>Передача исходных данных capacity-150</h1><p>Офлайн-предложение владельца данных для 150 групп. Машинные подсказки не являются проверкой или решением.</p></header>
<div class="shell"><section class="notice"><strong>Только локальная работа.</strong> Форма не выполняет сетевых запросов. Импортированные identity и lineage файлы и их SHA256 хранятся только в памяти вкладки: после перезагрузки или импорта черновика выберите их заново. Финальный файл не готов для curator или публикации.</section>
<section class="card" aria-labelledby="imports-title"><h2 id="imports-title">Обязательные глобальные привязки</h2><div class="grid"><label>Финальный training_lineage_manifest.json<input id="lineageFile" type="file" accept="application/json,.json"><small id="lineageStatus">Файл не импортирован.</small></label><label>Неизменяемый URL манифеста<input id="lineageUrl" type="url" placeholder="https://host/repo/raw/40-or-64-hex-revision/training_lineage_manifest.json"><small>HTTP(S), без credentials/query/fragment; путь содержит 40–64 hex revision.</small></label><label>Время получения манифеста (retrieved_at)<input id="lineageRetrieved" placeholder="2026-07-22T12:00:00Z"></label><label>Финальный identity_resolution_submission.json<input id="identityFile" type="file" accept="application/json,.json"><small id="identityStatus">Файл не импортирован.</small></label></div></section>
<div id="errors" class="errors" role="alert" hidden></div><div class="layout"><aside class="card sidebar"><label>Поиск групп<input id="search" type="search" placeholder="строка, ID, группа"></label><label>Фильтр<select id="filter"><option value="all">Все группы</option><option value="incomplete">Незаполненные</option><option value="identity">Identity actions</option><option value="probe">Машинный source probe</option></select></label><p id="progressText" class="meta"></p><progress id="progress" max="150" value="0">0/150</progress><div class="toolbar"><button id="firstIncomplete" type="button">Первая незаполненная</button></div><nav id="groupNav" class="nav-list" aria-label="Группы источников"></nav></aside><main><section id="groupForm" class="card" aria-live="polite"></section></main></div>
<section class="card"><h2>Черновик и финальный файл</h2><div class="actions"><button id="draftExport" type="button" class="secondary">Экспортировать черновик JSON</button><label>Импортировать черновик этой формы<input id="draftImport" type="file" accept="application/json,.json"></label><button id="clear" type="button" class="danger">Очистить локальный черновик</button><button id="finalExport" type="button">Проверить и экспортировать финальный JSON</button></div><p id="status" class="status storage-warning" aria-live="polite"></p></section></div>
"""

_SOURCE_JS = r"""
const KEY=`capacity-handoff:${BOOT.queue_fingerprint}:${BOOT.form_kind}`;
let currentRow=1;
let state={submissions:clone(BOOT.initial_submissions),lineage_reference:{source_url:"",retrieved_at:""}};
let bindings={identity:null,lineage:null};
let identityImportToken=0,lineageImportToken=0,draftImportToken=0,stateRevision=0;
const requirementByRow=new Map(BOOT.groups.map(group=>[group.row,group]));
const actionByRow=new Map(BOOT.identity_actions.map(action=>[action.row,action]));
const forbiddenLegacyHashes=new Set(BOOT.quarantined_legacy_sha256s);
function draftEnvelope(){
  return {kind:"capacity_source_submission_draft",artifact_version:1,queue_fingerprint:BOOT.queue_fingerprint,draft:clone(state)};
}
function validDraft(value){
  if(!exactKeys(value,["kind","artifact_version","queue_fingerprint","draft"])||
      value.kind!=="capacity_source_submission_draft"||value.artifact_version!==1||
      value.queue_fingerprint!==BOOT.queue_fingerprint)return false;
  const draft=value.draft;
  if(!exactKeys(draft,["submissions","lineage_reference"])||!Array.isArray(draft.submissions)||
      draft.submissions.length!==150||!exactKeys(draft.lineage_reference,["source_url","retrieved_at"])||
      Object.values(draft.lineage_reference).some(item=>typeof item!=="string"))return false;
  return draft.submissions.every((submission,index)=>{
    if(!exactKeys(submission,BOOT.export_contract.submission_fields)||
        submission.row!==BOOT.groups[index].row||submission.group_id!==BOOT.groups[index].group_id||
        !Array.isArray(submission.selected_task_ids)||submission.selected_task_ids.some(item=>typeof item!=="string")||
        !Array.isArray(submission.images)||!submission.images.length)return false;
    const special=new Set(["row","group_id","selected_task_ids","images","article_evidence","metadata_evidence"]);
    if(Object.entries(submission).some(([key,item])=>!special.has(key)&&typeof item!=="string"))return false;
    return submission.images.every(image=>exactKeys(image,BOOT.export_contract.image_fields)&&Object.values(image).every(item=>typeof item==="string"))&&
      exactKeys(submission.article_evidence,BOOT.export_contract.evidence_fields)&&Object.values(submission.article_evidence).every(item=>typeof item==="string")&&
      exactKeys(submission.metadata_evidence,BOOT.export_contract.evidence_fields)&&Object.values(submission.metadata_evidence).every(item=>typeof item==="string");
  });
}
function stateChanged(){stateRevision+=1;draftImportToken+=1}
function draftImportCurrent(input,file,token,revision){
  return token===draftImportToken&&revision===stateRevision&&Boolean(input.files)&&input.files[0]===file;
}
function save(){
  stateChanged();
  const stored=storageSet(KEY,JSON.stringify(draftEnvelope()));
  if(stored)setMessage("Черновик автоматически сохранен локально.");
  renderProgress();
  return stored;
}
function loadSaved(){
  const raw=storageGet(KEY);if(raw===null)return;
  try{const value=strictJsonParse(raw);if(!validDraft(value))throw new Error("структура не совпадает");state=clone(value.draft)}
  catch(error){setMessage(`Локальный черновик пропущен: ${String(error)}`,true)}
}
function evidenceErrors(evidence,label,optional=false){
  const errors=[],any=Object.values(evidence).some(Boolean);if(optional&&!any)return errors;
  if(!safeUrl(evidence.source_url))errors.push(`${label}: нужен безопасный HTTP(S) source_url без credentials/query/fragment`);
  if(!sha256(evidence.sha256))errors.push(`${label}: нужен lowercase SHA256`);
  if(!trimmed(evidence.retrieved_at))errors.push(`${label}: retrieved_at обязателен`);
  return errors;
}
function identityErrors(value){
  const errors=[];
  if(!exactKeys(value,["kind","artifact_version","queue_fingerprint","actions","policy","publication_ready"])||
      value.kind!=="capacity_identity_resolution_submission"||value.artifact_version!==1||
      value.queue_fingerprint!==BOOT.queue_fingerprint||value.publication_ready!==false)return ["identity submission имеет неверный kind/version/fingerprint или safety state"];
  const policyKeys=Object.keys(BOOT.safe_submission_policy);
  if(!exactKeys(value.policy,policyKeys)||policyKeys.some(key=>value.policy[key]!==BOOT.safe_submission_policy[key]))errors.push("policy identity submission не является fail-closed");
  if(!Array.isArray(value.actions)||value.actions.length!==BOOT.identity_actions.length)return [...errors,`identity submission должен содержать ровно ${BOOT.identity_actions.length} действий`];
  const occupied=new Set(BOOT.groups.filter(group=>!actionByRow.has(group.row)).map(group=>group.dossier_canonical_paper_id).filter(Boolean)),used=new Map();
  value.actions.forEach((resolution,index)=>{
    const action=BOOT.identity_actions[index],fields=["row","group_id","resolution_type","resolved_canonical_paper_id","evidence","capacity_plan_regenerated"];
    if(!exactKeys(resolution,fields)||resolution.row!==action.row||resolution.group_id!==action.group_id){errors.push(`identity action ${index+1}: inventory отличается`);return}
    if(!["confirm_current","replace_with_candidate","replace_with_other"].includes(resolution.resolution_type))errors.push(`строка ${action.row}: identity action не разрешен`);
    const resolved=resolution.resolved_canonical_paper_id;
    if(!canonicalPaperId(resolved))errors.push(`строка ${action.row}: resolved ID не является каноническим lowercase ID`);
    if(resolution.resolution_type==="confirm_current"&&resolved!==action.current_canonical_paper_id)errors.push(`строка ${action.row}: confirm_current должен сохранить текущий ID`);
    if(resolution.resolution_type==="replace_with_candidate"&&(!action.machine_candidate_canonical_paper_id||resolved!==action.machine_candidate_canonical_paper_id))errors.push(`строка ${action.row}: выбранный ID отличается от кандидата`);
    if(resolution.resolution_type==="replace_with_other"&&[action.current_canonical_paper_id,action.machine_candidate_canonical_paper_id].includes(resolved))errors.push(`строка ${action.row}: replace_with_other требует другой ID`);
    const changed=resolved!==action.current_canonical_paper_id;
    if(typeof resolution.capacity_plan_regenerated!=="boolean"||resolution.capacity_plan_regenerated!==changed)errors.push(`строка ${action.row}: capacity_plan_regenerated должен быть ${changed} только по факту изменения ID`);
    if(!exactKeys(resolution.evidence,["source_url","sha256","retrieved_at","notes"]))errors.push(`строка ${action.row}: неверные поля evidence`);
    else{
      if(!safeUrl(resolution.evidence.source_url))errors.push(`строка ${action.row}: неверный evidence URL`);
      if(!sha256(resolution.evidence.sha256))errors.push(`строка ${action.row}: неверный evidence SHA256`);
      for(const key of ["retrieved_at","notes"])if(!trimmed(resolution.evidence[key]))errors.push(`строка ${action.row}: evidence ${key} обязателен`);
    }
    if(canonicalPaperId(resolved)){
      if(occupied.has(resolved))errors.push(`строка ${action.row}: resolved ID уже занят группой без identity action`);
      if(used.has(resolved))errors.push(`строка ${action.row}: resolved ID повторяет identity action строки ${used.get(resolved)}`);
      else used.set(resolved,action.row);
    }
  });
  return errors;
}
function validateSource(){
  const errors=[];
  if(pendingHashCount>0)errors.push({row:null,message:`дождитесь вычисления SHA256: активных операций ${pendingHashCount}`});
  if(!Array.isArray(state.submissions)||state.submissions.length!==150)return [...errors,{row:null,message:"inventory должен содержать ровно 150 упорядоченных групп"}];
  const identityValue=bindings.identity&&bindings.identity.value;
  errors.push(...(identityValue?identityErrors(identityValue):["заново выберите финальный identity submission"]).map(message=>({row:null,message})));
  const lineageValue=bindings.lineage&&bindings.lineage.value;
  errors.push(...(lineageValue?manifestErrors(lineageValue):["заново выберите финальный training lineage manifest"]).map(message=>({row:null,message:`Training lineage: ${message}`})));
  if(!immutableUrl(state.lineage_reference.source_url))errors.push({row:null,message:"URL training lineage должен быть безопасным и содержать неизменяемый 40–64 hex revision"});
  if(!trimmed(state.lineage_reference.retrieved_at))errors.push({row:null,message:"training lineage retrieved_at обязателен"});
  const identityMap=new Map(identityValue&&Array.isArray(identityValue.actions)?identityValue.actions.map(action=>[action.row,action]):[]);
  const samples=new Map(),paperIds=new Map(),paths=new Map(),hashes=new Map();
  const training={paper:new Set(lineageValue?.paper_ids||[]),source:new Set(lineageValue?.source_document_ids||[]),creator:new Set(lineageValue?.creator_group_ids||[]),image:new Set(lineageValue?.image_sha256s||[])};
  for(const [index,submission] of state.submissions.entries()){
    const expected=BOOT.groups[index],row=expected.row,add=message=>errors.push({row,message});
    if(!exactKeys(submission,BOOT.export_contract.submission_fields)||submission.row!==row||submission.group_id!==expected.group_id){add("привязка или поля submission отличаются от контракта");continue}
    if(!Array.isArray(submission.selected_task_ids)||submission.selected_task_ids.length!==1||!expected.primary_task_ids.includes(submission.selected_task_ids[0]))add("выберите ровно одну primary task из primary_task_ids");
    const selectedTask=expected.task_summaries.find(task=>task.task_id===submission.selected_task_ids[0]);
    for(const key of ["sample_id","canonical_paper_id","stratum","model_task_prompt","source_document_id","creator_group_id","scientific_prompt_grounding"])if(!trimmed(submission[key]))add(`${key}: обязательная строка без пробелов по краям`);
    for(const key of ["system_instruction","notes"])if(!trimmed(submission[key],true))add(`${key}: недопустимая строка`);
    if(!["multimodal_hard","temporal_hard","easy_control"].includes(submission.stratum))add("stratum имеет недопустимое значение");
    if(selectedTask?.stratum&&submission.stratum!==selectedTask.stratum)add("stratum должен совпадать с явно выбранной primary task");
    if(!canonicalPaperId(submission.canonical_paper_id))add("canonical_paper_id должен быть каноническим lowercase DOI/arXiv/paper ID");
    const expectedId=identityMap.get(row)?.resolved_canonical_paper_id??expected.dossier_canonical_paper_id;
    if(submission.canonical_paper_id!==expectedId)add("canonical_paper_id не совпадает с dossier или импортированным identity resolution");
    for(const [map,key,label] of [[samples,"sample_id","sample_id"],[paperIds,"canonical_paper_id","canonical_paper_id"]]){
      const value=submission[key];if(value){if(map.has(value))add(`${label} уже используется в группе ${map.get(value)}`);else map.set(value,row)}
    }
    if(training.paper.has(submission.canonical_paper_id))add("canonical_paper_id пересекается с training lineage");
    if(training.source.has(submission.source_document_id))add("source_document_id пересекается с training lineage");
    if(training.creator.has(submission.creator_group_id))add("creator_group_id пересекается с training lineage");
    if(!Array.isArray(submission.images)||!submission.images.length)add("нужно хотя бы одно изображение");
    else submission.images.forEach((image,imageIndex)=>{
      const label=`изображение ${imageIndex+1}`;
      if(!exactKeys(image,BOOT.export_contract.image_fields)){add(`${label}: поля отличаются от контракта`);return}
      if(!safeAssetPath(image.image_path))add(`${label}: image_path должен быть ASCII Windows-safe путем внутри assets/images/`);
      if(!sha256(image.sha256))add(`${label}: нужен lowercase SHA256`);
      else{
        if(forbiddenLegacyHashes.has(image.sha256))add(`${label}: SHA256 принадлежит legacy image с release_usable=false`);
        if(training.image.has(image.sha256))add(`${label}: SHA256 пересекается с training lineage`);
        if(hashes.has(image.sha256))add(`${label}: SHA256 уже используется в группе ${hashes.get(image.sha256)}`);else hashes.set(image.sha256,row);
      }
      if(image.image_path){const pathKey=windowsPathKey(image.image_path),collisionRow=collidingAssetPathRow(paths,pathKey);if(collisionRow!==null)add(`${label}: image_path уже используется или конфликтует как file/directory с группой ${collisionRow}`);else paths.set(pathKey,row)}
      for(const key of ["page","locator","license","citation","retrieved_at"])if(!trimmed(image[key]))add(`${label}: ${key} обязателен`);
      if(!safeUrl(image.source_url))add(`${label}: source_url должен быть безопасным HTTP(S) URL`);
      if(!safeUrl(image.license_url))add(`${label}: license_url должен быть безопасным HTTP(S) URL`);
    });
    evidenceErrors(submission.article_evidence,"Article evidence").forEach(add);
    evidenceErrors(submission.metadata_evidence,"Metadata evidence",true).forEach(add);
  }
  return errors;
}
const STRATA=["multimodal_hard","temporal_hard","easy_control"];
function hasHint(requirement,path){return requirement.machine_prefilled_fields.includes(path)}
function selectTask(submission,task){
  submission.selected_task_ids=[task.task_id];
  if(task.stratum)submission.stratum=task.stratum;
}
function fieldGrid(parent,submission,requirement){
  const grid=el("div",undefined,"grid");
  addField(grid,"ID примера (sample_id)",submission,"sample_id");
  addField(grid,"Канонический ID статьи (canonical_paper_id)",submission,"canonical_paper_id",{machineHint:hasHint(requirement,"canonical_paper_id")});
  const label=el("label","Страта (stratum)"),select=el("select");select.append(el("option","Выберите…"));
  for(const value of STRATA){const option=el("option",value);option.value=value;select.append(option)}
  select.value=submission.stratum;select.addEventListener("change",()=>{submission.stratum=select.value;save()});label.append(select);grid.append(label);
  addField(grid,"ID исходного документа (source_document_id)",submission,"source_document_id");
  addField(grid,"ID группы создателей (creator_group_id)",submission,"creator_group_id");
  addField(grid,"Вопрос модели (model_task_prompt)",submission,"model_task_prompt",{textarea:true});
  addField(grid,"Необязательная системная инструкция",submission,"system_instruction",{textarea:true});
  addField(grid,"Научное обоснование вопроса",submission,"scientific_prompt_grounding",{textarea:true});
  addField(grid,"Примечания",submission,"notes",{textarea:true});parent.append(grid);
}
function evidenceBlock(parent,title,evidence,optional,machineHint){
  const box=el("section",undefined,"source");box.append(el("h3",title),el("p",optional?"Необязательно; если указано одно поле, заполните все три.":"Обязательны все три поля.","help"));
  const grid=el("div",undefined,"grid");addField(grid,"URL источника",evidence,"source_url",{machineHint});
  const shaInput=addField(grid,"SHA256 содержимого",evidence,"sha256");addField(grid,"Время получения (retrieved_at)",evidence,"retrieved_at");
  addFileHasher(grid,evidence,"sha256","Выбрать локальный evidence-файл и вычислить SHA256","",shaInput);box.append(grid);parent.append(box);
}
function imageBlock(parent,submission,image,index,requirement){
  const box=el("section",undefined,"image"),heading=el("div",undefined,"row");heading.append(el("h3",`Изображение ${index+1}`));
  const remove=el("button","Удалить","danger");remove.type="button";remove.disabled=submission.images.length===1;
  remove.addEventListener("click",()=>{cleanupFileHashers(image);submission.images.splice(index,1);save();renderGroup()});heading.append(remove);box.append(heading);
  const grid=el("div",undefined,"grid"),prefix=`images[${index}]`;let shaInput=null;
  addField(grid,"Путь image_path",image,"image_path",{placeholder:"assets/images/group/figure.png"});
  for(const [label,key] of [["SHA256","sha256"],["Страница","page"],["Локатор","locator"],["URL источника","source_url"],["Лицензия","license"],["URL лицензии","license_url"]]){const field=addField(grid,label,image,key,{machineHint:hasHint(requirement,`${prefix}.${key}`)});if(key==="sha256")shaInput=field}
  addField(grid,"Библиографическая ссылка",image,"citation",{textarea:true});addField(grid,"Время получения (retrieved_at)",image,"retrieved_at");
  addFileHasher(grid,image,"sha256","Выбрать локальное изображение, вычислить SHA256 и показать preview","image/*",shaInput);box.append(grid);parent.append(box);
}
function renderGroup(){
  cleanupFileHashers();const requirement=requirementByRow.get(currentRow),submission=state.submissions[currentRow-1],root=$("groupForm");root.replaceChildren();
  root.append(el("h2",`Группа ${currentRow} из 150`),el("p",requirement.group_id,"mono"),el("p",`Текущий ID: ${requirement.dossier_canonical_paper_id||"не разрешен"} | ${requirement.partition_category} | ${requirement.status}`,"meta"));
  const machine=el("section",undefined,"task machine");machine.append(el("strong","НЕПРОВЕРЕННЫЕ МАШИННЫЕ ПОДСКАЗКИ"),el("p","Это только retrieval hints. Они не выбирают задачу, не разрешают identity, не подтверждают права и не заменяют проверку evidence."));
  const labels={canonical_paper_id:"Кандидат ID",article_url:"URL статьи",image_url:"URL изображения",image_sha256:"SHA256 изображения",license:"Лицензия",license_url:"URL лицензии",citation_locator:"Локатор",machine_status:"Машинный статус",machine_reason:"Машинное обоснование"};
  if(requirement.machine_candidate)for(const [key,value] of Object.entries(requirement.machine_candidate))machine.append(el("p",`${labels[key]||key}: ${value??"нет"}`,"mono"));
  root.append(machine,el("h3","Явно выберите одну primary task"));
  const tasks=requirement.task_summaries.filter(task=>task.is_primary&&requirement.primary_task_ids.includes(task.task_id));
  for(const task of tasks){
    const label=el("label",undefined,"task check"),radio=el("input");radio.type="radio";radio.name="primary-task";radio.checked=submission.selected_task_ids[0]===task.task_id;
    radio.addEventListener("change",()=>{selectTask(submission,task);save();renderGroup()});
    const text=el("span");text.append(el("strong",task.task_id),el("div",`source_row_index=${task.source_row_index}; stratum=${task.stratum||"неизвестно"}; primary_endpoint=${task.primary_endpoint}`,"meta"),el("div",`Непроверенные коды: ${[...task.critical_codes,...task.warning_codes,...task.required_action_codes].join(", ")||"нет"}`,"meta"));
    if(task.machine_retrieval_url)text.append(el("div",`Машинный retrieval URL: ${task.machine_retrieval_url}`,"mono meta"));label.append(radio,text);root.append(label);
  }
  fieldGrid(root,submission,requirement);
  evidenceBlock(root,"Evidence статьи",submission.article_evidence,false,hasHint(requirement,"article_evidence.source_url"));
  evidenceBlock(root,"Evidence метаданных",submission.metadata_evidence,true,false);
  root.append(el("h3","Изображения"));submission.images.forEach((image,index)=>imageBlock(root,submission,image,index,requirement));
  const add=el("button","Добавить изображение","secondary");add.type="button";add.addEventListener("click",()=>{submission.images.push({image_path:"",sha256:"",page:"",locator:"",source_url:"",license:"",license_url:"",citation:"",retrieved_at:""});save();renderGroup()});root.append(add);
}
function renderProgress(){
  const errors=validateSource(),badRows=new Set(errors.filter(error=>error.row).map(error=>error.row)),complete=150-badRows.size;
  $("progress").value=complete;$("progressText").textContent=`Без построчных ошибок: ${complete}/150`;
  const query=$("search").value.toLowerCase().trim(),filter=$("filter").value,nav=$("groupNav");nav.replaceChildren();
  for(const group of BOOT.groups){
    const haystack=`${group.row} ${group.group_id} ${group.dossier_canonical_paper_id}`.toLowerCase();
    if(query&&!haystack.includes(query))continue;if(filter==="incomplete"&&!badRows.has(group.row))continue;
    if(filter==="identity"&&!actionByRow.has(group.row))continue;if(filter==="probe"&&!group.machine_candidate?.article_url)continue;
    const button=el("button",`#${group.row} ${group.dossier_canonical_paper_id||"не разрешен"}`);button.type="button";
    if(group.row===currentRow)button.classList.add("current");if(badRows.has(group.row))button.classList.add("incomplete");button.addEventListener("click",()=>go(group.row));nav.append(button);
  }
}
function go(row){if(row<1||row>150)return;currentRow=row;renderGroup();renderProgress();$("groupForm").scrollIntoView({block:"start"})}
function refreshImports(){
  $("lineageUrl").value=state.lineage_reference.source_url;$("lineageRetrieved").value=state.lineage_reference.retrieved_at;
  $("lineageStatus").textContent=bindings.lineage?`Импортирован в память вкладки; SHA256 файла ${bindings.lineage.sha256}`:"Файл не импортирован; выберите его заново.";
  $("identityStatus").textContent=bindings.identity?`Импортирован в память вкладки; SHA256 файла ${bindings.identity.sha256}`:"Файл не импортирован; выберите его заново.";
}
$("lineageUrl").addEventListener("input",event=>{state.lineage_reference.source_url=event.target.value;save()});
$("lineageRetrieved").addEventListener("input",event=>{state.lineage_reference.retrieved_at=event.target.value;save()});
$("lineageFile").addEventListener("change",async event=>{
  stateChanged();
  const input=event.target,file=input.files&&input.files[0],token=++lineageImportToken;bindings.lineage=null;refreshImports();if(!file)return;
  try{const imported=await readJsonWithHash(file);if(token!==lineageImportToken||!input.files||input.files[0]!==file)return;const errors=manifestErrors(imported.value);if(errors.length)throw new Error(errors.join("; "));bindings.lineage=imported;refreshImports();setMessage("Lineage manifest импортирован только в память вкладки.")}
  catch(error){if(token===lineageImportToken)setMessage(`Lineage import отклонен: ${String(error)}`,true)}
});
$("identityFile").addEventListener("change",async event=>{
  stateChanged();
  const input=event.target,file=input.files&&input.files[0],token=++identityImportToken;bindings.identity=null;refreshImports();if(!file)return;
  try{const imported=await readJsonWithHash(file);if(token!==identityImportToken||!input.files||input.files[0]!==file)return;const errors=identityErrors(imported.value);if(errors.length)throw new Error(errors.join("; "));bindings.identity=imported;refreshImports();setMessage("Identity submission импортирован только в память вкладки.")}
  catch(error){if(token===identityImportToken)setMessage(`Identity import отклонен: ${String(error)}`,true)}
});
$("search").addEventListener("input",renderProgress);$("filter").addEventListener("change",renderProgress);
$("firstIncomplete").addEventListener("click",()=>{const first=validateSource().find(error=>error.row);if(first)go(first.row);else setMessage("Построчных ошибок не найдено.")});
$("draftExport").addEventListener("click",()=>downloadJson("capacity_source_submission_draft.json",draftEnvelope()));
$("draftImport").addEventListener("change",async event=>{
  const input=event.target,file=input.files&&input.files[0],token=++draftImportToken,revision=stateRevision;if(!file)return;
  try{
    const value=await readJson(file);if(!draftImportCurrent(input,file,token,revision))return;
    if(!validDraft(value))throw new Error("kind/version/fingerprint, inventory или поля не совпадают");
    const nextState=clone(value.draft);cleanupFileHashers();state=nextState;bindings={identity:null,lineage:null};identityImportToken+=1;lineageImportToken+=1;$("identityFile").value="";$("lineageFile").value="";refreshImports();const persisted=save();renderGroup();
    if(persisted)setMessage("Черновик импортирован. Финальные binding-файлы нужно выбрать заново.");
    else setMessage("Черновик импортирован только в память: localStorage не обновлен. Финальные binding-файлы нужно выбрать заново.",true);
  }catch(error){if(draftImportCurrent(input,file,token,revision))setMessage(`Импорт черновика отклонен: ${String(error)}`,true)}
});
$("clear").addEventListener("click",()=>{
  if(!confirm("Очистить локальный черновик и все введенные метаданные этой формы?"))return;
  cleanupFileHashers();stateChanged();const removed=storageRemove(KEY);state={submissions:clone(BOOT.initial_submissions),lineage_reference:{source_url:"",retrieved_at:""}};bindings={identity:null,lineage:null};identityImportToken+=1;lineageImportToken+=1;$("identityFile").value="";$("lineageFile").value="";refreshImports();renderGroup();renderProgress();
  if(removed)setMessage("Локальный черновик очищен.");
  else setMessage("Черновик очищен только в памяти: сохраненная копия localStorage не удалена.",true);
});
$("finalExport").addEventListener("click",()=>{
  const errors=validateSource();renderErrorSummary(errors,go);
  if(errors.length){setMessage("Финальный экспорт заблокирован. Исправьте все перечисленные ошибки.",true);return}
  const identity=bindings.identity.value,lineage=bindings.lineage.value;
  const artifact={kind:"capacity_source_submission",artifact_version:1,queue_fingerprint:BOOT.queue_fingerprint,submissions:clone(state.submissions),identity_resolution_binding:{kind:identity.kind,artifact_version:identity.artifact_version,queue_fingerprint:identity.queue_fingerprint,sha256:bindings.identity.sha256,action_count:identity.actions.length,actions:clone(identity.actions)},training_lineage_manifest_binding:{filename:"training_lineage_manifest.json",sha256:bindings.lineage.sha256,source_url:state.lineage_reference.source_url,retrieved_at:state.lineage_reference.retrieved_at,manifest:clone(lineage),external_verification_claimed:false},policy:clone(BOOT.safe_submission_policy),publication_ready:false};
  downloadJson("capacity_source_submission.json",artifact);setMessage("Source submission экспортирован; downstream validation остается обязательным.");
});
window.addEventListener("beforeunload",()=>cleanupFileHashers());loadSaved();refreshImports();renderGroup();renderProgress();
"""

_IDENTITY_BODY = r"""
<header><h1>Разрешение идентичности статей</h1><p>Офлайн-предложения владельца данных для __ACTION_COUNT__ известных identity actions. Машинные кандидаты не проверены.</p></header><div class="shell"><section class="notice"><strong>Не финальное решение.</strong> Submission фиксирует предложения с evidence, но не решение curator и не заявление о внешней проверке. Resolved ID не может занимать ID другой группы или повторять другое identity action.</section><div id="errors" class="errors" role="alert" hidden></div><div class="layout"><aside class="card sidebar"><p id="progressText" class="meta"></p><progress id="progress" max="__ACTION_COUNT__" value="0">0/__ACTION_COUNT__</progress><div class="toolbar"><button id="firstIncomplete" type="button">Первая незаполненная</button></div><nav id="actionNav" class="nav-list" aria-label="Identity actions"></nav></aside><main><section id="actionForm" class="card" aria-live="polite"></section></main></div><section class="card"><h2>Черновик и финальный файл</h2><div class="actions"><button id="draftExport" type="button" class="secondary">Экспортировать черновик JSON</button><label>Импортировать черновик этой формы<input id="draftImport" type="file" accept="application/json,.json"></label><button id="clear" type="button" class="danger">Очистить локальный черновик</button><button id="finalExport" type="button">Проверить и экспортировать identity submission</button></div><p id="status" class="status storage-warning" aria-live="polite"></p></section></div>
"""

_IDENTITY_JS = r"""
const KEY=`capacity-handoff:${BOOT.queue_fingerprint}:${BOOT.form_kind}`;
let currentIndex=0,state={resolutions:clone(BOOT.initial_resolutions)};
let draftImportToken=0,stateRevision=0;
const occupiedPaperIds=new Set(BOOT.occupied_non_action_paper_ids);
function envelope(){return {kind:"capacity_identity_resolution_draft",artifact_version:1,queue_fingerprint:BOOT.queue_fingerprint,draft:clone(state)}}
function validDraft(value){
  if(!exactKeys(value,["kind","artifact_version","queue_fingerprint","draft"])||value.kind!=="capacity_identity_resolution_draft"||value.artifact_version!==1||value.queue_fingerprint!==BOOT.queue_fingerprint||!exactKeys(value.draft,["resolutions"])||!Array.isArray(value.draft.resolutions)||value.draft.resolutions.length!==BOOT.actions.length)return false;
  return value.draft.resolutions.every((resolution,index)=>exactKeys(resolution,BOOT.export_contract.resolution_fields)&&resolution.row===BOOT.actions[index].row&&resolution.group_id===BOOT.actions[index].group_id&&typeof resolution.resolution_type==="string"&&typeof resolution.resolved_canonical_paper_id==="string"&&typeof resolution.capacity_plan_regenerated==="boolean"&&exactKeys(resolution.evidence,BOOT.export_contract.evidence_fields)&&Object.values(resolution.evidence).every(item=>typeof item==="string"));
}
function stateChanged(){stateRevision+=1;draftImportToken+=1}
function draftImportCurrent(input,file,token,revision){return token===draftImportToken&&revision===stateRevision&&Boolean(input.files)&&input.files[0]===file}
function save(){stateChanged();const stored=storageSet(KEY,JSON.stringify(envelope()));if(stored)setMessage("Черновик автоматически сохранен локально.");renderNav();return stored}
function loadSaved(){const raw=storageGet(KEY);if(raw===null)return;try{const value=strictJsonParse(raw);if(!validDraft(value))throw new Error("структура не совпадает");state=clone(value.draft)}catch(error){setMessage(`Локальный черновик пропущен: ${String(error)}`,true)}}
function resolutionErrors(resolution,action){
  const errors=[];
  if(!["confirm_current","replace_with_candidate","replace_with_other","unresolved"].includes(resolution.resolution_type))errors.push("выберите тип resolution");
  else if(resolution.resolution_type==="unresolved")errors.push("unresolved допустим в черновике, но блокирует финальный экспорт");
  if(!canonicalPaperId(resolution.resolved_canonical_paper_id))errors.push("resolved canonical ID должен быть каноническим lowercase DOI/arXiv/paper ID");
  if(resolution.resolution_type==="confirm_current"&&resolution.resolved_canonical_paper_id!==action.current_canonical_paper_id)errors.push("confirm_current должен использовать текущий ID");
  if(resolution.resolution_type==="replace_with_candidate"&&(!action.machine_candidate_canonical_paper_id||resolution.resolved_canonical_paper_id!==action.machine_candidate_canonical_paper_id))errors.push("replace_with_candidate должен использовать показанный кандидат");
  if(resolution.resolution_type==="replace_with_other"&&[action.current_canonical_paper_id,action.machine_candidate_canonical_paper_id].includes(resolution.resolved_canonical_paper_id))errors.push("replace_with_other требует другой ID");
  const changed=resolution.resolved_canonical_paper_id!==action.current_canonical_paper_id;
  if(typeof resolution.capacity_plan_regenerated!=="boolean"||resolution.capacity_plan_regenerated!==changed)errors.push(`capacity_plan_regenerated должен быть ${changed}: регенерация нужна только при изменении ID`);
  if(!safeUrl(resolution.evidence.source_url))errors.push("evidence URL должен быть безопасным HTTP(S) URL без credentials/query/fragment");
  if(!sha256(resolution.evidence.sha256))errors.push("evidence SHA256 должен быть lowercase 64-hex");
  for(const key of ["retrieved_at","notes"])if(!trimmed(resolution.evidence[key]))errors.push(`evidence ${key} обязателен`);
  return errors;
}
function allErrors(){
  const errors=[];if(pendingHashCount>0)errors.push({row:null,message:`дождитесь вычисления SHA256: активных операций ${pendingHashCount}`});
  if(!Array.isArray(state.resolutions)||state.resolutions.length!==BOOT.actions.length)return [...errors,{row:null,message:`inventory должен содержать ровно ${BOOT.actions.length} упорядоченных actions`}];
  const used=new Map();
  state.resolutions.forEach((resolution,index)=>{
    const action=BOOT.actions[index];
    if(!exactKeys(resolution,BOOT.export_contract.resolution_fields)||resolution.row!==action.row||resolution.group_id!==action.group_id||!exactKeys(resolution.evidence,BOOT.export_contract.evidence_fields)){errors.push({row:action.row,message:"привязка или поля action отличаются от контракта"});return}
    resolutionErrors(resolution,action).forEach(message=>errors.push({row:resolution.row,message}));
    const resolved=resolution.resolved_canonical_paper_id;
    if(canonicalPaperId(resolved)){
      if(occupiedPaperIds.has(resolved))errors.push({row:resolution.row,message:"resolved ID уже занят группой без identity action"});
      if(used.has(resolved))errors.push({row:resolution.row,message:`resolved ID повторяет identity action строки ${used.get(resolved)}`});else used.set(resolved,resolution.row);
    }
  });return errors;
}
function goRow(row){const index=BOOT.actions.findIndex(action=>action.row===row);if(index>=0){currentIndex=index;renderForm();renderNav();$("actionForm").scrollIntoView({block:"start"})}}
function renderNav(){
  const errors=allErrors(),bad=new Set(errors.filter(error=>error.row).map(error=>error.row)),complete=BOOT.actions.length-bad.size;$("progress").value=complete;$("progressText").textContent=`Без построчных ошибок: ${complete}/${BOOT.actions.length}`;
  const nav=$("actionNav");nav.replaceChildren();BOOT.actions.forEach((action,index)=>{const button=el("button",`#${action.row} ${action.action_type}`);button.type="button";if(index===currentIndex)button.classList.add("current");if(bad.has(action.row))button.classList.add("incomplete");button.addEventListener("click",()=>goRow(action.row));nav.append(button)});
}
function renderForm(){
  cleanupFileHashers();const action=BOOT.actions[currentIndex],resolution=state.resolutions[currentIndex],root=$("actionForm");root.replaceChildren();
  root.append(el("h2",`Identity action ${currentIndex+1} из ${BOOT.actions.length} (исходная группа ${action.row})`),el("p",action.group_id,"mono"));
  const machine=el("section",undefined,"machine task");machine.append(el("strong","НЕПРОВЕРЕННАЯ МАШИННАЯ ПОДСКАЗКА"),el("p",`Текущий ID: ${action.current_canonical_paper_id||"пусто"}`,"mono"),el("p",`Кандидат: ${action.machine_candidate_canonical_paper_id||"нет"}`,"mono"),el("p",`Машинный статус: ${action.machine_status}`),el("p",`Машинное обоснование: ${action.machine_reason}`));root.append(machine);
  const grid=el("div",undefined,"grid"),label=el("label","Тип resolution"),select=el("select");
  for(const [value,text] of [["","Выберите явно…"],["confirm_current","Подтвердить текущий ID"],["replace_with_candidate","Заменить машинным кандидатом"],["replace_with_other","Заменить другим ID"],["unresolved","Не разрешено (только черновик)"]]){const option=el("option",text);option.value=value;select.append(option)}
  select.value=resolution.resolution_type;select.addEventListener("change",()=>{resolution.resolution_type=select.value;save()});label.append(select,el("small","Это состояние предложения, а не retain/exclude решение."));grid.append(label);
  addField(grid,"Разрешенный canonical paper ID",resolution,"resolved_canonical_paper_id");addField(grid,"URL evidence",resolution.evidence,"source_url");const shaInput=addField(grid,"SHA256 evidence",resolution.evidence,"sha256");addField(grid,"Время получения evidence",resolution.evidence,"retrieved_at");addField(grid,"Примечания evidence",resolution.evidence,"notes",{textarea:true});addFileHasher(grid,resolution.evidence,"sha256","Выбрать локальный evidence-файл и вычислить SHA256","",shaInput);
  const check=el("label",undefined,"check"),checkbox=el("input");checkbox.type="checkbox";checkbox.checked=resolution.capacity_plan_regenerated;checkbox.addEventListener("change",()=>{resolution.capacity_plan_regenerated=checkbox.checked;save()});check.append(checkbox,el("span","Capacity plan регенерирован (отмечайте только если resolved ID изменен)"));grid.append(check);root.append(grid);
}
$("firstIncomplete").addEventListener("click",()=>{const first=allErrors().find(error=>error.row);if(first)goRow(first.row);else setMessage("Построчных ошибок identity actions не найдено.")});
$("draftExport").addEventListener("click",()=>downloadJson("capacity_identity_resolution_draft.json",envelope()));
$("draftImport").addEventListener("change",async event=>{
  const input=event.target,file=input.files&&input.files[0],token=++draftImportToken,revision=stateRevision;if(!file)return;
  try{
    const value=await readJson(file);if(!draftImportCurrent(input,file,token,revision))return;
    if(!validDraft(value))throw new Error("kind/version/fingerprint, inventory или поля не совпадают");
    const nextState=clone(value.draft);cleanupFileHashers();state=nextState;const persisted=save();renderForm();
    if(persisted)setMessage("Черновик identity импортирован.");
    else setMessage("Identity-черновик импортирован только в память: localStorage не обновлен.",true);
  }catch(error){if(draftImportCurrent(input,file,token,revision))setMessage(`Импорт черновика отклонен: ${String(error)}`,true)}
});
$("clear").addEventListener("click",()=>{if(!confirm("Очистить локальный identity-черновик?"))return;cleanupFileHashers();stateChanged();const removed=storageRemove(KEY);state={resolutions:clone(BOOT.initial_resolutions)};renderForm();renderNav();if(removed)setMessage("Локальный identity-черновик очищен.");else setMessage("Identity-черновик очищен только в памяти: сохраненная копия localStorage не удалена.",true)});
$("finalExport").addEventListener("click",()=>{const errors=allErrors();renderErrorSummary(errors,goRow);if(errors.length){setMessage("Финальный экспорт заблокирован. Разрешите все actions и исправьте ошибки.",true);return}const artifact={kind:"capacity_identity_resolution_submission",artifact_version:1,queue_fingerprint:BOOT.queue_fingerprint,actions:clone(state.resolutions),policy:clone(BOOT.safe_submission_policy),publication_ready:false};downloadJson("identity_resolution_submission.json",artifact);setMessage("Identity proposal экспортирован; downstream validation остается обязательным.")});
window.addEventListener("beforeunload",()=>cleanupFileHashers());loadSaved();renderForm();renderNav();
"""

_LINEAGE_BODY = r"""
<header><h1>Манифест происхождения обучения</h1><p>Офлайн-конструктор training_lineage_manifest.schema.json. Финальный экспорт содержит ровно schema object.</p></header><div class="shell"><section class="notice"><strong>Точная lineage без заявления о внешней проверке.</strong> Закрепите неизменяемые revisions модели и датасетов, локально вычислите SHA256 каждого файла и явно подтвердите все пять измерений coverage.</section><div id="errors" class="errors" role="alert" hidden></div><section class="card"><h2>Источники обучения</h2><div id="sources"></div><button id="addSource" type="button" class="secondary">Добавить источник обучения</button></section><section class="card"><h2>Покрытие</h2><div id="coverage" class="grid"></div></section><section class="card"><h2>Покрытые идентификаторы</h2><p class="help">По одному непустому уникальному значению на строку.</p><div id="lists" class="grid"></div></section><section class="card"><h2>Черновик и финальный файл</h2><div class="actions"><button id="draftExport" type="button" class="secondary">Экспортировать черновик envelope</button><label>Импортировать черновик этой формы<input id="draftImport" type="file" accept="application/json,.json"></label><button id="clear" type="button" class="danger">Очистить локальный черновик</button><button id="finalExport" type="button">Проверить и экспортировать training_lineage_manifest.json</button></div><p id="status" class="status storage-warning" aria-live="polite"></p></section></div>
"""

_LINEAGE_JS = r"""
const KEY=`capacity-handoff:${BOOT.queue_fingerprint}:${BOOT.form_kind}`;
let state=clone(BOOT.initial_state);
let draftImportToken=0,stateRevision=0;
function envelope(){return {kind:"training_lineage_manifest_draft",artifact_version:1,queue_fingerprint:BOOT.queue_fingerprint,draft:clone(state)}}
function validDraft(value){
  if(!exactKeys(value,["kind","artifact_version","queue_fingerprint","draft"])||value.kind!=="training_lineage_manifest_draft"||value.artifact_version!==1||value.queue_fingerprint!==BOOT.queue_fingerprint)return false;
  const draft=value.draft;if(!exactKeys(draft,["training_sources","coverage","lists"])||!Array.isArray(draft.training_sources)||!draft.training_sources.length||!exactKeys(draft.coverage,BOOT.coverage_fields)||Object.values(draft.coverage).some(item=>typeof item!=="boolean")||!exactKeys(draft.lists,BOOT.list_fields)||Object.values(draft.lists).some(item=>typeof item!=="string"))return false;
  return draft.training_sources.every(source=>exactKeys(source,["repo_id","repo_type","revision","files"])&&typeof source.repo_id==="string"&&typeof source.repo_type==="string"&&typeof source.revision==="string"&&Array.isArray(source.files)&&source.files.length&&source.files.every(file=>exactKeys(file,["path","sha256","row_count"])&&typeof file.path==="string"&&typeof file.sha256==="string"&&(typeof file.row_count==="string"||typeof file.row_count==="number")));
}
function stateChanged(){stateRevision+=1;draftImportToken+=1}
function draftImportCurrent(input,file,token,revision){return token===draftImportToken&&revision===stateRevision&&Boolean(input.files)&&input.files[0]===file}
function save(){stateChanged();const stored=storageSet(KEY,JSON.stringify(envelope()));if(stored)setMessage("Черновик автоматически сохранен локально.");return stored}
function loadSaved(){const raw=storageGet(KEY);if(raw===null)return;try{const value=strictJsonParse(raw);if(!validDraft(value))throw new Error("структура не совпадает");state=clone(value.draft)}catch(error){setMessage(`Локальный черновик пропущен: ${String(error)}`,true)}}
function rowCount(value){return typeof value==="string"&&/^(0|[1-9][0-9]*)$/.test(value)?Number(value):value}
function manifestValue(){return {schema_version:1,training_sources:state.training_sources.map(source=>({repo_id:source.repo_id,repo_type:source.repo_type,revision:source.revision,files:source.files.map(file=>({path:file.path,sha256:file.sha256,row_count:rowCount(file.row_count)}))})),coverage:clone(state.coverage),paper_ids:listValues(state.lists.paper_ids),source_document_ids:listValues(state.lists.source_document_ids),creator_group_ids:listValues(state.lists.creator_group_ids),image_sha256s:listValues(state.lists.image_sha256s),prompt_sha256s:listValues(state.lists.prompt_sha256s)}}
function renderSources(){
  cleanupFileHashers();const root=$("sources");root.replaceChildren();state.training_sources.forEach((source,index)=>{
    const box=el("section",undefined,"source"),head=el("div",undefined,"row");head.append(el("h3",`Источник обучения ${index+1}`));
    const remove=el("button","Удалить источник","danger");remove.type="button";remove.disabled=state.training_sources.length===1;remove.addEventListener("click",()=>{source.files.forEach(file=>cleanupFileHashers(file));state.training_sources.splice(index,1);save();renderSources()});head.append(remove);box.append(head);
    const grid=el("div",undefined,"grid");addField(grid,"ID репозитория (repo_id)",source,"repo_id");const label=el("label","Тип репозитория (repo_type)"),select=el("select");for(const [value,text] of [["","Выберите…"],["model","Модель"],["dataset","Датасет"]]){const option=el("option",text);option.value=value;select.append(option)}select.value=source.repo_type;select.addEventListener("change",()=>{source.repo_type=select.value;save()});label.append(select);grid.append(label);addField(grid,"Неизменяемый lowercase revision из 40 hex",source,"revision");box.append(grid,el("h4","Файлы"));
    source.files.forEach((file,fileIndex)=>{const fileBox=el("div",undefined,"image"),fileHead=el("div",undefined,"row");fileHead.append(el("strong",`Файл ${fileIndex+1}`));const removeFile=el("button","Удалить файл","danger");removeFile.type="button";removeFile.disabled=source.files.length===1;removeFile.addEventListener("click",()=>{cleanupFileHashers(file);source.files.splice(fileIndex,1);save();renderSources()});fileHead.append(removeFile);fileBox.append(fileHead);const fileGrid=el("div",undefined,"grid");addField(fileGrid,"Путь в закрепленном репозитории",file,"path");const shaInput=addField(fileGrid,"SHA256",file,"sha256");addField(fileGrid,"Неотрицательный row_count",file,"row_count",{type:"number"});addFileHasher(fileGrid,file,"sha256","Выбрать локальный файл и вычислить SHA256","",shaInput);fileBox.append(fileGrid);box.append(fileBox)});
    const addFile=el("button","Добавить файл","secondary");addFile.type="button";addFile.addEventListener("click",()=>{source.files.push({path:"",sha256:"",row_count:""});save();renderSources()});box.append(addFile);root.append(box);
  });
}
function renderCoverage(){const root=$("coverage");root.replaceChildren();const labels={paper_ids:"ID статей",source_documents:"Исходные документы",creator_groups:"Группы создателей",image_bytes:"Байты изображений",prompts:"Промпты"};for(const key of BOOT.coverage_fields){const label=el("label",undefined,"check"),input=el("input");input.type="checkbox";input.checked=state.coverage[key];input.addEventListener("change",()=>{state.coverage[key]=input.checked;save()});label.append(input,el("span",`${labels[key]} (${key})`));root.append(label)}}
function renderLists(){const root=$("lists");root.replaceChildren();for(const key of BOOT.list_fields)addField(root,`${key} (по одному на строку)`,state.lists,key,{textarea:true,help:key.endsWith("sha256s")?"Каждое значение — lowercase SHA256 из 64 hex.":key==="paper_ids"?"Каждый ID — канонический lowercase DOI/arXiv/paper ID.":"Непустые уникальные значения без пробелов по краям."})}
$("addSource").addEventListener("click",()=>{state.training_sources.push({repo_id:"",repo_type:"",revision:"",files:[{path:"",sha256:"",row_count:""}]});save();renderSources()});
$("draftExport").addEventListener("click",()=>downloadJson("training_lineage_manifest_draft.json",envelope()));
$("draftImport").addEventListener("change",async event=>{
  const input=event.target,file=input.files&&input.files[0],token=++draftImportToken,revision=stateRevision;if(!file)return;
  try{
    const value=await readJson(file);if(!draftImportCurrent(input,file,token,revision))return;
    if(!validDraft(value))throw new Error("kind/version/fingerprint или поля черновика не совпадают");
    const nextState=clone(value.draft);cleanupFileHashers();state=nextState;const persisted=save();renderSources();renderCoverage();renderLists();
    if(persisted)setMessage("Черновик lineage импортирован.");
    else setMessage("Lineage-черновик импортирован только в память: localStorage не обновлен.",true);
  }catch(error){if(draftImportCurrent(input,file,token,revision))setMessage(`Импорт черновика отклонен: ${String(error)}`,true)}
});
$("clear").addEventListener("click",()=>{if(!confirm("Очистить локальный lineage-черновик?"))return;cleanupFileHashers();stateChanged();const removed=storageRemove(KEY);state=clone(BOOT.initial_state);renderSources();renderCoverage();renderLists();if(removed)setMessage("Локальный lineage-черновик очищен.");else setMessage("Lineage-черновик очищен только в памяти: сохраненная копия localStorage не удалена.",true)});
$("finalExport").addEventListener("click",()=>{const manifest=manifestValue(),errors=manifestErrors(manifest).map(message=>({row:null,message}));if(pendingHashCount>0)errors.unshift({row:null,message:`дождитесь вычисления SHA256: активных операций ${pendingHashCount}`});renderErrorSummary(errors);if(errors.length){setMessage("Финальный экспорт заблокирован. Исправьте все ошибки lineage.",true);return}downloadJson("training_lineage_manifest.json",manifest);setMessage("Schema-compatible training lineage manifest экспортирован.")});
window.addEventListener("beforeunload",()=>cleanupFileHashers());loadSaved();renderSources();renderCoverage();renderLists();
"""


def _training_lineage_form(queue_fingerprint: str) -> bytes:
    coverage_fields = [
        "paper_ids",
        "source_documents",
        "creator_groups",
        "image_bytes",
        "prompts",
    ]
    list_fields = [
        "paper_ids",
        "source_document_ids",
        "creator_group_ids",
        "image_sha256s",
        "prompt_sha256s",
    ]
    data = {
        "form_kind": "training_lineage_manifest_form",
        "artifact_version": ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "coverage_fields": coverage_fields,
        "list_fields": list_fields,
        "initial_state": {
            "training_sources": [
                {
                    "repo_id": "",
                    "repo_type": "",
                    "revision": "",
                    "files": [{"path": "", "sha256": "", "row_count": ""}],
                }
            ],
            "coverage": dict.fromkeys(coverage_fields, False),
            "lists": dict.fromkeys(list_fields, ""),
        },
        "export_contract": {
            "filename": "training_lineage_manifest.json",
            "schema_version": 1,
            "top_level_fields": [
                "schema_version",
                "training_sources",
                "coverage",
                *list_fields,
            ],
            "draft_kind": "training_lineage_manifest_draft",
        },
    }
    return _html_document("Манифест происхождения обучения", _LINEAGE_BODY, data, _LINEAGE_JS)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inventory(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"path": name, "sha256": _digest(data), "size_bytes": len(data)}
        for name, data in sorted(files.items())
    ]


def _artifacts(
    dossier_bytes: bytes,
    audit_bytes: bytes,
    requirements: Sequence[Mapping[str, Any]],
    row_category: Mapping[int, str],
    probes: Mapping[int, Mapping[str, Any]],
    exceptions: Mapping[int, Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    identity_routes: Mapping[int, str],
    queue_fingerprint: str,
    quarantined_hashes: Sequence[str],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    empty_dossier_action_count = sum(
        action["action_type"] == "unresolved_empty_dossier_id" for action in actions
    )
    noncanonical_dossier_action_count = sum(
        action["action_type"] == "noncanonical_dossier_id" for action in actions
    )
    duplicate_dossier_action_count = sum(
        action["action_type"] == "duplicate_dossier_id" for action in actions
    )
    payloads = {
        "enrichment_requirements.jsonl": _jsonl_bytes(requirements),
        "source_submission_template.csv": _source_template(requirements),
        "identity_resolution_template.csv": _identity_template(actions),
        "source_submission_form.html": _source_form(
            requirements, actions, queue_fingerprint, quarantined_hashes
        ),
        "identity_resolution_form.html": _identity_form(
            actions,
            queue_fingerprint,
            sorted(
                {
                    requirement["dossier_canonical_paper_id"]
                    for requirement in requirements
                    if requirement["row"] not in identity_routes
                    and requirement["dossier_canonical_paper_id"]
                }
            ),
        ),
        "training_lineage_form.html": _training_lineage_form(queue_fingerprint),
        "README_RU.md": _readme(
            len(actions),
            empty_dossier_action_count,
            noncanonical_dossier_action_count,
            duplicate_dossier_action_count,
        ),
    }
    if set(payloads) != PAYLOAD_FILENAMES:
        raise HandoffError("internal handoff payload inventory is incomplete")
    matching = sum(
        probe["dossier_canonical_paper_id"] == probe["verified_canonical_paper_id"]
        for probe in probes.values()
    )
    counts = {
        "total_rows": TOTAL_ROWS,
        "source_probe_count": len(probes),
        "matching_current_source_probe_count": matching,
        "identity_correction_source_probe_count": len(probes) - matching,
        "metadata_only_or_unlicensed_count": sum(
            category == "metadata_only_or_unlicensed" for category in row_category.values()
        ),
        "unresolved_identity_count": sum(
            category == "unresolved_identity" for category in row_category.values()
        ),
        "duplicate_candidate_count": sum(
            category == "duplicate_candidate" for category in row_category.values()
        ),
        "no_retrieval_hint_count": sum(
            category == "no_retrieval_hint" for category in row_category.values()
        ),
        "identity_exception_count": len(exceptions),
        "unresolved_empty_dossier_id_action_count": empty_dossier_action_count,
        "noncanonical_dossier_id_action_count": noncanonical_dossier_action_count,
        "duplicate_dossier_id_action_count": duplicate_dossier_action_count,
        "identity_action_count": len(actions),
    }
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "source_inputs": {
            "dossiers": {
                "sha256": _digest(dossier_bytes),
                "size_bytes": len(dossier_bytes),
            },
            "source_audit": {
                "sha256": _digest(audit_bytes),
                "size_bytes": len(audit_bytes),
            },
        },
        "counts": counts,
        "policy": {
            "network_retrieval_performed_by_generator": False,
            "source_probe_network_audit_present": True,
            "direct_image_bytes_sha256_machine_checked": True,
            "source_probes_machine_only": True,
            "source_probes_human_verified": False,
            "external_evidence_content_verified": False,
            "automatic_task_selection": False,
            "automatic_retain": False,
            "reviewer_identity_prefill": False,
            "attestation_prefill": False,
            "decision_prefill": False,
            "curator_importable": False,
            "capacity_plan_regenerated": False,
            "full_enrichment_validation_required": True,
        },
        "publication_ready": False,
        "file_count": len(payloads),
        "files": _inventory(payloads),
    }
    return {**payloads, MANIFEST_FILENAME: _json_bytes(manifest)}, manifest


def _workspace_matches(target: Path, files: Mapping[str, bytes]) -> bool:
    try:
        entries = list(target.iterdir())
    except OSError:
        return False
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        return False
    if {entry.name for entry in entries} != set(files):
        return False
    try:
        return all((target / name).read_bytes() == data for name, data in files.items())
    except OSError:
        return False


def _reject_output_link_components(target: Path) -> None:
    if target.drive and not target.root:
        raise HandoffError("output-dir must not use a drive-relative path")
    if target.is_absolute():
        current = Path(target.anchor)
        components = [current]
        parts = target.parts[1:]
    else:
        current = Path.cwd()
        components = [*reversed(current.parents), current]
        parts = target.parts
    for part in parts:
        if part in {"", "."}:
            continue
        current = current.parent if part == ".." else current / part
        components.append(current)

    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for component in components:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise HandoffError(f"cannot inspect output-dir component {component}: {exc}") from exc
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or attributes & reparse_point:
            raise HandoffError(
                f"output-dir component must not be a symlink or reparse point: {component}"
            )


def _publish_write_once(
    output_dir: str | Path,
    files: Mapping[str, bytes],
    protected_input_paths: Sequence[Path],
) -> None:
    target = Path(output_dir)
    _reject_output_link_components(target)
    try:
        resolved_target = target.resolve(strict=False)
        parent = resolved_target.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HandoffError(f"cannot resolve output-dir: {exc}") from exc
    if parent.is_symlink() or not parent.is_dir():
        raise HandoffError("output-dir parent must be an existing non-symlink directory")
    for source in protected_input_paths:
        source_root = source.parent
        if resolved_target == source_root or source_root in resolved_target.parents:
            raise HandoffError("output-dir must be outside immutable input workspaces")
    if resolved_target.exists():
        if resolved_target.is_dir() and _workspace_matches(resolved_target, files):
            return
        raise HandoffError("output-dir already contains a different handoff workspace")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_target.name}.capacity-handoff.", suffix=".tmp", dir=parent
        )
    )
    try:
        for name, data in sorted(files.items()):
            path = staging / name
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        if resolved_target.exists():
            if resolved_target.is_dir() and _workspace_matches(resolved_target, files):
                return
            raise HandoffError("output-dir appeared with different contents during publication")
        try:
            os.replace(staging, resolved_target)
        except OSError as exc:
            if resolved_target.is_dir() and _workspace_matches(resolved_target, files):
                return
            raise HandoffError(f"cannot publish handoff workspace atomically: {exc}") from exc
    finally:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)


def generate_handoff(
    dossiers_path: str | Path,
    source_audit_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate immutable inputs and publish a blocked, write-once handoff workspace."""

    (
        dossier_path,
        dossier_bytes,
        audit_path,
        audit_bytes,
        dossiers,
        _partition,
        row_category,
        probes,
        exceptions,
        queue_fingerprint,
    ) = _validated_inputs(dossiers_path, source_audit_path)
    identity_routes = _identity_routes(dossiers, probes, exceptions)
    actions = _identity_actions(dossiers, probes, exceptions, row_category, identity_routes)
    requirements = _requirements(
        dossiers,
        row_category,
        probes,
        exceptions,
        identity_routes,
        queue_fingerprint,
    )
    quarantined_hashes = sorted(
        {digest for dossier in dossiers for digest in dossier["quarantined_legacy_sha256s"]}
    )
    files, manifest = _artifacts(
        dossier_bytes,
        audit_bytes,
        requirements,
        row_category,
        probes,
        exceptions,
        actions,
        identity_routes,
        queue_fingerprint,
        quarantined_hashes,
    )
    _publish_write_once(output_dir, files, (dossier_path, audit_path))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic non-curator capacity enrichment handoff."
    )
    parser.add_argument("--dossiers", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = generate_handoff(args.dossiers, args.source_audit, args.output_dir)
    except (HandoffError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
