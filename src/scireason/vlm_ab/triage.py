# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build offline, queue-bound assisted triage proposals for human curators."""

from __future__ import annotations

import json
import os
import shutil
from csv import writer
from collections import Counter
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

from .curator import CURATOR_ARTIFACT_VERSION, _blank_edit
from .remediation import (
    RemediationError,
    _atomic_write,
    _json_bytes,
    _jsonl_bytes,
    _prepare_input_protection,
    _queue_material,
    _sha256_bytes,
    _separate_output_target,
    _trees_identical,
    _verify_queue_workspace,
)


TRIAGE_ARTIFACT_VERSION = 1
ASSISTED_REVIEW_DRAFT_FILENAME = "assisted_review_draft.json"
TRIAGE_FILENAME = "triage.jsonl"
TRIAGE_SUMMARY_FILENAME = "triage_summary.md"
TRIAGE_MANIFEST_FILENAME = "triage_manifest.json"
REVIEW_LOG_TEMPLATE_FILENAME = "external_review_log_template.csv"
_TRAINING_PAPER_OVERLAP_CODE = "training_paper_overlap"

# This is a routing aid, not a finding-to-fix claim. Each known audit code has one
# deterministic suggested action; unknown codes remain explicit human investigations.
_CODE_ACTIONS = {
    "cross_paper_image_reuse": "replace_and_verify_source_images",
    "duplicate_normalized_prompt": "rewrite_standalone_prompt",
    "duplicate_sample_id": "resolve_duplicate_id",
    "empty_benchmark": "human_review_other",
    "image_placeholder_mismatch": "replace_and_verify_source_images",
    "insufficient_primary_papers": "human_review_other",
    "likely_answer_leakage": "remove_answer_leakage",
    "missing_gold": "human_review_other",
    "missing_image": "replace_and_verify_source_images",
    "missing_training_lineage": "human_review_other",
    "provenance_mismatch": "create_complete_provenance",
    "residual_comparison_prompt": "rewrite_standalone_prompt",
    "schema_error": "repair_schema_and_split_provenance",
    "training_lineage_overlap": "human_review_other",
    "training_paper_overlap": "confirm_training_overlap_exclusion",
    "training_prompt_overlap": "rewrite_standalone_prompt",
    "unsafe_image_path": "replace_and_verify_source_images",
    "within_row_duplicate_image_bytes": "replace_and_verify_source_images",
}


def _task_string_list(task: Mapping[str, Any], field: str) -> list[str]:
    value = task.get(field)
    task_id = task.get("task_id")
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RemediationError(f"queue task {task_id!r} has malformed {field}")
    return sorted(set(value))


def _required_actions(
    critical_codes: Sequence[str],
    warning_codes: Sequence[str],
    *,
    has_training_overlap: bool,
) -> list[str]:
    """Return deterministic routing labels without resolving any source finding."""

    actions = {
        _CODE_ACTIONS.get(code, f"manual_investigation:{code}")
        for code in set(critical_codes) | set(warning_codes)
    }
    if has_training_overlap:
        actions.add("confirm_training_overlap_exclusion")
    if not actions:
        actions.add("human_review_other")
    return sorted(actions)


def _has_training_paper_overlap(
    critical_codes: Sequence[str], overlap_paper_ids: Sequence[str]
) -> bool:
    """Honor the audit code even when legacy task metadata lacks a canonical ID."""

    return bool(overlap_paper_ids) or _TRAINING_PAPER_OVERLAP_CODE in critical_codes


def _overlap_draft_edit(
    overlap_paper_ids: Sequence[str], *, task_id: str, source_row_index: int
) -> dict[str, Any]:
    edit = _blank_edit()
    edit["disposition"] = "exclude"
    identifiers = (
        f"Обнаружены canonical paper ID: {', '.join(overlap_paper_ids)}."
        if overlap_paper_ids
        else (
            "Audit установил критический код training_paper_overlap, но queue не содержит "
            "извлеченный canonical paper ID; сверить исходную запись и audit evidence."
        )
    )
    edit["exclusion_reason"] = (
        "Предварительное предложение об исключении по заранее выбранной политике: "
        "исключать все задачи с пересечением с training. "
        f"{identifiers} task_id={task_id}; source_row_index={source_row_index}. "
        "Это не завершенное решение: требуются две независимые человеческие аттестации."
    )
    return edit


def _triage_row(task: Mapping[str, Any], queue_fingerprint: str) -> dict[str, Any]:
    task_id = task.get("task_id")
    source_row_index = task.get("source_row_index")
    if not isinstance(task_id, str) or not task_id:
        raise RemediationError("queue task has no task_id")
    if isinstance(source_row_index, bool) or not isinstance(source_row_index, int):
        raise RemediationError(f"queue task {task_id} has malformed source_row_index")
    overlap_paper_ids = _task_string_list(task, "training_overlap_paper_ids")
    critical_codes = _task_string_list(task, "critical_codes")
    warning_codes = _task_string_list(task, "warning_codes")
    has_training_overlap = _has_training_paper_overlap(critical_codes, overlap_paper_ids)
    identifier_status = (
        "canonical_ids_available"
        if overlap_paper_ids
        else "audit_code_without_extracted_canonical_id"
        if has_training_overlap
        else "not_detected"
    )
    return {
        "artifact_version": TRIAGE_ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "task_id": task_id,
        "source_row_index": source_row_index,
        "proposed_disposition": (
            "exclude_pending_human_attestation"
            if has_training_overlap
            else "manual_review_required"
        ),
        "training_overlap_detected": has_training_overlap,
        "training_overlap_identifier_status": identifier_status,
        "training_overlap_paper_ids": overlap_paper_ids,
        "critical_codes": critical_codes,
        "warning_codes": warning_codes,
        "required_actions": _required_actions(
            critical_codes,
            warning_codes,
            has_training_overlap=has_training_overlap,
        ),
        "explanation": (
            "Машинное предложение об исключении по заранее выбранной политике; до завершения "
            "нужны две независимые человеческие аттестации."
            if has_training_overlap
            else "Автоматическое сохранение не выполняется: требуется ручная проверка по "
            "указанным машинно сформированным действиям."
        ),
    }


def _markdown_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _summary_markdown(
    triage_rows: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
    overlap_paper_ids: Sequence[str],
) -> bytes:
    action_counts = Counter(action for row in triage_rows for action in row["required_actions"])
    critical_counts = Counter(code for row in triage_rows for code in row["critical_codes"])
    warning_counts = Counter(code for row in triage_rows for code in row["warning_codes"])
    overlap_count = sum(bool(row["training_overlap_detected"]) for row in triage_rows)
    missing_identifier_count = sum(
        row["training_overlap_identifier_status"] == "audit_code_without_extracted_canonical_id"
        for row in triage_rows
    )
    manual_count = len(triage_rows) - overlap_count

    lines = [
        "# Assisted-triage для ручной курации",
        "",
        "Этот пакет создан локально из верифицированной immutable queue. Он содержит только "
        "машинную маршрутизацию и черновик для последующей человеческой проверки.",
        "",
        "## Сводка",
        "",
        f"- Queue fingerprint: `{queue_fingerprint}`",
        f"- Всего задач: {len(triage_rows)}",
        f"- Задач с training overlap: {overlap_count}",
        f"- Других задач, оставленных для ручной проверки: {manual_count}",
        f"- Уникальных canonical overlap paper ID: {len(overlap_paper_ids)}",
        f"- Overlap-задач без извлеченного canonical ID в queue: {missing_identifier_count}",
        "",
        "## Политика и обязательные действия",
        "",
        "Политика была явно выбрана при запуске: все задачи с критическим audit code "
        "`training_paper_overlap` или непустым `training_overlap_paper_ids` получают только "
        "предложение `exclude_pending_human_attestation`.",
        "",
        "В пакете нет завершенных решений: он не создает `completed_decisions.jsonl`, не "
        "заполняет reviewer ID и оставляет `independent_attestation=false`. Два реальных "
        "независимых человека должны проверить каждое предложение, заполнить свои ID и "
        "подтвердить аттестацию в master curator workspace.",
        "",
        "Ни один `retain` не создается автоматически. Все задачи без training overlap остаются "
        "для ручной курации; человек вправе отклонить или изменить предложение после проверки.",
        "",
        "Сопоставление кодов с required actions используется только для машинной маршрутизации. "
        "Оно не доказывает истинность исходной записи и не устанавливает причину исправления.",
        "",
        "## Группы по required actions",
        "",
    ]
    if action_counts:
        lines.extend(
            f"- {_markdown_value(action)}: {count}"
            for action, count in sorted(action_counts.items())
        )
    else:
        lines.append("- Нет действий.")

    lines.extend(["", "## Группы по критическим кодам", ""])
    if critical_counts:
        lines.extend(
            f"- {_markdown_value(code)}: {count}" for code, count in sorted(critical_counts.items())
        )
    else:
        lines.append("- Критические коды отсутствуют.")

    lines.extend(["", "## Группы по warning-кодам", ""])
    if warning_counts:
        lines.extend(
            f"- {_markdown_value(code)}: {count}" for code, count in sorted(warning_counts.items())
        )
    else:
        lines.append("- Warning-коды отсутствуют.")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _file_inventory(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "sha256": _sha256_bytes(data),
            "size_bytes": len(data),
        }
        for name, data in sorted(files.items())
    ]


def _review_log_template(triage_rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Emit a blank external accountability record without fabricating reviewers."""

    handle = StringIO(newline="")
    csv_writer = writer(handle, lineterminator="\n")
    csv_writer.writerow(
        [
            "task_id",
            "source_row_index",
            "machine_proposal",
            "training_overlap_identifier_status",
            "training_overlap_paper_ids",
            "reviewer_1_id",
            "reviewer_1_reviewed_at",
            "reviewer_1_signed_or_dated_record",
            "reviewer_2_id",
            "reviewer_2_reviewed_at",
            "reviewer_2_signed_or_dated_record",
            "human_final_disposition",
            "human_review_notes",
        ]
    )
    for row in triage_rows:
        csv_writer.writerow(
            [
                row["task_id"],
                row["source_row_index"],
                row["proposed_disposition"],
                row["training_overlap_identifier_status"],
                ";".join(row["training_overlap_paper_ids"]),
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
    return handle.getvalue().encode("utf-8")


def _existing_workspace_is_identical(staging: Path, target: Path, filenames: set[str]) -> bool:
    try:
        entries = list(target.iterdir())
    except OSError as exc:
        raise RemediationError(f"cannot inspect existing triage workspace: {exc}") from exc
    if {entry.name for entry in entries} != filenames or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        return False
    return _trees_identical(staging, target)


def _publish_workspace(target: Path, files: Mapping[str, bytes]) -> None:
    staging = target.with_name(f".{target.name}.triage.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise RemediationError(f"triage staging path already exists: {staging}")
    try:
        staging.mkdir()
        for name, data in sorted(files.items()):
            _atomic_write(staging / name, data)
        names = set(files)
        if target.exists() or target.is_symlink():
            if (
                target.is_dir()
                and not target.is_symlink()
                and _existing_workspace_is_identical(staging, target, names)
            ):
                return
            raise RemediationError("output_dir already contains a different triage workspace")
        try:
            os.replace(staging, target)
        except OSError as exc:
            if (
                target.is_dir()
                and not target.is_symlink()
                and _existing_workspace_is_identical(staging, target, names)
            ):
                return
            raise RemediationError(f"cannot publish triage workspace atomically: {exc}") from exc
    finally:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)


def generate_triage_package(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    queue_manifest: str | Path,
    output_dir: str | Path,
    *,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
    exclude_training_overlap: bool,
) -> dict[str, Any]:
    """Publish an immutable-queue-bound, human-only assisted-triage package."""

    if exclude_training_overlap is not True:
        raise RemediationError("exclude_training_overlap must be explicitly true")
    material = _queue_material(config, prepare_manifest, benchmark_schema, provenance_schema)
    queue_root, verified_queue = _verify_queue_workspace(material, queue_manifest)
    protected_directories, protected_files = _prepare_input_protection(material)
    target = _separate_output_target(
        output_dir,
        protected_directories=(queue_root, *protected_directories),
        protected_files=protected_files,
    )
    queue_fingerprint = verified_queue["queue_fingerprint"]

    tasks = sorted(
        material["tasks"],
        key=lambda task: (task["source_row_index"], task["task_id"]),
    )
    triage_rows = [_triage_row(task, queue_fingerprint) for task in tasks]
    edits = {
        row["task_id"]: (
            _overlap_draft_edit(
                row["training_overlap_paper_ids"],
                task_id=row["task_id"],
                source_row_index=row["source_row_index"],
            )
            if row["training_overlap_detected"]
            else _blank_edit()
        )
        for row in triage_rows
    }
    overlap_paper_ids = sorted(
        {paper_id for row in triage_rows for paper_id in row["training_overlap_paper_ids"]}
    )
    draft = {
        "artifact_version": CURATOR_ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "edits": edits,
    }
    payload_files = {
        ASSISTED_REVIEW_DRAFT_FILENAME: _json_bytes(draft, newline=True),
        TRIAGE_FILENAME: _jsonl_bytes(triage_rows),
        TRIAGE_SUMMARY_FILENAME: _summary_markdown(
            triage_rows,
            queue_fingerprint,
            overlap_paper_ids,
        ),
        REVIEW_LOG_TEMPLATE_FILENAME: _review_log_template(triage_rows),
    }
    manifest = {
        "artifact_version": TRIAGE_ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "task_count": len(triage_rows),
        "training_overlap_task_count": sum(
            bool(row["training_overlap_detected"]) for row in triage_rows
        ),
        "training_overlap_missing_identifier_task_count": sum(
            row["training_overlap_identifier_status"] == "audit_code_without_extracted_canonical_id"
            for row in triage_rows
        ),
        "overlap_paper_ids": overlap_paper_ids,
        "policy": {
            "exclude_training_overlap": True,
            "overlap_proposal": "exclude_pending_human_attestation",
            "nonoverlap_proposal": "manual_review_required",
            "automatic_retain": False,
        },
        # A manifest cannot hash itself; the inventory covers every other package file.
        "files": _file_inventory(payload_files),
    }
    files = {
        **payload_files,
        TRIAGE_MANIFEST_FILENAME: _json_bytes(manifest, newline=True),
    }
    _publish_workspace(target, files)
    return manifest


# The descriptive name is retained as a small public alias for callers that use the workflow term.
generate_assisted_triage_package = generate_triage_package


__all__ = ["generate_assisted_triage_package", "generate_triage_package"]
