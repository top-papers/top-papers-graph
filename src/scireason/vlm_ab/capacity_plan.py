# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Create queue-bound Phase 0 capacity review plans without making curation decisions."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from csv import DictWriter
from io import StringIO
from pathlib import Path
from typing import Any

from .audit import canonical_paper_id, paper_identity_errors
from .capacity import is_capacity_remediation_working_config
from .remediation import (
    RemediationError,
    _atomic_write,
    _json_bytes,
    _jsonl_bytes,
    _prepare_input_protection,
    _queue_material,
    _separate_output_target,
    _sha256_bytes,
    _trees_identical,
    _verify_queue_workspace,
)
from .triage import _has_training_paper_overlap, _required_actions


CAPACITY_PLAN_ARTIFACT_VERSION = 2
REVIEW_ASSIGNMENT_PLAN_FILENAME = "review_assignment_plan.csv"
CAPACITY_PAPER_GROUPS_FILENAME = "capacity_paper_groups.jsonl"
CAPACITY_PLAN_SUMMARY_FILENAME = "capacity_plan_summary.md"
CAPACITY_PLAN_MANIFEST_FILENAME = "capacity_plan_manifest.json"

_REVIEWER_SLOTS = ("reviewer-slot-1", "reviewer-slot-2", "reviewer-slot-3")
_REVIEWER_PAIRS = (
    (_REVIEWER_SLOTS[0], _REVIEWER_SLOTS[1]),
    (_REVIEWER_SLOTS[1], _REVIEWER_SLOTS[2]),
    (_REVIEWER_SLOTS[2], _REVIEWER_SLOTS[0]),
)
_ASSIGNMENT_FIELDS = (
    "task_id",
    "source_row_index",
    "workstream",
    "proposed_disposition",
    "reviewer_slot_1",
    "reviewer_slot_2",
    "critical_codes",
    "required_actions",
)


def _require_capacity_remediation_config(
    config: Mapping[str, Any], *, command: str = "curate-capacity-plan"
) -> int:
    power = config.get("power")
    if (
        not is_capacity_remediation_working_config(config)
        or not isinstance(power, Mapping)
        or power.get("require_exact_n_items") is not True
    ):
        raise RemediationError(
            f"{command} requires a capacity remediation working config with "
            "power.require_exact_n_items=true"
        )
    exact_n = power.get("n_items")
    if isinstance(exact_n, bool) or not isinstance(exact_n, int) or exact_n <= 0:
        raise RemediationError("capacity remediation config has an invalid exact n_items value")
    return exact_n


def _task_string_list(task: Mapping[str, Any], field: str) -> list[str]:
    value = task.get(field)
    task_id = task.get("task_id")
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RemediationError(f"queue task {task_id!r} has malformed {field}")
    return sorted(set(value))


def _task_details(
    tasks: Sequence[Mapping[str, Any]], primary_strata: Sequence[str]
) -> list[dict[str, Any]]:
    if (
        isinstance(primary_strata, (str, bytes))
        or not primary_strata
        or any(not isinstance(value, str) or not value for value in primary_strata)
    ):
        raise RemediationError("capacity routing requires configured primary strata")
    primary = frozenset(primary_strata)
    details: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    source_indices: set[int] = set()
    for task in tasks:
        task_id = task.get("task_id")
        source_row_index = task.get("source_row_index")
        row = task.get("original_row")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id != task_id.strip()
            or "\x00" in task_id
        ):
            raise RemediationError("queue task has an unsafe task_id")
        if task_id in task_ids:
            raise RemediationError(f"queue has duplicate task_id {task_id}")
        if (
            isinstance(source_row_index, bool)
            or not isinstance(source_row_index, int)
            or source_row_index < 0
        ):
            raise RemediationError(f"queue task {task_id} has malformed source_row_index")
        if source_row_index in source_indices:
            raise RemediationError(f"queue has duplicate source_row_index {source_row_index}")
        if not isinstance(row, Mapping):
            raise RemediationError(f"queue task {task_id} has no original_row object")

        critical_codes = _task_string_list(task, "critical_codes")
        warning_codes = _task_string_list(task, "warning_codes")
        overlap_paper_ids = _task_string_list(task, "training_overlap_paper_ids")
        has_training_overlap = _has_training_paper_overlap(critical_codes, overlap_paper_ids)
        task_ids.add(task_id)
        source_indices.add(source_row_index)
        details.append(
            {
                "task_id": task_id,
                "source_row_index": source_row_index,
                "row": row,
                "critical_codes": critical_codes,
                "warning_codes": warning_codes,
                "has_training_overlap": has_training_overlap,
                "required_actions": _required_actions(
                    critical_codes,
                    warning_codes,
                    has_training_overlap=has_training_overlap,
                ),
                "canonical_paper_id": canonical_paper_id(row.get("paper_id")),
                "paper_identity_errors": paper_identity_errors(row),
                "primary_endpoint": row.get("stratum") in primary,
            }
        )
    return details


def _assignment_rows(task_details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(task_details):
        reviewer_slot_1, reviewer_slot_2 = _REVIEWER_PAIRS[index % len(_REVIEWER_PAIRS)]
        has_training_overlap = bool(task["has_training_overlap"])
        rows.append(
            {
                "task_id": task["task_id"],
                "source_row_index": task["source_row_index"],
                "workstream": (
                    "training_overlap_confirmation" if has_training_overlap else "manual_curation"
                ),
                "proposed_disposition": (
                    "exclude_pending_human_attestation"
                    if has_training_overlap
                    else "manual_review_required"
                ),
                "reviewer_slot_1": reviewer_slot_1,
                "reviewer_slot_2": reviewer_slot_2,
                "critical_codes": ";".join(task["critical_codes"]),
                "required_actions": ";".join(task["required_actions"]),
            }
        )
    return rows


def _assignment_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    handle = StringIO(newline="")
    csv_writer = DictWriter(handle, fieldnames=_ASSIGNMENT_FIELDS, lineterminator="\n")
    csv_writer.writeheader()
    csv_writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _group_id(queue_fingerprint: str, paper_id: str, task_ids: Sequence[str]) -> str:
    payload = "\x00".join((queue_fingerprint, paper_id, *task_ids)).encode("utf-8")
    return _sha256_bytes(payload)


def _human_instruction(capacity_role: str) -> str:
    if capacity_role == "closed_training_overlap":
        return (
            "Подтвердить пересечение с training независимой ручной проверкой; группа закрыта для "
            "capacity до человеческого решения. Не заполнять решение, attestation или reviewer ID "
            "автоматически."
        )
    if capacity_role == "required_candidate_for_exact_150":
        return (
            "Проверить группу вручную как возможного кандидата exact-N. Не выбирать retained строку "
            "из duplicate group и не считать группу подтвержденной до человеческой проверки."
        )
    return (
        "Разрешить canonical identity вручную без fuzzy grouping или remapping. До этого группа не "
        "учитывается для capacity и не получает автоматического решения."
    )


def _primary_paper_groups(
    task_details: Sequence[Mapping[str, Any]], queue_fingerprint: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for task in task_details:
        paper_id = str(task["canonical_paper_id"])
        key = ("canonical", paper_id) if paper_id else ("unresolved", str(task["task_id"]))
        grouped.setdefault(key, []).append(task)

    groups: list[dict[str, Any]] = []
    for members in grouped.values():
        paper_id = str(members[0]["canonical_paper_id"])
        primary_members = [member for member in members if member["primary_endpoint"]]
        if not primary_members:
            continue
        task_ids = [str(member["task_id"]) for member in members]
        primary_task_ids = [str(member["task_id"]) for member in primary_members]
        training_overlap_closed = any(bool(member["has_training_overlap"]) for member in members)
        identity_clean = bool(paper_id) and all(
            not member["paper_identity_errors"] for member in members
        )
        if training_overlap_closed:
            capacity_role = "closed_training_overlap"
        elif identity_clean:
            capacity_role = "required_candidate_for_exact_150"
        else:
            capacity_role = "unresolved_not_counted"
        groups.append(
            {
                "artifact_version": CAPACITY_PLAN_ARTIFACT_VERSION,
                "queue_fingerprint": queue_fingerprint,
                "group_id": _group_id(queue_fingerprint, paper_id, task_ids),
                "canonical_paper_id": paper_id,
                "task_ids": task_ids,
                "primary_task_ids": primary_task_ids,
                "task_count": len(task_ids),
                "training_overlap_closed": training_overlap_closed,
                "identity_status": "clean" if identity_clean else "unresolved",
                "capacity_role": capacity_role,
                "human_instruction": _human_instruction(capacity_role),
            }
        )
    return groups


def _slot_load_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(row[field]) for row in rows for field in ("reviewer_slot_1", "reviewer_slot_2")
    )
    return {slot: counts[slot] for slot in _REVIEWER_SLOTS}


def _summary_markdown(
    *,
    queue_fingerprint: str,
    task_count: int,
    exact_n: int,
    primary_strata: Sequence[str],
    group_counts: Mapping[str, int],
    capacity_exact_target_available: bool,
    capacity_exact_remediation_pool_available: bool,
) -> bytes:
    capacity_candidates = group_counts["required_candidate_for_exact_150"]
    remediation_pool = capacity_candidates + group_counts["unresolved_not_counted"]
    availability = "true" if capacity_exact_target_available else "false"
    remediation_availability = (
        "true" if capacity_exact_remediation_pool_available else "false"
    )
    result = (
        "Количество clean capacity candidates равно exact N; это только счетчик маршрутизации, а не "
        "подтверждение paper или готовности к strict plan."
        if capacity_exact_target_available
        else "Количество clean capacity candidates не равно exact N; автоматического пути к plan нет."
    )
    lines = [
        "# Phase 0: план capacity review",
        "",
        "Этот standalone-пакет связан с immutable queue и предназначен только для ручного "
        "планирования review. Он не изменяет queue, формы или черновики.",
        "",
        "## Exact-N и результат capacity",
        "",
        f"- Queue fingerprint: `{queue_fingerprint}`",
        f"- Задач в immutable queue: {task_count}",
        f"- Точное требование: N={exact_n}",
        "- Primary membership выводится только из configured strata: "
        f"{', '.join(primary_strata)}",
        f"- Clean capacity candidates: {capacity_candidates}",
        f"- capacity_exact_target_available: `{availability}`",
        f"- Clean + identity-remediation pool: {remediation_pool}",
        f"- capacity_exact_remediation_pool_available: `{remediation_availability}`",
        f"- {result}",
        "",
        f"Для достаточности по capacity число clean capacity candidates должно быть равно ровно "
        f"N={exact_n}; большее или меньшее число не является достаточным.",
        "Identity-remediation pool разрешает только подготовку proposals для human review и не "
        "меняет clean capacity count до подтвержденного corrected release.",
        "",
        "## Группы primary paper",
        "",
        f"- Всего primary paper groups: {group_counts['primary_paper_groups']}",
        f"- `required_candidate_for_exact_150`: {capacity_candidates}",
        f"- `closed_training_overlap`: {group_counts['closed_training_overlap']}",
        f"- `unresolved_not_counted`: {group_counts['unresolved_not_counted']}",
        f"- Identity `clean`: {group_counts['identity_clean']}",
        f"- Identity `unresolved`: {group_counts['identity_unresolved']}",
        "",
        "`clean` означает только отсутствие ошибок canonical identity в metadata queue; это не "
        "утверждение о фактической проверке paper.",
        "",
        "## Обязательная ручная работа по группам",
        "",
        "- Для `closed_training_overlap` люди подтверждают overlap; группа не учитывается для "
        "capacity до человеческого решения.",
        "- Для `required_candidate_for_exact_150` люди проверяют группу как кандидата, но не выбирают "
        "retained row автоматически, включая duplicate rows.",
        "- Для `unresolved_not_counted` люди разрешают identity без fuzzy grouping или remapping; до "
        "этого группа не считается capacity candidate.",
        "",
        "Пакет не выполняет automatic retain, не выбирает duplicate row, не remap-ит paper ID и не "
        "заполняет reviewer ID, attestation или decision. Reviewer slots в CSV являются только "
        "нейтральными placeholders, а не идентификаторами людей.",
    ]
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


def _capacity_plan_artifacts(
    task_details: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
    exact_n: int,
    primary_strata: Sequence[str],
) -> tuple[dict[str, bytes], dict[str, Any], list[dict[str, Any]]]:
    assignments = _assignment_rows(task_details)
    groups = _primary_paper_groups(task_details, queue_fingerprint)
    role_counts = Counter(group["capacity_role"] for group in groups)
    identity_counts = Counter(group["identity_status"] for group in groups)
    group_counts = {
        "primary_paper_groups": len(groups),
        "required_candidate_for_exact_150": role_counts["required_candidate_for_exact_150"],
        "closed_training_overlap": role_counts["closed_training_overlap"],
        "unresolved_not_counted": role_counts["unresolved_not_counted"],
        "identity_clean": identity_counts["clean"],
        "identity_unresolved": identity_counts["unresolved"],
    }
    capacity_exact_target_available = group_counts["required_candidate_for_exact_150"] == exact_n
    capacity_remediation_pool_count = (
        group_counts["required_candidate_for_exact_150"]
        + group_counts["unresolved_not_counted"]
    )
    capacity_exact_remediation_pool_available = capacity_remediation_pool_count == exact_n
    payload_files = {
        REVIEW_ASSIGNMENT_PLAN_FILENAME: _assignment_csv(assignments),
        CAPACITY_PAPER_GROUPS_FILENAME: _jsonl_bytes(groups),
        CAPACITY_PLAN_SUMMARY_FILENAME: _summary_markdown(
            queue_fingerprint=queue_fingerprint,
            task_count=len(assignments),
            exact_n=exact_n,
            primary_strata=primary_strata,
            group_counts=group_counts,
            capacity_exact_target_available=capacity_exact_target_available,
            capacity_exact_remediation_pool_available=(
                capacity_exact_remediation_pool_available
            ),
        ),
    }
    manifest = {
        "artifact_version": CAPACITY_PLAN_ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "exact_n": exact_n,
        "task_count": len(assignments),
        "capacity_candidate_group_count": group_counts["required_candidate_for_exact_150"],
        "capacity_exact_target_available": capacity_exact_target_available,
        "capacity_remediation_pool_count": capacity_remediation_pool_count,
        "capacity_exact_remediation_pool_available": (
            capacity_exact_remediation_pool_available
        ),
        "group_counts": group_counts,
        "reviewer_slot_load_counts": _slot_load_counts(assignments),
        "file_count": len(payload_files),
        "file_counts": {
            REVIEW_ASSIGNMENT_PLAN_FILENAME: len(assignments),
            CAPACITY_PAPER_GROUPS_FILENAME: len(groups),
            CAPACITY_PLAN_SUMMARY_FILENAME: 1,
        },
        "policy": {
            "require_exact_n_items": True,
            "training_overlap_closes_group": True,
            "primary_endpoint_rule": "stratum_in_configured_primary_strata",
            "primary_strata": list(primary_strata),
            "identity_clean_rule": (
                "canonical_paper_id_nonempty_and_all_group_rows_have_no_paper_identity_errors"
            ),
            "fuzzy_grouping": False,
            "automatic_retain": False,
            "automatic_duplicate_row_selection": False,
            "automatic_paper_remapping": False,
            "reviewer_identity_prefill": False,
            "attestation_prefill": False,
            "decision_prefill": False,
            "paper_factual_verification_claimed": False,
            "reviewer_slots": list(_REVIEWER_SLOTS),
        },
        "files": _file_inventory(payload_files),
    }
    return (
        {
            **payload_files,
            CAPACITY_PLAN_MANIFEST_FILENAME: _json_bytes(manifest, newline=True),
        },
        manifest,
        groups,
    )


def _publish_workspace(target: Path, files: Mapping[str, bytes]) -> None:
    staging = target.with_name(f".{target.name}.capacity-plan.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise RemediationError(f"capacity plan staging path already exists: {staging}")
    try:
        staging.mkdir()
        for name, data in sorted(files.items()):
            _atomic_write(staging / name, data)
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink() and _trees_identical(staging, target):
                return
            raise RemediationError(
                "output_dir already contains a different capacity plan workspace"
            )
        try:
            os.replace(staging, target)
        except OSError as exc:
            if target.is_dir() and not target.is_symlink() and _trees_identical(staging, target):
                return
            raise RemediationError(
                f"cannot publish capacity plan workspace atomically: {exc}"
            ) from exc
    finally:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)


def generate_capacity_plan(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    queue_manifest: str | Path,
    output_dir: str | Path,
    *,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
) -> dict[str, Any]:
    """Publish a deterministic Phase 0 plan without changing curation state."""

    exact_n = _require_capacity_remediation_config(config)
    material = _queue_material(config, prepare_manifest, benchmark_schema, provenance_schema)
    queue_root, verified_queue = _verify_queue_workspace(material, queue_manifest)
    queue_fingerprint = verified_queue["queue_fingerprint"]
    primary_strata = verified_queue["policy"]["primary_strata"]
    task_details = _task_details(material["tasks"], primary_strata)
    files, manifest, _ = _capacity_plan_artifacts(
        task_details,
        queue_fingerprint,
        exact_n,
        primary_strata,
    )
    protected_directories, protected_files = _prepare_input_protection(material)
    target = _separate_output_target(
        output_dir,
        protected_directories=(queue_root, *protected_directories),
        protected_files=protected_files,
    )
    _publish_workspace(target, files)
    return manifest


__all__ = ["generate_capacity_plan"]
