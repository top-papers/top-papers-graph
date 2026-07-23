# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build fail-closed machine-assistance dossiers for exact-N capacity curation."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from csv import writer
from io import StringIO
from pathlib import Path
from typing import Any

from .capacity_plan import (
    CAPACITY_PAPER_GROUPS_FILENAME,
    CAPACITY_PLAN_ARTIFACT_VERSION,
    CAPACITY_PLAN_MANIFEST_FILENAME,
    CAPACITY_PLAN_SUMMARY_FILENAME,
    REVIEW_ASSIGNMENT_PLAN_FILENAME,
    _capacity_plan_artifacts,
    _primary_paper_groups,
    _require_capacity_remediation_config,
    _task_details,
)
from .remediation import (
    RemediationError,
    _atomic_write,
    _decision_binding,
    _json_bytes,
    _jsonl_bytes,
    _manifest_workspace_roots,
    _prepare_input_protection,
    _queue_material,
    _read_stable_bytes,
    _sha256_bytes,
    _separate_output_target,
    _strict_json,
    _strict_jsonl_bytes,
    _trees_identical,
    _verify_queue_workspace,
)


CAPACITY_ASSIST_ARTIFACT_VERSION = 2
CAPACITY_CANDIDATE_DOSSIERS_FILENAME = "capacity_candidate_dossiers.jsonl"
CAPACITY_ENRICHMENT_TEMPLATE_FILENAME = "capacity_enrichment_template.jsonl"
CAPACITY_HUMAN_REVIEW_FILENAME = "capacity_human_review_checklist.csv"
CAPACITY_ASSIST_SUMMARY_FILENAME = "capacity_assist_summary.md"
CAPACITY_ASSIST_MANIFEST_FILENAME = "capacity_assist_manifest.json"

_CLAIM_NAMES = (
    "paper_identity",
    "scientific_prompt",
    "image_provenance",
    "usage_rights",
    "paper_holdout",
    "source_holdout",
    "creator_holdout",
    "training_overlap",
)
_REQUIRED_ENRICHMENT_FIELDS = (
    "selected_task_ids",
    "benchmark_rows",
    "provenance_rows",
    "external_evidence",
    "claim_evidence",
    "human_review",
    "release_eligibility",
)


def _legacy_source_hints(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    task_id = task.get("task_id")
    rows = task.get("legacy_provenance_rows")
    if not isinstance(rows, list):
        raise RemediationError(f"queue task {task_id!r} has malformed legacy provenance rows")
    hints_by_bytes: dict[bytes, dict[str, Any]] = {}
    for entry in rows:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("row"), Mapping):
            raise RemediationError(f"queue task {task_id!r} has malformed legacy provenance entry")
        source_row = entry["row"]
        hint: dict[str, Any] = {
            "legacy_metadata_sha256": _sha256_bytes(_json_bytes(source_row)),
            "legacy_free_text_included": False,
        }
        source_index = entry.get("source_provenance_row_index")
        source_sha256 = entry.get("source_provenance_row_sha256")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise RemediationError(
                f"queue task {task_id!r} has malformed legacy provenance index"
            )
        if not isinstance(source_sha256, str):
            raise RemediationError(
                f"queue task {task_id!r} has malformed legacy provenance SHA256"
            )
        hint["source_provenance_row_index"] = source_index
        hint["source_provenance_row_sha256"] = source_sha256
        hint["evidence_status"] = "unverified_legacy_hint"
        encoded = _json_bytes(hint)
        hints_by_bytes[encoded] = hint
    return [hints_by_bytes[key] for key in sorted(hints_by_bytes)]


def _source_row_hint(task: Mapping[str, Any]) -> dict[str, Any]:
    task_id = task.get("task_id")
    row = task.get("original_row")
    if not isinstance(row, Mapping):
        raise RemediationError(f"queue task {task_id!r} has no original row")
    stratum = row.get("stratum")
    hint: dict[str, Any] = {
        "source_sample_id_sha256": (
            _sha256_bytes(row["sample_id"].encode("utf-8"))
            if isinstance(row.get("sample_id"), str)
            else None
        ),
        "source_paper_id_sha256": (
            _sha256_bytes(row["paper_id"].encode("utf-8"))
            if isinstance(row.get("paper_id"), str)
            else None
        ),
        "stratum": (
            stratum
            if stratum in {"multimodal_hard", "temporal_hard", "easy_control"}
            else ""
        ),
        "primary_endpoint": row.get("primary_endpoint") is True,
        "evidence_status": "unverified_legacy_hint",
    }
    prompt = row.get("model_task_prompt")
    hint["legacy_prompt_sha256"] = (
        _sha256_bytes(prompt.encode("utf-8")) if isinstance(prompt, str) else None
    )
    hint["legacy_prompt_text_included"] = False
    return hint


def _retrieval_url(canonical_id: str) -> str | None:
    if canonical_id.startswith("doi:"):
        return f"https://doi.org/{canonical_id.removeprefix('doi:')}"
    if canonical_id.startswith("arxiv:"):
        return f"https://arxiv.org/abs/{canonical_id.removeprefix('arxiv:')}"
    return None


def _verified_capacity_plan(
    capacity_plan_manifest: str | Path,
    *,
    expected_groups: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
    exact_n: int,
    task_details: Sequence[Mapping[str, Any]],
    primary_strata: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    manifest_path = Path(capacity_plan_manifest)
    manifest, manifest_bytes = _strict_json(manifest_path, "capacity plan manifest")
    root = manifest_path.parent
    expected_names = {
        REVIEW_ASSIGNMENT_PLAN_FILENAME,
        CAPACITY_PAPER_GROUPS_FILENAME,
        CAPACITY_PLAN_SUMMARY_FILENAME,
        CAPACITY_PLAN_MANIFEST_FILENAME,
    }
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise RemediationError(f"cannot inspect capacity plan workspace: {exc}") from exc
    if {entry.name for entry in entries} != expected_names or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise RemediationError("capacity plan workspace has an unexpected file inventory")
    expected_files, expected_manifest, reconstructed_groups = _capacity_plan_artifacts(
        task_details,
        queue_fingerprint,
        exact_n,
        primary_strata,
    )
    if reconstructed_groups != list(expected_groups):
        raise RemediationError("capacity plan reconstruction differs from expected routing")
    if (
        manifest != expected_manifest
        or manifest_bytes != expected_files[CAPACITY_PLAN_MANIFEST_FILENAME]
    ):
        raise RemediationError("capacity plan manifest differs from deterministic reconstruction")
    if (
        type(manifest.get("artifact_version")) is not int
        or manifest["artifact_version"] != CAPACITY_PLAN_ARTIFACT_VERSION
        or manifest.get("queue_fingerprint") != queue_fingerprint
        or manifest.get("exact_n") != exact_n
        or manifest.get("capacity_remediation_pool_count") != exact_n
        or manifest.get("capacity_exact_remediation_pool_available") is not True
        or manifest.get("file_count") != 3
    ):
        raise RemediationError("capacity plan manifest does not bind the exact candidate routing")
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(field) is not False
        for field in (
            "automatic_retain",
            "automatic_duplicate_row_selection",
            "automatic_paper_remapping",
            "reviewer_identity_prefill",
            "attestation_prefill",
            "decision_prefill",
            "paper_factual_verification_claimed",
        )
    ):
        raise RemediationError("capacity plan manifest has an unsafe routing policy")
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or len(inventory) != 3:
        raise RemediationError("capacity plan manifest file inventory is malformed")
    inventory_by_path: dict[str, Mapping[str, Any]] = {}
    for record in inventory:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise RemediationError("capacity plan manifest file inventory is malformed")
        path = record["path"]
        if path in inventory_by_path:
            raise RemediationError("capacity plan manifest file inventory contains duplicates")
        inventory_by_path[path] = record
    payload_names = expected_names - {CAPACITY_PLAN_MANIFEST_FILENAME}
    if set(inventory_by_path) != payload_names:
        raise RemediationError("capacity plan manifest file inventory is incomplete")
    payload_bytes: dict[str, bytes] = {}
    for name in sorted(payload_names):
        data = _read_stable_bytes(root / name, f"capacity plan {name}")
        payload_bytes[name] = data
        if data != expected_files[name]:
            raise RemediationError(
                f"capacity plan file differs from deterministic reconstruction: {name}"
            )
        record = inventory_by_path[name]
        if record.get("sha256") != _sha256_bytes(data) or record.get("size_bytes") != len(data):
            raise RemediationError(f"capacity plan file differs from its manifest: {name}")
    groups = _strict_jsonl_bytes(
        payload_bytes[CAPACITY_PAPER_GROUPS_FILENAME],
        "capacity paper groups JSONL",
    )
    if groups != list(expected_groups):
        raise RemediationError("capacity plan routing differs from the current verified queue")
    return groups, _sha256_bytes(manifest_bytes)


def _capacity_assist_candidates(
    groups: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        group
        for group in groups
        if group["capacity_role"]
        in {"required_candidate_for_exact_150", "unresolved_not_counted"}
    ]


def _separate_workspace_target(
    output_dir: str | Path,
    queue_root: Path,
    *protected_workspaces: str | Path,
    protected_directories: Sequence[str | Path] = (),
    protected_files: Sequence[str | Path] = (),
) -> Path:
    roots = [queue_root, *protected_directories]
    for protected in protected_workspaces:
        roots.extend(_manifest_workspace_roots(protected))
    return _separate_output_target(
        output_dir,
        protected_directories=roots,
        protected_files=protected_files,
    )


def _image_hints(task: Mapping[str, Any], critical_codes: Sequence[str]) -> list[dict[str, Any]]:
    task_id = task.get("task_id")
    records = task.get("audited_image_hashes")
    if not isinstance(records, list):
        raise RemediationError(f"queue task {task_id!r} has malformed audited image hashes")
    quarantined_for_reuse = "cross_paper_image_reuse" in critical_codes
    result: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RemediationError(f"queue task {task_id!r} has malformed audited image entry")
        path = record.get("path")
        sha256 = record.get("sha256")
        size_bytes = record.get("size_bytes")
        if (
            not isinstance(path, str)
            or not isinstance(sha256, str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise RemediationError(f"queue task {task_id!r} has malformed audited image entry")
        result.append(
            {
                "path": path,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "release_usable": False,
                "evidence_status": (
                    "quarantined_cross_paper_reuse"
                    if quarantined_for_reuse
                    else "pending_external_source_verification"
                ),
            }
        )
    return result


def _task_dossier(
    task: Mapping[str, Any], details: Mapping[str, Any], queue_fingerprint: str
) -> dict[str, Any]:
    if task.get("task_id") != details.get("task_id"):
        raise RemediationError("capacity assist task details do not match queue task")
    critical_codes = list(details["critical_codes"])
    return {
        "binding": _decision_binding(task, queue_fingerprint),
        "canonical_paper_id": details["canonical_paper_id"],
        "machine_retrieval_url": _retrieval_url(str(details["canonical_paper_id"])),
        "source_row_hint": _source_row_hint(task),
        "legacy_source_hints": _legacy_source_hints(task),
        "audited_image_hints": _image_hints(task, critical_codes),
        "critical_codes": critical_codes,
        "warning_codes": list(details["warning_codes"]),
        "required_actions": list(details["required_actions"]),
        "machine_status": "pending_external_enrichment",
    }


def _candidate_dossiers(
    groups: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    task_details: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
    capacity_plan_manifest_sha256: str,
) -> list[dict[str, Any]]:
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    details_by_id = {str(details["task_id"]): details for details in task_details}
    dossiers: list[dict[str, Any]] = []
    for group in groups:
        member_ids = list(group["task_ids"])
        if any(task_id not in tasks_by_id or task_id not in details_by_id for task_id in member_ids):
            raise RemediationError("capacity assist group references an unknown queue task")
        dossiers.append(
            {
                "artifact_version": CAPACITY_ASSIST_ARTIFACT_VERSION,
                "queue_fingerprint": queue_fingerprint,
                "capacity_plan_manifest_sha256": capacity_plan_manifest_sha256,
                "group_id": group["group_id"],
                "canonical_paper_id": group["canonical_paper_id"],
                "capacity_role": group["capacity_role"],
                "identity_status": group["identity_status"],
                "task_ids": member_ids,
                "primary_task_ids": list(group["primary_task_ids"]),
                "machine_selected_task_ids": [],
                "candidate_status": "blocked_pending_enrichment_and_human_attestation",
                "tasks": [
                    _task_dossier(tasks_by_id[task_id], details_by_id[task_id], queue_fingerprint)
                    for task_id in member_ids
                ],
                "human_review": {
                    "curator_ids": [],
                    "independent_attestation": False,
                },
            }
        )
    return dossiers


def _enrichment_templates(
    groups: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
    capacity_plan_manifest_sha256: str,
) -> list[dict]:
    return [
        {
            "artifact_version": CAPACITY_ASSIST_ARTIFACT_VERSION,
            "queue_fingerprint": queue_fingerprint,
            "capacity_plan_manifest_sha256": capacity_plan_manifest_sha256,
            "group_id": group["group_id"],
            "canonical_paper_id": group["canonical_paper_id"],
            "task_ids": list(group["task_ids"]),
            "selected_task_ids": [],
            "benchmark_rows": {},
            "provenance_rows": {},
            "external_evidence": [],
            "claim_evidence": {
                name: {"status": "pending", "evidence_ids": []} for name in _CLAIM_NAMES
            },
            "human_review": {
                "curator_ids": [],
                "independent_attestation": False,
            },
            "release_eligibility": "blocked",
        }
        for group in groups
    ]


def _review_checklist(groups: Sequence[Mapping[str, Any]]) -> bytes:
    handle = StringIO(newline="")
    csv_writer = writer(handle, lineterminator="\n")
    csv_writer.writerow(
        [
            "group_id",
            "canonical_paper_id",
            "task_ids",
            "curator_1_id",
            "curator_2_id",
            "paper_identity_confirmed",
            "prompt_grounding_confirmed",
            "image_source_confirmed",
            "usage_rights_confirmed",
            "training_holdout_confirmed",
            "final_notes",
        ]
    )
    for group in groups:
        csv_writer.writerow(
            [
                group["group_id"],
                group["canonical_paper_id"],
                ";".join(group["task_ids"]),
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


def _summary(
    *,
    queue_fingerprint: str,
    exact_n: int,
    group_count: int,
    clean_group_count: int,
    identity_remediation_group_count: int,
    task_count: int,
) -> bytes:
    lines = [
        "# Machine-assisted capacity curation",
        "",
        f"- Queue fingerprint: `{queue_fingerprint}`",
        f"- Exact target: N={exact_n}",
        f"- Candidate dossiers: {group_count}",
        f"- Clean candidate groups: {clean_group_count}",
        f"- Identity-remediation groups: {identity_remediation_group_count}",
        f"- Queue tasks represented in dossiers: {task_count}",
        "",
        "Пакет автоматически собирает только проверяемые bindings, source hints, audit findings и "
        "SHA256. Legacy prompts не включаются, а legacy images не объявляются пригодными для release.",
        "",
        "До внешнего enrichment и короткой проверки двумя реальными кураторами каждый кандидат "
        "остается blocked. Пакет не выбирает duplicate rows, не создает retain decisions, не "
        "заполняет human IDs и не подтверждает license, provenance или training holdout.",
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


def _capacity_assist_artifacts(
    candidates: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    task_details: Sequence[Mapping[str, Any]],
    queue_fingerprint: str,
    capacity_plan_manifest_sha256: str,
    exact_n: int,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    dossiers = _candidate_dossiers(
        candidates,
        tasks,
        task_details,
        queue_fingerprint,
        capacity_plan_manifest_sha256,
    )
    enrichment_templates = _enrichment_templates(
        candidates,
        queue_fingerprint,
        capacity_plan_manifest_sha256,
    )
    clean_group_count = sum(
        group["capacity_role"] == "required_candidate_for_exact_150" for group in candidates
    )
    identity_remediation_group_count = sum(
        group["capacity_role"] == "unresolved_not_counted" for group in candidates
    )
    represented_task_count = sum(len(group["task_ids"]) for group in candidates)
    payload_files = {
        CAPACITY_CANDIDATE_DOSSIERS_FILENAME: _jsonl_bytes(dossiers),
        CAPACITY_ENRICHMENT_TEMPLATE_FILENAME: _jsonl_bytes(enrichment_templates),
        CAPACITY_HUMAN_REVIEW_FILENAME: _review_checklist(candidates),
        CAPACITY_ASSIST_SUMMARY_FILENAME: _summary(
            queue_fingerprint=queue_fingerprint,
            exact_n=exact_n,
            group_count=len(candidates),
            clean_group_count=clean_group_count,
            identity_remediation_group_count=identity_remediation_group_count,
            task_count=represented_task_count,
        ),
    }
    manifest = {
        "artifact_version": CAPACITY_ASSIST_ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "capacity_plan_manifest_sha256": capacity_plan_manifest_sha256,
        "exact_n": exact_n,
        "candidate_group_count": len(candidates),
        "clean_candidate_group_count": clean_group_count,
        "identity_remediation_group_count": identity_remediation_group_count,
        "represented_task_count": represented_task_count,
        "required_enrichment_fields": list(_REQUIRED_ENRICHMENT_FIELDS),
        "policy": {
            "candidate_groups_from_verified_capacity_routing": True,
            "external_enrichment_performed": False,
            "automatic_task_selection": False,
            "automatic_retain": False,
            "automatic_duplicate_row_selection": False,
            "automatic_paper_remapping": False,
            "identity_remediation_proposals_allowed": True,
            "legacy_prompt_text_included": False,
            "legacy_images_release_usable": False,
            "reviewer_identity_prefill": False,
            "attestation_prefill": False,
            "paper_factual_verification_claimed": False,
        },
        "file_count": len(payload_files),
        "file_counts": {
            CAPACITY_CANDIDATE_DOSSIERS_FILENAME: len(dossiers),
            CAPACITY_ENRICHMENT_TEMPLATE_FILENAME: len(enrichment_templates),
            CAPACITY_HUMAN_REVIEW_FILENAME: len(candidates),
            CAPACITY_ASSIST_SUMMARY_FILENAME: 1,
        },
        "files": _file_inventory(payload_files),
    }
    return {
        **payload_files,
        CAPACITY_ASSIST_MANIFEST_FILENAME: _json_bytes(manifest, newline=True),
    }, manifest


def _publish_workspace(target: Path, files: Mapping[str, bytes]) -> None:
    staging = target.with_name(f".{target.name}.capacity-assist.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise RemediationError(f"capacity assist staging path already exists: {staging}")
    try:
        staging.mkdir()
        for name, data in sorted(files.items()):
            _atomic_write(staging / name, data)
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink() and _trees_identical(staging, target):
                return
            raise RemediationError(
                "output_dir already contains a different capacity assist workspace"
            )
        try:
            os.replace(staging, target)
        except OSError as exc:
            if target.is_dir() and not target.is_symlink() and _trees_identical(staging, target):
                return
            raise RemediationError(
                f"cannot publish capacity assist workspace atomically: {exc}"
            ) from exc
    finally:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)


def generate_capacity_assist_package(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    queue_manifest: str | Path,
    capacity_plan_manifest: str | Path,
    output_dir: str | Path,
    *,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
) -> dict[str, Any]:
    """Publish exact-N evidence dossiers without making scientific or human claims."""

    exact_n = _require_capacity_remediation_config(config, command="curate-capacity-assist")
    material = _queue_material(config, prepare_manifest, benchmark_schema, provenance_schema)
    queue_root, verified_queue = _verify_queue_workspace(material, queue_manifest)
    queue_fingerprint = verified_queue["queue_fingerprint"]
    primary_strata = verified_queue["policy"]["primary_strata"]
    task_details = _task_details(material["tasks"], primary_strata)
    expected_groups = _primary_paper_groups(task_details, queue_fingerprint)
    expected_candidates = _capacity_assist_candidates(expected_groups)
    if len(expected_candidates) != exact_n:
        raise RemediationError(
            "curate-capacity-assist requires exactly "
            f"{exact_n} clean-or-identity-remediation candidate groups; "
            f"found {len(expected_candidates)}"
        )
    groups, capacity_plan_manifest_sha256 = _verified_capacity_plan(
        capacity_plan_manifest,
        expected_groups=expected_groups,
        queue_fingerprint=queue_fingerprint,
        exact_n=exact_n,
        task_details=task_details,
        primary_strata=primary_strata,
    )
    candidates = _capacity_assist_candidates(groups)
    if candidates != expected_candidates:
        raise RemediationError("capacity plan candidate routing differs from the verified queue")

    files, manifest = _capacity_assist_artifacts(
        candidates,
        material["tasks"],
        task_details,
        queue_fingerprint,
        capacity_plan_manifest_sha256,
        exact_n,
    )
    protected_directories, protected_files = _prepare_input_protection(material)
    target = _separate_workspace_target(
        output_dir,
        queue_root,
        capacity_plan_manifest,
        protected_directories=protected_directories,
        protected_files=protected_files,
    )
    _publish_workspace(target, files)
    return manifest


__all__ = ["generate_capacity_assist_package"]
