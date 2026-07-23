# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate and publish machine-enriched capacity proposals for short human review."""

from __future__ import annotations

import copy
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .audit import canonical_paper_id
from .capacity_assist import (
    CAPACITY_ASSIST_ARTIFACT_VERSION,
    CAPACITY_ASSIST_MANIFEST_FILENAME,
    CAPACITY_ASSIST_SUMMARY_FILENAME,
    CAPACITY_CANDIDATE_DOSSIERS_FILENAME,
    CAPACITY_ENRICHMENT_TEMPLATE_FILENAME,
    CAPACITY_HUMAN_REVIEW_FILENAME,
    _capacity_assist_artifacts,
    _capacity_assist_candidates,
    _separate_workspace_target,
    _verified_capacity_plan,
)
from .capacity_plan import (
    CAPACITY_PLAN_MANIFEST_FILENAME,
    _capacity_plan_artifacts,
    _primary_paper_groups,
    _require_capacity_remediation_config,
    _task_details,
)
from .curator import CURATOR_ARTIFACT_VERSION, _blank_edit
from .remediation import (
    RemediationError,
    _atomic_write,
    _credential_free_http_url,
    _json_bytes,
    _jsonl_bytes,
    _prepare_input_protection,
    _queue_material,
    _read_stable_bytes,
    _safe_asset_reference,
    _schema_error,
    _sha256_bytes,
    _strict_json,
    _strict_jsonl,
    _strict_jsonl_bytes,
    _trees_identical,
    _validated_paper_id,
    _verify_queue_workspace,
)


CAPACITY_ENRICHMENT_ARTIFACT_VERSION = 2
MACHINE_ENRICHMENT_FILENAME = "machine_enrichment.jsonl"
MACHINE_ASSISTED_DRAFT_FILENAME = "machine_assisted_draft.json"
MACHINE_ENRICHMENT_SUMMARY_FILENAME = "machine_enrichment_summary.md"
MACHINE_ENRICHMENT_MANIFEST_FILENAME = "machine_enrichment_manifest.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_EVIDENCE_KINDS = frozenset({"article", "figure", "license", "metadata", "training_lineage"})
_CLAIM_NAMES = frozenset(
    {
        "paper_identity",
        "scientific_prompt",
        "image_provenance",
        "usage_rights",
        "paper_holdout",
        "source_holdout",
        "creator_holdout",
        "training_overlap",
    }
)
_CLAIM_EVIDENCE_KINDS = {
    "paper_identity": frozenset({"article", "metadata"}),
    "scientific_prompt": frozenset({"article", "figure"}),
    "image_provenance": frozenset({"figure"}),
    "usage_rights": frozenset({"license"}),
    "paper_holdout": frozenset({"training_lineage"}),
    "source_holdout": frozenset({"training_lineage"}),
    "creator_holdout": frozenset({"training_lineage"}),
    "training_overlap": frozenset({"training_lineage"}),
}
_ENRICHMENT_FIELDS = frozenset(
    {
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
)
_EVIDENCE_FIELDS = frozenset(
    {
        "evidence_id",
        "kind",
        "source_url",
        "content_sha256",
        "asserted_license",
        "retrieved_at",
        "notes",
    }
)
_ASSIST_FILENAMES = frozenset(
    {
        CAPACITY_CANDIDATE_DOSSIERS_FILENAME,
        CAPACITY_ENRICHMENT_TEMPLATE_FILENAME,
        CAPACITY_HUMAN_REVIEW_FILENAME,
        CAPACITY_ASSIST_SUMMARY_FILENAME,
        CAPACITY_ASSIST_MANIFEST_FILENAME,
    }
)


def _https_url(value: Any, label: str) -> str:
    return _credential_free_http_url(value, label)


def _verified_assist_manifest(
    assist_manifest_path: str | Path,
    *,
    queue_fingerprint: str,
    capacity_plan_manifest_sha256: str,
    candidate_groups: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    task_details: Sequence[Mapping[str, Any]],
    exact_n: int,
) -> str:
    path = Path(assist_manifest_path)
    manifest, raw = _strict_json(path, "capacity assist manifest")
    root = path.parent
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise RemediationError(f"cannot inspect capacity assist workspace: {exc}") from exc
    if {entry.name for entry in entries} != _ASSIST_FILENAMES or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise RemediationError("capacity assist workspace has an unexpected file inventory")
    expected_files, expected_manifest = _capacity_assist_artifacts(
        candidate_groups,
        tasks,
        task_details,
        queue_fingerprint,
        capacity_plan_manifest_sha256,
        exact_n,
    )
    if manifest != expected_manifest or raw != expected_files[CAPACITY_ASSIST_MANIFEST_FILENAME]:
        raise RemediationError("capacity assist manifest differs from deterministic reconstruction")
    if (
        type(manifest.get("artifact_version")) is not int
        or manifest["artifact_version"] != CAPACITY_ASSIST_ARTIFACT_VERSION
        or manifest.get("queue_fingerprint") != queue_fingerprint
        or manifest.get("capacity_plan_manifest_sha256") != capacity_plan_manifest_sha256
        or manifest.get("candidate_group_count") != len(candidate_groups)
        or manifest.get("exact_n") != len(candidate_groups)
        or manifest.get("file_count") != 4
    ):
        raise RemediationError("capacity assist manifest does not bind the expected candidates")
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(field) is not False
        for field in (
            "external_enrichment_performed",
            "automatic_task_selection",
            "automatic_retain",
            "automatic_duplicate_row_selection",
            "automatic_paper_remapping",
            "legacy_prompt_text_included",
            "legacy_images_release_usable",
            "reviewer_identity_prefill",
            "attestation_prefill",
            "paper_factual_verification_claimed",
        )
    ):
        raise RemediationError("capacity assist manifest has an unsafe policy")
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise RemediationError("capacity assist manifest file inventory is malformed")
    inventory_by_path: dict[str, Mapping[str, Any]] = {}
    for record in inventory:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise RemediationError("capacity assist manifest file inventory is malformed")
        name = record["path"]
        if name in inventory_by_path:
            raise RemediationError("capacity assist manifest file inventory contains duplicates")
        inventory_by_path[name] = record
    expected_payloads = _ASSIST_FILENAMES - {CAPACITY_ASSIST_MANIFEST_FILENAME}
    if set(inventory_by_path) != expected_payloads:
        raise RemediationError("capacity assist manifest file inventory is incomplete")
    payload_bytes: dict[str, bytes] = {}
    for name in sorted(expected_payloads):
        data = _read_stable_bytes(root / name, f"capacity assist {name}")
        payload_bytes[name] = data
        if data != expected_files[name]:
            raise RemediationError(
                f"capacity assist file differs from deterministic reconstruction: {name}"
            )
        record = inventory_by_path[name]
        if record.get("sha256") != _sha256_bytes(data) or record.get("size_bytes") != len(data):
            raise RemediationError(f"capacity assist file differs from its manifest: {name}")
    templates = _strict_jsonl_bytes(
        payload_bytes[CAPACITY_ENRICHMENT_TEMPLATE_FILENAME],
        "capacity enrichment template JSONL",
    )
    expected_by_group = {str(group["group_id"]): group for group in candidate_groups}
    if len(templates) != len(expected_by_group):
        raise RemediationError("capacity assist template count differs from candidate routing")
    seen: set[str] = set()
    for template in templates:
        group_id = template.get("group_id")
        group = expected_by_group.get(str(group_id))
        if group is None or group_id in seen:
            raise RemediationError("capacity assist templates have unknown or duplicate groups")
        seen.add(str(group_id))
        if (
            template.get("queue_fingerprint") != queue_fingerprint
            or template.get("capacity_plan_manifest_sha256")
            != capacity_plan_manifest_sha256
            or template.get("canonical_paper_id") != group["canonical_paper_id"]
            or template.get("task_ids") != group["task_ids"]
            or template.get("selected_task_ids") != []
            or template.get("benchmark_rows") != {}
            or template.get("provenance_rows") != {}
            or template.get("human_review")
            != {"curator_ids": [], "independent_attestation": False}
            or template.get("release_eligibility") != "blocked"
        ):
            raise RemediationError("capacity assist template bindings or blank state are invalid")
    return _sha256_bytes(raw)


def _validate_evidence(value: Any, group_id: str) -> tuple[list[dict[str, Any]], set[str]]:
    if not isinstance(value, list) or not value:
        raise RemediationError(f"enrichment group {group_id} requires external evidence")
    records: list[dict[str, Any]] = []
    evidence_ids: set[str] = set()
    for index, record in enumerate(value):
        if not isinstance(record, dict) or set(record) != _EVIDENCE_FIELDS:
            raise RemediationError(f"enrichment group {group_id} evidence {index} fields are invalid")
        evidence_id = record.get("evidence_id")
        if not isinstance(evidence_id, str) or not _EVIDENCE_ID_RE.fullmatch(evidence_id):
            raise RemediationError(f"enrichment group {group_id} has an unsafe evidence_id")
        if evidence_id in evidence_ids:
            raise RemediationError(f"enrichment group {group_id} has duplicate evidence_id")
        if record.get("kind") not in _EVIDENCE_KINDS:
            raise RemediationError(f"enrichment group {group_id} has an unsupported evidence kind")
        asserted_license = record.get("asserted_license")
        if record["kind"] == "license":
            if (
                not isinstance(asserted_license, str)
                or not asserted_license.strip()
                or asserted_license != asserted_license.strip()
            ):
                raise RemediationError(
                    f"enrichment group {group_id} license evidence requires asserted_license"
                )
        elif asserted_license is not None:
            raise RemediationError(
                f"enrichment group {group_id} non-license evidence cannot assert a license"
            )
        _https_url(record.get("source_url"), f"enrichment group {group_id} evidence URL")
        if not isinstance(record.get("content_sha256"), str) or not _SHA256_RE.fullmatch(
            record["content_sha256"]
        ):
            raise RemediationError(f"enrichment group {group_id} evidence requires SHA256")
        if not isinstance(record.get("retrieved_at"), str) or not record["retrieved_at"].strip():
            raise RemediationError(f"enrichment group {group_id} evidence requires retrieved_at")
        if not isinstance(record.get("notes"), str):
            raise RemediationError(f"enrichment group {group_id} evidence notes must be a string")
        evidence_ids.add(evidence_id)
        records.append(copy.deepcopy(record))
    return records, evidence_ids


def _validate_claims(
    value: Any,
    evidence_ids: set[str],
    evidence_kinds: Mapping[str, str],
    group_id: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CLAIM_NAMES:
        raise RemediationError(f"enrichment group {group_id} claim evidence is incomplete")
    result: dict[str, Any] = {}
    for name in sorted(_CLAIM_NAMES):
        claim = value[name]
        if not isinstance(claim, dict) or set(claim) != {"status", "evidence_ids"}:
            raise RemediationError(f"enrichment group {group_id} claim {name} is malformed")
        if claim.get("status") != "machine_supported":
            raise RemediationError(
                f"enrichment group {group_id} claim {name} is not machine-supported"
            )
        selected = claim.get("evidence_ids")
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(item, str) for item in selected)
            or len(selected) != len(set(selected))
            or not set(selected).issubset(evidence_ids)
        ):
            raise RemediationError(
                f"enrichment group {group_id} claim {name} has invalid evidence references"
            )
        if any(evidence_kinds[item] not in _CLAIM_EVIDENCE_KINDS[name] for item in selected):
            raise RemediationError(
                f"enrichment group {group_id} claim {name} uses incompatible evidence kinds"
            )
        result[name] = copy.deepcopy(claim)
    return result


def _validate_machine_provenance(
    row: Any,
    benchmark_row: Mapping[str, Any],
    schemas: Mapping[str, Any],
    forbidden_hashes: set[str],
    image_evidence: Sequence[Mapping[str, Any]],
    rights_evidence: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise RemediationError(f"{label} must be an object")
    candidate = copy.deepcopy(row)
    if set(candidate) != {"sample_id", "paper_id", "images"}:
        raise RemediationError(f"{label} fields are invalid")
    images = candidate.get("images")
    if not isinstance(images, list) or not images:
        raise RemediationError(f"{label} requires images")
    validation_copy = copy.deepcopy(candidate)
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise RemediationError(f"{label} image {index} must be an object")
        if set(image) != {
            "image_path",
            "sha256",
            "page",
            "locator",
            "source_url",
            "license",
            "verified_by",
            "citation",
        }:
            raise RemediationError(f"{label} image {index} fields are invalid")
        if image.get("verified_by") != []:
            raise RemediationError(f"{label} image {index} must leave verified_by empty")
        digest = image.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RemediationError(f"{label} image {index} requires SHA256")
        if digest in forbidden_hashes:
            raise RemediationError(f"{label} image {index} reuses quarantined legacy bytes")
        _safe_asset_reference(image.get("image_path"))
        source_url = _https_url(image.get("source_url"), f"{label} image {index} source_url")
        if not any(
            record["content_sha256"] == digest and record["source_url"] == source_url
            for record in image_evidence
        ):
            raise RemediationError(
                f"{label} image {index} has no matching cited figure hash and URL"
            )
        if not any(
            record["content_sha256"] == digest
            and record["asserted_license"] == image.get("license")
            for record in rights_evidence
        ):
            raise RemediationError(
                f"{label} image {index} has no matching cited rights evidence"
            )
        if not isinstance(image.get("citation"), str) or not image["citation"].strip():
            raise RemediationError(f"{label} image {index} requires citation")
        validation_copy["images"][index]["verified_by"] = [
            "machine-schema-placeholder-1",
            "machine-schema-placeholder-2",
        ]
    _schema_error(schemas["provenance_validator"], validation_copy, label)
    if candidate.get("sample_id") != benchmark_row.get("sample_id"):
        raise RemediationError(f"{label} sample_id differs from benchmark proposal")
    if candidate.get("paper_id") != benchmark_row.get("paper_id"):
        raise RemediationError(f"{label} paper_id differs from benchmark proposal")
    benchmark_images = benchmark_row.get("images")
    provenance_paths = [image["image_path"] for image in images]
    if benchmark_images != provenance_paths:
        raise RemediationError(f"{label} image paths differ from benchmark proposal")
    return candidate


def _system_instruction(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    text: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text.append(content)
        elif isinstance(content, list):
            text.extend(
                str(block["text"])
                for block in content
                if isinstance(block, Mapping)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
    return "\n".join(text)


def _proposal_edit(benchmark: Mapping[str, Any], provenance: Mapping[str, Any], group_id: str) -> dict:
    edit = _blank_edit()
    split = benchmark["split_provenance"]
    edit.update(
        {
            "notes": f"Machine proposal from capacity group {group_id}; requires human verification.",
            "sample_id": benchmark["sample_id"],
            "paper_id": benchmark["paper_id"],
            "stratum": benchmark["stratum"],
            "prompt": benchmark["model_task_prompt"],
            "system_instruction": _system_instruction(benchmark["messages"]),
            "source_document_id": split["source_document_id"],
            "creator_group_id": split["creator_group_id"],
            "images": [
                {
                    "image_path": image["image_path"],
                    "sha256": image["sha256"],
                    "page": str(image["page"]),
                    "locator": image["locator"],
                    "source_url": image["source_url"],
                    "license": image["license"],
                    "citation": image["citation"],
                    "verified_by": "",
                }
                for image in provenance["images"]
            ],
        }
    )
    return edit


def _validated_enrichment(
    rows: Sequence[Mapping[str, Any]],
    candidate_groups: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    schemas: Mapping[str, Any],
    queue_fingerprint: str,
    capacity_plan_manifest_sha256: str,
    primary_strata: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary = frozenset(primary_strata)
    groups_by_id = {str(group["group_id"]): group for group in candidate_groups}
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    forbidden_hashes = {
        str(image["sha256"])
        for task in tasks
        if "cross_paper_image_reuse" in task["critical_codes"]
        for image in task["audited_image_hashes"]
    }
    if len(rows) != len(groups_by_id):
        raise RemediationError("machine enrichment must contain exactly one row per candidate group")
    seen_groups: set[str] = set()
    selected_tasks: set[str] = set()
    known_paper_ids = {
        canonical_paper_id(task["original_row"].get("paper_id"))
        for task in tasks
        if isinstance(task.get("original_row"), Mapping)
    }
    known_paper_ids.discard("")
    proposed_identity_ids: set[str] = set()
    edits = {task_id: _blank_edit() for task_id in tasks_by_id}
    validated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ENRICHMENT_FIELDS:
            raise RemediationError("machine enrichment row fields are invalid")
        group_id = row.get("group_id")
        group = groups_by_id.get(str(group_id))
        if group is None or group_id in seen_groups:
            raise RemediationError("machine enrichment has an unknown or duplicate group")
        seen_groups.add(str(group_id))
        if (
            type(row.get("artifact_version")) is not int
            or row["artifact_version"] != CAPACITY_ENRICHMENT_ARTIFACT_VERSION
            or row.get("queue_fingerprint") != queue_fingerprint
            or row.get("capacity_plan_manifest_sha256") != capacity_plan_manifest_sha256
            or row.get("canonical_paper_id") != group["canonical_paper_id"]
            or row.get("task_ids") != group["task_ids"]
            or row.get("human_review")
            != {"curator_ids": [], "independent_attestation": False}
            or row.get("release_eligibility") != "blocked"
        ):
            raise RemediationError(f"machine enrichment group {group_id} bindings are invalid")
        selected = row.get("selected_task_ids")
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(task_id, str) for task_id in selected)
            or len(selected) != len(set(selected))
            or not set(selected).issubset(set(group["task_ids"]))
            or not set(selected).intersection(group["primary_task_ids"])
            or set(selected).intersection(selected_tasks)
        ):
            raise RemediationError(f"machine enrichment group {group_id} task selection is invalid")
        benchmark_by_task = row.get("benchmark_rows")
        provenance_by_task = row.get("provenance_rows")
        if (
            not isinstance(benchmark_by_task, dict)
            or not isinstance(provenance_by_task, dict)
            or set(benchmark_by_task) != set(selected)
            or set(provenance_by_task) != set(selected)
        ):
            raise RemediationError(f"machine enrichment group {group_id} proposals are incomplete")
        evidence, evidence_ids = _validate_evidence(row.get("external_evidence"), str(group_id))
        evidence_kinds = {record["evidence_id"]: record["kind"] for record in evidence}
        claims = _validate_claims(
            row.get("claim_evidence"), evidence_ids, evidence_kinds, str(group_id)
        )
        evidence_by_id = {record["evidence_id"]: record for record in evidence}
        image_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in claims["image_provenance"]["evidence_ids"]
        ]
        rights_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in claims["usage_rights"]["evidence_ids"]
        ]
        benchmark_rows: dict[str, Any] = {}
        provenance_rows: dict[str, Any] = {}
        identity_proposal: str | None = None
        for task_id in selected:
            benchmark = benchmark_by_task[task_id]
            if not isinstance(benchmark, dict):
                raise RemediationError(f"machine benchmark proposal {task_id} must be an object")
            if set(benchmark) != {
                "sample_id",
                "paper_id",
                "stratum",
                "primary_endpoint",
                "model_task_prompt",
                "messages",
                "images",
                "split_provenance",
            }:
                raise RemediationError(f"machine benchmark proposal {task_id} fields are invalid")
            _schema_error(schemas["benchmark_validator"], benchmark, f"machine benchmark {task_id}")
            if benchmark["primary_endpoint"] is not (benchmark["stratum"] in primary):
                raise RemediationError(
                    f"machine benchmark {task_id} primary_endpoint differs from configured strata"
                )
            paper_id = _validated_paper_id(benchmark, f"machine benchmark {task_id}")
            identity_remediation = group["capacity_role"] == "unresolved_not_counted"
            if (
                not identity_remediation
                and group["canonical_paper_id"]
                and paper_id != group["canonical_paper_id"]
            ):
                raise RemediationError(f"machine benchmark {task_id} changes the canonical paper")
            if identity_remediation:
                if identity_proposal is None:
                    if (
                        paper_id in known_paper_ids
                        and paper_id != group["canonical_paper_id"]
                    ) or paper_id in proposed_identity_ids:
                        raise RemediationError(
                            f"machine benchmark {task_id} proposes a duplicate canonical paper"
                        )
                    identity_proposal = paper_id
                    proposed_identity_ids.add(paper_id)
                elif paper_id != identity_proposal:
                    raise RemediationError(
                        f"machine benchmark {task_id} proposes an inconsistent canonical paper"
                    )
            provenance = _validate_machine_provenance(
                provenance_by_task[task_id],
                benchmark,
                schemas,
                forbidden_hashes,
                image_evidence,
                rights_evidence,
                f"machine provenance {task_id}",
            )
            edits[task_id] = _proposal_edit(benchmark, provenance, str(group_id))
            benchmark_rows[task_id] = copy.deepcopy(benchmark)
            provenance_rows[task_id] = provenance
        selected_tasks.update(selected)
        validated.append(
            {
                **{key: copy.deepcopy(row[key]) for key in _ENRICHMENT_FIELDS},
                "benchmark_rows": benchmark_rows,
                "provenance_rows": provenance_rows,
                "external_evidence": evidence,
                "claim_evidence": claims,
            }
        )
    draft = {
        "artifact_version": CURATOR_ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "edits": edits,
    }
    return validated, draft


def _file_inventory(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"path": name, "sha256": _sha256_bytes(data), "size_bytes": len(data)}
        for name, data in sorted(files.items())
    ]


def _enrichment_artifacts(
    validated: Sequence[Mapping[str, Any]],
    draft: Mapping[str, Any],
    *,
    queue_fingerprint: str,
    capacity_plan_manifest_sha256: str,
    capacity_assist_manifest_sha256: str,
    source_machine_enrichment_sha256: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    summary = (
        "# Machine enrichment contract validation\n\n"
        f"- Queue fingerprint: `{queue_fingerprint}`\n"
        f"- Candidate groups: {len(validated)}\n"
        "- Automatic retain decisions: 0\n"
        "- Human reviewer IDs and attestations: 0\n\n"
        "Импортируйте machine_assisted_draft.json только в чистый curator workspace. "
        "Каждый proposal остается без disposition и требует проверки двумя людьми.\n"
    ).encode("utf-8")
    payload_files = {
        MACHINE_ENRICHMENT_FILENAME: _jsonl_bytes(validated),
        MACHINE_ASSISTED_DRAFT_FILENAME: _json_bytes(draft, newline=True),
        MACHINE_ENRICHMENT_SUMMARY_FILENAME: summary,
    }
    manifest = {
        "artifact_version": CAPACITY_ENRICHMENT_ARTIFACT_VERSION,
        "queue_fingerprint": queue_fingerprint,
        "capacity_plan_manifest_sha256": capacity_plan_manifest_sha256,
        "capacity_assist_manifest_sha256": capacity_assist_manifest_sha256,
        "source_machine_enrichment_sha256": source_machine_enrichment_sha256,
        "candidate_group_count": len(validated),
        "prefilled_task_count": sum(len(row["selected_task_ids"]) for row in validated),
        "policy": {
            "machine_enrichment_schema_validated": True,
            "external_evidence_content_verified": False,
            "automatic_retain": False,
            "automatic_human_identity": False,
            "automatic_attestation": False,
            "quarantined_cross_paper_image_reuse_allowed": False,
            "human_verification_required": True,
        },
        "files": _file_inventory(payload_files),
    }
    return {
        **payload_files,
        MACHINE_ENRICHMENT_MANIFEST_FILENAME: _json_bytes(manifest, newline=True),
    }, manifest


def verify_capacity_enrichment_workspace(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    queue_manifest: str | Path,
    capacity_enrichment_manifest: str | Path,
    machine_enrichment_jsonl: str | Path,
    *,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
) -> dict[str, Any]:
    """Reconstruct a capacity-enrichment package from its queue and exact source bytes."""

    exact_n = _require_capacity_remediation_config(config, command="curate-assemble")
    material = _queue_material(config, prepare_manifest, benchmark_schema, provenance_schema)
    _, verified_queue = _verify_queue_workspace(material, queue_manifest)
    queue_fingerprint = verified_queue["queue_fingerprint"]
    primary_strata = verified_queue["policy"]["primary_strata"]
    details = _task_details(material["tasks"], primary_strata)
    expected_groups = _primary_paper_groups(details, queue_fingerprint)
    plan_files, _, reconstructed_groups = _capacity_plan_artifacts(
        details,
        queue_fingerprint,
        exact_n,
        primary_strata,
    )
    if reconstructed_groups != expected_groups:
        raise RemediationError("capacity enrichment plan reconstruction differs from queue routing")
    plan_sha256 = _sha256_bytes(plan_files[CAPACITY_PLAN_MANIFEST_FILENAME])
    candidates = _capacity_assist_candidates(reconstructed_groups)
    if len(candidates) != exact_n:
        raise RemediationError("capacity enrichment does not bind the exact candidate routing")
    assist_files, _ = _capacity_assist_artifacts(
        candidates,
        material["tasks"],
        details,
        queue_fingerprint,
        plan_sha256,
        exact_n,
    )
    assist_sha256 = _sha256_bytes(assist_files[CAPACITY_ASSIST_MANIFEST_FILENAME])
    enrichment_rows, enrichment_bytes = _strict_jsonl(
        machine_enrichment_jsonl,
        "machine capacity enrichment JSONL",
    )
    validated, draft = _validated_enrichment(
        enrichment_rows,
        candidates,
        material["tasks"],
        material["schemas"],
        queue_fingerprint,
        plan_sha256,
        primary_strata,
    )
    expected_files, expected_manifest = _enrichment_artifacts(
        validated,
        draft,
        queue_fingerprint=queue_fingerprint,
        capacity_plan_manifest_sha256=plan_sha256,
        capacity_assist_manifest_sha256=assist_sha256,
        source_machine_enrichment_sha256=_sha256_bytes(enrichment_bytes),
    )
    manifest_path = Path(capacity_enrichment_manifest)
    actual_manifest, actual_manifest_bytes = _strict_json(
        manifest_path,
        "capacity enrichment manifest",
    )
    root = manifest_path.parent
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise RemediationError(f"cannot inspect capacity enrichment workspace: {exc}") from exc
    if {entry.name for entry in entries} != set(expected_files) or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise RemediationError("capacity enrichment workspace has an unexpected file inventory")
    expected_manifest_bytes = expected_files[MACHINE_ENRICHMENT_MANIFEST_FILENAME]
    if actual_manifest != expected_manifest or actual_manifest_bytes != expected_manifest_bytes:
        raise RemediationError(
            "capacity enrichment manifest differs from deterministic reconstruction"
        )
    for name, expected_bytes in sorted(expected_files.items()):
        actual_bytes = _read_stable_bytes(root / name, f"capacity enrichment {name}")
        if actual_bytes != expected_bytes:
            raise RemediationError(
                f"capacity enrichment file differs from deterministic reconstruction: {name}"
            )
    return {
        "manifest": expected_manifest,
        "manifest_sha256": _sha256_bytes(expected_manifest_bytes),
        "source_machine_enrichment_bytes": enrichment_bytes,
        "files": expected_files,
    }


def _publish_workspace(target: Path, files: Mapping[str, bytes]) -> None:
    staging = target.with_name(f".{target.name}.capacity-enrichment.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise RemediationError(f"capacity enrichment staging path already exists: {staging}")
    try:
        staging.mkdir()
        for name, data in sorted(files.items()):
            _atomic_write(staging / name, data)
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink() and _trees_identical(staging, target):
                return
            raise RemediationError(
                "output_dir already contains a different capacity enrichment workspace"
            )
        try:
            os.replace(staging, target)
        except OSError as exc:
            if target.is_dir() and not target.is_symlink() and _trees_identical(staging, target):
                return
            raise RemediationError(
                f"cannot publish capacity enrichment workspace atomically: {exc}"
            ) from exc
    finally:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)


def generate_capacity_enrichment_package(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    queue_manifest: str | Path,
    capacity_plan_manifest: str | Path,
    capacity_assist_manifest: str | Path,
    machine_enrichment_jsonl: str | Path,
    output_dir: str | Path,
    *,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
) -> dict[str, Any]:
    """Publish validated machine proposals while preserving the human decision barrier."""

    exact_n = _require_capacity_remediation_config(config, command="curate-capacity-enrichment")
    material = _queue_material(config, prepare_manifest, benchmark_schema, provenance_schema)
    queue_root, verified_queue = _verify_queue_workspace(material, queue_manifest)
    queue_fingerprint = verified_queue["queue_fingerprint"]
    primary_strata = verified_queue["policy"]["primary_strata"]
    details = _task_details(material["tasks"], primary_strata)
    expected_groups = _primary_paper_groups(details, queue_fingerprint)
    groups, plan_sha256 = _verified_capacity_plan(
        capacity_plan_manifest,
        expected_groups=expected_groups,
        queue_fingerprint=queue_fingerprint,
        exact_n=exact_n,
        task_details=details,
        primary_strata=primary_strata,
    )
    candidates = _capacity_assist_candidates(groups)
    if len(candidates) != exact_n:
        raise RemediationError("capacity plan does not contain the exact enrichment candidate count")
    assist_sha256 = _verified_assist_manifest(
        capacity_assist_manifest,
        queue_fingerprint=queue_fingerprint,
        capacity_plan_manifest_sha256=plan_sha256,
        candidate_groups=candidates,
        tasks=material["tasks"],
        task_details=details,
        exact_n=exact_n,
    )
    enrichment_rows, enrichment_bytes = _strict_jsonl(
        machine_enrichment_jsonl,
        "machine capacity enrichment JSONL",
    )
    validated, draft = _validated_enrichment(
        enrichment_rows,
        candidates,
        material["tasks"],
        material["schemas"],
        queue_fingerprint,
        plan_sha256,
        primary_strata,
    )
    protected_directories, protected_files = _prepare_input_protection(material)
    target = _separate_workspace_target(
        output_dir,
        queue_root,
        capacity_plan_manifest,
        capacity_assist_manifest,
        protected_directories=protected_directories,
        protected_files=(*protected_files, machine_enrichment_jsonl),
    )
    files, manifest = _enrichment_artifacts(
        validated,
        draft,
        queue_fingerprint=queue_fingerprint,
        capacity_plan_manifest_sha256=plan_sha256,
        capacity_assist_manifest_sha256=assist_sha256,
        source_machine_enrichment_sha256=_sha256_bytes(enrichment_bytes),
    )
    _publish_workspace(target, files)
    return manifest


__all__ = ["generate_capacity_enrichment_package", "verify_capacity_enrichment_workspace"]
