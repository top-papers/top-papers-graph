# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from csv import DictReader
from io import StringIO
from pathlib import Path

import pytest

from scireason.vlm_ab import cli
from scireason.vlm_ab.capacity_plan import (
    CAPACITY_PAPER_GROUPS_FILENAME,
    CAPACITY_PLAN_MANIFEST_FILENAME,
    CAPACITY_PLAN_SUMMARY_FILENAME,
    REVIEW_ASSIGNMENT_PLAN_FILENAME,
    _group_id,
    _primary_paper_groups,
    _task_details,
    generate_capacity_plan,
)
from scireason.vlm_ab.remediation import RemediationError, _queue_material
from test_vlm_ab_capacity150_remediation import _archived_config
from test_vlm_ab_remediation import (
    BENCHMARK_SCHEMA,
    PROVENANCE_SCHEMA,
    _generate_queue,
    _prepare_bundle,
)


def _capacity_bundle(root: Path, **kwargs: object) -> dict:
    return _prepare_bundle(root, capacity_remediation=True, **kwargs)


def _generate(bundle: dict, queue: Path, output: Path) -> dict:
    return generate_capacity_plan(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        output,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_capacity_plan_is_queue_bound_deterministic_and_minimal(tmp_path: Path) -> None:
    bundle = _capacity_bundle(
        tmp_path / "bundle",
        row_count=386,
        contaminated_training=True,
    )
    queue = tmp_path / "queue"
    queue_manifest = _generate_queue(bundle, queue)
    queue_before = _files(queue)
    output = tmp_path / "capacity-plan"

    manifest = _generate(bundle, queue, output)
    first = _files(output)
    repeated = _generate(bundle, queue, output)
    assignments = list(
        DictReader(StringIO((output / REVIEW_ASSIGNMENT_PLAN_FILENAME).read_text(encoding="utf-8")))
    )
    groups = _jsonl(output / CAPACITY_PAPER_GROUPS_FILENAME)
    material = _queue_material(
        bundle["config"], bundle["prepare_manifest"], BENCHMARK_SCHEMA, PROVENANCE_SCHEMA
    )

    assert repeated == manifest
    assert _files(output) == first
    assert _files(queue) == queue_before
    assert set(first) == {
        REVIEW_ASSIGNMENT_PLAN_FILENAME,
        CAPACITY_PAPER_GROUPS_FILENAME,
        CAPACITY_PLAN_SUMMARY_FILENAME,
        CAPACITY_PLAN_MANIFEST_FILENAME,
    }
    assert "completed_decisions.jsonl" not in first
    assert (
        json.loads((output / CAPACITY_PLAN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        == manifest
    )
    assert manifest["queue_fingerprint"] == queue_manifest["queue_fingerprint"]
    assert manifest["task_count"] == 386
    assert manifest["exact_n"] == 150
    assert manifest["capacity_candidate_group_count"] == 385
    assert manifest["capacity_exact_target_available"] is False
    assert manifest["capacity_remediation_pool_count"] == 385
    assert manifest["capacity_exact_remediation_pool_available"] is False
    assert manifest["group_counts"] == {
        "primary_paper_groups": 386,
        "required_candidate_for_exact_150": 385,
        "closed_training_overlap": 1,
        "unresolved_not_counted": 0,
        "identity_clean": 386,
        "identity_unresolved": 0,
    }
    assert manifest["policy"]["automatic_retain"] is False
    assert manifest["policy"]["automatic_duplicate_row_selection"] is False
    assert manifest["policy"]["automatic_paper_remapping"] is False
    assert manifest["policy"]["reviewer_identity_prefill"] is False
    assert manifest["policy"]["attestation_prefill"] is False
    assert manifest["policy"]["decision_prefill"] is False
    assert manifest["policy"]["paper_factual_verification_claimed"] is False
    assert manifest["policy"]["primary_endpoint_rule"] == (
        "stratum_in_configured_primary_strata"
    )
    assert manifest["policy"]["primary_strata"] == ["multimodal_hard"]
    assert manifest["file_count"] == 3
    for entry in manifest["files"]:
        payload = first[entry["path"]]
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]

    assert list(assignments[0]) == [
        "task_id",
        "source_row_index",
        "workstream",
        "proposed_disposition",
        "reviewer_slot_1",
        "reviewer_slot_2",
        "critical_codes",
        "required_actions",
    ]
    assert len(assignments) == 386
    assert [row["task_id"] for row in assignments] == [
        task["task_id"] for task in material["tasks"]
    ]
    assert [int(row["source_row_index"]) for row in assignments] == list(range(386))
    assert assignments[0]["workstream"] == "training_overlap_confirmation"
    assert assignments[0]["proposed_disposition"] == "exclude_pending_human_attestation"
    assert all(
        row["workstream"] == "manual_curation"
        and row["proposed_disposition"] == "manual_review_required"
        for row in assignments[1:]
    )
    assert all(row["reviewer_slot_1"] != row["reviewer_slot_2"] for row in assignments)
    slot_loads = Counter(
        slot for row in assignments for slot in (row["reviewer_slot_1"], row["reviewer_slot_2"])
    )
    assert set(slot_loads) == {"reviewer-slot-1", "reviewer-slot-2", "reviewer-slot-3"}
    assert max(slot_loads.values()) - min(slot_loads.values()) <= 1
    assert manifest["reviewer_slot_load_counts"] == dict(sorted(slot_loads.items()))
    assert all(
        "reviewer-1" not in value and "reviewer-2" not in value
        for row in assignments
        for value in row.values()
    )

    expected_group_fields = {
        "artifact_version",
        "queue_fingerprint",
        "group_id",
        "canonical_paper_id",
        "task_ids",
        "primary_task_ids",
        "task_count",
        "training_overlap_closed",
        "identity_status",
        "capacity_role",
        "human_instruction",
    }
    assert len(groups) == 386
    assert all(set(group) == expected_group_fields for group in groups)
    assert groups[0]["group_id"] == _group_id(
        queue_manifest["queue_fingerprint"],
        groups[0]["canonical_paper_id"],
        groups[0]["task_ids"],
    )
    assert groups[0]["training_overlap_closed"] is True
    assert groups[0]["capacity_role"] == "closed_training_overlap"
    assert all(group["identity_status"] == "clean" for group in groups)
    group_text = (output / CAPACITY_PAPER_GROUPS_FILENAME).read_text(encoding="utf-8")
    assert bundle["rows"][0]["model_task_prompt"] not in group_text
    assert bundle["rows"][0]["reference_answer"] not in group_text
    assert "original_row" not in group_text
    assert "legacy_provenance_rows" not in group_text
    summary = (output / CAPACITY_PLAN_SUMMARY_FILENAME).read_text(encoding="utf-8")
    assert "N=150" in summary
    assert "Primary membership выводится только из configured strata" in summary
    assert "автоматического пути к plan нет" in summary
    assert "не выбирает duplicate row" in summary
    assert "не заполняет reviewer ID" in summary


def test_capacity_groups_close_overlap_and_exclude_unresolved_identity(tmp_path: Path) -> None:
    bundle = _capacity_bundle(
        tmp_path / "bundle",
        row_count=3,
        contaminated_training=True,
        duplicate_primary_paper=True,
        identity_conflict=True,
    )
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    output = tmp_path / "capacity-plan"

    manifest = _generate(bundle, queue, output)
    groups = _jsonl(output / CAPACITY_PAPER_GROUPS_FILENAME)

    closed = next(group for group in groups if group["capacity_role"] == "closed_training_overlap")
    unresolved = next(
        group for group in groups if group["capacity_role"] == "unresolved_not_counted"
    )
    assert closed["task_count"] == 2
    assert len(closed["primary_task_ids"]) == 2
    assert closed["training_overlap_closed"] is True
    assert closed["identity_status"] == "clean"
    assert unresolved["canonical_paper_id"] == "doi:10.5555/remediation.2"
    assert unresolved["identity_status"] == "unresolved"
    assert unresolved["training_overlap_closed"] is False
    assert manifest["capacity_candidate_group_count"] == 0
    assert manifest["capacity_exact_target_available"] is False


def test_capacity_plan_marks_an_exact_clean_candidate_count_available(tmp_path: Path) -> None:
    bundle = _capacity_bundle(tmp_path / "bundle", row_count=150)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    output = tmp_path / "capacity-plan"

    manifest = _generate(bundle, queue, output)

    assert manifest["capacity_candidate_group_count"] == 150
    assert manifest["capacity_exact_target_available"] is True
    assert manifest["capacity_remediation_pool_count"] == 150
    assert manifest["capacity_exact_remediation_pool_available"] is True
    assert "равно exact N" in (output / CAPACITY_PLAN_SUMMARY_FILENAME).read_text(encoding="utf-8")

    prepare_root = Path(bundle["prepared"]["dataset_root"])
    protected_output = prepare_root / "nested" / "capacity-plan"
    with pytest.raises(RemediationError, match="immutable input workspaces"):
        _generate(bundle, queue, protected_output)
    assert not (prepare_root / "nested").exists()


def test_capacity_routing_derives_primary_membership_from_configured_strata() -> None:
    tasks = [
        {
            "task_id": "task-primary",
            "source_row_index": 0,
            "original_row": {
                "paper_id": "doi:10.5555/primary",
                "stratum": "multimodal_hard",
                "primary_endpoint": False,
            },
            "critical_codes": [],
            "warning_codes": [],
            "training_overlap_paper_ids": [],
        },
        {
            "task_id": "task-control",
            "source_row_index": 1,
            "original_row": {
                "paper_id": "doi:10.5555/control",
                "stratum": "easy_control",
                "primary_endpoint": True,
            },
            "critical_codes": [],
            "warning_codes": [],
            "training_overlap_paper_ids": [],
        },
    ]

    details = _task_details(tasks, ["multimodal_hard"])
    groups = _primary_paper_groups(details, "f" * 64)

    assert [detail["primary_endpoint"] for detail in details] == [True, False]
    assert [group["canonical_paper_id"] for group in groups] == ["doi:10.5555/primary"]


def test_capacity_plan_exposes_exact_identity_remediation_pool(tmp_path: Path) -> None:
    bundle = _capacity_bundle(tmp_path / "bundle", row_count=150, identity_conflict=True)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)

    manifest = _generate(bundle, queue, tmp_path / "capacity-plan")

    assert manifest["capacity_candidate_group_count"] == 149
    assert manifest["capacity_exact_target_available"] is False
    assert manifest["capacity_remediation_pool_count"] == 150
    assert manifest["capacity_exact_remediation_pool_available"] is True


@pytest.mark.parametrize("kind", ["ordinary", "archived"])
def test_capacity_plan_rejects_non_capacity_configs(tmp_path: Path, kind: str) -> None:
    bundle = _capacity_bundle(tmp_path / "bundle")
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    config = bundle["config"] if kind == "ordinary" else _archived_config()
    if kind == "ordinary":
        config = copy.deepcopy(config)
        config["power"]["target_effect"] = 0.2

    with pytest.raises(RemediationError, match="capacity remediation working config"):
        generate_capacity_plan(
            config,
            bundle["prepare_manifest"],
            queue / "queue_manifest.json",
            tmp_path / f"{kind}-plan",
            benchmark_schema=BENCHMARK_SCHEMA,
            provenance_schema=PROVENANCE_SCHEMA,
        )


@pytest.mark.parametrize("location", ["same", "inside", "contains"])
def test_capacity_plan_rejects_queue_related_output(tmp_path: Path, location: str) -> None:
    bundle = _capacity_bundle(tmp_path / "bundle")
    queue = tmp_path / "container" / "queue"
    _generate_queue(bundle, queue)
    output = {
        "same": queue,
        "inside": queue / "capacity-plan",
        "contains": queue.parent,
    }[location]
    before = _files(queue)

    with pytest.raises(RemediationError, match="separate from"):
        _generate(bundle, queue, output)

    assert _files(queue) == before


def test_capacity_plan_rejects_tampered_queue(tmp_path: Path) -> None:
    bundle = _capacity_bundle(tmp_path / "bundle")
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    (queue / "tasks.jsonl").write_bytes((queue / "tasks.jsonl").read_bytes() + b"\n")

    with pytest.raises(RemediationError, match="tampered"):
        _generate(bundle, queue, tmp_path / "capacity-plan")

    assert not (tmp_path / "capacity-plan").exists()


def test_curate_capacity_plan_cli_parser_and_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    captured: dict = {}

    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: {"loaded": True})
    monkeypatch.setattr(cli, "_repo_root", lambda _path, _root: tmp_path)

    def fake_generate(*args, **kwargs) -> dict:
        captured.update(args=args, kwargs=kwargs)
        return {"task_count": 386}

    monkeypatch.setattr(cli, "generate_capacity_plan", fake_generate)
    code = cli.main(
        [
            "--config",
            str(config_path),
            "curate-capacity-plan",
            "--prepare-manifest",
            "prepare_manifest.json",
            "--queue-manifest",
            "queue/queue_manifest.json",
            "--output-dir",
            "capacity-plan",
        ]
    )

    assert code == 0
    assert captured["args"][0] == {"loaded": True}
    assert captured["kwargs"]["benchmark_schema"].name == "publication_benchmark_row.schema.json"
    assert captured["kwargs"]["provenance_schema"].name == "publication_provenance_row.schema.json"
