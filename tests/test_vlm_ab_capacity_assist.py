# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
from csv import DictReader
from io import StringIO
from pathlib import Path

import pytest

from scireason.vlm_ab import cli
from scireason.vlm_ab.capacity_assist import (
    CAPACITY_ASSIST_MANIFEST_FILENAME,
    CAPACITY_ASSIST_SUMMARY_FILENAME,
    CAPACITY_CANDIDATE_DOSSIERS_FILENAME,
    CAPACITY_ENRICHMENT_TEMPLATE_FILENAME,
    CAPACITY_HUMAN_REVIEW_FILENAME,
    _separate_workspace_target,
    generate_capacity_assist_package,
)
from scireason.vlm_ab.capacity_plan import CAPACITY_PLAN_SUMMARY_FILENAME, generate_capacity_plan
from scireason.vlm_ab.remediation import RemediationError
from test_vlm_ab_remediation import (
    BENCHMARK_SCHEMA,
    PROVENANCE_SCHEMA,
    _generate_queue,
    _prepare_bundle,
)


def _bundle(root: Path, *, row_count: int = 150, **kwargs: object) -> dict:
    return _prepare_bundle(
        root,
        row_count=row_count,
        capacity_remediation=True,
        **kwargs,
    )


def _generate_plan(bundle: dict, queue: Path, output: Path) -> Path:
    generate_capacity_plan(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        output,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )
    return output / "capacity_plan_manifest.json"


def _generate(bundle: dict, queue: Path, plan_manifest: Path, output: Path) -> dict:
    return generate_capacity_assist_package(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        plan_manifest,
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


def test_capacity_assist_is_exact_queue_bound_deterministic_and_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    queue = tmp_path / "queue"
    queue_manifest = _generate_queue(bundle, queue)
    plan_manifest = _generate_plan(bundle, queue, tmp_path / "capacity-plan")
    queue_before = _files(queue)
    output = tmp_path / "capacity-assist"

    manifest = _generate(bundle, queue, plan_manifest, output)
    first = _files(output)
    repeated = _generate(bundle, queue, plan_manifest, output)
    dossiers = _jsonl(output / CAPACITY_CANDIDATE_DOSSIERS_FILENAME)
    templates = _jsonl(output / CAPACITY_ENRICHMENT_TEMPLATE_FILENAME)
    checklist = list(
        DictReader(StringIO((output / CAPACITY_HUMAN_REVIEW_FILENAME).read_text("utf-8")))
    )

    assert repeated == manifest
    assert _files(output) == first
    assert _files(queue) == queue_before
    assert set(first) == {
        CAPACITY_CANDIDATE_DOSSIERS_FILENAME,
        CAPACITY_ENRICHMENT_TEMPLATE_FILENAME,
        CAPACITY_HUMAN_REVIEW_FILENAME,
        CAPACITY_ASSIST_SUMMARY_FILENAME,
        CAPACITY_ASSIST_MANIFEST_FILENAME,
    }
    assert "completed_decisions.jsonl" not in first
    assert manifest["queue_fingerprint"] == queue_manifest["queue_fingerprint"]
    assert manifest["capacity_plan_manifest_sha256"] == hashlib.sha256(
        plan_manifest.read_bytes()
    ).hexdigest()
    assert manifest["exact_n"] == 150
    assert manifest["candidate_group_count"] == 150
    assert manifest["represented_task_count"] == 150
    assert manifest["file_count"] == 4
    assert manifest["policy"] == {
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
    }
    for entry in manifest["files"]:
        payload = first[entry["path"]]
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]

    assert len(dossiers) == len(templates) == len(checklist) == 150
    assert all(dossier["machine_selected_task_ids"] == [] for dossier in dossiers)
    assert all(
        dossier["capacity_plan_manifest_sha256"]
        == manifest["capacity_plan_manifest_sha256"]
        for dossier in dossiers
    )
    assert all(dossier["human_review"] == {
        "curator_ids": [],
        "independent_attestation": False,
    } for dossier in dossiers)
    assert all(
        task["machine_status"] == "pending_external_enrichment"
        and task["source_row_hint"]["legacy_prompt_text_included"] is False
        and all(image["release_usable"] is False for image in task["audited_image_hints"])
        for dossier in dossiers
        for task in dossier["tasks"]
    )
    assert all(template["selected_task_ids"] == [] for template in templates)
    assert all(template["benchmark_rows"] == {} for template in templates)
    assert all(template["provenance_rows"] == {} for template in templates)
    assert all(template["release_eligibility"] == "blocked" for template in templates)
    assert all(
        claim == {"status": "pending", "evidence_ids": []}
        for template in templates
        for claim in template["claim_evidence"].values()
    )
    assert all(
        not row["curator_1_id"]
        and not row["curator_2_id"]
        and not row["paper_identity_confirmed"]
        for row in checklist
    )
    dossier_text = first[CAPACITY_CANDIDATE_DOSSIERS_FILENAME].decode("utf-8")
    assert bundle["rows"][0]["model_task_prompt"] not in dossier_text
    assert bundle["rows"][0]["reference_answer"] not in dossier_text
    assert '"paper_title"' not in dossier_text
    assert '"page_hint"' not in dossier_text
    summary = first[CAPACITY_ASSIST_SUMMARY_FILENAME].decode("utf-8")
    assert "N=150" in summary
    assert "не создает retain decisions" in summary


def test_capacity_assist_requires_exact_candidate_count(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", row_count=149)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    plan_manifest = _generate_plan(bundle, queue, tmp_path / "capacity-plan")
    output = tmp_path / "capacity-assist"

    with pytest.raises(RemediationError, match="requires exactly 150.*found 149"):
        _generate(bundle, queue, plan_manifest, output)

    assert not output.exists()


def test_capacity_assist_includes_blocked_identity_remediation_groups(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", row_count=150, identity_conflict=True)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    plan_manifest = _generate_plan(bundle, queue, tmp_path / "capacity-plan")
    output = tmp_path / "capacity-assist"

    manifest = _generate(bundle, queue, plan_manifest, output)
    dossiers = _jsonl(output / CAPACITY_CANDIDATE_DOSSIERS_FILENAME)
    unresolved = [row for row in dossiers if row["capacity_role"] == "unresolved_not_counted"]

    assert manifest["candidate_group_count"] == 150
    assert manifest["clean_candidate_group_count"] == 149
    assert manifest["identity_remediation_group_count"] == 1
    assert len(unresolved) == 1
    assert unresolved[0]["canonical_paper_id"] == "doi:10.5555/remediation.2"
    assert unresolved[0]["identity_status"] == "unresolved"
    assert unresolved[0]["machine_selected_task_ids"] == []


def test_capacity_assist_rejects_tampered_queue_and_workspace(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    plan_manifest = _generate_plan(bundle, queue, tmp_path / "capacity-plan")
    output = tmp_path / "capacity-assist"
    _generate(bundle, queue, plan_manifest, output)
    summary = output / CAPACITY_ASSIST_SUMMARY_FILENAME
    summary.write_bytes(summary.read_bytes() + b"tampered")

    with pytest.raises(RemediationError, match="different capacity assist workspace"):
        _generate(bundle, queue, plan_manifest, output)

    plan_summary = plan_manifest.parent / CAPACITY_PLAN_SUMMARY_FILENAME
    plan_summary.write_bytes(plan_summary.read_bytes() + b"tampered")
    plan = json.loads(plan_manifest.read_text(encoding="utf-8"))
    summary_record = next(
        record for record in plan["files"] if record["path"] == CAPACITY_PLAN_SUMMARY_FILENAME
    )
    summary_record.update(
        sha256=hashlib.sha256(plan_summary.read_bytes()).hexdigest(),
        size_bytes=plan_summary.stat().st_size,
    )
    plan_manifest.write_text(
        json.dumps(plan, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resealed_output = tmp_path / "resealed-plan-assist"
    with pytest.raises(RemediationError, match="deterministic reconstruction"):
        _generate(bundle, queue, plan_manifest, resealed_output)
    assert not resealed_output.exists()

    other_output = tmp_path / "other-capacity-assist"
    (queue / "tasks.jsonl").write_bytes((queue / "tasks.jsonl").read_bytes() + b"\n")
    with pytest.raises(RemediationError, match="tampered"):
        _generate(bundle, queue, plan_manifest, other_output)
    assert not other_output.exists()


@pytest.mark.parametrize("location", ["same", "inside", "contains"])
def test_capacity_assist_rejects_queue_related_output(tmp_path: Path, location: str) -> None:
    bundle = _bundle(tmp_path / "bundle")
    queue = tmp_path / "container" / "queue"
    _generate_queue(bundle, queue)
    plan_manifest = _generate_plan(bundle, queue, tmp_path / "capacity-plan")
    output = {
        "same": queue,
        "inside": queue / "capacity-assist",
        "contains": queue.parent,
    }[location]
    before = _files(queue)

    with pytest.raises(RemediationError, match="separate from"):
        _generate(bundle, queue, plan_manifest, output)

    assert _files(queue) == before


def test_capacity_assist_rejects_plan_related_output(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    queue = tmp_path / "queue-container" / "queue"
    _generate_queue(bundle, queue)
    plan = tmp_path / "plan-container" / "capacity-plan"
    plan_manifest = _generate_plan(bundle, queue, plan)
    before = _files(plan)

    for output in (plan, plan / "nested" / "capacity-assist", plan.parent):
        with pytest.raises(RemediationError, match="immutable input workspaces"):
            _separate_workspace_target(output, queue, plan_manifest)

    with pytest.raises(RemediationError, match="immutable input workspaces"):
        _generate(bundle, queue, plan_manifest, plan / "nested" / "capacity-assist")

    assert _files(plan) == before
    assert not (plan / "nested").exists()

    prepare_root = Path(bundle["prepared"]["dataset_root"])
    with pytest.raises(RemediationError, match="immutable input workspaces"):
        _generate(
            bundle,
            queue,
            plan_manifest,
            prepare_root / "nested" / "capacity-assist",
        )
    assert not (prepare_root / "nested").exists()

    malformed_workspace = tmp_path / "malformed-capacity-plan"
    malformed_manifest = malformed_workspace / "capacity_plan_manifest.json"
    malformed_manifest.mkdir(parents=True)
    with pytest.raises(RemediationError, match="regular non-symlink file"):
        _generate(
            bundle,
            queue,
            malformed_manifest,
            malformed_workspace / "nested" / "capacity-assist",
        )
    assert not (malformed_workspace / "nested").exists()


def test_curate_capacity_assist_cli_parser_and_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    captured: dict = {}
    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: {"loaded": True})
    monkeypatch.setattr(cli, "_repo_root", lambda _path, _root: tmp_path)

    def fake_generate(*args, **kwargs) -> dict:
        captured.update(args=args, kwargs=kwargs)
        return {"candidate_group_count": 150}

    monkeypatch.setattr(cli, "generate_capacity_assist_package", fake_generate)
    code = cli.main(
        [
            "--config",
            str(config_path),
            "curate-capacity-assist",
            "--prepare-manifest",
            "prepare_manifest.json",
            "--queue-manifest",
            "queue/queue_manifest.json",
            "--capacity-plan-manifest",
            "capacity-plan/capacity_plan_manifest.json",
            "--output-dir",
            "capacity-assist",
        ]
    )

    assert code == 0
    assert captured["args"][0] == {"loaded": True}
    assert captured["args"][3] == Path("capacity-plan/capacity_plan_manifest.json")
    assert captured["kwargs"]["benchmark_schema"].name == "publication_benchmark_row.schema.json"
    assert captured["kwargs"]["provenance_schema"].name == "publication_provenance_row.schema.json"
