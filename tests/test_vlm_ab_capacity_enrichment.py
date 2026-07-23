# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scireason.vlm_ab import cli
from scireason.vlm_ab.capacity_assist import (
    CAPACITY_ASSIST_MANIFEST_FILENAME,
    CAPACITY_CANDIDATE_DOSSIERS_FILENAME,
    CAPACITY_ENRICHMENT_TEMPLATE_FILENAME,
    generate_capacity_assist_package,
)
from scireason.vlm_ab.capacity_enrichment import (
    MACHINE_ASSISTED_DRAFT_FILENAME,
    MACHINE_ENRICHMENT_FILENAME,
    MACHINE_ENRICHMENT_MANIFEST_FILENAME,
    MACHINE_ENRICHMENT_SUMMARY_FILENAME,
    generate_capacity_enrichment_package,
    verify_capacity_enrichment_workspace,
)
from scireason.vlm_ab.capacity_plan import generate_capacity_plan
from scireason.vlm_ab.curator import _blank_edit
from scireason.vlm_ab.prepare import _validated_benchmark_assembly
from scireason.vlm_ab.remediation import (
    DecisionValidationError,
    RemediationError,
    assemble_corrected_release,
)
from test_vlm_ab_remediation import (
    BENCHMARK_SCHEMA,
    PROVENANCE_SCHEMA,
    _complete_provenance,
    _completed_decisions,
    _curated_replacements,
    _generate_queue,
    _prepare_bundle,
    _read_jsonl,
    _write_jsonl,
)


def _setup(tmp_path: Path, **bundle_kwargs: object) -> tuple[dict, Path, Path, Path]:
    bundle = _prepare_bundle(
        tmp_path / "bundle",
        row_count=150,
        capacity_remediation=True,
        **bundle_kwargs,
    )
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    plan = tmp_path / "capacity-plan"
    generate_capacity_plan(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        plan,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )
    assist = tmp_path / "capacity-assist"
    generate_capacity_assist_package(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        plan / "capacity_plan_manifest.json",
        assist,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )
    return bundle, queue, plan, assist


def _machine_rows(bundle: dict, queue: Path, assist: Path) -> list[dict]:
    templates = _read_jsonl(assist / CAPACITY_ENRICHMENT_TEMPLATE_FILENAME)
    dossiers = {
        row["group_id"]: row
        for row in _read_jsonl(assist / CAPACITY_CANDIDATE_DOSSIERS_FILENAME)
    }
    tasks = {row["task_id"]: row for row in _read_jsonl(queue / "tasks.jsonl")}
    rows: list[dict] = []
    for template in templates:
        row = copy.deepcopy(template)
        task_id = dossiers[row["group_id"]]["primary_task_ids"][0]
        task = tasks[task_id]
        source = copy.deepcopy(bundle["rows"][task["source_row_index"]])
        if dossiers[row["group_id"]]["capacity_role"] == "unresolved_not_counted":
            source["paper_id"] = f"doi:10.5555/resolved.{task['source_row_index']}"
        source["split_provenance"] = {
            "paper_holdout": True,
            "source_holdout": True,
            "creator_holdout": True,
            "training_overlap_checked": True,
            "source_document_id": f"source-document-{task['source_row_index']}",
            "creator_group_id": f"creator-group-{task['source_row_index']}",
        }
        source = {
            key: source[key]
            for key in (
                "sample_id",
                "paper_id",
                "stratum",
                "primary_endpoint",
                "model_task_prompt",
                "messages",
                "images",
                "split_provenance",
            )
        }
        provenance = _complete_provenance(source, bundle["image_bytes"])
        for image in provenance["images"]:
            image["verified_by"] = []
        image_sha256 = provenance["images"][0]["sha256"]
        evidence_ids = {
            kind: f"evidence-{kind}-{task['source_row_index']}"
            for kind in ("article", "figure", "license", "training_lineage")
        }
        claim_kinds = {
            "paper_identity": "article",
            "scientific_prompt": "article",
            "image_provenance": "figure",
            "usage_rights": "license",
            "paper_holdout": "training_lineage",
            "source_holdout": "training_lineage",
            "creator_holdout": "training_lineage",
            "training_overlap": "training_lineage",
        }
        row.update(
            {
                "selected_task_ids": [task_id],
                "benchmark_rows": {task_id: source},
                "provenance_rows": {task_id: provenance},
                "external_evidence": [
                    {
                        "evidence_id": evidence_ids[kind],
                        "kind": kind,
                        "source_url": provenance["images"][0]["source_url"],
                        "content_sha256": image_sha256,
                        "asserted_license": "CC-BY-4.0" if kind == "license" else None,
                        "retrieved_at": "2026-07-21T00:00:00Z",
                        "notes": "Machine acquisition fixture.",
                    }
                    for kind in ("article", "figure", "license", "training_lineage")
                ],
                "claim_evidence": {
                    name: {
                        "status": "machine_supported",
                        "evidence_ids": [evidence_ids[claim_kinds[name]]],
                    }
                    for name in row["claim_evidence"]
                },
            }
        )
        rows.append(row)
    return rows


def _generate(
    bundle: dict,
    queue: Path,
    plan: Path,
    assist: Path,
    machine_input: Path,
    output: Path,
) -> dict:
    return generate_capacity_enrichment_package(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        plan / "capacity_plan_manifest.json",
        assist / CAPACITY_ASSIST_MANIFEST_FILENAME,
        machine_input,
        output,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )


def test_capacity_enrichment_publishes_bound_prefill_without_human_decisions(
    tmp_path: Path,
) -> None:
    bundle, queue, plan, assist = _setup(tmp_path)
    rows = _machine_rows(bundle, queue, assist)
    machine_input = tmp_path / "machine-enrichment.jsonl"
    _write_jsonl(machine_input, rows)
    output = tmp_path / "enrichment-package"

    manifest = _generate(bundle, queue, plan, assist, machine_input, output)
    repeated = _generate(bundle, queue, plan, assist, machine_input, output)
    draft = json.loads((output / MACHINE_ASSISTED_DRAFT_FILENAME).read_text("utf-8"))
    published = _read_jsonl(output / MACHINE_ENRICHMENT_FILENAME)

    assert repeated == manifest
    assert manifest["candidate_group_count"] == 150
    assert manifest["prefilled_task_count"] == 150
    assert manifest["source_machine_enrichment_sha256"] == hashlib.sha256(
        machine_input.read_bytes()
    ).hexdigest()
    assert manifest["policy"] == {
        "machine_enrichment_schema_validated": True,
        "external_evidence_content_verified": False,
        "automatic_retain": False,
        "automatic_human_identity": False,
        "automatic_attestation": False,
        "quarantined_cross_paper_image_reuse_allowed": False,
        "human_verification_required": True,
    }
    assert set(path.name for path in output.iterdir()) == {
        MACHINE_ENRICHMENT_FILENAME,
        MACHINE_ASSISTED_DRAFT_FILENAME,
        MACHINE_ENRICHMENT_SUMMARY_FILENAME,
        MACHINE_ENRICHMENT_MANIFEST_FILENAME,
    }
    selected = {task_id for row in published for task_id in row["selected_task_ids"]}
    assert len(selected) == 150
    for task_id, edit in draft["edits"].items():
        if task_id in selected:
            assert edit["disposition"] == ""
            assert edit["reviewed_by"] == ""
            assert edit["independent_attestation"] is False
            assert edit["sample_id"]
            assert all(image["verified_by"] == "" for image in edit["images"])
        else:
            assert edit == _blank_edit()
    assert all(
        row["human_review"] == {"curator_ids": [], "independent_attestation": False}
        and row["release_eligibility"] == "blocked"
        for row in published
    )


def test_capacity_assembly_archives_exact_enrichment_without_bypassing_humans(
    tmp_path: Path,
) -> None:
    bundle, queue, plan, assist = _setup(tmp_path)
    machine_input = tmp_path / "machine-enrichment.jsonl"
    _write_jsonl(machine_input, _machine_rows(bundle, queue, assist))
    package = tmp_path / "enrichment-package"
    _generate(bundle, queue, plan, assist, machine_input, package)
    manifest_path = package / MACHINE_ENRICHMENT_MANIFEST_FILENAME
    verified = verify_capacity_enrichment_workspace(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        manifest_path,
        machine_input,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    pending_decisions = queue / "decision_template.jsonl"

    with pytest.raises(RemediationError, match="requires both.*enrichment manifest"):
        assemble_corrected_release(
            bundle["config"],
            bundle["prepare_manifest"],
            queue / "queue_manifest.json",
            pending_decisions,
            curated_root,
            tmp_path / "missing-evidence-release",
            benchmark_schema=BENCHMARK_SCHEMA,
            provenance_schema=PROVENANCE_SCHEMA,
        )
    with pytest.raises(DecisionValidationError, match="pending"):
        assemble_corrected_release(
            bundle["config"],
            bundle["prepare_manifest"],
            queue / "queue_manifest.json",
            pending_decisions,
            curated_root,
            tmp_path / "pending-release",
            benchmark_schema=BENCHMARK_SCHEMA,
            provenance_schema=PROVENANCE_SCHEMA,
            capacity_enrichment_manifest=manifest_path,
            machine_enrichment_jsonl=machine_input,
        )

    decisions = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions, _completed_decisions(queue, replacements))
    release = tmp_path / "release"
    assembly = assemble_corrected_release(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        decisions,
        curated_root,
        release,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
        capacity_enrichment_manifest=manifest_path,
        machine_enrichment_jsonl=machine_input,
    )
    repeated = assemble_corrected_release(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        decisions,
        curated_root,
        release,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
        capacity_enrichment_manifest=manifest_path,
        machine_enrichment_jsonl=machine_input,
    )

    binding = assembly["machine_enrichment_evidence"]
    assert repeated == assembly
    assert binding["package_manifest_sha256"] == verified["manifest_sha256"]
    assert binding["external_evidence_content_verified"] is False
    assert binding["human_verification_required"] is True
    assert (
        release / binding["source_machine_enrichment_path"]
    ).read_bytes() == machine_input.read_bytes()
    for name in verified["files"]:
        assert (
            release / "audit" / "machine_enrichment" / "package" / name
        ).read_bytes() == (package / name).read_bytes()
    inventory = {record["path"] for record in assembly["output_files"]}
    assert binding["source_machine_enrichment_path"] in inventory
    assert binding["package_manifest_path"] in inventory
    review_binding = assembly["human_review_evidence"]
    assert (release / review_binding["decisions_path"]).read_bytes() == decisions.read_bytes()
    assert (release / review_binding["queue_manifest_path"]).read_bytes() == (
        queue / "queue_manifest.json"
    ).read_bytes()
    assert review_binding["decisions_sha256"] == assembly["decisions_sha256"]
    assert review_binding["independent_attestation_required"] is True
    strict_source = copy.deepcopy(bundle["config"])
    strict_source["experiment"]["id"] = "renamed-remediation-study"
    assembly_path = release / "assembly_manifest.json"
    strict_source["benchmark"].update(
        {
            "data_file": "data/task3_vlm_generation.jsonl",
            "provenance_file": "article_image_sources.jsonl",
            "assembly_manifest_file": "assembly_manifest.json",
            "assembly_manifest_sha256": hashlib.sha256(assembly_path.read_bytes()).hexdigest(),
        }
    )
    assert _validated_benchmark_assembly(strict_source, release) == assembly_path

    canonical_path = package / MACHINE_ENRICHMENT_FILENAME
    canonical_path.write_bytes(canonical_path.read_bytes() + b"\n")
    resealed = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item for item in resealed["files"] if item["path"] == MACHINE_ENRICHMENT_FILENAME
    )
    record["sha256"] = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    record["size_bytes"] = canonical_path.stat().st_size
    manifest_path.write_text(
        json.dumps(resealed, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RemediationError, match="deterministic reconstruction"):
        verify_capacity_enrichment_workspace(
            bundle["config"],
            bundle["prepare_manifest"],
            queue / "queue_manifest.json",
            manifest_path,
            machine_input,
            benchmark_schema=BENCHMARK_SCHEMA,
            provenance_schema=PROVENANCE_SCHEMA,
        )


def test_capacity_enrichment_rejects_fabricated_human_state(tmp_path: Path) -> None:
    bundle, queue, plan, assist = _setup(tmp_path)
    rows = _machine_rows(bundle, queue, assist)
    rows[0]["human_review"] = {
        "curator_ids": ["fake-1", "fake-2"],
        "independent_attestation": True,
    }
    machine_input = tmp_path / "machine-enrichment.jsonl"
    _write_jsonl(machine_input, rows)
    output = tmp_path / "enrichment-package"

    with pytest.raises(RemediationError, match="bindings are invalid"):
        _generate(bundle, queue, plan, assist, machine_input, output)

    assert not output.exists()

    rows[0]["human_review"] = {"curator_ids": [], "independent_attestation": False}
    _write_jsonl(machine_input, rows)
    dossier = assist / CAPACITY_CANDIDATE_DOSSIERS_FILENAME
    dossier.write_bytes(b"{}\n")
    manifest_path = assist / CAPACITY_ASSIST_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dossier_record = next(
        record
        for record in manifest["files"]
        if record["path"] == CAPACITY_CANDIDATE_DOSSIERS_FILENAME
    )
    dossier_record.update(
        sha256=hashlib.sha256(dossier.read_bytes()).hexdigest(),
        size_bytes=dossier.stat().st_size,
    )
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RemediationError, match="deterministic reconstruction"):
        _generate(bundle, queue, plan, assist, machine_input, output)

    assert not output.exists()


def test_capacity_enrichment_publishes_blocked_identity_proposal(tmp_path: Path) -> None:
    bundle, queue, plan, assist = _setup(tmp_path, identity_conflict=True)
    rows = _machine_rows(bundle, queue, assist)
    machine_input = tmp_path / "machine-enrichment.jsonl"
    _write_jsonl(machine_input, rows)
    output = tmp_path / "enrichment-package"

    manifest = _generate(bundle, queue, plan, assist, machine_input, output)
    published = _read_jsonl(output / MACHINE_ENRICHMENT_FILENAME)
    unresolved = [
        row
        for row in published
        if any(
            benchmark["paper_id"] != row["canonical_paper_id"]
            for benchmark in row["benchmark_rows"].values()
        )
    ]

    assert manifest["candidate_group_count"] == 150
    assert len(unresolved) == 1
    task_id = unresolved[0]["selected_task_ids"][0]
    assert unresolved[0]["benchmark_rows"][task_id]["paper_id"].startswith(
        "doi:10.5555/resolved."
    )
    assert unresolved[0]["human_review"] == {
        "curator_ids": [],
        "independent_attestation": False,
    }
    assert unresolved[0]["release_eligibility"] == "blocked"


def test_capacity_enrichment_rejects_nested_human_state(tmp_path: Path) -> None:
    bundle, queue, plan, assist = _setup(tmp_path)
    rows = _machine_rows(bundle, queue, assist)
    task_id = rows[0]["selected_task_ids"][0]
    rows[0]["benchmark_rows"][task_id]["rubric"] = {
        "criteria": ["Machine-generated criterion"],
        "adjudicators": ["fake-1", "fake-2"],
    }
    machine_input = tmp_path / "machine-enrichment.jsonl"
    _write_jsonl(machine_input, rows)
    output = tmp_path / "enrichment-package"

    with pytest.raises(RemediationError, match="benchmark proposal.*fields are invalid"):
        _generate(bundle, queue, plan, assist, machine_input, output)

    assert not output.exists()


def test_capacity_enrichment_rejects_query_string_credentials(tmp_path: Path) -> None:
    bundle, queue, plan, assist = _setup(tmp_path)
    rows = _machine_rows(bundle, queue, assist)
    rows[0]["external_evidence"][0]["source_url"] += "?token=secret"
    machine_input = tmp_path / "machine-enrichment.jsonl"
    _write_jsonl(machine_input, rows)
    output = tmp_path / "enrichment-package"

    with pytest.raises(RemediationError, match=r"credential-free HTTP\(S\) URL"):
        _generate(bundle, queue, plan, assist, machine_input, output)

    assert not output.exists()

    rows[0]["external_evidence"][0]["source_url"] = rows[1]["external_evidence"][0][
        "source_url"
    ]
    rows[0]["claim_evidence"]["usage_rights"]["evidence_ids"] = [
        rows[0]["external_evidence"][1]["evidence_id"]
    ]
    _write_jsonl(machine_input, rows)
    with pytest.raises(RemediationError, match="uses incompatible evidence kinds"):
        _generate(bundle, queue, plan, assist, machine_input, output)

    assert not output.exists()

    rows[0]["claim_evidence"]["usage_rights"]["evidence_ids"] = [
        rows[0]["external_evidence"][2]["evidence_id"]
    ]
    task_id = rows[0]["selected_task_ids"][0]
    original_url = rows[0]["provenance_rows"][task_id]["images"][0]["source_url"]
    rows[0]["provenance_rows"][task_id]["images"][0]["source_url"] = (
        "https://example.org/unrelated-figure"
    )
    _write_jsonl(machine_input, rows)
    with pytest.raises(RemediationError, match="no matching cited figure hash and URL"):
        _generate(bundle, queue, plan, assist, machine_input, output)
    assert not output.exists()

    rows[0]["provenance_rows"][task_id]["images"][0]["source_url"] = original_url
    rows[0]["provenance_rows"][task_id]["images"][0]["license"] = "MIT"
    _write_jsonl(machine_input, rows)
    with pytest.raises(RemediationError, match="no matching cited rights evidence"):
        _generate(bundle, queue, plan, assist, machine_input, output)
    assert not output.exists()

    rows[0]["provenance_rows"][task_id]["images"][0]["license"] = "CC-BY-4.0"
    rows[0]["benchmark_rows"][task_id]["primary_endpoint"] = False
    _write_jsonl(machine_input, rows)
    with pytest.raises(RemediationError, match="primary_endpoint differs from configured strata"):
        _generate(bundle, queue, plan, assist, machine_input, output)
    assert not output.exists()

    rows[0]["benchmark_rows"][task_id]["primary_endpoint"] = True
    unrelated_hash = "f" * 64
    rows[0]["external_evidence"].append(
        {
            **rows[0]["external_evidence"][1],
            "evidence_id": "unrelated-figure",
            "content_sha256": unrelated_hash,
        }
    )
    rows[0]["provenance_rows"][task_id]["images"][0]["sha256"] = unrelated_hash
    _write_jsonl(machine_input, rows)
    with pytest.raises(RemediationError, match="no matching cited figure hash and URL"):
        _generate(bundle, queue, plan, assist, machine_input, output)

    assert not output.exists()


def test_capacity_enrichment_rejects_source_workspace_output(
    tmp_path: Path,
) -> None:
    bundle, queue, plan, assist = _setup(tmp_path)
    rows = _machine_rows(bundle, queue, assist)
    machine_input = tmp_path / "machine-enrichment.jsonl"
    _write_jsonl(machine_input, rows)
    protected_roots = (plan, assist, Path(bundle["prepared"]["dataset_root"]))
    before = {
        root: {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        for root in protected_roots
    }

    for root in protected_roots:
        with pytest.raises(RemediationError, match="immutable input workspaces"):
            _generate(
                bundle,
                queue,
                plan,
                assist,
                machine_input,
                root / "nested" / "enrichment-package",
            )

    for root in protected_roots:
        after = {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        assert after == before[root]
        assert not (root / "nested").exists()

    malformed_workspace = tmp_path / "malformed-assist"
    malformed_manifest = malformed_workspace / CAPACITY_ASSIST_MANIFEST_FILENAME
    malformed_manifest.mkdir(parents=True)
    with pytest.raises(RemediationError, match="regular non-symlink file"):
        generate_capacity_enrichment_package(
            bundle["config"],
            bundle["prepare_manifest"],
            queue / "queue_manifest.json",
            plan / "capacity_plan_manifest.json",
            malformed_manifest,
            machine_input,
            malformed_workspace / "nested" / "enrichment-package",
            benchmark_schema=BENCHMARK_SCHEMA,
            provenance_schema=PROVENANCE_SCHEMA,
        )
    assert not (malformed_workspace / "nested").exists()


def test_curate_capacity_enrichment_cli_parser_and_routing(
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

    monkeypatch.setattr(cli, "generate_capacity_enrichment_package", fake_generate)
    code = cli.main(
        [
            "--config",
            str(config_path),
            "curate-capacity-enrichment",
            "--prepare-manifest",
            "prepare.json",
            "--queue-manifest",
            "queue/queue_manifest.json",
            "--capacity-plan-manifest",
            "plan/capacity_plan_manifest.json",
            "--capacity-assist-manifest",
            "assist/capacity_assist_manifest.json",
            "--machine-enrichment-jsonl",
            "machine.jsonl",
            "--output-dir",
            "enrichment",
        ]
    )

    assert code == 0
    assert captured["args"][0] == {"loaded": True}
    assert captured["args"][5] == Path("machine.jsonl")
    assert captured["kwargs"]["benchmark_schema"].name == "publication_benchmark_row.schema.json"


def test_curate_assemble_cli_routes_capacity_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    captured: dict = {}
    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: {"loaded": True})
    monkeypatch.setattr(cli, "_repo_root", lambda _path, _root: tmp_path)

    def fake_assemble(*args, **kwargs) -> dict:
        captured.update(args=args, kwargs=kwargs)
        return {"scope": "candidate"}

    monkeypatch.setattr(cli, "assemble_corrected_release", fake_assemble)
    code = cli.main(
        [
            "--config",
            str(config_path),
            "curate-assemble",
            "--prepare-manifest",
            "prepare.json",
            "--queue-manifest",
            "queue/queue_manifest.json",
            "--decisions-jsonl",
            "decisions.jsonl",
            "--curated-dataset-root",
            "curated",
            "--capacity-enrichment-manifest",
            "enrichment/machine_enrichment_manifest.json",
            "--machine-enrichment-jsonl",
            "machine.jsonl",
            "--output-dir",
            "release",
        ]
    )

    assert code == 0
    assert captured["kwargs"]["capacity_enrichment_manifest"] == Path(
        "enrichment/machine_enrichment_manifest.json"
    )
    assert captured["kwargs"]["machine_enrichment_jsonl"] == Path("machine.jsonl")
