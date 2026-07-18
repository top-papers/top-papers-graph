# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scireason.vlm_ab.audit import audit_benchmark
from scireason.vlm_ab.config import validate_experiment_config
from scireason.vlm_ab.prepare import prepare_experiment, verify_prepared_audit
from scireason.vlm_ab.remediation import (
    DecisionValidationError,
    RemediationError,
    assemble_corrected_release,
    generate_curator_queues,
)


REVISION_A = "a" * 40
REVISION_B = "b" * 40
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCHEMA = (
    REPOSITORY_ROOT
    / "experiments"
    / "vlm_ab_evaluation"
    / "schemas"
    / "publication_benchmark_row.schema.json"
)
PROVENANCE_SCHEMA = (
    REPOSITORY_ROOT
    / "experiments"
    / "vlm_ab_evaluation"
    / "schemas"
    / "publication_provenance_row.schema.json"
)


def _config(*, n_items: int = 1, require_gold: bool = False) -> dict:
    return validate_experiment_config(
        {
            "schema_version": 1,
            "experiment": {
                "id": "remediation-test",
                "seed": 41,
                "output_dir": "runs/remediation-test",
            },
            "benchmark": {
                "repo_id": "example/benchmark",
                "revision": REVISION_A,
                "data_file": "data/benchmark.jsonl",
                "provenance_file": "provenance.jsonl",
                "require_gold": require_gold,
            },
            "training_audit": {"sources": []},
            "models": {
                "base": {
                    "base_model": {"id": "example/base", "revision": REVISION_A},
                    "model_kwargs": {"device_map": "cpu"},
                },
                "tuned": {
                    "base_model": {"id": "example/base", "revision": REVISION_A},
                    "adapter": {"id": "example/adapter", "revision": REVISION_B},
                    "model_kwargs": {"device_map": "cpu"},
                },
            },
            "processor": {"id": "example/processor", "revision": REVISION_B},
            "generation": {"do_sample": False, "max_new_tokens": 64},
            "conditions": ["original"],
            "review": {
                "reviewer_ids": ["reviewer-1", "reviewer-2"],
                "reviews_per_item": 2,
                "primary_condition": "original",
            },
            "statistics": {
                "primary_strata": ["multimodal_hard"],
                "bootstrap_resamples": 100,
                "randomization_resamples": 100,
            },
            "power": {
                "n_items": n_items,
                "reviews_per_item": 2,
                "evaluable_fraction": 1.0,
                "intracluster_correlation": 0.0,
                "alpha": 0.05,
                "target_power": 0.8,
                "score_sd": 0.5,
                "target_effect": 0.2,
            },
        }
    )


def _source_row(index: int, images: list[str], *, prompt: str | None = None) -> dict:
    prompt = prompt or f"Extract the measured trend from scientific figure {index + 1}."
    return {
        "sample_id": f"sample-{index}",
        "benchmark_version": "task3_hf_benchmark_v1",
        "task_family": "task3_vlm_ab_generation",
        "language": "en",
        "split": "test",
        "topic": "synthetic",
        "case_id": f"case-{index}",
        "stratum": "multimodal_hard",
        "primary_endpoint": True,
        "paper_title": f"Synthetic paper {index}",
        "paper_id": f"doi:10.5555/remediation.{index}",
        "year": "2026",
        "evidence_kind": "figure",
        "page_hint": f"Figure {index + 1}",
        "model_task_prompt": prompt,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Use only supplied evidence."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *({"type": "image"} for _ in images),
                ],
            },
        ],
        "images": images,
        "reference_answer": f"The synthetic trend for paper {index} is increasing.",
    }


def _complete_provenance(row: dict, image_bytes: dict[str, bytes]) -> dict:
    return {
        "sample_id": row["sample_id"],
        "paper_id": row["paper_id"],
        "images": [
            {
                "image_path": image,
                "sha256": hashlib.sha256(image_bytes[image]).hexdigest(),
                "page": image_index + 1,
                "locator": f"Figure {image_index + 1}",
                "source_url": f"https://doi.org/10.5555/remediation.{row['sample_id']}",
                "license": "CC-BY-4.0",
                "verified_by": ["source-curator-1", "source-curator-2"],
                "citation": f"Synthetic paper {row['sample_id']}, Figure {image_index + 1}",
            }
            for image_index, image in enumerate(row["images"])
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prepare_bundle(
    root: Path,
    *,
    row_count: int = 2,
    images_per_row: int = 1,
    n_items: int = 1,
    invalid_last_row: bool = False,
    duplicate_prompts: bool = False,
    contaminated_training: bool = False,
    require_gold: bool = False,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    config = _config(n_items=n_items, require_gold=require_gold)
    dataset = root / "source-dataset"
    (dataset / "data").mkdir(parents=True)
    (dataset / "assets" / "images").mkdir(parents=True)
    image_bytes: dict[str, bytes] = {}
    rows: list[dict] = []
    shared_prompt = "Extract the shared trend from the supplied scientific figure."
    for row_index in range(row_count):
        images = [
            f"assets/images/sample-{row_index}-{image_index}.png"
            for image_index in range(images_per_row)
        ]
        for image_index, image in enumerate(images):
            payload = f"image-{row_index}-{image_index}".encode()
            image_bytes[image] = payload
            (dataset / image).write_bytes(payload)
        rows.append(
            _source_row(
                row_index,
                images,
                prompt=shared_prompt if duplicate_prompts else None,
            )
        )
    if invalid_last_row:
        rows[-1]["messages"][1]["content"] = [
            {"type": "text", "text": rows[-1]["model_task_prompt"]}
        ]
    provenance = [_complete_provenance(row, image_bytes) for row in rows]
    _write_jsonl(dataset / "data" / "benchmark.jsonl", rows)
    _write_jsonl(dataset / "provenance.jsonl", provenance)

    training_files: list[Path] = []
    if contaminated_training:
        training = root / "training.jsonl"
        _write_jsonl(
            training,
            [
                {
                    "paper_id": rows[0]["paper_id"],
                    "messages": [{"role": "user", "content": "A distinct training instruction."}],
                }
            ],
        )
        training_files.append(training)
    prepared = prepare_experiment(
        config,
        root,
        benchmark_dir=dataset,
        training_files=training_files,
        exploratory=True,
    )
    return {
        "config": config,
        "prepared": prepared,
        "prepare_manifest": Path(prepared["prepare_manifest"]),
        "rows": rows,
        "image_bytes": image_bytes,
    }


def _generate_queue(bundle: dict, output: Path) -> dict:
    return generate_curator_queues(
        bundle["config"],
        bundle["prepare_manifest"],
        output,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )


def _curated_replacements(bundle: dict, root: Path) -> tuple[Path, list[tuple[dict, dict]]]:
    root.mkdir(parents=True)
    for image, payload in bundle["image_bytes"].items():
        destination = root / image
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    replacements: list[tuple[dict, dict]] = []
    for row_index, source_row in enumerate(bundle["rows"]):
        benchmark_row = copy.deepcopy(source_row)
        benchmark_row["split_provenance"] = {
            "paper_holdout": True,
            "source_holdout": True,
            "creator_holdout": True,
            "training_overlap_checked": True,
            "source_document_id": f"source-document-{row_index}",
            "creator_group_id": f"creator-group-{row_index}",
        }
        provenance_row = _complete_provenance(benchmark_row, bundle["image_bytes"])
        replacements.append((benchmark_row, provenance_row))
    return root, replacements


def _completed_decisions(
    queue_dir: Path,
    replacements: list[tuple[dict, dict]],
    *,
    excluded: set[int] | None = None,
) -> list[dict]:
    decisions = _read_jsonl(queue_dir / "decision_template.jsonl")
    for index, decision in enumerate(decisions):
        decision["status"] = "complete"
        decision["reviewed_by"] = ["curator-1", "curator-2"]
        decision["notes"] = "Independently checked against the source article."
        if index in (excluded or set()):
            decision["disposition"] = "exclude"
            decision["exclusion_reason"] = "Insufficient independent evidence."
        else:
            decision["disposition"] = "retain"
            decision["benchmark_row"], decision["provenance_row"] = copy.deepcopy(
                replacements[index]
            )
    return decisions


def _assemble(
    bundle: dict,
    queue_dir: Path,
    decisions_path: Path,
    curated_root: Path,
    output: Path,
) -> dict:
    return assemble_corrected_release(
        bundle["config"],
        bundle["prepare_manifest"],
        queue_dir / "queue_manifest.json",
        decisions_path,
        curated_root,
        output,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )


def test_queue_generation_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle")
    first_dir = tmp_path / "queue-first"
    second_dir = tmp_path / "queue-second"

    first = _generate_queue(bundle, first_dir)
    first_bytes = {path.name: path.read_bytes() for path in first_dir.iterdir()}
    repeated = _generate_queue(bundle, first_dir)
    second = _generate_queue(bundle, second_dir)

    assert repeated == first == second
    assert first_bytes == {path.name: path.read_bytes() for path in first_dir.iterdir()}
    assert first_bytes == {path.name: path.read_bytes() for path in second_dir.iterdir()}
    assert first["task_count"] == 2
    assert first["policy"]["warning_codes_must_be_zero"] == [
        "duplicate_normalized_prompt",
        "within_row_duplicate_image_bytes",
    ]
    assert str(tmp_path) not in json.dumps(first)

    tasks = _read_jsonl(first_dir / "tasks.jsonl")
    templates = _read_jsonl(first_dir / "decision_template.jsonl")
    assert [task["source_row_index"] for task in tasks] == [0, 1]
    assert [task["original_row"] for task in tasks] == bundle["rows"]
    assert all(template["status"] == "pending" for template in templates)
    assert all(template["disposition"] is None for template in templates)
    assert all(template["benchmark_row"] is None for template in templates)

    (first_dir / "tasks.jsonl").write_bytes(first_bytes["tasks.jsonl"] + b"\n")
    with pytest.raises(RemediationError, match="non-identical"):
        _generate_queue(bundle, first_dir)


@pytest.mark.parametrize("audit_version", [1, 2])
def test_retained_prepare_bundle_uses_its_audit_contract(
    tmp_path: Path, audit_version: int
) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle")
    prepared = bundle["prepared"]
    legacy_report = audit_benchmark(
        bundle["rows"],
        prepared["dataset_root"],
        provenance_rows=_read_jsonl(Path(prepared["provenance_file"])),
        primary_strata=bundle["config"]["statistics"]["primary_strata"],
        audit_version=audit_version,
    )
    audit_path = Path(prepared["audit_json"])
    audit_path.write_text(
        json.dumps(legacy_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = bundle["prepare_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audit_json_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert verify_prepared_audit(bundle["config"], manifest, manifest_path.parent) == legacy_report
    assert _generate_queue(bundle, tmp_path / "legacy-queue")["task_count"] == 2


def test_queue_and_nonfrozen_image_tampering_are_rejected(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "queue-bundle")
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions_path = tmp_path / "decisions.jsonl"
    decisions = _completed_decisions(queue_dir, replacements)
    decisions[0]["source_row_sha256"] = "0" * 64
    _write_jsonl(decisions_path, decisions)
    with pytest.raises(DecisionValidationError, match="tampered source_row_sha256"):
        _assemble(
            bundle,
            queue_dir,
            decisions_path,
            curated_root,
            tmp_path / "binding-release",
        )

    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))
    tasks_path = queue_dir / "tasks.jsonl"
    tasks_path.write_bytes(tasks_path.read_bytes() + b"\n")

    with pytest.raises(RemediationError, match="tampered"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")

    nonfrozen = _prepare_bundle(
        tmp_path / "nonfrozen-bundle",
        invalid_last_row=True,
    )
    assert "sample-1" not in nonfrozen["prepared"]["frozen_sample_ids"]
    nonfrozen_image = (
        Path(nonfrozen["prepared"]["dataset_root"]) / nonfrozen["rows"][1]["images"][0]
    )
    nonfrozen_image.write_bytes(b"tampered-nonfrozen-image")
    with pytest.raises(RemediationError, match="audited image hash changed"):
        _generate_queue(nonfrozen, tmp_path / "nonfrozen-queue")


@pytest.mark.parametrize("case", ["pending", "missing", "one-reviewer", "invalid-status"])
def test_pending_missing_and_one_reviewer_decisions_are_rejected(tmp_path: Path, case: str) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle")
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    if case == "pending":
        decisions = _read_jsonl(queue_dir / "decision_template.jsonl")
        expected = "pending"
    else:
        decisions = _completed_decisions(queue_dir, replacements)
        if case == "missing":
            decisions.pop()
            expected = "missing decisions"
        elif case == "one-reviewer":
            decisions[0]["reviewed_by"] = ["only-curator"]
            expected = "two distinct"
        else:
            decisions[0]["status"] = "approved"
            expected = "status must be complete"
    decisions_path = tmp_path / f"{case}.jsonl"
    _write_jsonl(decisions_path, decisions)

    with pytest.raises(DecisionValidationError, match=expected):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / f"release-{case}")


def test_full_corrected_release_is_relocatable_and_idempotent(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", n_items=2)
    queue_dir = tmp_path / "queue"
    queue_manifest = _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    replacements[0][0]["sample_id"] = "sample-z"
    replacements[0][1]["sample_id"] = "sample-z"
    replacements[1][0]["sample_id"] = "sample-a"
    replacements[1][1]["sample_id"] = "sample-a"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))
    output = tmp_path / "release"

    manifest = _assemble(bundle, queue_dir, decisions_path, curated_root, output)
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    repeated = _assemble(bundle, queue_dir, decisions_path, curated_root, output)

    assert repeated == manifest
    assert before == {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert manifest["scope"] == "validated_benchmark_release_candidate"
    assert manifest["publication_ready"] is False
    assert manifest["technical_audit_passed"] is True
    assert manifest["artifact_version"] == 2
    assert manifest["queue_fingerprint"] == queue_manifest["queue_fingerprint"]
    assert manifest["counts"] == {
        "source_tasks": 2,
        "retained_rows": 2,
        "excluded_rows": 0,
        "provenance_rows": 2,
        "copied_assets": 2,
        "unique_primary_papers": 2,
    }
    assert manifest["reviewer_ids"] == ["curator-1", "curator-2"]
    assert manifest["training_lineage"] == {
        "provided": False,
        "validated_in_candidate_audit": False,
        "required_for_strict_prepare": True,
        "unresolved_domains": [
            "paper_ids",
            "source_documents",
            "creator_groups",
            "image_bytes",
            "prompts",
        ],
    }
    assert manifest["remaining_requirements"] == [
        "publish immutable benchmark revision",
        "publish adapter lineage/revision",
        "create new experiment IDs and plan",
        "pass strict prepare",
    ]
    assert (output / "data" / "task3_vlm_generation.jsonl").is_file()
    assert (output / "article_image_sources.jsonl").is_file()
    assert (output / "audit" / "benchmark_audit.json").is_file()
    assert (output / "audit" / "benchmark_audit.md").is_file()
    candidate_audit = json.loads(
        (output / "audit" / "benchmark_audit.json").read_text(encoding="utf-8")
    )
    assert candidate_audit["status"] == "pass"
    assert candidate_audit["technical_audit_passed"] is True
    assert candidate_audit["publication_ready"] is False
    assert candidate_audit["scope"] == "validated_benchmark_release_candidate"
    candidate_markdown = (output / "audit" / "benchmark_audit.md").read_text(encoding="utf-8")
    assert "TECHNICAL PASS / PUBLICATION BLOCKED" in candidate_markdown
    assert "**FAIL:" not in candidate_markdown
    assert manifest["decisions_sha256"] == hashlib.sha256(decisions_path.read_bytes()).hexdigest()
    assert all(not Path(record["path"]).is_absolute() for record in manifest["output_files"])
    assert [
        row["sample_id"] for row in _read_jsonl(output / "data" / "task3_vlm_generation.jsonl")
    ] == ["sample-a", "sample-z"]
    for record in manifest["output_files"]:
        payload = (output / record["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert len(payload) == record["size_bytes"]


def test_exact_provenance_image_order_is_required(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1, images_per_row=2)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions = _completed_decisions(queue_dir, replacements)
    decisions[0]["provenance_row"]["images"].reverse()
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, decisions)

    with pytest.raises(DecisionValidationError, match="match benchmark images in order"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_declared_curated_image_hash_is_enforced(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions = _completed_decisions(queue_dir, replacements)
    decisions[0]["provenance_row"]["images"][0]["sha256"] = "0" * 64
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, decisions)

    with pytest.raises(DecisionValidationError, match="does not match bytes"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_fresh_audit_rejects_training_contamination_and_frozen_warnings(
    tmp_path: Path,
) -> None:
    contaminated = _prepare_bundle(
        tmp_path / "contaminated",
        row_count=1,
        contaminated_training=True,
    )
    contaminated_queue = tmp_path / "contaminated-queue"
    _generate_queue(contaminated, contaminated_queue)
    tasks = _read_jsonl(contaminated_queue / "tasks.jsonl")
    assert tasks[0]["training_overlap_paper_ids"] == ["doi:10.5555/remediation.0"]
    curated_root, replacements = _curated_replacements(
        contaminated, tmp_path / "contaminated-curated"
    )
    decisions_path = tmp_path / "contaminated-decisions.jsonl"
    _write_jsonl(
        decisions_path,
        _completed_decisions(contaminated_queue, replacements),
    )
    with pytest.raises(DecisionValidationError, match="training_paper_overlap"):
        _assemble(
            contaminated,
            contaminated_queue,
            decisions_path,
            curated_root,
            tmp_path / "contaminated-release",
        )

    duplicated = _prepare_bundle(
        tmp_path / "duplicated",
        duplicate_prompts=True,
    )
    duplicated_queue = tmp_path / "duplicated-queue"
    _generate_queue(duplicated, duplicated_queue)
    duplicated_curated, duplicated_replacements = _curated_replacements(
        duplicated, tmp_path / "duplicated-curated"
    )
    duplicated_decisions = tmp_path / "duplicated-decisions.jsonl"
    _write_jsonl(
        duplicated_decisions,
        _completed_decisions(duplicated_queue, duplicated_replacements),
    )
    with pytest.raises(DecisionValidationError, match="duplicate_normalized_prompt"):
        _assemble(
            duplicated,
            duplicated_queue,
            duplicated_decisions,
            duplicated_curated,
            tmp_path / "duplicated-release",
        )


def test_minimum_primary_paper_gate_is_enforced(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", n_items=2)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions = _completed_decisions(queue_dir, replacements, excluded={1})
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, decisions)

    with pytest.raises(DecisionValidationError, match="1 < 2"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_prepare_configuration_binding_cannot_be_substituted(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", n_items=2)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(RemediationError, match="different configuration"):
        assemble_corrected_release(
            _config(n_items=1),
            bundle["prepare_manifest"],
            queue_dir / "queue_manifest.json",
            decisions_path,
            curated_root,
            tmp_path / "release",
            benchmark_schema=BENCHMARK_SCHEMA,
            provenance_schema=PROVENANCE_SCHEMA,
        )


def test_noncanonical_schema_override_is_rejected(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle")
    permissive = tmp_path / "permissive.schema.json"
    permissive.write_text(
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RemediationError, match="byte-identical canonical"):
        generate_curator_queues(
            bundle["config"],
            bundle["prepare_manifest"],
            tmp_path / "queue",
            benchmark_schema=permissive,
            provenance_schema=PROVENANCE_SCHEMA,
        )


@pytest.mark.parametrize(
    "bad_path, expected",
    [
        ("audit/benchmark_audit.json.", "trailing-dot/space"),
        ("assets/CON.png", "reserved Windows device"),
        ("assets/COM\u00b9.png", "reserved Windows device"),
        ("assets/CONIN$.png", "reserved Windows device"),
        ("assets/NUL .txt", "reserved Windows device"),
        ("audit/benchmark_audit.json/nested.png", "collides with a release artifact"),
        ("images/outside-prefix.png", "assets/images"),
    ],
)
def test_windows_aliases_and_artifact_prefixes_are_rejected(
    tmp_path: Path, bad_path: str, expected: str
) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    replacements[0][0]["images"] = [bad_path]
    replacements[0][1]["images"][0]["image_path"] = bad_path
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match=expected):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_conflicting_identifier_fields_cannot_mask_paper_identity(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=2)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    for benchmark_row, provenance_row in replacements:
        benchmark_row["doi"] = "10.5555/conflicting-shared-id"
        provenance_row["doi"] = "10.5555/conflicting-shared-id"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match="conflict with canonical paper_id"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_normalized_paper_id_aliases_cannot_overwrite_conflicts(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    benchmark_row, provenance_row = replacements[0]
    replacements[0] = (
        {"paper-id": "doi:10.5555/conflicting-alias", **benchmark_row},
        {"paper-id": "doi:10.5555/conflicting-alias", **provenance_row},
    )
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match="conflict with canonical paper_id"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


@pytest.mark.parametrize("alias_key", ["paper\u200b_id", "paper%5Fid", "paper%2\u200b55fid"])
def test_invisible_paper_alias_key_is_rejected(tmp_path: Path, alias_key: str) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    replacements[0][0][alias_key] = "doi:10.5555/hidden-conflict"
    replacements[0][1][alias_key] = "doi:10.5555/hidden-conflict"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match="unsafe characters"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_nested_and_unsafe_paper_alias_values_are_rejected(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    benchmark_row, provenance_row = replacements[0]
    benchmark_row["doi"] = f"{benchmark_row['paper_id'][4:]}\u200bhidden"
    provenance_row["payload"] = {"paper_id": "doi:10.5555/nested-conflict"}
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match="safe canonical identifiers"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_release_manifest_normalizes_reviewer_roster(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=2, n_items=2)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions = _completed_decisions(queue_dir, replacements)
    decisions[0]["reviewed_by"] = ["Alice", "Bob"]
    decisions[1]["reviewed_by"] = ["ALICE", "BOB"]
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, decisions)

    manifest = _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")

    assert manifest["reviewer_ids"] == ["alice", "bob"]


def test_percent_encoded_unicode_cannot_inflate_primary_paper_count(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=2, n_items=2)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    encoded_first_paper = (
        "doi%3A%EF%BC%91%EF%BC%90.%EF%BC%95%EF%BC%95%EF%BC%95%EF%BC%95%2Fremediation.0"
    )
    replacements[1][0]["paper_id"] = encoded_first_paper
    replacements[1][1]["paper_id"] = encoded_first_paper
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match="canonical representation"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_normalized_verifier_aliases_are_not_independent(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    replacements[0][1]["images"][0]["verified_by"] = ["same-verifier", "SAME-VERIFIER"]
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match="requires two distinct identifiers"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_invisible_reviewer_aliases_are_rejected(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions = _completed_decisions(queue_dir, replacements)
    decisions[0]["reviewed_by"] = ["same-reviewer", "same-\u200breviewer"]
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, decisions)

    with pytest.raises(DecisionValidationError, match="invisible or control"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_invisible_paper_identity_cannot_reach_minimum_paper_gate(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=2, n_items=2)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    replacements[1][0]["paper_id"] = "paper-\u034fidentity"
    replacements[1][1]["paper_id"] = "paper-\u034fidentity"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match="invisible or control"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")


def test_required_gold_needs_substantive_gold_and_rubric(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1, require_gold=True)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    with pytest.raises(DecisionValidationError, match="requires adjudicated gold_answer"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "missing-gold")

    benchmark_row = replacements[0][0]
    benchmark_row["gold_answer"] = {
        "answer": "The measured trend increases.",
        "evidence_used": ["The visual panel"],
        "visual_facts": ["The plotted series rises."],
        "temporal_facts": ["Later values exceed earlier values."],
        "uncertainty": None,
        "missing_evidence": None,
    }
    benchmark_row["rubric"] = {
        "criteria": ["The answer states the direction of change."],
        "adjudicators": ["adjudicator-1", "adjudicator-2"],
    }
    _write_jsonl(decisions_path, _completed_decisions(queue_dir, replacements))

    manifest = _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")
    assert manifest["technical_audit_passed"] is True


def test_duplicate_json_keys_in_decisions_are_rejected(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue_dir = tmp_path / "queue"
    _generate_queue(bundle, queue_dir)
    curated_root, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions = _completed_decisions(queue_dir, replacements)
    payload = json.dumps(decisions[0], separators=(",", ":"), sort_keys=True)
    payload = payload.replace(
        '"status":"complete"',
        '"status":"complete","status":"complete"',
    )
    decisions_path = tmp_path / "duplicate-key.jsonl"
    decisions_path.write_text(payload + "\n", encoding="utf-8")

    with pytest.raises(RemediationError, match="duplicate JSON key 'status'"):
        _assemble(bundle, queue_dir, decisions_path, curated_root, tmp_path / "release")
