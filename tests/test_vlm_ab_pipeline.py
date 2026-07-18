# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import scireason.vlm_ab.prepare as prepare_module
from scireason.vlm_ab.cli import (
    _run_fingerprint,
    command_aggregate,
    command_blind,
    command_infer,
    command_plan,
)
from scireason.vlm_ab.config import ExperimentConfigError, validate_experiment_config
from scireason.vlm_ab.prepare import (
    PublicationGateError,
    _apply_publication_schema_failure,
    _load_strict_json_object,
    _validate_adapter_base_metadata,
    _validate_publication_schemas,
    _validate_training_lineage_schema,
    prepare_experiment,
    resolve_prepare_manifest,
    verify_prepare_manifest,
)


REVISION_A = "a" * 40
REVISION_B = "b" * 40
REVISION_C = "c" * 40


def _config() -> dict:
    return validate_experiment_config(
        {
            "schema_version": 1,
            "experiment": {
                "id": "pipeline-smoke",
                "seed": 73,
                "output_dir": "runs/pipeline-smoke",
            },
            "benchmark": {
                "repo_id": "example/benchmark",
                "revision": REVISION_A,
                "data_file": "data/benchmark.jsonl",
                "provenance_file": "provenance.jsonl",
                "require_gold": False,
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
            "conditions": ["original", "text_only", "shuffled_images"],
            "review": {
                "reviewer_ids": ["reviewer-1", "reviewer-2"],
                "reviews_per_item": 2,
                "primary_condition": "original",
            },
            "statistics": {
                "bootstrap_resamples": 100,
                "randomization_resamples": 100,
            },
            "power": {
                "n_items": 1,
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


def _dataset(root: Path) -> Path:
    dataset = root / "dataset"
    (dataset / "data").mkdir(parents=True)
    (dataset / "assets" / "images").mkdir(parents=True)
    image = dataset / "assets" / "images" / "page.png"
    image.write_bytes(b"publication-smoke-image")
    row = {
        "sample_id": "sample-1",
        "benchmark_version": "task3_hf_benchmark_v1",
        "task_family": "task3_vlm_ab_generation",
        "language": "en",
        "split": "test",
        "topic": "smoke",
        "case_id": "case-1",
        "stratum": "multimodal_hard",
        "primary_endpoint": True,
        "paper_title": "Smoke paper",
        "paper_id": "doi:10.5555/smoke",
        "year": "2025",
        "evidence_kind": "figure",
        "page_hint": "Fig. 1",
        "model_task_prompt": "Extract the trend shown in Figure 1.",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Use evidence."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the trend shown in Figure 1."},
                    {"type": "image"},
                ],
            },
        ],
        "images": ["assets/images/page.png"],
    }
    (dataset / "data" / "benchmark.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    provenance = {
        "sample_id": row["sample_id"],
        "paper_id": row["paper_id"],
        "image_path": row["images"][0],
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }
    (dataset / "provenance.jsonl").write_text(json.dumps(provenance) + "\n", encoding="utf-8")
    return dataset


def _review_export(public: dict, preference: str) -> dict:
    assignment_ids = [item["assignment_id"] for item in public["assignments"]]
    return {
        "artifact_version": public["artifact_version"],
        "experiment_id": public["experiment_id"],
        "study_fingerprint": public["study_fingerprint"],
        "reviewer_id": public["reviewer_id"],
        "assignments": assignment_ids,
        "responses": [
            {
                "assignment_id": assignment_id,
                "overall_preference": preference,
                "evidence_preference": "tie",
                "visual_preference": "tie",
                "temporal_preference": "skip",
                "left_error_tags": [],
                "right_error_tags": [],
                "confidence": 4,
                "comments": "smoke",
            }
            for assignment_id in assignment_ids
        ],
    }


def test_mock_pipeline_reaches_blind_review_and_aggregation(tmp_path: Path) -> None:
    config = _config()
    dataset = _dataset(tmp_path)
    prepared = prepare_experiment(
        config,
        tmp_path,
        benchmark_dir=dataset,
        exploratory=True,
    )
    verify_prepare_manifest(json.loads(Path(prepared["prepare_manifest"]).read_text()))

    base_args = argparse.Namespace(
        exploratory=True,
        arm="base",
        backend="mock",
        limit=None,
    )
    base_inference = command_infer(base_args, config, tmp_path)
    assert set(base_inference["arms"]) == {"base"}
    tuned_args = argparse.Namespace(
        exploratory=True,
        arm="tuned",
        backend="mock",
        limit=None,
    )
    inference = command_infer(tuned_args, config, tmp_path)
    assert inference["arms"]["base"]["successful_rows"] == 3
    assert inference["arms"]["tuned"]["successful_rows"] == 3

    blind_args = argparse.Namespace(
        exploratory=True,
        reviewers=None,
        reviews_per_item=None,
    )
    blind = command_blind(blind_args, config, tmp_path)
    blind_manifest_path = tmp_path / "runs" / "pipeline-smoke" / "blind_review_manifest.json"
    blind_manifest_path.unlink()
    recovered_blind = command_blind(blind_args, config, tmp_path)
    assert recovered_blind["study_fingerprint"] == blind["study_fingerprint"]
    assert command_blind(blind_args, config, tmp_path) == recovered_blind
    blind = recovered_blind
    secret = tmp_path / "runs" / "pipeline-smoke" / "blind_review" / "owner_only"
    assert (secret / "randomization_secret.txt").exists()
    assert Path(blind["owner_mapping"]).parent == secret

    reviews = tmp_path / "reviews"
    reviews.mkdir()
    for index, package in enumerate(blind["reviewer_packages"].values()):
        public = json.loads(Path(package["assignment"]).read_text(encoding="utf-8"))
        exported = _review_export(public, "left" if index == 0 else "right")
        (reviews / f"review-{index}.json").write_text(json.dumps(exported), encoding="utf-8")

    aggregate_args = argparse.Namespace(
        exploratory=True,
        review_file=[],
        reviews_dir=reviews,
        allow_incomplete=False,
    )
    prediction = tmp_path / "runs" / "pipeline-smoke" / "predictions" / "base.jsonl"
    prediction_bytes = prediction.read_bytes()
    prediction.write_bytes(prediction_bytes + b"\n")
    with pytest.raises(RuntimeError, match="prediction bytes differ"):
        command_aggregate(aggregate_args, config, tmp_path)
    prediction.write_bytes(prediction_bytes)

    prediction_manifest = Path(f"{prediction}.manifest.json")
    prediction_manifest_bytes = prediction_manifest.read_bytes()
    edited_rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    edited_rows[0]["parsed_response"]["answer"] = "Edited after blinded review."
    edited_rows[0]["raw_response"] = json.dumps(edited_rows[0]["parsed_response"])
    edited_bytes = "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in edited_rows
    ).encode("utf-8")
    prediction.write_bytes(edited_bytes)
    edited_manifest = json.loads(prediction_manifest_bytes)
    edited_manifest["output_sha256"] = hashlib.sha256(edited_bytes).hexdigest()
    prediction_manifest.write_text(json.dumps(edited_manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different prediction bytes"):
        command_aggregate(aggregate_args, config, tmp_path)
    prediction.write_bytes(prediction_bytes)
    prediction_manifest.write_bytes(prediction_manifest_bytes)

    owner = secret / "owner_only.json"
    owner_bytes = owner.read_bytes()
    owner.write_bytes(owner_bytes + b"\n")
    with pytest.raises(RuntimeError, match="owner mapping bytes differ"):
        command_aggregate(aggregate_args, config, tmp_path)
    owner.write_bytes(owner_bytes)

    injected = secret.parent / "public" / "injected.txt"
    injected.write_text("unlisted reviewer handoff file", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unlisted or missing files"):
        command_aggregate(aggregate_args, config, tmp_path)
    injected.unlink()

    result = command_aggregate(aggregate_args, config, tmp_path)
    assert result["publication_artifacts_ready"] is False
    assert result["superiority_claim_supported"] is False
    assert Path(result["report"]).exists()
    assert Path(result["figure"]).exists()


def test_prepare_hash_verification_detects_image_tampering(tmp_path: Path) -> None:
    config = _config()
    dataset = _dataset(tmp_path)
    prepared = prepare_experiment(
        config,
        tmp_path,
        benchmark_dir=dataset,
        exploratory=True,
    )
    manifest = json.loads(Path(prepared["prepare_manifest"]).read_text(encoding="utf-8"))
    (Path(prepared["dataset_root"]) / "assets" / "images" / "page.png").write_bytes(b"tampered")
    with pytest.raises(PublicationGateError, match="image hash changed"):
        verify_prepare_manifest(manifest)


def test_prepare_hash_verification_detects_diagnostic_tampering(tmp_path: Path) -> None:
    prepared = prepare_experiment(
        _config(),
        tmp_path,
        benchmark_dir=_dataset(tmp_path),
        exploratory=True,
    )
    manifest = json.loads(Path(prepared["prepare_manifest"]).read_text(encoding="utf-8"))
    original_manifest = copy.deepcopy(manifest)
    markdown_path = Path(prepared["audit_markdown"])
    markdown_bytes = markdown_path.read_bytes()
    markdown_path.write_text("tampered diagnostics", encoding="utf-8")

    with pytest.raises(PublicationGateError, match="artifact hash changed"):
        verify_prepare_manifest(manifest)
    markdown_path.write_bytes(markdown_bytes)
    manifest.pop("audit_markdown_sha256")
    with pytest.raises(PublicationGateError, match="incomplete file hash"):
        verify_prepare_manifest(manifest)

    missing_markdown = copy.deepcopy(original_manifest)
    missing_markdown["artifact_paths"].pop("audit_markdown")
    missing_markdown.pop("audit_markdown")
    missing_markdown.pop("audit_markdown_sha256")
    with pytest.raises(PublicationGateError, match="required artifact path"):
        verify_prepare_manifest(missing_markdown)

    missing_schema_binding = copy.deepcopy(original_manifest)
    missing_schema_binding.pop("publication_schema_hashes")
    with pytest.raises(PublicationGateError, match="schema hashes are invalid"):
        verify_prepare_manifest(missing_schema_binding)


def test_downstream_recomputes_audit_instead_of_trusting_edited_status(tmp_path: Path) -> None:
    config = _config()
    prepared = prepare_experiment(
        config,
        tmp_path,
        benchmark_dir=_dataset(tmp_path),
        exploratory=True,
    )
    manifest_path = Path(prepared["prepare_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_path = Path(prepared["audit_json"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["status"] = "pass"
    audit["publication_ready"] = True
    audit["critical_findings"] = []
    audit["summary"]["total_rows"] = 999
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest["audit_json_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest["benchmark_audit_status"] = "pass"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PublicationGateError, match="fresh recomputation"):
        command_infer(
            argparse.Namespace(
                exploratory=True,
                arm="base",
                backend="mock",
                limit=None,
            ),
            config,
            tmp_path,
        )


def test_run_fingerprint_binds_preregistered_plan_bytes() -> None:
    config = _config()
    prepared = {
        "code_provenance": {"source_fingerprint": "a" * 64},
        "frozen_benchmark_sha256": "b" * 64,
        "audit_json_sha256": "c" * 64,
        "power_plan_sha256": "d" * 64,
    }
    first = _run_fingerprint(config, prepared)
    second = _run_fingerprint(config, {**prepared, "power_plan_sha256": "e" * 64})
    assert first != second


def test_strict_local_benchmark_override_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PublicationGateError, match="local benchmark overrides"):
        prepare_experiment(_config(), tmp_path, benchmark_dir=_dataset(tmp_path))


def test_strict_local_training_override_is_rejected(tmp_path: Path) -> None:
    training = tmp_path / "training.jsonl"
    training.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PublicationGateError, match="training-file overrides"):
        prepare_experiment(
            _config(),
            tmp_path,
            benchmark_dir=_dataset(tmp_path),
            training_files=[training],
        )


def test_strict_schema_gate_rejects_legacy_flat_provenance(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    rows = [
        json.loads(line) for line in (dataset / "data" / "benchmark.jsonl").read_text().splitlines()
    ]
    rows[0]["split_provenance"] = {
        "paper_holdout": True,
        "source_holdout": True,
        "creator_holdout": True,
        "training_overlap_checked": True,
        "source_document_id": "source-document-1",
        "creator_group_id": "creator-group-1",
    }
    provenance = [json.loads((dataset / "provenance.jsonl").read_text())]
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(PublicationGateError, match="strict provenance row 0"):
        _validate_publication_schemas(rows, provenance, repo_root)

    provenance[0] = {
        "sample_id": rows[0]["sample_id"],
        "paper_id": rows[0]["paper_id"],
        "images": [
            {
                "image_path": rows[0]["images"][0],
                "sha256": provenance[0]["sha256"],
                "page": 1,
                "locator": "Figure 1",
                "source_url": "https://doi.org/10.5555/smoke",
                "license": "CC-BY-4.0",
                "verified_by": ["curator-1", "curator-2"],
            }
        ],
    }
    _validate_publication_schemas(rows, provenance, repo_root)

    provenance[0]["images"][0]["verified_by"] = ["Verifier", "verifier"]
    with pytest.raises(PublicationGateError, match="normalized-distinct"):
        _validate_publication_schemas(rows, provenance, repo_root)
    provenance[0]["images"][0]["verified_by"] = ["curator-1", "curator-2"]

    rows[0]["messages"][1]["content"].append(
        {"type": "video", "url": "https://example.invalid/video"}
    )
    with pytest.raises(PublicationGateError, match="strict benchmark row 0"):
        _validate_publication_schemas(rows, provenance, repo_root)

    rows[0]["messages"][1]["content"].pop()
    rows[0]["paper-id"] = "doi:10.5555/conflicting-alias"
    with pytest.raises(PublicationGateError, match="conflicting paper identity"):
        _validate_publication_schemas(rows, provenance, repo_root)

    rows[0].pop("paper-id")
    rows[0]["paper\u200b_id"] = "doi:10.5555/hidden-conflicting-alias"
    with pytest.raises(PublicationGateError, match="conflicting paper identity"):
        _validate_publication_schemas(rows, provenance, repo_root)
    rows[0].pop("paper\u200b_id")

    rows[0]["paper%5Fid"] = "doi:10.5555/encoded-conflicting-alias"
    with pytest.raises(PublicationGateError, match="conflicting paper identity"):
        _validate_publication_schemas(rows, provenance, repo_root)
    rows[0].pop("paper%5Fid")

    rows[0]["paper%2\u200b55fid"] = "doi:10.5555/encoded-after-filter"
    with pytest.raises(PublicationGateError, match="conflicting paper identity"):
        _validate_publication_schemas(rows, provenance, repo_root)
    rows[0].pop("paper%2\u200b55fid")

    rows[0]["payload"] = {"paper_id": "doi:10.5555/nested-conflict"}
    with pytest.raises(PublicationGateError, match="conflicting paper identity"):
        _validate_publication_schemas(rows, provenance, repo_root)
    rows[0].pop("payload")

    rows[0]["paper_id"] = "DOI:10.5555/SMOKE"
    with pytest.raises(PublicationGateError, match="not in canonical form"):
        _validate_publication_schemas(rows, provenance, repo_root)
    rows[0]["paper_id"] = "doi:10.5555/smoke"

    provenance[0]["paper-id"] = "doi:10.5555/conflicting-alias"
    with pytest.raises(PublicationGateError, match="strict provenance row 0"):
        _validate_publication_schemas(rows, provenance, repo_root)


@pytest.mark.parametrize(
    "reviewers",
    [
        ["reviewer-a", "REVIEWER-A"],
        ["reviewer-a", "reviewer-\u034fa"],
        ["reviewer-a", "reviewer-\u180ba"],
    ],
)
def test_config_rejects_spoofed_reviewer_identities(reviewers: list[str]) -> None:
    config = _config()
    config["review"]["reviewer_ids"] = reviewers

    with pytest.raises(ExperimentConfigError, match="reviewer_ids"):
        validate_experiment_config(config)


def test_config_requires_a_distinct_lineage_attestation_revision() -> None:
    config = _config()
    config["training_audit"] = {
        "require_lineage_manifest": True,
        "sources": [
            {
                "repo_id": "example/adapter",
                "repo_type": "model",
                "revision": REVISION_B,
                "files": ["artifacts/data/training.jsonl"],
            }
        ],
        "lineage_manifest": {
            "repo_id": "example/adapter",
            "repo_type": "model",
            "revision": REVISION_C,
            "file": "artifacts/training_lineage_manifest.json",
        },
    }

    validated = validate_experiment_config(copy.deepcopy(config))
    assert validated["models"]["tuned"]["adapter"]["revision"] == REVISION_B
    assert validated["training_audit"]["lineage_manifest"]["revision"] == REVISION_C

    self_referential = copy.deepcopy(config)
    self_referential["training_audit"]["lineage_manifest"]["revision"] = REVISION_B
    with pytest.raises(ExperimentConfigError, match="distinct immutable attestation revision"):
        validate_experiment_config(self_referential)

    wrong_repository = copy.deepcopy(config)
    wrong_repository["training_audit"]["lineage_manifest"]["repo_id"] = "example/attestations"
    with pytest.raises(ExperimentConfigError, match="evaluated adapter repository"):
        validate_experiment_config(wrong_repository)

    missing_adapter_source = copy.deepcopy(config)
    missing_adapter_source["training_audit"]["sources"][0].update(
        {"repo_id": "example/training", "repo_type": "dataset"}
    )
    with pytest.raises(ExperimentConfigError, match="include the evaluated adapter revision"):
        validate_experiment_config(missing_adapter_source)

    optional_lineage = copy.deepcopy(config)
    optional_lineage["training_audit"]["require_lineage_manifest"] = False
    validate_experiment_config(optional_lineage)
    optional_lineage["training_audit"]["lineage_manifest"]["revision"] = REVISION_B
    with pytest.raises(ExperimentConfigError, match="distinct immutable attestation revision"):
        validate_experiment_config(optional_lineage)


@pytest.mark.parametrize("n_items", [True, 0, -1, 1.5, "240"])
def test_config_requires_positive_integer_power_n_items(n_items: object) -> None:
    config = _config()
    config["power"]["n_items"] = n_items

    with pytest.raises(ExperimentConfigError, match="power.n_items"):
        validate_experiment_config(config)


def test_download_inputs_fetches_adapter_r_and_distinct_attestation_m(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["training_audit"] = {
        "require_lineage_manifest": True,
        "sources": [
            {
                "repo_id": "example/adapter",
                "repo_type": "model",
                "revision": REVISION_B,
                "files": ["artifacts/data/training.jsonl"],
            }
        ],
        "lineage_manifest": {
            "repo_id": "example/adapter",
            "repo_type": "model",
            "revision": REVISION_C,
            "file": "artifacts/training_lineage_manifest.json",
        },
    }
    config = validate_experiment_config(config)
    benchmark_root = tmp_path / "benchmark"
    adapter_root = tmp_path / "adapter-r"
    attestation_root = tmp_path / "attestation-m"
    benchmark_root.mkdir()
    (adapter_root / "artifacts" / "data").mkdir(parents=True)
    (adapter_root / "artifacts" / "data" / "training.jsonl").write_text("{}\n")
    (adapter_root / "adapter_config.json").write_text("{}")
    (attestation_root / "artifacts").mkdir(parents=True)
    (attestation_root / "artifacts" / "training_lineage_manifest.json").write_text("{}")

    def fake_snapshot(
        repo_id: str,
        repo_type: str,
        revision: str,
        allow_patterns: list[str],
        cache_dir: Path | None,
    ) -> tuple[Path, str]:
        del repo_id, repo_type, cache_dir
        if revision == REVISION_C:
            return attestation_root, revision
        if revision == REVISION_B:
            assert allow_patterns in (
                ["artifacts/data/training.jsonl"],
                ["adapter_config.json"],
            )
            return adapter_root, revision
        return benchmark_root, revision

    monkeypatch.setattr(prepare_module, "_snapshot", fake_snapshot)
    _, training, lineage, adapter_config, sources = prepare_module._download_inputs(config, None)

    assert training == [adapter_root / "artifacts" / "data" / "training.jsonl"]
    assert lineage == attestation_root / "artifacts" / "training_lineage_manifest.json"
    assert adapter_config == adapter_root / "adapter_config.json"
    assert sources[-2]["resolved_revision"] == REVISION_C
    assert sources[-1]["role"] == "evaluated_adapter_metadata"
    assert sources[-1]["resolved_revision"] == REVISION_B


def test_training_lineage_schema_is_executed_by_strict_prepare() -> None:
    manifest = {
        "schema_version": 1,
        "training_sources": [
            {
                "repo_id": "example/adapter",
                "repo_type": "model",
                "revision": REVISION_B,
                "files": [
                    {
                        "path": "artifacts/data/training.jsonl",
                        "sha256": "0" * 64,
                        "row_count": 1,
                    }
                ],
            }
        ],
        "coverage": {
            "paper_ids": True,
            "source_documents": True,
            "creator_groups": True,
            "image_bytes": True,
            "prompts": True,
        },
        "paper_ids": ["doi:10.5555/training"],
        "source_document_ids": ["source-1"],
        "creator_group_ids": ["creator-1"],
        "image_sha256s": ["1" * 64],
        "prompt_sha256s": ["2" * 64],
    }
    _validate_training_lineage_schema(manifest, Path(__file__).resolve().parents[1])

    manifest["image_sha256s"] = ["A" * 64]
    with pytest.raises(PublicationGateError, match="canonical schema"):
        _validate_training_lineage_schema(manifest, Path(__file__).resolve().parents[1])


def test_adapter_metadata_must_bind_the_configured_base_revision() -> None:
    config = _config()
    metadata = {
        "base_model_name_or_path": "example/base",
        "revision": REVISION_A,
    }
    _validate_adapter_base_metadata(metadata, config)

    metadata["revision"] = None
    with pytest.raises(PublicationGateError, match="immutable configured base revision"):
        _validate_adapter_base_metadata(metadata, config)

    metadata.update(
        {
            "base_model_name_or_path": "example/other-base",
            "revision": REVISION_A,
        }
    )
    with pytest.raises(PublicationGateError, match="base_model_name_or_path"):
        _validate_adapter_base_metadata(metadata, config)


def test_security_metadata_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"revision":"a","revision":"b"}', encoding="utf-8")
    with pytest.raises(PublicationGateError, match="duplicate JSON key 'revision'"):
        _load_strict_json_object(metadata, "metadata")

    metadata.write_text('{"revision":NaN}', encoding="utf-8")
    with pytest.raises(PublicationGateError, match="non-finite JSON constant"):
        _load_strict_json_object(metadata, "metadata")

    metadata.write_text('{"overflow":1e999}', encoding="utf-8")
    with pytest.raises(PublicationGateError, match="non-finite JSON number"):
        _load_strict_json_object(metadata, "metadata")


def test_strict_prepare_requires_lineage_even_when_config_flag_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    dataset = _dataset(tmp_path)
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text(
        json.dumps(
            {
                "base_model_name_or_path": config["models"]["tuned"]["base_model"]["id"],
                "revision": config["models"]["tuned"]["base_model"]["revision"],
            }
        ),
        encoding="utf-8",
    )
    sources = [
        {
            "repo_id": config["benchmark"]["repo_id"],
            "repo_type": "dataset",
            "configured_revision": REVISION_A,
            "resolved_revision": REVISION_A,
            "snapshot_root": str(dataset),
        },
        {
            "repo_id": config["models"]["tuned"]["adapter"]["id"],
            "repo_type": "model",
            "configured_revision": REVISION_B,
            "resolved_revision": REVISION_B,
            "files": ["adapter_config.json"],
            "role": "evaluated_adapter_metadata",
            "snapshot_root": str(tmp_path),
        },
    ]
    monkeypatch.setattr(
        prepare_module,
        "_download_inputs",
        lambda _config, _cache: (dataset, [], None, adapter_config, sources),
    )
    original_audit = prepare_module.audit_benchmark
    captured: dict[str, bool] = {}

    def capture_lineage_requirement(*args: object, **kwargs: object) -> dict:
        captured["required"] = kwargs["require_training_lineage"] is True
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(prepare_module, "audit_benchmark", capture_lineage_requirement)

    with pytest.raises(PublicationGateError, match="publication gate failed"):
        prepare_experiment(config, tmp_path)
    assert captured == {"required": True}


def test_strict_schema_failure_clears_eligibility_for_diagnostics() -> None:
    report = {
        "status": "pass",
        "publication_ready": True,
        "critical_findings": [],
        "eligible_sample_ids": ["sample-1"],
        "summary": {
            "critical_findings": 0,
            "critical_finding_count": 0,
            "critical_by_code": {},
            "eligible_samples": 1,
            "eligible_sample_count": 1,
        },
        "per_sample_findings": {"sample-1": {"eligible": True, "critical_findings": []}},
    }

    _apply_publication_schema_failure(report, "strict benchmark row 0 failed")

    assert report["status"] == "fail"
    assert report["publication_ready"] is False
    assert report["eligible_sample_ids"] == []
    assert report["summary"]["eligible_samples"] == 0
    assert report["per_sample_findings"]["sample-1"]["eligible"] is False


def test_prepare_bundle_is_relocatable(tmp_path: Path) -> None:
    config = _config()
    prepared = prepare_experiment(
        config,
        tmp_path,
        benchmark_dir=_dataset(tmp_path),
        exploratory=True,
    )
    source_run = Path(prepared["prepare_manifest"]).parent
    relocated = tmp_path / "relocated-run"
    shutil.copytree(source_run, relocated)
    manifest = json.loads((relocated / "prepare_manifest.json").read_text(encoding="utf-8"))

    verify_prepare_manifest(manifest, relocated)
    resolved = resolve_prepare_manifest(manifest, relocated)

    assert Path(resolved["dataset_root"]).is_relative_to(relocated)
    assert Path(resolved["benchmark_file"]).is_relative_to(relocated)


def test_strict_mode_rejects_protocol_overrides_before_loading_artifacts(
    tmp_path: Path,
) -> None:
    config = _config()
    with pytest.raises(RuntimeError, match="allowed only with --exploratory"):
        command_infer(
            argparse.Namespace(
                exploratory=False,
                arm="base",
                backend="mock",
                limit=None,
            ),
            config,
            tmp_path,
        )


def test_strict_mode_does_not_trust_edited_prepare_readiness(tmp_path: Path) -> None:
    config = _config()
    prepared = prepare_experiment(
        config,
        tmp_path,
        benchmark_dir=_dataset(tmp_path),
        exploratory=True,
    )
    manifest_path = Path(prepared["prepare_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["publication_ready"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PublicationGateError, match="status is inconsistent|modes must match"):
        command_infer(
            argparse.Namespace(
                exploratory=False,
                arm="base",
                backend="transformers",
                limit=None,
            ),
            config,
            tmp_path,
        )


def test_strict_prepare_requires_an_immutable_preregistered_plan(tmp_path: Path) -> None:
    config = _config()
    config["experiment"]["require_preregistered_plan"] = True
    dataset = _dataset(tmp_path)
    with pytest.raises(PublicationGateError, match="power plan"):
        prepare_experiment(config, tmp_path, benchmark_dir=dataset)

    command_plan(argparse.Namespace(), config, tmp_path)
    with pytest.raises(PublicationGateError, match="local benchmark overrides"):
        prepare_experiment(config, tmp_path, benchmark_dir=dataset)

    plan_path = tmp_path / config["experiment"]["output_dir"] / "design" / "power_plan.json"
    plan_bytes = plan_path.read_bytes()
    invalid_plan = json.loads(plan_bytes)
    invalid_plan["created_at"] = ""
    plan_path.write_text(json.dumps(invalid_plan), encoding="utf-8")
    with pytest.raises(PublicationGateError, match="ISO 8601 timestamp"):
        command_plan(argparse.Namespace(), config, tmp_path)
    plan_path.write_bytes(plan_bytes)

    changed = copy.deepcopy(config)
    changed["generation"]["max_new_tokens"] += 1
    with pytest.raises(RuntimeError, match="another protocol"):
        command_plan(argparse.Namespace(), changed, tmp_path)
