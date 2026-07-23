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
import scireason.vlm_ab.cli as cli_module
from scireason.vlm_ab.adapter_checkpoint import inspect_plain_fp32_lora_safetensors
from scireason.vlm_ab.cli import (
    _run_fingerprint,
    command_aggregate,
    command_blind,
    command_infer,
    command_plan,
)
from scireason.vlm_ab.config import (
    ExperimentConfigError,
    FP32_LORA_ADAPTER_KWARGS,
    NF4_SENSITIVITY_MODEL_KWARGS,
    load_experiment_config,
    validate_experiment_config,
)
from scireason.vlm_ab.prepare import (
    PublicationGateError,
    _apply_publication_schema_failure,
    _load_strict_json_object,
    _validate_adapter_base_metadata,
    _validate_publication_schemas,
    _validate_training_lineage_schema,
    code_provenance,
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


def _plain_adapter_metadata(config: dict) -> dict[str, object]:
    return {
        "base_model_name_or_path": config["models"]["tuned"]["base_model"]["id"],
        "revision": config["models"]["tuned"]["base_model"]["revision"],
        "peft_type": "LORA",
        "peft_version": "0.19.1",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "bias": "none",
        "modules_to_save": None,
        "trainable_token_indices": None,
        "init_lora_weights": True,
        "use_dora": False,
        "use_rslora": False,
        "use_qalora": False,
        "use_bdlora": None,
        "lora_bias": False,
        "ensure_weight_tying": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "loftq_config": {},
        "alora_invocation_tokens": None,
        "arrow_config": None,
        "corda_config": None,
        "eva_config": None,
        "layer_replication": None,
        "lora_ga_config": None,
        "megatron_config": None,
        "target_parameters": None,
        "target_modules": ["q_proj"],
    }


def _write_plain_lora_checkpoint(
    path: Path, *, dtype: str = "F32", extra_tensor: str | None = None
) -> None:
    tensors: dict[str, object] = {
        "base_model.model.q_proj.lora_A.weight": {
            "dtype": dtype,
            "shape": [1, 1],
            "data_offsets": [0, 4],
        },
        "base_model.model.q_proj.lora_B.weight": {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [4, 8],
        },
    }
    data = b"\0" * 8
    if extra_tensor is not None:
        tensors[extra_tensor] = {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [8, 12],
        }
        data += b"\0" * 4
    header = json.dumps(tensors, separators=(",", ":"), sort_keys=True).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + data)


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


def _write_assembly_manifest(dataset: Path) -> Path:
    audit_dir = dataset / "audit"
    audit_dir.mkdir()
    (audit_dir / "benchmark_audit.json").write_text("{}\n", encoding="utf-8")
    (audit_dir / "benchmark_audit.md").write_text("technical pass\n", encoding="utf-8")
    review_root = audit_dir / "human_review"
    queue_root = review_root / "queue"
    queue_root.mkdir(parents=True)
    benchmark = json.loads((dataset / "data" / "benchmark.jsonl").read_text(encoding="utf-8"))
    provenance = json.loads((dataset / "provenance.jsonl").read_text(encoding="utf-8"))
    config_fingerprint = "6" * 64
    source_hashes = {
        "prepare_manifest_sha256": "1" * 64,
        "benchmark_file_sha256": "2" * 64,
        "provenance_file_sha256": "3" * 64,
        "audit_json_sha256": "4" * 64,
    }
    source_row_sha256 = hashlib.sha256(
        json.dumps(
            benchmark,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    task_id = "task_" + hashlib.sha256(
        f"{'1' * 64}\x000\x00{source_row_sha256}".encode("ascii")
    ).hexdigest()
    task = {
        "artifact_version": 3,
        "task_id": task_id,
        "prepare_manifest_sha256": source_hashes["prepare_manifest_sha256"],
        "source_benchmark_sha256": source_hashes["benchmark_file_sha256"],
        "source_provenance_sha256": source_hashes["provenance_file_sha256"],
        "source_audit_sha256": source_hashes["audit_json_sha256"],
        "source_row_index": 0,
        "source_row_sha256": source_row_sha256,
        "original_row": benchmark,
        "legacy_provenance_rows": [],
        "audited_image_hashes": [],
        "critical_codes": [],
        "warning_codes": [],
        "training_overlap_paper_ids": [],
    }

    def json_bytes(value: object, *, newline: bool = False) -> bytes:
        suffix = "\n" if newline else ""
        return (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + suffix
        ).encode("utf-8")

    tasks_bytes = json_bytes(task, newline=True)
    tasks_sha256 = hashlib.sha256(tasks_bytes).hexdigest()
    schema_hashes = {"schema": "8" * 64}
    policy = {"exact_primary_papers": 1, "primary_strata": ["multimodal_hard"]}
    fingerprint_payload = {
        "artifact_version": 3,
        "config_fingerprint": config_fingerprint,
        "source_hashes": source_hashes,
        "schema_hashes": schema_hashes,
        "policy": policy,
        "task_count": 1,
        "tasks_sha256": tasks_sha256,
    }
    queue_fingerprint = hashlib.sha256(json_bytes(fingerprint_payload)).hexdigest()
    decision_binding = {
        key: task[key]
        for key in (
            "artifact_version",
            "task_id",
            "prepare_manifest_sha256",
            "source_benchmark_sha256",
            "source_provenance_sha256",
            "source_audit_sha256",
            "source_row_index",
            "source_row_sha256",
        )
    }
    decision_binding["queue_fingerprint"] = queue_fingerprint
    template = {
        **decision_binding,
        "status": "pending",
        "disposition": None,
        "exclusion_reason": None,
        "benchmark_row": None,
        "provenance_row": None,
        "reviewed_by": [],
        "independent_attestation": False,
        "notes": "",
    }
    decision = {
        **decision_binding,
        "status": "complete",
        "disposition": "retain",
        "exclusion_reason": None,
        "benchmark_row": benchmark,
        "provenance_row": provenance,
        "reviewed_by": ["curator-1", "curator-2"],
        "independent_attestation": True,
        "notes": "Independently reviewed.",
    }
    template_bytes = json_bytes(template, newline=True)
    decisions_bytes = json_bytes(decision, newline=True)
    queue_manifest = {
        **fingerprint_payload,
        "tasks_file": "tasks.jsonl",
        "decision_template_file": "decision_template.jsonl",
        "decision_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "queue_fingerprint": queue_fingerprint,
    }
    queue_manifest_bytes = json_bytes(queue_manifest, newline=True)
    (queue_root / "tasks.jsonl").write_bytes(tasks_bytes)
    (queue_root / "decision_template.jsonl").write_bytes(template_bytes)
    (queue_root / "queue_manifest.json").write_bytes(queue_manifest_bytes)
    (review_root / "completed_decisions.jsonl").write_bytes(decisions_bytes)
    release_files = [
        "data/benchmark.jsonl",
        "provenance.jsonl",
        "assets/images/page.png",
        "audit/benchmark_audit.json",
        "audit/benchmark_audit.md",
        "audit/human_review/queue/tasks.jsonl",
        "audit/human_review/queue/decision_template.jsonl",
        "audit/human_review/queue/queue_manifest.json",
        "audit/human_review/completed_decisions.jsonl",
    ]
    output_files = []
    for relative in release_files:
        payload = (dataset / relative).read_bytes()
        output_files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest = {
        "artifact_version": 3,
        "scope": "validated_benchmark_release_candidate",
        "publication_ready": False,
        "technical_audit_passed": True,
        "config_fingerprint": config_fingerprint,
        "queue_fingerprint": queue_fingerprint,
        "decisions_sha256": hashlib.sha256(decisions_bytes).hexdigest(),
        "reviewer_ids": ["curator-1", "curator-2"],
        "human_review_evidence": {
            "artifact_version": 3,
            "archive_root": "audit/human_review",
            "queue_manifest_path": "audit/human_review/queue/queue_manifest.json",
            "queue_manifest_sha256": hashlib.sha256(queue_manifest_bytes).hexdigest(),
            "tasks_path": "audit/human_review/queue/tasks.jsonl",
            "tasks_sha256": tasks_sha256,
            "decision_template_path": "audit/human_review/queue/decision_template.jsonl",
            "decision_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
            "decisions_path": "audit/human_review/completed_decisions.jsonl",
            "decisions_sha256": hashlib.sha256(decisions_bytes).hexdigest(),
            "independent_attestation_required": True,
        },
        "counts": {
            "source_tasks": 1,
            "retained_rows": 1,
            "excluded_rows": 0,
            "provenance_rows": 1,
            "copied_assets": 1,
            "unique_primary_papers": 1,
        },
        "output_files": output_files,
    }
    path = dataset / "assembly_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _review_export(public: dict, preference: str) -> dict:
    assignment_ids = [item["assignment_id"] for item in public["assignments"]]
    return {
        "artifact_version": public["artifact_version"],
        "experiment_id": public["experiment_id"],
        "study_fingerprint": public["study_fingerprint"],
        "rubric_version": public["rubric_version"],
        "rubric_sha256": public["rubric_sha256"],
        "reviewer_id": public["reviewer_id"],
        "package_nonce": public["package_nonce"],
        "independent_review_attestation": True,
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


def test_nf4_mode_is_automatic_only_and_cannot_enter_human_review(tmp_path: Path) -> None:
    config = _config()
    config["experiment"].update(
        {
            "precision_mode": "nf4-sensitivity",
            "require_clean_code": True,
            "require_preregistered_plan": True,
        }
    )
    config["power"].update(
        {"n_items": 150, "require_exact_n_items": True, "reviews_per_item": 2}
    )
    config["benchmark"].update(
        {
            "assembly_manifest_file": "assembly_manifest.json",
            "assembly_manifest_sha256": "d" * 64,
        }
    )
    for arm_name in ("base", "tuned"):
        config["models"][arm_name]["model_kwargs"] = copy.deepcopy(
            NF4_SENSITIVITY_MODEL_KWARGS
        )
    config["models"]["tuned"]["adapter_kwargs"] = copy.deepcopy(
        FP32_LORA_ADAPTER_KWARGS
    )
    config = validate_experiment_config(config)
    assert cli_module._inference_result_scope(config, False) == "automatic_sensitivity_only"
    assert cli_module._inference_result_scope(config, True) == "exploratory_not_for_publication"

    args = argparse.Namespace(exploratory=False)
    for command in (command_blind, command_aggregate, cli_module.command_run):
        with pytest.raises(RuntimeError, match="automatic diagnostics only"):
            command(args, config, tmp_path)


def test_unlabeled_config_cannot_claim_publication_or_enter_strict_review(tmp_path: Path) -> None:
    config = _config()
    assert cli_module._inference_result_scope(config, False) == "unspecified"

    with pytest.raises(RuntimeError, match="explicit fp16-primary"):
        command_blind(
            argparse.Namespace(
                exploratory=False,
                reviewers=None,
                reviews_per_item=None,
            ),
            config,
            tmp_path,
        )


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
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "## Human Review Diagnostics" in report
    assert "Krippendorff alpha" in report


@pytest.mark.parametrize(
    ("require_exact_n_items", "expected_ready"),
    [(False, True), (True, False)],
)
def test_aggregate_exact_n_gate_rejects_extra_assigned_papers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_exact_n_items: bool,
    expected_ready: bool,
) -> None:
    config = _config()
    config["power"].update(
        {
            "n_items": 2,
            "require_exact_n_items": require_exact_n_items,
            "target_effect": 4.0,
        }
    )
    config = validate_experiment_config(config)
    primary = {
        "n_evaluable_reviews": 6,
        "n_reviews": 6,
        "n_papers": 3,
        "n_assigned_papers": 3,
        "estimate": 0.6,
        "bootstrap_ci": [0.55, 0.65],
        "missingness_worst_best_case_bounds": [0.55, 0.65],
        "missingness_worst_case_bootstrap_ci": [0.55, 0.65],
        "missingness_worst_case_p_value": 0.01,
        "p_value": 0.01,
    }
    human = {
        "primary": primary,
        "n_assigned_papers": 3,
        "n_reviewers": len(config["review"]["reviewer_ids"]),
    }

    monkeypatch.setattr(
        cli_module,
        "_load_prepare",
        lambda *_args: {"frozen_benchmark": "frozen.jsonl", "publication_ready": True},
    )
    monkeypatch.setattr(cli_module, "load_jsonl", lambda _path: [{"sample_id": "sample-1"}])
    monkeypatch.setattr(
        cli_module,
        "_validate_prediction_manifests",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(cli_module, "_validate_paired_outputs", lambda *_args: None)
    monkeypatch.setattr(
        cli_module,
        "_verify_blind_manifest",
        lambda *_args: ({"study_fingerprint": "study"}, tmp_path / "owner.json", "0" * 64),
    )
    monkeypatch.setattr(cli_module, "_validate_realized_review_design", lambda *_args: None)
    monkeypatch.setattr(cli_module, "_review_paths", lambda _args: [])
    monkeypatch.setattr(cli_module, "deblind_reviews", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli_module, "_check_review_completeness", lambda *_args: (6, 0))
    monkeypatch.setattr(cli_module, "_join_review_metadata", lambda *_args: [])
    monkeypatch.setattr(cli_module, "summarize_reviews", lambda *_args, **_kwargs: human)
    monkeypatch.setattr(cli_module, "write_paper_scores", lambda *_args: None)
    monkeypatch.setattr(cli_module, "write_effect_svg", lambda *_args: None)
    monkeypatch.setattr(cli_module, "build_markdown_report", lambda *_args: "")
    monkeypatch.setattr(cli_module, "_require_human_review_mode", lambda *_args, **_kwargs: None)

    result = command_aggregate(
        argparse.Namespace(
            exploratory=False,
            review_file=[],
            reviews_dir=None,
            allow_incomplete=False,
        ),
        config,
        tmp_path,
    )
    stored = json.loads(Path(result["results"]).read_text(encoding="utf-8"))

    assert result["publication_artifacts_ready"] is expected_ready
    assert stored["power_plan"]["require_exact_n_items"] is require_exact_n_items


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


def test_prepare_preserves_and_verifies_reviewed_assembly_manifest(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    assembly = _write_assembly_manifest(dataset)
    config = _config()
    config["power"]["require_exact_n_items"] = True
    config["benchmark"].update(
        {
            "assembly_manifest_file": assembly.relative_to(dataset).as_posix(),
            "assembly_manifest_sha256": hashlib.sha256(assembly.read_bytes()).hexdigest(),
        }
    )
    config = validate_experiment_config(config)

    prepared = prepare_experiment(
        config,
        tmp_path,
        benchmark_dir=dataset,
        exploratory=True,
    )
    manifest = json.loads(Path(prepared["prepare_manifest"]).read_text(encoding="utf-8"))

    assert Path(prepared["assembly_manifest"]).read_bytes() == assembly.read_bytes()
    assert prepared["assembly_manifest_sha256"] == config["benchmark"][
        "assembly_manifest_sha256"
    ]
    verify_prepare_manifest(manifest)
    benchmark_path = Path(prepared["benchmark_file"])
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_path.write_bytes(benchmark_bytes + b"\n")
    resealed = copy.deepcopy(manifest)
    resealed["benchmark_file_sha256"] = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    with pytest.raises(PublicationGateError, match="assembly output file differs"):
        prepare_module.verify_prepared_audit(config, resealed)
    benchmark_path.write_bytes(benchmark_bytes)
    missing_binding = copy.deepcopy(manifest)
    missing_binding["artifact_paths"].pop("assembly_manifest")
    missing_binding.pop("assembly_manifest")
    missing_binding.pop("assembly_manifest_sha256")
    with pytest.raises(PublicationGateError, match="does not preserve.*assembly"):
        prepare_module.verify_prepared_audit(config, missing_binding)
    Path(prepared["assembly_manifest"]).write_bytes(b"tampered\n")
    with pytest.raises(PublicationGateError, match="artifact hash changed"):
        verify_prepare_manifest(manifest)


@pytest.mark.parametrize("tamper", ["manifest", "release-file", "resealed-review"])
def test_prepare_rejects_tampered_reviewed_assembly(
    tmp_path: Path, tamper: str
) -> None:
    dataset = _dataset(tmp_path)
    assembly = _write_assembly_manifest(dataset)
    config = _config()
    config["power"]["require_exact_n_items"] = True
    config["benchmark"].update(
        {
            "assembly_manifest_file": assembly.relative_to(dataset).as_posix(),
            "assembly_manifest_sha256": hashlib.sha256(assembly.read_bytes()).hexdigest(),
        }
    )
    config = validate_experiment_config(config)
    if tamper == "manifest":
        assembly.write_bytes(assembly.read_bytes() + b"\n")
        expected = "assembly manifest SHA256 differs"
    elif tamper == "release-file":
        (dataset / "assets" / "images" / "page.png").write_bytes(b"tampered release image")
        expected = "assembly output file differs"
    else:
        decisions_path = dataset / "audit" / "human_review" / "completed_decisions.jsonl"
        decision = json.loads(decisions_path.read_text(encoding="utf-8"))
        decision["status"] = "pending"
        decisions_path.write_text(
            json.dumps(decision, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resealed = json.loads(assembly.read_text(encoding="utf-8"))
        digest = hashlib.sha256(decisions_path.read_bytes()).hexdigest()
        resealed["decisions_sha256"] = digest
        resealed["human_review_evidence"]["decisions_sha256"] = digest
        output_record = next(
            record
            for record in resealed["output_files"]
            if record["path"] == "audit/human_review/completed_decisions.jsonl"
        )
        output_record.update(sha256=digest, size_bytes=decisions_path.stat().st_size)
        assembly.write_text(json.dumps(resealed, sort_keys=True) + "\n", encoding="utf-8")
        config["benchmark"]["assembly_manifest_sha256"] = hashlib.sha256(
            assembly.read_bytes()
        ).hexdigest()
        config = validate_experiment_config(config)
        expected = "curator decision bindings are invalid"

    with pytest.raises(PublicationGateError, match=expected):
        prepare_experiment(
            config,
            tmp_path,
            benchmark_dir=dataset,
            exploratory=True,
        )


def test_required_machine_enrichment_archive_cannot_be_omitted(tmp_path: Path) -> None:
    with pytest.raises(PublicationGateError, match="no archived machine enrichment evidence"):
        prepare_module._validate_archived_machine_enrichment(
            {"counts": {"retained_rows": 150}},
            {},
            tmp_path,
            required=True,
            queue_tasks=[],
            queue_policy={},
        )


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


def _nf4_kwargs() -> dict:
    return {
        "device_map": "balanced",
        "quantization_config": {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "torch.float16",
            "bnb_4bit_use_double_quant": True,
        },
    }


def test_config_accepts_strict_nf4_for_both_arms() -> None:
    config = _config()
    config["experiment"].update(
        {
            "precision_mode": "nf4-sensitivity",
            "require_clean_code": True,
            "require_preregistered_plan": True,
        }
    )
    config["power"].update(
        {"n_items": 150, "require_exact_n_items": True, "reviews_per_item": 2}
    )
    config["benchmark"].update(
        {
            "assembly_manifest_file": "assembly_manifest.json",
            "assembly_manifest_sha256": "d" * 64,
        }
    )
    for arm_name in ("base", "tuned"):
        config["models"][arm_name]["model_kwargs"] = copy.deepcopy(
            NF4_SENSITIVITY_MODEL_KWARGS
        )
    config["models"]["tuned"]["adapter_kwargs"] = copy.deepcopy(
        FP32_LORA_ADAPTER_KWARGS
    )

    validated = validate_experiment_config(config)

    assert validated["models"]["base"]["model_kwargs"] == NF4_SENSITIVITY_MODEL_KWARGS
    assert validated["models"]["tuned"]["model_kwargs"] == NF4_SENSITIVITY_MODEL_KWARGS


def test_strict_exact_config_requires_reviewed_assembly_binding() -> None:
    config = _config()
    config["experiment"].update(
        {"require_clean_code": True, "require_preregistered_plan": True}
    )
    config["power"]["require_exact_n_items"] = True

    with pytest.raises(ExperimentConfigError, match="reviewed benchmark assembly manifest"):
        validate_experiment_config(config)


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"load_in_4bit": True},
        {"quantization_config": []},
        {"quantization_config": {}},
        {
            "quantization_config": {
                **_nf4_kwargs()["quantization_config"],
                "bnb_4bit_compute_dtype": "bfloat16",
            }
        },
        {
            "quantization_config": {
                **_nf4_kwargs()["quantization_config"],
                "bnb_4bit_use_double_quant": 1,
            }
        },
        {
            "quantization_config": {
                **_nf4_kwargs()["quantization_config"],
                "unexpected": True,
            }
        },
    ],
)
@pytest.mark.parametrize("arm_name", ["base", "tuned"])
def test_config_rejects_malformed_nf4_before_runtime(model_kwargs: dict, arm_name: str) -> None:
    config = _config()
    config["models"][arm_name]["model_kwargs"] = model_kwargs

    with pytest.raises(ExperimentConfigError, match="quantization|quantization_config"):
        validate_experiment_config(config)


@pytest.mark.parametrize("n_items", [True, 0, -1, 1.5, "240"])
def test_config_requires_positive_integer_power_n_items(n_items: object) -> None:
    config = _config()
    config["power"]["n_items"] = n_items

    with pytest.raises(ExperimentConfigError, match="power.n_items"):
        validate_experiment_config(config)


def test_config_validates_opt_in_exact_primary_paper_guard() -> None:
    config = _config()
    assert "require_exact_n_items" not in config["power"]
    assert config["power"].get("require_exact_n_items", False) is False

    config["power"]["require_exact_n_items"] = True
    assert validate_experiment_config(config)["power"]["require_exact_n_items"] is True

    config["power"]["require_exact_n_items"] = 1
    with pytest.raises(ExperimentConfigError, match="require_exact_n_items"):
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
    _write_plain_lora_checkpoint(adapter_root / "adapter_model.safetensors")
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
                ["adapter_config.json", "adapter_model.safetensors"],
            )
            return adapter_root, revision
        return benchmark_root, revision

    monkeypatch.setattr(prepare_module, "_snapshot", fake_snapshot)
    _, training, lineage, adapter_config, adapter_checkpoint, sources = (
        prepare_module._download_inputs(config, None)
    )

    assert training == [adapter_root / "artifacts" / "data" / "training.jsonl"]
    assert lineage == attestation_root / "artifacts" / "training_lineage_manifest.json"
    assert adapter_config == adapter_root / "adapter_config.json"
    assert adapter_checkpoint == adapter_root / "adapter_model.safetensors"
    assert sources[-2]["resolved_revision"] == REVISION_C
    assert sources[-1]["role"] == "evaluated_adapter_checkpoint"
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
    metadata = _plain_adapter_metadata(config)
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

    metadata.update(
        {
            "base_model_name_or_path": "example/base",
            "modules_to_save": ["lm_head"],
        }
    )
    with pytest.raises(PublicationGateError, match="modules_to_save"):
        _validate_adapter_base_metadata(metadata, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("init_lora_weights", "pissa"),
        ("use_dora", True),
        ("use_rslora", True),
        ("lora_bias", True),
        ("target_parameters", ["model.weight"]),
    ],
)
def test_adapter_metadata_rejects_non_plain_lora_variants(field: str, value: object) -> None:
    config = _config()
    metadata = _plain_adapter_metadata(config)
    metadata[field] = value

    with pytest.raises(PublicationGateError, match=field):
        _validate_adapter_base_metadata(metadata, config)


def test_adapter_checkpoint_attests_only_paired_native_fp32_lora(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter_model.safetensors"
    _write_plain_lora_checkpoint(checkpoint)

    attestation = inspect_plain_fp32_lora_safetensors(
        checkpoint,
        repo_id="example/adapter",
        revision=REVISION_B,
        error_type=PublicationGateError,
    )

    assert attestation["tensor_dtype"] == "F32"
    assert attestation["tensor_count"] == 2
    assert attestation["lora_module_count"] == 1
    assert len(attestation["sha256"]) == 64


@pytest.mark.parametrize(
    ("dtype", "extra_tensor", "message"),
    [
        ("F16", None, "native F32"),
        ("F32", "base_model.model.q_proj.weight", "non-plain-LoRA"),
    ],
)
def test_adapter_checkpoint_rejects_castable_or_base_weights(
    tmp_path: Path,
    dtype: str,
    extra_tensor: str | None,
    message: str,
) -> None:
    checkpoint = tmp_path / "adapter_model.safetensors"
    _write_plain_lora_checkpoint(checkpoint, dtype=dtype, extra_tensor=extra_tensor)

    with pytest.raises(PublicationGateError, match=message):
        inspect_plain_fp32_lora_safetensors(
            checkpoint,
            repo_id="example/adapter",
            revision=REVISION_B,
            error_type=PublicationGateError,
        )


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


@pytest.mark.parametrize(
    ("suffix", "text", "message"),
    [
        (".json", '{"schema_version":1,"schema_version":1}', "duplicate JSON key"),
        (".json", '{"schema_version":NaN}', "non-finite"),
        (".json", '{"schema_version":1e999}', "non-finite"),
        (".yaml", "schema_version: 1\nschema_version: 1\n", "duplicate key"),
        (".yaml", "schema_version: .nan\n", "non-finite"),
    ],
)
def test_experiment_config_rejects_ambiguous_or_nonfinite_documents(
    tmp_path: Path, suffix: str, text: str, message: str
) -> None:
    path = tmp_path / f"config{suffix}"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match=message):
        load_experiment_config(path)


def test_strict_prepare_requires_lineage_even_when_config_flag_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    dataset = _dataset(tmp_path)
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text(json.dumps(_plain_adapter_metadata(config)), encoding="utf-8")
    adapter_checkpoint = tmp_path / "adapter_model.safetensors"
    _write_plain_lora_checkpoint(adapter_checkpoint)
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
            "files": ["adapter_config.json", "adapter_model.safetensors"],
            "role": "evaluated_adapter_checkpoint",
            "snapshot_root": str(tmp_path),
        },
    ]
    monkeypatch.setattr(
        prepare_module,
        "_download_inputs",
        lambda _config, _cache: (
            dataset,
            [],
            None,
            adapter_config,
            adapter_checkpoint,
            sources,
        ),
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


@pytest.mark.parametrize(
    ("n_items", "actual_primary_papers"),
    [(2, 1), (1, 2)],
)
def test_strict_prepare_blocks_an_exact_primary_paper_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_items: int,
    actual_primary_papers: int,
) -> None:
    config = _config()
    config["power"].update({"n_items": n_items, "require_exact_n_items": True})
    config = validate_experiment_config(config)
    dataset = _dataset(tmp_path)
    if actual_primary_papers == 2:
        benchmark_path = dataset / "data" / "benchmark.jsonl"
        first_row = json.loads(benchmark_path.read_text(encoding="utf-8"))
        second_row = copy.deepcopy(first_row)
        second_row["sample_id"] = "sample-2"
        second_row["paper_id"] = "doi:10.5555/smoke.two"
        second_row["model_task_prompt"] = "Extract the trend shown in Figure 2."
        second_row["messages"][1]["content"][0]["text"] = second_row["model_task_prompt"]
        benchmark_path.write_text(
            "\n".join(json.dumps(row) for row in (first_row, second_row)) + "\n",
            encoding="utf-8",
        )
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text(json.dumps(_plain_adapter_metadata(config)), encoding="utf-8")
    adapter_checkpoint = tmp_path / "adapter_model.safetensors"
    _write_plain_lora_checkpoint(adapter_checkpoint)
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
            "files": ["adapter_config.json", "adapter_model.safetensors"],
            "role": "evaluated_adapter_checkpoint",
            "snapshot_root": str(tmp_path),
        },
    ]
    monkeypatch.setattr(
        prepare_module,
        "_download_inputs",
        lambda _config, _cache: (
            dataset,
            [],
            None,
            adapter_config,
            adapter_checkpoint,
            sources,
        ),
    )

    with pytest.raises(PublicationGateError, match="publication gate failed"):
        prepare_experiment(config, tmp_path)

    report = json.loads(
        (
            tmp_path / config["experiment"]["output_dir"] / "audit" / "benchmark_audit.json"
        ).read_text(encoding="utf-8")
    )
    mismatch = next(
        finding
        for finding in report["critical_findings"]
        if finding["code"] == "primary_paper_count_mismatch"
    )
    assert mismatch["details"] == {"actual": actual_primary_papers, "required": n_items}


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


def test_code_provenance_excludes_generated_cache_and_sensitive_files(tmp_path: Path) -> None:
    included = [
        tmp_path / "src" / "scireason" / "vlm_ab" / "module.py",
        tmp_path / "experiments" / "vlm_ab_evaluation" / "run_pipeline.py",
        tmp_path / "experiments" / "vlm_ab_evaluation" / "kaggle" / "build_payload.py",
    ]
    excluded = [
        tmp_path / "experiments" / "vlm_ab_evaluation" / "kaggle" / "build" / "x.py",
        tmp_path / "experiments" / "vlm_ab_evaluation" / "kaggle" / "staging" / "x.py",
        tmp_path / "experiments" / "vlm_ab_evaluation" / "kaggle" / "downloads" / "x.py",
        tmp_path / "src" / "scireason" / "vlm_ab" / "__pycache__" / "module.pyc",
        tmp_path / "experiments" / "vlm_ab_evaluation" / "caches" / "runtime.bin",
        tmp_path / "experiments" / "vlm_ab_evaluation" / "kaggle.json",
        tmp_path / "experiments" / "vlm_ab_evaluation" / ".env.local",
        tmp_path / "src" / "scireason" / "vlm_ab" / "private-key.pem",
        tmp_path / "experiments" / "vlm_ab_evaluation" / ".envrc",
        tmp_path / "experiments" / "vlm_ab_evaluation" / ".env.d" / "value",
        tmp_path / "experiments" / "vlm_ab_evaluation" / ".git-credentials",
        tmp_path / "experiments" / "vlm_ab_evaluation" / "client_secret.json",
        tmp_path / "src" / "scireason" / "vlm_ab" / "id_ecdsa",
    ]
    for path in included + excluded:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")

    first = code_provenance(tmp_path)
    paths = {item["path"] for item in first["files"]}
    for path in included:
        assert path.relative_to(tmp_path).as_posix() in paths
    for path in excluded:
        assert path.relative_to(tmp_path).as_posix() not in paths

    for path in excluded:
        path.write_text("changed generated content", encoding="utf-8")
    assert code_provenance(tmp_path)["source_fingerprint"] == first["source_fingerprint"]


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


def test_prediction_manifest_validation_uses_prepared_adapter_attestation(
    tmp_path: Path,
) -> None:
    config = _config()
    attestation_path = tmp_path / "adapter-checkpoint.json"
    attestation = {"artifact_version": 1, "sha256": "a" * 64}
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    prepared = {"adapter_checkpoint_attestation": str(attestation_path)}
    predictions = tmp_path / config["experiment"]["output_dir"] / "predictions"
    predictions.mkdir(parents=True)
    runtime = {"python": "test"}
    expected_rows = len(config["conditions"])
    for arm in ("base", "tuned"):
        output = predictions / f"{arm}.jsonl"
        output.write_text("{}\n", encoding="utf-8")
        manifest = {
            "status": "complete",
            "completed_rows": expected_rows,
            "expected_rows": expected_rows,
            "arm": arm,
            "experiment_fingerprint": "b" * 64,
            "result_scope": "exploratory_not_for_publication",
            "conditions": config["conditions"],
            "seed": config["experiment"]["seed"],
            "backend": "mock",
            "limit": None,
            "configuration": {
                "arm": cli_module._arm_config(config, arm, prepared),
                "processor": cli_module._processor_config(config),
                "generation": config["generation"],
            },
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "protocol_fingerprint": "c" * 64,
            "runtime": runtime,
        }
        Path(f"{output}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    validated = cli_module._validate_prediction_manifests(
        config,
        tmp_path,
        expected_samples=1,
        exploratory=True,
        run_fingerprint="b" * 64,
        prepared=prepared,
    )

    assert validated["tuned"]["configuration"]["arm"][
        "adapter_checkpoint_attestation"
    ] == attestation


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


def test_confirmatory_plan_requires_exactly_two_reviewers(tmp_path: Path) -> None:
    config = _config()
    config["review"]["reviewer_ids"].append("reviewer-3")
    config = validate_experiment_config(config)

    with pytest.raises(PublicationGateError, match="exactly two reviewer IDs"):
        command_plan(argparse.Namespace(), config, tmp_path)


def test_command_plan_requires_clean_git_provenance_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["experiment"]["require_clean_code"] = True
    config = validate_experiment_config(config)
    plan_path = tmp_path / config["experiment"]["output_dir"] / "design" / "power_plan.json"

    monkeypatch.setattr(
        cli_module,
        "code_provenance",
        lambda _root: {
            "source_fingerprint": "f" * 64,
            "git_dirty": True,
            "git_head": "a" * 40,
        },
    )
    with pytest.raises(PublicationGateError, match="clean evaluation source tree"):
        command_plan(argparse.Namespace(), config, tmp_path)
    assert not plan_path.exists()

    monkeypatch.setattr(
        cli_module,
        "code_provenance",
        lambda _root: {
            "source_fingerprint": "f" * 64,
            "git_dirty": False,
            "git_head": "a" * 40,
        },
    )
    plan = command_plan(argparse.Namespace(), config, tmp_path)

    assert plan_path.is_file()
    assert plan["code_fingerprint"] == "f" * 64


def test_command_plan_records_exact_primary_paper_guard(tmp_path: Path) -> None:
    config = _config()
    config["power"]["require_exact_n_items"] = True
    config = validate_experiment_config(config)

    first = command_plan(argparse.Namespace(), config, tmp_path)
    second = command_plan(argparse.Namespace(), config, tmp_path)
    plan_path = tmp_path / config["experiment"]["output_dir"] / "design" / "power_plan.json"

    assert first["require_exact_n_items"] is True
    assert second == first
    assert json.loads(plan_path.read_text(encoding="utf-8"))["require_exact_n_items"] is True
