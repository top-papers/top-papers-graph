#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exact local scoring for the synthetic exploratory VLM A/B harness only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    RESULT_SCOPE,
    SCOPE_LABELS,
    SyntheticHarnessError,
    canonical_json,
    inventory,
    read_json_object,
    read_jsonl,
    repo_root,
    require_scope_labels,
    scope_metadata,
    scope_text,
    sha256_file,
    verify_generated_run,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


_ARMS = ("base", "tuned")
_CONDITIONS = ("original", "text_only", "shuffled_images")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FENCED_JSON = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_raw_response(raw_response: Any) -> Any | None:
    """Independently parse strict JSON from the model's raw response only."""

    if not isinstance(raw_response, str) or not raw_response.strip():
        return None
    candidates = [raw_response.strip()]
    candidates.extend(match.group(1).strip() for match in _FENCED_JSON.finditer(raw_response))
    decoder = json.JSONDecoder(
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(
                candidate,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError):
            pass
        for index, character in enumerate(candidate):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate, index)
            except (json.JSONDecodeError, ValueError):
                continue
            return value
    return None


def _schema_valid(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    if not isinstance(response.get("answer"), str) or not response["answer"].strip():
        return False
    evidence = response.get("evidence_used")
    if not isinstance(evidence, list):
        return False
    for item in evidence:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(field), str) for field in ("kind", "locator", "description")
        ):
            return False
    for field in ("visual_facts", "temporal_facts", "missing_evidence"):
        value = response.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return False
    return response.get("uncertainty") in {"low", "medium", "high"}


def _load_prepared(root: Path, generated: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from scireason.vlm_ab.audit import load_jsonl as load_pipeline_jsonl
    from scireason.vlm_ab.prepare import (
        resolve_prepare_manifest,
        verify_prepare_manifest,
        verify_prepared_audit,
    )

    run = generated["run"]
    receipt = read_json_object(run / "synthetic_prepare_receipt.json", "synthetic prepare receipt")
    require_scope_labels(receipt, "synthetic prepare receipt")
    if receipt.get("result_scope") != RESULT_SCOPE or receipt.get("exploratory") is not True:
        raise SyntheticHarnessError("synthetic prepare receipt has the wrong scope")
    if receipt.get("generator_manifest_sha256") != sha256_file(run / "generator_manifest.json"):
        raise SyntheticHarnessError("synthetic prepare receipt generator binding changed")
    if receipt.get("synthetic_config_sha256") != sha256_file(run / "synthetic_config.yaml"):
        raise SyntheticHarnessError("synthetic prepare receipt config binding changed")
    manifest_path = run / "prepare_manifest.json"
    if receipt.get("prepare_manifest_sha256") != sha256_file(manifest_path):
        raise SyntheticHarnessError("synthetic prepare receipt manifest binding changed")
    manifest = read_json_object(manifest_path, "prepare manifest")
    verify_prepare_manifest(manifest, run)
    verify_prepared_audit(generated["config"], manifest, run)
    if manifest.get("exploratory") is not True or manifest.get("result_scope") != RESULT_SCOPE:
        raise SyntheticHarnessError("prepare manifest is not an exploratory synthetic result")
    if manifest.get("publication_ready") is not False:
        raise SyntheticHarnessError("synthetic prepare manifest must not be publication-ready")
    expected_count = generated["manifest"]["sample_count"]
    if manifest.get("frozen_rows") != expected_count or manifest.get("excluded_rows") != 0:
        raise SyntheticHarnessError("prepare manifest does not contain the complete synthetic source")
    prepared = resolve_prepare_manifest(manifest, run)
    frozen = load_pipeline_jsonl(prepared["frozen_benchmark"])
    if len(frozen) != expected_count:
        raise SyntheticHarnessError("frozen benchmark row count is inconsistent")
    return prepared, frozen


def _prediction_inputs(
    root: Path,
    generated: dict[str, Any],
    prepared: dict[str, Any],
    frozen: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from scireason.vlm_ab.inference import build_condition_rows

    config = generated["config"]
    rows = build_condition_rows(
        frozen,
        prepared["dataset_root"],
        conditions=config["conditions"],
        seed=config["experiment"]["seed"],
    )
    expected = {(row["sample_id"], row["condition"]): row for row in rows}
    if len(expected) != len(rows):
        raise SyntheticHarnessError("prepared synthetic condition keys are not unique")
    return expected


def _verify_sidecar(
    run: Path,
    arm: str,
    config: dict[str, Any],
    expected_rows: int,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    prediction = run / "predictions" / f"{arm}.jsonl"
    sidecar_path = Path(f"{prediction}.manifest.json")
    if not prediction.is_file() or prediction.is_symlink() or not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise SyntheticHarnessError(f"{arm} prediction and regular sidecar are required")
    sidecar = read_json_object(sidecar_path, f"{arm} prediction sidecar")
    backend = sidecar.get("backend")
    if backend not in {"transformers", "mock"}:
        raise SyntheticHarnessError(f"{arm} sidecar has an unsupported backend")
    if (
        sidecar.get("status") != "complete"
        or sidecar.get("arm") != arm
        or sidecar.get("result_scope") != RESULT_SCOPE
        or sidecar.get("conditions") != config["conditions"]
        or sidecar.get("seed") != config["experiment"]["seed"]
        or sidecar.get("expected_rows") != expected_rows
        or sidecar.get("completed_rows") != expected_rows
    ):
        raise SyntheticHarnessError(f"{arm} prediction sidecar is incomplete or mismatched")
    successful = sidecar.get("successful_rows")
    errors = sidecar.get("error_rows")
    if (
        isinstance(successful, bool)
        or isinstance(errors, bool)
        or not isinstance(successful, int)
        or not isinstance(errors, int)
        or successful < 0
        or errors < 0
        or successful + errors != expected_rows
    ):
        raise SyntheticHarnessError(f"{arm} prediction sidecar has invalid row counts")
    output_hash = sidecar.get("output_sha256")
    if not isinstance(output_hash, str) or not _SHA256_RE.fullmatch(output_hash):
        raise SyntheticHarnessError(f"{arm} prediction sidecar has no valid output SHA256")
    if sha256_file(prediction) != output_hash:
        raise SyntheticHarnessError(f"{arm} prediction bytes do not match its sidecar")
    configuration = sidecar.get("configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("arm") != config["models"][arm]
        or configuration.get("processor") != config["processor"]
        or configuration.get("generation") != config["generation"]
    ):
        raise SyntheticHarnessError(f"{arm} prediction sidecar differs from synthetic config")
    rows = read_jsonl(prediction, f"{arm} predictions")
    if len(rows) != expected_rows:
        raise SyntheticHarnessError(f"{arm} prediction JSONL row count is invalid")
    return prediction, sidecar, rows


def _verify_row_controls(
    row: dict[str, Any],
    expected: dict[str, Any],
    keys_by_sample: dict[str, dict[str, Any]],
    keys_by_paper: dict[str, dict[str, Any]],
    arm: str,
    backend: str,
) -> None:
    if row.get("arm") != arm or row.get("backend") != backend:
        raise SyntheticHarnessError(f"{arm} prediction row has another arm or backend")
    sample_id = row.get("sample_id")
    condition = row.get("condition")
    if not isinstance(sample_id, str) or not isinstance(condition, str):
        raise SyntheticHarnessError("prediction row has an invalid sample/condition key")
    if row.get("paper_id") != expected["paper_id"]:
        raise SyntheticHarnessError("prediction row paper ID differs from frozen benchmark")
    if row.get("condition_input_fingerprint") != expected["condition_input_fingerprint"]:
        raise SyntheticHarnessError("prediction row condition input fingerprint differs")
    if row.get("input_image_hashes") != expected["input_image_hashes"]:
        raise SyntheticHarnessError("prediction row input image hashes differ")
    if row.get("shuffle_source_paper_id") != expected["shuffle_source_paper_id"]:
        raise SyntheticHarnessError("prediction row shuffled donor differs")
    hashes = row.get("input_image_hashes")
    if not isinstance(hashes, list) or row.get("input_image_count") != len(hashes):
        raise SyntheticHarnessError("prediction row has inconsistent image count")
    key = keys_by_sample.get(sample_id)
    if key is None:
        raise SyntheticHarnessError("prediction row references an unknown answer-key sample")
    if condition == "original":
        if hashes != [key["original_image_sha256"]] or row.get("shuffle_source_paper_id") != key["paper_id"]:
            raise SyntheticHarnessError("original control does not use the answer-key image")
    elif condition == "text_only":
        if hashes or row.get("shuffle_source_paper_id") is not None:
            raise SyntheticHarnessError("text_only control must contain no image or donor")
    elif condition == "shuffled_images":
        donor = row.get("shuffle_source_paper_id")
        donor_key = keys_by_paper.get(donor) if isinstance(donor, str) else None
        if donor_key is None or donor == key["paper_id"] or hashes != [donor_key["original_image_sha256"]]:
            raise SyntheticHarnessError("shuffled_images control does not use another keyed image")
    else:
        raise SyntheticHarnessError("prediction row has an unsupported condition")


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        raise SyntheticHarnessError("cannot compute a metric from an empty condition")
    exact = sum(row["exact_match"] for row in rows)
    parsed = sum(row["parse_valid"] for row in rows)
    schema = sum(row["schema_valid"] for row in rows)
    generation_errors = sum(row["generation_error"] for row in rows)
    return {
        "n": count,
        "exact_match_count": exact,
        "exact_accuracy": exact / count,
        "parse_valid_count": parsed,
        "parse_rate": parsed / count,
        "schema_valid_count": schema,
        "schema_rate": schema / count,
        "generation_error_count": generation_errors,
        "generation_error_rate": generation_errors / count,
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Synthetic Exploratory Exact-Match Report",
        "",
        f"**{scope_text()}**",
        "",
        "This is an automated synthetic-data-only exact-match summary. No human review was performed.",
        "It is not suitable for publication and permits no publication or superiority claim.",
        "",
        "## Metrics",
        "",
        "| Arm | Condition | N | Exact accuracy | Parse rate | Schema rate | Generation errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in _ARMS:
        for condition in _CONDITIONS:
            metric = report["metrics"][arm][condition]
            lines.append(
                "| "
                f"{arm} | {condition} | {metric['n']} | {metric['exact_accuracy']:.6f} | "
                f"{metric['parse_rate']:.6f} | {metric['schema_rate']:.6f} | "
                f"{metric['generation_error_count']} |"
            )
    lines.extend(["", "## Original Minus Controls", ""])
    for arm in _ARMS:
        values = report["original_minus_control_exact_accuracy"][arm]
        lines.append(
            f"- `{arm}`: original minus text_only = {values['text_only']:.6f}; "
            f"original minus shuffled_images = {values['shuffled_images']:.6f}."
        )
    lines.extend(
        [
            "",
            "These descriptive differences are not inferential statistics and do not establish a claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _input_bindings(run: Path) -> dict[str, str]:
    paths = {
        "generator_manifest": run / "generator_manifest.json",
        "answer_key": run / "answer_key.jsonl",
        "synthetic_config": run / "synthetic_config.yaml",
        "prepare_manifest": run / "prepare_manifest.json",
        "prepare_receipt": run / "synthetic_prepare_receipt.json",
        "base_predictions": run / "predictions" / "base.jsonl",
        "base_sidecar": run / "predictions" / "base.jsonl.manifest.json",
        "tuned_predictions": run / "predictions" / "tuned.jsonl",
        "tuned_sidecar": run / "predictions" / "tuned.jsonl.manifest.json",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _verify_existing_score(score_dir: Path, bindings: dict[str, str]) -> dict[str, Any]:
    manifest = read_json_object(score_dir / "score_manifest.json", "automated score manifest")
    require_scope_labels(manifest, "automated score manifest")
    if manifest.get("result_scope") != RESULT_SCOPE or manifest.get("input_sha256s") != bindings:
        raise SyntheticHarnessError("existing automated score belongs to different inputs")
    entries = manifest.get("output_inventory")
    if not isinstance(entries, list):
        raise SyntheticHarnessError("existing automated score has no output inventory")
    expected: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise SyntheticHarnessError("existing automated score has a malformed output inventory")
        expected[entry["path"]] = entry
    actual = inventory(score_dir)
    actual_without_manifest = [entry for entry in actual if entry["path"] != "score_manifest.json"]
    if actual_without_manifest != entries or {entry["path"] for entry in actual} != {
        *expected,
        "score_manifest.json",
    }:
        raise SyntheticHarnessError("existing automated score files differ from its inventory")
    report = read_json_object(score_dir / "synthetic_report.json", "synthetic score report")
    require_scope_labels(report, "synthetic score report")
    return {
        **scope_metadata(),
        "score_dir": str(score_dir),
        "report": str(score_dir / "synthetic_report.json"),
        "scored_rows": str(score_dir / "scored_rows.jsonl"),
        "idempotent": True,
    }


def score(repo_root_value: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Validate immutable predictions and write one local exact-match score bundle."""

    root = repo_root(repo_root_value)
    generated = verify_generated_run(root, run_dir)
    run = generated["run"]
    prepared, frozen = _load_prepared(root, generated)
    expected_inputs = _prediction_inputs(root, generated, prepared, frozen)
    expected_keys = {
        (sample_id, condition)
        for sample_id in {row["sample_id"] for row in frozen}
        for condition in _CONDITIONS
    }
    if set(expected_inputs) != expected_keys:
        raise SyntheticHarnessError("synthetic config does not generate all required controls")
    expected_row_count = len(expected_keys)
    bindings_before = _input_bindings(run)
    keys_by_sample = {record["sample_id"]: record for record in generated["keys"]}
    keys_by_paper = {record["paper_id"]: record for record in generated["keys"]}

    scored: list[dict[str, Any]] = []
    all_metrics: dict[str, dict[str, Any]] = {}
    protocol_fingerprints: set[str] = set()
    shared_backends: set[str] = set()
    for arm in _ARMS:
        prediction_path, sidecar, rows = _verify_sidecar(
            run,
            arm,
            generated["config"],
            expected_row_count,
        )
        del prediction_path
        protocol = sidecar.get("protocol_fingerprint")
        if not isinstance(protocol, str) or not _SHA256_RE.fullmatch(protocol):
            raise SyntheticHarnessError(f"{arm} sidecar has no valid protocol fingerprint")
        protocol_fingerprints.add(protocol)
        backend = sidecar["backend"]
        shared_backends.add(backend)
        seen: set[tuple[str, str]] = set()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            sample_id = row.get("sample_id")
            condition = row.get("condition")
            if not isinstance(sample_id, str) or not isinstance(condition, str):
                raise SyntheticHarnessError("prediction row has no valid key")
            key = (sample_id, condition)
            if key in seen or key not in expected_inputs:
                raise SyntheticHarnessError("prediction rows have duplicate or unexpected keys")
            seen.add(key)
            _verify_row_controls(
                row,
                expected_inputs[key],
                keys_by_sample,
                keys_by_paper,
                arm,
                backend,
            )
            status = row.get("status")
            if status not in {"success", "error"}:
                raise SyntheticHarnessError("prediction row has an unsupported generation status")
            parsed = _parse_raw_response(row.get("raw_response")) if status == "success" else None
            parse_valid = parsed is not None
            schema_valid = _schema_valid(parsed)
            answer = parsed.get("answer") if isinstance(parsed, dict) else None
            exact_match = status == "success" and answer == keys_by_sample[sample_id]["expected_answer"]
            raw = row.get("raw_response")
            scored_row = {
                **scope_metadata(),
                "arm": arm,
                "backend": backend,
                "sample_id": sample_id,
                "paper_id": row["paper_id"],
                "condition": condition,
                "generation_error": status != "success",
                "parse_valid": parse_valid,
                "schema_valid": schema_valid,
                "exact_match": exact_match,
                "input_image_hashes": row["input_image_hashes"],
                "shuffle_source_paper_id": row["shuffle_source_paper_id"],
                "raw_response_sha256": (
                    hashlib.sha256(raw.encode("utf-8")).hexdigest() if isinstance(raw, str) else None
                ),
            }
            scored.append(scored_row)
            grouped[condition].append(scored_row)
        if seen != expected_keys:
            raise SyntheticHarnessError(f"{arm} prediction keyset does not match all synthetic controls")
        all_metrics[arm] = {condition: _metric(grouped[condition]) for condition in _CONDITIONS}
    if len(protocol_fingerprints) != 1 or len(shared_backends) != 1:
        raise SyntheticHarnessError("base and tuned predictions do not share one protocol/backend")

    differences = {
        arm: {
            condition: all_metrics[arm]["original"]["exact_accuracy"]
            - all_metrics[arm][condition]["exact_accuracy"]
            for condition in ("text_only", "shuffled_images")
        }
        for arm in _ARMS
    }
    report = {
        **scope_metadata(),
        "artifact_version": 1,
        "result_scope": RESULT_SCOPE,
        "automated_exact_match_only": True,
        "human_review_performed": False,
        "publication_or_superiority_claim_allowed": False,
        "sample_count": generated["manifest"]["sample_count"],
        "conditions": list(_CONDITIONS),
        "arms": list(_ARMS),
        "backend": next(iter(shared_backends)),
        "protocol_fingerprint": next(iter(protocol_fingerprints)),
        "metrics": all_metrics,
        "original_minus_control_exact_accuracy": differences,
    }
    report_markdown = _report_markdown(report)
    score_dir = run / "automated_score"
    if score_dir.exists() or score_dir.is_symlink():
        return _verify_existing_score(score_dir, bindings_before)
    temporary = score_dir.with_name(f".{score_dir.name}.{uuid.uuid4().hex}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary score directory already exists: {temporary}")
    try:
        temporary.mkdir()
        scored.sort(key=lambda row: (row["arm"], row["sample_id"], _CONDITIONS.index(row["condition"])))
        write_jsonl_new(temporary / "scored_rows.jsonl", scored)
        write_json_new(temporary / "synthetic_report.json", report)
        write_text_new(temporary / "report.md", report_markdown)
        output_inventory = inventory(temporary)
        score_manifest = {
            **scope_metadata(),
            "artifact_version": 1,
            "result_scope": RESULT_SCOPE,
            "input_sha256s": bindings_before,
            "output_inventory": output_inventory,
        }
        write_json_new(temporary / "score_manifest.json", score_manifest)
        if _input_bindings(run) != bindings_before:
            raise SyntheticHarnessError("predictions or sidecars changed while scoring; no result was written")
        os.rename(temporary, score_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        **scope_metadata(),
        "score_dir": str(score_dir),
        "report": str(score_dir / "synthetic_report.json"),
        "scored_rows": str(score_dir / "scored_rows.jsonl"),
        "idempotent": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = score(args.repo_root, args.run_dir)
    except (OSError, RuntimeError, ValueError, SyntheticHarnessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
