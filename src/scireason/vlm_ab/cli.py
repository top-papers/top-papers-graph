# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command-line orchestration for the publication VLM A/B experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .audit import canonical_paper_id, load_jsonl
from .blind import build_blind_review_packages, deblind_reviews
from .config import config_fingerprint, load_experiment_config
from .curator import generate_curator_workspace
from .inference import manifest_path_for, run_inference
from .prepare import (
    PublicationGateError,
    code_provenance,
    prepare_experiment,
    resolve_prepare_manifest,
    verify_prepared_audit,
    verify_prepare_manifest,
)
from .remediation import assemble_corrected_release, generate_curator_queues
from .reporting import (
    automatic_output_metrics,
    build_markdown_report,
    error_tag_metrics,
    review_missingness_metrics,
    write_effect_svg,
    write_paper_scores,
)
from .stats import power_mde_plan, summarize_reviews


def _repo_root(config_path: Path, explicit_root: Path | None = None) -> Path:
    def validated(root: Path) -> Path:
        expected_module = root / "src" / "scireason" / "vlm_ab" / "cli.py"
        try:
            if expected_module.resolve(strict=True) != Path(__file__).resolve(strict=True):
                raise RuntimeError("executing VLM A/B package differs from --repo-root source tree")
        except OSError as exc:
            raise RuntimeError("repository root has no executable VLM A/B source tree") from exc
        return root

    if explicit_root is not None:
        root = explicit_root.resolve(strict=True)
        if not root.is_dir():
            raise RuntimeError("--repo-root must identify a directory")
        return validated(root)
    config_parent = config_path.resolve(strict=True).parent
    for candidate in (config_parent, *config_parent.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return validated(candidate)
    raise RuntimeError("cannot locate repository root; pass --repo-root explicitly")


def _output_dir(config: dict[str, Any], repo_root: Path) -> Path:
    return (repo_root / config["experiment"]["output_dir"]).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise PublicationGateError(f"{label} must be a non-empty ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicationGateError(f"{label} is not a valid ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PublicationGateError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hmac(payload: dict[str, Any], secret: str) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(bytes.fromhex(secret), encoded, hashlib.sha256).hexdigest()


def _safe_artifact(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"artifact path escapes its root: {relative}") from exc
    if not candidate.is_file():
        raise RuntimeError(f"artifact is not a file: {relative}")
    return candidate


def _load_prepare(config: dict[str, Any], repo_root: Path, exploratory: bool) -> dict[str, Any]:
    path = _output_dir(config, repo_root) / "prepare_manifest.json"
    manifest = _read_json(path)
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise RuntimeError("prepare manifest was created from a different configuration")
    expected_code = (manifest.get("code_provenance") or {}).get("source_fingerprint")
    current_code = code_provenance(repo_root)
    if expected_code != current_code.get("source_fingerprint"):
        raise RuntimeError("evaluation source tree changed after prepare")
    stored_code = manifest.get("code_provenance") or {}
    clean_code_attested = current_code.get("git_dirty") is False or (
        current_code.get("git_dirty") is None
        and stored_code.get("git_dirty") is False
        and isinstance(stored_code.get("git_head"), str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", stored_code["git_head"]))
    )
    if (
        not exploratory
        and config["experiment"].get("require_clean_code", False)
        and not clean_code_attested
    ):
        raise RuntimeError("publication inference requires a clean evaluation source tree")
    verify_prepare_manifest(manifest, path.parent)
    prepared = resolve_prepare_manifest(manifest, path.parent)
    audit = verify_prepared_audit(config, manifest, path.parent)
    if config["experiment"].get("require_preregistered_plan", False) and not exploratory:
        if not prepared.get("power_plan"):
            raise PublicationGateError("strict run has no preregistered power plan")
        power_plan = _read_json(Path(prepared["power_plan"]))
        plan_created_at = _parse_timestamp(power_plan.get("created_at"), "power plan created_at")
        prepare_created_at = _parse_timestamp(manifest.get("created_at"), "prepare created_at")
        if (
            power_plan.get("config_fingerprint") != config_fingerprint(config)
            or power_plan.get("code_fingerprint") != expected_code
            or plan_created_at > prepare_created_at
        ):
            raise PublicationGateError("preregistered power plan is inconsistent or post-dated")
    sources = manifest.get("sources")
    source_revisions_verified = bool(
        isinstance(sources, list)
        and sources
        and all(
            isinstance(source, dict)
            and source.get("configured_revision") == source.get("resolved_revision")
            for source in sources
        )
    )
    expected_code_gate = (
        not config["experiment"].get("require_clean_code", False) or clean_code_attested
    )
    expected_publication_ready = bool(
        audit.get("publication_ready") is True
        and expected_code_gate
        and source_revisions_verified
        and manifest.get("exploratory") is False
    )
    if (
        manifest.get("benchmark_audit_status") != audit.get("status")
        or manifest.get("code_gate_passed") is not expected_code_gate
        or manifest.get("publication_ready") is not expected_publication_ready
    ):
        raise PublicationGateError("prepare manifest status is inconsistent with verified inputs")
    if bool(manifest.get("exploratory")) != exploratory:
        raise PublicationGateError("prepare and downstream exploratory modes must match")
    expected_scope = "exploratory_not_for_publication" if exploratory else "publication_ready"
    if manifest.get("result_scope") != expected_scope:
        raise PublicationGateError("prepare manifest has the wrong result scope")
    if not expected_publication_ready and not exploratory:
        raise PublicationGateError(
            "benchmark is not publication-ready; pass --exploratory only for diagnostics"
        )
    return prepared


def _arm_config(config: dict[str, Any], arm: str) -> dict[str, Any]:
    return dict(config["models"][arm])


def _processor_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["processor"])


def _prediction_path(config: dict[str, Any], repo_root: Path, arm: str) -> Path:
    return _output_dir(config, repo_root) / "predictions" / f"{arm}.jsonl"


def _run_fingerprint(config: dict[str, Any], prepared: dict[str, Any]) -> str:
    payload = {
        "config_fingerprint": config_fingerprint(config),
        "code_fingerprint": (prepared.get("code_provenance") or {}).get("source_fingerprint"),
        "frozen_benchmark_sha256": prepared.get("frozen_benchmark_sha256"),
        "audit_json_sha256": prepared.get("audit_json_sha256"),
        "power_plan_sha256": prepared.get("power_plan_sha256"),
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def command_plan(args: argparse.Namespace, config: dict[str, Any], root: Path) -> dict[str, Any]:
    power = dict(config["power"])
    n_items = int(power.pop("n_items"))
    reviews = float(power.pop("reviews_per_item", config["review"]["reviews_per_item"]))
    plan = {
        "artifact_version": 1,
        "created_at": None,
        "experiment_id": config["experiment"]["id"],
        "config_fingerprint": config_fingerprint(config),
        "code_fingerprint": code_provenance(root)["source_fingerprint"],
        **power_mde_plan(n_items, reviews, **power),
    }
    path = _output_dir(config, root) / "design" / "power_plan.json"
    if path.exists():
        existing = _read_json(path)
        created_at = _parse_timestamp(existing.get("created_at"), "power plan created_at")
        if created_at > datetime.now(timezone.utc):
            raise PublicationGateError("power plan preregistration timestamp is in the future")
        comparable = dict(existing)
        comparable["created_at"] = None
        if comparable != plan:
            raise RuntimeError(
                "power plan already exists for another protocol; use a new experiment output_dir"
            )
        return {"power_plan": str(path), **existing}
    plan["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write_json(path, plan)
    return {"power_plan": str(path), **plan}


def command_prepare(args: argparse.Namespace, config: dict[str, Any], root: Path) -> dict[str, Any]:
    return prepare_experiment(
        config,
        root,
        benchmark_dir=args.benchmark_dir,
        training_files=args.training_file,
        cache_dir=args.cache_dir,
        exploratory=args.exploratory,
    )


def _remediation_schema(root: Path, name: str) -> Path:
    return root / "experiments" / "vlm_ab_evaluation" / "schemas" / name


def command_curate_queue(
    args: argparse.Namespace, config: dict[str, Any], root: Path
) -> dict[str, Any]:
    return generate_curator_queues(
        config,
        args.prepare_manifest,
        args.output_dir,
        benchmark_schema=_remediation_schema(
            root,
            "publication_benchmark_row.schema.json",
        ),
        provenance_schema=_remediation_schema(
            root,
            "publication_provenance_row.schema.json",
        ),
    )


def command_curate_forms(
    args: argparse.Namespace, config: dict[str, Any], root: Path
) -> dict[str, Any]:
    return generate_curator_workspace(
        config,
        args.prepare_manifest,
        args.queue_manifest,
        args.output_dir,
        benchmark_schema=_remediation_schema(root, "publication_benchmark_row.schema.json"),
        provenance_schema=_remediation_schema(root, "publication_provenance_row.schema.json"),
    )


def command_curate_assemble(
    args: argparse.Namespace, config: dict[str, Any], root: Path
) -> dict[str, Any]:
    return assemble_corrected_release(
        config,
        args.prepare_manifest,
        args.queue_manifest,
        args.decisions_jsonl,
        args.curated_dataset_root,
        args.output_dir,
        benchmark_schema=_remediation_schema(
            root,
            "publication_benchmark_row.schema.json",
        ),
        provenance_schema=_remediation_schema(
            root,
            "publication_provenance_row.schema.json",
        ),
    )


def command_infer(args: argparse.Namespace, config: dict[str, Any], root: Path) -> dict[str, Any]:
    if not args.exploratory and args.arm == "all":
        raise RuntimeError(
            "strict inference requires one arm per process; pass --arm base or tuned"
        )
    if not args.exploratory and (args.backend != "transformers" or args.limit is not None):
        raise RuntimeError("mock inference and --limit are allowed only with --exploratory")
    prepared = _load_prepare(config, root, args.exploratory)
    benchmark = load_jsonl(prepared["frozen_benchmark"])
    run_fingerprint = _run_fingerprint(config, prepared)
    arms = ("base", "tuned") if args.arm == "all" else (args.arm,)
    summaries: dict[str, Any] = {}
    protocol_fingerprint = None
    for arm in arms:
        summaries[arm] = run_inference(
            benchmark=benchmark,
            dataset_root=prepared["dataset_root"],
            arm_name=arm,
            arm_config=_arm_config(config, arm),
            processor_config=_processor_config(config),
            generation_config=config["generation"],
            output_jsonl=_prediction_path(config, root, arm),
            conditions=config["conditions"],
            seed=config["experiment"]["seed"],
            limit=args.limit,
            backend=args.backend,
            experiment_fingerprint=run_fingerprint,
            result_scope=(
                "exploratory_not_for_publication" if args.exploratory else "publication_candidate"
            ),
        )
        current = summaries[arm]["protocol_fingerprint"]
        if protocol_fingerprint is not None and current != protocol_fingerprint:
            raise RuntimeError("base and tuned inference protocols differ")
        protocol_fingerprint = current
        _write_json(
            _output_dir(config, root) / "predictions" / f"{arm}.summary.json",
            summaries[arm],
        )
        _release_gpu()
    summary_path = _output_dir(config, root) / "inference_summary.json"
    experiment_fingerprint = config_fingerprint(config)
    result_scope = (
        "exploratory_not_for_publication" if args.exploratory else "publication_candidate"
    )
    merged_arms: dict[str, Any] = {}
    if summary_path.exists():
        previous = _read_json(summary_path)
        if previous.get("experiment_config_fingerprint") != experiment_fingerprint:
            raise RuntimeError("existing inference summary belongs to another configuration")
        if previous.get("result_scope") != result_scope:
            raise RuntimeError("existing inference summary has another result scope")
        if previous.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("existing inference summary belongs to another prepared run")
        if isinstance(previous.get("arms"), dict):
            merged_arms.update(previous["arms"])
    merged_arms.update(summaries)
    protocols = {
        str(summary.get("protocol_fingerprint") or "")
        for summary in merged_arms.values()
        if isinstance(summary, dict)
    }
    if len(protocols) != 1 or "" in protocols:
        raise RuntimeError("merged base and tuned inference summaries use different protocols")
    result = {
        "experiment_id": config["experiment"]["id"],
        "experiment_config_fingerprint": experiment_fingerprint,
        "run_fingerprint": run_fingerprint,
        "exploratory": bool(args.exploratory),
        "result_scope": result_scope,
        "protocol_fingerprint": next(iter(protocols)),
        "arms": merged_arms,
    }
    _write_json(summary_path, result)
    return result


def _validate_paired_outputs(
    base: list[dict[str, Any]],
    tuned: list[dict[str, Any]],
    expected_original_ids: set[str] | None = None,
    expected_conditions: list[str] | None = None,
    manifests: dict[str, dict[str, Any]] | None = None,
) -> None:
    base_protocols = {row.get("protocol_fingerprint") for row in base}
    tuned_protocols = {row.get("protocol_fingerprint") for row in tuned}
    if len(base_protocols) != 1 or base_protocols != tuned_protocols:
        raise RuntimeError("base and tuned outputs were not generated by one identical protocol")
    base_keys = {(row.get("sample_id"), row.get("condition")) for row in base}
    tuned_keys = {(row.get("sample_id"), row.get("condition")) for row in tuned}
    if len(base_keys) != len(base) or len(tuned_keys) != len(tuned):
        raise RuntimeError("prediction files contain duplicate sample/condition keys")
    if base_keys != tuned_keys:
        raise RuntimeError("base and tuned prediction key sets differ")
    if manifests is not None:
        for arm, rows in (("base", base), ("tuned", tuned)):
            manifest = manifests[arm]
            expected_backend = manifest.get("backend")
            expected_config = manifest.get("config_fingerprint")
            for row in rows:
                if row.get("arm") != arm:
                    raise RuntimeError(f"{arm} prediction file contains a row from another arm")
                if row.get("backend") != expected_backend:
                    raise RuntimeError(f"{arm} prediction backend differs from its manifest")
                if row.get("config_fingerprint") != expected_config:
                    raise RuntimeError(f"{arm} prediction config differs from its manifest")
    if expected_original_ids is not None:
        conditions = expected_conditions or ["original"]
        expected = {
            (sample_id, condition)
            for sample_id in expected_original_ids
            for condition in conditions
        }
        if base_keys != expected:
            raise RuntimeError(
                "paired prediction key set differs from the frozen benchmark and conditions"
            )


def _validate_prediction_manifests(
    config: dict[str, Any],
    root: Path,
    *,
    expected_samples: int,
    exploratory: bool,
    run_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    protocols: set[str] = set()
    runtimes: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    expected_rows = expected_samples * len(config["conditions"])
    for arm in ("base", "tuned"):
        output = _prediction_path(config, root, arm)
        manifest = _read_json(manifest_path_for(output))
        manifests[arm] = manifest
        if manifest.get("status") != "complete":
            raise RuntimeError(f"{arm} inference manifest is not complete")
        if manifest.get("completed_rows") != manifest.get("expected_rows"):
            raise RuntimeError(f"{arm} inference manifest has incomplete rows")
        if manifest.get("expected_rows") != expected_rows:
            raise RuntimeError(f"{arm} inference manifest has the wrong expected row count")
        if manifest.get("arm") != arm:
            raise RuntimeError(f"{arm} inference manifest identifies another arm")
        if manifest.get("experiment_fingerprint") != run_fingerprint:
            raise RuntimeError(f"{arm} inference manifest belongs to another prepared run")
        expected_scope = (
            "exploratory_not_for_publication" if exploratory else "publication_candidate"
        )
        if manifest.get("result_scope") != expected_scope:
            raise RuntimeError(f"{arm} inference manifest has the wrong result scope")
        if manifest.get("conditions") != config["conditions"]:
            raise RuntimeError(f"{arm} inference conditions differ from the configuration")
        if manifest.get("seed") != config["experiment"]["seed"]:
            raise RuntimeError(f"{arm} inference seed differs from the configuration")
        if not exploratory and (
            manifest.get("backend") != "transformers" or manifest.get("limit") is not None
        ):
            raise RuntimeError(f"{arm} inference is not a full transformers run")
        configuration = manifest.get("configuration")
        if not isinstance(configuration, dict) or (
            configuration.get("arm") != _arm_config(config, arm)
            or configuration.get("processor") != _processor_config(config)
            or configuration.get("generation") != config["generation"]
        ):
            raise RuntimeError(f"{arm} inference configuration differs from the current config")
        expected_hash = manifest.get("output_sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise RuntimeError(f"{arm} inference manifest has no valid output SHA256")
        if _sha256(output.resolve(strict=True)) != expected_hash:
            raise RuntimeError(f"{arm} prediction bytes differ from the complete manifest")
        protocols.add(str(manifest.get("protocol_fingerprint") or ""))
        runtime = manifest.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError(f"{arm} inference manifest has no runtime environment")
        runtimes.append(runtime)
    if len(protocols) != 1 or "" in protocols:
        raise RuntimeError("base and tuned inference manifests use different protocols")
    if runtimes[0] != runtimes[1]:
        raise RuntimeError("base and tuned inference runtime environments differ")
    return manifests


def _blinding_secret(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        value = path.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(f"invalid existing blinding secret: {path}")
        return value
    value = secrets.token_hex(32)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(value + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return value


def command_blind(args: argparse.Namespace, config: dict[str, Any], root: Path) -> dict[str, Any]:
    if not args.exploratory and (args.reviewers or args.reviews_per_item is not None):
        raise RuntimeError("review design overrides are allowed only with --exploratory")
    prepared = _load_prepare(config, root, args.exploratory)
    benchmark = load_jsonl(prepared["frozen_benchmark"])
    base = load_jsonl(_prediction_path(config, root, "base"))
    tuned = load_jsonl(_prediction_path(config, root, "tuned"))
    run_fingerprint = _run_fingerprint(config, prepared)
    manifests = _validate_prediction_manifests(
        config,
        root,
        expected_samples=len(benchmark),
        exploratory=args.exploratory,
        run_fingerprint=run_fingerprint,
    )
    _validate_paired_outputs(
        base,
        tuned,
        {row["sample_id"] for row in benchmark},
        config["conditions"],
        manifests,
    )
    reviewers = args.reviewers or config["review"]["reviewer_ids"]
    reviews_per_item = args.reviews_per_item or config["review"]["reviews_per_item"]
    output_dir = _output_dir(config, root)
    review_root = output_dir / "blind_review"
    blind_manifest_path = output_dir / "blind_review_manifest.json"
    if blind_manifest_path.is_symlink():
        raise RuntimeError("blind review manifest path is a symlink")
    if blind_manifest_path.is_file():
        manifest, _, _ = _verify_blind_manifest(
            config,
            output_dir,
            args.exploratory,
            run_fingerprint,
        )
        if (
            manifest.get("reviewer_ids") != sorted(reviewers)
            or manifest.get("reviews_per_item") != reviews_per_item
        ):
            raise RuntimeError("existing blind package uses another reviewer design")
        return manifest
    if blind_manifest_path.exists():
        raise RuntimeError("blind review manifest path is not a regular file")
    public_root = review_root / "public"
    if public_root.is_symlink():
        raise RuntimeError("incomplete blind public path is a symlink")
    if public_root.exists():
        if not public_root.is_dir():
            raise RuntimeError("incomplete blind public path is not a directory")
        shutil.rmtree(public_root)
    secret_path = review_root / "owner_only" / "randomization_secret.txt"
    blinding_secret = _blinding_secret(secret_path)
    owner_path = review_root / "owner_only" / "owner_only.json"
    build = build_blind_review_packages(
        benchmark,
        base,
        tuned,
        prepared["dataset_root"],
        review_root / "public",
        reviewers,
        reviews_per_item,
        blinding_secret,
        primary_condition=config["review"].get("primary_condition", "original"),
        experiment_id=config["experiment"].get("public_id"),
        owner_mapping_path=owner_path,
    )
    result = {
        "experiment_id": build.experiment_id,
        "study_fingerprint": build.study_fingerprint,
        "experiment_config_fingerprint": config_fingerprint(config),
        "run_fingerprint": run_fingerprint,
        "result_scope": (
            "exploratory_not_for_publication" if args.exploratory else "publication_candidate"
        ),
        "reviewer_ids": sorted(reviewers),
        "reviews_per_item": reviews_per_item,
        "prediction_sha256s": {arm: manifests[arm]["output_sha256"] for arm in ("base", "tuned")},
        "prediction_manifest_sha256s": {
            arm: _sha256(
                manifest_path_for(_prediction_path(config, root, arm)).resolve(strict=True)
            )
            for arm in ("base", "tuned")
        },
        "paired_items": build.paired_item_count,
        "total_assignments": build.total_assignment_count,
        "owner_mapping": str(build.owner_mapping_path),
        "randomization_secret_sha256": hashlib.sha256(blinding_secret.encode("ascii")).hexdigest(),
        "reviewer_packages": {
            reviewer: {
                "directory": str(package.package_dir),
                "html": str(package.html_path),
                "assignment": str(package.assignment_path),
            }
            for reviewer, package in build.reviewer_packages.items()
        },
    }
    public_files = []
    for package in build.reviewer_packages.values():
        for path in (package.assignment_path, package.html_path, *package.image_paths):
            public_files.append(
                {
                    "path": path.relative_to(public_root).as_posix(),
                    "sha256": _sha256(path),
                }
            )
    result["public_files"] = sorted(public_files, key=lambda entry: entry["path"])
    result["owner_mapping_sha256"] = _sha256(build.owner_mapping_path)
    result["integrity_hmac_sha256"] = _manifest_hmac(result, blinding_secret)
    _write_json_atomic(blind_manifest_path, result)
    _verify_blind_manifest(
        config,
        output_dir,
        args.exploratory,
        run_fingerprint,
    )
    return result


def _verify_blind_manifest(
    config: dict[str, Any],
    output_dir: Path,
    exploratory: bool,
    run_fingerprint: str,
) -> tuple[dict[str, Any], Path, str]:
    review_root = output_dir / "blind_review"
    secret_path = review_root / "owner_only" / "randomization_secret.txt"
    try:
        secret = secret_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("cannot read the owner-only randomization secret") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", secret):
        raise RuntimeError("owner-only randomization secret is malformed")
    manifest = _read_json(output_dir / "blind_review_manifest.json")
    signature = manifest.get("integrity_hmac_sha256")
    if not isinstance(signature, str):
        raise RuntimeError("blind manifest has no integrity HMAC")
    unsigned = {key: value for key, value in manifest.items() if key != "integrity_hmac_sha256"}
    if not hmac.compare_digest(signature, _manifest_hmac(unsigned, secret)):
        raise RuntimeError("blind manifest integrity HMAC does not match")
    if manifest.get("experiment_config_fingerprint") != config_fingerprint(config):
        raise RuntimeError("blind manifest belongs to another experiment config")
    if manifest.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError("blind manifest belongs to another prepared run")
    current_prediction_hashes = {
        arm: _read_json(manifest_path_for(output_dir / "predictions" / f"{arm}.jsonl")).get(
            "output_sha256"
        )
        for arm in ("base", "tuned")
    }
    if manifest.get("prediction_sha256s") != current_prediction_hashes:
        raise RuntimeError("blind packages belong to different prediction bytes")
    current_manifest_hashes = {
        arm: _sha256(
            manifest_path_for(output_dir / "predictions" / f"{arm}.jsonl").resolve(strict=True)
        )
        for arm in ("base", "tuned")
    }
    if manifest.get("prediction_manifest_sha256s") != current_manifest_hashes:
        raise RuntimeError("prediction manifest bytes changed after blind packaging")
    expected_scope = "exploratory_not_for_publication" if exploratory else "publication_candidate"
    if manifest.get("result_scope") != expected_scope:
        raise RuntimeError("blind manifest has the wrong result scope")
    if not exploratory and manifest.get("reviewer_ids") != sorted(config["review"]["reviewer_ids"]):
        raise RuntimeError("blind manifest reviewer roster differs from the configuration")
    if not exploratory and manifest.get("reviews_per_item") != config["review"]["reviews_per_item"]:
        raise RuntimeError("blind manifest reviews_per_item differs from the configuration")
    if (
        manifest.get("randomization_secret_sha256")
        != hashlib.sha256(secret.encode("ascii")).hexdigest()
    ):
        raise RuntimeError("blind manifest references another randomization secret")
    owner = review_root / "owner_only" / "owner_only.json"
    if _sha256(owner.resolve(strict=True)) != manifest.get("owner_mapping_sha256"):
        raise RuntimeError("owner mapping bytes differ from the blind manifest")
    public_root = review_root / "public"
    raw_files = manifest.get("public_files")
    if not isinstance(raw_files, list) or not raw_files:
        raise RuntimeError("blind manifest contains no public file inventory")
    listed_paths: set[str] = set()
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise RuntimeError("blind manifest contains a malformed public file entry")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise RuntimeError("blind manifest contains an incomplete public file entry")
        if relative in listed_paths:
            raise RuntimeError("blind manifest contains a duplicate public file entry")
        listed_paths.add(relative)
        if _sha256(_safe_artifact(public_root, relative)) != expected_hash:
            raise RuntimeError(f"blind public artifact changed: {relative}")
    actual_paths = {
        path.relative_to(public_root).as_posix()
        for path in public_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != listed_paths:
        raise RuntimeError("blind public directory contains unlisted or missing files")
    return manifest, owner, secret


def _review_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path).resolve(strict=True) for path in args.review_file]
    if args.reviews_dir:
        directory = Path(args.reviews_dir).resolve(strict=True)
        paths.extend(sorted(directory.glob("*.json")))
    unique = sorted(set(paths))
    if not unique:
        raise RuntimeError("no review exports were supplied")
    return unique


def _join_review_metadata(
    rows: list[dict[str, Any]], benchmark: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    index = {row["sample_id"]: row for row in benchmark}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        sample = index.get(row["sample_id"])
        if sample is None:
            raise RuntimeError(f"review references unknown sample {row['sample_id']!r}")
        paper_id = canonical_paper_id(sample.get("paper_id"))
        if not paper_id:
            raise RuntimeError(f"sample {row['sample_id']!r} has no canonical paper ID")
        if not isinstance(sample.get("primary_endpoint"), bool):
            raise RuntimeError(f"sample {row['sample_id']!r} lacks primary_endpoint metadata")
        enriched.append(
            {
                **row,
                "paper_id": paper_id,
                "stratum": str(sample.get("stratum") or "unspecified"),
                "primary_endpoint": sample["primary_endpoint"],
                "evidence_kind": str(sample.get("evidence_kind") or "unspecified"),
            }
        )
    return enriched


def _check_review_completeness(
    rows: list[dict[str, Any]], owner_path: Path, allow_incomplete: bool
) -> tuple[int, int]:
    owner = _read_json(owner_path)
    expected = {(row["reviewer_id"], row["assignment_id"]) for row in owner.get("assignments", [])}
    observed = {(row["reviewer_id"], row["assignment_id"]) for row in rows}
    missing = expected - observed
    extra = observed - expected
    if extra:
        raise RuntimeError(f"review exports contain {len(extra)} unknown assignments")
    if missing and not allow_incomplete:
        raise RuntimeError(f"review exports are missing {len(missing)} assigned judgments")
    return len(expected), len(missing)


def _validate_realized_review_design(
    owner_path: Path,
    benchmark: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    owner = _read_json(owner_path)
    assignments = owner.get("assignments")
    if not isinstance(assignments, list):
        raise RuntimeError("owner mapping has no assignment inventory")
    expected_samples = {row["sample_id"] for row in benchmark}
    expected_reviewers = set(config["review"]["reviewer_ids"])
    primary_condition = config["review"].get("primary_condition", "original")
    per_sample: dict[str, set[str]] = {sample_id: set() for sample_id in expected_samples}
    observed_reviewers: set[str] = set()
    for entry in assignments:
        if not isinstance(entry, dict):
            raise RuntimeError("owner mapping contains a malformed assignment")
        sample_id = entry.get("sample_id")
        reviewer_id = entry.get("reviewer_id")
        if sample_id not in per_sample or reviewer_id not in expected_reviewers:
            raise RuntimeError("owner mapping contains an unexpected sample or reviewer")
        if entry.get("condition") != primary_condition:
            raise RuntimeError("owner mapping contains a non-primary review condition")
        if reviewer_id in per_sample[sample_id]:
            raise RuntimeError("a reviewer is assigned to the same sample more than once")
        per_sample[sample_id].add(reviewer_id)
        observed_reviewers.add(reviewer_id)
    expected_depth = config["review"]["reviews_per_item"]
    if any(len(reviewers) != expected_depth for reviewers in per_sample.values()):
        raise RuntimeError("realized reviewer assignment depth differs from the configuration")
    if observed_reviewers != expected_reviewers:
        raise RuntimeError("realized reviewer roster differs from the configuration")


def _stratum_summaries(rows: list[dict[str, Any]], statistics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in sorted({row["stratum"] for row in rows}):
        summary = summarize_reviews(
            rows,
            {
                **statistics,
                "primary_strata": [stratum],
                "primary_endpoint_only": False,
                "secondary_criteria": [],
            },
        )
        summary["inference_scope"] = "exploratory_unadjusted"
        result[stratum] = summary
    return result


def _evidence_summaries(rows: list[dict[str, Any]], statistics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for evidence_kind in sorted({row["evidence_kind"] for row in rows}):
        selected = [row for row in rows if row["evidence_kind"] == evidence_kind]
        summary = summarize_reviews(
            selected,
            {**statistics, "secondary_criteria": []},
        )
        summary["inference_scope"] = "exploratory_unadjusted"
        result[evidence_kind] = summary
    return result


def command_aggregate(
    args: argparse.Namespace, config: dict[str, Any], root: Path
) -> dict[str, Any]:
    if not args.exploratory and args.allow_incomplete:
        raise RuntimeError("--allow-incomplete is allowed only with --exploratory")
    prepared = _load_prepare(config, root, args.exploratory)
    output_dir = _output_dir(config, root)
    benchmark = load_jsonl(prepared["frozen_benchmark"])
    base_predictions = load_jsonl(_prediction_path(config, root, "base"))
    tuned_predictions = load_jsonl(_prediction_path(config, root, "tuned"))
    manifests = _validate_prediction_manifests(
        config,
        root,
        expected_samples=len(benchmark),
        exploratory=args.exploratory,
        run_fingerprint=_run_fingerprint(config, prepared),
    )
    _validate_paired_outputs(
        base_predictions,
        tuned_predictions,
        {row["sample_id"] for row in benchmark},
        config["conditions"],
        manifests,
    )
    blind_manifest, owner, blinding_secret = _verify_blind_manifest(
        config,
        output_dir,
        args.exploratory,
        _run_fingerprint(config, prepared),
    )
    if not args.exploratory:
        _validate_realized_review_design(owner, benchmark, config)
    review_paths = _review_paths(args)
    deblinded = deblind_reviews(review_paths, owner, blinding_secret=blinding_secret)
    expected_reviews, missing_reviews = _check_review_completeness(
        deblinded, owner, args.allow_incomplete
    )
    rows = _join_review_metadata(deblinded, benchmark)
    statistics = {
        "primary_criterion": "overall_preference",
        "secondary_criteria": [
            "evidence_preference",
            "visual_preference",
            "temporal_preference",
        ],
        **config["statistics"],
        "seed": config["experiment"]["seed"],
    }
    human = summarize_reviews(rows, statistics)
    power_config = dict(config["power"])
    planned_papers = int(power_config.pop("n_items"))
    planned_reviews = float(
        power_config.pop("reviews_per_item", config["review"]["reviews_per_item"])
    )
    power = power_mde_plan(planned_papers, planned_reviews, **power_config)
    review_evaluable_rate = (
        human["primary"]["n_evaluable_reviews"] / human["primary"]["n_reviews"]
        if human["primary"]["n_reviews"]
        else 0.0
    )
    paper_evaluable_rate = (
        human["primary"]["n_papers"] / human["primary"]["n_assigned_papers"]
        if human["primary"]["n_assigned_papers"]
        else 0.0
    )
    artifact_ready = bool(
        prepared.get("publication_ready")
        and not args.exploratory
        and not missing_reviews
        and human.get("n_assigned_papers", 0) >= planned_papers
        and human.get("n_reviewers", 0) == len(config["review"]["reviewer_ids"])
        and paper_evaluable_rate >= float(config["power"].get("evaluable_fraction", 1.0))
        and float(power.get("achieved_power") or 0.0)
        >= float(config["power"].get("target_power", 0.8))
    )
    primary = human["primary"]
    ci = primary.get("bootstrap_ci") or [None, None]
    missingness_bounds = primary.get("missingness_worst_best_case_bounds") or [None, None]
    missingness_worst_ci = primary.get("missingness_worst_case_bootstrap_ci") or [None, None]
    alpha = float(config["power"].get("alpha", 0.05))
    superiority_supported = bool(
        artifact_ready
        and primary.get("estimate") is not None
        and primary["estimate"] > 0.5
        and ci[0] is not None
        and ci[0] > 0.5
        and missingness_bounds[0] is not None
        and missingness_bounds[0] > 0.5
        and missingness_worst_ci[0] is not None
        and missingness_worst_ci[0] > 0.5
        and primary.get("missingness_worst_case_p_value") is not None
        and primary["missingness_worst_case_p_value"] < alpha
        and primary.get("p_value") is not None
        and primary["p_value"] < alpha
    )
    automatic_metrics = automatic_output_metrics(base_predictions + tuned_predictions)
    results = {
        "artifact_version": 1,
        "experiment_id": config["experiment"]["id"],
        "blind_study_fingerprint": blind_manifest["study_fingerprint"],
        "publication_artifacts_ready": artifact_ready,
        "superiority_claim_supported": superiority_supported,
        "publication_claim_allowed": superiority_supported,
        "result_scope": (
            "publication_ready" if artifact_ready else "exploratory_not_for_publication"
        ),
        "power_plan": power,
        "review_completeness": {
            "expected": expected_reviews,
            "observed": len(rows),
            "missing": missing_reviews,
            "primary_review_evaluable_rate": review_evaluable_rate,
            "primary_paper_evaluable_rate": paper_evaluable_rate,
        },
        "flow": {
            "input_rows": prepared.get("input_rows", 0),
            "frozen_rows": prepared.get("frozen_rows", 0),
            "excluded_rows": prepared.get("excluded_rows", 0),
            "excluded_sample_ids": prepared.get("excluded_sample_ids", []),
            "generation_errors": {
                arm: {
                    condition: values["n"] - values["generation_success"]
                    for condition, values in conditions.items()
                }
                for arm, conditions in automatic_metrics.items()
            },
        },
        "human_review": human,
        "strata": _stratum_summaries(rows, statistics),
        "evidence_kinds": _evidence_summaries(rows, statistics),
        "error_tags": error_tag_metrics(rows),
        "missingness": review_missingness_metrics(rows),
        "automatic_metrics": automatic_metrics,
        "source_artifacts": {
            "prepare_manifest": prepared.get("prepare_manifest")
            or str(output_dir / "prepare_manifest.json"),
            "review_exports": [str(path) for path in review_paths],
            "owner_mapping": str(owner),
        },
    }
    analysis_dir = output_dir / "analysis"
    _write_jsonl(analysis_dir / "deblinded_reviews.jsonl", rows)
    _write_json(analysis_dir / "results.json", results)
    write_paper_scores(analysis_dir / "tables" / "paper_scores.csv", human)
    write_effect_svg(analysis_dir / "figures" / "preference_effects.svg", results)
    with (analysis_dir / "report.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown_report(results, config))
    return {
        "publication_artifacts_ready": artifact_ready,
        "superiority_claim_supported": superiority_supported,
        "publication_claim_allowed": superiority_supported,
        "results": str(analysis_dir / "results.json"),
        "report": str(analysis_dir / "report.md"),
        "figure": str(analysis_dir / "figures" / "preference_effects.svg"),
    }


def command_run(args: argparse.Namespace, config: dict[str, Any], root: Path) -> dict[str, Any]:
    if not args.exploratory:
        raise RuntimeError(
            "strict runs require separate base and tuned processes; use plan/prepare/infer/blind"
        )
    plan = command_plan(args, config, root)
    prepared = command_prepare(args, config, root)
    inference = command_infer(args, config, root)
    blind = command_blind(args, config, root)
    return {"plan": plan, "prepare": prepared, "inference": inference, "blind": blind}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publication-grade paired A/B evaluation for Qwen3-VL SciReason."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="Compute the preregistered power/MDE plan.")

    prepare = subparsers.add_parser("prepare", help="Download, audit, and freeze the benchmark.")
    prepare.add_argument("--benchmark-dir", type=Path)
    prepare.add_argument("--training-file", type=Path, action="append", default=[])
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--exploratory", action="store_true")

    curate_queue = subparsers.add_parser(
        "curate-queue",
        help="Build immutable human-remediation tasks from a failed prepare run.",
    )
    curate_queue.add_argument("--prepare-manifest", type=Path, required=True)
    curate_queue.add_argument("--output-dir", type=Path, required=True)

    curate_forms = subparsers.add_parser(
        "curate-forms",
        help="Build a verified standalone offline curator form workspace.",
    )
    curate_forms.add_argument("--prepare-manifest", type=Path, required=True)
    curate_forms.add_argument("--queue-manifest", type=Path, required=True)
    curate_forms.add_argument("--output-dir", type=Path, required=True)

    curate_assemble = subparsers.add_parser(
        "curate-assemble",
        help="Validate complete curator decisions and stage a corrected release candidate.",
    )
    curate_assemble.add_argument("--prepare-manifest", type=Path, required=True)
    curate_assemble.add_argument("--queue-manifest", type=Path, required=True)
    curate_assemble.add_argument("--decisions-jsonl", type=Path, required=True)
    curate_assemble.add_argument("--curated-dataset-root", type=Path, required=True)
    curate_assemble.add_argument("--output-dir", type=Path, required=True)

    infer = subparsers.add_parser("infer", help="Run or resume model inference.")
    infer.add_argument("--arm", choices=("base", "tuned", "all"), required=True)
    infer.add_argument("--backend", choices=("transformers", "mock"), default="transformers")
    infer.add_argument("--limit", type=int)
    infer.add_argument("--exploratory", action="store_true")

    blind = subparsers.add_parser("blind", help="Create randomized offline reviewer packages.")
    blind.add_argument("--reviewers", nargs="+")
    blind.add_argument("--reviews-per-item", type=int)
    blind.add_argument("--exploratory", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Validate, deblind, and analyze reviews.")
    aggregate.add_argument("--review-file", type=Path, action="append", default=[])
    aggregate.add_argument("--reviews-dir", type=Path)
    aggregate.add_argument("--allow-incomplete", action="store_true")
    aggregate.add_argument("--exploratory", action="store_true")

    run = subparsers.add_parser(
        "run", help="Run plan, prepare, inference, and blind packaging up to the human barrier."
    )
    run.add_argument("--benchmark-dir", type=Path)
    run.add_argument("--training-file", type=Path, action="append", default=[])
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--exploratory", action="store_true")
    run.add_argument("--arm", choices=("all",), default="all")
    run.add_argument("--backend", choices=("transformers", "mock"), default="transformers")
    run.add_argument("--limit", type=int)
    run.add_argument("--reviewers", nargs="+")
    run.add_argument("--reviews-per-item", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_experiment_config(args.config)
        root = _repo_root(args.config, args.repo_root)
        commands = {
            "plan": command_plan,
            "prepare": command_prepare,
            "curate-queue": command_curate_queue,
            "curate-forms": command_curate_forms,
            "curate-assemble": command_curate_assemble,
            "infer": command_infer,
            "blind": command_blind,
            "aggregate": command_aggregate,
            "run": command_run,
        }
        result = commands[args.command](args, config, root)
        print(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (PublicationGateError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
