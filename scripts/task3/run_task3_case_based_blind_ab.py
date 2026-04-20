from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9\-\s_]+", "", (text or "").strip().lower())
    value = re.sub(r"\s+", "-", value).strip("-")
    return value[:80] or "task3"


def _parse_identifiers(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,;\n]", str(text)) if item.strip()]


def _load_yaml_doc(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a top-level object: {path}")
    return payload


def _task_meta_from_yaml(base_doc: dict[str, Any], *, fallback_query: str, domain_id: str, submission_id_override: str = "") -> dict[str, Any]:
    expert = base_doc.get("expert") if isinstance(base_doc.get("expert"), dict) else {}
    full_name = " ".join(
        x for x in [str(expert.get("last_name") or "").strip(), str(expert.get("first_name") or "").strip(), str(expert.get("patronymic") or "-").strip()] if x
    ).strip()
    latin_slug = _slugify(full_name)
    submission_id = str(submission_id_override or base_doc.get("submission_id") or "").strip()
    if not submission_id:
        seed_text = fallback_query or str(base_doc.get("topic") or "task3_case_based")
        short_hash = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12]
        submission_id = f'{latin_slug or "expert"}__{short_hash}'
    topic_value = str(base_doc.get("topic") or fallback_query or "").strip()
    return {
        "topic": topic_value,
        "submission_id": submission_id,
        "cutoff_year": str(base_doc.get("cutoff_year") or "").strip(),
        "domain": str(base_doc.get("domain") or domain_id or "").strip(),
        "expert": {"full_name": full_name, "latin_slug": latin_slug},
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _extract_case_manifest(path: Path, target_dir: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(target_dir)
    candidates = sorted(target_dir.rglob("task3_ab_case_manifest*.json"))
    if not candidates:
        candidates = sorted(target_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError("No JSON case manifest found inside the provided ZIP archive")
    return candidates[0]


def _build_export_zip(out_dir: Path) -> Path | None:
    try:
        export_zip = out_dir.parent / f"{out_dir.name}__outputs.zip"
        with zipfile.ZipFile(export_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(out_dir.parent)))
        return export_zip
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Task 3 case-based blind A/B review from a creator-defined case manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectory", type=Path, default=None, help="Path to Task 1 / expert YAML.")
    parser.add_argument("--case-manifest", type=Path, required=True, help="JSON or ZIP created by the test-set creator.")
    parser.add_argument("--query", default="", help="Fallback query string if YAML topic is empty.")
    parser.add_argument("--identifiers", default="", help="Comma/newline separated identifiers.")
    parser.add_argument("--identifiers-file", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None, help="Curated processed_papers directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/task3_case_based_blind_ab"))
    parser.add_argument("--domain-id", default="science")
    parser.add_argument("--submission-id", default="")
    parser.add_argument("--search-limit", type=int, default=25)
    parser.add_argument("--top-papers", type=int, default=12)
    parser.add_argument("--top-hypotheses", type=int, default=16)
    parser.add_argument("--candidate-top-k", type=int, default=24)
    parser.add_argument("--annoy-n-trees", type=int, default=32)
    parser.add_argument("--annoy-top-k", type=int, default=6)
    parser.add_argument("--include-multimodal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-vlm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-mode", default="auto")
    parser.add_argument("--link-prediction-backend", default="auto")
    parser.add_argument("--hf-offline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--create-expert-bundle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--g4f-model", default=None)
    parser.add_argument("--model-a-owner-label", default="base_local_model")
    parser.add_argument("--model-a-local-text-model", default=None)
    parser.add_argument("--model-a-vlm-backend", default="qwen2_vl")
    parser.add_argument("--model-a-vlm-model-id", default="")
    parser.add_argument("--model-b-owner-label", default="finetuned_local_model")
    parser.add_argument("--model-b-local-text-model", default=None)
    parser.add_argument("--model-b-vlm-backend", default="qwen2_vl")
    parser.add_argument("--model-b-vlm-model-id", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    os.environ.setdefault("G4F_ASYNC_ENABLED", "1")
    os.environ.setdefault("G4F_ASYNC_MAX_CONCURRENCY", "3")
    os.environ.setdefault("LLM_REQUEST_TIMEOUT_SECONDS", "25")
    os.environ.setdefault("VLM_REQUEST_TIMEOUT_SECONDS", "45")
    os.environ.setdefault("LOCAL_VLM_REQUEST_TIMEOUT_SECONDS", "300")
    os.environ.setdefault("LOCAL_VLM_STARTUP_TIMEOUT_SECONDS", "900")
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from scireason.task3_hypothesis_bundle import prepare_task3_hypothesis_bundle
    from scireason.task3_ab_case_manifest import build_task3_case_based_blind_review_package, build_task3_case_based_expert_bundle

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_manifest_path = _extract_case_manifest(args.case_manifest, out_dir / "creator_inputs")
    identifiers = _parse_identifiers(args.identifiers)
    trajectory_doc = _load_yaml_doc(args.trajectory)
    effective_query = str(args.query or "").strip() or str(trajectory_doc.get("topic") or "").strip()
    task_meta = _task_meta_from_yaml(trajectory_doc, fallback_query=effective_query, domain_id=args.domain_id, submission_id_override=args.submission_id)

    model_a_cfg = {
        "owner_label": str(args.model_a_owner_label or "base_local_model").strip() or "base_local_model",
        "vlm_backend": str(args.model_a_vlm_backend or "qwen2_vl").strip() or "qwen2_vl",
        "vlm_model_id": str(args.model_a_vlm_model_id or "").strip(),
        "local_text_model": str(args.model_a_local_text_model or "").strip() or None,
    }
    model_b_cfg = {
        "owner_label": str(args.model_b_owner_label or "finetuned_local_model").strip() or "finetuned_local_model",
        "vlm_backend": str(args.model_b_vlm_backend or "qwen2_vl").strip() or "qwen2_vl",
        "vlm_model_id": str(args.model_b_vlm_model_id or "").strip(),
        "local_text_model": str(args.model_b_local_text_model or "").strip() or None,
    }

    run_vlm_value = bool(args.run_vlm)
    if model_a_cfg["vlm_backend"] == "none" or model_b_cfg["vlm_backend"] == "none":
        run_vlm_value = False
    if run_vlm_value:
        if model_a_cfg["vlm_backend"] != "none" and not model_a_cfg["vlm_model_id"]:
            parser.error("Model α requires --model-a-vlm-model-id when VLM backend is enabled")
        if model_b_cfg["vlm_backend"] != "none" and not model_b_cfg["vlm_model_id"]:
            parser.error("Model β requires --model-b-vlm-model-id when VLM backend is enabled")

    common_kwargs = dict(
        trajectory=args.trajectory,
        query=effective_query,
        identifiers=identifiers,
        identifiers_file=args.identifiers_file,
        domain_id=args.domain_id,
        search_limit=int(args.search_limit),
        top_papers=int(args.top_papers),
        top_hypotheses=int(args.top_hypotheses),
        candidate_top_k=int(args.candidate_top_k),
        include_multimodal=bool(args.include_multimodal),
        run_vlm=run_vlm_value,
        edge_mode=str(args.edge_mode),
        link_prediction_backend=str(args.link_prediction_backend),
        annoy_n_trees=int(args.annoy_n_trees),
        annoy_top_k=int(args.annoy_top_k),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        g4f_model=args.g4f_model,
    )

    try:
        print("[info] Step 1/3 — model α", flush=True)
        bundle_alpha = prepare_task3_hypothesis_bundle(
            **dict(
                common_kwargs,
                processed_dir=args.processed_dir,
                out_dir=out_dir / "variant_alpha",
                local_model=model_a_cfg["local_text_model"],
                vlm_backend=model_a_cfg["vlm_backend"],
                vlm_model_id=model_a_cfg["vlm_model_id"],
            )
        )
        shared_processed_dir = Path(args.processed_dir) if args.processed_dir is not None else Path(bundle_alpha.bundle_dir) / "processed_papers"
        if not shared_processed_dir.exists():
            raise FileNotFoundError("Shared processed_papers directory was not found after model α run")

        print("[info] Step 2/3 — model β", flush=True)
        bundle_beta = prepare_task3_hypothesis_bundle(
            **dict(
                common_kwargs,
                processed_dir=shared_processed_dir,
                processed_dir_link_mode="hardlink",
                processed_dir_strip_mm_vlm_metadata=True,
                out_dir=out_dir / "variant_beta",
                local_model=model_b_cfg["local_text_model"],
                vlm_backend=model_b_cfg["vlm_backend"],
                vlm_model_id=model_b_cfg["vlm_model_id"],
            )
        )

        manifest_alpha = json.loads(Path(bundle_alpha.manifest_path).read_text(encoding="utf-8"))
        manifest_beta = json.loads(Path(bundle_beta.manifest_path).read_text(encoding="utf-8"))

        print("[info] Step 3/3 — case-based blind review package", flush=True)
        review_assets = build_task3_case_based_blind_review_package(
            manifest_alpha,
            manifest_beta,
            case_manifest_path,
            task_meta,
            model_a_descriptor=model_a_cfg,
            model_b_descriptor=model_b_cfg,
        )
        expert_bundle_path = None
        if args.create_expert_bundle:
            expert_bundle_path = build_task3_case_based_expert_bundle(
                manifest_alpha,
                manifest_beta,
                case_manifest_path,
                task_meta,
                model_a_descriptor=model_a_cfg,
                model_b_descriptor=model_b_cfg,
            )

        payload = {
            "generated_at": _utc_now(),
            "repo_root": str(REPO_ROOT),
            "trajectory": str(args.trajectory) if args.trajectory else None,
            "case_manifest": str(case_manifest_path),
            "query": effective_query,
            "identifiers": identifiers,
            "shared_processed_dir": str(shared_processed_dir),
            "task_meta": task_meta,
            "config": {
                "top_papers": int(args.top_papers),
                "top_hypotheses": int(args.top_hypotheses),
                "candidate_top_k": int(args.candidate_top_k),
                "include_multimodal": bool(args.include_multimodal),
                "run_vlm": run_vlm_value,
                "edge_mode": str(args.edge_mode),
                "link_prediction_backend": str(args.link_prediction_backend),
            },
            "model_a": model_a_cfg,
            "model_b": model_b_cfg,
            "artifacts": {
                "run_alpha_bundle": str(bundle_alpha.bundle_dir),
                "run_beta_bundle": str(bundle_beta.bundle_dir),
                "manifest_alpha": str(bundle_alpha.manifest_path),
                "manifest_beta": str(bundle_beta.manifest_path),
                "offline_form": str(review_assets.offline_html_path),
                "owner_key": str(review_assets.owner_mapping_path),
                "public_manifest": str(review_assets.public_manifest_path),
                "expert_bundle": str(expert_bundle_path) if expert_bundle_path else None,
            },
        }
        manifest_path = _write_json(out_dir / "task3_case_based_run_manifest.json", payload)
        export_zip = _build_export_zip(out_dir)
        if export_zip is not None:
            payload["artifacts"]["export_zip"] = str(export_zip)
            _write_json(manifest_path, payload)

        print(
            json.dumps(
                {
                    "run_manifest": str(manifest_path),
                    "offline_form": str(review_assets.offline_html_path),
                    "owner_key": str(review_assets.owner_mapping_path),
                    "expert_bundle": str(expert_bundle_path) if expert_bundle_path else None,
                    "export_zip": str(export_zip) if export_zip else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        error_payload = {
            "generated_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(out_dir / "task3_case_based_run_error.json", error_payload)
        print(json.dumps(error_payload, ensure_ascii=False, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
