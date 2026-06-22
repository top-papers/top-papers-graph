#!/usr/bin/env python3
"""Prepare and upload fine-tuned VLM artifacts to a Hugging Face Hub model repo."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
}


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def sizeof(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, exclude_roots: Iterable[Path] = ()) -> None:
    """Copy a tree while avoiding recursive copies into the bundle itself."""
    if not src.exists():
        return
    src = src.resolve()
    dst = dst.resolve()
    excluded = [p.resolve() for p in exclude_roots]
    for path in src.rglob("*"):
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if any(resolved == root or root in resolved.parents for root in excluded):
            continue
        if any(part in DEFAULT_EXCLUDE_DIRS for part in path.parts):
            continue
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def copy_grpo_root_files(grpo_dir: Path, bundle_dir: Path) -> list[str]:
    """Expose final adapter/processor files at repo root for easier loading.

    The historical name is kept for backward compatibility with the legacy
    SFT->GRPO uploader.  The input directory may now be a DPO or GRPO adapter.
    """
    copied: list[str] = []
    if not grpo_dir.exists():
        return copied
    for child in sorted(grpo_dir.iterdir()):
        if child.is_file():
            target = bundle_dir / child.name
            shutil.copy2(child, target)
            copied.append(child.name)
    return copied


def adapter_has_root_files(path: Path | None) -> bool:
    if path is None or not path.exists() or not path.is_dir():
        return False
    root_file_names = {child.name for child in path.iterdir() if child.is_file()}
    return bool(root_file_names & {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"})


def resolve_final_adapter_dir(*, sft_dir: Path, dpo_dir: Path | None, grpo_dir: Path, final_dir: Path | None, final_stage: str) -> tuple[Path, str]:
    """Select the adapter exposed at the Hub repo root.

    In the DPO-first pipeline GRPO is optional.  When it is skipped, the DPO
    adapter is the main candidate and must be exposed at repo root instead of
    forcing the upload script to fail because the GRPO directory only contains
    planned_run_config.json.
    """
    if final_dir is not None:
        return final_dir, final_stage if final_stage != "auto" else "custom"
    candidates: list[tuple[str, Path | None]]
    if final_stage == "grpo":
        candidates = [("grpo", grpo_dir)]
    elif final_stage == "dpo":
        candidates = [("dpo", dpo_dir)]
    elif final_stage == "sft":
        candidates = [("sft", sft_dir)]
    else:
        candidates = [("grpo", grpo_dir), ("dpo", dpo_dir), ("sft", sft_dir)]
    for stage, path in candidates:
        if adapter_has_root_files(path):
            return path, stage
    checked = {stage: str(path) for stage, path in candidates}
    raise SystemExit(f"Could not find a final adapter directory with adapter root files. Checked: {checked}")


def write_model_card(
    path: Path,
    *,
    repo_id: str,
    base_model: str,
    dataset_id: str,
    out_prefix: str,
    sft_dir: Path,
    dpo_dir: Path | None,
    grpo_dir: Path,
    final_dir: Path,
    final_stage: str,
    generated_at: str,
) -> None:
    text = f"""---
base_model: {base_model}
library_name: peft
tags:
- qwen3-vl
- vision-language
- lora
- sft
- dpo
- grpo
- scireason
datasets:
- {dataset_id}
---

# {repo_id}

This repository contains the fine-tuned SciReason VLM artifacts produced by the
DataSphere SciReason alignment pipeline.  The recommended production route is
SFT -> DPO, with optional short GRPO polish when reward-audit gates pass.

## Contents

- Root files: final `{final_stage}` adapter and processor files copied from `{final_dir}` for convenient loading.
- `artifacts/sft_lora/`: SFT LoRA adapter directory copied from `{sft_dir}`.
- `artifacts/dpo_lora/`: DPO LoRA adapter directory copied from `{dpo_dir}` when available.
- `artifacts/grpo_lora/`: GRPO output directory copied from `{grpo_dir}` when available.
- `artifacts/archives/`: compressed `.tar.gz` archives produced by the job.
- `artifacts/data/`: generated train/eval JSONL files and dataset summary.
- `artifacts/reports/`: budget, final summary, upload manifest and runtime reports.

## Training metadata

- Base model: `{base_model}`
- Dataset: `{dataset_id}`
- Output prefix: `{out_prefix}`
- Uploaded at UTC: `{generated_at}`

## Loading note

The root of this repository is prepared as the final `{final_stage}` adapter
directory. For LoRA/PEFT loading, use the same base model listed above and load
this repository as the adapter. The complete SFT, DPO and GRPO directories are
preserved under `artifacts/` when present for auditability and reproducibility.
"""
    path.write_text(text, encoding="utf-8")


def build_manifest(bundle_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    files = []
    total_size = 0
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(bundle_dir).as_posix()
            size = path.stat().st_size
            total_size += size
            files.append({"path": rel, "size_bytes": size})
    payload = {**payload, "total_size_bytes": total_size, "file_count": len(files), "files": files}
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload fine-tuned VLM model and artifacts to Hugging Face Hub.")
    ap.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", "top-papers/Qwen3-VL-8B-Instruct-scireason"))
    ap.add_argument("--repo-type", default=os.environ.get("HF_REPO_TYPE", "model"))
    ap.add_argument("--revision", default=os.environ.get("HF_REVISION", "main"))
    ap.add_argument("--path-in-repo", default=os.environ.get("HF_UPLOAD_PATH_PREFIX", ""), help="Optional subdirectory in the Hub repo. Empty means repo root.")
    ap.add_argument("--commit-message", default=os.environ.get("HF_UPLOAD_COMMIT_MESSAGE", "Upload fine-tuned SciReason VLM artifacts"))
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    ap.add_argument("--private", action="store_true", default=parse_bool(os.environ.get("HF_REPO_PRIVATE"), False))
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--sft-dir", type=Path, required=True)
    ap.add_argument("--dpo-dir", type=Path, default=None, help="Optional DPO adapter/output directory for DPO-first pipelines.")
    ap.add_argument("--grpo-dir", type=Path, required=True)
    ap.add_argument("--final-dir", type=Path, default=None, help="Optional explicit final adapter directory to expose at repo root.")
    ap.add_argument("--final-stage", choices=["auto", "sft", "dpo", "grpo"], default="auto")
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.token:
        raise SystemExit(
            "HF upload is enabled, but HF_TOKEN/HUGGING_FACE_HUB_TOKEN is not set. "
            "Create a Hugging Face token with write access to the target repo/org and expose it as HF_TOKEN in the DataSphere job environment."
        )
    if args.repo_type != "model":
        raise SystemExit("This uploader is intended for Hugging Face model repositories. Use --repo-type model.")
    final_dir, resolved_final_stage = resolve_final_adapter_dir(
        sft_dir=args.sft_dir,
        dpo_dir=args.dpo_dir,
        grpo_dir=args.grpo_dir,
        final_dir=args.final_dir,
        final_stage=args.final_stage,
    )

    bundle_dir = args.bundle_dir.resolve()
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    root_files = copy_grpo_root_files(final_dir, bundle_dir)
    copy_tree(args.sft_dir, bundle_dir / "artifacts" / "sft_lora", exclude_roots=[bundle_dir])
    if args.dpo_dir is not None and args.dpo_dir.exists():
        copy_tree(args.dpo_dir, bundle_dir / "artifacts" / "dpo_lora", exclude_roots=[bundle_dir])
    if args.grpo_dir.exists():
        copy_tree(args.grpo_dir, bundle_dir / "artifacts" / "grpo_lora", exclude_roots=[bundle_dir])

    for name in [
        "summary.json",
        "README.md",
        "export_summary.json",
        "ARTICLE_IMAGE_SOURCES.md",
        "article_image_sources.jsonl",
        "sft_all.jsonl",
        "sft_train.jsonl",
        "sft_eval.jsonl",
        "sft_text_train.jsonl",
        "sft_text_eval.jsonl",
        "sft_vlm_train.jsonl",
        "sft_vlm_eval.jsonl",
        "dpo_all.jsonl",
        "dpo_train.jsonl",
        "dpo_eval.jsonl",
        "grpo_all.jsonl",
        "grpo_train.jsonl",
        "grpo_eval.jsonl",
        "grpo_train_all.jsonl",
        "grpo_eval_all.jsonl",
        "grpo_all_verified.jsonl",
        "grpo_train_verified.jsonl",
        "grpo_eval_verified.jsonl",
    ]:
        copy_file(args.data_dir / name, bundle_dir / "artifacts" / "data" / name)

    for archive in [
        Path("outputs") / f"{args.out_prefix}_text_sft_lora.tar.gz",
        Path("outputs") / f"{args.out_prefix}_vlm_sft_lora.tar.gz",
        Path("outputs") / f"{args.out_prefix}_sft_lora.tar.gz",
        Path("outputs") / f"{args.out_prefix}_dpo_lora.tar.gz",
        Path("outputs") / f"{args.out_prefix}_grpo_lora.tar.gz",
        Path("reports") / f"{args.out_prefix}_datasphere_reports.tar.gz",
    ]:
        copy_file(archive, bundle_dir / "artifacts" / "archives" / archive.name)

    copy_tree(args.report_dir, bundle_dir / "artifacts" / "reports", exclude_roots=[bundle_dir])

    (bundle_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
        "*.tar.gz filter=lfs diff=lfs merge=lfs -text\n"
        "*.jsonl filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    write_model_card(
        bundle_dir / "README.md",
        repo_id=args.repo_id,
        base_model=args.base_model,
        dataset_id=args.dataset_id,
        out_prefix=args.out_prefix,
        sft_dir=args.sft_dir,
        dpo_dir=args.dpo_dir,
        grpo_dir=args.grpo_dir,
        final_dir=final_dir,
        final_stage=resolved_final_stage,
        generated_at=generated_at,
    )

    upload_manifest_payload = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "path_in_repo": args.path_in_repo or ".",
        "base_model": args.base_model,
        "dataset_id": args.dataset_id,
        "out_prefix": args.out_prefix,
        "generated_at_utc": generated_at,
        "bundle_dir": str(bundle_dir),
        "final_stage": resolved_final_stage,
        "final_dir": str(final_dir),
        "root_files": root_files,
        "source_sizes_bytes": {
            "sft_dir": sizeof(args.sft_dir),
            "dpo_dir": sizeof(args.dpo_dir) if args.dpo_dir is not None else 0,
            "grpo_dir": sizeof(args.grpo_dir),
            "data_dir": sizeof(args.data_dir),
            "report_dir": sizeof(args.report_dir),
        },
    }
    manifest = build_manifest(bundle_dir, upload_manifest_payload)
    (bundle_dir / "artifacts" / "reports").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "artifacts" / "reports" / "hf_upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Rebuild so the manifest includes itself.
    manifest = build_manifest(bundle_dir, upload_manifest_payload)
    (bundle_dir / "artifacts" / "reports" / "hf_upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    upload_kwargs: dict[str, Any] = {
        "folder_path": str(bundle_dir),
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "commit_message": args.commit_message,
    }
    if args.path_in_repo:
        upload_kwargs["path_in_repo"] = args.path_in_repo.strip("/")
    commit_info = api.upload_folder(**upload_kwargs)

    summary = {
        "status": "uploaded",
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "path_in_repo": args.path_in_repo or ".",
        "commit_message": args.commit_message,
        "commit_info": str(commit_info),
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(bundle_dir / "artifacts" / "reports" / "hf_upload_manifest.json"),
        "uploaded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_count": manifest["file_count"],
        "total_size_bytes": manifest["total_size_bytes"],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "hf_upload_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
