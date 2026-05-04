from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence

import yaml

from .asset_index import AssetIndex, AssetRecord, clone_image, parse_locator_number
from .discovery import discover_mixed_inputs, discover_task1_files, discover_task2_inputs
from .download import collect_refs_from_task1_files, collect_refs_from_task2_inputs, download_and_ingest_refs
from .hf_hub import upload_export_to_hf
from .vendor.common.datacls import (
    AssistantMessage,
    Chat,
    ExpertSignals,
    GRPOMetadata,
    GRPOSample,
    ImageContent,
    SFTMetadata,
    SFTSample,
    SystemMessage,
    TextContent,
    UserMessage,
)
from .vendor.common.utils.io import write_json, write_jsonl
from .vendor.common.utils.paper_ids import resolve
from .vendor.common.utils.prompts import (
    GRPO_SYSTEM_PROMPT,
    SFT_SYSTEM_PROMPT,
    build_gold_assertion_sft_prompt,
    build_grpo_user_prompt,
    build_sft_user_prompt,
)
from .vendor.normalize_task1.normalizer import normalize_file as normalize_task1_file
from .vendor.normalize_task2.normalizer import normalize_bundle as normalize_task2_bundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportResult:
    output_root: Path
    normalized_task1_dir: Path
    normalized_task2_dir: Path
    sft_path: Path
    grpo_path: Path
    summary_path: Path
    stats: dict[str, Any]
    hf_repo_url: str | None = None


def export_dataset(
    *,
    task1_files: Sequence[Path] = (),
    task1_dirs: Sequence[Path] = (),
    task2_inputs: Sequence[Path] = (),
    task2_dirs: Sequence[Path] = (),
    input_dirs: Sequence[Path] = (),
    out_dir: Path,
    processed_papers_dirs: Sequence[Path] = (),
    copy_assets: bool = True,
    max_images_per_sample: int = 8,
    max_multimodal_records_per_sample: int = 0,
    discover_recursive: bool = True,
    download_referenced_papers: bool = False,
    download_unpaywall_email: str | None = None,
    download_root: Path | None = None,
    download_processed_papers_dir: Path | None = None,
    ingest_downloaded_papers: bool = True,
    download_multimodal: bool = True,
    download_run_vlm: bool = True,
    prefer_cached_downloads: bool = True,
    hf_upload: bool = False,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    hf_private: bool | None = None,
    hf_path_in_repo: str | None = None,
    hf_commit_message: str | None = None,
    hf_commit_description: str | None = None,
    hf_create_repo_if_missing: bool = True,
    hf_generate_readme: bool = True,
) -> ExportResult:
    out_dir = out_dir.resolve()
    norm_task1_dir = out_dir / "normalized_task1"
    norm_task2_dir = out_dir / "normalized_task2"
    norm_task1_dir.mkdir(parents=True, exist_ok=True)
    norm_task2_dir.mkdir(parents=True, exist_ok=True)

    mixed_task1, mixed_task2 = discover_mixed_inputs(input_dirs, recursive=discover_recursive)
    task1_paths = discover_task1_files([*task1_files, *mixed_task1], task1_dirs, recursive=discover_recursive)
    task2_paths = discover_task2_inputs([*task2_inputs, *mixed_task2], task2_dirs, recursive=discover_recursive)

    extracted_roots: list[Path] = []
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    download_stats: dict[str, Any] = {}
    hf_repo_url: str | None = None
    try:
        for path in task1_paths:
            normalize_task1_file(Path(path), norm_task1_dir, wiki_cache={})

        for path in task2_paths:
            bundle_refs, temp_obj = _materialize_task2_input(Path(path))
            if temp_obj is not None:
                temp_dirs.append(temp_obj)
            for bundle_path, source_path, source_identity in bundle_refs:
                extracted_roots.append(bundle_path)
                normalize_task2_bundle(
                    bundle_path,
                    norm_task2_dir,
                    source_path=source_path,
                    source_identity=source_identity,
                )

        processed_roots = [Path(p).resolve() for p in processed_papers_dirs]
        scan_roots = list(extracted_roots)
        if download_referenced_papers:
            refs = []
            refs.extend(collect_refs_from_task1_files(task1_paths))
            refs.extend(collect_refs_from_task2_inputs(extracted_roots))
            download_summary = download_and_ingest_refs(
                refs=refs,
                download_root=(download_root or (out_dir / "downloads")).resolve(),
                processed_papers_dir=(download_processed_papers_dir or (out_dir / "downloaded_processed_papers")).resolve() if ingest_downloaded_papers else None,
                existing_processed_papers_dirs=processed_roots,
                unpaywall_email=download_unpaywall_email,
                ingest_downloaded=ingest_downloaded_papers,
                multimodal=download_multimodal,
                run_vlm=download_run_vlm,
                prefer_cached=prefer_cached_downloads,
            )
            if download_summary.produced_processed_papers_dir is not None:
                processed_roots.append(download_summary.produced_processed_papers_dir)
            scan_roots.append(download_summary.download_root)
            download_stats = {
                "download_refs_total": download_summary.refs_total,
                "download_refs_supported": download_summary.refs_supported,
                "download_pdf_downloaded": download_summary.pdf_downloaded,
                "download_html_downloaded": download_summary.html_downloaded,
                "download_ingested_processed_papers": download_summary.ingested_processed_papers,
                "download_skipped_existing": download_summary.skipped_existing,
                "download_errors": download_summary.errors,
                "download_root": str(download_summary.download_root),
                "download_processed_papers_dir": str(download_summary.produced_processed_papers_dir) if download_summary.produced_processed_papers_dir else "",
            }

        asset_index = AssetIndex()
        asset_index.scan_processed_papers(processed_roots)
        asset_index.scan_bundle_inputs(scan_roots)

        sft_rows, sft_stats = _build_sft(
            norm_task1_dir=norm_task1_dir,
            norm_task2_dir=norm_task2_dir,
            asset_index=asset_index,
            assets_root=out_dir / "assets",
            copy_assets=copy_assets,
            max_images_per_sample=max_images_per_sample,
            max_multimodal_records_per_sample=max_multimodal_records_per_sample,
        )
        grpo_rows, grpo_stats = _build_grpo(
            norm_task2_dir=norm_task2_dir,
            asset_index=asset_index,
            assets_root=out_dir / "assets",
            copy_assets=copy_assets,
            max_images_per_sample=max_images_per_sample,
            max_multimodal_records_per_sample=max_multimodal_records_per_sample,
        )
        task1_retention_stats = _task1_source_retention_stats(norm_task1_dir, sft_rows)

        sft_path = out_dir / "sft.jsonl"
        grpo_path = out_dir / "grpo.jsonl"
        write_jsonl(sft_path, sft_rows)
        write_jsonl(grpo_path, grpo_rows)

        stats: dict[str, Any] = {
            **sft_stats,
            **grpo_stats,
            **download_stats,
            **task1_retention_stats,
            "normalized_task1_submissions": sum(1 for p in norm_task1_dir.iterdir() if p.is_dir()),
            "normalized_task2_bundles": sum(1 for p in norm_task2_dir.iterdir() if p.is_dir()),
            "sft_rows": len(sft_rows),
            "grpo_rows": len(grpo_rows),
            "sft_rows_with_images": sum(1 for row in sft_rows if row.get("images")),
            "grpo_rows_with_images": sum(1 for row in grpo_rows if row.get("images")),
            "sft_image_refs": sum(len(row.get("images") or []) for row in sft_rows),
            "grpo_image_refs": sum(len(row.get("images") or []) for row in grpo_rows),
            "processed_papers_roots": [str(p) for p in processed_roots],
            "task1_inputs": [str(Path(p)) for p in task1_paths],
            "task2_inputs": [str(Path(p)) for p in task2_paths],
            "task1_dirs": [str(Path(p)) for p in task1_dirs],
            "task2_dirs": [str(Path(p)) for p in task2_dirs],
            "input_dirs": [str(Path(p)) for p in input_dirs],
            "discovered_task1_files": len(task1_paths),
            "discovered_task2_inputs": len(task2_paths),
        }
        if hf_upload:
            if not hf_repo_id:
                raise ValueError("hf_repo_id is required when hf_upload=True")
            hf_result = upload_export_to_hf(
                out_dir,
                repo_id=hf_repo_id,
                token=hf_token,
                private=hf_private,
                path_in_repo=hf_path_in_repo,
                commit_message=hf_commit_message,
                commit_description=hf_commit_description,
                create_repo_if_missing=hf_create_repo_if_missing,
                generate_readme=hf_generate_readme,
                stats=stats,
            )
            hf_repo_url = hf_result.repo_url
            stats.update({
                "hf_uploaded": True,
                "hf_repo_id": hf_result.repo_id,
                "hf_repo_url": hf_result.repo_url,
                "hf_path_in_repo": hf_result.path_in_repo,
                "hf_commit_message": hf_result.commit_message,
                "hf_created_repo": hf_result.created_repo,
                "hf_private": hf_result.private,
            })
        else:
            stats.update({"hf_uploaded": False})
        summary_path = out_dir / "export_summary.json"
        write_json(summary_path, stats)
        return ExportResult(
            output_root=out_dir,
            normalized_task1_dir=norm_task1_dir,
            normalized_task2_dir=norm_task2_dir,
            sft_path=sft_path,
            grpo_path=grpo_path,
            summary_path=summary_path,
            stats=stats,
            hf_repo_url=hf_repo_url,
        )
    finally:
        for tmp in temp_dirs:
            tmp.cleanup()


def _materialize_task2_input(path: Path) -> tuple[list[tuple[Path, Path, str]], Optional[tempfile.TemporaryDirectory[str]]]:
    """Return every Task 2 bundle root represented by a path.

    Some form uploads are ZIPs that contain several bundle directories. The
    previous implementation picked only the first detected bundle root, which
    made the remaining valid input files disappear from normalized_task2 and
    downstream SFT/GRPO datasets.
    """
    if path.is_dir():
        return [(bundle, bundle, str(bundle.resolve())) for bundle in _find_bundle_roots(path)], None
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported Task 2 input: {path}")
    tmp = tempfile.TemporaryDirectory(prefix="task2_bundle_")
    root = Path(tmp.name)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(root)
    refs: list[tuple[Path, Path, str]] = []
    for bundle in _find_bundle_roots(root):
        try:
            rel = bundle.relative_to(root).as_posix()
        except ValueError:
            rel = bundle.name
        refs.append((bundle, path, f"{path.resolve()}!{rel}"))
    return refs, tmp


def _is_task2_bundle_root(path: Path) -> bool:
    return any((path / name).exists() for name in ("edge_reviews.json", "review_templates", "task2_notebook_manifest.json"))


def _find_bundle_roots(root: Path) -> list[Path]:
    if _is_task2_bundle_root(root):
        return [root]

    bundle_dirs: list[Path] = []
    seen: set[str] = set()
    for candidate in sorted(p for p in root.rglob("*") if p.is_dir() and p.name != "__MACOSX"):
        if not _is_task2_bundle_root(candidate):
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        # If a parent is already a bundle root, do not add nested implementation
        # folders as separate bundles.
        if any(str(candidate.resolve()).startswith(str(parent.resolve()) + "/") for parent in bundle_dirs):
            continue
        seen.add(key)
        bundle_dirs.append(candidate)

    if bundle_dirs:
        return bundle_dirs

    children = [p for p in root.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return [children[0]]
    return [root]



def _portable_image_ref(image_path: str, export_root: Path) -> str:
    """Return an image reference that survives uploading/downloading the export folder."""
    raw = str(image_path or "").strip()
    if not raw:
        return ""
    if re.match(r"https?://", raw):
        return raw
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(export_root.resolve()).as_posix()
        except Exception:
            return candidate.as_posix()
    return candidate.as_posix()


def _canonical_content_block(block: dict[str, Any], *, images: list[str], export_root: Path) -> dict[str, Any] | None:
    kind = str(block.get("type") or "text").strip().lower()
    if kind == "image":
        image_ref = _portable_image_ref(str(block.get("image") or ""), export_root)
        if image_ref and image_ref not in images:
            images.append(image_ref)
        # TRL expects image placeholders in message content; the actual paths live
        # in the top-level `images` column.
        return {"type": "image"}
    if kind == "text":
        value = str(block.get("text") or "")
        if not value:
            return None
        return {"type": "text", "text": value}
    return None


def _canonical_messages(messages: Sequence[dict[str, Any]], *, export_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    canonical: list[dict[str, Any]] = []
    images: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict):
                    converted = _canonical_content_block(block, images=images, export_root=export_root)
                    if converted is not None:
                        blocks.append(converted)
                elif isinstance(block, str) and block.strip():
                    blocks.append({"type": "text", "text": block})
            canonical.append({"role": role, "content": blocks})
        elif isinstance(content, str):
            canonical.append({"role": role, "content": [{"type": "text", "text": content}]})
    return canonical, images


def _add_trl_sft_fields(row: dict[str, Any], *, export_root: Path) -> dict[str, Any]:
    """Add TRL-compatible `messages` and `images` while keeping legacy `chat`."""
    chat = row.get("chat") if isinstance(row.get("chat"), dict) else {}
    legacy_messages = chat.get("messages") if isinstance(chat.get("messages"), list) else []
    messages, images = _canonical_messages(legacy_messages, export_root=export_root)
    if messages:
        row["messages"] = messages
    # Keep `images` on every row: TRL/HF datasets support mixed VLM + text rows
    # when image rows contain paths/PIL objects and text-only rows contain [] .
    row["images"] = images
    if images:
        extra = row.setdefault("metadata", {}).setdefault("extra", {})
        extra["image_paths"] = images
        extra["image_count"] = len(images)
    return row


def _add_trl_grpo_fields(row: dict[str, Any], *, export_root: Path) -> dict[str, Any]:
    """Add TRL-compatible `prompt` and `images` while keeping legacy `prompt_chat`."""
    prompt_chat = row.get("prompt_chat") if isinstance(row.get("prompt_chat"), dict) else {}
    legacy_messages = prompt_chat.get("messages") if isinstance(prompt_chat.get("messages"), list) else []
    prompt, images = _canonical_messages(legacy_messages, export_root=export_root)
    if prompt:
        row["prompt"] = prompt
    row["images"] = images
    if images:
        extra = row.setdefault("metadata", {}).setdefault("extra", {})
        extra["image_paths"] = images
        extra["image_count"] = len(images)
    return row

def _read_source_marker(sub_dir: Path) -> str:
    marker = sub_dir / ".source_path"
    try:
        return marker.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _task1_source_retention_stats(norm_task1_dir: Path, sft_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows_by_submission: dict[str, int] = {}
    for row in sft_rows:
        if row.get("task_family") != "trajectory_reasoning":
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        submission_id = str(metadata.get("submission_id") or "")
        if submission_id:
            rows_by_submission[submission_id] = rows_by_submission.get(submission_id, 0) + 1

    normalized_sources: list[str] = []
    sources_without_rows: list[str] = []
    for sub_dir in sorted(p for p in norm_task1_dir.iterdir() if p.is_dir()):
        source = _read_source_marker(sub_dir) or str(sub_dir)
        normalized_sources.append(source)
        if rows_by_submission.get(sub_dir.name, 0) <= 0:
            sources_without_rows.append(source)

    return {
        "normalized_task1_source_files": normalized_sources,
        "normalized_task1_source_file_count": len(normalized_sources),
        "task1_sources_with_sft_rows": len(normalized_sources) - len(sources_without_rows),
        "task1_sources_without_sft_rows": sources_without_rows,
    }


def _build_sft(
    *,
    norm_task1_dir: Path,
    norm_task2_dir: Path,
    asset_index: AssetIndex,
    assets_root: Path,
    copy_assets: bool,
    max_images_per_sample: int,
    max_multimodal_records_per_sample: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {"trajectory_reasoning": 0, "assertion_reconstruction": 0}
    for sub_dir in sorted(norm_task1_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        yaml_candidates = [p for p in sub_dir.glob("*.yaml") if p.stem == sub_dir.name] or list(sub_dir.glob("*.yaml"))
        if not yaml_candidates:
            continue
        samples = _trajectory_samples(
            yaml_candidates[0],
            asset_index=asset_index,
            assets_root=assets_root / sub_dir.name,
            copy_assets=copy_assets,
            max_images_per_sample=max_images_per_sample,
            max_multimodal_records_per_sample=max_multimodal_records_per_sample,
        )
        dump = [_add_trl_sft_fields(sample.model_dump(exclude_none=True), export_root=assets_root.parent) for sample in samples]
        write_jsonl(sub_dir / "sft.jsonl", dump)
        rows.extend(dump)
        stats["trajectory_reasoning"] += len(samples)

    for sub_dir in sorted(norm_task2_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        gold_path = sub_dir / "gold.json"
        if not gold_path.exists():
            continue
        samples = _gold_assertion_samples(
            gold_path,
            asset_index=asset_index,
            assets_root=assets_root / sub_dir.name,
            copy_assets=copy_assets,
            max_images_per_sample=max_images_per_sample,
            max_multimodal_records_per_sample=max_multimodal_records_per_sample,
        )
        dump = [_add_trl_sft_fields(sample.model_dump(exclude_none=True), export_root=assets_root.parent) for sample in samples]
        write_jsonl(sub_dir / "sft.jsonl", dump)
        rows.extend(dump)
        stats["assertion_reconstruction"] += len(samples)
    return rows, stats


def _build_grpo(
    *,
    norm_task2_dir: Path,
    asset_index: AssetIndex,
    assets_root: Path,
    copy_assets: bool,
    max_images_per_sample: int,
    max_multimodal_records_per_sample: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {"assertion_review_rl": 0}
    for sub_dir in sorted(norm_task2_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        auto_path = sub_dir / "auto.json"
        if not auto_path.exists():
            continue
        samples = _grpo_samples(
            auto_path,
            asset_index=asset_index,
            assets_root=assets_root / sub_dir.name,
            copy_assets=copy_assets,
            max_images_per_sample=max_images_per_sample,
            max_multimodal_records_per_sample=max_multimodal_records_per_sample,
        )
        dump = [_add_trl_grpo_fields(sample.model_dump(exclude_none=True), export_root=assets_root.parent) for sample in samples]
        write_jsonl(sub_dir / "grpo.jsonl", dump)
        rows.extend(dump)
        stats["assertion_review_rl"] += len(samples)
    return rows, stats


def _expert_key(expert: dict[str, Any] | None) -> str:
    expert = expert or {}
    return str(expert.get("latin_slug") or expert.get("full_name") or "").strip()


def _trajectory_samples(
    yaml_path: Path,
    *,
    asset_index: AssetIndex,
    assets_root: Path,
    copy_assets: bool,
    max_images_per_sample: int,
    max_multimodal_records_per_sample: int,
) -> list[SFTSample]:
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        return []
    expert_key = _expert_key(doc.get("expert"))
    submission_id = str(doc.get("submission_id") or yaml_path.stem)
    topic = str(doc.get("topic") or "")
    domain = str(doc.get("domain") or "")
    steps = doc.get("steps") if isinstance(doc.get("steps"), list) else []

    samples: list[SFTSample] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        previous = [s for s in steps[:idx] if isinstance(s, dict)]
        prompt_text = build_sft_user_prompt(doc, step, previous)
        refs = _refs_for_step(doc, step)
        selected_records, all_records = _select_records_for_step(step, refs, asset_index)
        user_message = _user_message(
            prompt_text=prompt_text,
            selected_records=selected_records,
            all_records=all_records,
            assets_out_dir=assets_root / f"step_{step.get('step_id')}",
            copy_assets=copy_assets,
            max_images_per_sample=max_images_per_sample,
            max_multimodal_records_per_sample=max_multimodal_records_per_sample,
        )
        assistant_text = json.dumps(
            {
                "inference": step.get("inference", ""),
                "next_question": step.get("next_question", ""),
            },
            ensure_ascii=False,
        )
        chat = Chat(
            messages=[
                SystemMessage(content=[TextContent(text=SFT_SYSTEM_PROMPT)]),
                user_message,
                AssistantMessage(content=[TextContent(text=assistant_text, trainable=True)]),
            ]
        )
        samples.append(
            SFTSample(
                id=f"trajectory:{submission_id}:{step.get('step_id')}",
                task_family="trajectory_reasoning",
                domain=domain,
                topic=topic,
                expert_key=expert_key,
                source_file=str(yaml_path),
                chat=chat,
                metadata=SFTMetadata(
                    submission_id=submission_id,
                    step_id=step.get("step_id"),
                    assertion_id=f"{submission_id}:step{step.get('step_id')}",
                    cutoff_year=doc.get("cutoff_year"),
                    importance=step.get("importance"),
                    start_date=step.get("start_date"),
                    end_date=step.get("end_date"),
                    time_source=step.get("time_source"),
                    extra={
                        "multimodal_selected": len(selected_records),
                        "multimodal_available": len(all_records),
                    },
                ),
            )
        )
    return samples


def _gold_assertion_samples(
    bundle_path: Path,
    *,
    asset_index: AssetIndex,
    assets_root: Path,
    copy_assets: bool,
    max_images_per_sample: int,
    max_multimodal_records_per_sample: int,
) -> list[SFTSample]:
    doc = json.loads(bundle_path.read_text(encoding="utf-8"))
    assertions = doc.get("assertions") if isinstance(doc.get("assertions"), list) else []
    submission_id = str(doc.get("submission_id") or bundle_path.parent.name)
    topic = str(doc.get("topic") or "")
    domain = str(doc.get("domain") or "")

    samples: list[SFTSample] = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        prompt_text = build_gold_assertion_sft_prompt(doc, assertion)
        refs = _refs_for_assertion(assertion)
        selected_records, all_records = _select_records_for_assertion(assertion, refs, asset_index)
        user_message = _user_message(
            prompt_text=prompt_text,
            selected_records=selected_records,
            all_records=all_records,
            assets_out_dir=assets_root / f"gold_{assertion.get('assertion_id')}",
            copy_assets=copy_assets,
            max_images_per_sample=max_images_per_sample,
            max_multimodal_records_per_sample=max_multimodal_records_per_sample,
        )
        assistant_text = json.dumps(
            {
                "subject": assertion.get("subject", ""),
                "predicate": assertion.get("predicate", ""),
                "object": assertion.get("object", ""),
                "start_date": assertion.get("start_date"),
                "end_date": assertion.get("end_date"),
            },
            ensure_ascii=False,
        )
        chat = Chat(
            messages=[
                SystemMessage(content=[TextContent(text=SFT_SYSTEM_PROMPT)]),
                user_message,
                AssistantMessage(content=[TextContent(text=assistant_text, trainable=True)]),
            ]
        )
        samples.append(
            SFTSample(
                id=f"assertion_reconstruction:{submission_id}:{assertion.get('assertion_id')}",
                task_family="assertion_reconstruction",
                domain=domain,
                topic=topic,
                expert_key=str(doc.get("reviewer_id") or ""),
                source_file=str(bundle_path),
                chat=chat,
                metadata=SFTMetadata(
                    submission_id=submission_id,
                    assertion_id=assertion.get("assertion_id"),
                    graph_kind=assertion.get("graph_kind"),
                    importance_score=assertion.get("importance_score"),
                    start_date=assertion.get("start_date"),
                    end_date=assertion.get("end_date"),
                    extra={
                        "multimodal_selected": len(selected_records),
                        "multimodal_available": len(all_records),
                    },
                ),
            )
        )
    return samples


def _grpo_samples(
    auto_path: Path,
    *,
    asset_index: AssetIndex,
    assets_root: Path,
    copy_assets: bool,
    max_images_per_sample: int,
    max_multimodal_records_per_sample: int,
) -> list[GRPOSample]:
    doc = json.loads(auto_path.read_text(encoding="utf-8"))
    assertions = doc.get("assertions") if isinstance(doc.get("assertions"), list) else []
    submission_id = str(doc.get("submission_id") or auto_path.parent.name)
    topic = str(doc.get("topic") or "")
    domain = str(doc.get("domain") or "")
    reviewer_id = str(doc.get("reviewer_id") or "")
    accepted = {"accepted", "rejected", "needs_time_fix", "needs_evidence_fix", "uncertain", "added", "confirmed", "modified"}

    samples: list[GRPOSample] = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        expert = assertion.get("expert") if isinstance(assertion.get("expert"), dict) else {}
        verdict = str(expert.get("verdict") or "").strip().lower()
        if not verdict or verdict not in accepted:
            continue
        prompt_text = build_grpo_user_prompt(doc, assertion)
        refs = _refs_for_assertion(assertion)
        selected_records, all_records = _select_records_for_assertion(assertion, refs, asset_index)
        user_message = _user_message(
            prompt_text=prompt_text,
            selected_records=selected_records,
            all_records=all_records,
            assets_out_dir=assets_root / f"grpo_{assertion.get('assertion_id')}",
            copy_assets=copy_assets,
            max_images_per_sample=max_images_per_sample,
            max_multimodal_records_per_sample=max_multimodal_records_per_sample,
        )
        prompt_chat = Chat(
            messages=[
                SystemMessage(content=[TextContent(text=GRPO_SYSTEM_PROMPT)]),
                user_message,
            ]
        )
        evidence = assertion.get("evidence") if isinstance(assertion.get("evidence"), dict) else {}
        samples.append(
            GRPOSample(
                id=f"assertion_review_rl:{submission_id}:{assertion.get('assertion_id')}",
                sample_id=f"assertion_review:{submission_id}:{assertion.get('assertion_id')}",
                domain=domain,
                topic=topic,
                expert_key=reviewer_id,
                source_file=str(auto_path),
                prompt_chat=prompt_chat,
                reference_json=json.dumps({"verdict": verdict, "rationale": expert.get("rationale", "")}, ensure_ascii=False),
                reference_assertions_json=json.dumps([
                    {
                        "subject": assertion.get("subject", ""),
                        "predicate": assertion.get("predicate", ""),
                        "object": assertion.get("object", ""),
                    }
                ], ensure_ascii=False),
                reference_temporal_json=json.dumps({
                    "start_date": assertion.get("start_date"),
                    "end_date": assertion.get("end_date"),
                }, ensure_ascii=False),
                expected_verdict=verdict,
                evidence_text=str(evidence.get("text") or "").strip(),
                metadata=GRPOMetadata(
                    submission_id=submission_id,
                    assertion_id=assertion.get("assertion_id"),
                    importance_score=assertion.get("importance_score"),
                    expert=ExpertSignals(
                        semantic_correctness=str(expert.get("semantic_correctness") or ""),
                        evidence_sufficiency=str(expert.get("evidence_sufficiency") or ""),
                        scope_match=str(expert.get("scope_match") or ""),
                        hypothesis_role=str(expert.get("hypothesis_role") or ""),
                        causal_status=str(expert.get("causal_status") or ""),
                        severity=str(expert.get("severity") or ""),
                        leakage_risk=str(expert.get("leakage_risk") or ""),
                        time_confidence=str(expert.get("time_confidence") or ""),
                        mm_verdict=str(expert.get("mm_verdict") or ""),
                    ),
                    extra={
                        "multimodal_selected": len(selected_records),
                        "multimodal_available": len(all_records),
                    },
                ),
            )
        )
    return samples


def _refs_for_step(doc: dict[str, Any], step: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    papers = doc.get("papers") if isinstance(doc.get("papers"), list) else []
    paper_map = {str(p.get("id")): p for p in papers if isinstance(p, dict) and p.get("id")}
    for src in step.get("sources") or []:
        if not isinstance(src, dict):
            continue
        ref = str(src.get("paper_ref_id") or "").strip()
        if ref:
            refs.append(ref)
            continue
        raw = str(src.get("source") or "").strip()
        if raw:
            try:
                refs.append(resolve(raw).id)
            except Exception:
                refs.append(raw)
    if not refs:
        refs.extend(paper_map.keys())
    return _dedupe(refs)


def _refs_for_assertion(assertion: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for ref in assertion.get("paper_ids") or []:
        if isinstance(ref, str) and ref.strip():
            refs.append(ref.strip())
    evidence = assertion.get("evidence") if isinstance(assertion.get("evidence"), dict) else {}
    if evidence.get("paper_id"):
        refs.insert(0, str(evidence.get("paper_id")))
    return _dedupe(refs)


def _select_records_for_step(step: dict[str, Any], refs: Sequence[str], asset_index: AssetIndex) -> tuple[list[AssetRecord], list[AssetRecord]]:
    all_records = _sort_records(asset_index.records_for_refs(refs))
    selected: list[AssetRecord] = []
    sources = step.get("sources") if isinstance(step.get("sources"), list) else []
    for src in sources:
        if not isinstance(src, dict):
            continue
        src_ref = str(src.get("paper_ref_id") or src.get("source") or "").strip()
        page = _coerce_int(src.get("page"))
        locator = str(src.get("locator") or "").strip()
        kind = str(src.get("type") or "text").strip().lower()
        pool = _sort_records(asset_index.records_for_ref(src_ref) if src_ref else all_records)
        matched = _match_records(pool, page=page, locator=locator, kind=kind)
        selected.extend(matched)
    if not selected:
        selected = _fallback_records(all_records)
    return _dedupe_records(selected), all_records


def _select_records_for_assertion(assertion: dict[str, Any], refs: Sequence[str], asset_index: AssetIndex) -> tuple[list[AssetRecord], list[AssetRecord]]:
    all_records = _sort_records(asset_index.records_for_refs(refs))
    evidence = assertion.get("evidence") if isinstance(assertion.get("evidence"), dict) else {}
    page = _coerce_int(evidence.get("page"))
    locator = str(evidence.get("figure_or_table") or "").strip()
    matched = _match_records(all_records, page=page, locator=locator, kind="image" if locator else "text")
    if not matched and evidence.get("image_path"):
        image_path = str(evidence.get("image_path") or "").strip()
        matched = [rec for rec in all_records if rec.image_path == image_path]
    if not matched:
        matched = _fallback_records(all_records)
    return _dedupe_records(matched), all_records


def _fallback_records(records: Sequence[AssetRecord]) -> list[AssetRecord]:
    image_first = [rec for rec in records if rec.has_image]
    if image_first:
        return image_first[:3]
    return list(records[:3])


def _match_records(records: Sequence[AssetRecord], *, page: Optional[int], locator: str, kind: str) -> list[AssetRecord]:
    out: list[AssetRecord] = []
    loc_kind, loc_num = parse_locator_number(locator)
    for rec in records:
        if page is not None and _page_matches(rec.page, page):
            out.append(rec)
            continue
        if locator and rec.locator and locator.lower() in rec.locator.lower():
            out.append(rec)
            continue
        if loc_kind and rec.modality in {"figure", "table"} and rec.locator:
            rec_kind, rec_num = parse_locator_number(rec.locator)
            if rec_kind == loc_kind and loc_num is not None and rec_num == loc_num:
                out.append(rec)
                continue
        if kind in {"image", "table"} and rec.has_image and rec not in out:
            if rec.tables_md or rec.caption or rec.modality in {"figure", "table", "page"}:
                out.append(rec)
    return out


def _page_matches(candidate: Optional[int], requested: int) -> bool:
    if candidate is None:
        return False
    return candidate == requested or candidate + 1 == requested or candidate - 1 == requested


def _sort_records(records: Sequence[AssetRecord]) -> list[AssetRecord]:
    def key(rec: AssetRecord) -> tuple[int, int, int, str]:
        modality_rank = 0 if rec.modality in {"figure", "table"} else (1 if rec.modality == "page" else 2)
        image_rank = 0 if rec.has_image else 1
        page_rank = rec.page if rec.page is not None else 10_000
        return (image_rank, modality_rank, page_rank, rec.record_id)
    return sorted(records, key=key)


def _dedupe_records(records: Sequence[AssetRecord]) -> list[AssetRecord]:
    seen: set[str] = set()
    out: list[AssetRecord] = []
    for rec in records:
        if rec.record_id in seen:
            continue
        seen.add(rec.record_id)
        out.append(rec)
    return out


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, "", "unknown"):
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _user_message(
    *,
    prompt_text: str,
    selected_records: Sequence[AssetRecord],
    all_records: Sequence[AssetRecord],
    assets_out_dir: Path,
    copy_assets: bool,
    max_images_per_sample: int,
    max_multimodal_records_per_sample: int,
) -> UserMessage:
    content: list[TextContent | ImageContent] = [TextContent(text=prompt_text)]
    mm_text = _build_multimodal_text(all_records, max_records=max_multimodal_records_per_sample)
    if mm_text:
        content.append(TextContent(text=mm_text, meta={"role": "multimodal_context"}))

    image_records = _image_records(selected_records, all_records, max_images_per_sample=max_images_per_sample)
    for idx, rec in enumerate(image_records):
        image_path = rec.image_path
        if copy_assets:
            image_path = clone_image(image_path, assets_out_dir)
        if not image_path:
            continue
        content.append(
            ImageContent(
                image=image_path,
                meta={
                    "role": "primary_figure" if idx == 0 else "extra_figure",
                    "paper_id": rec.paper_id,
                    "page": rec.page,
                    "locator": rec.locator,
                    "source": rec.source,
                },
            )
        )
    return UserMessage(content=content)


def _image_records(selected_records: Sequence[AssetRecord], all_records: Sequence[AssetRecord], *, max_images_per_sample: int) -> list[AssetRecord]:
    chosen = [rec for rec in selected_records if rec.has_image]
    if max_images_per_sample == 0:
        limit = None
    else:
        limit = max_images_per_sample
    if limit is not None:
        chosen = chosen[:limit]
    if limit is None or len(chosen) < limit:
        needed = None if limit is None else (limit - len(chosen))
        existing = {rec.record_id for rec in chosen}
        for rec in all_records:
            if not rec.has_image or rec.record_id in existing:
                continue
            chosen.append(rec)
            existing.add(rec.record_id)
            if needed is not None and len(chosen) >= limit:
                break
    return chosen


def _build_multimodal_text(records: Sequence[AssetRecord], *, max_records: int) -> str:
    if not records:
        return ""
    limit = len(records) if max_records == 0 else min(len(records), max_records)
    lines = ["Multimodal evidence extracted from cited articles:"]
    for rec in list(records)[:limit]:
        page_part = f"page={rec.page}" if rec.page is not None else "page=?"
        locator = f" locator={rec.locator}" if rec.locator else ""
        summary = _trim_text(rec.summary_text(), limit=600)
        line = f"- paper={rec.paper_id} | modality={rec.modality} | {page_part}{locator}"
        if summary:
            line += f" | {summary}"
        lines.append(line)
    if limit < len(records):
        lines.append(f"- ... plus {len(records) - limit} more multimodal records")
    return "\n".join(lines)


def _trim_text(text: str, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
