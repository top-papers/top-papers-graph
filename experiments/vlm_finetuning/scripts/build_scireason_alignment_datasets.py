#!/usr/bin/env python3
"""Build leakage-safe SFT/DPO/GRPO datasets for SciReason VLM fine-tuning.

This v2 builder deliberately uses the prepared HF export files
(`exports/colab-run-001/sft.jsonl` and `grpo.jsonl`) instead of the generic
HF imagefolder viewer.  The viewer is useful for browsing assets, but it is not
an instruction/reasoning dataset.

Outputs are designed for the recommended training sequence:
  1. text-only SFT with assistant-only loss,
  2. multimodal SFT on evidence-bearing rows,
  3. DPO on gold-vs-negative preference pairs,
  4. optional short GRPO only on reward-ready rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from huggingface_hub import snapshot_download

DEFAULT_DATASET_ID = "top-papers/top-papers-graph-experts-data"
DEFAULT_EXPORT_SUBDIR = "exports/colab-run-001"
DEFAULT_OUT_DIR = Path("data/derived/hf_top_papers_scireason_v2")

TEXT_KEYS = (
    "id",
    "sample_id",
    "task_family",
    "domain",
    "topic",
    "claim",
    "question",
    "prompt",
    "reference_label",
    "label_text",
    "expected_verdict",
    "verdict",
    "rationale",
    "evidence",
    "paper_id",
    "source_file",
    "submission_id",
    "expert_key",
)
IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|webp|tiff?|bmp)$", re.IGNORECASE)
TOKEN_RE = re.compile(r"[\wА-Яа-яёЁ]{3,}", re.UNICODE)
REWARD_READY_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "image_label_rl": ("reference_label",),
    "trajectory_reasoning_rl": ("reference_assertions_json", "reference_json", "expected_output"),
    "assertion_review_rl": ("expected_verdict", "reference_verdict", "verdict"),
    "mm_review_rl": ("expected_verdict", "reference_verdict", "mm_verdict"),
    "temporal_fix_rl": ("reference_temporal_json", "corrected_time", "corrected_interval", "expected_output"),
}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(obj)!r}")
            rows.append(obj)
    return rows


def as_list(value: Any) -> List[Any]:
    if value in (None, "", []):
        return []
    return value if isinstance(value, list) else [value]


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def normalize_token_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def tokenize(value: Any) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(stable_json(value))}


def first_nonempty(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return default


def compact_dict(obj: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, Mapping):
            vv = compact_dict(v)
            if vv:
                out[k] = vv
        elif isinstance(v, list):
            vv = [x for x in v if x not in (None, "", [], {})]
            if vv:
                out[k] = vv
        elif v not in (None, "", [], {}):
            out[k] = v
    return out


def metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, Mapping) else {}


def nested_get(row: Mapping[str, Any], *keys: str) -> Any:
    meta = metadata(row)
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
        value = meta.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def row_id(row: Mapping[str, Any], fallback_index: int) -> str:
    value = nested_get(row, "id", "sample_id", "assertion_id", "step_id")
    if value not in (None, ""):
        return str(value)
    payload = stable_json({k: row.get(k) for k in TEXT_KEYS if k in row})
    digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"row:{fallback_index:06d}:{digest}"


def leakage_group_key(row: Mapping[str, Any], fallback_index: int) -> str:
    """Coarse source key used to avoid train/eval leakage.

    The export has evolved over time, so the function looks for multiple common
    field names.  It intentionally groups rows by paper/source/expert before
    falling back to the sample id.
    """
    candidates = [
        nested_get(row, "paper_id", "paper", "doi", "arxiv_id", "pmid"),
        nested_get(row, "source_file", "input_file", "source_path"),
        nested_get(row, "submission_id", "source_id"),
        nested_get(row, "expert_key", "expert_id", "reviewer"),
    ]
    parts = [normalize_token_text(x) for x in candidates if x not in (None, "", [], {})]
    if parts:
        return "::".join(parts[:4])
    return row_id(row, fallback_index)


def deterministic_group_split(rows: Sequence[Dict[str, Any]], eval_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    task_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for idx, row in enumerate(rows):
        key = leakage_group_key(row, idx)
        groups[key].append(idx)
        task_counts[key][str(row.get("task_family") or "unknown")] += 1

    ordered = sorted(groups, key=lambda key: hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest())
    target_eval_rows = int(round(len(rows) * eval_ratio))
    eval_indices: set[int] = set()
    eval_groups: List[str] = []
    for key in ordered:
        if len(eval_indices) >= target_eval_rows and eval_indices:
            break
        if len(rows) - (len(eval_indices) + len(groups[key])) <= 0:
            continue
        eval_indices.update(groups[key])
        eval_groups.append(key)

    train = [row for idx, row in enumerate(rows) if idx not in eval_indices]
    eval_rows = [row for idx, row in enumerate(rows) if idx in eval_indices]
    train_groups = set(groups) - set(eval_groups)
    leakage_overlap = sorted(set(eval_groups).intersection(train_groups))
    report = {
        "strategy": "leakage_safe_group_split",
        "rows_total": len(rows),
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "groups_total": len(groups),
        "train_groups": len(train_groups),
        "eval_groups": len(eval_groups),
        "eval_ratio_requested": eval_ratio,
        "eval_ratio_actual": round(len(eval_rows) / len(rows), 4) if rows else 0.0,
        "group_overlap_count": len(leakage_overlap),
        "group_overlap_preview": leakage_overlap[:20],
        "task_family_train": dict(sorted(Counter(str(r.get("task_family") or "unknown") for r in train).items())),
        "task_family_eval": dict(sorted(Counter(str(r.get("task_family") or "unknown") for r in eval_rows).items())),
    }
    return train, eval_rows, report


def resolve_image_ref(value: Any, repo_root: Path, export_dir: Path) -> Tuple[str | None, bool, str | None]:
    raw = value
    if isinstance(value, Mapping):
        value = first_nonempty(value.get("path"), value.get("image"), value.get("url"), value.get("bytes"))
    if value in (None, "", []):
        return None, True, None
    if not isinstance(value, str):
        return value, True, stable_json(raw)
    text = value.strip()
    if not text:
        return None, True, None
    if text.startswith("http://") or text.startswith("https://"):
        return text, True, text
    path = Path(text)
    candidates = [path] if path.is_absolute() else [repo_root / path, export_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve().as_posix(), True, text
    return candidates[0].resolve().as_posix(), False, text


def image_relevance_score(image_ref: Any, row: Mapping[str, Any], raw_index: int) -> float:
    """Prefer evidence/figure/table/page-matching images while preserving order.

    This is intentionally lightweight and deterministic: it uses only fields that
    are already in the export row, so it can run in DataSphere before model load.
    """
    image_text = stable_json(image_ref).lower()
    row_tokens = tokenize({k: row.get(k) for k in TEXT_KEYS if k in row})
    evidence_tokens = tokenize(first_nonempty(row.get("evidence"), row.get("evidence_json"), row.get("metadata"), default={}))
    score = 0.0
    basename_tokens = tokenize(Path(str(image_text)).name)
    score += 3.0 * len(basename_tokens.intersection(evidence_tokens))
    score += 0.75 * len(basename_tokens.intersection(row_tokens))
    if "figure" in image_text or "fig" in image_text:
        score += 1.0
    if "table" in image_text:
        score += 1.0
    if "page" in image_text or re.search(r"p(?:age)?[_-]?\d+", image_text):
        score += 0.4
    # Stable tiny tie-breaker keeps original order when scores are equal.
    return score - raw_index * 1e-6


def select_relevant_images(raw_images: List[Any], row: Mapping[str, Any], max_images: int) -> Tuple[List[Any], Dict[str, Any]]:
    if max_images <= 0 or len(raw_images) <= max_images:
        return list(raw_images), {"policy": "all", "truncated": False, "before": len(raw_images), "after": len(raw_images)}
    scored = [(image_relevance_score(img, row, idx), idx, img) for idx, img in enumerate(raw_images)]
    chosen = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_images]
    chosen_sorted = [img for _, _, img in sorted(chosen, key=lambda x: x[1])]
    return chosen_sorted, {
        "policy": "relevance_top_k_then_original_order",
        "truncated": True,
        "before": len(raw_images),
        "after": len(chosen_sorted),
        "score_preview": [round(x[0], 4) for x in sorted(chosen, key=lambda x: x[1])],
    }


def normalise_export_rows(rows: Sequence[Dict[str, Any]], repo_root: Path, export_dir: Path, max_images: int, seed: int, max_samples: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected = list(rows)
    if max_samples and max_samples > 0 and len(selected) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = selected[:max_samples]

    out: List[Dict[str, Any]] = []
    missing_refs = 0
    before_refs = 0
    after_refs = 0
    truncated_rows = 0
    rows_with_images = 0
    ids = Counter()
    for idx, row in enumerate(selected):
        norm = dict(row)
        rid = row_id(norm, idx)
        norm.setdefault("id", rid)
        norm.setdefault("sample_id", rid)
        norm["leakage_group"] = leakage_group_key(norm, idx)
        raw_images = as_list(norm.get("images")) or as_list(norm.get("image"))
        chosen_raw, selection_meta = select_relevant_images(raw_images, norm, max_images=max_images)
        resolved: List[str] = []
        for image in chosen_raw:
            resolved_path, exists, _raw_text = resolve_image_ref(image, repo_root, export_dir)
            if resolved_path is None:
                continue
            if not exists:
                missing_refs += 1
            if resolved_path not in resolved:
                resolved.append(resolved_path)
        before_refs += len(raw_images)
        after_refs += len(resolved)
        truncated_rows += int(bool(selection_meta.get("truncated")))
        rows_with_images += int(bool(resolved))
        norm["images"] = resolved
        if resolved:
            norm["image"] = resolved[0]
        else:
            norm.pop("image", None)
        norm.setdefault("metadata", {})
        if isinstance(norm["metadata"], dict):
            norm["metadata"].setdefault("image_selection", selection_meta)
            norm["metadata"].setdefault("leakage_group", norm["leakage_group"])
        ids[str(norm["id"])] += 1
        out.append(norm)

    image_selection_policy = "all_available_refs" if max_images <= 0 else "relevance_top_k_then_original_order"
    stats = {
        "rows": len(out),
        "rows_with_images": rows_with_images,
        "image_refs_before_selection": before_refs,
        "image_refs_after_selection": after_refs,
        "missing_image_refs": missing_refs,
        "rows_with_truncated_images": truncated_rows,
        "max_images_per_example": max_images,
        "image_selection_policy": image_selection_policy,
        "duplicate_ids": {k: v for k, v in ids.items() if v > 1},
        "task_family_counts": dict(sorted(Counter(str(r.get("task_family") or "unknown") for r in out).items())),
    }
    return out, stats


def assistant_text_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, Mapping):
                    if block.get("type") == "text" and block.get("text") not in (None, ""):
                        parts.append(str(block.get("text")))
                elif block not in (None, ""):
                    parts.append(str(block))
            return "\n".join(parts).strip() or None
    return None


def prompt_messages_without_answer(row: Mapping[str, Any]) -> Any:
    if isinstance(row.get("messages"), list):
        return [msg for msg in row["messages"] if not (isinstance(msg, Mapping) and msg.get("role") == "assistant")]
    return row.get("prompt") or row.get("prompt_chat") or row.get("question") or row.get("claim") or ""


def strip_images(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out.pop("image", None)
    out.pop("images", None)
    if isinstance(out.get("messages"), list):
        messages: List[Dict[str, Any]] = []
        for msg in out["messages"]:
            if not isinstance(msg, Mapping):
                messages.append(msg)
                continue
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, list):
                new_msg["content"] = [b for b in content if not (isinstance(b, Mapping) and b.get("type") == "image")]
            messages.append(new_msg)
        out["messages"] = messages
    return out


def is_multimodal(row: Mapping[str, Any]) -> bool:
    return bool(as_list(row.get("images")) or row.get("image"))


def task_hard_negative(task_family: str) -> str:
    if task_family == "trajectory_reasoning":
        return json.dumps({"inference": "unsupported", "next_question": "unknown", "extracted_assertions": []}, ensure_ascii=False)
    if task_family == "assertion_reconstruction":
        return json.dumps({"subject": "unknown", "predicate": "unknown", "object": "unknown", "evidence": []}, ensure_ascii=False)
    if task_family in {"assertion_review", "assertion_review_rl"}:
        return json.dumps({"verdict": "accept", "rationale": "No issues found."}, ensure_ascii=False)
    if task_family in {"temporal_fix", "temporal_fix_rl"}:
        return json.dumps({"start_date": "unknown", "end_date": "unknown", "time_source": "missing"}, ensure_ascii=False)
    return json.dumps({"answer": "I do not know.", "evidence": []}, ensure_ascii=False)


def make_dpo_rows_from_sft(rows: Sequence[Dict[str, Any]], synthetic_negatives: bool) -> List[Dict[str, Any]]:
    dpo_rows: List[Dict[str, Any]] = []
    for row in rows:
        chosen = first_nonempty(row.get("chosen"), row.get("preferred"), row.get("reference_response"), assistant_text_from_messages(row.get("messages")))
        rejected = first_nonempty(row.get("rejected"), row.get("dispreferred"), row.get("negative_response"))
        if not chosen:
            continue
        if not rejected and synthetic_negatives:
            rejected = task_hard_negative(str(row.get("task_family") or "unknown"))
        if not rejected or normalize_token_text(chosen) == normalize_token_text(rejected):
            continue
        rec = {
            "id": f"dpo:{row.get('id')}",
            "task_family": str(row.get("task_family") or "unknown"),
            "domain": row.get("domain"),
            "prompt": prompt_messages_without_answer(row),
            "chosen": chosen if isinstance(chosen, str) else stable_json(chosen),
            "rejected": rejected if isinstance(rejected, str) else stable_json(rejected),
            "metadata": {
                "source_id": row.get("id"),
                "leakage_group": row.get("leakage_group"),
                "synthetic_negative": not bool(first_nonempty(row.get("rejected"), row.get("dispreferred"), row.get("negative_response"))),
            },
        }
        if is_multimodal(row):
            rec["images"] = row.get("images") or as_list(row.get("image"))
            if rec["images"]:
                rec["image"] = rec["images"][0]
        dpo_rows.append(rec)
    return dpo_rows



VERDICT_ALIASES = {
    "accept": {"accept", "accepted", "valid", "supported", "true", "correct", "ok", "approve", "approved"},
    "reject": {"reject", "rejected", "invalid", "unsupported", "false", "incorrect", "wrong", "deny", "denied"},
    "revise": {"revise", "revision", "needs revision", "needs_revision", "fix", "correct", "modify", "partial"},
}


def canonical_verdict(value: Any) -> str:
    text = normalize_token_text(value).replace("_", " ").replace("-", " ")
    if not text:
        return "unknown"
    # Exact alias matching first; avoid classifying "unsupported" as "accept"
    # merely because it contains the substring "supported".
    for canonical, aliases in VERDICT_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
    if text.startswith("un") and ("support" in text or "valid" in text or "correct" in text):
        return "reject"
    if "not" in text and ("support" in text or "valid" in text or "correct" in text):
        return "reject"
    for canonical, aliases in VERDICT_ALIASES.items():
        if any((" " in alias) and alias in text for alias in aliases):
            return canonical
    return text


def opposite_verdict(verdict: str) -> str:
    verdict = canonical_verdict(verdict)
    if verdict == "accept":
        return "reject"
    if verdict == "reject":
        return "accept"
    if verdict == "revise":
        return "accept"
    return "unknown"


def ensure_text_response(value: Any) -> str:
    if isinstance(value, str):
        return value
    return stable_json(value)


def make_dpo_rows_from_grpo(rows: Sequence[Dict[str, Any]], synthetic_negatives: bool) -> List[Dict[str, Any]]:
    """Build additional offline preference pairs from explicit RL targets.

    The last successful run showed that GRPO reward variance is still weak.  These
    rows move the deterministic supervision signal into DPO first, using the RL
    export's expected verdict/temporal/assertion targets as preferred answers and
    a task-aware hard negative as the rejected completion.
    """
    dpo_rows: List[Dict[str, Any]] = []
    for row in rows:
        tf = str(row.get("task_family") or "unknown")
        chosen_obj: Any | None = None
        rejected_obj: Any | None = None
        if tf in {"assertion_review_rl", "mm_review_rl"}:
            expected = first_nonempty(
                nested_get(row, "expected_verdict", "reference_verdict", "verdict", "mm_verdict"),
                default=None,
            )
            if expected in (None, ""):
                continue
            verdict = canonical_verdict(expected)
            chosen_obj = {
                "verdict" if tf == "assertion_review_rl" else "mm_verdict": verdict,
                "rationale": first_nonempty(row.get("rationale"), row.get("evidence_text"), row.get("evidence"), default="Use the provided paper evidence and expert annotation."),
            }
            rejected_obj = {
                "verdict" if tf == "assertion_review_rl" else "mm_verdict": opposite_verdict(verdict),
                "rationale": "Incorrect preference bootstrap negative: contradicts the expert target.",
            }
        elif tf == "temporal_fix_rl":
            ref = first_nonempty(
                nested_get(row, "reference_temporal_json", "expected_output"),
                compact_dict({
                    "start_date": nested_get(row, "corrected_start_date", "start_date"),
                    "end_date": nested_get(row, "corrected_end_date", "end_date"),
                    "time_source": nested_get(row, "time_source"),
                }),
                default=None,
            )
            if ref in (None, "", [], {}):
                continue
            chosen_obj = ref
            rejected_obj = {"start_date": "unknown", "end_date": "unknown", "time_source": "missing"}
        elif tf == "trajectory_reasoning_rl":
            ref = first_nonempty(nested_get(row, "reference_assertions_json", "reference_json", "expected_output"), default=None)
            if ref in (None, "", [], {}):
                continue
            chosen_obj = {"inference": "supported", "next_question": "verify remaining evidence", "extracted_assertions": ref}
            rejected_obj = {"inference": "unsupported", "next_question": "unknown", "extracted_assertions": []}
        elif tf == "image_label_rl":
            label = first_nonempty(nested_get(row, "reference_label", "label_text"), default=None)
            if not label:
                continue
            chosen_obj = {"label": label}
            rejected_obj = {"label": "unknown"}
        else:
            continue

        if not synthetic_negatives and not first_nonempty(row.get("rejected"), row.get("dispreferred"), row.get("negative_response")):
            continue
        chosen = ensure_text_response(chosen_obj)
        rejected = ensure_text_response(first_nonempty(row.get("rejected"), row.get("dispreferred"), row.get("negative_response"), rejected_obj))
        if normalize_token_text(chosen) == normalize_token_text(rejected):
            continue
        rec: Dict[str, Any] = {
            "id": f"dpo-grpo:{row.get('id')}",
            "task_family": tf.replace("_rl", ""),
            "domain": row.get("domain"),
            "prompt": prompt_messages_without_answer(row),
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "source_id": row.get("id"),
                "source_task_family": tf,
                "leakage_group": row.get("leakage_group"),
                "synthetic_negative": not bool(first_nonempty(row.get("rejected"), row.get("dispreferred"), row.get("negative_response"))),
                "preference_source": "grpo_target_bootstrap",
            },
        }
        if is_multimodal(row):
            rec["images"] = row.get("images") or as_list(row.get("image"))
            if rec["images"]:
                rec["image"] = rec["images"][0]
        dpo_rows.append(rec)
    return dpo_rows


def dedupe_dpo_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = hashlib.sha1(stable_json({
            "prompt": row.get("prompt"),
            "chosen": row.get("chosen"),
            "rejected": row.get("rejected"),
            "images": row.get("images"),
        }).encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

def row_has_any(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    for key in keys:
        if row.get(key) not in (None, "", [], {}):
            return True
        if metadata(row).get(key) not in (None, "", [], {}):
            return True
    return False


def filter_reward_ready_grpo(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    dropped_by_reason = Counter()
    coverage: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        tf = str(row.get("task_family") or "unknown")
        coverage[tf]["rows"] += 1
        required = REWARD_READY_FIELDS.get(tf)
        if required is None:
            dropped_by_reason["unsupported_task_family"] += 1
            coverage[tf]["unsupported"] += 1
            continue
        if not row_has_any(row, required):
            dropped_by_reason["missing_reward_target"] += 1
            coverage[tf]["missing_reward_target"] += 1
            continue
        kept.append(row)
        coverage[tf]["kept"] += 1
    report = {
        "kept_rows": len(kept),
        "input_rows": len(rows),
        "kept_ratio": round(len(kept) / len(rows), 4) if rows else 0.0,
        "dropped_by_reason": dict(sorted(dropped_by_reason.items())),
        "by_task_family": {tf: dict(counter) for tf, counter in sorted(coverage.items())},
        "gate_note": "Use GRPO only for task families with explicit reward targets and non-trivial reward variance.",
    }
    return kept, report


def load_hf_export(args: argparse.Namespace) -> Tuple[Path, Path, List[Dict[str, Any]], List[Dict[str, Any]]]:
    export_subdir = args.export_subdir.strip("/")
    metadata_patterns = [
        f"{export_subdir}/sft.jsonl",
        f"{export_subdir}/grpo.jsonl",
        f"{export_subdir}/README.md",
        f"{export_subdir}/export_summary.json",
        f"{export_subdir}/ARTICLE_IMAGE_SOURCES.md",
        f"{export_subdir}/article_image_sources.jsonl",
    ]
    allow_patterns = list(metadata_patterns)
    if not args.metadata_only:
        allow_patterns.append(f"{export_subdir}/assets/**")

    repo_root = Path(snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=allow_patterns,
        max_workers=args.hf_download_max_workers,
        local_dir=args.snapshot_dir,
        local_dir_use_symlinks=False if args.snapshot_dir else None,
    )).resolve()
    export_dir = repo_root / export_subdir
    sft_path = export_dir / "sft.jsonl"
    grpo_path = export_dir / "grpo.jsonl"
    if not sft_path.exists() or not grpo_path.exists():
        raise FileNotFoundError(f"Expected sft.jsonl and grpo.jsonl under {export_dir}; imagefolder fallback is intentionally disabled.")
    return repo_root, export_dir, read_jsonl(sft_path), read_jsonl(grpo_path)


def copy_export_metadata(export_dir: Path, out_dir: Path) -> None:
    for filename in ["README.md", "export_summary.json", "ARTICLE_IMAGE_SOURCES.md", "article_image_sources.jsonl"]:
        src = export_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)


def write_report(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build export-only SciReason alignment datasets with leakage-safe splits.")
    ap.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--export-subdir", default=DEFAULT_EXPORT_SUBDIR)
    ap.add_argument("--snapshot-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--eval-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-sft-samples", type=int, default=0)
    ap.add_argument("--max-grpo-samples", type=int, default=0)
    ap.add_argument("--max-images-per-example-sft", type=int, default=0, help="0 means preserve all image refs from the export.")
    ap.add_argument("--max-images-per-example-grpo", type=int, default=0, help="0 means preserve all image refs from the export.")
    ap.add_argument("--hf-download-max-workers", type=int, default=2)
    ap.add_argument("--metadata-only", action="store_true", help="Do not download assets; useful for fast CI smoke checks.")
    ap.add_argument("--no-synthetic-dpo-negatives", action="store_true", help="Only emit DPO rows that already contain explicit rejected answers.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.eval_ratio < 0.5):
        raise ValueError("--eval-ratio must be in [0, 0.5).")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root, export_dir, sft_raw, grpo_raw = load_hf_export(args)
    sft_rows, sft_stats = normalise_export_rows(sft_raw, repo_root, export_dir, args.max_images_per_example_sft, args.seed, args.max_sft_samples)
    grpo_rows, grpo_stats = normalise_export_rows(grpo_raw, repo_root, export_dir, args.max_images_per_example_grpo, args.seed + 17, args.max_grpo_samples)

    sft_train, sft_eval, sft_leakage = deterministic_group_split(sft_rows, args.eval_ratio, args.seed)
    grpo_train_all, grpo_eval_all, grpo_leakage = deterministic_group_split(grpo_rows, args.eval_ratio, args.seed + 31)

    dpo_from_sft = make_dpo_rows_from_sft(sft_rows, synthetic_negatives=not args.no_synthetic_dpo_negatives)
    dpo_from_grpo = make_dpo_rows_from_grpo(grpo_rows, synthetic_negatives=not args.no_synthetic_dpo_negatives)
    dpo_all = dedupe_dpo_rows(dpo_from_sft + dpo_from_grpo)
    dpo_train, dpo_eval, dpo_leakage = deterministic_group_split(dpo_all, args.eval_ratio, args.seed + 61)

    grpo_ready_all, reward_audit_all = filter_reward_ready_grpo(grpo_rows)
    grpo_ready_train, grpo_ready_eval, grpo_ready_leakage = deterministic_group_split(grpo_ready_all, args.eval_ratio, args.seed + 91)

    write_jsonl(out_dir / "sft_text_train.jsonl", [strip_images(r) for r in sft_train])
    write_jsonl(out_dir / "sft_text_eval.jsonl", [strip_images(r) for r in sft_eval])
    write_jsonl(out_dir / "sft_vlm_train.jsonl", [r for r in sft_train if is_multimodal(r)])
    write_jsonl(out_dir / "sft_vlm_eval.jsonl", [r for r in sft_eval if is_multimodal(r)])
    write_jsonl(out_dir / "sft_all.jsonl", sft_rows)
    write_jsonl(out_dir / "dpo_train.jsonl", dpo_train)
    write_jsonl(out_dir / "dpo_eval.jsonl", dpo_eval)
    write_jsonl(out_dir / "dpo_all.jsonl", dpo_all)
    write_jsonl(out_dir / "grpo_train_verified.jsonl", grpo_ready_train)
    write_jsonl(out_dir / "grpo_eval_verified.jsonl", grpo_ready_eval)
    write_jsonl(out_dir / "grpo_all_verified.jsonl", grpo_ready_all)
    write_jsonl(out_dir / "grpo_train_all.jsonl", grpo_train_all)
    write_jsonl(out_dir / "grpo_eval_all.jsonl", grpo_eval_all)

    copy_export_metadata(export_dir, out_dir)

    leakage_report = {
        "sft": sft_leakage,
        "dpo": dpo_leakage,
        "grpo_all": grpo_leakage,
        "grpo_verified": grpo_ready_leakage,
    }
    image_report = {"sft": sft_stats, "grpo": grpo_stats}
    summary = {
        "source_mode": "export_only_scireason_v2",
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "export_subdir": args.export_subdir.strip("/"),
        "snapshot_dir": repo_root.as_posix(),
        "export_dir": export_dir.as_posix(),
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "max_sft_samples": args.max_sft_samples,
        "max_grpo_samples": args.max_grpo_samples,
        "max_images_per_example_sft": args.max_images_per_example_sft,
        "max_images_per_example_grpo": args.max_images_per_example_grpo,
        "row_sampling_limited": bool(args.max_sft_samples or args.max_grpo_samples),
        "image_selection_limited": bool(args.max_images_per_example_sft > 0 or args.max_images_per_example_grpo > 0),
        "full_data_policy": "all_rows_and_all_image_refs" if not (args.max_sft_samples or args.max_grpo_samples or args.max_images_per_example_sft > 0 or args.max_images_per_example_grpo > 0) else "limited_by_cli_arguments",
        "raw_sft_rows_total": len(sft_raw),
        "raw_grpo_rows_total": len(grpo_raw),
        "outputs": {
            "sft_text_train": str(out_dir / "sft_text_train.jsonl"),
            "sft_text_eval": str(out_dir / "sft_text_eval.jsonl"),
            "sft_vlm_train": str(out_dir / "sft_vlm_train.jsonl"),
            "sft_vlm_eval": str(out_dir / "sft_vlm_eval.jsonl"),
            "dpo_train": str(out_dir / "dpo_train.jsonl"),
            "dpo_eval": str(out_dir / "dpo_eval.jsonl"),
            "grpo_train_verified": str(out_dir / "grpo_train_verified.jsonl"),
            "grpo_eval_verified": str(out_dir / "grpo_eval_verified.jsonl"),
        },
        "counts": {
            "sft_text_train": len(sft_train),
            "sft_text_eval": len(sft_eval),
            "sft_vlm_train": sum(1 for r in sft_train if is_multimodal(r)),
            "sft_vlm_eval": sum(1 for r in sft_eval if is_multimodal(r)),
            "dpo_train": len(dpo_train),
            "dpo_eval": len(dpo_eval),
            "dpo_from_sft": len(dpo_from_sft),
            "dpo_from_grpo": len(dpo_from_grpo),
            "grpo_train_verified": len(grpo_ready_train),
            "grpo_eval_verified": len(grpo_ready_eval),
        },
        "quality_gates": {
            "imagefolder_fallback": "disabled",
            "split": "leakage_safe_group_split",
            "image_selection": "all_available_refs" if not (args.max_images_per_example_sft > 0 or args.max_images_per_example_grpo > 0) else "relevance_top_k_then_original_order",
            "full_data_usage_audit": "recommended",
            "dpo_negatives": "synthetic_bootstrap_enabled" if not args.no_synthetic_dpo_negatives else "explicit_pairs_only",
            "dpo_sources": {"sft_rows": len(dpo_from_sft), "grpo_reward_targets": len(dpo_from_grpo)},
            "grpo": "verified_reward_target_rows_only",
        },
    }
    write_report(out_dir / "leakage_report.json", leakage_report)
    write_report(out_dir / "image_resolution_report.json", image_report)
    write_report(out_dir / "reward_audit_by_task_family.json", reward_audit_all)
    write_report(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
