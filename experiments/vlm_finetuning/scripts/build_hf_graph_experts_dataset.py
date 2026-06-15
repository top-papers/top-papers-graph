#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import ClassLabel, load_dataset
from huggingface_hub import snapshot_download
from PIL import Image

DEFAULT_EXPORT_SUBDIR = "exports/colab-run-001"
DEFAULT_PROMPT = (
    "Classify this scientific graph expert image into the exact dataset class. "
    "Return only the class label, without explanations."
)


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
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(row)!r}")
            rows.append(row)
    return rows


def normalize_label(value: Any, names: List[str] | None = None) -> Tuple[int | None, str]:
    if isinstance(value, int):
        if names and 0 <= value < len(names):
            return value, str(names[value])
        return value, str(value)
    text = str(value)
    if names and text in names:
        return names.index(text), text
    return None, text


def save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, dict) and image.get("path"):
        img = Image.open(image["path"])
    else:
        raise TypeError(f"Unsupported image value: {type(image)!r}")
    if img.mode not in {"RGB", "RGBA"}:
        img = img.convert("RGB")
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    img.save(path, quality=95)


def make_sft_record(sample_id: str, image_path: str, label_text: str, label_id: int | None, prompt: str) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "task_family": "hf_image_label_sft",
        "image": image_path,
        "images": [image_path],
        "label": label_id,
        "label_text": label_text,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a careful VLM classifier for scientific graph expert data.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label_text}]},
        ],
    }


def make_grpo_record(sample_id: str, image_path: str, label_text: str, label_id: int | None, prompt: str) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "task_family": "image_label_rl",
        "domain": "scientific_graph_experts",
        "images": [image_path],
        "reference_label": label_text,
        "reference_label_id": label_id,
        "label_text": label_text,
        "prompt": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a careful VLM classifier. Answer with the exact class label only.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }


def stable_row_key(row: Dict[str, Any], fallback_index: int) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("id", "sample_id", "submission_id", "assertion_id", "step_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row:{fallback_index:06d}"


def sort_for_split(rows: List[Dict[str, Any]], rng: random.Random) -> List[int]:
    # Deterministic shuffle without depending on the original HF/Xet chunk order.
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return indices


def stratified_split(rows: List[Dict[str, Any]], eval_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_family") or "unknown")].append(row)

    train: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for task_family in sorted(grouped):
        group = grouped[task_family]
        order = sort_for_split(group, rng)
        if len(group) <= 1 or eval_ratio <= 0:
            eval_count = 0
        else:
            eval_count = max(1, int(round(len(group) * eval_ratio)))
            eval_count = min(eval_count, max(1, len(group) - 1))
        eval_set = set(order[:eval_count])
        for i, row in enumerate(group):
            (eval_rows if i in eval_set else train).append(row)
    return train, eval_rows


def as_list(value: Any) -> List[Any]:
    if value in (None, "", []):
        return []
    return value if isinstance(value, list) else [value]


def resolve_image_path(value: Any, repo_root: Path, export_dir: Path) -> Tuple[str | None, bool]:
    if isinstance(value, dict):
        value = value.get("path") or value.get("image") or value.get("url") or value.get("bytes")
    if value in (None, ""):
        return None, True
    if not isinstance(value, str):
        return value, True
    text = value.strip()
    if not text:
        return None, True
    if text.startswith("http://") or text.startswith("https://"):
        return text, True

    path = Path(text)
    candidates: List[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        # The export README states that image paths are relative to the HF
        # snapshot root. Older intermediate exports used paths relative to the
        # export directory, so keep that fallback.
        candidates = [repo_root / path, export_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve().as_posix(), True
    # Keep a resolvable absolute-looking path for easier debugging, but mark it.
    fallback = candidates[0] if candidates else path
    return fallback.resolve().as_posix(), False


def normalise_images(
    row: Dict[str, Any],
    repo_root: Path,
    export_dir: Path,
    max_images: int,
) -> Tuple[Dict[str, Any], int, int, int]:
    out = dict(row)
    raw_images = as_list(out.get("images")) or as_list(out.get("image"))
    before = len(raw_images)
    resolved: List[str] = []
    missing = 0
    for item in raw_images:
        image_path, exists = resolve_image_path(item, repo_root, export_dir)
        if image_path is None:
            continue
        if not exists:
            missing += 1
        if image_path not in resolved:
            resolved.append(image_path)
    if max_images > 0:
        kept = resolved[:max_images]
    else:
        kept = resolved
    out["images"] = kept
    if kept:
        out["image"] = kept[0]
    elif "image" in out:
        out.pop("image", None)
    return out, before, len(kept), missing


def normalise_export_rows(
    rows: List[Dict[str, Any]],
    repo_root: Path,
    export_dir: Path,
    max_images: int,
    max_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(rows)
    if max_samples and max_samples > 0 and len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]

    output: List[Dict[str, Any]] = []
    image_refs_before = 0
    image_refs_after = 0
    missing_image_refs = 0
    truncated_rows = 0
    ids = Counter()
    for i, row in enumerate(selected):
        norm_row, before, after, missing = normalise_images(row, repo_root, export_dir, max_images)
        sample_id = stable_row_key(norm_row, i)
        norm_row.setdefault("id", sample_id)
        norm_row.setdefault("sample_id", sample_id)
        ids[str(norm_row["id"])] += 1
        image_refs_before += before
        image_refs_after += after
        missing_image_refs += missing
        if max_images > 0 and before > after:
            truncated_rows += 1
        output.append(norm_row)

    duplicate_ids = {key: value for key, value in ids.items() if value > 1}
    stats = {
        "rows": len(output),
        "image_refs_before_cap": image_refs_before,
        "image_refs_after_cap": image_refs_after,
        "missing_image_refs": missing_image_refs,
        "rows_with_truncated_images": truncated_rows,
        "duplicate_ids": duplicate_ids,
        "task_family_counts": dict(sorted(Counter(str(r.get("task_family") or "unknown") for r in output).items())),
    }
    return output, stats



def select_debug_rows(rows: List[Dict[str, Any]], max_samples: int, seed: int) -> List[Dict[str, Any]]:
    """Deterministically cap rows before downloading export assets for smoke jobs."""
    selected = list(rows)
    if max_samples and max_samples > 0 and len(selected) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = selected[:max_samples]
    return selected


def export_metadata_allow_patterns(export_subdir: str) -> List[str]:
    return [
        f"{export_subdir}/sft.jsonl",
        f"{export_subdir}/grpo.jsonl",
        f"{export_subdir}/README.md",
        f"{export_subdir}/export_summary.json",
        f"{export_subdir}/article_image_sources.jsonl",
        f"{export_subdir}/ARTICLE_IMAGE_SOURCES.md",
    ]


def _asset_allow_pattern(value: Any, export_subdir: str) -> str | None:
    """Convert a JSON image reference to an exact HF snapshot allow_pattern."""
    if isinstance(value, dict):
        value = value.get("path") or value.get("image") or value.get("url") or value.get("bytes")
    if not isinstance(value, str):
        return None
    text = value.strip().lstrip("./")
    if not text or text.startswith("http://") or text.startswith("https://") or text.startswith("/"):
        return None
    if text.startswith(f"{export_subdir}/assets/"):
        return text
    if text.startswith("assets/"):
        return f"{export_subdir}/{text}"
    marker = f"/{export_subdir}/assets/"
    if marker in text:
        return f"{export_subdir}/assets/{text.split(marker, 1)[1]}"
    marker = "/assets/"
    if marker in text:
        return f"{export_subdir}/assets/{text.split(marker, 1)[1]}"
    return None


def collect_asset_allow_patterns(
    rows: List[Dict[str, Any]],
    export_subdir: str,
    max_images: int,
) -> List[str]:
    patterns: List[str] = []
    seen: set[str] = set()
    for row in rows:
        raw_images = as_list(row.get("images")) or as_list(row.get("image"))
        if max_images > 0:
            raw_images = raw_images[:max_images]
        for item in raw_images:
            pattern = _asset_allow_pattern(item, export_subdir)
            if pattern and pattern not in seen:
                seen.add(pattern)
                patterns.append(pattern)
    return patterns

def build_from_export(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_subdir = args.export_subdir.strip("/")
    metadata_patterns = export_metadata_allow_patterns(export_subdir)

    # Smoke/debug runs should not ask Hugging Face Hub for every image in the
    # export. Download metadata first, select a small deterministic subset, then
    # request only the assets referenced by that subset. This avoids thousands of
    # HEAD requests and the 429 backoff visible in DataSphere smoke logs.
    using_debug_caps = bool(
        (args.max_sft_samples and args.max_sft_samples > 0)
        or (args.max_grpo_samples and args.max_grpo_samples > 0)
    )
    initial_allow_patterns = metadata_patterns if using_debug_caps else [*metadata_patterns, f"{export_subdir}/assets/**"]

    repo_root = Path(
        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=initial_allow_patterns,
            local_dir=args.snapshot_dir,
            local_dir_use_symlinks=False if args.snapshot_dir else None,
        )
    ).resolve()
    export_dir = repo_root / export_subdir
    sft_path = export_dir / "sft.jsonl"
    grpo_path = export_dir / "grpo.jsonl"
    if not sft_path.exists() or not grpo_path.exists():
        raise FileNotFoundError(
            f"Expected export files not found under {export_dir}. "
            f"Got sft={sft_path.exists()}, grpo={grpo_path.exists()}"
        )

    sft_rows_raw_all = read_jsonl(sft_path)
    grpo_rows_raw_all = read_jsonl(grpo_path)
    sft_rows_raw = select_debug_rows(sft_rows_raw_all, args.max_sft_samples, args.seed)
    grpo_rows_raw = select_debug_rows(grpo_rows_raw_all, args.max_grpo_samples, args.seed + 17)

    asset_patterns: List[str] = []
    if using_debug_caps:
        asset_patterns.extend(
            collect_asset_allow_patterns(sft_rows_raw, export_subdir, args.max_images_per_example_sft)
        )
        asset_patterns.extend(
            p for p in collect_asset_allow_patterns(grpo_rows_raw, export_subdir, args.max_images_per_example_grpo)
            if p not in asset_patterns
        )
        if asset_patterns:
            repo_root = Path(
                snapshot_download(
                    repo_id=args.dataset_id,
                    repo_type="dataset",
                    revision=args.revision,
                    allow_patterns=[*metadata_patterns, *asset_patterns],
                    local_dir=args.snapshot_dir,
                    local_dir_use_symlinks=False if args.snapshot_dir else None,
                )
            ).resolve()
            export_dir = repo_root / export_subdir

    sft_rows, sft_stats = normalise_export_rows(
        sft_rows_raw,
        repo_root,
        export_dir,
        max_images=args.max_images_per_example_sft,
        max_samples=0,
        seed=args.seed,
    )
    grpo_rows, grpo_stats = normalise_export_rows(
        grpo_rows_raw,
        repo_root,
        export_dir,
        max_images=args.max_images_per_example_grpo,
        max_samples=0,
        seed=args.seed + 17,
    )

    sft_train, sft_eval = stratified_split(sft_rows, args.eval_ratio, args.seed)
    grpo_train, grpo_eval = stratified_split(grpo_rows, args.eval_ratio, args.seed + 31)

    write_jsonl(out_dir / "sft_train.jsonl", sft_train)
    write_jsonl(out_dir / "sft_eval.jsonl", sft_eval)
    write_jsonl(out_dir / "sft_all.jsonl", [*sft_train, *sft_eval])
    write_jsonl(out_dir / "grpo_train.jsonl", grpo_train)
    write_jsonl(out_dir / "grpo_eval.jsonl", grpo_eval)
    write_jsonl(out_dir / "grpo_all.jsonl", [*grpo_train, *grpo_eval])

    for filename in ["README.md", "export_summary.json", "ARTICLE_IMAGE_SOURCES.md", "article_image_sources.jsonl"]:
        src = export_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)

    summary = {
        "source_mode": "export",
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "export_subdir": export_subdir,
        "snapshot_dir": repo_root.as_posix(),
        "export_dir": export_dir.as_posix(),
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "max_sft_samples": args.max_sft_samples,
        "max_grpo_samples": args.max_grpo_samples,
        "raw_sft_rows_total": len(sft_rows_raw_all),
        "raw_grpo_rows_total": len(grpo_rows_raw_all),
        "sample_limited_asset_download": using_debug_caps,
        "asset_patterns_requested": len(asset_patterns),
        "max_images_per_example_sft": args.max_images_per_example_sft,
        "max_images_per_example_grpo": args.max_images_per_example_grpo,
        "sft": {
            **sft_stats,
            "train": len(sft_train),
            "eval": len(sft_eval),
        },
        "grpo": {
            **grpo_stats,
            "train": len(grpo_train),
            "eval": len(grpo_eval),
        },
        "outputs": {
            "sft_train": str(out_dir / "sft_train.jsonl"),
            "sft_eval": str(out_dir / "sft_eval.jsonl"),
            "grpo_train": str(out_dir / "grpo_train.jsonl"),
            "grpo_eval": str(out_dir / "grpo_eval.jsonl"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

def build_from_imagefolder(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir.resolve()
    image_dir = out_dir / "images"
    ds = load_dataset(args.dataset_id, split=args.split)

    label_feature = ds.features.get("label")
    label_names = label_feature.names if isinstance(label_feature, ClassLabel) else None

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    if args.max_sft_samples and args.max_sft_samples > 0:
        indices = indices[: args.max_sft_samples]

    eval_count = max(1, int(round(len(indices) * args.eval_ratio))) if len(indices) > 1 else 0
    eval_ids = set(indices[:eval_count])

    sft_train: List[Dict[str, Any]] = []
    sft_eval: List[Dict[str, Any]] = []
    grpo_train: List[Dict[str, Any]] = []
    grpo_eval: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row_idx in indices:
        row = ds[row_idx]
        label_id, label_text = normalize_label(row.get("label"), label_names)
        abs_image = image_dir / f"sample_{row_idx:06d}.jpg"
        save_image(row["image"], abs_image)
        try:
            rel_image = abs_image.relative_to(Path.cwd()).as_posix()
        except ValueError:
            rel_image = abs_image.as_posix()
        sample_id = f"hf_top_papers_graph_experts:{args.split}:{row_idx:06d}"
        sft_row = make_sft_record(sample_id, rel_image, label_text, label_id, args.prompt)
        grpo_row = make_grpo_record(sample_id, rel_image, label_text, label_id, args.prompt)
        counts[label_text] += 1
        if row_idx in eval_ids:
            sft_eval.append(sft_row)
            grpo_eval.append(grpo_row)
        else:
            sft_train.append(sft_row)
            grpo_train.append(grpo_row)

    write_jsonl(out_dir / "sft_train.jsonl", sft_train)
    write_jsonl(out_dir / "sft_eval.jsonl", sft_eval)
    write_jsonl(out_dir / "sft_all.jsonl", [*sft_train, *sft_eval])
    write_jsonl(out_dir / "grpo_train.jsonl", grpo_train)
    write_jsonl(out_dir / "grpo_eval.jsonl", grpo_eval)
    write_jsonl(out_dir / "grpo_all.jsonl", [*grpo_train, *grpo_eval])

    summary = {
        "source_mode": "imagefolder",
        "dataset_id": args.dataset_id,
        "split": args.split,
        "rows_total": len(indices),
        "sft_train": len(sft_train),
        "sft_eval": len(sft_eval),
        "grpo_train": len(grpo_train),
        "grpo_eval": len(grpo_eval),
        "label_count": len(counts),
        "labels": dict(sorted(counts.items())),
        "image_dir": str(image_dir),
        "outputs": {
            "sft_train": str(out_dir / "sft_train.jsonl"),
            "sft_eval": str(out_dir / "sft_eval.jsonl"),
            "grpo_train": str(out_dir / "grpo_train.jsonl"),
            "grpo_eval": str(out_dir / "grpo_eval.jsonl"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build local SFT/GRPO JSONL files from the HF top-papers graph experts dataset.")
    ap.add_argument("--dataset-id", default="top-papers/top-papers-graph-experts-data")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--split", default="validation", help="Imagefolder fallback split.")
    ap.add_argument("--source-mode", choices=["export", "imagefolder"], default="export")
    ap.add_argument("--export-subdir", default=DEFAULT_EXPORT_SUBDIR)
    ap.add_argument("--snapshot-dir", type=Path, default=None, help="Optional local snapshot directory. Default uses HF cache.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/derived/hf_top_papers_graph_experts"))
    ap.add_argument("--eval-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-sft-samples", type=int, default=0, help="Optional SFT debug cap. 0 means all rows.")
    ap.add_argument("--max-grpo-samples", type=int, default=0, help="Optional GRPO debug cap. 0 means all rows.")
    ap.add_argument("--max-samples", type=int, default=0, help="Backward-compatible cap applied to both SFT and GRPO if set.")
    ap.add_argument("--max-images-per-example-sft", type=int, default=3)
    ap.add_argument("--max-images-per-example-grpo", type=int, default=2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples and args.max_samples > 0:
        if args.max_sft_samples <= 0:
            args.max_sft_samples = args.max_samples
        if args.max_grpo_samples <= 0:
            args.max_grpo_samples = args.max_samples
    if not (0 <= args.eval_ratio < 0.5):
        raise ValueError("--eval-ratio must be in [0, 0.5).")

    if args.source_mode == "export":
        summary = build_from_export(args)
    else:
        summary = build_from_imagefolder(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
