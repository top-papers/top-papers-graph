#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import ClassLabel, load_dataset
from PIL import Image

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build local SFT/GRPO JSONL files from the HF top-papers graph experts dataset.")
    ap.add_argument("--dataset-id", default="top-papers/top-papers-graph-experts-data")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out-dir", type=Path, default=Path("data/derived/hf_top_papers_graph_experts"))
    ap.add_argument("--eval-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-samples", type=int, default=0, help="Optional debug cap. 0 means all rows.")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    image_dir = out_dir / "images"
    ds = load_dataset(args.dataset_id, split=args.split)

    label_feature = ds.features.get("label")
    label_names = label_feature.names if isinstance(label_feature, ClassLabel) else None

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    if args.max_samples and args.max_samples > 0:
        indices = indices[: args.max_samples]

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
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
