#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Create a deterministic, local-only synthetic chart benchmark for exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ADAPTER_ID,
    ADAPTER_REVISION,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    GENERATOR_VERSION,
    SCOPE_LABELS,
    SyntheticHarnessError,
    generator_binding,
    inventory,
    json_sha256,
    repo_root,
    resolve_run,
    safe_relative_path,
    scope_metadata,
    sha256_file,
    verify_generated_run,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_PROMPT = (
    "Read the integer displayed in the target panel of this synthetic chart. "
    "Return one strict SciReason JSON object with fields answer, evidence_used, visual_facts, "
    "temporal_facts, uncertainty, and missing_evidence. Set answer exactly to VALUE=<integer>, "
    "where <integer> is read from the chart. Do not add prose outside the JSON object."
)


def _font(size: int) -> Any:
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _panel_marker(index: int, seed: int) -> str:
    digest = hashlib.sha256(f"panel\x00{seed}\x00{index}".encode("ascii")).hexdigest()
    return "PANEL-" + "".join(chr(65 + int(char, 16) % 26) for char in digest[:8])


def _render_chart(path: Path, *, index: int, seed: int, value: int, marker: str) -> None:
    """Render one high-contrast chart whose target value only occurs in pixels."""

    try:
        from PIL import Image, ImageDraw, PngImagePlugin
    except ImportError as exc:  # pragma: no cover - Pillow is a declared VLM dependency
        raise SyntheticHarnessError("Pillow is required to generate synthetic PNG charts") from exc

    rng = random.Random(f"chart\x00{seed}\x00{index}")
    image = Image.new("RGB", (1200, 760), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    title_font = _font(34)
    label_font = _font(22)
    value_font = _font(80)
    small_font = _font(16)

    draw.rectangle((0, 0, 1200, 82), fill="#102a43")
    draw.text((38, 22), f"SYNTHETIC CHART {marker}", font=title_font, fill="#ffffff")
    chart = (70, 150, 760, 620)
    draw.rectangle(chart, outline="#102a43", width=5, fill="#ffffff")
    draw.line((120, 560, 700, 560), fill="#102a43", width=4)
    draw.line((120, 560, 120, 200), fill="#102a43", width=4)

    points: list[tuple[int, int]] = []
    for point_index in range(6):
        x = 150 + point_index * 100
        y = 500 - rng.randrange(40, 280)
        points.append((x, y))
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="#d64545", outline="#102a43", width=2)
    draw.line(points, fill="#d64545", width=6)
    draw.text((145, 584), "synthetic signal", font=label_font, fill="#102a43")
    draw.text((155, 170), "high contrast chart", font=label_font, fill="#102a43")

    panel = (820, 180, 1130, 550)
    draw.rounded_rectangle(panel, radius=28, fill="#102a43", outline="#ffcc4d", width=7)
    draw.text((860, 225), "TARGET VALUE", font=label_font, fill="#ffffff")
    value_text = str(value)
    bbox = draw.textbbox((0, 0), value_text, font=value_font)
    text_width = bbox[2] - bbox[0]
    draw.text((975 - text_width // 2, 330), value_text, font=value_font, fill="#ffcc4d")
    draw.text((855, 465), "read the panel", font=label_font, fill="#ffffff")

    draw.rectangle((0, 680, 1200, 760), fill="#102a43")
    draw.text(
        (30, 696),
        "SYNTHETIC_DATA_ONLY | EXPLORATORY_NOT_FOR_PUBLICATION | NO_HUMAN_REVIEW",
        font=small_font,
        fill="#ffffff",
    )
    draw.text(
        (30, 722),
        "NO_PUBLICATION_OR_SUPERIORITY_CLAIM",
        font=small_font,
        fill="#ffffff",
    )
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("scope_labels", "|".join(SCOPE_LABELS))
    metadata.add_text("synthetic_data_only", "true")
    metadata.add_text("human_review_performed", "false")
    metadata.add_text("publication_or_superiority_claim_allowed", "false")
    image.save(path, format="PNG", pnginfo=metadata, compress_level=9)


def _slug_for_run(run_relative: Path) -> str:
    raw = re.sub(r"[^a-z0-9]+", "-", run_relative.as_posix().lower()).strip("-")
    return raw[-42:] or "run"


def _config(
    run_relative: Path,
    sample_count: int,
    seed: int,
    base_revision: str,
    adapter_revision: str,
) -> dict[str, Any]:
    run_hash = hashlib.sha256(run_relative.as_posix().encode("utf-8")).hexdigest()[:12]
    identity = f"synthetic-exploratory-{_slug_for_run(run_relative)}-{run_hash}"
    nf4_kwargs = {
        "torch_dtype": "float16",
        "device_map": "balanced",
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "trust_remote_code": False,
        "quantization_config": {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
        },
    }
    return {
        "schema_version": 1,
        **scope_metadata(),
        "experiment": {
            "id": identity,
            "public_id": f"synthetic-exploratory-not-for-publication-{run_hash}",
            "seed": seed,
            "require_clean_code": False,
            "require_preregistered_plan": False,
            "output_dir": run_relative.as_posix(),
        },
        "benchmark": {
            "repo_id": "top-papers/synthetic-local-only",
            "revision": "0" * 40,
            "data_file": "data/benchmark.jsonl",
            "provenance_file": "article_image_sources.jsonl",
            "require_gold": False,
            "require_complete_provenance": True,
            "require_split_provenance": True,
        },
        "training_audit": {"require_lineage_manifest": False, "sources": []},
        "models": {
            "base": {
                "base_model": {"id": BASE_MODEL_ID, "revision": base_revision},
                "model_kwargs": nf4_kwargs,
            },
            "tuned": {
                "base_model": {"id": BASE_MODEL_ID, "revision": base_revision},
                "adapter": {
                    "id": ADAPTER_ID,
                    "revision": adapter_revision,
                    "adapter_name": "default",
                },
                "model_kwargs": nf4_kwargs,
            },
        },
        "processor": {
            "id": ADAPTER_ID,
            "revision": adapter_revision,
            "settings": {"max_pixels": 1003520, "trust_remote_code": False},
            "call_settings": {"padding": True},
            "chat_template_settings": {"add_vision_id": False},
        },
        "generation": {"do_sample": False, "max_new_tokens": 192, "use_cache": True},
        "conditions": ["original", "text_only", "shuffled_images"],
        "review": {
            "reviewer_ids": ["synthetic-automation-a", "synthetic-automation-b"],
            "reviews_per_item": 1,
            "primary_condition": "original",
        },
        "statistics": {
            "primary_condition": "original",
            "primary_endpoint_only": False,
            "primary_strata": ["multimodal_hard"],
        },
        "power": {
            "n_items": sample_count,
            "reviews_per_item": 1,
            "evaluable_fraction": 1.0,
            "intracluster_correlation": 0.0,
            "alpha": 0.05,
            "target_power": 0.8,
            "score_sd": 0.5,
            "target_effect": 0.2,
        },
    }


def _validate_generated_source(root: Path, source: Path, rows: list[dict[str, Any]], provenance: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Use the existing schema/audit code without entering strict publication mode."""

    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from scireason.vlm_ab.audit import audit_benchmark
    from scireason.vlm_ab.config import validate_experiment_config
    from scireason.vlm_ab.prepare import _validate_publication_schemas

    validate_experiment_config(config)
    _validate_publication_schemas(rows, provenance, root)
    report = audit_benchmark(
        rows,
        source,
        provenance_rows=provenance,
        require_gold=False,
        require_complete_provenance=True,
        require_split_provenance=True,
        require_citation=True,
        primary_strata=config["statistics"]["primary_strata"],
    )
    technical = {
        "duplicate_sample_id",
        "empty_benchmark",
        "image_placeholder_mismatch",
        "missing_image",
        "schema_error",
        "unsafe_image_path",
    }
    observed = set(report["summary"]["critical_by_code"])
    if observed.intersection(technical):
        raise SyntheticHarnessError("synthetic source has a technical audit blocker")


def _write_config(path: Path, config: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a project dependency
        raise SyntheticHarnessError("PyYAML is required to write synthetic_config.yaml") from exc
    header = "# " + " | ".join(SCOPE_LABELS) + "\n"
    text = header + yaml.safe_dump(config, allow_unicode=False, sort_keys=False)
    write_text_new(path, text)


def generate(
    repo_root_value: str | Path,
    run_dir: str | Path,
    sample_count: int = 24,
    seed: int = 20260719,
    base_revision: str = BASE_MODEL_REVISION,
    adapter_revision: str = ADAPTER_REVISION,
) -> dict[str, Any]:
    """Build one atomic synthetic-only run directory and return its safe summary."""

    root = repo_root(repo_root_value)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise SyntheticHarnessError("sample-count must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SyntheticHarnessError("seed must be an integer")
    for label, revision in (("base revision", base_revision), ("adapter revision", adapter_revision)):
        if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
            raise SyntheticHarnessError(f"{label} must be a 40-character commit SHA")
    run_relative, destination = resolve_run(root, run_dir, require_exists=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite run directory: {destination}")
    parent_relative = run_relative.parent
    if parent_relative != Path("."):
        for parent in (root / parent_relative, *(root / parent_relative).parents):
            if parent == root.parent:
                break
            if parent.exists() and parent.is_symlink():
                raise SyntheticHarnessError("run-dir parent traverses a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary run directory already exists: {temporary}")

    try:
        temporary.mkdir()
        source = temporary / "synthetic_source"
        image_dir = source / "assets" / "images"
        image_dir.mkdir(parents=True)
        rng = random.Random(seed)
        values: list[int] = []
        run_text = run_relative.as_posix()
        while len(values) < sample_count:
            candidate = rng.randrange(1_000_000_000, 9_999_999_999)
            if candidate not in values and str(candidate) not in run_text:
                values.append(candidate)

        rows: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        key_records: list[dict[str, Any]] = []
        for index, value in enumerate(values, start=1):
            sample_id = f"syn-{index:04d}"
            paper_id = f"paper:synthetic-{index:04d}"
            image_relative = f"assets/images/syn-{index:04d}.png"
            marker = _panel_marker(index, seed)
            image_path = source / image_relative
            _render_chart(image_path, index=index, seed=seed, value=value, marker=marker)
            image_hash = sha256_file(image_path)
            row = {
                **scope_metadata(),
                "sample_id": sample_id,
                "benchmark_version": "synthetic_chart_benchmark_v1",
                "task_family": "synthetic_chart_exact_value",
                "language": "en",
                "split": "synthetic_exploratory",
                "topic": "synthetic_chart_reading",
                "case_id": f"synthetic-case-{index:04d}",
                "stratum": "easy_control",
                "primary_endpoint": False,
                "paper_title": f"Synthetic chart panel {marker}",
                "paper_id": paper_id,
                "year": "2026",
                "evidence_kind": "synthetic_chart",
                "page_hint": f"Synthetic panel {marker}",
                "model_task_prompt": _PROMPT,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only the requested complete SciReason JSON object.",
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": _PROMPT}, {"type": "image"}],
                    },
                ],
                "images": [image_relative],
                "generation_target_schema": {
                    "type": "object",
                    "required": [
                        "answer",
                        "evidence_used",
                        "visual_facts",
                        "temporal_facts",
                        "uncertainty",
                        "missing_evidence",
                    ],
                },
                "split_provenance": {
                    "paper_holdout": True,
                    "source_holdout": True,
                    "creator_holdout": True,
                    "training_overlap_checked": True,
                    "source_document_id": f"synthetic-source-{index:04d}",
                    "creator_group_id": "synthetic-generator-only",
                },
            }
            rows.append(row)
            provenance.append(
                {
                    **scope_metadata(),
                    "sample_id": sample_id,
                    "paper_id": paper_id,
                    "images": [
                        {
                            "image_path": image_relative,
                            "sha256": image_hash,
                            "page": "synthetic-panel",
                            "locator": f"Synthetic panel {marker}",
                            "source_url": f"https://top-papers.example/synthetic/{sample_id}",
                            "license": "CC0-1.0",
                            "citation": "Synthetic chart generated locally; no external source.",
                            "verified_by": ["synthetic-verifier-a", "synthetic-verifier-b"],
                            **scope_metadata(),
                        }
                    ],
                }
            )
            key_records.append(
                {
                    **scope_metadata(),
                    "sample_id": sample_id,
                    "paper_id": paper_id,
                    "expected_answer": f"VALUE={value}",
                    "original_image_sha256": image_hash,
                }
            )

        write_jsonl_new(source / "data" / "benchmark.jsonl", rows)
        write_jsonl_new(source / "article_image_sources.jsonl", provenance)
        config = _config(run_relative, sample_count, seed, base_revision.lower(), adapter_revision.lower())
        _write_config(temporary / "synthetic_config.yaml", config)
        _validate_generated_source(root, source, rows, provenance, config)

        source_inventory = inventory(source)
        manifest_base: dict[str, Any] = {
            **scope_metadata(),
            "artifact_version": GENERATOR_VERSION,
            "generator_version": GENERATOR_VERSION,
            "run_dir": run_relative.as_posix(),
            "sample_count": sample_count,
            "seed": seed,
            "source_inventory": source_inventory,
            "source_inventory_sha256": json_sha256(source_inventory),
            "synthetic_config_sha256": sha256_file(temporary / "synthetic_config.yaml"),
            "key_record_count": sample_count,
        }
        binding = generator_binding(manifest_base)
        manifest_base["generator_manifest_binding"] = binding
        for record in key_records:
            record["generator_manifest_binding"] = binding
        write_jsonl_new(temporary / "answer_key.jsonl", key_records)
        manifest = {
            **manifest_base,
            "answer_key_sha256": sha256_file(temporary / "answer_key.jsonl"),
            "answer_key_location": "LOCAL_ONLY_OUTSIDE_SYNTHETIC_SOURCE",
        }
        write_json_new(temporary / "generator_manifest.json", manifest)

        # The source text and source filenames may never disclose any panel value.
        source_text = "\n".join(
            [
                (source / "data" / "benchmark.jsonl").read_text(encoding="utf-8"),
                (source / "article_image_sources.jsonl").read_text(encoding="utf-8"),
            ]
        )
        source_names = "\n".join(path.name for path in source.rglob("*") if path.is_file())
        for value in values:
            number = str(value)
            if number in source_text or number in source_names:
                raise SyntheticHarnessError("generated answer value leaked into source text or filename")

        # os.rename is non-overwriting on the supported Windows workflow.
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    verified = verify_generated_run(root, run_relative)
    return {
        **scope_metadata(),
        "run_dir": run_relative.as_posix(),
        "sample_count": verified["manifest"]["sample_count"],
        "generator_manifest": str(destination / "generator_manifest.json"),
        "synthetic_config": str(destination / "synthetic_config.yaml"),
        "answer_key": str(destination / "answer_key.jsonl"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", required=True, help="Safe relative path below --repo-root.")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--base-revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--adapter-revision", default=ADAPTER_REVISION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = generate(
            args.repo_root,
            safe_relative_path(args.run_dir, "run-dir"),
            args.sample_count,
            args.seed,
            args.base_revision,
            args.adapter_revision,
        )
    except (OSError, ValueError, SyntheticHarnessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
