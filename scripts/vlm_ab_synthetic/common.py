#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared integrity helpers for the synthetic exploratory VLM A/B harness.

This module deliberately lives outside the publication evaluation source tree.
It only verifies and labels synthetic exploratory artifacts; it never changes
the publication pipeline contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


SCOPE_LABELS = (
    "EXPLORATORY_NOT_FOR_PUBLICATION",
    "SYNTHETIC_DATA_ONLY",
    "NO_HUMAN_REVIEW",
    "NO_PUBLICATION_OR_SUPERIORITY_CLAIM",
)
RESULT_SCOPE = "exploratory_not_for_publication"
GENERATOR_VERSION = 1
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BASE_MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
ADAPTER_ID = "top-papers/Qwen3-VL-8B-Instruct-scireason"
ADAPTER_REVISION = "45936868f4b2acdbbc4245044137072411fd11cc"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ANSWER_RE = re.compile(r"^VALUE=([0-9]+)$")


class SyntheticHarnessError(RuntimeError):
    """Raised when a synthetic-only harness artifact is malformed or unsafe."""


def scope_metadata() -> dict[str, Any]:
    """Return the fixed non-publication labels carried by synthetic artifacts."""

    return {
        "scope_labels": list(SCOPE_LABELS),
        "synthetic_data_only": True,
        "human_review_performed": False,
        "publication_or_superiority_claim_allowed": False,
    }


def scope_text() -> str:
    return " | ".join(SCOPE_LABELS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SyntheticHarnessError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SyntheticHarnessError(f"{label} must be a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise SyntheticHarnessError(f"{label} has a blank JSONL line {line_number}")
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_object_without_duplicate_keys,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise SyntheticHarnessError(
                        f"{label} has invalid JSON on line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise SyntheticHarnessError(f"{label} line {line_number} must be an object")
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        raise SyntheticHarnessError(f"cannot read {label}: {path}: {exc}") from exc
    return rows


def safe_relative_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if (
        not str(value)
        or path.is_absolute()
        or path.drive
        or path.anchor
        or ".." in path.parts
        or path == Path(".")
    ):
        raise SyntheticHarnessError(f"{label} must be a non-empty safe relative path")
    return path


def _assert_no_symlink_components(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise SyntheticHarnessError(f"path traverses a symlink: {relative.as_posix()}")


def repo_root(path: str | Path) -> Path:
    try:
        root = Path(path).resolve(strict=True)
    except OSError as exc:
        raise SyntheticHarnessError(f"repository root does not exist: {path}") from exc
    if not root.is_dir() or not (root / "pyproject.toml").is_file():
        raise SyntheticHarnessError("repo-root must contain pyproject.toml")
    return root


def resolve_run(root: Path, run_dir: str | Path, *, require_exists: bool = True) -> tuple[Path, Path]:
    relative = safe_relative_path(run_dir, "run-dir")
    _assert_no_symlink_components(root, relative)
    candidate = root / relative
    if require_exists:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise SyntheticHarnessError(f"run directory does not exist: {candidate}") from exc
        if not resolved.is_dir() or candidate.is_symlink():
            raise SyntheticHarnessError(f"run directory must be a regular directory: {candidate}")
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise SyntheticHarnessError("run directory escapes repo-root") from exc
        return relative, resolved
    return relative, candidate


def require_scope_labels(value: Mapping[str, Any], label: str) -> None:
    labels = value.get("scope_labels")
    if not isinstance(labels, list) or set(labels) != set(SCOPE_LABELS) or len(labels) != len(SCOPE_LABELS):
        raise SyntheticHarnessError(f"{label} must contain exactly the synthetic scope labels")
    if value.get("synthetic_data_only") is not True:
        raise SyntheticHarnessError(f"{label} must be marked synthetic_data_only")
    if value.get("human_review_performed") is not False:
        raise SyntheticHarnessError(f"{label} must state that no human review was performed")
    if value.get("publication_or_superiority_claim_allowed") is not False:
        raise SyntheticHarnessError(f"{label} must forbid publication or superiority claims")


def safe_walk_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise SyntheticHarnessError(f"expected a regular directory: {root}")
    paths = sorted(root.rglob("*"), key=lambda path: path.as_posix())
    if any(path.is_symlink() for path in paths):
        raise SyntheticHarnessError(f"symlinks are forbidden below {root}")
    return [path for path in paths if path.is_file()]


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in safe_walk_files(root)
    ]


def verify_inventory(root: Path, entries: Any, label: str) -> None:
    if not isinstance(entries, list) or not entries:
        raise SyntheticHarnessError(f"{label} inventory must be a non-empty list")
    declared: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise SyntheticHarnessError(f"{label} inventory has a malformed entry")
        relative = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(relative, str):
            raise SyntheticHarnessError(f"{label} inventory path is invalid")
        safe_relative_path(relative, f"{label} inventory path")
        if (
            relative in declared
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise SyntheticHarnessError(f"{label} inventory entry is invalid: {relative!r}")
        declared[relative] = entry
    actual = inventory(root)
    actual_by_path = {entry["path"]: entry for entry in actual}
    if set(declared) != set(actual_by_path):
        raise SyntheticHarnessError(f"{label} inventory paths do not match files on disk")
    for relative, entry in declared.items():
        actual_entry = actual_by_path[relative]
        if entry["sha256"] != actual_entry["sha256"] or entry["size"] != actual_entry["size"]:
            raise SyntheticHarnessError(f"{label} inventory hash mismatch: {relative}")


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, value: Any) -> None:
    """Write a new JSON file through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary artifact already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_regular_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise SyntheticHarnessError(f"source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite payload file: {destination}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from scireason.vlm_ab.config import load_experiment_config
    except ImportError as exc:  # pragma: no cover - repository layout error
        raise SyntheticHarnessError("cannot import the existing VLM config validator") from exc
    return load_experiment_config(path)


def load_synthetic_config(root: Path, run: Path, run_relative: Path) -> dict[str, Any]:
    config_path = run / "synthetic_config.yaml"
    if not config_path.is_file() or config_path.is_symlink():
        raise SyntheticHarnessError("generated run has no regular synthetic_config.yaml")
    config = _load_config(config_path, root)
    require_scope_labels(config, "synthetic configuration")
    if config.get("experiment", {}).get("output_dir") != run_relative.as_posix():
        raise SyntheticHarnessError("synthetic config output_dir does not match run-dir")
    experiment = config.get("experiment", {})
    identity = " ".join(str(experiment.get(key, "")) for key in ("id", "public_id")).lower()
    if "synthetic" not in identity or "exploratory" not in identity:
        raise SyntheticHarnessError("synthetic config identity is not explicitly exploratory")
    return config


def _answer_number(expected: Any, label: str) -> str:
    if not isinstance(expected, str):
        raise SyntheticHarnessError(f"{label} expected_answer must be a string")
    match = _ANSWER_RE.fullmatch(expected)
    if match is None:
        raise SyntheticHarnessError(f"{label} expected_answer must use exact VALUE=<integer> form")
    return match.group(1)


def _assert_no_answer_leakage(
    source: Path,
    values: Iterable[str],
) -> None:
    text_files = [
        source / "data" / "benchmark.jsonl",
        source / "article_image_sources.jsonl",
    ]
    texts: list[str] = []
    for path in text_files:
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            raise SyntheticHarnessError(f"cannot scan synthetic model-facing text: {path}") from exc
    filenames = "\n".join(path.name for path in safe_walk_files(source))
    for number in values:
        number_pattern = re.compile(rf"(?<![0-9]){re.escape(number)}(?![0-9])")
        for text in texts:
            if f"VALUE={number}" in text or number_pattern.search(text):
                raise SyntheticHarnessError("synthetic expected answer leaked into model-facing text")
        if f"VALUE={number}" in filenames or number_pattern.search(filenames):
            raise SyntheticHarnessError("synthetic expected answer leaked into a source filename")


def generator_binding(manifest: Mapping[str, Any]) -> str:
    material = {
        "artifact_version": manifest.get("artifact_version"),
        "generator_version": manifest.get("generator_version"),
        "sample_count": manifest.get("sample_count"),
        "seed": manifest.get("seed"),
        "source_inventory": manifest.get("source_inventory"),
    }
    return json_sha256(material)


def verify_generated_run(root_value: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Verify the local synthetic dataset, key, config, and generator binding."""

    root = repo_root(root_value)
    run_relative, run = resolve_run(root, run_dir)
    manifest_path = run / "generator_manifest.json"
    manifest = read_json_object(manifest_path, "generator manifest")
    require_scope_labels(manifest, "generator manifest")
    if manifest.get("artifact_version") != GENERATOR_VERSION:
        raise SyntheticHarnessError("generator manifest has an unsupported artifact version")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise SyntheticHarnessError("generator manifest has an unsupported generator version")
    if manifest.get("run_dir") != run_relative.as_posix():
        raise SyntheticHarnessError("generator manifest run_dir does not match the requested run")
    sample_count = manifest.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise SyntheticHarnessError("generator manifest sample_count must be an integer >= 2")
    if isinstance(manifest.get("seed"), bool) or not isinstance(manifest.get("seed"), int):
        raise SyntheticHarnessError("generator manifest seed must be an integer")

    source = run / "synthetic_source"
    verify_inventory(source, manifest.get("source_inventory"), "synthetic source")
    if manifest.get("source_inventory_sha256") != json_sha256(manifest["source_inventory"]):
        raise SyntheticHarnessError("generator manifest source inventory digest is invalid")
    if manifest.get("generator_manifest_binding") != generator_binding(manifest):
        raise SyntheticHarnessError("generator manifest binding is invalid")

    config_path = run / "synthetic_config.yaml"
    if manifest.get("synthetic_config_sha256") != sha256_file(config_path):
        raise SyntheticHarnessError("generator manifest synthetic config hash is invalid")
    config = load_synthetic_config(root, run, run_relative)

    key_path = run / "answer_key.jsonl"
    if not key_path.is_file() or key_path.is_symlink():
        raise SyntheticHarnessError("generated run has no regular local answer_key.jsonl")
    if manifest.get("answer_key_sha256") != sha256_file(key_path):
        raise SyntheticHarnessError("generator manifest answer key hash is invalid")
    keys = read_jsonl(key_path, "answer key")
    if len(keys) != sample_count or manifest.get("key_record_count") != sample_count:
        raise SyntheticHarnessError("answer key count does not match the generator manifest")

    benchmark = read_jsonl(source / "data" / "benchmark.jsonl", "synthetic benchmark")
    provenance = read_jsonl(source / "article_image_sources.jsonl", "synthetic provenance")
    if len(benchmark) != sample_count or len(provenance) != sample_count:
        raise SyntheticHarnessError("synthetic source row count does not match the generator manifest")
    benchmark_by_id = {row.get("sample_id"): row for row in benchmark}
    provenance_by_id = {row.get("sample_id"): row for row in provenance}
    if len(benchmark_by_id) != sample_count or len(provenance_by_id) != sample_count:
        raise SyntheticHarnessError("synthetic source has duplicate sample IDs")

    expected_values: list[str] = []
    seen_answers: set[str] = set()
    seen_hashes: set[str] = set()
    for key in keys:
        require_scope_labels(key, "answer key record")
        sample_id = key.get("sample_id")
        paper_id = key.get("paper_id")
        if not isinstance(sample_id, str) or not isinstance(paper_id, str):
            raise SyntheticHarnessError("answer key record lacks a sample_id or paper_id")
        number = _answer_number(key.get("expected_answer"), "answer key record")
        if key.get("generator_manifest_binding") != manifest["generator_manifest_binding"]:
            raise SyntheticHarnessError("answer key record is not bound to this generator manifest")
        image_hash = key.get("original_image_sha256")
        if not isinstance(image_hash, str) or not _SHA256_RE.fullmatch(image_hash):
            raise SyntheticHarnessError("answer key record has an invalid original image hash")
        row = benchmark_by_id.get(sample_id)
        provenance_row = provenance_by_id.get(sample_id)
        if not isinstance(row, dict) or not isinstance(provenance_row, dict):
            raise SyntheticHarnessError("answer key references an unknown synthetic source sample")
        require_scope_labels(row, "synthetic benchmark row")
        require_scope_labels(provenance_row, "synthetic provenance row")
        if row.get("paper_id") != paper_id or provenance_row.get("paper_id") != paper_id:
            raise SyntheticHarnessError("answer key paper mapping differs from synthetic source")
        images = row.get("images")
        provenance_images = provenance_row.get("images")
        if (
            not isinstance(images, list)
            or len(images) != 1
            or not isinstance(images[0], str)
            or not isinstance(provenance_images, list)
            or len(provenance_images) != 1
            or not isinstance(provenance_images[0], dict)
        ):
            raise SyntheticHarnessError("synthetic source must contain exactly one image per sample")
        image_path = source / images[0]
        if not image_path.is_file() or image_path.is_symlink() or sha256_file(image_path) != image_hash:
            raise SyntheticHarnessError("answer key original image hash does not match source bytes")
        if provenance_images[0].get("sha256") != image_hash:
            raise SyntheticHarnessError("synthetic provenance image hash does not match answer key")
        if number in seen_answers or image_hash in seen_hashes:
            raise SyntheticHarnessError("synthetic answer values and image bytes must be unique")
        seen_answers.add(number)
        seen_hashes.add(image_hash)
        expected_values.append(number)
    _assert_no_answer_leakage(source, expected_values)
    return {
        "repo_root": root,
        "run_dir": run_relative,
        "run": run,
        "source": source,
        "config": config,
        "manifest": manifest,
        "keys": keys,
        "benchmark": benchmark,
        "provenance": provenance,
    }


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_REVISION",
    "BASE_MODEL_ID",
    "BASE_MODEL_REVISION",
    "GENERATOR_VERSION",
    "RESULT_SCOPE",
    "SCOPE_LABELS",
    "SyntheticHarnessError",
    "canonical_json",
    "copy_regular_file",
    "generator_binding",
    "inventory",
    "json_sha256",
    "load_synthetic_config",
    "read_json_object",
    "read_jsonl",
    "repo_root",
    "require_scope_labels",
    "resolve_run",
    "safe_relative_path",
    "safe_walk_files",
    "scope_metadata",
    "scope_text",
    "sha256_file",
    "verify_generated_run",
    "verify_inventory",
    "write_json_atomic",
    "write_json_new",
    "write_jsonl_new",
    "write_text_new",
]
