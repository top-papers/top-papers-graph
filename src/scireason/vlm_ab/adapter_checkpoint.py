# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fail-closed inspection for the evaluated plain FP32 LoRA checkpoint."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_TENSORS = 100_000
_LORA_KEY_RE = re.compile(r"^(?P<prefix>.+)\.(?P<kind>lora_A|lora_B)\.weight$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(text: str, *, error_type: type[Exception]) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise error_type(f"adapter safetensors header contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"adapter safetensors header is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise error_type("adapter safetensors header must be a JSON object")
    return value


def inspect_plain_fp32_lora_safetensors(
    path: str | Path,
    *,
    repo_id: str,
    revision: str,
    error_type: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Inspect without materializing tensors and return a content-bound attestation."""

    candidate = Path(path)
    filename = candidate.name
    source = candidate.resolve(strict=True)
    if filename != "adapter_model.safetensors" or not source.is_file():
        raise error_type("the evaluated adapter must use adapter_model.safetensors")
    size = source.stat().st_size
    if size <= 8:
        raise error_type("adapter_model.safetensors is truncated")
    with source.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
        if header_length <= 1 or header_length > min(_MAX_HEADER_BYTES, size - 8):
            raise error_type("adapter safetensors header length is invalid")
        raw_header = handle.read(header_length)
    try:
        header_text = raw_header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise error_type("adapter safetensors header is not UTF-8") from exc
    header = _strict_object(header_text, error_type=error_type)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors or len(tensors) > _MAX_TENSORS:
        raise error_type("adapter safetensors tensor inventory is empty or too large")

    data_size = size - 8 - header_length
    inventory: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    pairs: dict[str, set[str]] = {}
    for key, raw in sorted(tensors.items()):
        match = _LORA_KEY_RE.fullmatch(key)
        if match is None:
            raise error_type(f"adapter checkpoint contains a non-plain-LoRA tensor: {key}")
        if not isinstance(raw, Mapping) or raw.get("dtype") != "F32":
            raise error_type(f"adapter tensor must be native F32: {key}")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not shape
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        ):
            raise error_type(f"adapter tensor has an invalid shape: {key}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        ):
            raise error_type(f"adapter tensor has invalid data offsets: {key}")
        start, end = offsets
        if start < 0 or end <= start or end > data_size:
            raise error_type(f"adapter tensor data offsets are out of bounds: {key}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - start != elements * 4:
            raise error_type(f"adapter F32 tensor byte length differs from its shape: {key}")
        prefix = match.group("prefix")
        pairs.setdefault(prefix, set()).add(match.group("kind"))
        inventory[key] = {"dtype": "F32", "shape": shape, "data_offsets": offsets}
        intervals.append((start, end, key))

    incomplete = sorted(prefix for prefix, kinds in pairs.items() if kinds != {"lora_A", "lora_B"})
    if incomplete:
        raise error_type(f"adapter checkpoint has unpaired LoRA tensors: {incomplete[:5]}")
    cursor = 0
    for start, end, key in sorted(intervals):
        if start != cursor:
            raise error_type(f"adapter tensor data layout has a gap or overlap before {key}")
        cursor = end
    if cursor != data_size:
        raise error_type("adapter tensor data does not cover the complete safetensors payload")

    encoded_inventory = json.dumps(
        inventory,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "artifact_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "filename": filename,
        "size": size,
        "sha256": _sha256(source),
        "tensor_count": len(inventory),
        "lora_module_count": len(pairs),
        "tensor_dtype": "F32",
        "tensor_inventory_sha256": hashlib.sha256(encoded_inventory).hexdigest(),
    }


def verify_plain_fp32_lora_safetensors(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str,
    error_type: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Recompute and compare an adapter checkpoint attestation exactly."""

    actual = inspect_plain_fp32_lora_safetensors(
        path,
        repo_id=repo_id,
        revision=revision,
        error_type=error_type,
    )
    if dict(expected) != actual:
        raise error_type("pinned adapter checkpoint differs from the prepared FP32 LoRA attestation")
    return actual


__all__ = [
    "inspect_plain_fp32_lora_safetensors",
    "verify_plain_fp32_lora_safetensors",
]
