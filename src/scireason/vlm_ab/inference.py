# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resumable, deterministic inference for paired Qwen3-VL evaluation.

The module intentionally imports only the Python standard library at import time.
PyTorch, Transformers, PEFT, and qwen-vl-utils are loaded only when the
``transformers`` backend or an explicit loader is used.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from .paths import resolve_dataset_file


ARTIFACT_VERSION = 1
ENGINE_VERSION = "3"
DEFAULT_CONDITIONS = ("original", "text_only", "shuffled_images")
SUPPORTED_CONDITIONS = frozenset(DEFAULT_CONDITIONS)
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})
_FENCED_JSON = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_ASSISTANT_ROLES = frozenset({"assistant", "bot", "gpt", "model"})
_IMAGE_FIELDS = ("images", "image_paths", "evidence_images", "image", "image_path")
_IMAGE_REFERENCE_FIELDS = (
    "image",
    "image_path",
    "path",
    "file",
    "file_name",
    "filename",
    "image_url",
    "url",
)
_OUTPUT_FIELDS = frozenset(
    {
        "record_version",
        "sample_id",
        "paper_id",
        "condition",
        "arm",
        "backend",
        "status",
        "error",
        "runtime_seconds",
        "raw_response",
        "parsed_response",
        "parse_valid",
        "schema_valid",
        "schema_errors",
        "input_image_hashes",
        "input_image_count",
        "shuffle_source_paper_id",
        "condition_input_fingerprint",
        "protocol_fingerprint",
        "config_fingerprint",
    }
)


class InferenceError(RuntimeError):
    """Base error for inference setup, input, and resume failures."""


class InferenceInputError(InferenceError, ValueError):
    """Raised when benchmark input is malformed or unsafe."""


class InferenceConfigurationError(InferenceError, ValueError):
    """Raised for invalid or incompatible inference configuration."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json_decoder() -> json.JSONDecoder:
    return json.JSONDecoder(
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def _strict_json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def parse_json_response(response: Any) -> Any | None:
    """Extract the first strict JSON value from a model response.

    Plain JSON, fenced JSON, and a JSON object/array surrounded by prose are
    accepted. Duplicate object keys and non-finite numbers are rejected.
    ``None`` is returned when no strict JSON value can be decoded.
    """

    if isinstance(response, (dict, list)):
        try:
            return _strict_json_loads(_strict_json_text(response))
        except (TypeError, ValueError):
            return None
    if isinstance(response, bytes):
        try:
            response = response.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(response, str) or not response.strip():
        return None

    text = response.strip()
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in _FENCED_JSON.finditer(text))
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _strict_json_loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    decoder = _json_decoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        return value
    return None


def _scireason_schema_errors(response: Any) -> list[str]:
    if not isinstance(response, Mapping):
        return ["response must be a JSON object"]

    errors: list[str] = []
    answer = response.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("answer must be a non-empty string")
    evidence = response.get("evidence_used")
    if not isinstance(evidence, list):
        errors.append("evidence_used must be an array")
    else:
        for index, item in enumerate(evidence):
            if not isinstance(item, Mapping):
                errors.append(f"evidence_used[{index}] must be an object")
                continue
            for field in ("kind", "locator", "description"):
                if not isinstance(item.get(field), str):
                    errors.append(f"evidence_used[{index}].{field} must be a string")
    for field in ("visual_facts", "temporal_facts", "missing_evidence"):
        value = response.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{field} must be an array of strings")
    if response.get("uncertainty") not in {"low", "medium", "high"}:
        errors.append("uncertainty must be low, medium, or high")
    return errors


def validate_scireason_response(response: Any) -> bool:
    """Return whether a parsed response satisfies the Task 3 benchmark schema."""

    return not _scireason_schema_errors(response)


def _canonical_data(value: Any, *, label: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InferenceConfigurationError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise InferenceConfigurationError(f"{label} object keys must be strings")
            result[key] = _canonical_data(child, label=f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_data(child, label=f"{label}[{index}]") for index, child in enumerate(value)
        ]
    raise InferenceConfigurationError(
        f"{label} must contain only JSON-compatible configuration values"
    )


def _fingerprint(value: Any) -> str:
    text = _strict_json_text(_canonical_data(value))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identifier(value: Any, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise InferenceInputError(f"{field} must be a non-empty identifier")
    text = str(value)
    if not text or text != text.strip() or "\x00" in text:
        raise InferenceInputError(f"{field} must be a non-empty, trimmed identifier")
    return text


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise InferenceConfigurationError("limit must be a non-negative integer")
    if limit < 0:
        raise InferenceConfigurationError("limit must be a non-negative integer")
    return limit


def _materialize_benchmark(
    benchmark: Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    limit = _validate_limit(limit)
    if isinstance(benchmark, Mapping):
        source: Iterable[tuple[Any, Any]] = benchmark.items()
        keyed = True
    else:
        source = enumerate(benchmark)
        keyed = False

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, raw in source:
        if limit is not None and len(rows) >= limit:
            break
        if not isinstance(raw, Mapping):
            raise InferenceInputError("each benchmark row must be an object")
        row = copy.deepcopy(dict(raw))
        embedded = row.get("sample_id", row.get("id"))
        if keyed:
            sample_id = _identifier(key, "sample_id")
            if embedded is not None and _identifier(embedded, "sample_id") != sample_id:
                raise InferenceInputError("benchmark key conflicts with row sample_id")
        else:
            sample_id = _identifier(embedded, "sample_id")
        if sample_id in seen:
            raise InferenceInputError(f"duplicate benchmark sample_id: {sample_id}")
        seen.add(sample_id)
        row["sample_id"] = sample_id
        rows.append(row)
    return rows


def _paper_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    candidates = [row.get("paper_id"), row.get("paper")]
    if isinstance(metadata, Mapping):
        candidates.extend((metadata.get("paper_id"), metadata.get("paper")))
    for value in candidates:
        if value not in (None, ""):
            return _identifier(value, "paper_id")
    return _identifier(row["sample_id"], "sample_id")


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return _strict_json_text(_canonical_data(value, label="message content"))
    except InferenceConfigurationError as exc:
        raise InferenceInputError(str(exc)) from exc


def _prompt_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw: Any = row.get("messages")
    if raw is None:
        raw = row.get("prompt_chat")
    if raw is None:
        raw = row.get("prompt")

    if isinstance(raw, list) and all(
        isinstance(message, Mapping) and ("role" in message or "from" in message) for message in raw
    ):
        source = raw
    elif raw is not None:
        source = [{"role": "user", "content": raw}]
    else:
        value = next(
            (
                row[field]
                for field in ("question", "instruction", "task", "claim")
                if row.get(field) not in (None, "", [], {})
            ),
            None,
        )
        if value is None:
            raise InferenceInputError(
                f"benchmark sample {row['sample_id']!r} has no messages or prompt"
            )
        source = [{"role": "user", "content": _text_content(value)}]

    messages: list[dict[str, Any]] = []
    for index, raw_message in enumerate(source):
        if not isinstance(raw_message, Mapping):
            raise InferenceInputError(f"messages[{index}] must be an object")
        message = copy.deepcopy(dict(raw_message))
        role = str(message.get("role", message.get("from", "user"))).strip().lower()
        if role in _ASSISTANT_ROLES:
            continue
        if role == "human":
            role = "user"
        if not role or "\x00" in role:
            raise InferenceInputError(f"messages[{index}].role is invalid")
        content = message.get("content", message.get("value", ""))
        message.pop("from", None)
        message.pop("value", None)
        message["role"] = role
        message["content"] = copy.deepcopy(content)
        messages.append(message)
    if not messages:
        raise InferenceInputError(
            f"benchmark sample {row['sample_id']!r} has no non-assistant prompt messages"
        )
    return messages


def _row_image_references(row: Mapping[str, Any]) -> list[Any]:
    for field in _IMAGE_FIELDS:
        raw = row.get(field)
        if raw in (None, "", []):
            continue
        if isinstance(raw, (str, os.PathLike, Mapping)):
            return [raw]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return list(raw)
        raise InferenceInputError(f"benchmark {field} must be a path or an array of paths")
    return []


def _reference_value(reference: Any) -> str:
    while isinstance(reference, Mapping):
        for field in _IMAGE_REFERENCE_FIELDS:
            if reference.get(field) not in (None, ""):
                reference = reference[field]
                break
        else:
            raise InferenceInputError("image object has no local path")
    if not isinstance(reference, (str, os.PathLike)):
        raise InferenceInputError("image reference must be a local filesystem path")
    value = os.fspath(reference)
    if not value or "\x00" in value:
        raise InferenceInputError("image path must be non-empty and must not contain NUL")
    return value


def _path_from_reference(reference: Any) -> Path:
    value = _reference_value(reference)
    direct_path = Path(value)
    if direct_path.is_absolute():
        return direct_path
    parsed = urlparse(value)
    if parsed.scheme:
        if parsed.scheme.lower() != "file":
            raise InferenceInputError("image references must be local paths or file URIs")
        if parsed.netloc not in ("", "localhost"):
            raise InferenceInputError("remote file URI authorities are not allowed")
        return Path(url2pathname(unquote(parsed.path)))
    return Path(value)


def _resolve_image(reference: Any, root: Path) -> Path:
    candidate = _path_from_reference(reference)
    try:
        resolved = resolve_dataset_file(root, candidate)
    except (OSError, RuntimeError) as exc:
        raise InferenceInputError(f"evidence image does not exist: {candidate}") from exc
    except ValueError as exc:
        raise InferenceInputError(f"evidence image escapes dataset_root: {candidate}") from exc
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise InferenceInputError(f"unsupported evidence image type: {candidate}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_item(path: Path, hash_cache: dict[Path, str]) -> dict[str, str]:
    if path not in hash_cache:
        hash_cache[path] = _file_sha256(path)
    return {"uri": str(path), "sha256": hash_cache[path]}


def _is_image_block(block: Any) -> bool:
    if not isinstance(block, Mapping):
        return False
    block_type = str(block.get("type") or "").strip().lower()
    return block_type in {"image", "image_url"} or "image" in block


def _block_image_reference(block: Mapping[str, Any]) -> Any | None:
    for field in _IMAGE_REFERENCE_FIELDS:
        if block.get(field) not in (None, ""):
            return block[field]
    return None


def _canonical_image_block(item: Mapping[str, str]) -> dict[str, str]:
    return {"type": "image", "image": item["uri"]}


def _inject_images(
    messages: list[dict[str, Any]], image_items: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    if not image_items:
        return messages
    image_blocks = [_canonical_image_block(item) for item in image_items]
    target = next(
        (message for message in messages if str(message.get("role")).lower() == "user"),
        None,
    )
    if target is None:
        messages.append({"role": "user", "content": image_blocks})
        return messages
    content = target.get("content", "")
    if isinstance(content, list):
        target["content"] = image_blocks + content
    elif isinstance(content, Mapping):
        target["content"] = image_blocks + [copy.deepcopy(dict(content))]
    elif content in (None, ""):
        target["content"] = image_blocks
    else:
        target["content"] = image_blocks + [{"type": "text", "text": _text_content(content)}]
    return messages


def _resolve_message_images(
    messages: list[dict[str, Any]],
    top_level_references: Sequence[Any],
    root: Path,
    hash_cache: dict[Path, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    unused_top = [
        (_resolve_image(reference, root), reference) for reference in top_level_references
    ]
    resolved_messages = copy.deepcopy(messages)
    used: list[dict[str, str]] = []

    for message_index, message in enumerate(resolved_messages):
        content = message.get("content")
        if isinstance(content, Mapping):
            blocks: list[Any] = [content]
        elif isinstance(content, list):
            blocks = content
        else:
            continue
        normalized: list[Any] = []
        for block_index, block in enumerate(blocks):
            if not _is_image_block(block):
                normalized.append(block)
                continue
            assert isinstance(block, Mapping)
            reference = _block_image_reference(block)
            if reference is None:
                if not unused_top:
                    raise InferenceInputError(
                        f"messages[{message_index}].content[{block_index}] has an "
                        "image placeholder without a matching top-level image"
                    )
                path, _ = unused_top.pop(0)
            else:
                path = _resolve_image(reference, root)
                matching = next(
                    (index for index, (candidate, _) in enumerate(unused_top) if candidate == path),
                    None,
                )
                if matching is not None:
                    unused_top.pop(matching)
            item = _image_item(path, hash_cache)
            used.append(item)
            normalized.append(_canonical_image_block(item))
        message["content"] = normalized

    remaining = [_image_item(path, hash_cache) for path, _ in unused_top]
    if remaining:
        _inject_images(resolved_messages, remaining)
        used = remaining + used
    return resolved_messages, used


def _replace_message_images(
    messages: Sequence[Mapping[str, Any]], image_items: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    result = copy.deepcopy(list(messages))
    next_image = 0
    for message in result:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        replaced: list[Any] = []
        for block in content:
            if not _is_image_block(block):
                replaced.append(block)
                continue
            if next_image < len(image_items):
                replaced.append(_canonical_image_block(image_items[next_image]))
                next_image += 1
        message["content"] = replaced
    if next_image < len(image_items):
        _inject_images(result, image_items[next_image:])
    return result


def _portable_message_fingerprint_data(
    messages: Sequence[Mapping[str, Any]], image_items: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    result = copy.deepcopy(list(messages))
    image_index = 0
    for message in result:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        normalized: list[Any] = []
        for block in content:
            if not _is_image_block(block):
                normalized.append(block)
                continue
            if image_index >= len(image_items):
                raise InferenceInputError("message image count exceeds resolved evidence images")
            normalized.append({"type": "image", "sha256": image_items[image_index]["sha256"]})
            image_index += 1
        message["content"] = normalized
    if image_index != len(image_items):
        raise InferenceInputError("resolved evidence image count exceeds message images")
    return result


def _normalize_conditions(conditions: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(conditions, str):
        raw = [conditions]
    else:
        raw = list(conditions)
    if not raw:
        raise InferenceConfigurationError("at least one condition is required")
    normalized: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            raise InferenceConfigurationError("conditions must be strings")
        condition = value.strip().lower()
        if condition not in SUPPORTED_CONDITIONS:
            raise InferenceConfigurationError(
                f"unsupported condition {value!r}; expected one of {sorted(SUPPORTED_CONDITIONS)}"
            )
        if condition in normalized:
            raise InferenceConfigurationError(f"duplicate condition: {condition}")
        normalized.append(condition)
    return tuple(normalized)


def _derived_random(seed: int, purpose: str) -> random.Random:
    material = f"{seed}\x00{purpose}".encode("utf-8")
    number = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return random.Random(number)


def _paper_derangement(
    paper_images: Mapping[str, Mapping[str, Mapping[str, str]]], seed: int
) -> dict[str, str]:
    papers = sorted(paper_id for paper_id, images in paper_images.items() if images)
    if len(papers) < 2:
        return {paper_id: paper_id for paper_id in papers}
    order = list(papers)
    _derived_random(seed, "paper-image-derangement").shuffle(order)
    return {paper_id: order[(index + 1) % len(order)] for index, paper_id in enumerate(order)}


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InferenceConfigurationError("seed must be an integer")
    return seed


def build_condition_rows(
    benchmark: Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    dataset_root: str | os.PathLike[str],
    conditions: Sequence[str] | str = DEFAULT_CONDITIONS,
    seed: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build safe, JSON-compatible prompts for every sample and condition.

    Assistant answer turns are removed. Every local image is resolved beneath
    ``dataset_root``, hashed, and represented in messages by an absolute ``file:``
    URI. The shuffled condition uses one deterministic donor paper for every
    image-bearing paper; with fewer than two such papers it retains the original
    images because a derangement is impossible.
    """

    seed = _validate_seed(seed)
    normalized_conditions = _normalize_conditions(conditions)
    try:
        root = Path(dataset_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InferenceInputError(f"dataset_root does not exist: {dataset_root}") from exc
    if not root.is_dir():
        raise InferenceInputError("dataset_root must be a directory")

    benchmark_rows = _materialize_benchmark(benchmark, limit)
    hash_cache: dict[Path, str] = {}
    prepared: list[dict[str, Any]] = []
    paper_images: dict[str, dict[str, dict[str, str]]] = {}
    for row in benchmark_rows:
        sample_id = row["sample_id"]
        paper_id = _paper_id(row)
        messages, image_items = _resolve_message_images(
            _prompt_messages(row),
            _row_image_references(row),
            root,
            hash_cache,
        )
        prepared.append(
            {
                "sample_id": sample_id,
                "paper_id": paper_id,
                "messages": messages,
                "images": image_items,
            }
        )
        aggregate = paper_images.setdefault(paper_id, {})
        for item in image_items:
            aggregate.setdefault(item["uri"], item)

    donors = _paper_derangement(paper_images, seed)
    output: list[dict[str, Any]] = []
    for row in prepared:
        for condition in normalized_conditions:
            source_paper: str | None
            if condition == "original":
                image_items = row["images"]
                source_paper = row["paper_id"] if image_items else None
            elif condition == "text_only":
                image_items = []
                source_paper = None
            else:
                donor = donors.get(row["paper_id"])
                if row["images"] and donor is not None:
                    donor_items = [paper_images[donor][uri] for uri in sorted(paper_images[donor])]
                    image_items = [
                        donor_items[index % len(donor_items)] for index in range(len(row["images"]))
                    ]
                    source_paper = donor
                else:
                    image_items = []
                    source_paper = None
            condition_messages = _replace_message_images(row["messages"], image_items)
            condition_fingerprint = _fingerprint(
                {
                    "sample_id": row["sample_id"],
                    "paper_id": row["paper_id"],
                    "condition": condition,
                    "messages": _portable_message_fingerprint_data(condition_messages, image_items),
                    "input_image_hashes": [item["sha256"] for item in image_items],
                    "shuffle_source_paper_id": source_paper,
                }
            )
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "paper_id": row["paper_id"],
                    "condition": condition,
                    "messages": condition_messages,
                    "image_uris": [item["uri"] for item in image_items],
                    "input_image_hashes": [item["sha256"] for item in image_items],
                    "shuffle_source_paper_id": source_paper,
                    "condition_input_fingerprint": condition_fingerprint,
                }
            )
    return output


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InferenceConfigurationError(f"{label} must be an object")
    return dict(value)


def _config_text(
    config: Mapping[str, Any], names: Sequence[str], label: str, *, required: bool = True
) -> str | None:
    for name in names:
        value = config.get(name)
        if value not in (None, ""):
            if not isinstance(value, (str, os.PathLike)):
                raise InferenceConfigurationError(f"{label} must be a non-empty string")
            text = os.fspath(value).strip()
            if not text or "\x00" in text:
                raise InferenceConfigurationError(f"{label} must be a non-empty string")
            return text
    if required:
        raise InferenceConfigurationError(f"{label} is required")
    return None


def _merged_kwargs(
    section: Mapping[str, Any],
    section_names: Sequence[str],
    config: Mapping[str, Any],
    config_names: Sequence[str],
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in section_names:
        if name in section:
            result.update(_mapping(section[name], f"{label}.{name}"))
    for name in config_names:
        if name in config:
            result.update(_mapping(config[name], f"{label}.{name}"))
    return result


def _import_optional(module_name: str, requirement: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise InferenceConfigurationError(
            f"{requirement} is required for the transformers backend"
        ) from exc


def _resolve_torch_dtype(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    value = model_kwargs.get("torch_dtype")
    if not isinstance(value, str) or value in {"auto", ""}:
        return model_kwargs
    name = value.removeprefix("torch.")
    if name not in {"float16", "bfloat16", "float32", "float64"}:
        raise InferenceConfigurationError(f"unsupported torch_dtype: {value}")
    torch = _import_optional("torch", "torch")
    result = dict(model_kwargs)
    result["torch_dtype"] = getattr(torch, name)
    return result


def _assert_no_reserved_kwargs(kwargs: Mapping[str, Any], reserved: set[str], label: str) -> None:
    conflict = sorted(reserved & set(kwargs))
    if conflict:
        raise InferenceConfigurationError(
            f"{label} must not override explicit arguments: {conflict}"
        )


def _reject_local_model_path(identifier: str, label: str) -> None:
    try:
        if Path(identifier).expanduser().exists():
            raise InferenceConfigurationError(
                f"{label} resolves to a local path instead of the pinned Hub repository"
            )
    except OSError as exc:
        raise InferenceConfigurationError(f"cannot validate {label} as a Hub identifier") from exc


def _active_adapter_names(model: Any) -> list[str]:
    value = getattr(model, "active_adapters", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(model, "active_adapter", None)
        if callable(value):
            value = value()
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def load_transformers_model(arm_name: str, arm_config: Mapping[str, Any]) -> Any:
    """Explicitly load a pinned Qwen3-VL base and, for ``tuned``, a pinned adapter.

    The base always comes from
    ``Qwen3VLForConditionalGeneration.from_pretrained``. The tuned arm then calls
    ``PeftModel.from_pretrained`` on that concrete base model. Both revisions are
    mandatory; implicit Transformers adapter auto-loading is never used.
    """

    if not isinstance(arm_name, str) or arm_name.strip().lower() not in {"base", "tuned"}:
        raise InferenceConfigurationError("arm_name must be 'base' or 'tuned'")
    arm = arm_name.strip().lower()
    config = _mapping(arm_config, "arm_config")
    base_section = _mapping(config.get("base_model"), "arm_config.base_model")

    base_id = _config_text(
        {**config, **base_section},
        ("base_model_id", "model_id", "id", "repo_id"),
        "base model id",
    )
    base_revision = _config_text(
        {**config, **base_section},
        ("base_revision", "model_revision", "revision"),
        "base model revision",
    )
    _reject_local_model_path(base_id, "base model id")
    model_kwargs = _merged_kwargs(
        base_section,
        ("settings", "kwargs", "model_kwargs"),
        config,
        ("model_kwargs",),
        "arm_config",
    )
    _assert_no_reserved_kwargs(model_kwargs, {"revision"}, "model_kwargs")
    model_kwargs = _resolve_torch_dtype(model_kwargs)

    transformers = _import_optional("transformers", "transformers")
    model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_class is None:
        raise InferenceConfigurationError(
            "transformers.Qwen3VLForConditionalGeneration is unavailable"
        )
    base_model = model_class.from_pretrained(
        base_id,
        revision=base_revision,
        **model_kwargs,
    )

    if arm == "base":
        if config.get("adapter_id") or config.get("adapter_model_id") or config.get("adapter"):
            raise InferenceConfigurationError("the base arm must not configure an adapter")
        model = base_model
    else:
        adapter_section = _mapping(config.get("adapter"), "arm_config.adapter")
        combined_adapter = {**config, **adapter_section}
        adapter_id = _config_text(
            combined_adapter,
            ("adapter_id", "adapter_model_id", "id", "repo_id", "path"),
            "adapter id",
        )
        adapter_revision = _config_text(
            combined_adapter,
            ("adapter_revision", "revision"),
            "adapter revision",
        )
        _reject_local_model_path(adapter_id, "adapter id")
        adapter_name = (
            _config_text(
                combined_adapter,
                ("adapter_name",),
                "adapter name",
                required=False,
            )
            or "default"
        )
        adapter_kwargs = _merged_kwargs(
            adapter_section,
            ("settings", "kwargs", "adapter_kwargs"),
            config,
            ("adapter_kwargs",),
            "arm_config.adapter",
        )
        _assert_no_reserved_kwargs(
            adapter_kwargs,
            {"revision", "adapter_name", "is_trainable"},
            "adapter_kwargs",
        )
        peft = _import_optional("peft", "peft")
        peft_class = getattr(peft, "PeftModel", None)
        if peft_class is None:
            raise InferenceConfigurationError("peft.PeftModel is unavailable")
        model = peft_class.from_pretrained(
            base_model,
            adapter_id,
            revision=adapter_revision,
            adapter_name=adapter_name,
            is_trainable=False,
            **adapter_kwargs,
        )
        if not isinstance(model, peft_class):
            raise InferenceConfigurationError("adapter loader did not return a PeftModel")
        active_adapters = _active_adapter_names(model)
        if adapter_name not in active_adapters:
            raise InferenceConfigurationError("requested PEFT adapter is not active")
        peft_config = getattr(model, "peft_config", None)
        if not isinstance(peft_config, Mapping):
            raise InferenceConfigurationError("PEFT model has no adapter config mapping")
        if adapter_name not in peft_config:
            raise InferenceConfigurationError("active PEFT adapter has no config")
        if peft_config[adapter_name] is None:
            raise InferenceConfigurationError("active PEFT adapter config is empty")

    eval_method = getattr(model, "eval", None)
    if not callable(eval_method):
        raise InferenceConfigurationError("loaded model does not provide eval()")
    eval_method()
    return model


def load_transformers_processor(processor_config: Mapping[str, Any]) -> Any:
    """Load the shared processor from an explicit ID and revision."""

    config = _mapping(processor_config, "processor_config")
    processor_id = _config_text(
        config,
        ("processor_id", "model_id", "id", "repo_id"),
        "processor id",
    )
    revision = _config_text(
        config,
        ("processor_revision", "revision"),
        "processor revision",
    )
    _reject_local_model_path(processor_id, "processor id")
    kwargs = _merged_kwargs(
        {},
        (),
        config,
        ("settings", "kwargs", "processor_kwargs"),
        "processor_config",
    )
    _assert_no_reserved_kwargs(kwargs, {"revision"}, "processor settings")
    transformers = _import_optional("transformers", "transformers")
    processor_class = getattr(transformers, "AutoProcessor", None)
    if processor_class is None:
        raise InferenceConfigurationError("transformers.AutoProcessor is unavailable")
    return processor_class.from_pretrained(processor_id, revision=revision, **kwargs)


def _generation_settings(generation_config: Mapping[str, Any] | None) -> dict[str, Any]:
    config = _mapping(generation_config, "generation_config")
    result = {"do_sample": False, "max_new_tokens": 256}
    result.update(config)
    max_tokens = result.get("max_new_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise InferenceConfigurationError("max_new_tokens must be a positive integer")
    if not isinstance(result.get("do_sample"), bool):
        raise InferenceConfigurationError("do_sample must be boolean")
    return _canonical_data(result, label="generation_config")


def _seed_transformers(seed: int, torch: Any, transformers: Any) -> None:
    normalized = seed % (2**63 - 1)
    random.seed(normalized)
    torch.manual_seed(normalized)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "manual_seed_all", None)):
        cuda.manual_seed_all(normalized)
    set_seed = getattr(transformers, "set_seed", None)
    if callable(set_seed):
        set_seed(normalized)


def _row_seed(seed: int, sample_id: str, condition: str) -> int:
    digest = hashlib.sha256(f"{seed}\x00{sample_id}\x00{condition}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class _TransformersRunner:
    def __init__(
        self,
        model: Any,
        processor: Any,
        processor_config: Mapping[str, Any],
        generation_config: Mapping[str, Any],
        seed: int,
    ) -> None:
        self.model = model
        self.processor = processor
        self.generation_config = dict(generation_config)
        self.seed = seed
        self.torch = _import_optional("torch", "torch")
        self.transformers = _import_optional("transformers", "transformers")
        config = _mapping(processor_config, "processor_config")
        self.call_kwargs = _mapping(config.get("call_settings"), "processor call_settings")
        self.chat_kwargs = _mapping(
            config.get("chat_template_settings"), "processor chat_template_settings"
        )
        _assert_no_reserved_kwargs(
            self.call_kwargs,
            {"add_generation_prompt", "return_dict", "return_tensors", "tokenize"},
            "processor call_settings",
        )
        _assert_no_reserved_kwargs(
            self.chat_kwargs,
            {"add_generation_prompt", "return_dict", "return_tensors", "tokenize"},
            "processor chat_template_settings",
        )

    def __call__(self, row: Mapping[str, Any]) -> str:
        messages = row["messages"]
        _seed_transformers(
            _row_seed(self.seed, str(row["sample_id"]), str(row["condition"])),
            self.torch,
            self.transformers,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_kwargs,
            **self.call_kwargs,
        )

        device = getattr(self.model, "device", None)
        if device is not None and callable(getattr(inputs, "to", None)):
            inputs = inputs.to(device)
        input_ids = inputs["input_ids"]
        input_length = int(input_ids.shape[-1])
        self.model.eval()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **self.generation_config)
        sequences = getattr(generated, "sequences", generated)
        generated_suffix = sequences[:, input_length:]
        decoded = self.processor.batch_decode(
            generated_suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            return ""
        return str(decoded[0])


def _mock_response(row: Mapping[str, Any], arm_name: str, seed: int) -> str:
    token = hashlib.sha256(
        (
            f"{seed}\x00{arm_name}\x00{row['sample_id']}\x00{row['condition']}\x00"
            f"{row['condition_input_fingerprint']}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    return _strict_json_text(
        {
            "answer": f"Deterministic mock answer {token}.",
            "evidence_used": [],
            "visual_facts": [],
            "temporal_facts": [],
            "uncertainty": "high",
            "missing_evidence": ["Mock backend does not inspect evidence."],
        }
    )


def manifest_path_for(output_jsonl: str | os.PathLike[str]) -> Path:
    """Return the sidecar manifest path used for an output JSONL file."""

    return Path(f"{os.fspath(output_jsonl)}.manifest.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_environment(backend: str) -> dict[str, Any]:
    packages = {
        name: version
        for name, distribution in (
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("peft", "peft"),
            ("qwen_vl_utils", "qwen-vl-utils"),
            ("numpy", "numpy"),
            ("torchvision", "torchvision"),
            ("accelerate", "accelerate"),
            ("pillow", "Pillow"),
        )
        if (version := _package_version(distribution)) is not None
    }
    result: dict[str, Any] = {
        "backend": backend,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }
    if backend == "transformers":
        try:
            import torch  # type: ignore

            result["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
            if torch.cuda.is_available():
                result["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
                result["gpu_count"] = torch.cuda.device_count()
        except (ImportError, RuntimeError):
            pass
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise InferenceConfigurationError(f"refusing to overwrite symlink: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise InferenceConfigurationError(f"temporary manifest path already exists: {temporary}")
    text = _strict_json_text(payload, indent=2) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise InferenceConfigurationError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InferenceConfigurationError(f"{label} must be a JSON object: {path}")
    return value


def _validate_existing_row(
    row: Any,
    line_number: int,
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    arm_name: str,
    backend: str,
    protocol_fingerprint: str,
    config_fingerprint: str,
) -> tuple[str, str]:
    if not isinstance(row, dict):
        raise InferenceConfigurationError(f"output line {line_number} must be a JSON object")
    if set(row) != _OUTPUT_FIELDS:
        missing = sorted(_OUTPUT_FIELDS - set(row))
        extra = sorted(set(row) - _OUTPUT_FIELDS)
        raise InferenceConfigurationError(
            f"output line {line_number} has incompatible fields; missing={missing}, extra={extra}"
        )
    if isinstance(row["record_version"], bool) or row["record_version"] != ARTIFACT_VERSION:
        raise InferenceConfigurationError(f"output line {line_number} has wrong record_version")
    if row["config_fingerprint"] != config_fingerprint:
        raise InferenceConfigurationError(
            f"output config fingerprint mismatch on line {line_number}"
        )
    if row["protocol_fingerprint"] != protocol_fingerprint:
        raise InferenceConfigurationError(
            f"output protocol fingerprint mismatch on line {line_number}"
        )
    if row["arm"] != arm_name or row["backend"] != backend:
        raise InferenceConfigurationError(f"output line {line_number} belongs to another run")
    if not isinstance(row["sample_id"], str) or not isinstance(row["condition"], str):
        raise InferenceConfigurationError(f"output line {line_number} has an invalid key")
    key = (row["sample_id"], row["condition"])
    expected_row = expected.get(key)
    if expected_row is None:
        raise InferenceConfigurationError(f"output line {line_number} has an unexpected key: {key}")
    if row["paper_id"] != expected_row["paper_id"]:
        raise InferenceConfigurationError(f"output line {line_number} has the wrong paper_id")
    if row["condition_input_fingerprint"] != expected_row["condition_input_fingerprint"]:
        raise InferenceConfigurationError(
            f"output input fingerprint mismatch on line {line_number}"
        )
    if row["input_image_hashes"] != expected_row["input_image_hashes"]:
        raise InferenceConfigurationError(f"output image hashes mismatch on line {line_number}")
    if (
        isinstance(row["input_image_count"], bool)
        or not isinstance(row["input_image_count"], int)
        or row["input_image_count"] != len(expected_row["input_image_hashes"])
    ):
        raise InferenceConfigurationError(f"output image count mismatch on line {line_number}")
    if row["shuffle_source_paper_id"] != expected_row["shuffle_source_paper_id"]:
        raise InferenceConfigurationError(f"output shuffle source mismatch on line {line_number}")
    if row["status"] not in {"success", "error"}:
        raise InferenceConfigurationError(f"output line {line_number} has an invalid status")
    runtime = row["runtime_seconds"]
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)):
        raise InferenceConfigurationError(f"output line {line_number} has invalid runtime")
    if not math.isfinite(float(runtime)) or runtime < 0:
        raise InferenceConfigurationError(f"output line {line_number} has invalid runtime")
    if not isinstance(row["parse_valid"], bool) or not isinstance(row["schema_valid"], bool):
        raise InferenceConfigurationError(f"output line {line_number} has invalid flags")
    if not isinstance(row["schema_errors"], list) or not all(
        isinstance(value, str) for value in row["schema_errors"]
    ):
        raise InferenceConfigurationError(f"output line {line_number} has schema errors")
    if row["status"] == "success" and not isinstance(row["raw_response"], str):
        raise InferenceConfigurationError(f"output line {line_number} has no raw response")
    if row["status"] == "success" and row["error"] is not None:
        raise InferenceConfigurationError(f"output line {line_number} has a spurious error")
    if row["status"] == "success":
        reparsed = parse_json_response(row["raw_response"])
        expected_parse_valid = reparsed is not None
        expected_schema_errors = _scireason_schema_errors(reparsed)
        if reparsed != row["parsed_response"] or row["parse_valid"] != expected_parse_valid:
            raise InferenceConfigurationError(
                f"output line {line_number} has inconsistent parsed response"
            )
        if row["schema_errors"] != expected_schema_errors or row["schema_valid"] != (
            expected_parse_valid and not expected_schema_errors
        ):
            raise InferenceConfigurationError(
                f"output line {line_number} has inconsistent schema flags"
            )
    else:
        error = row["error"]
        if not isinstance(error, Mapping):
            raise InferenceConfigurationError(f"output line {line_number} has no error object")
        if set(error) != {"type", "message"} or not all(
            isinstance(error[field], str) for field in ("type", "message")
        ):
            raise InferenceConfigurationError(
                f"output line {line_number} has an invalid error object"
            )
        if (
            row["raw_response"] is not None
            or row["parsed_response"] is not None
            or row["parse_valid"]
            or row["schema_valid"]
        ):
            raise InferenceConfigurationError(
                f"output line {line_number} has inconsistent error fields"
            )
    return key


def _load_existing_rows(
    path: Path,
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    arm_name: str,
    backend: str,
    protocol_fingerprint: str,
    config_fingerprint: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    if path.is_symlink():
        raise InferenceConfigurationError(f"refusing to use symlinked output: {path}")
    if not path.is_file():
        raise InferenceConfigurationError(f"output_jsonl is not a file: {path}")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise InferenceConfigurationError(
                        f"output line {line_number} is blank or incomplete"
                    )
                try:
                    row = _strict_json_loads(line)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise InferenceConfigurationError(
                        f"invalid JSONL output on line {line_number}: {exc}"
                    ) from exc
                key = _validate_existing_row(
                    row,
                    line_number,
                    expected,
                    arm_name,
                    backend,
                    protocol_fingerprint,
                    config_fingerprint,
                )
                if key in result:
                    raise InferenceConfigurationError(f"duplicate output key: {key}")
                result[key] = row
    except UnicodeError as exc:
        raise InferenceConfigurationError(f"output_jsonl is not valid UTF-8: {path}") from exc
    return result


def _encoded_row(row: Mapping[str, Any]) -> bytes:
    if set(row) != _OUTPUT_FIELDS:
        raise AssertionError("internal output row does not match the strict schema")
    return (_strict_json_text(row) + "\n").encode("utf-8")


def _append_row(handle: Any, encoded_row: bytes) -> None:
    handle.write(encoded_row)
    handle.flush()
    os.fsync(handle.fileno())


def _checkpoint_state(path: Path, completed_rows: int) -> tuple[str, bytes] | None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for _ in range(completed_rows):
            line = handle.readline()
            if not line:
                return None
            digest.update(line)
        return digest.hexdigest(), handle.read()


def _row_counts(rows: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
    success = 0
    errors = 0
    for row in rows:
        if row.get("status") == "success":
            success += 1
        else:
            errors += 1
    return success, errors


def _manifest_payload(
    *,
    previous: Mapping[str, Any] | None,
    output_path: Path,
    arm_name: str,
    arm_config: Mapping[str, Any],
    processor_config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    conditions: Sequence[str],
    seed: int,
    limit: int | None,
    backend: str,
    input_fingerprint: str,
    protocol_fingerprint: str,
    config_fingerprint: str,
    experiment_fingerprint: str | None,
    result_scope: str,
    runtime_environment: Mapping[str, Any],
    expected_rows: int,
    completed_rows: int,
    successful_rows: int,
    error_rows: int,
    status: str,
    started_at: str,
    pending_row: Mapping[str, str] | None = None,
    output_sha256: str | None = None,
    last_error: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    created_at = previous.get("created_at") if previous else now
    resume_count = int(previous.get("resume_count", 0)) if previous else 0
    return {
        "artifact_version": ARTIFACT_VERSION,
        "engine": "scireason.vlm_ab.inference",
        "engine_version": ENGINE_VERSION,
        "status": status,
        "created_at": created_at,
        "started_at": started_at,
        "updated_at": now,
        "finished_at": now if status in {"complete", "failed"} else None,
        "resume_count": resume_count,
        "output_jsonl": str(output_path),
        "output_sha256": output_sha256,
        "arm": arm_name,
        "backend": backend,
        "experiment_fingerprint": experiment_fingerprint,
        "result_scope": result_scope,
        "conditions": list(conditions),
        "seed": seed,
        "limit": limit,
        "expected_rows": expected_rows,
        "completed_rows": completed_rows,
        "successful_rows": successful_rows,
        "error_rows": error_rows,
        "input_fingerprint": input_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "shared_config_fingerprint": protocol_fingerprint,
        "config_fingerprint": config_fingerprint,
        "fingerprints": {
            "input": input_fingerprint,
            "protocol": protocol_fingerprint,
            "config": config_fingerprint,
            "processor": _fingerprint(processor_config),
            "generation": _fingerprint(generation_config),
            "arm": _fingerprint(arm_config),
        },
        "configuration": {
            "arm": _canonical_data(arm_config, label="arm_config"),
            "processor": _canonical_data(processor_config, label="processor_config"),
            "generation": _canonical_data(generation_config, label="generation_config"),
        },
        "runtime": dict(runtime_environment),
        "pending_row": dict(pending_row) if pending_row else None,
        "last_error": dict(last_error) if last_error else None,
    }


def run_inference(
    benchmark: Mapping[Any, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    dataset_root: str | os.PathLike[str],
    arm_name: str,
    arm_config: Mapping[str, Any],
    processor_config: Mapping[str, Any] | None,
    generation_config: Mapping[str, Any] | None,
    output_jsonl: str | os.PathLike[str],
    conditions: Sequence[str] | str = DEFAULT_CONDITIONS,
    seed: int = 0,
    limit: int | None = None,
    backend: str = "transformers",
    experiment_fingerprint: str | None = None,
    result_scope: str = "unspecified",
) -> dict[str, Any]:
    """Run or resume one arm of a paired Qwen3-VL evaluation.

    One strict JSONL record is durably appended per sample/condition. Existing
    records are validated and skipped by their unique ``(sample_id, condition)``
    key. Any manifest or row fingerprint mismatch aborts without changing the
    output. Run the base and tuned arms sequentially with the same shared
    processor/generation arguments and separate output files.
    """

    seed = _validate_seed(seed)
    limit = _validate_limit(limit)
    if not isinstance(arm_name, str) or arm_name.strip().lower() not in {"base", "tuned"}:
        raise InferenceConfigurationError("arm_name must be 'base' or 'tuned'")
    arm = arm_name.strip().lower()
    if not isinstance(backend, str) or backend.strip().lower() not in {"transformers", "mock"}:
        raise InferenceConfigurationError("backend must be 'transformers' or 'mock'")
    backend_name = backend.strip().lower()
    runtime_environment = _runtime_environment(backend_name)
    if experiment_fingerprint is not None and not re.fullmatch(
        r"[0-9a-f]{64}", experiment_fingerprint
    ):
        raise InferenceConfigurationError("experiment_fingerprint must be a SHA256 digest")
    if result_scope not in {
        "unspecified",
        "publication_candidate",
        "exploratory_not_for_publication",
    }:
        raise InferenceConfigurationError("result_scope is invalid")
    arm_settings = _mapping(arm_config, "arm_config")
    processor_settings = _mapping(processor_config, "processor_config")
    generation_settings = _generation_settings(generation_config)
    forbidden = {
        "processor",
        "processor_config",
        "processor_id",
        "processor_revision",
        "processor_settings",
        "generation",
        "generation_config",
        "generation_kwargs",
        "do_sample",
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "num_beams",
        "repetition_penalty",
    }
    if forbidden & set(arm_settings):
        raise InferenceConfigurationError(
            "processor and generation settings must be shared, not stored in arm_config"
        )
    normalized_conditions = _normalize_conditions(conditions)
    if backend_name == "transformers":
        _config_text(
            processor_settings,
            ("processor_id", "model_id", "id", "repo_id"),
            "processor id",
        )
        _config_text(
            processor_settings,
            ("processor_revision", "revision"),
            "processor revision",
        )

    condition_rows = build_condition_rows(
        benchmark,
        dataset_root,
        normalized_conditions,
        seed,
        limit,
    )
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in condition_rows:
        key = (row["sample_id"], row["condition"])
        if key in expected:
            raise InferenceInputError(f"duplicate condition key: {key}")
        expected[key] = row

    fingerprint_rows = sorted(
        condition_rows,
        key=lambda row: (row["sample_id"], row["condition"]),
    )
    input_fingerprint = _fingerprint(
        [
            {
                "sample_id": row["sample_id"],
                "paper_id": row["paper_id"],
                "condition": row["condition"],
                "condition_input_fingerprint": row["condition_input_fingerprint"],
            }
            for row in fingerprint_rows
        ]
    )
    protocol_fingerprint = _fingerprint(
        {
            "engine_version": ENGINE_VERSION,
            "backend": backend_name,
            "processor_config": processor_settings,
            "generation_config": generation_settings,
            "conditions": list(normalized_conditions),
            "seed": seed,
            "limit": limit,
            "input_fingerprint": input_fingerprint,
        }
    )
    config_fingerprint = _fingerprint(
        {
            "protocol_fingerprint": protocol_fingerprint,
            "arm": arm,
            "arm_config": arm_settings,
        }
    )

    output_path = Path(output_jsonl)
    if output_path.exists() and output_path.is_symlink():
        raise InferenceConfigurationError(f"refusing to use symlinked output: {output_path}")
    if output_path.exists() and not output_path.is_file():
        raise InferenceConfigurationError(f"output_jsonl is not a file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.absolute()
    sidecar_path = manifest_path_for(output_path)
    if sidecar_path.exists() and sidecar_path.is_symlink():
        raise InferenceConfigurationError(f"refusing to use symlinked manifest: {sidecar_path}")
    if output_path.exists() and not sidecar_path.exists():
        raise InferenceConfigurationError("prediction output exists without its integrity manifest")

    previous_manifest: dict[str, Any] | None = None
    checkpoint_rows: int | None = None
    recovered_pending_key: tuple[str, str] | None = None
    if sidecar_path.exists():
        previous_manifest = _read_json_object(sidecar_path, "runtime manifest")
        previous_fingerprint = (previous_manifest.get("fingerprints") or {}).get("config")
        if previous_fingerprint != config_fingerprint:
            raise InferenceConfigurationError("runtime manifest config fingerprint mismatch")
        if previous_manifest.get("experiment_fingerprint") != experiment_fingerprint:
            raise InferenceConfigurationError("runtime manifest experiment fingerprint mismatch")
        if previous_manifest.get("result_scope") != result_scope:
            raise InferenceConfigurationError("runtime manifest result scope mismatch")
        if previous_manifest.get("runtime") != runtime_environment:
            raise InferenceConfigurationError("runtime environment changed during resume")
        previous_output_hash = previous_manifest.get("output_sha256")
        completed_rows = previous_manifest.get("completed_rows")
        if (
            isinstance(completed_rows, bool)
            or not isinstance(completed_rows, int)
            or completed_rows < 0
        ):
            raise InferenceConfigurationError("runtime manifest completed_rows is invalid")
        checkpoint_rows = completed_rows
        pending_row = previous_manifest.get("pending_row")
        pending_key: tuple[str, str] | None = None
        pending_hash: str | None = None
        if pending_row is not None:
            if not isinstance(pending_row, Mapping) or set(pending_row) != {
                "sample_id",
                "condition",
                "sha256",
            }:
                raise InferenceConfigurationError("runtime manifest pending_row is invalid")
            sample_id = pending_row.get("sample_id")
            condition = pending_row.get("condition")
            pending_hash = pending_row.get("sha256")
            if (
                not isinstance(sample_id, str)
                or not isinstance(condition, str)
                or not isinstance(pending_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", pending_hash)
            ):
                raise InferenceConfigurationError("runtime manifest pending_row is invalid")
            pending_key = (sample_id, condition)
            if pending_key not in expected:
                raise InferenceConfigurationError("runtime manifest pending_row is unexpected")
        if output_path.exists():
            checkpoint = _checkpoint_state(output_path, completed_rows)
            if checkpoint is None:
                raise InferenceConfigurationError(
                    "prediction bytes changed after the previous manifest update"
                )
            checkpoint_hash, tail = checkpoint
            expected_checkpoint_hash = previous_output_hash
            if expected_checkpoint_hash is None and completed_rows == 0:
                expected_checkpoint_hash = hashlib.sha256(b"").hexdigest()
            if (
                not isinstance(expected_checkpoint_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_hash)
                or not hmac.compare_digest(checkpoint_hash, expected_checkpoint_hash)
            ):
                raise InferenceConfigurationError(
                    "prediction bytes changed after the previous manifest update"
                )
            if tail:
                if (
                    pending_hash is None
                    or tail.count(b"\n") != 1
                    or not tail.endswith(b"\n")
                    or not hmac.compare_digest(hashlib.sha256(tail).hexdigest(), pending_hash)
                ):
                    raise InferenceConfigurationError(
                        "prediction bytes changed after the previous manifest update"
                    )
                recovered_pending_key = pending_key
        elif previous_output_hash is not None or completed_rows != 0:
            raise InferenceConfigurationError(
                "prediction output disappeared after the previous manifest update"
            )
        previous_manifest["resume_count"] = int(previous_manifest.get("resume_count", 0)) + 1

    existing = _load_existing_rows(
        output_path,
        expected,
        arm,
        backend_name,
        protocol_fingerprint,
        config_fingerprint,
    )
    if checkpoint_rows is not None:
        recovered_rows = 1 if recovered_pending_key is not None else 0
        if len(existing) != checkpoint_rows + recovered_rows:
            raise InferenceConfigurationError(
                "prediction rows differ from the previous manifest checkpoint"
            )
        if recovered_pending_key is not None and recovered_pending_key not in existing:
            raise InferenceConfigurationError(
                "recovered prediction row differs from its write-ahead commitment"
            )
    pending = [
        row for row in condition_rows if (row["sample_id"], row["condition"]) not in existing
    ]
    started_at = _utc_now()
    success_count, error_count = _row_counts(existing.values())
    output_hasher = hashlib.sha256()
    if output_path.exists():
        with output_path.open("rb") as existing_output:
            for block in iter(lambda: existing_output.read(1024 * 1024), b""):
                output_hasher.update(block)
    manifest = _manifest_payload(
        previous=previous_manifest,
        output_path=output_path,
        arm_name=arm,
        arm_config=arm_settings,
        processor_config=processor_settings,
        generation_config=generation_settings,
        conditions=normalized_conditions,
        seed=seed,
        limit=limit,
        backend=backend_name,
        input_fingerprint=input_fingerprint,
        protocol_fingerprint=protocol_fingerprint,
        config_fingerprint=config_fingerprint,
        experiment_fingerprint=experiment_fingerprint,
        result_scope=result_scope,
        runtime_environment=runtime_environment,
        expected_rows=len(condition_rows),
        completed_rows=len(existing),
        successful_rows=success_count,
        error_rows=error_count,
        status="running" if pending else "complete",
        started_at=started_at,
        output_sha256=output_hasher.hexdigest() if output_path.exists() else None,
    )
    _write_json_atomic(sidecar_path, manifest)

    if not pending:
        return {
            "output_jsonl": str(output_path),
            "manifest_path": str(sidecar_path),
            "config_fingerprint": config_fingerprint,
            "protocol_fingerprint": protocol_fingerprint,
            "expected_rows": len(condition_rows),
            "existing_rows": len(existing),
            "written_rows": 0,
            "successful_rows": success_count,
            "error_rows": error_count,
            "output_sha256": manifest["output_sha256"],
        }

    random.seed(seed)
    try:
        if backend_name == "mock":

            def runner(row: Mapping[str, Any]) -> str:
                return _mock_response(row, arm, seed)
        else:
            model = load_transformers_model(arm, arm_settings)
            processor = load_transformers_processor(processor_settings)
            runner = _TransformersRunner(
                model,
                processor,
                processor_settings,
                generation_settings,
                seed,
            )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        failed_manifest = _manifest_payload(
            previous=manifest,
            output_path=output_path,
            arm_name=arm,
            arm_config=arm_settings,
            processor_config=processor_settings,
            generation_config=generation_settings,
            conditions=normalized_conditions,
            seed=seed,
            limit=limit,
            backend=backend_name,
            input_fingerprint=input_fingerprint,
            protocol_fingerprint=protocol_fingerprint,
            config_fingerprint=config_fingerprint,
            experiment_fingerprint=experiment_fingerprint,
            result_scope=result_scope,
            runtime_environment=runtime_environment,
            expected_rows=len(condition_rows),
            completed_rows=len(existing),
            successful_rows=success_count,
            error_rows=error_count,
            status="failed",
            started_at=started_at,
            output_sha256=output_hasher.hexdigest() if output_path.exists() else None,
            last_error=error,
        )
        _write_json_atomic(sidecar_path, failed_manifest)
        raise

    written = 0
    with output_path.open("ab") as handle:
        for condition_row in pending:
            started = time.perf_counter()
            error: dict[str, str] | None = None
            raw_response: str | None = None
            parsed_response: Any | None = None
            parse_valid = False
            schema_errors: list[str]
            try:
                raw_response = runner(condition_row)
                if not isinstance(raw_response, str):
                    raw_response = str(raw_response)
                parsed_response = parse_json_response(raw_response)
                parse_valid = parsed_response is not None
                schema_errors = _scireason_schema_errors(parsed_response)
                status = "success"
            except Exception as exc:
                status = "error"
                error = {"type": type(exc).__name__, "message": str(exc)}
                schema_errors = ["generation failed"]
            runtime_seconds = time.perf_counter() - started
            output_row = {
                "record_version": ARTIFACT_VERSION,
                "sample_id": condition_row["sample_id"],
                "paper_id": condition_row["paper_id"],
                "condition": condition_row["condition"],
                "arm": arm,
                "backend": backend_name,
                "status": status,
                "error": error,
                "runtime_seconds": runtime_seconds,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
                "parse_valid": parse_valid,
                "schema_valid": parse_valid and not schema_errors,
                "schema_errors": schema_errors,
                "input_image_hashes": condition_row["input_image_hashes"],
                "input_image_count": len(condition_row["input_image_hashes"]),
                "shuffle_source_paper_id": condition_row["shuffle_source_paper_id"],
                "condition_input_fingerprint": condition_row["condition_input_fingerprint"],
                "protocol_fingerprint": protocol_fingerprint,
                "config_fingerprint": config_fingerprint,
            }
            encoded_row = _encoded_row(output_row)
            pending_manifest = _manifest_payload(
                previous=manifest,
                output_path=output_path,
                arm_name=arm,
                arm_config=arm_settings,
                processor_config=processor_settings,
                generation_config=generation_settings,
                conditions=normalized_conditions,
                seed=seed,
                limit=limit,
                backend=backend_name,
                input_fingerprint=input_fingerprint,
                protocol_fingerprint=protocol_fingerprint,
                config_fingerprint=config_fingerprint,
                experiment_fingerprint=experiment_fingerprint,
                result_scope=result_scope,
                runtime_environment=runtime_environment,
                expected_rows=len(condition_rows),
                completed_rows=len(existing),
                successful_rows=success_count,
                error_rows=error_count,
                status="running",
                started_at=started_at,
                pending_row={
                    "sample_id": condition_row["sample_id"],
                    "condition": condition_row["condition"],
                    "sha256": hashlib.sha256(encoded_row).hexdigest(),
                },
                output_sha256=output_hasher.hexdigest(),
            )
            _write_json_atomic(sidecar_path, pending_manifest)
            manifest = pending_manifest
            _append_row(handle, encoded_row)
            output_hasher.update(encoded_row)
            key = (condition_row["sample_id"], condition_row["condition"])
            existing[key] = output_row
            written += 1
            success_count, error_count = _row_counts(existing.values())
            manifest = _manifest_payload(
                previous=manifest,
                output_path=output_path,
                arm_name=arm,
                arm_config=arm_settings,
                processor_config=processor_settings,
                generation_config=generation_settings,
                conditions=normalized_conditions,
                seed=seed,
                limit=limit,
                backend=backend_name,
                input_fingerprint=input_fingerprint,
                protocol_fingerprint=protocol_fingerprint,
                config_fingerprint=config_fingerprint,
                experiment_fingerprint=experiment_fingerprint,
                result_scope=result_scope,
                runtime_environment=runtime_environment,
                expected_rows=len(condition_rows),
                completed_rows=len(existing),
                successful_rows=success_count,
                error_rows=error_count,
                status="running",
                started_at=started_at,
                output_sha256=output_hasher.hexdigest(),
            )
            _write_json_atomic(sidecar_path, manifest)

    final_manifest = _manifest_payload(
        previous=manifest,
        output_path=output_path,
        arm_name=arm,
        arm_config=arm_settings,
        processor_config=processor_settings,
        generation_config=generation_settings,
        conditions=normalized_conditions,
        seed=seed,
        limit=limit,
        backend=backend_name,
        input_fingerprint=input_fingerprint,
        protocol_fingerprint=protocol_fingerprint,
        config_fingerprint=config_fingerprint,
        experiment_fingerprint=experiment_fingerprint,
        result_scope=result_scope,
        runtime_environment=runtime_environment,
        expected_rows=len(condition_rows),
        completed_rows=len(existing),
        successful_rows=success_count,
        error_rows=error_count,
        status="complete",
        started_at=started_at,
        output_sha256=output_hasher.hexdigest(),
    )
    _write_json_atomic(sidecar_path, final_manifest)
    return {
        "output_jsonl": str(output_path),
        "manifest_path": str(sidecar_path),
        "config_fingerprint": config_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "expected_rows": len(condition_rows),
        "existing_rows": len(existing) - written,
        "written_rows": written,
        "successful_rows": success_count,
        "error_rows": error_count,
        "output_sha256": final_manifest["output_sha256"],
    }


__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_CONDITIONS",
    "InferenceConfigurationError",
    "InferenceError",
    "InferenceInputError",
    "build_condition_rows",
    "load_transformers_model",
    "load_transformers_processor",
    "manifest_path_for",
    "parse_json_response",
    "run_inference",
    "validate_scireason_response",
]
