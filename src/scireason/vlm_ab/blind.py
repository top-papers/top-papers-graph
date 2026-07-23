# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deterministic, identity-blind packages for paired VLM review."""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import random
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identities import normalize_identity
from .paths import resolve_dataset_file

ARTIFACT_VERSION = 3
PUBLIC_ASSIGNMENT_FILENAME = "assignments.json"
REVIEW_HTML_FILENAME = "review.html"
OWNER_MAPPING_FILENAME = "owner_only.json"

PREFERENCE_VALUES = frozenset({"left", "right", "tie", "skip"})
ERROR_TAGS = (
    "hallucination",
    "unsupported_claim",
    "wrong_evidence",
    "missed_evidence",
    "wrong_visual",
    "missed_visual",
    "wrong_temporal",
    "incomplete",
    "invalid_format",
    "other",
)
RUBRIC_VERSION = "vlm-ab-paired-v1.0"
RUBRIC = {
    "version": RUBRIC_VERSION,
    "instructions": [
        "Оценивайте только показанные задание, изображения и два ответа. Не используйте внешний поиск.",
        "Работайте независимо: не обсуждайте решения со вторым экспертом до сдачи обоих файлов.",
        "Не пытайтесь определить модель по стилю ответа. Метки left/right не связаны с одной системой.",
        "Фактическая корректность и опора на свидетельства важнее стиля и краткости.",
        "Пустой, оборванный ответ или ошибка генерации являются недостатком ответа, а не причиной skip.",
    ],
    "criteria": {
        "overall_preference": (
            "Итоговое содержательное качество: корректность, полнота, релевантность и выполнение задачи."
        ),
        "evidence_preference": (
            "Корректность использования научных свидетельств и отсутствие неподтвержденных выводов."
        ),
        "visual_preference": (
            "Соответствие утверждений показанным графикам, таблицам и другим изображениям."
        ),
        "temporal_preference": (
            "Корректность временного порядка, динамики, трендов и причинно-временных связей."
        ),
    },
    "preferences": {
        "left": "Левый ответ содержательно лучше по выбранному критерию.",
        "right": "Правый ответ содержательно лучше по выбранному критерию.",
        "tie": "Ответы эквивалентны, различие несущественно или критерий одинаково неприменим.",
        "skip": (
            "Оценка критерия невозможна из-за дефекта задания или пакета. Требуется объяснение."
        ),
    },
    "confidence": {
        "1": "очень низкая",
        "2": "низкая",
        "3": "средняя",
        "4": "высокая",
        "5": "очень высокая",
    },
    "error_tags": {
        "hallucination": "выдуманный факт или объект",
        "unsupported_claim": "утверждение не подтверждено данными",
        "wrong_evidence": "неверно использовано свидетельство",
        "missed_evidence": "пропущено существенное свидетельство",
        "wrong_visual": "неверно прочитано изображение",
        "missed_visual": "пропущена существенная визуальная деталь",
        "wrong_temporal": "ошибка во времени, порядке или тренде",
        "incomplete": "существенно неполный ответ",
        "invalid_format": "нарушен обязательный формат ответа",
        "other": "другая содержательная ошибка, поясненная в комментарии",
    },
}
RUBRIC_SHA256 = hashlib.sha256(
    json.dumps(
        RUBRIC,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

_RESPONSE_FIELDS = (
    "raw_response",
    "response",
    "output",
    "prediction",
    "generated_text",
    "text",
    "result",
)
_SUCCESS_STATUSES = frozenset({"success", "succeeded", "ok", "complete", "completed", "finished"})
_IMAGE_FIELDS = ("images", "image_paths", "evidence_images", "image", "image_path")
_IMAGE_PATH_FIELDS = ("path", "image_path", "file", "file_name", "filename")
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})
_SAFE_TOP_LEVEL_METADATA = (
    "domain",
    "task_family",
    "stratum",
    "paper_title",
    "paper_id",
    "year",
    "evidence_kind",
    "page_hint",
)
_REVIEW_TOP_LEVEL_FIELDS = frozenset(
    {
        "artifact_version",
        "experiment_id",
        "study_fingerprint",
        "rubric_version",
        "rubric_sha256",
        "reviewer_id",
        "package_nonce",
        "independent_review_attestation",
        "assignments",
        "responses",
    }
)
_REVIEW_RESPONSE_FIELDS = frozenset(
    {
        "assignment_id",
        "overall_preference",
        "evidence_preference",
        "visual_preference",
        "temporal_preference",
        "left_error_tags",
        "right_error_tags",
        "confidence",
        "comments",
    }
)
_OWNER_TOP_LEVEL_FIELDS = frozenset(
    {
        "artifact_version",
        "experiment_id",
        "study_fingerprint",
        "assignments",
        "integrity_hmac_sha256",
    }
)
_OWNER_ENTRY_FIELDS = frozenset(
    {
        "assignment_id",
        "reviewer_id",
        "sample_id",
        "condition",
        "left_arm",
        "right_arm",
        "display_position",
        "package_nonce",
    }
)
_MISSING = object()
_MAX_JSON_BYTES = 32 * 1024 * 1024


class BlindReviewError(ValueError):
    """Raised when a blind package cannot be built safely."""


class ReviewValidationError(BlindReviewError):
    """Raised when an exported review violates the review contract."""


@dataclass(frozen=True)
class ReviewerPackage:
    reviewer_id: str
    package_dir: Path
    html_path: Path
    assignment_path: Path
    image_paths: tuple[Path, ...]

    @property
    def public_assignment_path(self) -> Path:
        return self.assignment_path


@dataclass(frozen=True)
class BlindReviewBuild:
    experiment_id: str
    study_fingerprint: str
    output_dir: Path
    owner_mapping_path: Path
    reviewer_packages: dict[str, ReviewerPackage]
    paired_item_count: int
    total_assignment_count: int

    @property
    def owner_key_path(self) -> Path:
        return self.owner_mapping_path

    def for_reviewer(self, reviewer_id: str) -> ReviewerPackage:
        return self.reviewer_packages[reviewer_id]


@dataclass(frozen=True)
class _PairedItem:
    assignment_id: str
    sample_id: str
    condition: str
    task: Any
    metadata: dict[str, Any]
    images: tuple[Path, ...]
    base_response: Any
    tuned_response: Any
    sensitive_values: frozenset[str]


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BlindReviewError(f"{field} must be a non-empty, trimmed string")
    if "\x00" in value:
        raise BlindReviewError(f"{field} must not contain NUL")
    if not normalize_identity(value):
        raise BlindReviewError(f"{field} contains unsafe identifier characters")
    return value


def _json_text(value: Any, *, indent: int | None = None, sort_keys: bool = True) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=sort_keys,
        )
    except (TypeError, ValueError) as exc:
        raise BlindReviewError(f"value is not strict JSON data: {exc}") from exc


def _json_copy(value: Any) -> Any:
    return json.loads(_json_text(value))


def _write_json(path: Path, payload: Any) -> None:
    if path.is_symlink():
        raise BlindReviewError(f"refusing to overwrite symlink: {path}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_json_text(payload, indent=2) + "\n")


def _derived_rng(seed: int | str, purpose: str) -> random.Random:
    material = f"{type(seed).__name__}:{seed}\x00{purpose}".encode("utf-8")
    number = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return random.Random(number)


def _digest_id(prefix: str, *parts: Any, length: int = 24) -> str:
    material = "\x00".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(material).hexdigest()[:length]}"


def _hmac_key(seed: int | str) -> bytes:
    return hashlib.sha256(f"{type(seed).__name__}:{seed}".encode("utf-8")).digest()


def _payload_hmac(payload: Mapping[str, Any], seed: int | str) -> str:
    return hmac.new(
        _hmac_key(seed), _json_text(payload).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _reviewer_package_nonce(seed: int | str, experiment_id: str, reviewer_id: str) -> str:
    material = f"{experiment_id}\x00{reviewer_id}\x00review-package".encode("utf-8")
    return hmac.new(_hmac_key(seed), material, hashlib.sha256).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_key(key: str) -> str:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key.strip())
    return snake_case.lower().replace("-", "_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalized_key(key)
    tokens = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
    banned_tokens = {
        "adapter",
        "annotator",
        "annotation",
        "arm",
        "base",
        "baseline",
        "checkpoint",
        "filename",
        "label",
        "model",
        "models",
        "path",
        "rationale",
        "review",
        "reviewer",
        "truth",
        "tuned",
    }
    if tokens & banned_tokens:
        return True
    sensitive_names = {
        "answer",
        "condition",
        "creator_id",
        "creator_notes",
        "creator_rationale",
        "error_taxonomy",
        "expected_error_modes",
        "expected_error_tags",
        "expected_winner",
        "file",
        "file_name",
        "ground_truth",
        "image_path",
        "image_paths",
        "owner_key",
        "primary_endpoint",
        "reference_answer",
        "run_id",
        "sample_id",
        "seed",
    }
    return (
        normalized in sensitive_names
        or normalized.startswith("expected_")
        or normalized.startswith("model_")
    )


def _is_secret_value_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized == "model_task_prompt":
        return False
    tokens = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
    if tokens & {"adapter", "checkpoint", "model", "models"}:
        return True
    return normalized in {
        "arm_truth",
        "creator_notes",
        "creator_rationale",
        "expected_winner",
        "ground_truth",
        "owner_key",
        "reference_answer",
    }


def _collect_sensitive_values(value: Any, *, under_sensitive_key: bool = False) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_is_sensitive = isinstance(key, str) and _is_secret_value_key(key)
            found.update(
                _collect_sensitive_values(
                    child,
                    under_sensitive_key=under_sensitive_key or key_is_sensitive,
                )
            )
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_collect_sensitive_values(child, under_sensitive_key=under_sensitive_key))
    elif under_sensitive_key and isinstance(value, str) and len(value) >= 4:
        found.add(value)
    return found


def _sanitize_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise BlindReviewError("public metadata keys must be strings")
            if not _is_sensitive_key(key):
                clean[key] = _sanitize_public_value(child)
        return clean
    if isinstance(value, (list, tuple)):
        return [_sanitize_public_value(child) for child in value]
    return _json_copy(value)


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, Mapping)):
        return bool(value)
    return True


def _extract_task(row: Mapping[str, Any], sample_id: str) -> Any:
    raw_messages = row.get("messages")
    if isinstance(raw_messages, list):
        messages: list[dict[str, Any]] = []
        for message in raw_messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or message.get("from") or "").strip().lower()
            if role in {"assistant", "bot", "gpt", "model"}:
                continue
            content = message.get("content", message.get("value"))
            if isinstance(content, list):
                public_content: list[Any] = []
                for block in content:
                    if isinstance(block, Mapping) and str(block.get("type") or "").lower() in {
                        "image",
                        "image_url",
                        "input_image",
                    }:
                        public_content.append({"type": "image"})
                    else:
                        public_content.append(_sanitize_public_value(block))
                content = public_content
            else:
                content = _sanitize_public_value(content)
            messages.append({"role": role, "content": content})
        if messages:
            return messages
    for field in (
        "model_task_prompt",
        "task",
        "prompt",
        "question",
        "instruction",
        "creator_prompt",
    ):
        if field in row and _has_content(row[field]):
            task = _sanitize_public_value(row[field])
            if _has_content(task):
                return task
    raise BlindReviewError(f"benchmark sample {sample_id!r} has no task or prompt")


def _extract_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise BlindReviewError("benchmark metadata must be an object")
    metadata = _sanitize_public_value(raw)
    for key in _SAFE_TOP_LEVEL_METADATA:
        if key in row and key not in metadata:
            metadata[key] = _sanitize_public_value(row[key])
    return metadata


def _image_references(row: Mapping[str, Any]) -> list[Any]:
    for field in _IMAGE_FIELDS:
        if field not in row or row[field] in (None, ""):
            continue
        raw = row[field]
        if isinstance(raw, (str, os.PathLike, Mapping)):
            return [raw]
        if isinstance(raw, Sequence):
            return list(raw)
        raise BlindReviewError(f"benchmark {field} must be a path or a list of paths")
    return []


def _image_path_from_reference(reference: Any) -> Path:
    if isinstance(reference, Mapping):
        for field in _IMAGE_PATH_FIELDS:
            if field in reference and reference[field] not in (None, ""):
                reference = reference[field]
                break
        else:
            raise BlindReviewError("image object has no local path")
    if not isinstance(reference, (str, os.PathLike)):
        raise BlindReviewError("image reference must be a local filesystem path")
    text = os.fspath(reference)
    if not text or "\x00" in text:
        raise BlindReviewError("image path must be non-empty and must not contain NUL")
    return Path(text)


def _resolve_image(reference: Any, dataset_root: Path) -> Path:
    candidate = _image_path_from_reference(reference)
    try:
        resolved = resolve_dataset_file(dataset_root, candidate)
    except (OSError, RuntimeError) as exc:
        raise BlindReviewError(f"evidence image does not exist: {candidate}") from exc
    except ValueError as exc:
        raise BlindReviewError(f"evidence image escapes dataset_root: {candidate}") from exc
    if resolved.suffix.lower() not in _IMAGE_SUFFIXES:
        raise BlindReviewError(f"unsupported evidence image type: {candidate}")
    return resolved


def _index_benchmark_rows(
    rows: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if isinstance(rows, Mapping):
        source: Iterable[tuple[Any, Any]] = rows.items()
        keyed = True
    else:
        source = enumerate(rows)
        keyed = False
    for key, value in source:
        if not isinstance(value, Mapping):
            raise BlindReviewError("each benchmark row must be an object")
        row = dict(value)
        embedded = row.get("sample_id")
        sample_id = _require_identifier(key if keyed else embedded, "sample_id")
        if keyed and embedded is not None and embedded != sample_id:
            raise BlindReviewError("benchmark mapping key conflicts with row sample_id")
        if sample_id in indexed:
            raise BlindReviewError(f"duplicate benchmark sample_id: {sample_id}")
        row["sample_id"] = sample_id
        indexed[sample_id] = row
    return indexed


def _index_output_rows(
    rows: Mapping[tuple[str, str], Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(rows, Mapping):
        source: Iterable[tuple[Any, Any]] = rows.items()
        keyed = True
    else:
        source = enumerate(rows)
        keyed = False
    for key, value in source:
        if not isinstance(value, Mapping):
            if keyed and isinstance(key, tuple) and len(key) == 2:
                value = {"response": value}
            else:
                raise BlindReviewError(f"each {label} output row must be an object")
        row = dict(value)
        if keyed and isinstance(key, tuple) and len(key) == 2:
            sample_raw, condition_raw = key
            if row.get("sample_id") is not None and row["sample_id"] != sample_raw:
                raise BlindReviewError(f"{label} output key conflicts with row sample_id")
            if row.get("condition") is not None and row["condition"] != condition_raw:
                raise BlindReviewError(f"{label} output key conflicts with row condition")
        else:
            sample_raw = row.get("sample_id")
            condition_raw = row.get("condition")
        sample_id = _require_identifier(sample_raw, f"{label} output sample_id")
        condition = _require_identifier(condition_raw, f"{label} output condition")
        output_key = (sample_id, condition)
        if output_key in indexed:
            raise BlindReviewError(f"duplicate {label} output key: {output_key!r}")
        row["sample_id"] = sample_id
        row["condition"] = condition
        indexed[output_key] = row
    return indexed


def _success_flag(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _SUCCESS_STATUSES | {"true", "yes"}
    return False


def _is_successful_output(row: Mapping[str, Any]) -> bool:
    for field in ("error", "exception", "failure"):
        if field in row and row[field] not in (None, "", False, [], {}):
            return False
    if "status" in row and not _success_flag(row["status"]):
        return False
    for field in ("success", "ok", "complete", "completed"):
        if field in row and not _success_flag(row[field]):
            return False
    return True


def _extract_response(row: Mapping[str, Any]) -> Any:
    for field in _RESPONSE_FIELDS:
        if field in row and row[field] is not None:
            return _json_copy(row[field])
    error = row.get("error")
    if isinstance(error, Mapping):
        return {
            "generation_status": "error",
            "error_type": str(error.get("type") or "GenerationError"),
        }
    if not _is_successful_output(row):
        status = str(row.get("status") or "GenerationError").strip() or "GenerationError"
        return {"generation_status": "error", "error_type": status}
    return _MISSING


def _validate_seed(seed: Any) -> int | str:
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise BlindReviewError("seed must be an integer or non-empty string")
    if isinstance(seed, str) and not seed:
        raise BlindReviewError("seed must be an integer or non-empty string")
    return seed


def _derive_experiment_id(
    items: Sequence[dict[str, Any]],
    reviewer_ids: Sequence[str],
    reviews_per_item: int,
    seed: int | str,
    condition: str,
) -> str:
    fingerprint = {
        "condition": condition,
        "items": items,
        "reviewer_ids": sorted(reviewer_ids),
        "reviews_per_item": reviews_per_item,
        "seed": seed,
    }
    digest = hashlib.sha256(_json_text(fingerprint).encode("utf-8")).hexdigest()[:20]
    return f"vlm-ab-{digest}"


def _prepare_output_dir(output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = target.parent.resolve(strict=True) / target.name
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise BlindReviewError(f"output_dir is not a safe directory: {target}")
    if target.is_dir() and any(target.iterdir()):
        raise BlindReviewError(
            "output_dir already contains reviewer artifacts; use a new experiment run"
        )
    if target.is_dir():
        target.rmdir()
    staging = target.with_name(f".{target.name}.building")
    if staging.is_symlink():
        raise BlindReviewError(f"refusing to use symlinked staging directory: {staging}")
    if staging.exists():
        if not staging.is_dir():
            raise BlindReviewError(f"staging path is not a directory: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()
    return staging.resolve(strict=True)


def _prepare_child_dir(root: Path, name: str) -> Path:
    child = root / name
    if child.is_symlink():
        raise BlindReviewError(f"refusing to use symlinked package directory: {child}")
    child.mkdir(parents=True, exist_ok=True)
    resolved = child.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BlindReviewError(f"package directory escapes output_dir: {child}") from exc
    return resolved


def _json_for_script(payload: Any) -> str:
    text = _json_text(payload)
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blind paired VLM review</title>
  <style>
    :root { --ink:#18212b; --muted:#657383; --line:#ccd6df; --paper:#fff;
      --bg:#edf1f3; --accent:#075985; --soft:#e0f2fe; --danger:#b42318; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--bg);
      font:15px/1.45 Georgia, "Times New Roman", serif; }
    header { position:sticky; top:0; z-index:2; padding:15px max(18px, calc((100% - 1320px)/2));
      color:#fff; background:#132f3e; box-shadow:0 2px 8px #0003; }
    header h1 { margin:0 0 5px; font-size:23px; }
    header p { margin:0; color:#d9e8ef; }
    main { max-width:1320px; margin:auto; padding:20px 18px 60px; }
    .notice, .card { background:var(--paper); border:1px solid var(--line); border-radius:8px; }
    .notice { padding:13px 16px; margin-bottom:16px; }
    .rubric h2 { margin-top:14px; }
    .rubric h2:first-child { margin-top:0; }
    .rubric p { margin:6px 0; }
    .rubric ul { margin:7px 0; padding-left:22px; }
    .card { padding:18px; margin:18px 0; box-shadow:0 2px 8px #23313d0d; }
    h2 { margin:0 0 12px; font-size:19px; }
    h3 { margin:12px 0 7px; font-size:15px; color:var(--accent); }
    .muted { color:var(--muted); }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .response { min-width:0; padding:12px; border:1px solid var(--line); border-radius:7px; }
    pre { margin:0; padding:11px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere;
      font:13px/1.45 ui-monospace, Consolas, monospace; background:#f7f9fa; border-radius:5px; }
    .images { display:flex; flex-wrap:wrap; gap:10px; }
    .images img { max-width:min(100%, 520px); max-height:520px; object-fit:contain;
      border:1px solid var(--line); background:#fff; }
    .controls { display:grid; grid-template-columns:repeat(4, minmax(150px, 1fr)); gap:10px;
      margin-top:14px; }
    label { display:grid; gap:4px; font-weight:600; }
    select, textarea, button { font:inherit; border:1px solid #9aa9b7; border-radius:5px;
      background:#fff; padding:8px; color:var(--ink); }
    textarea { min-height:80px; resize:vertical; }
    .wide { grid-column:1/-1; }
    fieldset { min-width:0; border:1px solid var(--line); border-radius:6px; }
    fieldset label { display:inline-flex; grid-auto-flow:column; align-items:center; margin:3px 12px 3px 0;
      font:13px/1.3 ui-sans-serif, sans-serif; }
    .footer { display:flex; align-items:center; gap:12px; position:sticky; bottom:0; padding:12px;
      background:#edf1f3ee; border-top:1px solid var(--line); }
    button { cursor:pointer; color:#fff; background:var(--accent); font-weight:700; }
    .attestation { display:flex; grid-auto-flow:unset; align-items:flex-start; gap:9px; font-weight:600; }
    .attestation input { margin-top:4px; flex:0 0 auto; }
    #status.error { color:var(--danger); font-weight:700; }
    #status.ok { color:#166534; font-weight:700; }
    @media (max-width:850px) { .grid, .controls { grid-template-columns:1fr; }
      .wide { grid-column:auto; } header { position:static; } }
  </style>
</head>
<body>
  <header>
    <h1>Слепая парная оценка VL-моделей</h1>
    <p>Оценивайте только показанные материалы. Идентификаторы моделей отсутствуют в пакете.</p>
  </header>
  <main>
    <div id="rubric" class="notice rubric"></div>
    <div id="items"></div>
    <div class="notice">
      <label class="attestation"><input id="independent-attestation" type="checkbox">
        <span>Подтверждаю, что выполнил(а) оценку самостоятельно, не обсуждал(а) ответы со вторым
        экспертом, не использовал(а) внешний поиск и не пытался(ась) раскрыть модели.</span></label>
    </div>
    <div class="footer">
      <button id="export" type="button">Экспортировать итоговый JSON</button>
      <span id="status" class="muted"></span>
    </div>
  </main>
  <script>
  "use strict";
  const APP = __APP_DATA__;
  const ERROR_TAGS = __ERROR_TAGS__;
  const PREFS = ["left", "right", "tie", "skip"];
  const STORAGE_KEY = `blind-paired-review:${APP.experiment_id}:${APP.reviewer_id}`;
  const text = (tag, value, className="") => {
    const node = document.createElement(tag);
    node.textContent = String(value ?? "");
    if (className) node.className = className;
    return node;
  };
  const format = (value) => typeof value === "string" ? value : JSON.stringify(value, null, 2);
  const blank = (id) => ({ assignment_id:id, overall_preference:"", evidence_preference:"",
    visual_preference:"", temporal_preference:"", left_error_tags:[], right_error_tags:[],
    confidence:0, comments:"" });
  const state = Object.fromEntries(APP.assignments.map((item) => [item.assignment_id, blank(item.assignment_id)]));
  let independentAttestation = false;

  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (saved && typeof saved === "object" && saved.responses) APP.assignments.forEach((item) => {
      const row = saved.responses[item.assignment_id];
      if (row && typeof row === "object") state[item.assignment_id] = Object.assign(blank(item.assignment_id), row);
    });
    independentAttestation = saved?.independent_review_attestation === true;
  } catch (_) { /* Local drafts are optional. */ }
  const save = () => { try { localStorage.setItem(STORAGE_KEY, JSON.stringify({
    responses:state, independent_review_attestation:independentAttestation
  })); } catch (_) {} };

  function renderRubric() {
    const host = document.getElementById("rubric");
    host.append(text("h2", `Инструкция и критерии (${APP.rubric_version})`));
    const instructions = document.createElement("ul");
    APP.rubric.instructions.forEach((value) => {
      const item = document.createElement("li"); item.textContent = value; instructions.append(item);
    });
    host.append(instructions, text("h3", "Критерии"));
    Object.entries(APP.rubric.criteria).forEach(([key, value]) =>
      host.append(text("p", `${key}: ${value}`)));
    host.append(text("h3", "Значения выбора"));
    Object.entries(APP.rubric.preferences).forEach(([key, value]) =>
      host.append(text("p", `${key}: ${value}`)));
    host.append(text("p", `Rubric SHA256: ${APP.rubric_sha256}`, "muted"));
  }

  function preference(label, row, field) {
    const wrapper = document.createElement("label");
    wrapper.append(text("span", label));
    const select = document.createElement("select");
    select.append(new Option("Select...", ""));
    PREFS.forEach((value) => select.append(new Option(value, value)));
    select.value = row[field];
    select.addEventListener("change", () => { row[field] = select.value; save(); updateStatus(); });
    wrapper.append(select);
    return wrapper;
  }

  function errorTags(label, row, field) {
    const box = document.createElement("fieldset");
    box.append(text("legend", label));
    const selected = Array.isArray(row[field]) ? row[field] : [];
    ERROR_TAGS.forEach((tag) => {
      const labelNode = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = selected.includes(tag);
      input.addEventListener("change", () => {
        row[field] = ERROR_TAGS.filter((candidate) =>
          box.querySelector(`input[data-tag="${candidate}"]`).checked);
        save();
      });
      input.dataset.tag = tag;
      labelNode.title = APP.rubric.error_tags[tag] || "";
      labelNode.append(input, document.createTextNode(
        `${tag}: ${APP.rubric.error_tags[tag] || ""}`));
      box.append(labelNode);
    });
    return box;
  }

  function render() {
    const host = document.getElementById("items");
    APP.assignments.forEach((item, index) => {
      const row = state[item.assignment_id];
      const card = document.createElement("article");
      card.className = "card";
      card.dataset.assignmentId = item.assignment_id;
      card.append(text("h2", `Пример ${index + 1} из ${APP.assignments.length}`));
      card.append(text("div", item.assignment_id, "muted"));
      card.append(text("h3", "Задание"), text("pre", format(item.task)));
      card.append(text("h3", "Метаданные"), text("pre", format(item.metadata)));
      if (item.images.length) {
        card.append(text("h3", "Изображения-свидетельства"));
        const gallery = document.createElement("div");
        gallery.className = "images";
        item.images.forEach((src, imageIndex) => {
          const image = document.createElement("img");
          image.src = src;
          image.alt = `Evidence image ${imageIndex + 1}`;
          image.loading = "lazy";
          gallery.append(image);
        });
        card.append(gallery);
      }
      const compared = document.createElement("div");
      compared.className = "grid";
      [["Левый ответ", item.left_response], ["Правый ответ", item.right_response]].forEach(([label, value]) => {
        const panel = document.createElement("section");
        panel.className = "response";
        panel.append(text("h3", label), text("pre", format(value)));
        compared.append(panel);
      });
      card.append(compared);
      const controls = document.createElement("div");
      controls.className = "controls";
      controls.append(
        preference("Итоговое предпочтение", row, "overall_preference"),
        preference("Работа со свидетельствами", row, "evidence_preference"),
        preference("Визуальная корректность", row, "visual_preference"),
        preference("Временная корректность", row, "temporal_preference")
      );
      const confidence = document.createElement("label");
      confidence.append(text("span", "Уверенность"));
      const confidenceSelect = document.createElement("select");
      confidenceSelect.append(new Option("Выберите...", "0"));
      [1,2,3,4,5].forEach((value) => confidenceSelect.append(
        new Option(`${value} - ${APP.rubric.confidence[String(value)]}`, String(value))));
      confidenceSelect.value = String(row.confidence || 0);
      confidenceSelect.addEventListener("change", () => {
        row.confidence = Number(confidenceSelect.value); save(); updateStatus();
      });
      confidence.append(confidenceSelect);
      controls.append(confidence, errorTags("Ошибки левого ответа", row, "left_error_tags"),
        errorTags("Ошибки правого ответа", row, "right_error_tags"));
      const comments = document.createElement("label");
      comments.className = "wide";
      comments.append(text("span", "Краткое обоснование решения (обязательно)"));
      const area = document.createElement("textarea");
      area.value = typeof row.comments === "string" ? row.comments : "";
      area.addEventListener("input", () => { row.comments = area.value; save(); });
      comments.append(area);
      controls.append(comments);
      card.append(controls);
      host.append(card);
    });
    updateStatus();
  }

  function validationErrors() {
    const errors = [];
    APP.assignments.forEach((item, index) => {
      const row = state[item.assignment_id];
      ["overall_preference", "evidence_preference", "visual_preference", "temporal_preference"].forEach((field) => {
        if (!PREFS.includes(row[field])) errors.push(`Item ${index + 1}: ${field} is incomplete.`);
      });
      if (!Number.isInteger(row.confidence) || row.confidence < 1 || row.confidence > 5)
        errors.push(`Item ${index + 1}: confidence is incomplete.`);
      ["left_error_tags", "right_error_tags"].forEach((field) => {
        if (!Array.isArray(row[field]) || row[field].some((tag) => !ERROR_TAGS.includes(tag)) ||
            new Set(row[field]).size !== row[field].length)
          errors.push(`Item ${index + 1}: ${field} is malformed.`);
      });
      if (typeof row.comments !== "string") errors.push(`Item ${index + 1}: comments is malformed.`);
      if (!row.comments.trim()) errors.push(`Item ${index + 1}: add a brief rationale.`);
    });
    if (!independentAttestation) errors.push("Confirm the independent-review attestation.");
    return errors;
  }

  function updateStatus() {
    const status = document.getElementById("status");
    const errors = validationErrors();
    status.className = errors.length ? "muted" : "ok";
    status.textContent = errors.length ? `${errors.length} required fields remain.` : "Ready to export.";
  }

  function exportReview() {
    const errors = validationErrors();
    const status = document.getElementById("status");
    if (errors.length) {
      status.className = "error";
      status.textContent = errors[0];
      const match = /Item (\d+)/.exec(errors[0]);
      if (match) document.querySelectorAll(".card")[Number(match[1]) - 1]?.scrollIntoView({behavior:"smooth"});
      return;
    }
    const payload = {
      artifact_version: APP.artifact_version,
      experiment_id: APP.experiment_id,
      study_fingerprint: APP.study_fingerprint,
      rubric_version: APP.rubric_version,
      rubric_sha256: APP.rubric_sha256,
      reviewer_id: APP.reviewer_id,
      package_nonce: APP.package_nonce,
      independent_review_attestation: independentAttestation,
      assignments: APP.assignments.map((item) => item.assignment_id),
      responses: APP.assignments.map((item) => {
        const row = state[item.assignment_id];
        return {
          assignment_id: item.assignment_id,
          overall_preference: row.overall_preference,
          evidence_preference: row.evidence_preference,
          visual_preference: row.visual_preference,
          temporal_preference: row.temporal_preference,
          left_error_tags: row.left_error_tags.slice(),
          right_error_tags: row.right_error_tags.slice(),
          confidence: row.confidence,
          comments: row.comments
        };
      })
    };
    const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], {type:"application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "blind_review_export.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const attestation = document.getElementById("independent-attestation");
  attestation.checked = independentAttestation;
  attestation.addEventListener("change", () => {
    independentAttestation = attestation.checked; save(); updateStatus();
  });
  document.getElementById("export").addEventListener("click", exportReview);
  renderRubric();
  render();
  </script>
</body>
</html>
"""


def _render_html(public_payload: Mapping[str, Any]) -> str:
    return (
        _HTML_TEMPLATE.replace("__APP_DATA__", _json_for_script(public_payload))
        .replace("__ERROR_TAGS__", _json_for_script(ERROR_TAGS))
        .replace("__PAGE_TITLE__", html.escape("Blind paired VLM review"))
    )


def _assert_public_payload_is_blind(
    public_payload: Mapping[str, Any], sensitive_values: Iterable[str]
) -> None:
    serialized = _json_text(public_payload).casefold()
    leaked = sorted(
        value for value in set(sensitive_values) if value and value.casefold() in serialized
    )
    if leaked:
        raise BlindReviewError("sensitive model or authoring metadata appears in public content")


def build_blind_review_packages(
    benchmark_rows: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    base_output_rows: Mapping[tuple[str, str], Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    tuned_output_rows: Mapping[tuple[str, str], Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    dataset_root: str | Path,
    output_dir: str | Path,
    reviewer_ids: Sequence[str],
    reviews_per_item: int,
    seed: int | str,
    *,
    primary_condition: str = "original",
    experiment_id: str | None = None,
    owner_mapping_path: str | Path | None = None,
) -> BlindReviewBuild:
    """Build one offline blind-review package per reviewer.

    Output rows may be mappings keyed by ``(sample_id, condition)`` or iterables
    carrying those two fields. Explicit generation failures remain reviewable.
    Missing responses without an explicit failure marker stop package creation.
    """

    seed = _validate_seed(seed)
    condition = _require_identifier(primary_condition, "primary_condition")
    reviewers = sorted(_require_identifier(value, "reviewer_id") for value in reviewer_ids)
    normalized_reviewers = [normalize_identity(value, ascii_reviewer=True) for value in reviewers]
    if any(not value for value in normalized_reviewers):
        raise BlindReviewError("reviewer_ids must be safe ASCII identifiers")
    if len(normalized_reviewers) != len(set(normalized_reviewers)):
        raise BlindReviewError("reviewer_ids must be normalized-distinct")
    if not reviewers:
        raise BlindReviewError("at least one reviewer_id is required")
    if isinstance(reviews_per_item, bool) or not isinstance(reviews_per_item, int):
        raise BlindReviewError("reviews_per_item must be an integer")
    if not 1 <= reviews_per_item <= len(reviewers):
        raise BlindReviewError("reviews_per_item must be between 1 and reviewer count")

    try:
        root = Path(dataset_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BlindReviewError(f"dataset_root does not exist: {dataset_root}") from exc
    if not root.is_dir():
        raise BlindReviewError("dataset_root must be a directory")

    benchmark = _index_benchmark_rows(benchmark_rows)
    base_outputs = _index_output_rows(base_output_rows, "base")
    tuned_outputs = _index_output_rows(tuned_output_rows, "tuned")

    prepared: list[dict[str, Any]] = []
    expected_keys = {(sample_id, condition) for sample_id in benchmark}
    missing_base = sorted(expected_keys - set(base_outputs))
    missing_tuned = sorted(expected_keys - set(tuned_outputs))
    if missing_base or missing_tuned:
        raise BlindReviewError(
            "paired outputs are incomplete: "
            f"missing_base={len(missing_base)}, missing_tuned={len(missing_tuned)}"
        )
    for sample_id in sorted(benchmark):
        key = (sample_id, condition)
        base_row = base_outputs.get(key)
        tuned_row = tuned_outputs.get(key)
        assert base_row is not None and tuned_row is not None
        base_response = _extract_response(base_row)
        tuned_response = _extract_response(tuned_row)
        missing_arms = [
            arm
            for arm, response in (("base", base_response), ("tuned", tuned_response))
            if response is _MISSING
        ]
        if missing_arms:
            raise BlindReviewError(
                f"sample {sample_id!r} has no response or explicit generation error for "
                + ", ".join(missing_arms)
            )
        benchmark_row = benchmark[sample_id]
        task = _extract_task(benchmark_row, sample_id)
        metadata = _extract_metadata(benchmark_row)
        images = tuple(
            _resolve_image(reference, root) for reference in _image_references(benchmark_row)
        )
        sensitive = set()
        sensitive.update(_collect_sensitive_values(benchmark_row))
        sensitive.update(_collect_sensitive_values(base_row))
        sensitive.update(_collect_sensitive_values(tuned_row))
        prepared.append(
            {
                "sample_id": sample_id,
                "condition": condition,
                "task": task,
                "metadata": metadata,
                "images": images,
                "base_response": base_response,
                "tuned_response": tuned_response,
                "sensitive_values": frozenset(sensitive),
            }
        )
    if not prepared:
        raise BlindReviewError("no paired outputs for the primary condition")

    fingerprint_items = [
        {
            "sample_id": row["sample_id"],
            "condition": row["condition"],
            "task": row["task"],
            "metadata": row["metadata"],
            "images": [
                {
                    "path": str(path.relative_to(root)).replace(os.sep, "/"),
                    "sha256": _file_sha256(path),
                }
                for path in row["images"]
            ],
            "base_response": row["base_response"],
            "tuned_response": row["tuned_response"],
        }
        for row in prepared
    ]
    if experiment_id is None:
        experiment_id = _derive_experiment_id(
            fingerprint_items, reviewers, reviews_per_item, seed, condition
        )
    else:
        experiment_id = _require_identifier(experiment_id, "experiment_id")
    study_fingerprint = hashlib.sha256(
        _json_text(
            {
                "artifact_version": ARTIFACT_VERSION,
                "condition": condition,
                "experiment_id": experiment_id,
                "items": fingerprint_items,
                "reviewer_ids": reviewers,
                "reviews_per_item": reviews_per_item,
                "rubric_version": RUBRIC_VERSION,
                "rubric_sha256": RUBRIC_SHA256,
                "seed_commitment": hashlib.sha256(
                    f"{type(seed).__name__}:{seed}".encode("utf-8")
                ).hexdigest(),
            }
        ).encode("utf-8")
    ).hexdigest()

    items: list[_PairedItem] = []
    assignment_ids: set[str] = set()
    for row in prepared:
        assignment_id = _digest_id(
            "item", study_fingerprint, seed, row["sample_id"], row["condition"]
        )
        if assignment_id in assignment_ids:
            raise BlindReviewError("opaque assignment_id collision")
        assignment_ids.add(assignment_id)
        items.append(
            _PairedItem(
                assignment_id=assignment_id,
                sample_id=row["sample_id"],
                condition=row["condition"],
                task=row["task"],
                metadata=row["metadata"],
                images=row["images"],
                base_response=row["base_response"],
                tuned_response=row["tuned_response"],
                sensitive_values=row["sensitive_values"],
            )
        )

    scheduled_items = list(items)
    _derived_rng(seed, f"{experiment_id}:reviewer-subset-items").shuffle(scheduled_items)
    assigned: dict[str, list[_PairedItem]] = {reviewer: [] for reviewer in reviewers}
    paired_two_expert_design = len(reviewers) == reviews_per_item == 2
    paired_side_plan: dict[str, bool] = {}
    if paired_two_expert_design:
        assigned[reviewers[0]] = list(scheduled_items)
        assigned[reviewers[1]] = list(reversed(scheduled_items))
        side_rng = _derived_rng(seed, f"{experiment_id}:paired-side-balance")
        tuned_left_count = len(scheduled_items) // 2
        if len(scheduled_items) % 2 and side_rng.randrange(2):
            tuned_left_count += 1
        flags = [True] * tuned_left_count + [False] * (len(scheduled_items) - tuned_left_count)
        side_rng.shuffle(flags)
        paired_side_plan = {
            item.assignment_id: tuned_left for item, tuned_left in zip(scheduled_items, flags)
        }
    else:
        reviewer_cycle = list(reviewers)
        _derived_rng(seed, f"{experiment_id}:reviewer-subset-reviewers").shuffle(reviewer_cycle)
        slot = 0
        for item in scheduled_items:
            for offset in range(reviews_per_item):
                reviewer = reviewer_cycle[(slot + offset) % len(reviewer_cycle)]
                assigned[reviewer].append(item)
            slot += reviews_per_item

    published_output = Path(output_dir)
    published_output.parent.mkdir(parents=True, exist_ok=True)
    published_output = published_output.parent.resolve(strict=True) / published_output.name
    root_output = _prepare_output_dir(published_output)
    reviewer_packages: dict[str, ReviewerPackage] = {}
    owner_entries: list[dict[str, Any]] = []
    used_package_names: set[str] = set()

    for reviewer_id in reviewers:
        package_name = _digest_id("reviewer", experiment_id, reviewer_id, length=20)
        if package_name in used_package_names:
            raise BlindReviewError("opaque reviewer directory collision")
        used_package_names.add(package_name)
        package_dir = _prepare_child_dir(root_output, package_name)
        image_dir = _prepare_child_dir(package_dir, "images")

        reviewer_items = list(assigned[reviewer_id])
        if paired_two_expert_design:
            invert = reviewer_id == reviewers[1]
            tuned_left_flags = [
                paired_side_plan[item.assignment_id] ^ invert for item in reviewer_items
            ]
        else:
            _derived_rng(seed, f"{experiment_id}:{reviewer_id}:item-order").shuffle(reviewer_items)
            side_rng = _derived_rng(seed, f"{experiment_id}:{reviewer_id}:side-balance")
            tuned_left_count = len(reviewer_items) // 2
            if len(reviewer_items) % 2 and side_rng.randrange(2):
                tuned_left_count += 1
            tuned_left_flags = [True] * tuned_left_count + [False] * (
                len(reviewer_items) - tuned_left_count
            )
            side_rng.shuffle(tuned_left_flags)
        package_nonce = _reviewer_package_nonce(seed, experiment_id, reviewer_id)

        public_assignments: list[dict[str, Any]] = []
        copied_images: list[Path] = []
        package_sensitive: set[str] = set()
        for display_position, (item, tuned_left) in enumerate(
            zip(reviewer_items, tuned_left_flags), start=1
        ):
            public_images: list[str] = []
            for image_index, source in enumerate(item.images):
                image_name = (
                    _digest_id(
                        "img",
                        experiment_id,
                        reviewer_id,
                        item.assignment_id,
                        image_index,
                        length=32,
                    )
                    + source.suffix.lower()
                )
                destination = image_dir / image_name
                if destination.is_symlink():
                    raise BlindReviewError(f"refusing to overwrite symlink: {destination}")
                shutil.copyfile(source, destination)
                copied_images.append(destination)
                public_images.append(f"images/{image_name}")

            left_arm = "tuned" if tuned_left else "base"
            right_arm = "base" if tuned_left else "tuned"
            public_assignments.append(
                {
                    "assignment_id": item.assignment_id,
                    "task": item.task,
                    "metadata": item.metadata,
                    "images": public_images,
                    "left_response": item.tuned_response if tuned_left else item.base_response,
                    "right_response": item.base_response if tuned_left else item.tuned_response,
                }
            )
            package_sensitive.update(item.sensitive_values)
            owner_entries.append(
                {
                    "assignment_id": item.assignment_id,
                    "reviewer_id": reviewer_id,
                    "sample_id": item.sample_id,
                    "condition": item.condition,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "display_position": display_position,
                    "package_nonce": package_nonce,
                }
            )

        public_payload = {
            "artifact_version": ARTIFACT_VERSION,
            "experiment_id": experiment_id,
            "study_fingerprint": study_fingerprint,
            "rubric_version": RUBRIC_VERSION,
            "rubric_sha256": RUBRIC_SHA256,
            "rubric": RUBRIC,
            "reviewer_id": reviewer_id,
            "package_nonce": package_nonce,
            "assignments": public_assignments,
        }
        _assert_public_payload_is_blind(public_payload, package_sensitive)
        assignment_path = package_dir / PUBLIC_ASSIGNMENT_FILENAME
        html_path = package_dir / REVIEW_HTML_FILENAME
        _write_json(assignment_path, public_payload)
        if html_path.is_symlink():
            raise BlindReviewError(f"refusing to overwrite symlink: {html_path}")
        with html_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(_render_html(public_payload))
        reviewer_packages[reviewer_id] = ReviewerPackage(
            reviewer_id=reviewer_id,
            package_dir=package_dir,
            html_path=html_path,
            assignment_path=assignment_path,
            image_paths=tuple(copied_images),
        )

    reviewer_rank = {reviewer: index for index, reviewer in enumerate(reviewers)}
    owner_entries.sort(key=lambda row: (reviewer_rank[row["reviewer_id"]], row["assignment_id"]))
    owner_core = {
        "artifact_version": ARTIFACT_VERSION,
        "experiment_id": experiment_id,
        "study_fingerprint": study_fingerprint,
        "assignments": owner_entries,
    }
    owner_payload = {
        **owner_core,
        "integrity_hmac_sha256": _payload_hmac(owner_core, seed),
    }
    owner_path = (
        Path(owner_mapping_path).resolve()
        if owner_mapping_path is not None
        else root_output / OWNER_MAPPING_FILENAME
    )
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(owner_path, owner_payload)
    try:
        owner_path.chmod(0o600)
    except OSError:
        pass

    if published_output.exists() or published_output.is_symlink():
        raise BlindReviewError("reviewer output appeared while the package was being built")
    os.replace(root_output, published_output)

    def published(path: Path) -> Path:
        try:
            return published_output / path.relative_to(root_output)
        except ValueError:
            return path

    published_packages = {
        reviewer_id: ReviewerPackage(
            reviewer_id=package.reviewer_id,
            package_dir=published(package.package_dir),
            html_path=published(package.html_path),
            assignment_path=published(package.assignment_path),
            image_paths=tuple(published(path) for path in package.image_paths),
        )
        for reviewer_id, package in reviewer_packages.items()
    }
    return BlindReviewBuild(
        experiment_id=experiment_id,
        study_fingerprint=study_fingerprint,
        output_dir=published_output,
        owner_mapping_path=published(owner_path),
        reviewer_packages=published_packages,
        paired_item_count=len(items),
        total_assignment_count=len(owner_entries),
    )


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReviewValidationError(
            f"{label} fields do not match schema; missing={missing}, extra={extra}"
        )


def _validated_string(value: Any, field: str) -> str:
    try:
        return _require_identifier(value, field)
    except BlindReviewError as exc:
        raise ReviewValidationError(str(exc)) from exc


def _validated_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ReviewValidationError(f"{field} must be a lowercase SHA256 digest")
    return value


def _validate_preference(value: Any, field: str) -> str:
    if not isinstance(value, str) or value not in PREFERENCE_VALUES:
        raise ReviewValidationError(f"{field} must be one of {sorted(PREFERENCE_VALUES)}")
    return value


def _validate_error_tags(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ReviewValidationError(f"{field} must be an array")
    if any(not isinstance(tag, str) or tag not in ERROR_TAGS for tag in value):
        raise ReviewValidationError(f"{field} contains an unknown error tag")
    if len(value) != len(set(value)):
        raise ReviewValidationError(f"{field} must not contain duplicate tags")
    return list(value)


def validate_review_export(
    payload: Mapping[str, Any],
    *,
    expected_experiment_id: str | None = None,
    expected_study_fingerprint: str | None = None,
    expected_reviewer_id: str | None = None,
    expected_assignment_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate and return a normalized copy of a complete browser export."""

    if not isinstance(payload, Mapping):
        raise ReviewValidationError("review export must be a JSON object")
    _exact_fields(payload, _REVIEW_TOP_LEVEL_FIELDS, "review export")
    version = payload["artifact_version"]
    if isinstance(version, bool) or version != ARTIFACT_VERSION:
        raise ReviewValidationError(f"artifact_version must be {ARTIFACT_VERSION}")
    experiment_id = _validated_string(payload["experiment_id"], "experiment_id")
    study_fingerprint = _validated_sha256(payload["study_fingerprint"], "study_fingerprint")
    rubric_version = _validated_string(payload["rubric_version"], "rubric_version")
    rubric_sha256 = _validated_sha256(payload["rubric_sha256"], "rubric_sha256")
    reviewer_id = _validated_string(payload["reviewer_id"], "reviewer_id")
    package_nonce = _validated_sha256(payload["package_nonce"], "package_nonce")
    if rubric_version != RUBRIC_VERSION or rubric_sha256 != RUBRIC_SHA256:
        raise ReviewValidationError("review export uses an unknown rubric version or hash")
    if payload["independent_review_attestation"] is not True:
        raise ReviewValidationError("independent_review_attestation must be true")
    if expected_experiment_id is not None and experiment_id != expected_experiment_id:
        raise ReviewValidationError("experiment_id does not match the expected package")
    if expected_reviewer_id is not None and reviewer_id != expected_reviewer_id:
        raise ReviewValidationError("reviewer_id does not match the expected package")
    if expected_study_fingerprint is not None and study_fingerprint != expected_study_fingerprint:
        raise ReviewValidationError("study_fingerprint does not match the expected package")

    raw_assignments = payload["assignments"]
    if not isinstance(raw_assignments, list):
        raise ReviewValidationError("assignments must be an array")
    assignments = [
        _validated_string(value, f"assignments[{index}]")
        for index, value in enumerate(raw_assignments)
    ]
    if len(assignments) != len(set(assignments)):
        raise ReviewValidationError("assignments must be unique")
    if expected_assignment_ids is not None:
        expected = list(expected_assignment_ids)
        if assignments != expected:
            raise ReviewValidationError("assignments do not match the expected package order")

    raw_responses = payload["responses"]
    if not isinstance(raw_responses, list):
        raise ReviewValidationError("responses must be an array")
    if len(raw_responses) != len(assignments):
        raise ReviewValidationError("responses must contain exactly one row per assignment")
    responses: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_responses):
        if not isinstance(raw, Mapping):
            raise ReviewValidationError(f"responses[{index}] must be an object")
        _exact_fields(raw, _REVIEW_RESPONSE_FIELDS, f"responses[{index}]")
        assignment_id = _validated_string(raw["assignment_id"], f"responses[{index}].assignment_id")
        if assignment_id != assignments[index]:
            raise ReviewValidationError(
                f"responses[{index}].assignment_id is out of order or unknown"
            )
        confidence = raw["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, int):
            raise ReviewValidationError(f"responses[{index}].confidence must be an integer")
        if not 1 <= confidence <= 5:
            raise ReviewValidationError(f"responses[{index}].confidence must be from 1 to 5")
        comments = raw["comments"]
        if not isinstance(comments, str):
            raise ReviewValidationError(f"responses[{index}].comments must be a string")
        if not comments.strip():
            raise ReviewValidationError(f"responses[{index}].comments must contain a rationale")
        responses.append(
            {
                "assignment_id": assignment_id,
                "overall_preference": _validate_preference(
                    raw["overall_preference"],
                    f"responses[{index}].overall_preference",
                ),
                "evidence_preference": _validate_preference(
                    raw["evidence_preference"],
                    f"responses[{index}].evidence_preference",
                ),
                "visual_preference": _validate_preference(
                    raw["visual_preference"],
                    f"responses[{index}].visual_preference",
                ),
                "temporal_preference": _validate_preference(
                    raw["temporal_preference"],
                    f"responses[{index}].temporal_preference",
                ),
                "left_error_tags": _validate_error_tags(
                    raw["left_error_tags"], f"responses[{index}].left_error_tags"
                ),
                "right_error_tags": _validate_error_tags(
                    raw["right_error_tags"], f"responses[{index}].right_error_tags"
                ),
                "confidence": confidence,
                "comments": comments,
            }
        )
    return {
        "artifact_version": ARTIFACT_VERSION,
        "experiment_id": experiment_id,
        "study_fingerprint": study_fingerprint,
        "rubric_version": rubric_version,
        "rubric_sha256": rubric_sha256,
        "reviewer_id": reviewer_id,
        "package_nonce": package_nonce,
        "independent_review_attestation": True,
        "assignments": assignments,
        "responses": responses,
    }


def _reject_json_constant(value: str) -> None:
    raise ReviewValidationError(f"non-finite JSON number is not allowed: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_file(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ReviewValidationError(f"cannot read {label}: {source}") from exc
    if size > _MAX_JSON_BYTES:
        raise ReviewValidationError(f"{label} exceeds {_MAX_JSON_BYTES} bytes")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ReviewValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewValidationError(f"{label} must contain a JSON object")
    return payload


def load_review_export(
    path: str | Path,
    *,
    expected_experiment_id: str | None = None,
    expected_study_fingerprint: str | None = None,
    expected_reviewer_id: str | None = None,
    expected_assignment_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load a strict JSON review export and validate it."""

    payload = _load_json_file(path, "review export")
    return validate_review_export(
        payload,
        expected_experiment_id=expected_experiment_id,
        expected_study_fingerprint=expected_study_fingerprint,
        expected_reviewer_id=expected_reviewer_id,
        expected_assignment_ids=expected_assignment_ids,
    )


def _validate_owner_mapping(
    payload: Mapping[str, Any], blinding_secret: int | str | None = None
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ReviewValidationError("owner mapping must be a JSON object")
    _exact_fields(payload, _OWNER_TOP_LEVEL_FIELDS, "owner mapping")
    version = payload["artifact_version"]
    if isinstance(version, bool) or version != ARTIFACT_VERSION:
        raise ReviewValidationError(f"owner artifact_version must be {ARTIFACT_VERSION}")
    experiment_id = _validated_string(payload["experiment_id"], "owner experiment_id")
    study_fingerprint = _validated_sha256(payload["study_fingerprint"], "owner study_fingerprint")
    integrity_hmac = _validated_sha256(
        payload["integrity_hmac_sha256"], "owner integrity_hmac_sha256"
    )
    raw_entries = payload["assignments"]
    if not isinstance(raw_entries, list):
        raise ReviewValidationError("owner assignments must be an array")
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise ReviewValidationError(f"owner assignments[{index}] must be an object")
        _exact_fields(raw, _OWNER_ENTRY_FIELDS, f"owner assignments[{index}]")
        string_fields = _OWNER_ENTRY_FIELDS - {"display_position", "package_nonce"}
        entry: dict[str, Any] = {
            field: _validated_string(raw[field], f"owner assignments[{index}].{field}")
            for field in string_fields
        }
        entry["package_nonce"] = _validated_sha256(
            raw["package_nonce"], f"owner assignments[{index}].package_nonce"
        )
        display_position = raw["display_position"]
        if (
            isinstance(display_position, bool)
            or not isinstance(display_position, int)
            or display_position < 1
        ):
            raise ReviewValidationError(
                f"owner assignments[{index}].display_position must be a positive integer"
            )
        entry["display_position"] = display_position
        if {entry["left_arm"], entry["right_arm"]} != {"base", "tuned"}:
            raise ReviewValidationError("owner left_arm/right_arm must be opposite base/tuned arms")
        key = (entry["reviewer_id"], entry["assignment_id"])
        if key in seen:
            raise ReviewValidationError("owner mapping contains duplicate reviewer assignment")
        seen.add(key)
        entries.append(entry)
    normalized = {
        "artifact_version": ARTIFACT_VERSION,
        "experiment_id": experiment_id,
        "study_fingerprint": study_fingerprint,
        "assignments": entries,
        "integrity_hmac_sha256": integrity_hmac,
    }
    if blinding_secret is not None:
        owner_core = {
            key: normalized[key]
            for key in ("artifact_version", "experiment_id", "study_fingerprint", "assignments")
        }
        if not hmac.compare_digest(integrity_hmac, _payload_hmac(owner_core, blinding_secret)):
            raise ReviewValidationError("owner mapping integrity HMAC does not match")
    return normalized


def _load_owner_mapping(
    source: Mapping[str, Any] | str | Path,
    blinding_secret: int | str | None = None,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return _validate_owner_mapping(source, blinding_secret)
    return _validate_owner_mapping(_load_json_file(source, "owner mapping"), blinding_secret)


def _review_sources(
    exports: Mapping[str, Any] | str | Path | Iterable[Mapping[str, Any] | str | Path],
) -> list[Mapping[str, Any] | str | Path]:
    if isinstance(exports, (Mapping, str, Path)):
        return [exports]
    return list(exports)


def _normalize_displayed_preference(displayed: str, owner_entry: Mapping[str, Any]) -> str:
    if displayed in {"tie", "skip"}:
        return displayed
    return owner_entry[f"{displayed}_arm"]


def deblind_reviews(
    review_exports: Mapping[str, Any] | str | Path | Iterable[Mapping[str, Any] | str | Path],
    owner_mapping: Mapping[str, Any] | str | Path,
    *,
    blinding_secret: int | str | None = None,
) -> list[dict[str, Any]]:
    """Deblind validated exports into one normalized row per reviewer assignment."""

    owner = _load_owner_mapping(owner_mapping, blinding_secret)
    owner_index = {
        (entry["reviewer_id"], entry["assignment_id"]): entry for entry in owner["assignments"]
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in _review_sources(review_exports):
        review = (
            validate_review_export(source)
            if isinstance(source, Mapping)
            else load_review_export(source)
        )
        if review["experiment_id"] != owner["experiment_id"]:
            raise ReviewValidationError("review experiment_id does not match owner mapping")
        if review["study_fingerprint"] != owner["study_fingerprint"]:
            raise ReviewValidationError("review study_fingerprint does not match owner mapping")
        reviewer_id = review["reviewer_id"]
        for response in review["responses"]:
            key = (reviewer_id, response["assignment_id"])
            if key in seen:
                raise ReviewValidationError("duplicate review for reviewer assignment")
            seen.add(key)
            entry = owner_index.get(key)
            if entry is None:
                raise ReviewValidationError("review assignment is absent from owner mapping")
            if review["package_nonce"] != entry["package_nonce"]:
                raise ReviewValidationError(
                    "review package nonce does not match the reviewer package"
                )
            tuned_side = "left" if entry["left_arm"] == "tuned" else "right"
            base_side = "right" if tuned_side == "left" else "left"
            displayed = response["overall_preference"]
            normalized = _normalize_displayed_preference(displayed, entry)
            row = {
                "artifact_version": ARTIFACT_VERSION,
                "experiment_id": review["experiment_id"],
                "study_fingerprint": review["study_fingerprint"],
                "reviewer_id": reviewer_id,
                "assignment_id": response["assignment_id"],
                "sample_id": entry["sample_id"],
                "condition": entry["condition"],
                "display_position": entry["display_position"],
                "tuned_side": tuned_side,
                "base_side": base_side,
                "displayed_preference": displayed,
                "preference": normalized,
                "displayed_overall_preference": displayed,
                "overall_preference": normalized,
            }
            for criterion in ("evidence", "visual", "temporal"):
                field = f"{criterion}_preference"
                criterion_displayed = response[field]
                row[f"displayed_{field}"] = criterion_displayed
                row[field] = _normalize_displayed_preference(criterion_displayed, entry)
            row.update(
                {
                    "left_error_tags": list(response["left_error_tags"]),
                    "right_error_tags": list(response["right_error_tags"]),
                    "tuned_error_tags": list(response[f"{tuned_side}_error_tags"]),
                    "base_error_tags": list(response[f"{base_side}_error_tags"]),
                    "confidence": response["confidence"],
                    "comments": response["comments"],
                    "rubric_version": review["rubric_version"],
                    "rubric_sha256": review["rubric_sha256"],
                    "independent_review_attestation": review["independent_review_attestation"],
                }
            )
            rows.append(row)
    return rows


build_blind_review_package = build_blind_review_packages
validate_exported_review = validate_review_export
load_exported_review = load_review_export
deblind_review_exports = deblind_reviews


__all__ = [
    "ARTIFACT_VERSION",
    "BlindReviewBuild",
    "BlindReviewError",
    "ERROR_TAGS",
    "RUBRIC",
    "RUBRIC_SHA256",
    "RUBRIC_VERSION",
    "ReviewerPackage",
    "ReviewValidationError",
    "build_blind_review_package",
    "build_blind_review_packages",
    "deblind_review_exports",
    "deblind_reviews",
    "load_exported_review",
    "load_review_export",
    "validate_exported_review",
    "validate_review_export",
]
