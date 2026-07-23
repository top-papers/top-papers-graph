# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dependency-light publication audit for the Task 3 HF VLM benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import unquote

from .identities import contains_unsafe_identifier_codepoint, normalize_identity
from .paths import resolve_dataset_file

__all__ = [
    "BenchmarkAuditError",
    "audit_benchmark",
    "benchmark_paper_ids",
    "canonical_paper_id",
    "load_jsonl",
    "paper_identity_errors",
    "write_audit_report",
]

AUDIT_VERSION = 3
_SUPPORTED_AUDIT_VERSIONS = frozenset({1, 2, AUDIT_VERSION})

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
_ARXIV_NEW_RE = re.compile(
    r"(?<!\d)(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)",
    re.IGNORECASE,
)
_ARXIV_OLD_RE = re.compile(
    r"(?:arxiv\s*:\s*)?([a-z][a-z0-9.-]*/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
_OPAQUE_PAPER_RE = re.compile(
    r"(?<![\w])paper\s*:\s*([a-z0-9](?:[a-z0-9._:/+-]*[a-z0-9])?)",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
_URI_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)

_PROMPT_FIELDS = (
    "model_task_prompt",
    "prompt",
    "question",
    "instruction",
    "task",
    "input_text",
)
_TRAINING_PROMPT_KEYS = frozenset(
    {
        "input",
        "input_text",
        "instruction",
        "model_task_prompt",
        "prompt",
        "query",
        "question",
        "task",
        "user_prompt",
    }
)
_PAPER_ID_KEYS = frozenset(
    {
        "arxiv",
        "arxiv_id",
        "canonical_paper_id",
        "doi",
        "doi_id",
        "paper_id",
        "paper_ids",
        "paper_identifier",
        "publication_id",
        "publication_identifier",
        "source_paper_id",
    }
)
_PAPER_CONTEXT_KEYS = frozenset({"article", "paper", "publication", "source_paper"})
_GOLD_KEYS = frozenset(
    {
        "answer_key",
        "expected_answer",
        "gold",
        "gold_answer",
        "ground_truth",
        "ground_truth_answer",
    }
)
_REFERENCE_KEYS = frozenset(
    {"reference", "reference_answer", "reference_output", "reference_response"}
)
_RUBRIC_KEYS = frozenset({"evaluation_rubric", "grading_rubric", "rubric", "scoring_rubric"})
_LEAKAGE_VALUE_KEYS = (
    _GOLD_KEYS
    | _REFERENCE_KEYS
    | _RUBRIC_KEYS
    | frozenset(
        {
            "creator_rationale",
            "expected_error_modes",
            "expected_winner",
            "review_focus",
        }
    )
)
_IMAGE_PATH_KEYS = ("path", "image_path", "file", "file_name", "filename")

_RU_COMPARE_STEMS = (
    "\u0441\u0440\u0430\u0432\u043d",
    "\u0441\u043e\u043f\u043e\u0441\u0442\u0430\u0432",
)
_RU_SUBJECT_STEMS = (
    "\u0432\u0430\u0440\u0438\u0430\u043d\u0442",
    "\u043e\u0442\u0432\u0435\u0442",
    "\u043c\u043e\u0434\u0435\u043b",
    "vlm",
)
_RU_EACH_STEM = "\u043a\u0430\u0436\u0434"
_RU_WHICH = "\u043a\u0430\u043a\u043e\u0439"
_RU_BETTER_STEMS = ("\u043b\u0443\u0447\u0448", "\u0442\u043e\u0447\u043d")
_RU_TWO_WORDS = ("\u0434\u0432\u0430", "\u0434\u0432\u0435")
_RU_ANSWER = "\u043e\u0442\u0432\u0435\u0442"
_RU_LEAK_STEMS = (
    "\u044d\u0442\u0430\u043b\u043e\u043d\u043d",
    "\u043f\u0440\u0430\u0432\u0438\u043b\u044c\u043d",
    "\u043e\u0436\u0438\u0434\u0430\u0435\u043c",
)
_RU_CRITERIA = "\u043a\u0440\u0438\u0442\u0435\u0440\u0438"
_RU_ASSESSMENT = "\u043e\u0446\u0435\u043d"


class BenchmarkAuditError(ValueError):
    """Raised when an audit input or output cannot be handled safely."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _stable_percent_decode(value: str) -> str | None:
    text = value
    while True:
        try:
            decoded = unquote(text, errors="strict")
        except (UnicodeError, ValueError):
            return None
        if decoded == text:
            return text
        if len(decoded) >= len(text):
            return None
        text = decoded


def _normalized_key_details(value: Any) -> tuple[str, bool]:
    text = str(value)
    unsafe = False
    seen: set[str] = set()
    while text not in seen:
        seen.add(text)
        normalized = unicodedata.normalize("NFKC", text).strip()
        unsafe = unsafe or normalized != text
        filtered = "".join(
            character
            for character in normalized
            if not contains_unsafe_identifier_codepoint(character)
        )
        unsafe = unsafe or filtered != normalized
        decoded = _stable_percent_decode(filtered)
        if decoded is None:
            return "", True
        unsafe = unsafe or decoded != filtered
        if decoded == text:
            text = decoded
            break
        text = decoded
    else:
        return "", True
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return text.casefold().replace("-", "_"), unsafe


def _normalized_key(value: Any) -> str:
    return _normalized_key_details(value)[0]


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _compact_prompt(value: str) -> str:
    return re.sub(r"\s+", "", _normalize_text(value))


def _strip_doi_tail(value: str) -> str:
    value = value.rstrip(".,;:")
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    return value


def canonical_paper_id(value: Any) -> str:
    """Return a stable DOI, arXiv, or opaque paper identifier.

    DOI case and URL wrappers are removed. arXiv version suffixes are removed so
    versions of the same arXiv record compare equal. Other identifiers receive a
    ``paper:`` prefix after Unicode, case, and whitespace normalization.
    """

    if value is None or isinstance(value, (Mapping, list, tuple, set, bool)):
        return ""
    try:
        text = str(value)
    except (TypeError, ValueError):
        return ""
    if not text or contains_unsafe_identifier_codepoint(text):
        return ""
    decoded = _stable_percent_decode(text)
    if decoded is None:
        return ""
    text = decoded
    if contains_unsafe_identifier_codepoint(text):
        return ""
    try:
        text = unicodedata.normalize("NFKC", text).strip()
    except (TypeError, ValueError):
        return ""
    if not text or contains_unsafe_identifier_codepoint(text):
        return ""

    doi_match = _DOI_RE.search(text)
    if doi_match:
        doi = _strip_doi_tail(doi_match.group(0)).casefold()
        return f"doi:{doi}"
    arxiv_match = _ARXIV_NEW_RE.search(text)
    if arxiv_match:
        return f"arxiv:{arxiv_match.group(1).casefold()}"
    arxiv_match = _ARXIV_OLD_RE.search(text)
    if arxiv_match:
        return f"arxiv:{arxiv_match.group(1).casefold()}"

    opaque = re.sub(r"\s+", " ", text).strip().casefold()
    if opaque.startswith("paper:"):
        opaque = opaque[6:].strip()
    return f"paper:{opaque}" if opaque else ""


def _identifiers_in_text(value: str) -> set[str]:
    decoded = value
    while True:
        candidate = unquote(decoded, errors="replace")
        if candidate == decoded or len(candidate) >= len(decoded):
            break
        decoded = candidate
    texts = [unicodedata.normalize("NFKC", value)]
    if decoded != value:
        texts.append(unicodedata.normalize("NFKC", decoded))
    found: set[str] = set()
    for text in texts:
        for pattern in (_DOI_RE, _ARXIV_NEW_RE, _ARXIV_OLD_RE):
            for match in pattern.finditer(text):
                canonical = canonical_paper_id(match.group(0))
                if canonical.startswith(("doi:", "arxiv:")):
                    found.add(canonical)
        for match in _OPAQUE_PAPER_RE.finditer(text):
            canonical = canonical_paper_id(f"paper:{match.group(1)}")
            if canonical.startswith("paper:"):
                found.add(canonical)
    return found


def _extract_identifiers(
    value: Any,
    *,
    key_hint: str = "",
    paper_context: bool = False,
    seen: set[int] | None = None,
) -> set[str]:
    if seen is None:
        seen = set()
    found: set[str] = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            return found
        seen.add(marker)
        for raw_key, child in value.items():
            key = _normalized_key(raw_key)
            child_context = paper_context or key in _PAPER_CONTEXT_KEYS
            found.update(
                _extract_identifiers(
                    child,
                    key_hint=key,
                    paper_context=child_context,
                    seen=seen,
                )
            )
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        marker = id(value)
        if marker in seen:
            return found
        seen.add(marker)
        for child in value:
            found.update(
                _extract_identifiers(
                    child,
                    key_hint=key_hint,
                    paper_context=paper_context,
                    seen=seen,
                )
            )
        return found
    if isinstance(value, str):
        found.update(_identifiers_in_text(value))
        if key_hint in _PAPER_ID_KEYS or (
            paper_context and key_hint in {"id", "identifier", "identifiers"}
        ):
            canonical = canonical_paper_id(value)
            if canonical:
                found.add(canonical)
    elif key_hint in _PAPER_ID_KEYS and isinstance(value, (int, float)):
        canonical = canonical_paper_id(value)
        if canonical:
            found.add(canonical)
    return found


def _paper_sort_key(value: str) -> tuple[int, str]:
    if value.startswith("doi:"):
        return (0, value)
    if value.startswith("arxiv:"):
        return (1, value)
    return (2, value)


def benchmark_paper_ids(row: Mapping[str, Any]) -> list[str]:
    """Extract every canonical paper identifier declared by a row."""

    found: set[str] = set()
    for key, value in row.items():
        normalized = _normalized_key(key)
        if normalized in _PAPER_ID_KEYS:
            found.update(_extract_identifiers(value, key_hint=normalized))
        elif normalized in _PAPER_CONTEXT_KEYS:
            found.update(_extract_identifiers(value, key_hint=normalized, paper_context=True))
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            normalized = _normalized_key(key)
            if normalized in _PAPER_ID_KEYS:
                found.update(_extract_identifiers(value, key_hint=normalized))
            elif normalized in _PAPER_CONTEXT_KEYS:
                found.update(_extract_identifiers(value, key_hint=normalized, paper_context=True))
    for _, key, value, _ in _paper_identifier_declarations(row):
        found.update(_extract_identifiers(value, key_hint=key, paper_context=True))
    return sorted(found, key=_paper_sort_key)


def _paper_identifier_declarations(
    value: Any,
    *,
    paper_context: bool = False,
    seen: set[int] | None = None,
) -> list[tuple[str, str, Any, bool]]:
    if seen is None:
        seen = set()
    declarations: list[tuple[str, str, Any, bool]] = []
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            return declarations
        seen.add(marker)
        for raw_key, child in value.items():
            key_text = str(raw_key)
            key, unsafe_key = _normalized_key_details(raw_key)
            is_identifier = key in _PAPER_ID_KEYS or (
                paper_context and key in {"id", "identifier", "identifiers"}
            )
            if is_identifier:
                declarations.append((key_text, key, child, unsafe_key))
            child_context = paper_context or key in _PAPER_CONTEXT_KEYS
            declarations.extend(
                _paper_identifier_declarations(
                    child,
                    paper_context=child_context,
                    seen=seen,
                )
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        marker = id(value)
        if marker in seen:
            return declarations
        seen.add(marker)
        for child in value:
            declarations.extend(
                _paper_identifier_declarations(
                    child,
                    paper_context=paper_context,
                    seen=seen,
                )
            )
    return declarations


def _declared_identifier_value_is_safe(value: Any, key: str) -> bool:
    if isinstance(value, str):
        decoded = _stable_percent_decode(value)
        if (
            decoded is None
            or contains_unsafe_identifier_codepoint(value)
            or contains_unsafe_identifier_codepoint(decoded)
        ):
            return False
        return bool(_extract_identifiers(value, key_hint=key, paper_context=True))
    if isinstance(value, Mapping):
        return bool(value) and all(
            not _normalized_key_details(raw_key)[1]
            and _declared_identifier_value_is_safe(child, _normalized_key(raw_key))
            for raw_key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return bool(value) and all(
            _declared_identifier_value_is_safe(child, key) for child in value
        )
    return bool(_extract_identifiers(value, key_hint=key, paper_context=True))


def paper_identity_errors(row: Mapping[str, Any]) -> list[str]:
    """Return fail-closed errors for canonical and aliased paper identifiers."""

    canonical = canonical_paper_id(row.get("paper_id"))
    errors: set[str] = set()
    if "paper_id" not in row or not canonical:
        errors.add("paper_id must be a safe canonical paper identifier")

    declarations = _paper_identifier_declarations(row)
    for raw_key, key, value, unsafe_key in declarations:
        if unsafe_key:
            errors.add(f"paper identifier alias key {raw_key!r} contains unsafe characters")
        if not _declared_identifier_value_is_safe(value, key):
            errors.add(
                f"paper identifier alias {raw_key!r} must contain safe canonical identifiers"
            )

    if canonical and set(benchmark_paper_ids(row)) - {canonical}:
        errors.add("paper identifier aliases conflict with top-level paper_id")
    return sorted(errors)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load strict UTF-8 JSONL rows and reject duplicate object keys."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        handle = source.open("r", encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise BenchmarkAuditError(f"cannot read JSONL file {source}: {exc}") from exc
    try:
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_object_without_duplicate_keys,
                        parse_constant=_reject_json_constant,
                    )
                except _DuplicateJsonKeyError as exc:
                    raise BenchmarkAuditError(
                        f"duplicate JSON key {exc.args[0]!r} in {source} line {line_number}"
                    ) from exc
                except (json.JSONDecodeError, ValueError) as exc:
                    raise BenchmarkAuditError(
                        f"invalid JSON in {source} line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise BenchmarkAuditError(
                        f"JSONL row in {source} line {line_number} must be an object"
                    )
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        raise BenchmarkAuditError(f"cannot read JSONL file {source}: {exc}") from exc
    return rows


def _materialize_rows(value: Any, label: str) -> list[Any]:
    if isinstance(value, Mapping):
        row_markers = {"sample_id", "messages", "images", "model_task_prompt", "paper_id"}
        if row_markers.intersection(value):
            return [value]
        rows: list[Any] = []
        for key in sorted(value, key=lambda item: str(item)):
            row = value[key]
            if isinstance(row, Mapping) and "sample_id" not in row:
                row = {"sample_id": str(key), **row}
            rows.append(row)
        return rows
    if isinstance(value, (str, bytes, bytearray)):
        raise BenchmarkAuditError(f"{label} must be an iterable of JSON objects")
    try:
        return list(value)
    except TypeError as exc:
        raise BenchmarkAuditError(f"{label} must be an iterable of JSON objects") from exc


def _strict_sample_id(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        return None
    return value


def _message_texts(messages: Any, *, user_only: bool = False) -> list[str]:
    texts: list[str] = []
    if not isinstance(messages, list):
        return texts
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or message.get("from") or "").casefold()
        if user_only and role not in {"human", "user"}:
            continue
        content = message.get("content", message.get("value"))
        if isinstance(content, str) and content.strip():
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str) and item.strip():
                    texts.append(item)
                elif isinstance(item, Mapping):
                    item_type = str(item.get("type") or "").casefold()
                    text = item.get("text")
                    if item_type in {"", "text", "input_text"} and isinstance(text, str):
                        if text.strip():
                            texts.append(text)
    return texts


def _primary_prompt(row: Mapping[str, Any]) -> str:
    for field in _PROMPT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    user_texts = _message_texts(row.get("messages"), user_only=True)
    return "\n".join(user_texts).strip()


def _model_facing_texts(row: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for field in _PROMPT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            texts.append(value)
    texts.extend(_message_texts(row.get("messages")))
    unique: list[str] = []
    seen: set[str] = set()
    for text in texts:
        normalized = _normalize_text(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(text)
    return unique


def _validate_messages(messages: Any) -> tuple[int, list[str]]:
    if not isinstance(messages, list):
        return (0, ["messages must be an array"])
    errors: list[str] = []
    placeholders = 0
    has_user = False
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            errors.append(f"messages[{message_index}] must be an object")
            continue
        if set(message) != {"role", "content"}:
            errors.append(f"messages[{message_index}] must contain exactly role and content")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            errors.append(f"messages[{message_index}].role is unsupported")
        elif role == "user":
            has_user = True
        content = message.get("content")
        if isinstance(content, str):
            if not content.strip():
                errors.append(f"messages[{message_index}].content must be non-empty")
            continue
        if not isinstance(content, list):
            errors.append(f"messages[{message_index}].content must be text or an array")
            continue
        if not content:
            errors.append(f"messages[{message_index}].content must be non-empty")
        for content_index, item in enumerate(content):
            if not isinstance(item, Mapping):
                errors.append(
                    f"messages[{message_index}].content[{content_index}] must be an object"
                )
                continue
            item_type = item.get("type")
            if not isinstance(item_type, str) or not item_type:
                errors.append(
                    f"messages[{message_index}].content[{content_index}].type is required"
                )
                continue
            if item_type == "image":
                placeholders += 1
                if set(item) != {"type"}:
                    errors.append(
                        f"messages[{message_index}].content[{content_index}] must be "
                        "a path-free {'type': 'image'} placeholder"
                    )
            elif item_type == "text":
                if (
                    set(item) != {"type", "text"}
                    or not isinstance(item.get("text"), str)
                    or not item["text"].strip()
                ):
                    errors.append(
                        f"messages[{message_index}].content[{content_index}] must be an exact "
                        "non-empty text block"
                    )
            else:
                errors.append(
                    f"messages[{message_index}].content[{content_index}].type is unsupported"
                )
    if not has_user:
        errors.append("messages must contain a user role")
    return (placeholders, errors)


def _comparison_markers(value: str) -> list[str]:
    text = _normalize_text(value)
    markers: set[str] = set()
    if re.search(r"(?<!\w)(?:a\s*/\s*b|\u0430\s*/\s*\u0431)(?!\w)", text):
        markers.add("a/b wording")
    subjects = r"models?|variants?|answers?|responses?|outputs?|systems?|vlms?"
    if re.search(rf"\bcompar(?:e|es|ed|ing)\b[^.!?]{{0,160}}\b(?:{subjects})\b", text):
        markers.add("English comparison wording")
    if re.search(rf"\b(?:{subjects})\b[^.!?]{{0,160}}\bcompar(?:e|es|ed|ing)\b", text):
        markers.add("English comparison wording")
    if re.search(
        rf"\bwhich\b[^.!?]{{0,100}}\b(?:{subjects})\b[^.!?]{{0,100}}\b(?:better|best|more accurate)\b",
        text,
    ):
        markers.add("English model/variant selection wording")
    if re.search(rf"\b(?:each|both|two)\s+(?:{subjects})\b", text):
        markers.add("English multiple-model/variant wording")
    if re.search(r"\b(?:model|variant|answer|response|output)\s*(?:a|b|1|2)\b", text):
        markers.add("English labeled model/variant wording")

    if any(stem in text for stem in _RU_COMPARE_STEMS) and any(
        stem in text for stem in _RU_SUBJECT_STEMS
    ):
        markers.add("Russian comparison wording")
    if _RU_EACH_STEM in text and any(stem in text for stem in _RU_SUBJECT_STEMS):
        markers.add("Russian multiple-model/variant wording")
    if (
        _RU_WHICH in text
        and any(stem in text for stem in _RU_SUBJECT_STEMS)
        and any(stem in text for stem in _RU_BETTER_STEMS)
    ):
        markers.add("Russian model/variant selection wording")
    if any(word in text for word in _RU_TWO_WORDS) and any(
        stem in text for stem in _RU_SUBJECT_STEMS
    ):
        markers.add("Russian multiple-model/variant wording")
    if any(
        re.search(rf"\b{re.escape(stem)}\w*\s+(?:a|b|\u0430|\u0431)\b", text)
        for stem in _RU_SUBJECT_STEMS
        if stem != "vlm"
    ):
        markers.add("Russian labeled model/variant wording")
    return sorted(markers)


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        return bool(value)
    return True


def _normalized_identity(value: Any) -> str | None:
    identity = normalize_identity(value, ascii_reviewer=True)
    return identity or None


def _substantive_entry(value: Any) -> bool:
    return (isinstance(value, str) and bool(value.strip())) or (
        isinstance(value, Mapping) and bool(value)
    )


def _canonical_gold_errors(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    gold = row.get("gold_answer")
    if not isinstance(gold, Mapping):
        errors.append("gold_answer must be a top-level object")
    else:
        required_gold = {
            "answer",
            "evidence_used",
            "visual_facts",
            "temporal_facts",
            "uncertainty",
            "missing_evidence",
        }
        missing = sorted(required_gold - set(gold))
        if missing:
            errors.append(f"gold_answer is missing fields: {missing}")
        if not isinstance(gold.get("answer"), str) or not gold["answer"].strip():
            errors.append("gold_answer.answer must be non-empty")
        for field in ("evidence_used", "visual_facts", "temporal_facts"):
            values = gold.get(field)
            if (
                not isinstance(values, list)
                or not values
                or any(not _substantive_entry(value) for value in values)
            ):
                errors.append(f"gold_answer.{field} must contain substantive entries")

    rubric = row.get("rubric")
    if not isinstance(rubric, Mapping):
        errors.append("rubric must be a top-level object")
    else:
        criteria = rubric.get("criteria")
        if (
            not isinstance(criteria, list)
            or not criteria
            or any(not _substantive_entry(value) for value in criteria)
        ):
            errors.append("rubric.criteria must contain substantive entries")
        adjudicators = rubric.get("adjudicators")
        normalized = (
            [_normalized_identity(value) for value in adjudicators]
            if isinstance(adjudicators, list)
            else []
        )
        if (
            len(normalized) < 2
            or any(value is None for value in normalized)
            or len(set(normalized)) != len(normalized)
        ):
            errors.append("rubric.adjudicators requires normalized-distinct identifiers")
    return errors


def _evaluation_material(row: Mapping[str, Any]) -> tuple[dict[str, bool], list[str]]:
    presence = {
        "gold": _has_content(row.get("gold_answer")),
        "reference": _has_content(row.get("reference_answer")),
        "rubric": _has_content(row.get("rubric")),
    }
    values: list[str] = []
    seen: set[int] = set()

    def collect_strings(value: Any, local_seen: set[int]) -> None:
        if isinstance(value, str):
            if value.strip():
                values.append(value)
        elif isinstance(value, Mapping):
            marker = id(value)
            if marker in local_seen:
                return
            local_seen.add(marker)
            for child in value.values():
                collect_strings(child, local_seen)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            marker = id(value)
            if marker in local_seen:
                return
            local_seen.add(marker)
            for child in value:
                collect_strings(child, local_seen)

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
            for raw_key, child in value.items():
                key = _normalized_key(raw_key)
                category = ""
                if key in _GOLD_KEYS:
                    category = "gold"
                elif key in _REFERENCE_KEYS:
                    category = "reference"
                elif key in _RUBRIC_KEYS:
                    category = "rubric"
                if key in _LEAKAGE_VALUE_KEYS and _has_content(child):
                    collect_strings(child, set())
                if not category:
                    walk(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
            for child in value:
                walk(child)

    walk(row)
    return presence, values


def _answer_leakage_reasons(
    model_texts: Sequence[str], evaluation_values: Sequence[str]
) -> list[str]:
    combined = _normalize_text("\n".join(model_texts))
    if not combined:
        return []
    reasons: set[str] = set()
    english_patterns = (
        r"\b(?:gold|reference|expected|correct)\s+answer\b",
        r"\bground[\s_-]*truth\b",
        r"\b(?:evaluation|grading|scoring)\s+rubric\b",
        r"\b(?:creator[_ ]rationale|expected[_ ]error[_ ]modes?)\b",
    )
    if any(re.search(pattern, combined) for pattern in english_patterns):
        reasons.add("answer/rubric language appears in model-facing text")
    if any(stem in combined for stem in _RU_LEAK_STEMS) and _RU_ANSWER in combined:
        reasons.add("answer-key language appears in model-facing text")
    if _RU_CRITERIA in combined and _RU_ASSESSMENT in combined:
        reasons.add("rubric language appears in model-facing text")
    for value in evaluation_values:
        normalized = _normalize_text(value)
        if len(normalized) >= 8 and normalized in combined:
            reasons.add("evaluation material appears verbatim in model-facing text")
            break
    return sorted(reasons)


def _normalize_image_reference(value: Any) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("image path must be a string")
    text = os.fspath(value)
    if not text or "\x00" in text:
        raise ValueError("image path must be non-empty and contain no NUL")
    if _URI_SCHEME_RE.match(text) or text.startswith(("//", "\\\\")):
        raise ValueError("image path must be a local relative path")
    windows = PureWindowsPath(text)
    normalized_text = text.replace("\\", "/")
    posix = PurePosixPath(normalized_text)
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        raise ValueError("image path must be relative to dataset_root")
    if ".." in posix.parts:
        raise ValueError("image path contains parent traversal")
    parts = [part for part in posix.parts if part not in {"", "."}]
    if not parts:
        raise ValueError("image path does not name a file")
    if any(":" in part for part in parts):
        raise ValueError("image path contains a drive or alternate data stream marker")
    return "/".join(parts)


def _resolve_image(reference: Any, root: Path) -> tuple[str, Path]:
    normalized = _normalize_image_reference(reference)
    try:
        resolved = resolve_dataset_file(root, Path(*normalized.split("/")))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"image does not exist: {normalized}") from exc
    return normalized, resolved


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _extract_training_prompts(value: Any, *, combine_fragments: bool = False) -> set[str]:
    prompts: set[str] = set()
    seen: set[int] = set()
    user_fragments: list[str] = []
    prompt_field_fragments: list[str] = []

    def text_fragments(child: Any) -> list[str]:
        if isinstance(child, str):
            return [child] if child.strip() else []
        if isinstance(child, Mapping):
            item_type = str(child.get("type") or "").casefold()
            text = child.get("text")
            if item_type in {"text", "input_text"} and isinstance(text, str):
                return [text] if text.strip() else []
            fragments: list[str] = []
            for key, nested in child.items():
                if _normalized_key(key) in {"from", "role", "type"}:
                    continue
                fragments.extend(text_fragments(nested))
            return fragments
        if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
            fragments = []
            for nested in child:
                fragments.extend(text_fragments(nested))
            return fragments
        return []

    def add_combined_variants(fragments: list[str]) -> None:
        if len(fragments) < 2:
            return
        for separator in ("\n", ""):
            combined = _normalize_text(separator.join(fragments))
            if combined:
                prompts.add(combined)

    def add_strings(child: Any) -> None:
        if combine_fragments:
            fragments = text_fragments(child)
            for fragment in fragments:
                normalized = _normalize_text(fragment)
                if normalized:
                    prompts.add(normalized)
            add_combined_variants(fragments)
            return
        if isinstance(child, str):
            normalized = _normalize_text(child)
            if normalized:
                prompts.add(normalized)
        elif isinstance(child, Mapping):
            for nested in child.values():
                add_strings(nested)
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
            for nested in child:
                add_strings(nested)

    def walk(child: Any) -> None:
        if isinstance(child, Mapping):
            marker = id(child)
            if marker in seen:
                return
            seen.add(marker)
            role = str(child.get("role") or child.get("from") or "").casefold()
            if role in {"human", "user"}:
                content = child.get("content", child.get("value"))
                if combine_fragments:
                    user_fragments.extend(text_fragments(content))
                add_strings(content)
            for raw_key, nested in child.items():
                key = _normalized_key(raw_key)
                if key in _TRAINING_PROMPT_KEYS:
                    if combine_fragments:
                        prompt_field_fragments.extend(text_fragments(nested))
                    add_strings(nested)
                else:
                    walk(nested)
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
            marker = id(child)
            if marker in seen:
                return
            seen.add(marker)
            for nested in child:
                walk(nested)

    walk(value)
    if combine_fragments:
        for fragments in (user_fragments, prompt_field_fragments):
            add_combined_variants(fragments)
        prompts.update(compact for prompt in list(prompts) if (compact := _compact_prompt(prompt)))
    return prompts


def _prompt_overlap_entry(
    prompt: str, sample_ids: Iterable[str], *, training_rows: Iterable[int] | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_preview": prompt[:160],
        "sample_ids": sorted(set(sample_ids)),
    }
    if training_rows is not None:
        entry["training_row_indices"] = sorted(set(training_rows))
    return entry


def _provenance_image_entries(
    row: Mapping[str, Any], expected_paths: Sequence[str]
) -> tuple[list[tuple[str, str | None]], list[str]]:
    errors: list[str] = []
    raw_images: Any = None
    image_field = ""
    for field in ("images", "image_paths", "image", "image_path", "path"):
        if field in row:
            raw_images = row[field]
            image_field = field
            break
    if raw_images is None and isinstance(row.get("image_index"), int):
        index = row["image_index"]
        if 0 <= index < len(expected_paths):
            raw_images = expected_paths[index]
        else:
            errors.append(f"image_index {index} is outside the benchmark image list")
    if (
        raw_images is None
        and len(expected_paths) == 1
        and any(field in row for field in ("hash", "image_hashes", "image_sha256", "sha256"))
    ):
        raw_images = expected_paths[0]
    if raw_images is None:
        references: list[Any] = []
    elif isinstance(raw_images, (str, os.PathLike, Mapping)):
        references = [raw_images]
    elif isinstance(raw_images, Sequence) and not isinstance(raw_images, (str, bytes, bytearray)):
        references = list(raw_images)
    else:
        errors.append(f"{image_field or 'image'} must be a path or an array")
        references = []

    top_hashes = row.get(
        "image_hashes",
        row.get("image_sha256", row.get("sha256", row.get("hash"))),
    )
    entries: list[tuple[str, str | None]] = []
    for index, reference in enumerate(references):
        declared_hash: Any = None
        if isinstance(reference, Mapping):
            path_value: Any = None
            for field in _IMAGE_PATH_KEYS:
                if field in reference:
                    path_value = reference[field]
                    break
            declared_hash = reference.get(
                "sha256", reference.get("image_sha256", reference.get("hash"))
            )
            reference = path_value
        if declared_hash is None:
            if isinstance(top_hashes, Mapping):
                declared_hash = top_hashes.get(str(reference))
            elif isinstance(top_hashes, Sequence) and not isinstance(
                top_hashes, (str, bytes, bytearray)
            ):
                if index < len(top_hashes):
                    declared_hash = top_hashes[index]
            elif len(references) == 1:
                declared_hash = top_hashes
        try:
            normalized = _normalize_image_reference(reference)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid provenance image at index {index}: {exc}")
            continue
        normalized_hash: str | None = None
        if declared_hash not in (None, ""):
            if not isinstance(declared_hash, str) or not _SHA256_RE.fullmatch(declared_hash):
                errors.append(f"invalid SHA256 for provenance image {normalized}")
            else:
                normalized_hash = declared_hash.casefold()
        entries.append((normalized, normalized_hash))
    return entries, errors


def _complete_provenance_errors(
    row: Mapping[str, Any], *, require_citation: bool = False
) -> list[str]:
    raw_entries = row.get("images")
    if (
        isinstance(raw_entries, list)
        and raw_entries
        and all(isinstance(entry, Mapping) for entry in raw_entries)
    ):
        entries = list(raw_entries)
    else:
        entries = [row]
    errors: list[str] = []
    for index, entry in enumerate(entries):
        prefix = f"provenance image {index}"
        digest = entry.get("sha256", entry.get("image_sha256", entry.get("hash")))
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(f"{prefix} requires a lowercase SHA256")
        page = entry.get("page")
        if isinstance(page, bool) or not isinstance(page, (int, str)) or str(page).strip() == "":
            errors.append(f"{prefix} requires page")
        required_text_fields = ["locator", "source_url", "license"]
        if require_citation:
            required_text_fields.append("citation")
        for field in required_text_fields:
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix} requires {field}")
        source_url = entry.get("source_url")
        if isinstance(source_url, str) and not re.match(r"^https?://", source_url):
            errors.append(f"{prefix} source_url must be HTTP(S)")
        verified_by = entry.get("verified_by")
        normalized_verifiers = (
            [_normalized_identity(value) for value in verified_by]
            if isinstance(verified_by, list)
            else []
        )
        if (
            len(normalized_verifiers) < 2
            or any(value is None for value in normalized_verifiers)
            or len(set(normalized_verifiers)) != len(normalized_verifiers)
        ):
            errors.append(f"{prefix} requires two distinct verified_by identifiers")
    return errors


def _finding_sort_key(finding: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(finding.get("code", "")),
        "\x00".join(str(value) for value in finding.get("sample_ids", [])),
        json.dumps(finding.get("details", {}), ensure_ascii=False, sort_keys=True),
    )


def audit_benchmark(
    rows: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    dataset_root: str | os.PathLike[str],
    *,
    provenance_rows: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    training_rows: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    training_lineage: Mapping[str, Any] | None = None,
    expected_training_sources: Sequence[Mapping[str, Any]] | None = None,
    require_gold: bool = False,
    require_complete_provenance: bool = False,
    require_split_provenance: bool = False,
    require_training_lineage: bool = False,
    require_provenance_order: bool = False,
    require_citation: bool = False,
    blocked_warning_codes: Sequence[str] = (),
    minimum_primary_papers: int = 0,
    exact_primary_papers: int | None = None,
    primary_strata: Sequence[str] = ("multimodal_hard", "temporal_hard"),
    audit_version: int = AUDIT_VERSION,
) -> dict[str, Any]:
    """Audit benchmark rows and return a deterministic JSON-serializable report."""

    if (
        isinstance(audit_version, bool)
        or not isinstance(audit_version, int)
        or audit_version not in _SUPPORTED_AUDIT_VERSIONS
    ):
        raise BenchmarkAuditError(
            f"audit_version must be one of {sorted(_SUPPORTED_AUDIT_VERSIONS)}"
        )
    if not isinstance(require_gold, bool):
        raise BenchmarkAuditError("require_gold must be a boolean")
    if not isinstance(require_complete_provenance, bool):
        raise BenchmarkAuditError("require_complete_provenance must be a boolean")
    if not isinstance(require_split_provenance, bool):
        raise BenchmarkAuditError("require_split_provenance must be a boolean")
    if not isinstance(require_training_lineage, bool):
        raise BenchmarkAuditError("require_training_lineage must be a boolean")
    if not isinstance(require_provenance_order, bool):
        raise BenchmarkAuditError("require_provenance_order must be a boolean")
    if not isinstance(require_citation, bool):
        raise BenchmarkAuditError("require_citation must be a boolean")
    if isinstance(blocked_warning_codes, (str, bytes)) or any(
        not isinstance(code, str) or not code for code in blocked_warning_codes
    ):
        raise BenchmarkAuditError("blocked_warning_codes must contain non-empty strings")
    blocked_warnings = set(blocked_warning_codes)
    if (
        isinstance(minimum_primary_papers, bool)
        or not isinstance(minimum_primary_papers, int)
        or minimum_primary_papers < 0
    ):
        raise BenchmarkAuditError("minimum_primary_papers must be a non-negative integer")
    if exact_primary_papers is not None and (
        isinstance(exact_primary_papers, bool)
        or not isinstance(exact_primary_papers, int)
        or exact_primary_papers <= 0
    ):
        raise BenchmarkAuditError("exact_primary_papers must be a positive integer or None")
    if training_lineage is not None and not isinstance(training_lineage, Mapping):
        raise BenchmarkAuditError("training_lineage must be an object")
    normalized_primary_strata = set(primary_strata)
    if not normalized_primary_strata or not normalized_primary_strata <= {
        "multimodal_hard",
        "temporal_hard",
        "easy_control",
    }:
        raise BenchmarkAuditError("primary_strata contains an unsupported stratum")
    try:
        root = Path(dataset_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BenchmarkAuditError(f"dataset_root does not exist: {dataset_root}") from exc
    if not root.is_dir():
        raise BenchmarkAuditError("dataset_root must be a directory")

    benchmark_rows = _materialize_rows(rows, "rows")
    provenance = (
        None if provenance_rows is None else _materialize_rows(provenance_rows, "provenance_rows")
    )
    training = None if training_rows is None else _materialize_rows(training_rows, "training_rows")

    critical_findings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    per_sample: dict[str, dict[str, Any]] = {}

    def ensure_sample(sample_key: str) -> dict[str, Any]:
        if sample_key not in per_sample:
            per_sample[sample_key] = {
                "row_indices": [],
                "canonical_paper_ids": [],
                "prompt_sha256s": [],
                "image_hashes": [],
                "gold_reference_rubric": {
                    "gold": False,
                    "reference": False,
                    "rubric": False,
                },
                "critical_findings": [],
                "warnings": [],
                "eligible": False,
            }
        return per_sample[sample_key]

    def add_finding(
        severity: str,
        code: str,
        message: str,
        sample_keys: Iterable[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if severity == "warning" and code in blocked_warnings:
            severity = "critical"
        keys = sorted(set(sample_keys))
        finding: dict[str, Any] = {
            "code": code,
            "message": message,
            "sample_ids": keys,
        }
        if details:
            finding["details"] = dict(details)
        target = critical_findings if severity == "critical" else warnings
        target.append(finding)
        per_key_field = "critical_findings" if severity == "critical" else "warnings"
        for key in keys:
            ensure_sample(key)[per_key_field].append(finding)

    prepared: list[dict[str, Any]] = []
    valid_id_indices: dict[str, list[int]] = defaultdict(list)
    valid_sample_ids: set[str] = set()
    for row_index, raw_row in enumerate(benchmark_rows):
        row = raw_row if isinstance(raw_row, Mapping) else {}
        sample_id = _strict_sample_id(row.get("sample_id")) if row else None
        sample_key = sample_id if sample_id is not None else f"<row:{row_index:06d}>"
        sample_report = ensure_sample(sample_key)
        sample_report["row_indices"].append(row_index)
        if sample_id is not None:
            valid_id_indices[sample_id].append(row_index)
            valid_sample_ids.add(sample_id)
        prepared.append(
            {
                "row_index": row_index,
                "row": row,
                "raw_is_mapping": isinstance(raw_row, Mapping),
                "sample_id": sample_id,
                "sample_key": sample_key,
                "paper_ids": [],
                "primary_paper_id": "",
                "prompt": "",
                "image_paths": [],
                "hashes_by_path": defaultdict(set),
            }
        )

    if not benchmark_rows:
        add_finding(
            "critical",
            "empty_benchmark",
            "The benchmark contains no rows.",
        )

    duplicate_ids = {
        sample_id: indices for sample_id, indices in valid_id_indices.items() if len(indices) > 1
    }
    for sample_id in sorted(duplicate_ids):
        add_finding(
            "critical",
            "duplicate_sample_id",
            f"sample_id {sample_id!r} occurs more than once.",
            [sample_id],
            {"row_indices": duplicate_ids[sample_id]},
        )

    image_cache: dict[Path, tuple[str, int]] = {}
    image_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_samples: dict[str, list[str]] = defaultdict(list)
    declared_image_count = 0
    hashed_image_count = 0
    within_row_image_duplicates: list[dict[str, Any]] = []

    for item in prepared:
        row = item["row"]
        sample_key = item["sample_key"]
        schema_errors: list[str] = []
        if not item["raw_is_mapping"]:
            schema_errors.append("row must be an object")
        if item["sample_id"] is None:
            schema_errors.append("sample_id must be a non-empty, trimmed string")

        paper_ids = benchmark_paper_ids(row)
        declared_paper_id = canonical_paper_id(row.get("paper_id"))
        item["paper_ids"] = paper_ids
        item["primary_paper_id"] = declared_paper_id or (paper_ids[0] if paper_ids else "")
        sample_report = ensure_sample(sample_key)
        sample_report["canonical_paper_ids"] = sorted(
            set(sample_report["canonical_paper_ids"]) | set(paper_ids),
            key=_paper_sort_key,
        )
        if audit_version >= 2:
            schema_errors.extend(paper_identity_errors(row))
        elif not paper_ids:
            schema_errors.append("a non-empty paper_id, DOI, or arXiv identifier is required")

        messages = row.get("messages")
        if "messages" not in row:
            schema_errors.append("messages is required")
        placeholder_count, message_errors = _validate_messages(messages)
        schema_errors.extend(message_errors)

        model_task_prompt = row.get("model_task_prompt")
        if not isinstance(model_task_prompt, str) or not model_task_prompt.strip():
            schema_errors.append("model_task_prompt must be a non-empty string")
        else:
            canonical_user_prompt = "\n".join(_message_texts(messages, user_only=True)).strip()
            if _normalize_text(model_task_prompt) != _normalize_text(canonical_user_prompt):
                schema_errors.append(
                    "model_task_prompt must exactly match the canonical user message text"
                )
        prompt = _primary_prompt(row)
        item["prompt"] = prompt
        if not prompt:
            schema_errors.append("a non-empty model task prompt is required")
        else:
            normalized_prompt = _normalize_text(prompt)
            prompt_samples[normalized_prompt].append(sample_key)
            prompt_digest = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
            sample_report["prompt_sha256s"] = sorted(
                set(sample_report["prompt_sha256s"]) | {prompt_digest}
            )

        stratum = row.get("stratum")
        if stratum not in {
            "multimodal_hard",
            "temporal_hard",
            "easy_control",
        }:
            schema_errors.append("stratum must be a declared benchmark stratum")
        if not isinstance(row.get("primary_endpoint"), bool):
            schema_errors.append("primary_endpoint must be boolean")
        elif stratum in {"multimodal_hard", "temporal_hard", "easy_control"} and row[
            "primary_endpoint"
        ] != (stratum in normalized_primary_strata):
            schema_errors.append("primary_endpoint is inconsistent with primary_strata")
        if require_split_provenance:
            split_provenance = row.get("split_provenance")
            if not isinstance(split_provenance, Mapping):
                schema_errors.append("split_provenance must be an object")
            else:
                for field in (
                    "paper_holdout",
                    "source_holdout",
                    "creator_holdout",
                    "training_overlap_checked",
                ):
                    if split_provenance.get(field) is not True:
                        schema_errors.append(f"split_provenance.{field} must be true")
                for field in ("source_document_id", "creator_group_id"):
                    value = split_provenance.get(field)
                    if not isinstance(value, str) or not value.strip() or value != value.strip():
                        schema_errors.append(
                            f"split_provenance.{field} must be a non-empty, trimmed string"
                        )

        raw_images = row.get("images")
        if "images" not in row:
            schema_errors.append("images is required")
            images: list[Any] = []
        elif not isinstance(raw_images, list):
            schema_errors.append("images must be an array of local relative paths")
            images = []
        else:
            images = raw_images
        declared_image_count += len(images)
        if not images:
            add_finding(
                "critical",
                "missing_image",
                "The VLM benchmark row declares no local evidence images.",
                [sample_key],
                {"row_index": item["row_index"]},
            )
        for image_index, reference in enumerate(images):
            if not isinstance(reference, str):
                schema_errors.append(f"images[{image_index}] must be a string path")

        if schema_errors:
            add_finding(
                "critical",
                "schema_error",
                "The row does not satisfy the Task 3 benchmark schema.",
                [sample_key],
                {"errors": sorted(set(schema_errors)), "row_index": item["row_index"]},
            )

        if placeholder_count != len(images):
            add_finding(
                "critical",
                "image_placeholder_mismatch",
                "The number of message image placeholders does not match images.",
                [sample_key],
                {
                    "declared_images": len(images),
                    "image_placeholders": placeholder_count,
                    "row_index": item["row_index"],
                },
            )

        row_hashes: dict[str, list[str]] = defaultdict(list)
        for image_index, reference in enumerate(images):
            if not isinstance(reference, str):
                continue
            try:
                normalized_path, resolved = _resolve_image(reference, root)
            except FileNotFoundError as exc:
                add_finding(
                    "critical",
                    "missing_image",
                    str(exc),
                    [sample_key],
                    {"image_index": image_index, "path": reference},
                )
                continue
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                add_finding(
                    "critical",
                    "unsafe_image_path",
                    f"Unsafe image path {reference!r}: {exc}",
                    [sample_key],
                    {"image_index": image_index, "path": reference},
                )
                continue
            try:
                if resolved not in image_cache:
                    image_cache[resolved] = _sha256_file(resolved)
                digest, size = image_cache[resolved]
            except OSError as exc:
                add_finding(
                    "critical",
                    "missing_image",
                    f"Image cannot be read: {normalized_path}",
                    [sample_key],
                    {"error": str(exc), "image_index": image_index, "path": normalized_path},
                )
                continue
            hashed_image_count += 1
            item["image_paths"].append(normalized_path)
            item["hashes_by_path"][normalized_path].add(digest)
            hash_record = {
                "path": normalized_path,
                "sha256": digest,
                "size_bytes": size,
                "row_index": item["row_index"],
            }
            sample_report["image_hashes"].append(hash_record)
            row_hashes[digest].append(normalized_path)
            image_occurrences[digest].append(
                {
                    "sample_key": sample_key,
                    "path": normalized_path,
                    "paper_id": item["primary_paper_id"],
                    "row_index": item["row_index"],
                    "size_bytes": size,
                }
            )

        for digest, paths in sorted(row_hashes.items()):
            if len(paths) < 2:
                continue
            detail = {
                "paths": sorted(paths),
                "row_index": item["row_index"],
                "sha256": digest,
            }
            within_row_image_duplicates.append({"sample_id": sample_key, **detail})
            add_finding(
                "warning",
                "within_row_duplicate_image_bytes",
                "Multiple image entries in this row have identical bytes.",
                [sample_key],
                detail,
            )

        model_texts = _model_facing_texts(row)
        comparison_markers = sorted(
            {marker for text in model_texts for marker in _comparison_markers(text)}
        )
        if comparison_markers:
            add_finding(
                "critical",
                "residual_comparison_prompt",
                "Model-facing text contains residual A/B, model, or variant comparison wording.",
                [sample_key],
                {"markers": comparison_markers, "row_index": item["row_index"]},
            )

        presence, evaluation_values = _evaluation_material(row)
        for category in presence:
            sample_report["gold_reference_rubric"][category] = (
                sample_report["gold_reference_rubric"][category] or presence[category]
            )
        canonical_gold_errors = _canonical_gold_errors(row) if require_gold else []
        missing_required_gold = bool(canonical_gold_errors)
        if missing_required_gold or not any(presence.values()):
            add_finding(
                "critical" if missing_required_gold else "warning",
                "missing_gold",
                (
                    "An adjudicated gold answer and evaluation rubric are required."
                    if missing_required_gold
                    else "No gold answer, reference answer, or evaluation rubric is present."
                ),
                [sample_key],
                (
                    {
                        "errors": canonical_gold_errors,
                        "row_index": item["row_index"],
                    }
                    if canonical_gold_errors
                    else {"row_index": item["row_index"]}
                ),
            )
        leakage_reasons = _answer_leakage_reasons(model_texts, evaluation_values)
        if leakage_reasons:
            add_finding(
                "critical",
                "likely_answer_leakage",
                "Model-facing text likely exposes answer or evaluation material.",
                [sample_key],
                {"reasons": leakage_reasons, "row_index": item["row_index"]},
            )

    cross_paper_image_reuse: list[dict[str, Any]] = []
    for digest, occurrences in sorted(image_occurrences.items()):
        paper_ids = sorted(
            {entry["paper_id"] for entry in occurrences if entry["paper_id"]},
            key=_paper_sort_key,
        )
        if len(paper_ids) < 2:
            continue
        sample_keys = sorted({entry["sample_key"] for entry in occurrences})
        references = sorted(
            (
                {
                    "paper_id": entry["paper_id"],
                    "path": entry["path"],
                    "row_index": entry["row_index"],
                    "sample_id": entry["sample_key"],
                }
                for entry in occurrences
            ),
            key=lambda entry: (
                entry["sample_id"],
                entry["paper_id"],
                entry["path"],
                entry["row_index"],
            ),
        )
        detail = {
            "paper_ids": paper_ids,
            "references": references,
            "sample_ids": sample_keys,
            "sha256": digest,
            "interpretation": (
                "SHA256 equality establishes byte reuse only; it does not prove that the "
                "records describe the same semantic paper."
            ),
        }
        cross_paper_image_reuse.append(detail)
        add_finding(
            "critical",
            "cross_paper_image_reuse",
            (
                "Identical image bytes are referenced under different canonical paper IDs. "
                "This establishes byte reuse only, not semantic paper identity."
            ),
            sample_keys,
            detail,
        )

    benchmark_prompt_overlaps: list[dict[str, Any]] = []
    for prompt, sample_keys in sorted(prompt_samples.items()):
        if len(sample_keys) < 2:
            continue
        entry = _prompt_overlap_entry(prompt, sample_keys)
        entry["row_count"] = len(sample_keys)
        benchmark_prompt_overlaps.append(entry)
        add_finding(
            "warning",
            "duplicate_normalized_prompt",
            "The exact normalized model task prompt occurs in multiple benchmark rows.",
            sample_keys,
            entry,
        )

    training_prompt_overlaps: list[dict[str, Any]] = []
    training_paper_overlaps: list[dict[str, Any]] = []
    training_identifiers: dict[str, list[int]] = defaultdict(list)
    training_prompts: dict[str, list[int]] = defaultdict(list)
    if training is not None:
        for training_index, training_row in enumerate(training):
            if not isinstance(training_row, Mapping):
                add_finding(
                    "warning",
                    "invalid_training_row",
                    "A training row is not a JSON object and could not be checked.",
                    details={"training_row_index": training_index},
                )
                continue
            for identifier in sorted(_extract_identifiers(training_row), key=_paper_sort_key):
                training_identifiers[identifier].append(training_index)
            for prompt in sorted(
                _extract_training_prompts(
                    training_row,
                    combine_fragments=audit_version >= 3,
                )
            ):
                training_prompts[prompt].append(training_index)

        benchmark_identifier_samples: dict[str, set[str]] = defaultdict(set)
        for item in prepared:
            for identifier in item["paper_ids"]:
                benchmark_identifier_samples[identifier].add(item["sample_key"])
        for identifier in sorted(
            set(benchmark_identifier_samples).intersection(training_identifiers),
            key=_paper_sort_key,
        ):
            sample_keys = sorted(benchmark_identifier_samples[identifier])
            entry = {
                "paper_id": identifier,
                "sample_ids": sample_keys,
                "training_row_indices": sorted(set(training_identifiers[identifier])),
            }
            training_paper_overlaps.append(entry)
            add_finding(
                "critical",
                "training_paper_overlap",
                f"Canonical paper identifier {identifier!r} occurs in actual training rows.",
                sample_keys,
                entry,
            )

        if audit_version >= 3:
            benchmark_by_key: dict[str, dict[str, set[str]]] = {}
            for prompt, sample_keys in prompt_samples.items():
                key = _compact_prompt(prompt)
                record = benchmark_by_key.setdefault(key, {"prompts": set(), "samples": set()})
                record["prompts"].add(prompt)
                record["samples"].update(sample_keys)
            training_by_key: dict[str, set[int]] = defaultdict(set)
            for prompt, training_rows in training_prompts.items():
                training_by_key[_compact_prompt(prompt)].update(training_rows)
            for key in sorted(set(benchmark_by_key).intersection(training_by_key)):
                prompt = sorted(benchmark_by_key[key]["prompts"])[0]
                sample_keys = sorted(benchmark_by_key[key]["samples"])
                entry = _prompt_overlap_entry(
                    prompt,
                    sample_keys,
                    training_rows=training_by_key[key],
                )
                entry["match_normalization"] = "unicode-casefold-whitespace-insensitive"
                training_prompt_overlaps.append(entry)
                add_finding(
                    "critical",
                    "training_prompt_overlap",
                    "The benchmark prompt occurs in actual training rows after strict normalization.",
                    sample_keys,
                    entry,
                )
        else:
            for prompt in sorted(set(prompt_samples).intersection(training_prompts)):
                sample_keys = sorted(set(prompt_samples[prompt]))
                entry = _prompt_overlap_entry(
                    prompt,
                    sample_keys,
                    training_rows=training_prompts[prompt],
                )
                training_prompt_overlaps.append(entry)
                add_finding(
                    "critical",
                    "training_prompt_overlap",
                    "The exact normalized benchmark prompt occurs in actual training rows.",
                    sample_keys,
                    entry,
                )

    lineage_audit: dict[str, Any] = {
        "provided": training_lineage is not None,
        "issues": [],
        "overlaps": {},
    }
    if training_lineage is None:
        if require_training_lineage:
            add_finding(
                "critical",
                "missing_training_lineage",
                "A complete immutable training lineage manifest is required.",
            )
    else:
        lineage_issues: list[str] = []
        required_fields = {
            "schema_version",
            "training_sources",
            "coverage",
            "paper_ids",
            "source_document_ids",
            "creator_group_ids",
            "image_sha256s",
            "prompt_sha256s",
        }
        if set(training_lineage) != required_fields:
            lineage_issues.append("training lineage fields do not match schema version 1")
        if training_lineage.get("schema_version") != 1:
            lineage_issues.append("training lineage schema_version must be 1")
        coverage = training_lineage.get("coverage")
        required_coverage = {
            "paper_ids",
            "source_documents",
            "creator_groups",
            "image_bytes",
            "prompts",
        }
        if not isinstance(coverage, Mapping) or any(
            coverage.get(field) is not True for field in required_coverage
        ):
            lineage_issues.append("training lineage coverage must explicitly cover every domain")

        expected_sources = [
            {
                "repo_id": source.get("repo_id"),
                "repo_type": source.get("repo_type", "model"),
                "revision": source.get("revision"),
                "files": [
                    dict(file) if isinstance(file, Mapping) else file
                    for file in source.get("files", [])
                ],
            }
            for source in (expected_training_sources or [])
        ]
        declared_sources = training_lineage.get("training_sources")
        if not isinstance(declared_sources, list) or declared_sources != expected_sources:
            lineage_issues.append("training lineage sources differ from configured training inputs")
        declared_row_count = 0
        if isinstance(declared_sources, list):
            for source in declared_sources:
                files = source.get("files") if isinstance(source, Mapping) else None
                if not isinstance(files, list) or not files:
                    lineage_issues.append("each training lineage source must list files")
                    continue
                for file in files:
                    if not isinstance(file, Mapping):
                        lineage_issues.append("training lineage file records must be objects")
                        continue
                    path = file.get("path")
                    digest = file.get("sha256")
                    row_count = file.get("row_count")
                    if (
                        set(file) != {"path", "sha256", "row_count"}
                        or not isinstance(path, str)
                        or not path.strip()
                        or path != path.strip()
                        or not isinstance(digest, str)
                        or not _SHA256_RE.fullmatch(digest)
                        or isinstance(row_count, bool)
                        or not isinstance(row_count, int)
                        or row_count < 0
                    ):
                        lineage_issues.append("training lineage file records are invalid")
                        continue
                    declared_row_count += row_count
        if training is not None and declared_row_count != len(training):
            lineage_issues.append(
                "training lineage row counts differ from the audited training files"
            )

        lineage_sets: dict[str, set[str]] = {}
        for field in (
            "paper_ids",
            "source_document_ids",
            "creator_group_ids",
            "image_sha256s",
            "prompt_sha256s",
        ):
            values = training_lineage.get(field)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or any(value != value.strip() for value in values if isinstance(value, str))
            ):
                lineage_issues.append(
                    f"training lineage {field} must be a non-empty trimmed string array"
                )
                lineage_sets[field] = set()
                continue
            if field == "paper_ids":
                normalized = [canonical_paper_id(value) for value in values]
            elif field in {"source_document_ids", "creator_group_ids"}:
                normalized = [_normalize_text(value) for value in values]
            else:
                normalized = [value.lower() for value in values]
                if any(not _SHA256_RE.fullmatch(value) for value in values):
                    lineage_issues.append(f"training lineage {field} contains an invalid SHA256")
            if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
                lineage_issues.append(
                    f"training lineage {field} contains duplicate or invalid normalized values"
                )
            lineage_sets[field] = {value for value in normalized if value}

        canonical_training_papers = lineage_sets.get("paper_ids", set())
        actual_training_papers = set(training_identifiers)
        if actual_training_papers - canonical_training_papers:
            lineage_issues.append("training lineage omits paper IDs found in audited training rows")
        actual_training_prompt_hashes = {
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in training_prompts
        }
        if actual_training_prompt_hashes - lineage_sets.get("prompt_sha256s", set()):
            lineage_issues.append("training lineage omits prompts found in audited training rows")
        benchmark_papers = {identifier for item in prepared for identifier in item["paper_ids"]}
        benchmark_sources = {
            _normalize_text(str(split["source_document_id"]))
            for item in prepared
            if isinstance((split := item["row"].get("split_provenance")), Mapping)
            and isinstance(split.get("source_document_id"), str)
        }
        benchmark_creators = {
            _normalize_text(str(split["creator_group_id"]))
            for item in prepared
            if isinstance((split := item["row"].get("split_provenance")), Mapping)
            and isinstance(split.get("creator_group_id"), str)
        }
        benchmark_hashes = set(image_occurrences)
        benchmark_prompt_hashes = {
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompt_samples
        }
        if audit_version >= 3:
            benchmark_prompt_hashes.update(
                hashlib.sha256(_compact_prompt(prompt).encode("utf-8")).hexdigest()
                for prompt in prompt_samples
            )
        overlaps = {
            "paper_ids": sorted(benchmark_papers & canonical_training_papers, key=_paper_sort_key),
            "source_document_ids": sorted(
                benchmark_sources & lineage_sets.get("source_document_ids", set())
            ),
            "creator_group_ids": sorted(
                benchmark_creators & lineage_sets.get("creator_group_ids", set())
            ),
            "image_sha256s": sorted(benchmark_hashes & lineage_sets.get("image_sha256s", set())),
            "prompt_sha256s": sorted(
                benchmark_prompt_hashes & lineage_sets.get("prompt_sha256s", set())
            ),
        }
        lineage_audit["overlaps"] = overlaps
        lineage_audit["issues"] = sorted(set(lineage_issues))
        if lineage_issues:
            add_finding(
                "critical",
                "invalid_training_lineage",
                "The training lineage manifest is incomplete or inconsistent.",
                details={"issues": sorted(set(lineage_issues))},
            )
        nonempty_overlaps = {field: values for field, values in overlaps.items() if values}
        if nonempty_overlaps:
            add_finding(
                "critical",
                "training_lineage_overlap",
                "The benchmark intersects the immutable training lineage manifest.",
                details={"overlaps": nonempty_overlaps},
            )

    provenance_mismatches: list[dict[str, Any]] = []
    provenance_required = (
        require_complete_provenance or require_provenance_order or require_citation
    )
    if provenance_required and provenance is None:
        add_finding(
            "critical",
            "provenance_mismatch",
            "Complete image provenance is required but no provenance rows were supplied.",
        )
    if provenance is not None:
        expected_by_id: dict[str, dict[str, Any]] = {}
        for item in prepared:
            sample_id = item["sample_id"]
            if sample_id is None:
                continue
            expected = expected_by_id.setdefault(
                sample_id,
                {
                    "images": [],
                    "papers": set(),
                    "hashes": defaultdict(set),
                },
            )
            expected["images"].extend(item["image_paths"])
            expected["papers"].update(item["paper_ids"])
            for path, hashes in item["hashes_by_path"].items():
                expected["hashes"][path].update(hashes)

        actual_by_id: dict[str, dict[str, Any]] = {}
        for provenance_index, provenance_row in enumerate(provenance):
            if not isinstance(provenance_row, Mapping):
                detail = {
                    "issues": ["provenance row must be an object"],
                    "provenance_row_index": provenance_index,
                }
                provenance_mismatches.append(detail)
                add_finding(
                    "critical",
                    "provenance_mismatch",
                    "A provenance row is malformed.",
                    details=detail,
                )
                continue
            sample_id = _strict_sample_id(provenance_row.get("sample_id"))
            if sample_id is None:
                detail = {
                    "issues": ["provenance sample_id must be a non-empty, trimmed string"],
                    "provenance_row_index": provenance_index,
                }
                provenance_mismatches.append(detail)
                add_finding(
                    "critical",
                    "provenance_mismatch",
                    "A provenance row has no valid sample mapping.",
                    details=detail,
                )
                continue
            if sample_id not in expected_by_id:
                detail = {
                    "issues": [f"unknown benchmark sample_id {sample_id!r}"],
                    "provenance_row_index": provenance_index,
                    "sample_id": sample_id,
                }
                provenance_mismatches.append(detail)
                add_finding(
                    "critical",
                    "provenance_mismatch",
                    "Provenance references an unknown benchmark sample.",
                    details=detail,
                )
                continue
            expected_paths = expected_by_id[sample_id]["images"]
            entries, entry_errors = _provenance_image_entries(provenance_row, expected_paths)
            actual = actual_by_id.setdefault(
                sample_id,
                {
                    "images": [],
                    "papers": set(),
                    "hashes": defaultdict(set),
                    "errors": [],
                    "rows": [],
                },
            )
            actual["rows"].append(provenance_index)
            actual["errors"].extend(entry_errors)
            if require_complete_provenance or require_citation:
                actual["errors"].extend(
                    _complete_provenance_errors(
                        provenance_row,
                        require_citation=require_citation,
                    )
                )
            provenance_paper_ids = benchmark_paper_ids(provenance_row)
            if audit_version >= 2:
                actual["errors"].extend(
                    f"provenance {error}" for error in paper_identity_errors(provenance_row)
                )
            actual["papers"].update(provenance_paper_ids)
            for path, digest in entries:
                actual["images"].append(path)
                if digest:
                    actual["hashes"][path].add(digest)

        for sample_id in sorted(expected_by_id):
            expected = expected_by_id[sample_id]
            actual = actual_by_id.get(sample_id)
            issues: list[str] = []
            if actual is None:
                issues.append("sample has no provenance row")
                provenance_rows_for_sample: list[int] = []
            else:
                provenance_rows_for_sample = sorted(actual["rows"])
                issues.extend(actual["errors"])
                if require_complete_provenance and len(actual["rows"]) != 1:
                    issues.append("sample must have exactly one complete provenance row")
                if set(actual["papers"]) != set(expected["papers"]):
                    issues.append(
                        "paper mapping differs: "
                        f"benchmark={sorted(expected['papers'], key=_paper_sort_key)!r}, "
                        f"provenance={sorted(actual['papers'], key=_paper_sort_key)!r}"
                    )
                if Counter(actual["images"]) != Counter(expected["images"]):
                    issues.append(
                        "image mapping differs: "
                        f"benchmark={sorted(expected['images'])!r}, "
                        f"provenance={sorted(actual['images'])!r}"
                    )
                elif require_provenance_order and actual["images"] != expected["images"]:
                    issues.append(
                        "image order differs: "
                        f"benchmark={expected['images']!r}, provenance={actual['images']!r}"
                    )
                for path, declared_hashes in actual["hashes"].items():
                    expected_hashes = expected["hashes"].get(path, set())
                    for digest in declared_hashes:
                        if digest not in expected_hashes:
                            issues.append(
                                f"SHA256 mismatch for {path!r}: provenance={digest!r}, "
                                f"benchmark={sorted(expected_hashes)!r}"
                            )
                if require_complete_provenance:
                    for path in expected["images"]:
                        if not actual["hashes"].get(path):
                            issues.append(f"provenance SHA256 is missing for {path!r}")
            if issues:
                detail = {
                    "issues": sorted(set(issues)),
                    "provenance_row_indices": provenance_rows_for_sample,
                    "sample_id": sample_id,
                }
                provenance_mismatches.append(detail)
                add_finding(
                    "critical",
                    "provenance_mismatch",
                    "Benchmark and provenance mappings are inconsistent.",
                    [sample_id],
                    detail,
                )

    if exact_primary_papers is not None:
        primary_items = [item for item in prepared if item["row"].get("primary_endpoint") is True]
        valid_primary_papers = {
            canonical_paper_id(item["row"].get("paper_id"))
            for item in primary_items
            if not paper_identity_errors(item["row"])
        }
        valid_primary_papers.discard("")
        if len(valid_primary_papers) != exact_primary_papers:
            add_finding(
                "critical",
                "primary_paper_count_mismatch",
                "The benchmark does not contain exactly the preregistered number of unique "
                "valid primary papers.",
                [item["sample_key"] for item in primary_items],
                {
                    "actual": len(valid_primary_papers),
                    "required": exact_primary_papers,
                },
            )

    if minimum_primary_papers:
        primary_items = [item for item in prepared if item["row"].get("primary_endpoint") is True]
        primary_papers = {
            item["primary_paper_id"] for item in primary_items if item["primary_paper_id"]
        }
        if len(primary_papers) < minimum_primary_papers:
            add_finding(
                "critical",
                "insufficient_primary_papers",
                "The benchmark has fewer unique primary papers than the preregistered minimum.",
                [item["sample_key"] for item in primary_items],
                {
                    "actual": len(primary_papers),
                    "required": minimum_primary_papers,
                },
            )

    critical_findings.sort(key=_finding_sort_key)
    warnings.sort(key=_finding_sort_key)
    for sample_key, sample_report in per_sample.items():
        sample_report["row_indices"] = sorted(set(sample_report["row_indices"]))
        sample_report["canonical_paper_ids"] = sorted(
            set(sample_report["canonical_paper_ids"]), key=_paper_sort_key
        )
        sample_report["prompt_sha256s"] = sorted(set(sample_report["prompt_sha256s"]))
        sample_report["image_hashes"].sort(
            key=lambda record: (record["row_index"], record["path"], record["sha256"])
        )
        sample_report["critical_findings"].sort(key=_finding_sort_key)
        sample_report["warnings"].sort(key=_finding_sort_key)
        sample_report["eligible"] = (
            sample_key in valid_sample_ids and not sample_report["critical_findings"]
        )

    eligible_sample_ids = sorted(
        sample_id for sample_id in valid_sample_ids if per_sample[sample_id]["eligible"]
    )
    critical_by_code = Counter(finding["code"] for finding in critical_findings)
    warnings_by_code = Counter(finding["code"] for finding in warnings)
    sample_reports = [per_sample[key] for key in valid_sample_ids]
    status = "pass" if not critical_findings else "fail"
    critical_count = len(critical_findings)
    warning_count = len(warnings)
    summary = {
        "total_rows": len(benchmark_rows),
        "total_samples": len(valid_sample_ids),
        "unique_sample_ids": len(valid_sample_ids),
        "eligible_samples": len(eligible_sample_ids),
        "eligible_sample_count": len(eligible_sample_ids),
        "critical_findings": critical_count,
        "critical_finding_count": critical_count,
        "warnings": warning_count,
        "warning_count": warning_count,
        "declared_images": declared_image_count,
        "hashed_images": hashed_image_count,
        "duplicate_sample_ids": len(duplicate_ids),
        "missing_images": critical_by_code.get("missing_image", 0),
        "unsafe_image_paths": critical_by_code.get("unsafe_image_path", 0),
        "image_placeholder_mismatches": critical_by_code.get("image_placeholder_mismatch", 0),
        "within_row_duplicate_image_groups": len(within_row_image_duplicates),
        "cross_paper_image_reuse_groups": len(cross_paper_image_reuse),
        "provenance_mismatches": len(provenance_mismatches),
        "residual_comparison_prompts": critical_by_code.get("residual_comparison_prompt", 0),
        "benchmark_prompt_overlap_groups": len(benchmark_prompt_overlaps),
        "training_prompt_overlap_groups": len(training_prompt_overlaps),
        "training_paper_overlap_ids": len(training_paper_overlaps),
        "samples_with_gold": sum(
            bool(report["gold_reference_rubric"]["gold"]) for report in sample_reports
        ),
        "samples_with_reference": sum(
            bool(report["gold_reference_rubric"]["reference"]) for report in sample_reports
        ),
        "samples_with_rubric": sum(
            bool(report["gold_reference_rubric"]["rubric"]) for report in sample_reports
        ),
        "critical_by_code": dict(sorted(critical_by_code.items())),
        "warnings_by_code": dict(sorted(warnings_by_code.items())),
    }
    report: dict[str, Any] = {
        "audit_version": audit_version,
        "status": status,
        "publication_ready": status == "pass",
        "summary": summary,
        "critical_findings": critical_findings,
        "warnings": warnings,
        "per_sample_findings": dict(sorted(per_sample.items())),
        "contamination": {
            "cross_paper_image_reuse": cross_paper_image_reuse,
            "exact_normalized_prompt_overlaps": {
                "within_benchmark": benchmark_prompt_overlaps,
                "with_training": training_prompt_overlaps,
            },
            "training_paper_overlaps": training_paper_overlaps,
            "training_identifiers_extracted": sorted(training_identifiers, key=_paper_sort_key),
            "training_lineage": lineage_audit,
        },
        "provenance": {
            "provided": provenance is not None,
            "rows_checked": 0 if provenance is None else len(provenance),
            "mismatches": provenance_mismatches,
        },
        "eligible_sample_ids": eligible_sample_ids,
        "interpretation": {
            "image_hash_equality": (
                "Matching SHA256 values establish byte-for-byte reuse only; they do not "
                "prove that records describe the same semantic paper."
            )
        },
    }
    try:
        json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise BenchmarkAuditError(f"internal audit report is not strict JSON: {exc}") from exc
    return report


def _markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _finding_markdown(finding: Mapping[str, Any]) -> str:
    samples = finding.get("sample_ids") or []
    suffix = f" Samples: {', '.join(_markdown_text(item) for item in samples)}." if samples else ""
    return (
        f"- `{_markdown_text(finding.get('code', 'unknown'))}`: "
        f"{_markdown_text(finding.get('message', ''))}{suffix}"
    )


def write_audit_report(
    report: Mapping[str, Any],
    json_path: str | os.PathLike[str],
    markdown_path: str | os.PathLike[str],
) -> None:
    """Write deterministic JSON and a publication-oriented Markdown summary."""

    if not isinstance(report, Mapping):
        raise BenchmarkAuditError("report must be a mapping")
    try:
        json_text = (
            json.dumps(
                report,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkAuditError(f"report is not strict JSON data: {exc}") from exc

    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    if json_target.absolute() == markdown_target.absolute():
        raise BenchmarkAuditError("json_path and markdown_path must be different files")
    for target in (json_target, markdown_target):
        if target.is_symlink():
            raise BenchmarkAuditError(f"refusing to overwrite symlink: {target}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BenchmarkAuditError(
                f"cannot create report directory for {target}: {exc}"
            ) from exc

    status = str(report.get("status", "fail")).upper()
    publication_ready = report.get("publication_ready") is True
    technical_candidate = (
        status == "PASS" and report.get("technical_audit_passed") is True and not publication_ready
    )
    display_status = "TECHNICAL PASS / PUBLICATION BLOCKED" if technical_candidate else status
    markdown: list[str] = [
        "# Task 3 HF Benchmark Publication Audit",
        "",
        f"Status: **{_markdown_text(display_status)}**",
        "",
    ]
    if publication_ready:
        markdown.append("**PASS: No critical publication-gate findings were detected.**")
    elif technical_candidate:
        markdown.append(
            "**TECHNICAL PASS: Candidate data checks passed, but publication readiness remains "
            "blocked.**"
        )
    else:
        markdown.append(
            "**FAIL: This audit failure blocks publication claims for this benchmark.**"
        )
    markdown.extend(["", "## Summary", "", "| Metric | Count |", "| --- | ---: |"])
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        for key in sorted(summary):
            value = summary[key]
            if isinstance(value, Mapping):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            markdown.append(f"| {_markdown_text(key)} | {_markdown_text(value)} |")

    blockers = report.get("publication_readiness_blockers")
    if isinstance(blockers, list) and blockers:
        markdown.extend(["", "## Publication Readiness Blockers", ""])
        markdown.extend(f"- {_markdown_text(value)}" for value in blockers)

    markdown.extend(["", "## Critical Findings", ""])
    critical = report.get("critical_findings")
    if isinstance(critical, list) and critical:
        markdown.extend(
            _finding_markdown(finding) for finding in critical if isinstance(finding, Mapping)
        )
    else:
        markdown.append("- None.")

    markdown.extend(["", "## Warnings", ""])
    report_warnings = report.get("warnings")
    if isinstance(report_warnings, list) and report_warnings:
        markdown.extend(
            _finding_markdown(finding)
            for finding in report_warnings
            if isinstance(finding, Mapping)
        )
    else:
        markdown.append("- None.")

    eligible = report.get("eligible_sample_ids")
    markdown.extend(["", "## Eligible Samples", ""])
    if isinstance(eligible, list) and eligible:
        markdown.extend(f"- `{_markdown_text(sample_id)}`" for sample_id in eligible)
    else:
        markdown.append("- None.")

    contamination = report.get("contamination", {})
    markdown.extend(
        [
            "",
            "## Contamination Details",
            "",
            "```json",
            json.dumps(contamination, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Interpretation",
            "",
            (
                "Matching SHA256 values establish byte-for-byte reuse only; they do not prove "
                "that records describe the same semantic paper."
            ),
            "",
        ]
    )
    markdown_text = "\n".join(markdown)
    try:
        with json_target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json_text)
        with markdown_target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown_text)
    except OSError as exc:
        raise BenchmarkAuditError(f"cannot write audit report: {exc}") from exc
