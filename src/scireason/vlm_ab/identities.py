# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Normalization rules for security-sensitive publication identifiers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_ASCII_REVIEWER_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:@/+ -]*[A-Za-z0-9])?$")
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)


def contains_unsafe_identifier_codepoint(value: str) -> bool:
    """Return whether text contains controls or Unicode default-ignorables."""

    for character in value:
        codepoint = ord(character)
        if unicodedata.category(character).startswith("C") or any(
            start <= codepoint <= end for start, end in _DEFAULT_IGNORABLE_RANGES
        ):
            return True
    return False


def normalize_identity(value: Any, *, ascii_reviewer: bool = False) -> str:
    """Return a comparison key, or an empty string for an unsafe identifier."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or contains_unsafe_identifier_codepoint(value)
    ):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    if contains_unsafe_identifier_codepoint(normalized):
        return ""
    if ascii_reviewer and not _ASCII_REVIEWER_ID_RE.fullmatch(normalized):
        return ""
    return re.sub(r"\s+", " ", normalized).casefold().strip()


__all__ = ["contains_unsafe_identifier_codepoint", "normalize_identity"]
