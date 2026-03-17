from __future__ import annotations

import re
from typing import Any, Dict, Optional

NEG_INFINITY = "-infinity"
POS_INFINITY = "+infinity"
LEGACY_TIME_PLACEHOLDER = "If applicable: year range / before-after / condition window"

_YEAR_RANGE_RE = re.compile(r"\b(?P<start>(?:18|19|20)\d{2})(?:\s*[-–/]\s*(?P<end>(?:18|19|20)\d{2}))?\b")


def _strip(value: Any) -> str:
    return str(value or "").strip()


def normalize_infinity_token(value: Any, *, default: Optional[str] = None) -> Optional[str]:
    raw = _strip(value)
    if not raw:
        return default
    low = raw.lower().replace("∞", "infinity")
    if low in {"unknown", "n/a", "na", "none", "null", "?", LEGACY_TIME_PLACEHOLDER.lower()}:
        return default
    if low in {"-inf", "-infinity", "minus infinity", "neg infinity", "negative infinity"}:
        return NEG_INFINITY
    if low in {"+inf", "+infinity", "inf", "infinity", "plus infinity", "pos infinity", "positive infinity"}:
        return POS_INFINITY
    return raw


def infer_start_end_from_legacy_interval(interval: Any) -> tuple[Optional[str], Optional[str]]:
    raw = _strip(interval)
    if not raw or raw == LEGACY_TIME_PLACEHOLDER:
        return None, None
    m = _YEAR_RANGE_RE.search(raw)
    if not m:
        return None, None
    start = m.group("start")
    end = m.group("end") or start
    return start, end


def build_interval(start_date: Any, end_date: Any) -> str:
    start = normalize_infinity_token(start_date)
    end = normalize_infinity_token(end_date)
    if not start and not end:
        return "unknown"
    if start and end:
        if start == end:
            return start
        return f"{start}..{end}"
    return start or end or "unknown"


def normalize_temporal_payload(
    record: Dict[str, Any],
    *,
    end_date_hint: Optional[str] = None,
    start_date_hint: Optional[str] = None,
    valid_from_hint: Optional[str] = None,
    valid_to_hint: Optional[str] = None,
    temporal_basis: Optional[str] = None,
) -> Dict[str, Any]:
    start_date = normalize_infinity_token(record.get("start_date"), default=None)
    end_date = normalize_infinity_token(record.get("end_date"), default=None)

    if not start_date or not end_date:
        legacy_start, legacy_end = infer_start_end_from_legacy_interval(record.get("time_interval"))
        start_date = start_date or legacy_start
        end_date = end_date or legacy_end

    start_date = start_date or normalize_infinity_token(start_date_hint, default=NEG_INFINITY)
    end_date = end_date or normalize_infinity_token(end_date_hint, default=POS_INFINITY)

    valid_from = normalize_infinity_token(record.get("valid_from"), default=None)
    valid_to = normalize_infinity_token(record.get("valid_to"), default=None)
    valid_from = valid_from or normalize_infinity_token(valid_from_hint, default=end_date or start_date or NEG_INFINITY)
    valid_to = valid_to or normalize_infinity_token(valid_to_hint, default=POS_INFINITY)

    basis = _strip(record.get("temporal_basis") or temporal_basis or "") or "unknown"

    return {
        "start_date": start_date,
        "end_date": end_date,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "time_interval": build_interval(start_date, end_date),
        "validity_interval": build_interval(valid_from, valid_to),
        "temporal_basis": basis,
    }
