from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple

from .schemas import TimeInterval


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def infer_time_from_query(query: str) -> Optional[TimeInterval]:
    """Грубый парсер временных ограничений из запроса.

    MVP: вытаскиваем годы (2020, 2021...). Для курса этого достаточно как базовый слой,
    а далее можно заменить на LLM-парсер или dateparser.
    """
    years = _YEAR_RE.findall(query)
    if not years:
        return None
    years_int = sorted({int(y) for y in years})
    if len(years_int) == 1:
        y = years_int[0]
        return TimeInterval(start=str(y), end=str(y), granularity="year")
    return TimeInterval(start=str(years_int[0]), end=str(years_int[-1]), granularity="year")


def default_time_from_paper_year(year: Optional[int]) -> Optional[TimeInterval]:
    if not year:
        return None
    return TimeInterval(start=str(year), end=str(year), granularity="year")
