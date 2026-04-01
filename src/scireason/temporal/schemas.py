from __future__ import annotations

from hashlib import sha1
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


Granularity = Literal["year", "month", "day", "interval"]


def _infer_granularity_from_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "year"
    if text in {"unknown", "+inf", "-inf"}:
        return "year"
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return "day"
    if len(text) >= 7 and text[4:5] == "-":
        return "month"
    return "year"


def normalize_granularity(value: Any, *, start: Any = None, end: Any = None, default: str = "year") -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": default,
        "unknown": default,
        "year": "year",
        "annual": "year",
        "month": "month",
        "monthly": "month",
        "day": "day",
        "daily": "day",
        "interval": "interval",
        "range": "interval",
        "period": "interval",
        "timespan": "interval",
        "time_span": "interval",
        "date_range": "interval",
    }
    if text in aliases:
        return aliases[text]
    inferred_start = _infer_granularity_from_value(start)
    inferred_end = _infer_granularity_from_value(end)
    if start not in (None, "") and end not in (None, "") and str(start) != str(end):
        return "interval"
    if inferred_start == inferred_end:
        return inferred_start
    if start not in (None, "") or end not in (None, ""):
        return "interval"
    return default
EventSplit = Literal["train", "valid", "test"]
EventType = Literal["extracted", "reviewed", "corrected"]


class TimeInterval(BaseModel):
    """Временной интервал, который «привязан» к утверждению.

    В TG-RAG/T-GRAG важно отличать один и тот же факт в разные периоды.
    """

    start: Optional[str] = Field(default=None, description="ISO дата или год: YYYY / YYYY-MM / YYYY-MM-DD")
    end: Optional[str] = Field(default=None, description="ISO дата/год конца интервала (включительно)")
    granularity: Granularity = "year"

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        payload["granularity"] = normalize_granularity(
            payload.get("granularity"),
            start=payload.get("start"),
            end=payload.get("end"),
            default="year",
        )
        return payload

    def key(self) -> str:
        start = self.start or ""
        end = self.end or start
        return f"{self.granularity}:{start}:{end}"


class TemporalTriplet(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    polarity: Literal["supports", "contradicts", "unknown"] = "unknown"
    time: Optional[TimeInterval] = None
    time_source: Literal["extracted", "paper_year_fallback"] = "extracted"
    # Provenance (MVP): a short quote/snippet from the input chunk/page that supports the assertion.
    evidence_quote: Optional[str] = Field(
        default=None,
        description="Короткая цитата/фрагмент (<=200 символов) из исходного текста, подтверждающий триплет.",
    )

    def as_text(self) -> str:
        time_part = ""
        if self.time is not None:
            time_part = f" | time={self.time.key()}"
        quote_part = f" | evidence={self.evidence_quote}" if self.evidence_quote else ""
        return f"{self.subject} | {self.predicate} | {self.object}{time_part}{quote_part}"


class TemporalEvent(BaseModel):
    """Atomic event for training or evaluating temporal KG predictors.

    The event layer complements Assertion nodes: Assertions remain the explainable/provenance layer,
    while TemporalEvent provides a chronologically ordered stream suitable for TGNN/TGN-style models.
    """

    event_id: Optional[str] = None
    paper_id: str
    chunk_id: Optional[str] = None
    assertion_id: Optional[str] = None
    subject: str
    predicate: str
    object: str
    ts_start: Optional[str] = None
    ts_end: Optional[str] = None
    granularity: Granularity = "year"

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        payload["granularity"] = normalize_granularity(
            payload.get("granularity"),
            start=payload.get("ts_start"),
            end=payload.get("ts_end"),
            default="year",
        )
        return payload

    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    polarity: Literal["supports", "contradicts", "unknown"] = "unknown"
    split: Optional[EventSplit] = None
    event_type: EventType = "extracted"
    extraction_method: str = "llm_triplet"
    weight: float = 1.0
    evidence_quote: Optional[str] = None

    def time_key(self) -> str:
        start = self.ts_start or ""
        end = self.ts_end or start
        return f"{self.granularity}:{start}:{end}"

    def pair_key(self) -> tuple[str, str, str]:
        return (self.subject.strip().lower(), self.predicate.strip().lower(), self.object.strip().lower())

    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.ts_start or "",
            self.ts_end or "",
            self.subject.strip().lower(),
            self.object.strip().lower(),
        )

    def as_text(self) -> str:
        quote_part = f" | evidence={self.evidence_quote}" if self.evidence_quote else ""
        return (
            f"{self.subject} | {self.predicate} | {self.object}"
            f" | time={self.time_key()} | conf={self.confidence:.3f}{quote_part}"
        )

    def stable_id(self) -> str:
        if self.event_id:
            return self.event_id
        raw = (
            f"{self.paper_id}|{self.chunk_id or ''}|{self.subject}|{self.predicate}|{self.object}|"
            f"{self.ts_start or ''}|{self.ts_end or ''}|{self.granularity}|{self.event_type}"
        )
        return sha1(raw.encode('utf-8')).hexdigest()[:16]
