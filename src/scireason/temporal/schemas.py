from __future__ import annotations

from hashlib import sha1
from typing import Literal, Optional

from pydantic import BaseModel, Field


Granularity = Literal["year", "month", "day"]
EventSplit = Literal["train", "valid", "test"]
EventType = Literal["extracted", "reviewed", "corrected"]


class TimeInterval(BaseModel):
    """Временной интервал, который «привязан» к утверждению.

    В TG-RAG/T-GRAG важно отличать один и тот же факт в разные периоды.
    """

    start: Optional[str] = Field(default=None, description="ISO дата или год: YYYY / YYYY-MM / YYYY-MM-DD")
    end: Optional[str] = Field(default=None, description="ISO дата/год конца интервала (включительно)")
    granularity: Granularity = "year"

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
