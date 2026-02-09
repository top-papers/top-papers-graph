from __future__ import annotations

from datetime import date
from typing import Optional, Literal
from pydantic import BaseModel, Field


Granularity = Literal["year", "month", "day"]


class TimeInterval(BaseModel):
    """Временной интервал, который «привязан» к утверждению.

    В TG-RAG/T-GRAG важно отличать один и тот же факт в разные периоды. 
    """
    start: Optional[str] = Field(default=None, description="ISO дата или год: YYYY / YYYY-MM / YYYY-MM-DD")
    end: Optional[str] = Field(default=None, description="ISO дата/год конца интервала (включительно)")
    granularity: Granularity = "year"


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
