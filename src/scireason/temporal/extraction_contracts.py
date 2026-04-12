from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter

from .schemas import TemporalTriplet, TimeInterval, normalize_granularity


class TripletExtractionPayload(BaseModel):
    triplets: list[TemporalTriplet] = Field(default_factory=list)


class TripletVerificationRecord(BaseModel):
    subject: str
    predicate: str
    object: str
    supported: bool = True
    support_level: Literal["none", "weak", "moderate", "strong"] = "moderate"
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    polarity: Literal["supports", "contradicts", "unknown"] = "unknown"
    evidence_quote: Optional[str] = None
    time: Optional[TimeInterval] = None
    time_source: Literal["extracted", "paper_year_fallback"] = "extracted"
    rationale: Optional[str] = None


class TripletVerificationPayload(BaseModel):
    verifications: list[TripletVerificationRecord] = Field(default_factory=list)


EXTRACTION_SCHEMA_HINT = json.dumps(
    {
        "name": "temporal_triplet_extraction",
        "strict": True,
        "schema": TripletExtractionPayload.model_json_schema(),
    },
    ensure_ascii=False,
    indent=2,
)

VERIFICATION_SCHEMA_HINT = json.dumps(
    {
        "name": "temporal_triplet_verification",
        "strict": True,
        "schema": TripletVerificationPayload.model_json_schema(),
    },
    ensure_ascii=False,
    indent=2,
)


_extraction_adapter = TypeAdapter(TripletExtractionPayload)
_verification_adapter = TypeAdapter(TripletVerificationPayload)


def validate_extraction_payload(data: Any) -> list[TemporalTriplet]:
    rows = data
    if isinstance(data, dict):
        rows = data.get("triplets", [])
    if not isinstance(rows, list):
        rows = []
    out: list[TemporalTriplet] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            out.append(TemporalTriplet.model_validate(row))
        except Exception:
            continue
    return out


def validate_verification_payload(data: Any) -> list[TripletVerificationRecord]:
    if isinstance(data, list):
        data = {"verifications": data}
    payload = _verification_adapter.validate_python(data)
    return list(payload.verifications or [])
