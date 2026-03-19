from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .schemas import Citation, HypothesisDraft
from .temporal.schemas import TemporalEvent


ChunkModality = Literal["text", "table", "figure", "formula", "page", "unknown"]


class ChunkRecord(BaseModel):
    """Primary document artifact flowing through OCR -> retrieval -> graph.

    The repo historically stored chunks as `{chunk_id, text}`. This richer contract keeps that
    backward compatibility while making the record usable for document OCR/layout pipelines,
    temporal provenance, and multimodal verification.
    """

    chunk_id: str
    text: str = ""
    paper_id: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    modality: ChunkModality = "text"
    source_backend: str = "unknown"
    reading_order: Optional[int] = None
    lang: Optional[str] = None
    table_html: Optional[str] = None
    table_md: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "page": self.page,
            "modality": self.modality,
            "source_backend": self.source_backend,
            "reading_order": self.reading_order,
            "lang": self.lang,
            "text": self.text,
        }
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        if self.table_html:
            payload["table_html"] = self.table_html
        if self.table_md:
            payload["table_md"] = self.table_md
        if self.image_path:
            payload["image_path"] = self.image_path
        if self.metadata:
            payload.update({k: v for k, v in self.metadata.items() if k not in payload})
        return payload


class HypothesisArtifact(BaseModel):
    """Stable product artifact for `ChunkRecord -> TemporalEvent -> Hypothesis`."""

    hypothesis: HypothesisDraft
    candidate_kind: str = "unknown"
    source_term: Optional[str] = None
    target_term: Optional[str] = None
    predicate: Optional[str] = None
    time_scope: Optional[str] = None
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    supporting_events: List[str] = Field(default_factory=list)
    graph_signals: Dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_draft(
        cls,
        draft: HypothesisDraft,
        *,
        candidate_kind: str,
        source_term: Optional[str] = None,
        target_term: Optional[str] = None,
        predicate: Optional[str] = None,
        time_scope: Optional[str] = None,
        evidence_chunk_ids: Optional[List[str]] = None,
        supporting_events: Optional[List[str]] = None,
        graph_signals: Optional[Dict[str, float]] = None,
    ) -> "HypothesisArtifact":
        return cls(
            hypothesis=draft,
            candidate_kind=candidate_kind,
            source_term=source_term,
            target_term=target_term,
            predicate=predicate,
            time_scope=time_scope,
            evidence_chunk_ids=list(evidence_chunk_ids or []),
            supporting_events=list(supporting_events or []),
            graph_signals=dict(graph_signals or {}),
        )


class VerificationBundle(BaseModel):
    """Bundle passed to multimodal verification stage."""

    event: TemporalEvent
    evidence: List[Citation] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list)
    graph_context: Dict[str, Any] = Field(default_factory=dict)
