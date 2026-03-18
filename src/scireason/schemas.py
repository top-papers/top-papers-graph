from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str = Field(description="Стабильный ID источника (DOI/arXiv/OpenAlex/S2 PaperId).")
    text_snippet: str = Field(description="Короткая цитата/фрагмент текста из источника.")


class Triplet(BaseModel):
    subject: str
    predicate: str
    object: str
    evidence: Optional[Citation] = None
    polarity: Literal["supports", "contradicts", "unknown"] = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)


class HypothesisDraft(BaseModel):
    title: str
    premise: str = Field(description="Основное предположение")
    mechanism: str = Field(description="Предполагаемый механизм")
    time_scope: str = Field(default="", description="Условия/время применимости (например: T, SOC окно, химия, формат).")
    proposed_experiment: str = Field(description="Как проверить гипотезу в эксперименте (дизайн/метод/метрика).")
    supporting_evidence: List[Citation] = Field(default_factory=list)
    confidence_score: int = Field(description="Уверенность 1-10", ge=1, le=10, default=5)


class CritiqueReview(BaseModel):
    logical_fallacies: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    is_accepted: bool = False
    improvement_suggestions: str = ""


class DebateResult(BaseModel):
    final_hypothesis: Optional[HypothesisDraft] = None
    last_critique: Optional[CritiqueReview] = None
    rounds: int = 0
    verdict: Literal["accepted", "rejected", "max_rounds"] = "max_rounds"
