from __future__ import annotations

"""Hypothesis testing / verification.

The repository already provides a *generation* loop (`agents.debate_graph`). This module adds a
thin, domain-agnostic verification layer that can be used for:

1) literature-consistency checks (GraphRAG retrieval -> LLM reviewer)
2) temporal consistency checks (optional Neo4jTemporalStore)
3) pluggable experiment execution (future, domain-specific)

The goal is to keep the MVP easy to run locally while providing an extensible API for future work
by 200-300 experts.
"""

import json

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter

from ..config import settings
from ..llm import chat_json
from ..demos.retriever import retrieve_demos
from ..demos.render import render_demos_block
from ..schemas import DebateResult, HypothesisDraft
from ..graph.graphrag_query import retrieve_context
from ..graph.temporal_neo4j_store import Neo4jTemporalStore


class EvidenceItem(BaseModel):
    source_id: str = Field(description="paper_id or assertion_id")
    text_snippet: str = Field(description="short quote/summary that supports/contradicts")
    score: Optional[float] = Field(default=None, description="retriever score if known")


Verdict = Literal["supported", "contradicted", "insufficient_evidence", "needs_revision"]


class HypothesisTestResult(BaseModel):
    verdict: Verdict
    summary: str
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list)
    temporal_notes: Optional[str] = None
    recommended_experiments: Optional[str] = None
    confidence_score: int = Field(ge=1, le=10, default=5)


_ENTITIES_SCHEMA = """JSON объект:
{ "entities": ["entity1", "entity2", ...] }
Правила:
- 5-10 сущностей максимум.
- Сущности должны быть терминами/объектами из гипотезы (материалы, параметры, процессы).
"""


_TEST_SCHEMA = """JSON объект HypothesisTestResult:
{ "verdict": "supported|contradicted|insufficient_evidence|needs_revision",
  "summary": "...",
  "supporting_evidence": [{"source_id":"...","text_snippet":"...","score":0.0}],
  "contradicting_evidence": [{"source_id":"...","text_snippet":"...","score":0.0}],
  "temporal_notes": "...",
  "recommended_experiments": "...",
  "confidence_score": 1-10 }

Правила:
- НЕ выдумывай источники: source_id должен соответствовать paper_id из контекста или assertion_id из TG.
- text_snippet должен быть коротким и опираться на контекст.
"""


def _extract_entities(domain: str, hypothesis_text: str) -> List[str]:
    """Extract key entities/variables from a hypothesis (LLM).

    If LLM is unavailable, returns an empty list.
    """
    system = f"Ты — помощник исследователя в области {domain}."
    user = f"""Гипотеза:
{hypothesis_text}

Выдели ключевые сущности/переменные, которые нужно проверить."""
    try:
        data = chat_json(system=system, user=user, schema_hint=_ENTITIES_SCHEMA, temperature=0.0)
        ents = data.get("entities")
        if isinstance(ents, list):
            return [str(x).strip() for x in ents if str(x).strip()][:10]
    except Exception:
        pass
    return []


def _collect_temporal_signals(entities: List[str], limit_per_entity: int = 8) -> List[Dict[str, Any]]:
    """Query Neo4jTemporalStore for assertions related to entities (best effort)."""
    if not entities:
        return []
    out: List[Dict[str, Any]] = []
    try:
        store = Neo4jTemporalStore()
        store.ensure_schema()
        for e in entities:
            try:
                rows = store.query_assertions(entity=e, time=None, limit=limit_per_entity)
                for r in rows:
                    r = dict(r)
                    r["entity"] = e
                    out.append(r)
            except Exception:
                continue
        store.close()
    except Exception:
        return []
    return out


def test_hypothesis(
    *,
    domain: str,
    hypothesis: HypothesisDraft,
    collection_text: str,
    k: int = 12,
    use_demos: Optional[bool] = None,
    ctx_override: Optional[List[Dict[str, Any]]] = None,
) -> HypothesisTestResult:
    """Verify a hypothesis against retrieved literature + optional temporal KG signals."""

    hyp_text = (
        f"Title: {hypothesis.title}\n\n"
        f"Premise: {hypothesis.premise}\n\n"
        f"Mechanism: {hypothesis.mechanism}\n\n"
        f"Time/conditions scope: {hypothesis.time_scope}\n\n"
        f"Proposed experiment: {hypothesis.proposed_experiment}\n"
    )

    # 1) Retrieve literature context (GraphRAG)
    query = f"{hypothesis.title}\n{hypothesis.premise}\n{hypothesis.mechanism}"
    ctx = ctx_override if ctx_override is not None else retrieve_context(collection=collection_text, query=query, limit=k)
    ctx_lines = []
    for c in ctx:
        payload = c.get("payload") or {}
        pid = str(payload.get("paper_id") or "unknown")
        txt = str(payload.get("text") or "")
        score = c.get("score")
        ctx_lines.append(f"[{pid}] (score={score}) {txt}")
    ctx_text = "\n\n".join(ctx_lines)

    # 2) Temporal KG signals (optional)
    entities = _extract_entities(domain=domain, hypothesis_text=hyp_text)
    temporal_rows = _collect_temporal_signals(entities)
    temporal_text = ""
    if temporal_rows:
        # keep short
        lines = []
        for r in temporal_rows[:50]:
            lines.append(
                f"entity={r.get('entity')} id={r.get('id')} pred={r.get('predicate')} polarity={r.get('polarity')} "
                f"time={r.get('granularity')}:{r.get('t_start')}..{r.get('t_end')} conf={r.get('confidence')}"
            )
        temporal_text = "\n".join(lines)

    # 2.5) Retrieval few-shot demos (optional)
    enabled = getattr(settings, "demo_enabled", True) if use_demos is None else use_demos
    demo_block = ""
    if enabled:
        dk = int(getattr(settings, "demo_top_k_hypothesis", 2))
        demo_query = query  # same as retrieval query
        demos = retrieve_demos(task="hypothesis_test", domain=domain, query=demo_query, k=dk)
        demo_block = render_demos_block(
            demos,
            max_total_chars=int(getattr(settings, "demo_max_chars_total", 3500)),
            title="Эталонные примеры проверки гипотез",
        )

    system = f"""Ты — строгий научный рецензент и экспериментатор в области {domain}.
Твоя задача — проверить гипотезу: есть ли поддержка в литературе, есть ли противоречия и что
нужно сделать, чтобы проверить её экспериментально.

Требования:
1) НЕ выдумывай факты и источники.
2) Опирайся только на предоставленные фрагменты и сигналы из темпорального графа.
3) Если данных недостаточно — verdict=insufficient_evidence.
"""

    user = f"""{demo_block}Гипотеза:
{hyp_text}

Литературный контекст (GraphRAG):
{ctx_text}

Сигналы из темпорального графа (если есть):
{temporal_text if temporal_text else '[none]'}

Верни результат проверки по схеме HypothesisTestResult."""

    data = chat_json(system=system, user=user, schema_hint=_TEST_SCHEMA, temperature=0.1)
    return TypeAdapter(HypothesisTestResult).validate_python(data)


def load_hypothesis_from_json(path: Path) -> HypothesisDraft:
    """Load either HypothesisDraft or DebateResult JSON and return HypothesisDraft."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    # Try DebateResult first
    try:
        dr = TypeAdapter(DebateResult).validate_python(obj)
        if dr.final_hypothesis is not None:
            return dr.final_hypothesis
    except Exception:
        pass
    return TypeAdapter(HypothesisDraft).validate_python(obj)
