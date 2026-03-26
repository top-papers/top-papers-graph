from __future__ import annotations

from typing import Dict, Any, TypedDict, Optional
import os
from pathlib import Path
from pydantic import TypeAdapter
from rich.console import Console

from ..schemas import HypothesisDraft, CritiqueReview, DebateResult
from ..llm import chat_json
from ..reward import RuleBasedReward

console = Console()

HYPOTHESIS_SCHEMA = """JSON объект HypothesisDraft:
{ "title": "...", "premise":"...", "mechanism":"...", "time_scope":"...", "proposed_experiment":"...", "supporting_evidence":[{"source_id":"...","text_snippet":"..."}], "confidence_score": 1-10 }"""

CRITIQUE_SCHEMA = """JSON объект CritiqueReview:
{ "logical_fallacies":[...], "missing_evidence":[...], "contradictions":[...], "is_accepted": true|false, "improvement_suggestions":"..." }"""

MEDIATOR_SCHEMA = """JSON объект:
{ "verdict": "accepted|rejected|revise|max_rounds", "reason":"...", "rounds": <int> }"""


class DebateState(TypedDict, total=False):
    domain: str
    context: str
    hypothesis: Dict[str, Any]
    critique: Dict[str, Any]
    rounds: int
    max_rounds: int
    verdict: str


def _enthusiast(state: DebateState) -> DebateState:
    domain = state.get("domain", "Science")
    context = state.get("context", "")
    prev = state.get("critique")

    system = f"""Ты — ведущий исследователь в области {domain}.
Твоя задача — предложить смелую, но обоснованную гипотезу на основе контекста литературы.
Принципы:
1) НЕ выдумывай факты. Каждое важное утверждение должно ссылаться на контекст (цитатой/ID).
2) Разделяй факты и предположение.
3) Гипотеза должна быть проверяема экспериментом.
"""

    user = f"""Контекст (фрагменты литературы):
{context}

"""
    if prev:
        user += f"""Предыдущая критика:
{prev}

Исправь гипотезу так, чтобы снять критические замечания, сохранив идею.
"""
    user += f"""Верни гипотезу в JSON по схеме HypothesisDraft."""

    hyp = chat_json(system=system, user=user, schema_hint=HYPOTHESIS_SCHEMA, temperature=0.6)
    state["hypothesis"] = hyp
    return state


def _skeptic(state: DebateState) -> DebateState:
    domain = state.get("domain", "Science")
    context = state.get("context", "")
    hyp = state.get("hypothesis") or {}

    checklist_path = os.getenv("SKEPTIC_CHECKLIST_PATH")
    checklist = None
    if checklist_path and Path(checklist_path).exists():
        checklist = Path(checklist_path).read_text(encoding="utf-8")
    else:
        checklist = """- Корреляция vs каузальность
- Ошибки обобщения (вид/модель/условия)
- Противоречия с данным контекстом
- Проверяемость (эксперимент возможен?)"""

    system = f"""Ты — строгий научный рецензент ("Reviewer #2") в области {domain}.
Твоя задача — выявить логические ошибки, галлюцинации и недостаток доказательств.

Checklist (из экспертов/конфига):
{checklist}
"""

    user = f"""Контекст:
{context}

Гипотеза:
{hyp}

Проанализируй и верни CritiqueReview (JSON)."""

    critique = chat_json(system=system, user=user, schema_hint=CRITIQUE_SCHEMA, temperature=0.1)
    state["critique"] = critique
    return state


def _mediator(state: DebateState) -> DebateState:
    rounds = int(state.get("rounds", 0))
    max_rounds = int(state.get("max_rounds", 3))
    critique = state.get("critique") or {}
    is_accepted = bool(critique.get("is_accepted", False))

    # Optional: rule-based reward that reacts to expert feedback instantly.
    overrides_path = os.getenv("EXPERT_OVERRIDES_PATH")
    reward = None
    if overrides_path:
        try:
            rb = RuleBasedReward(Path(overrides_path))
            hyp = state.get("hypothesis") or {}
            reward = rb.score(hyp)
            state["reward"] = {"score": reward.score, "reasons": reward.reasons}
        except Exception:
            pass
    # Hard guardrails: if reward is very low, force revise (even if critique is lenient).
    if reward is not None and reward.score <= -1.0:
        state["verdict"] = "revise"
        return state

    if is_accepted:
        state["verdict"] = "accepted"
        return state

    if rounds + 1 >= max_rounds:
        state["verdict"] = "max_rounds"
        return state

    state["verdict"] = "revise"
    return state


def run_debate(domain: str, context: str, max_rounds: int = 3) -> DebateResult:
    """Лёгкая версия “дебатов” без жесткой зависимости от API LangGraph.
    (LangGraph отлично подходит для продакшена, но для учебного проекта важно, чтобы всё работало из коробки.)
    """
    state: DebateState = {"domain": domain, "context": context, "rounds": 0, "max_rounds": max_rounds}
    for i in range(max_rounds):
        state = _enthusiast(state)
        state = _skeptic(state)
        state = _mediator(state)
        state["rounds"] = i + 1
        if state.get("verdict") in ("accepted", "max_rounds"):
            break

    hyp = None
    crit = None
    try:
        hyp = TypeAdapter(HypothesisDraft).validate_python(state.get("hypothesis"))
    except Exception:
        pass
    try:
        crit = TypeAdapter(CritiqueReview).validate_python(state.get("critique"))
    except Exception:
        pass

    verdict = state.get("verdict", "max_rounds")
    if verdict == "accepted":
        v = "accepted"
    elif verdict == "max_rounds":
        v = "max_rounds"
    else:
        v = "rejected"

    return DebateResult(final_hypothesis=hyp, last_critique=crit, rounds=int(state.get("rounds", 0)), verdict=v)
