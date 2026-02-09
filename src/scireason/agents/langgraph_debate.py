from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any

from ..schemas import DebateResult, HypothesisDraft, CritiqueReview
from .debate_graph import _enthusiast, _skeptic, _mediator  # reuse node logic

# LangGraph — основной “боевой” оркестратор (state machine).
# Если у вас не установился langgraph, используйте run_debate из debate_graph.py.


class State(TypedDict, total=False):
    domain: str
    context: str
    hypothesis: Dict[str, Any]
    critique: Dict[str, Any]
    rounds: int
    max_rounds: int
    verdict: str


def build_graph():
    from langgraph.graph import StateGraph, END  # type: ignore

    g = StateGraph(State)
    g.add_node("enthusiast", _enthusiast)
    g.add_node("skeptic", _skeptic)
    g.add_node("mediator", _mediator)

    g.set_entry_point("enthusiast")
    g.add_edge("enthusiast", "skeptic")
    g.add_edge("skeptic", "mediator")

    def route(state: State):
        v = state.get("verdict", "revise")
        if v in ("accepted", "max_rounds"):
            return END
        return "enthusiast"

    g.add_conditional_edges("mediator", route, {"enthusiast": "enthusiast", END: END})
    return g.compile()
