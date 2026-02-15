from __future__ import annotations

"""Agentic (code-writing) candidate generator for hypothesis discovery.

The repo already had an LLM-based "writer" step for hypotheses. This module adds an
*agentic* step that reasons over the temporal knowledge graph using classic (logical)
+ modern (vector-ish) graph algorithms.

Design
------
- Inspired by Hugging Face smolagents CodeAgent: the agent *writes python code* and we
  execute it in a constrained sandbox.
- Tooling is intentionally small and open-source: NetworkX for community detection,
  centrality and link prediction.
- If no LLM is available, the pipeline still works (agent disabled or LLM_PROVIDER=mock).
"""

import ast
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console

from ..agentic.code_agent import CodeAgent, Tool
from ..agentic.graph_tools import (
    build_nx_graph,
    communities_greedy_modularity,
    cross_community_bridges,
    graph_summary,
    link_prediction,
    spectral_link_prediction,
    top_central_nodes,
)

# Optional GNN link prediction (PyTorch Geometric). Must not break base installation.
try:  # pragma: no cover
    from ..gnn.pyg_link_prediction import PyGLinkPredConfig, PyGUnavailableError, pyg_link_prediction
except Exception:  # pragma: no cover
    PyGLinkPredConfig = None
    PyGUnavailableError = Exception
    pyg_link_prediction = None
from ..schemas import Citation
from ..temporal.temporal_kg_builder import PaperRecord, TemporalKnowledgeGraph
from ..config import settings

console = Console()


@dataclass(frozen=True)
class AgentCandidate:
    kind: str
    source: str
    target: str
    predicate: str
    score: float
    time_scope: str
    graph_signals: Dict[str, float]
    evidence: List[Citation]


def _time_scope_from_kg(kg: TemporalKnowledgeGraph, recent_window_years: int = 3) -> str:
    years = kg.meta.get("years") or []
    if not years:
        return ""
    try:
        max_year = int(max(years))
        start = max_year - max(1, int(recent_window_years)) + 1
        return f"{start}-{max_year}"
    except Exception:
        return ""


def _evidence_stub(papers: Sequence[PaperRecord], a: str, b: str, k: int = 2) -> List[Citation]:
    ev: List[Citation] = []
    al, bl = (a or "").lower(), (b or "").lower()
    for p in papers:
        t = (p.text or "")
        tl = t.lower()
        if al in tl and bl in tl:
            # very small snippet (first 220 chars around a)
            idx = tl.find(al)
            if idx < 0:
                idx = tl.find(bl)
            start = max(0, idx - 80)
            end = min(len(t), idx + 140)
            sn = t[start:end].strip().replace("\n", " ")
            if len(sn) > 220:
                sn = sn[:219] + "…"
            ev.append(Citation(source_id=p.paper_id, text_snippet=sn or t[:220]))
        if len(ev) >= k:
            break
    return ev


def agent_generate_candidates(
    kg: TemporalKnowledgeGraph,
    *,
    papers: Sequence[PaperRecord],
    query: str,
    domain: str,
    top_k: int = 10,
    recent_window_years: int = 3,
) -> List[AgentCandidate]:
    """Use a code-writing agent to propose candidate hypotheses from a KG.

    Returns candidates with graph signals from:
    - communities (greedy modularity)
    - centrality
    - classic link prediction (Adamic-Adar / Jaccard / preferential attachment)
    - cross-community "bridge" suggestions
    """

    if not getattr(settings, "hyp_agent_enabled", True):
        return []

    backend = (getattr(settings, "hyp_agent_backend", "internal") or "internal").strip().lower()

    time_scope = _time_scope_from_kg(kg, recent_window_years=recent_window_years)

    # Tools capture KG in closures so the agent code stays import-free.
    def _build_graph() -> Any:
        return build_nx_graph(kg, directed=False, min_total_count=1)

    def _graph_summary(G: Any) -> Dict[str, Any]:
        return graph_summary(G)

    def _communities(G: Any) -> List[List[str]]:
        return communities_greedy_modularity(G)

    def _centrality(G: Any, k: int = 10) -> Dict[str, Any]:
        return top_central_nodes(G, k=int(k))

    def _link_pred(G: Any, method: str = "adamic_adar", k: int = 30) -> List[List[Any]]:
        return [list(x) for x in link_prediction(G, method=method, k=int(k))]

    def _cross_bridges(G: Any, comms: List[List[str]], top_k: int = 20) -> List[List[Any]]:
        return [list(x) for x in cross_community_bridges(G, comms, top_k=int(top_k))]

    def _spectral_lp(G: Any, dim: int = 8, k: int = 30) -> List[List[Any]]:
        return [list(x) for x in spectral_link_prediction(G, dim=int(dim), k=int(k))]

    def _gnn_lp(G: Any, k: int = 30) -> List[List[Any]]:
        """GNN-based link prediction via PyTorch Geometric (optional)."""
        if pyg_link_prediction is None or PyGLinkPredConfig is None:
            return []
        try:
            cfg = PyGLinkPredConfig(
                epochs=int(getattr(settings, "hyp_gnn_epochs", 80) or 80),
                hidden_dim=int(getattr(settings, "hyp_gnn_hidden_dim", 64) or 64),
                lr=float(getattr(settings, "hyp_gnn_lr", 0.01) or 0.01),
                node_cap=int(getattr(settings, "hyp_gnn_node_cap", 300) or 300),
                seed=int(getattr(settings, "hyp_gnn_seed", 7) or 7),
                device="cpu",
            )
            return [list(x) for x in pyg_link_prediction(G, top_k=int(k), config=cfg)]
        except PyGUnavailableError:
            return []
        except Exception:
            return []

    system_prompt = (
    "You are a graph reasoning agent. Use the provided tools to analyze the graph and propose "
    "candidate scientific hypotheses as missing links or emerging bridges. "
    "Prefer candidates that connect different communities and involve central nodes. "
    "If you need more evidence, you may (best-effort) search open APIs for papers and store/retrieve "
    "snippets via the lightweight vector store tools."
)

# Select agent backend.

    raw: Any
    if backend == "smolagents":
        try:
            from ..smolagents_integration.runner import run_code_agent
            from ..smolagents_integration.tools import make_graph_tools

            # Convert KG edges into a lightweight edge list + weights for tools.
            edges: List[Tuple[str, str]] = []
            weights: List[float] = []
            for e in kg.edges:
                if int(getattr(e, "total_count", 0) or 0) < 1:
                    continue
                u, v = (e.source or ""), (e.target or "")
                if not u or not v or u == v:
                    continue
                edges.append((u, v))
                try:
                    weights.append(float(getattr(e, "score", 1.0) or 1.0))
                except Exception:
                    weights.append(1.0)

            smol_tools = make_graph_tools(edges=edges, weights=weights, directed=False)

            task = (
                "Using the temporal knowledge graph, propose up to {k} candidate hypotheses as edges (source, predicate, target).
"
                "Return a LIST of dicts with keys: kind, source, target, predicate, score, graph_signals.
"
                "Where:
"
                "- kind is one of: cross_bridge, link_prediction, central_emergence
"
                "- predicate is a short snake_case relation like may_relate_to, influences, correlates_with
"
                "- score is a float where higher is better
"
                "- graph_signals is a dict of floats (e.g. adamic_adar, pagerank_u, pagerank_v, comm_u, comm_v)

"
                "Core graph tools you may call:
"
                "- build_graph()
"
                "- communities(G, method='greedy', max_communities=...)
"
                "- centrality(G, k=...)
"
                "- cross_bridges(G, comms, top_k=...)
"
                "- link_prediction(G, method=..., k=...)
"
                "- spectral_link_prediction(G, dim=..., k=...)

"
                "Optional research + storage tools (best-effort, may be empty offline):
"
                "- api_search_papers(query, limit=..., sources='semantic_scholar,openalex', with_abstracts=True)
"
                "- vector_index(collection, texts, ids=None, backend='auto')
"
                "- vector_search(collection, query, limit=..., backend='auto')
"
                "- graph_store_put(name, G), graph_store_neighbors(name, node), graph_store_shortest_path(name, src, dst)

"
                "IMPORTANT: end by calling final_answer(<the_list>)."
            ).format(k=int(top_k))

            context = f"Domain: {domain}\nQuery: {query}\nTime scope (recent): {time_scope}\n"

            raw = run_code_agent(
                task=task,
                tools=smol_tools,
                system_prompt=system_prompt,
                context=context,
                max_steps=int(getattr(settings, "hyp_agent_max_steps", 4) or 4),
                executor_type=getattr(settings, "smol_executor", "local") or "local",
                additional_authorized_imports=[
                    "math",
                    "statistics",
                    "numpy",
                    "numpy.*",
                    "networkx",
                ],
            )
        except Exception as e:
            console.print(
                f"[yellow]smolagents backend unavailable ({type(e).__name__}: {e}). Falling back to internal agent.[/yellow]"
            )
            backend = "internal"

    if backend != "smolagents":
        tools = [
            Tool("build_graph", "Convert TemporalKnowledgeGraph -> NetworkX graph", _build_graph),
            Tool("graph_summary", "Basic graph stats: nodes/edges/density", _graph_summary),
            Tool("communities", "Community detection via greedy modularity", _communities),
            Tool("centrality", "Top nodes by pagerank/degree/betweenness", _centrality),
            Tool(
                "link_prediction",
                "Link prediction: adamic_adar|jaccard|preferential_attachment|common_neighbor_centrality",
                _link_pred,
            ),
            Tool("spectral_link_prediction", "Vector baseline: spectral embedding + cosine similarity", _spectral_lp),
            Tool("gnn_link_prediction", "(optional) GNN link prediction (PyG GraphSAGE + negative sampling)", _gnn_lp),
            Tool("cross_bridges", "Cross-community candidate edges based on Adamic-Adar", _cross_bridges),
        ]

        agent = CodeAgent(
            tools=tools,
            system_prompt=system_prompt,
            max_steps=int(getattr(settings, "hyp_agent_max_steps", 4) or 4),
            timeout_s=int(getattr(settings, "hyp_agent_timeout_seconds", 20) or 20),
        )

        task = (
            "Using the temporal knowledge graph, propose up to {k} candidate hypotheses as edges (source, predicate, target).\n"
            "Return a LIST of dicts with keys: kind, source, target, predicate, score, graph_signals.\n"
            "Where:\n"
            "- kind is one of: cross_bridge, link_prediction, central_emergence\n"
            "- predicate is a short snake_case relation like may_relate_to, influences, correlates_with\n"
            "- score is a float where higher is better\n"
            "- graph_signals is a dict of floats (e.g. adamic_adar, pagerank_u, pagerank_v, comm_u, comm_v)\n"
            "You may call: build_graph(), communities(G), centrality(G), link_prediction(G, method=..., k=...), spectral_link_prediction(G, dim=..., k=...), cross_bridges(G, comms, top_k=...)."
        ).format(k=int(top_k))

        context = f"Domain: {domain}\nQuery: {query}\nTime scope (recent): {time_scope}\n"

        try:
            raw = agent.run(task, context=context)
        except Exception as e:
            console.print(
                f"[yellow]Graph candidate agent failed ({type(e).__name__}: {e}). Falling back.[/yellow]"
            )
            return []


    # Normalize
    out: List[AgentCandidate] = []
    # smolagents often returns a string; try to parse.
    if isinstance(raw, str):
        txt = raw.strip()
        if txt:
            try:
                raw = json.loads(txt)
            except Exception:
                try:
                    raw = ast.literal_eval(txt)
                except Exception:
                    raw = []

    if isinstance(raw, dict) and "candidates" in raw:
        raw = raw.get("candidates")

    if not isinstance(raw, list):
        return []

    for r in raw[: int(top_k)]:
        if not isinstance(r, dict):
            continue
        src = str(r.get("source") or "").strip().lower()
        tgt = str(r.get("target") or "").strip().lower()
        if not src or not tgt or src == tgt:
            continue
        kind = str(r.get("kind") or "link_prediction")
        pred = str(r.get("predicate") or "may_relate_to").strip()
        try:
            score = float(r.get("score") or 0.0)
        except Exception:
            score = 0.0
        gs = r.get("graph_signals") if isinstance(r.get("graph_signals"), dict) else {}
        graph_signals: Dict[str, float] = {}
        for k, v in (gs or {}).items():
            try:
                graph_signals[str(k)] = float(v)
            except Exception:
                continue

        ev = _evidence_stub(papers, src, tgt, k=2)
        out.append(
            AgentCandidate(
                kind=kind,
                source=src,
                target=tgt,
                predicate=pred,
                score=score,
                time_scope=time_scope,
                graph_signals=graph_signals,
                evidence=ev,
            )
        )

    # stable sort
    out.sort(key=lambda c: c.score, reverse=True)
    return out[: int(top_k)]
