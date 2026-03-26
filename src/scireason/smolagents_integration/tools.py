from __future__ import annotations

"""smolagents tool set.

This module provides **safe, course-friendly** tools for the smolagents CodeAgent.

Goals
-----
- Keep the base project runnable without heavy infra.
- Still expose the core project capabilities as tools:
  - graph algorithms (NetworkX)
  - optional research via open APIs (paper search)
  - lightweight vector store (local in-memory; optional Qdrant)
  - lightweight graph store (local in-memory)

Notes
-----
* Tools are deliberately *small* and *composable* so students can read/extend them.
* If optional deps are missing, tools degrade gracefully.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

try:  # pragma: no cover
    import networkx as nx
except Exception as e:  # pragma: no cover
    raise RuntimeError("networkx is required for smolagents graph tools") from e

try:  # pragma: no cover
    from smolagents import tool
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "smolagents is not installed. Install optional dependency: pip install -e '.[agents]'"
    ) from e

from ..agentic.graph_tools import (
    communities_greedy_modularity,
    cross_community_bridges,
    graph_summary as _graph_summary,
    link_prediction as _link_prediction,
    spectral_link_prediction as _spectral_link_prediction,
    top_central_nodes,
)
from ..config import settings
from ..llm import embed
from ..papers.service import search_papers

# Optional: qdrant for a real vector DB (still free/open-source).
try:  # pragma: no cover
    from ..graph.qdrant_store import QdrantStore

    _HAS_QDRANT = True
except Exception:  # pragma: no cover
    QdrantStore = None  # type: ignore
    _HAS_QDRANT = False


# ---------------------------------------------------------------------------
# Simple local stores (for CI/offline/classroom)
# ---------------------------------------------------------------------------

_LOCAL_VECTOR_STORES: Dict[str, Dict[str, Any]] = {}
_LOCAL_GRAPH_STORES: Dict[str, nx.Graph] = {}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += float(x) * float(y)
        da += float(x) * float(x)
        db += float(y) * float(y)
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return num / (math.sqrt(da) * math.sqrt(db))


def _local_vs_upsert(collection: str, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
    store = _LOCAL_VECTOR_STORES.setdefault(collection, {"ids": [], "vecs": [], "payloads": []})
    store["ids"].extend(list(ids))
    store["vecs"].extend(list(vectors))
    store["payloads"].extend(list(payloads))


def _local_vs_search(collection: str, query_vector: List[float], limit: int = 8) -> List[Dict[str, Any]]:
    store = _LOCAL_VECTOR_STORES.get(collection)
    if not store:
        return []
    ids = store.get("ids") or []
    vecs = store.get("vecs") or []
    payloads = store.get("payloads") or []

    scored: List[Tuple[float, int]] = []
    for i, v in enumerate(vecs):
        try:
            s = _cosine(query_vector, v)
        except Exception:
            s = 0.0
        scored.append((s, i))
    scored.sort(key=lambda t: t[0], reverse=True)

    out: List[Dict[str, Any]] = []
    for s, i in scored[: int(limit)]:
        out.append({"id": ids[i], "score": float(s), "payload": payloads[i]})
    return out


def _graph_store_put(name: str, G: nx.Graph) -> None:
    _LOCAL_GRAPH_STORES[name] = G


def _graph_store_get(name: str) -> Optional[nx.Graph]:
    return _LOCAL_GRAPH_STORES.get(name)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def make_graph_tools(
    *,
    edges: List[Tuple[str, str]],
    weights: Optional[List[float]] = None,
    directed: bool = False,
) -> List[Any]:
    """Create a standard tool set for the graph reasoning agent.

    Parameters
    ----------
    edges:
        Edge list (u, v).
    weights:
        Optional per-edge weights.
    directed:
        Whether to build a DiGraph.
    """

    w = weights or [1.0 for _ in edges]

    @tool
    def build_graph() -> Any:
        """Build an in-memory NetworkX graph from the KG edge list."""
        G = nx.DiGraph() if directed else nx.Graph()
        for (u, v), ww in zip(edges, w, strict=True):
            uu = str(u).strip().lower()
            vv = str(v).strip().lower()
            if not uu or not vv or uu == vv:
                continue
            if G.has_edge(uu, vv):
                G[uu][vv]["weight"] = float(G[uu][vv].get("weight", 1.0)) + float(ww)
            else:
                G.add_edge(uu, vv, weight=float(ww))
        return G

    @tool
    def graph_summary(G: Any) -> Dict[str, Any]:
        """Return basic graph stats: #nodes, #edges, density."""
        try:
            return _graph_summary(G)
        except Exception:
            try:
                return {"nodes": int(getattr(G, "number_of_nodes")()), "edges": int(getattr(G, "number_of_edges")())}
            except Exception:
                return {}

    @tool
    def communities(G: Any, method: str = "greedy", max_communities: int = 12) -> List[List[str]]:
        """Community detection.

        method: only 'greedy' is supported in this lightweight mode.
        """
        try:
            comms = communities_greedy_modularity(G)
            return [list(c)[:500] for c in comms[: int(max_communities)]]
        except Exception:
            return []

    @tool
    def centrality(G: Any, k: int = 10) -> Dict[str, Any]:
        """Centrality ranking (pagerank/degree/betweenness)."""
        try:
            return top_central_nodes(G, k=int(k))
        except Exception:
            return {}

    @tool
    def link_prediction(G: Any, method: str = "adamic_adar", k: int = 30) -> List[List[Any]]:
        """Classic link prediction (adamic_adar|jaccard|preferential_attachment|common_neighbor_centrality)."""
        try:
            return [list(x) for x in _link_prediction(G, method=method, k=int(k))]
        except Exception:
            return []

    @tool
    def spectral_link_prediction(G: Any, dim: int = 8, k: int = 30) -> List[List[Any]]:
        """Vector baseline: spectral embedding + cosine similarity."""
        try:
            return [list(x) for x in _spectral_link_prediction(G, dim=int(dim), k=int(k))]
        except Exception:
            return []

    @tool
    def cross_bridges(G: Any, comms: List[List[str]], top_k: int = 20) -> List[List[Any]]:
        """Cross-community candidate edges."""
        try:
            return [list(x) for x in cross_community_bridges(G, comms, top_k=int(top_k))]
        except Exception:
            return []

    # ----------------------
    # Research / storage tools
    # ----------------------

    @tool
    def api_search_papers(
        query: str,
        limit: int = 10,
        sources: str = "semantic_scholar,openalex",
        with_abstracts: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search papers using free/open APIs (best-effort).

        Returns a list of normalized metadata dicts.

        sources: comma-separated list. Supported: semantic_scholar, openalex, crossref, pubmed, europe_pmc, arxiv.
        """
        q = (query or "").strip()
        if not q:
            return []
        # map to enum values used in service
        from ..papers.schema import PaperSource

        srcs: List[PaperSource] = []
        for s in (sources or "").split(","):
            name = s.strip().lower()
            if not name:
                continue
            try:
                srcs.append(PaperSource(name))
            except Exception:
                continue
        try:
            papers = search_papers(q, limit=int(limit), sources=srcs or None, with_abstracts=bool(with_abstracts))
            return [p.model_dump(mode="json") for p in papers]
        except Exception:
            return []

    @tool
    def vector_index(
        collection: str,
        texts: List[str],
        ids: Optional[List[str]] = None,
        backend: str = "auto",
    ) -> Dict[str, Any]:
        """Index texts into a vector store.

        backend:
          - auto (prefer Qdrant if available)
          - local (always available, in-memory)
          - qdrant (requires qdrant-client)

        Returns {collection, backend, count}.
        """
        col = (collection or "agent_tmp").strip() or "agent_tmp"
        if not texts:
            return {"collection": col, "backend": "none", "count": 0}
        ids = ids or [f"item:{i}" for i in range(len(texts))]
        try:
            vecs = embed(list(texts))
        except Exception:
            vecs = []
        payloads = [{"text": t} for t in texts]

        use_qdrant = (backend or "auto").lower() in {"auto", "qdrant"} and _HAS_QDRANT
        if use_qdrant:
            try:
                store = QdrantStore(url=getattr(settings, "qdrant_url", ":memory:"))  # type: ignore
                store.ensure_collection(col, vector_size=len(vecs[0]) if vecs else int(getattr(settings, "hash_embed_dim", 384)))
                store.upsert(col, ids=list(ids), vectors=list(vecs), payloads=payloads)
                return {"collection": col, "backend": "qdrant", "count": len(texts)}
            except Exception:
                # fall back to local
                pass

        _local_vs_upsert(col, ids=list(ids), vectors=list(vecs), payloads=payloads)
        return {"collection": col, "backend": "local", "count": len(texts)}

    @tool
    def vector_search(collection: str, query: str, limit: int = 8, backend: str = "auto") -> List[Dict[str, Any]]:
        """Vector search over indexed texts."""
        col = (collection or "agent_tmp").strip() or "agent_tmp"
        q = (query or "").strip()
        if not q:
            return []
        try:
            qv = embed([q])[0]
        except Exception:
            qv = []

        use_qdrant = (backend or "auto").lower() in {"auto", "qdrant"} and _HAS_QDRANT
        if use_qdrant:
            try:
                store = QdrantStore(url=getattr(settings, "qdrant_url", ":memory:"))  # type: ignore
                return store.search(col, query_vector=list(qv), limit=int(limit))
            except Exception:
                pass

        return _local_vs_search(col, query_vector=list(qv), limit=int(limit))

    @tool
    def graph_store_put(name: str, G: Any) -> Dict[str, Any]:
        """Persist the current graph in a lightweight in-memory graph store."""
        n = (name or "agent_graph").strip() or "agent_graph"
        try:
            _graph_store_put(n, G)
            return {"ok": True, "name": n, "nodes": int(G.number_of_nodes()), "edges": int(G.number_of_edges())}
        except Exception as e:
            return {"ok": False, "name": n, "error": f"{type(e).__name__}: {e}"}

    @tool
    def graph_store_neighbors(name: str, node: str, limit: int = 20) -> List[str]:
        """Get neighbors for a node from a stored graph."""
        n = (name or "agent_graph").strip() or "agent_graph"
        G = _graph_store_get(n)
        if G is None:
            return []
        nd = (node or "").strip().lower()
        if not nd or nd not in G:
            return []
        nbrs = list(G.neighbors(nd))
        return [str(x) for x in nbrs[: int(limit)]]

    @tool
    def graph_store_shortest_path(name: str, source: str, target: str, cutoff: int = 6) -> List[str]:
        """Shortest path in stored graph (unweighted)."""
        n = (name or "agent_graph").strip() or "agent_graph"
        G = _graph_store_get(n)
        if G is None:
            return []
        s = (source or "").strip().lower()
        t = (target or "").strip().lower()
        if not s or not t or s not in G or t not in G:
            return []
        try:
            p = nx.shortest_path(G, s, t)
            if len(p) - 1 > int(cutoff):
                return []
            return [str(x) for x in p]
        except Exception:
            return []

    return [
        build_graph,
        graph_summary,
        communities,
        centrality,
        link_prediction,
        spectral_link_prediction,
        cross_bridges,
        api_search_papers,
        vector_index,
        vector_search,
        graph_store_put,
        graph_store_neighbors,
        graph_store_shortest_path,
    ]
