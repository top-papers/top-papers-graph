from __future__ import annotations

"""Graph analysis tools exposed to the code-writing agent.

We focus on methods that are:
- open source
- easy to install
- suitable for small/medium classroom graphs

Under the hood we use NetworkX for classic algorithms (community detection, shortest paths,
centrality, link prediction, etc.). NetworkX link prediction & community APIs are well
maintained and easy to reason about.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..temporal.temporal_kg_builder import TemporalKnowledgeGraph


def build_nx_graph(
    kg: TemporalKnowledgeGraph,
    *,
    directed: bool = False,
    weight: str = "score",
    min_total_count: int = 1,
) -> nx.Graph:
    """Convert TemporalKnowledgeGraph to a NetworkX graph.

    Nodes: terms.
    Edges: aggregated relations. Weight defaults to edge.score.
    """

    G: nx.Graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for term in kg.nodes.keys():
        G.add_node(term)

    for e in kg.edges:
        if int(getattr(e, "total_count", 0) or 0) < int(min_total_count):
            continue
        w = float(getattr(e, weight, 1.0) or 1.0) if hasattr(e, weight) else float(e.features.get(weight, 1.0) or 1.0)
        a, b = e.source, e.target
        if not a or not b or a == b:
            continue
        # Keep the best weight if multiple predicates map to same pair.
        if G.has_edge(a, b):
            prev = G[a][b].get("weight", 0.0)
            if w > prev:
                G[a][b]["weight"] = w
        else:
            G.add_edge(a, b, weight=w, predicate=e.predicate, total_count=e.total_count, trend=float(e.features.get("trend", 0.0) or 0.0))

    return G


def graph_summary(G: nx.Graph) -> Dict[str, Any]:
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "is_directed": G.is_directed(),
        "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
    }


def top_central_nodes(G: nx.Graph, *, k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    """Compute a few centrality rankings."""

    out: Dict[str, List[Tuple[str, float]]] = {}
    if G.number_of_nodes() == 0:
        return out

    try:
        pr = nx.pagerank(G, weight="weight")
        out["pagerank"] = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:k]
    except Exception:
        pass

    try:
        dc = nx.degree_centrality(G)
        out["degree"] = sorted(dc.items(), key=lambda kv: kv[1], reverse=True)[:k]
    except Exception:
        pass

    try:
        bc = nx.betweenness_centrality(G, weight=None)
        out["betweenness"] = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)[:k]
    except Exception:
        pass

    return out


def communities_greedy_modularity(G: nx.Graph, *, max_communities: int = 12) -> List[List[str]]:
    """Community detection via greedy modularity maximization."""

    if G.number_of_nodes() == 0:
        return []

    # Works for undirected graphs; if directed, project to undirected.
    H = G.to_undirected() if G.is_directed() else G
    comms = list(nx.algorithms.community.greedy_modularity_communities(H, weight="weight"))
    comms = sorted(comms, key=lambda c: len(c), reverse=True)
    return [sorted(list(c)) for c in comms[:max_communities]]


def communities_label_propagation(G: nx.Graph, *, max_communities: int = 12) -> List[List[str]]:
    """Fast community detection via label propagation."""

    if G.number_of_nodes() == 0:
        return []
    H = G.to_undirected() if G.is_directed() else G
    try:
        comms = list(nx.algorithms.community.label_propagation_communities(H))
        comms = sorted(comms, key=lambda c: len(c), reverse=True)
        return [sorted(list(c)) for c in comms[:max_communities]]
    except Exception:
        return []


def shortest_path_terms(G: nx.Graph, source: str, target: str, *, weight: Optional[str] = None) -> List[str]:
    if not source or not target:
        return []
    if source not in G or target not in G:
        return []
    try:
        return list(nx.shortest_path(G, source=source, target=target, weight=weight))
    except Exception:
        return []


def link_prediction(
    G: nx.Graph,
    *,
    method: str = "adamic_adar",
    k: int = 30,
    exclude_existing: bool = True,
) -> List[Tuple[str, str, float]]:
    """Classic structural link prediction heuristics."""

    H = G.to_undirected() if G.is_directed() else G

    # Candidate pairs: non-edges
    ebunch = None
    if exclude_existing:
        # networkx predictors accept ebunch; for small graphs leaving it None is fine.
        ebunch = None

    if method == "jaccard":
        preds = nx.jaccard_coefficient(H, ebunch)
    elif method == "preferential_attachment":
        preds = nx.preferential_attachment(H, ebunch)
    elif method == "common_neighbor_centrality":
        preds = nx.common_neighbor_centrality(H, ebunch)
    else:
        preds = nx.adamic_adar_index(H, ebunch)

    out = [(u, v, float(p)) for (u, v, p) in preds]
    out.sort(key=lambda t: t[2], reverse=True)

    # Filter out existing edges if requested
    if exclude_existing:
        out = [(u, v, s) for (u, v, s) in out if not H.has_edge(u, v)]

    return out[:k]


def cross_community_bridges(G: nx.Graph, communities: Sequence[Sequence[str]], *, top_k: int = 20) -> List[Tuple[str, str, float]]:
    """Suggest cross-community candidate edges by combining community structure + link prediction.

    A simple heuristic:
    - compute adamic-adar candidates
    - keep only pairs from different communities
    """

    comm_of: Dict[str, int] = {}
    for i, c in enumerate(communities):
        for n in c:
            comm_of[n] = i

    preds = link_prediction(G, method="adamic_adar", k=200)
    out: List[Tuple[str, str, float]] = []
    for u, v, s in preds:
        if comm_of.get(u) is None or comm_of.get(v) is None:
            continue
        if comm_of[u] == comm_of[v]:
            continue
        out.append((u, v, s))
        if len(out) >= top_k:
            break
    return out


def spectral_link_prediction(
    G: nx.Graph,
    *,
    dim: int = 8,
    k: int = 30,
    degree_cap: int = 300,
) -> List[Tuple[str, str, float]]:
    """A simple "vector" baseline: spectral embedding + cosine similarity.

    We compute a low-rank embedding of the adjacency matrix (eigenvectors of A) and
    propose non-edge pairs with high cosine similarity.
    """

    H = G.to_undirected() if G.is_directed() else G
    nodes = list(H.nodes())
    if len(nodes) < 3:
        return []

    # For large graphs, restrict to high-degree nodes to keep O(n^2) manageable.
    if len(nodes) > degree_cap:
        nodes = sorted(nodes, key=lambda n: H.degree(n), reverse=True)[:degree_cap]

    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for u, v, data in H.edges(data=True):
        if u not in idx or v not in idx:
            continue
        w = float(data.get("weight", 1.0) or 1.0)
        iu, iv = idx[u], idx[v]
        A[iu, iv] = w
        A[iv, iu] = w

    # Eigen-decomposition (dense; fine for classroom sizes)
    try:
        vals, vecs = np.linalg.eigh(A)
    except Exception:
        return []

    # Take top-|vals| components
    order = np.argsort(vals)[::-1]
    d = max(2, min(int(dim), n - 1))
    V = vecs[:, order[:d]]

    # Normalize for cosine similarity
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Vn = V / norms

    out: List[Tuple[str, str, float]] = []
    for i in range(n):
        ui = nodes[i]
        for j in range(i + 1, n):
            vj = nodes[j]
            if H.has_edge(ui, vj):
                continue
            score = float(np.dot(Vn[i], Vn[j]))
            out.append((ui, vj, score))

    out.sort(key=lambda t: t[2], reverse=True)
    return out[:k]
