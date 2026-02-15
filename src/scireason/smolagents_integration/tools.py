from __future__ import annotations

"""smolagents Tool wrappers for graph algorithms.

We intentionally reuse the *exact same* pure-python implementations used by the built-in
agent (NetworkX + NumPy + optional PyG), but expose them through the `smolagents.tool`
decorator so they show up correctly in CodeAgent prompts.
"""

import importlib.util
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from ..agentic import graph_tools
from ..config import settings


def _require_smolagents():
    if importlib.util.find_spec("smolagents") is None:
        raise RuntimeError(
            "smolagents не установлен. Установите зависимости: pip install -e '.[agents]'"
        )


def make_graph_tools(
    *,
    edges: List[Tuple[str, str]],
    weights: Optional[List[float]] = None,
    directed: bool = False,
) -> List[Any]:
    """Create a list of smolagents tools with closures over the current graph data."""

    _require_smolagents()

    from smolagents import tool  # type: ignore

    # Build a graph once and reuse it.
    _G_cache: Dict[str, nx.Graph] = {}

    @tool
    def build_graph() -> Any:
        """Build a NetworkX graph from the current temporal knowledge graph edges.

        Returns:
            A NetworkX Graph (or DiGraph) with edge weights.
        """

        if "G" in _G_cache:
            return _G_cache["G"]
        G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
        ws = weights or [1.0] * len(edges)
        for (u, v), w in zip(edges, ws):
            if not u or not v or u == v:
                continue
            G.add_edge(str(u), str(v), weight=float(w))
        _G_cache["G"] = G
        return G

    @tool
    def graph_summary(G: Any) -> Dict[str, Any]:
        """Return basic statistics for a graph.

        Args:
            G: networkx Graph.
        """

        return graph_tools.graph_summary(G)

    @tool
    def communities(G: Any, method: str = "greedy", max_communities: int = 12) -> List[List[str]]:
        """Detect communities.

        Args:
            G: networkx Graph.
            method: greedy|lpa.
            max_communities: maximum number of communities to return.
        """

        method = (method or "greedy").lower()
        if method in {"lpa", "label", "label_propagation"}:
            return graph_tools.communities_label_propagation(G, max_communities=max_communities)
        return graph_tools.communities_greedy_modularity(G, max_communities=max_communities)

    @tool
    def centrality(G: Any, k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Compute centrality rankings (pagerank, degree, betweenness).

        Args:
            G: networkx Graph.
            k: top-k per ranking.
        """

        return graph_tools.top_central_nodes(G, k=k)

    @tool
    def shortest_path_terms(G: Any, source: str, target: str) -> List[str]:
        """Shortest path between two terms.

        Args:
            G: networkx Graph.
            source: source term.
            target: target term.
        """

        return graph_tools.shortest_path_terms(G, source, target)

    @tool
    def link_prediction(G: Any, method: str = "adamic_adar", k: int = 30) -> List[Tuple[str, str, float]]:
        """Predict missing edges using classic heuristics.

        Args:
            G: networkx Graph.
            method: adamic_adar|jaccard|preferential_attachment|common_neighbor_centrality.
            k: top-k candidates.
        """

        return graph_tools.link_prediction(G, method=method, k=k)

    @tool
    def spectral_link_prediction(G: Any, dim: int = 8, k: int = 30) -> List[Tuple[str, str, float]]:
        """Predict missing edges using a simple spectral embedding + cosine similarity.

        Args:
            G: networkx Graph.
            dim: embedding dimension.
            k: top-k candidates.
        """

        return graph_tools.spectral_link_prediction(G, dim=dim, k=k)

    @tool
    def cross_bridges(G: Any, comms: Sequence[Sequence[str]], top_k: int = 20) -> List[Tuple[str, str, float]]:
        """Suggest cross-community candidate edges.

        Args:
            G: networkx Graph.
            comms: communities (list of lists of node ids).
            top_k: number of candidates.
        """

        return graph_tools.cross_community_bridges(G, comms, top_k=top_k)

    # Optional PyG tool
    extra_tools: List[Any] = []
    if getattr(settings, "hyp_gnn_enabled", False):

        @tool
        def gnn_link_prediction(G: Any, k: int = 30) -> List[Tuple[str, str, float]]:
            """(optional) GNN link prediction using PyTorch Geometric (GraphSAGE baseline).

            Args:
                G: networkx Graph.
                k: top-k candidates.
            """

            from ..hypotheses.gnn_link_prediction import gnn_link_prediction as _gnn

            return _gnn(G, k=k)

        extra_tools.append(gnn_link_prediction)

    return [
        build_graph,
        graph_summary,
        communities,
        centrality,
        shortest_path_terms,
        link_prediction,
        spectral_link_prediction,
        cross_bridges,
        *extra_tools,
    ]
