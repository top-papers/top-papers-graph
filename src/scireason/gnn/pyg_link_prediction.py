from __future__ import annotations

"""PyTorch Geometric (PyG) link prediction helper.

Why this exists
--------------
NetworkX heuristics (Adamic-Adar, Jaccard, etc.) are great baselines, but a GNN can:
- combine multiple hops of structure into node embeddings
- generalize better on sparse graphs where common-neighbor heuristics degrade

This module is optional and must *not* break the base installation.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import networkx as nx


class PyGUnavailableError(RuntimeError):
    """Raised when PyG is required but not importable."""


def pyg_available() -> bool:
    """Return True if torch_geometric is importable."""
    try:
        import torch_geometric  # noqa: F401

        return True
    except Exception:
        return False


def _require_pyg() -> None:
    if not pyg_available():
        raise PyGUnavailableError(
            "PyTorch Geometric (torch_geometric) is not installed. "
            "Install optional dependencies: pip install -e '.[gnn]'"
        )


@dataclass(frozen=True)
class PyGLinkPredConfig:
    epochs: int = 80
    hidden_dim: int = 64
    lr: float = 0.01
    node_cap: int = 300
    seed: int = 7
    device: str = "cpu"


def _top_nodes_by_degree(G: nx.Graph, *, node_cap: int) -> List[str]:
    nodes = list(G.nodes())
    if node_cap <= 0 or len(nodes) <= node_cap:
        return nodes

    # Prefer weighted degree if available.
    try:
        deg = dict(G.degree(weight="weight"))
    except Exception:
        deg = dict(G.degree())

    nodes_sorted = sorted(nodes, key=lambda n: float(deg.get(n, 0.0)), reverse=True)
    return nodes_sorted[: int(node_cap)]


def pyg_link_prediction(
    G: nx.Graph,
    *,
    top_k: int = 30,
    config: Optional[PyGLinkPredConfig] = None,
) -> List[Tuple[str, str, float]]:
    """Run a small GraphSAGE link prediction model on an (induced) subgraph.

    Parameters
    ----------
    G:
        NetworkX graph (undirected recommended).
    top_k:
        Number of candidate non-edges to return.
    config:
        Training/runtime knobs.

    Returns
    -------
    List[(u, v, score)]
        Proposed missing edges sorted by decreasing score.

    Notes
    -----
    - For large graphs we train on the induced subgraph of `node_cap` highest-degree nodes.
    - This is meant as a *course-friendly* GNN baseline, not a SOTA temporal GNN.
    """

    cfg = config or PyGLinkPredConfig()

    if G.number_of_nodes() < 3 or G.number_of_edges() < 2:
        return []

    # Lazy import so base installation stays clean.
    _require_pyg()
    import random

    import torch
    import torch.nn.functional as F
    from torch import nn

    # These imports are the minimal surface we rely on.
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import negative_sampling

    # ---- deterministic ----
    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(cfg.device or "cpu")

    # ---- induced subgraph for speed ----
    nodes = _top_nodes_by_degree(G, node_cap=int(cfg.node_cap))
    H = G.subgraph(nodes).copy()

    if H.number_of_nodes() < 3 or H.number_of_edges() < 2:
        return []

    node_list = list(H.nodes())
    idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    # Positive edges for supervision (unique, undirected)
    pos_edges: List[Tuple[int, int]] = []
    for u, v in H.edges():
        if u == v:
            continue
        iu, iv = idx[u], idx[v]
        if iu == iv:
            continue
        if iu < iv:
            pos_edges.append((iu, iv))
        else:
            pos_edges.append((iv, iu))

    # Deduplicate
    pos_edges = sorted(list(set(pos_edges)))
    if len(pos_edges) < 2:
        return []

    # Message-passing edges (both directions)
    mp_edges: List[Tuple[int, int]] = []
    for iu, iv in pos_edges:
        mp_edges.append((iu, iv))
        mp_edges.append((iv, iu))

    edge_index = torch.tensor(mp_edges, dtype=torch.long).t().contiguous().to(device)
    pos_edge_label_index = torch.tensor(pos_edges, dtype=torch.long).t().contiguous().to(device)

    class _SageLP(nn.Module):
        def __init__(self, n_nodes: int, hidden_dim: int) -> None:
            super().__init__()
            self.emb = nn.Embedding(n_nodes, hidden_dim)
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.emb.weight
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

        def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
            src, dst = edge_label_index[0], edge_label_index[1]
            return (z[src] * z[dst]).sum(dim=-1)

    model = _SageLP(num_nodes, int(cfg.hidden_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    # ---- train ----
    for _ in range(int(cfg.epochs)):
        model.train()
        opt.zero_grad()

        z = model.encode(edge_index)

        neg_edge_label_index = negative_sampling(
            edge_index=pos_edge_label_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_label_index.size(1),
            force_undirected=True,
        ).to(device)

        edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
        edge_label = torch.cat(
            [torch.ones(pos_edge_label_index.size(1)), torch.zeros(neg_edge_label_index.size(1))],
            dim=0,
        ).to(device)

        logits = model.decode(z, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, edge_label)
        loss.backward()
        opt.step()

    # ---- predict ----
    model.eval()
    with torch.no_grad():
        z = model.encode(edge_index)
        scores = torch.sigmoid(z @ z.t())

        # remove self-loops and existing edges
        scores.fill_diagonal_(0.0)
        for iu, iv in pos_edges:
            scores[iu, iv] = 0.0
            scores[iv, iu] = 0.0

        # Upper triangle ranking
        tri = torch.triu_indices(num_nodes, num_nodes, offset=1)
        tri_scores = scores[tri[0], tri[1]]
        if tri_scores.numel() == 0:
            return []

        k = min(int(top_k), int(tri_scores.numel()))
        vals, best = torch.topk(tri_scores, k=k)

        out: List[Tuple[str, str, float]] = []
        for v, idx_flat in zip(vals.tolist(), best.tolist()):
            iu = int(tri[0][idx_flat])
            iv = int(tri[1][idx_flat])
            u = node_list[iu]
            w = node_list[iv]
            if not u or not w or u == w:
                continue
            out.append((str(u), str(w), float(v)))

    out.sort(key=lambda t: t[2], reverse=True)
    return out[: int(top_k)]
