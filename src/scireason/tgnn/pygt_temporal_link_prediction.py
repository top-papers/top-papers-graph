from __future__ import annotations

"""Snapshot-based temporal link prediction using PyTorch Geometric Temporal.

The repo already ships with a lightweight TGNN/TGN-oriented predictor that works with the
base installation. This module adds an optional backend built on top of
`torch-geometric-temporal` for users who want recurrent snapshot models such as GConvGRU.

Design principles:
- keep imports optional so the base package stays lightweight;
- train on yearly graph snapshots derived from the temporal KG event stream;
- expose the same `(source, target, score)` contract as the existing TGNN predictor;
- gracefully fail so callers can fall back to the heuristic/TGN path.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..temporal.schemas import TemporalEvent

try:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:  # pragma: no cover
    from torch_geometric_temporal.nn.recurrent import GConvGRU
except Exception:  # pragma: no cover
    GConvGRU = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PyGTemporalLinkPredConfig:
    hidden_dim: int = 48
    k_hops: int = 2
    epochs: int = 20
    lr: float = 0.01
    weight_decay: float = 1e-4
    recent_window_years: int = 3
    min_candidate_score: float = 0.05
    negative_ratio: float = 1.0
    seed: int = 7
    device: str = "cpu"


class PyGTemporalUnavailableError(RuntimeError):
    pass


class _TemporalEdgeScorer(nn.Module):  # type: ignore[misc]
    def __init__(self, in_channels: int, hidden_dim: int, k_hops: int) -> None:
        super().__init__()
        self.encoder = GConvGRU(in_channels, hidden_dim, k_hops)
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, x, edge_index, edge_weight=None, hidden=None):
        if hidden is None:
            try:
                z = self.encoder(x, edge_index, edge_weight)
            except TypeError:
                z = self.encoder(x, edge_index)
        else:
            try:
                z = self.encoder(x, edge_index, edge_weight, H=hidden)
            except TypeError:
                try:
                    z = self.encoder(x, edge_index, edge_weight, hidden)
                except TypeError:
                    z = self.encoder(x, edge_index, hidden)
        return self.project(z)

    @staticmethod
    def pair_scores(z, pairs):
        if pairs.numel() == 0:
            return z.new_zeros((0,))
        lhs = z[pairs[:, 0]]
        rhs = z[pairs[:, 1]]
        return (lhs * rhs).sum(dim=-1)


def pygt_temporal_available() -> bool:
    return all(x is not None for x in (torch, nn, F, GConvGRU))


def _safe_year(ts: Optional[str], fallback: int = 0) -> int:
    try:
        return int(str(ts or "")[:4])
    except Exception:
        return int(fallback)


def _pair(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _build_year_aggregates(events: Sequence[TemporalEvent]) -> tuple[List[int], List[str], Dict[int, List[Tuple[int, int, float]]]]:
    nodes = sorted({str(ev.subject) for ev in events if ev.subject} | {str(ev.object) for ev in events if ev.object})
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    by_year: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for ev in events:
        year = _safe_year(ev.ts_start or ev.ts_end)
        if year <= 0:
            continue
        u = str(ev.subject or "").strip()
        v = str(ev.object or "").strip()
        if not u or not v or u == v:
            continue
        weight = float(max(ev.weight, 1.0)) * float(ev.confidence or 0.0 or 1.0)
        by_year[year].append((node_to_idx[u], node_to_idx[v], weight))

    years = sorted(by_year)
    return years, nodes, by_year


def _degree_from_edges(num_nodes: int, edges: Dict[Tuple[int, int], float]) -> np.ndarray:
    deg = np.zeros((num_nodes,), dtype=np.float32)
    for (u, v), w in edges.items():
        deg[u] += float(w)
        deg[v] += float(w)
    return deg


def _make_snapshots(
    events: Sequence[TemporalEvent],
    *,
    recent_window_years: int,
) -> tuple[List[Dict[str, np.ndarray]], List[str], set[Tuple[int, int]], Dict[int, Dict[Tuple[int, int], float]], Dict[int, List[Tuple[int, int]]], Dict[int, np.ndarray], Dict[int, int]]:
    years, nodes, by_year = _build_year_aggregates(events)
    if len(years) < 2:
        return [], nodes, set(), {}, {}, {}, {}

    num_nodes = len(nodes)
    cumulative_edges: Dict[Tuple[int, int], float] = {}
    yearly_edges: Dict[int, Dict[Tuple[int, int], float]] = {}
    yearly_positive_pairs: Dict[int, List[Tuple[int, int]]] = {}
    degree_history: Dict[int, np.ndarray] = {}
    last_seen: Dict[int, int] = {idx: years[0] for idx in range(num_nodes)}
    snapshots: List[Dict[str, np.ndarray]] = []

    for year in years:
        yearly_map: Dict[Tuple[int, int], float] = defaultdict(float)
        for u, v, w in by_year.get(year, []):
            p = _pair(u, v)
            yearly_map[p] += float(w)
            cumulative_edges[p] = cumulative_edges.get(p, 0.0) + float(w)
            last_seen[u] = year
            last_seen[v] = year
        yearly_edges[year] = dict(yearly_map)
        yearly_positive_pairs[year] = sorted(yearly_map)
        degree_history[year] = _degree_from_edges(num_nodes, cumulative_edges)

        if not cumulative_edges:
            continue
        edge_list: List[Tuple[int, int]] = []
        edge_weights: List[float] = []
        for (u, v), w in sorted(cumulative_edges.items()):
            edge_list.append((u, v))
            edge_list.append((v, u))
            edge_weights.extend([float(w), float(w)])

        edge_index = np.asarray(edge_list, dtype=np.int64).T
        edge_weight = np.asarray(edge_weights, dtype=np.float32)
        cumulative_degree = degree_history[year]

        recent_degree = np.zeros((num_nodes,), dtype=np.float32)
        for prev_year, degree_vec in degree_history.items():
            if prev_year < year - max(1, recent_window_years) + 1:
                continue
            recent_degree += degree_vec

        recency_gap = np.asarray([float(max(0, year - last_seen.get(idx, year))) for idx in range(num_nodes)], dtype=np.float32)
        current_year_degree = _degree_from_edges(num_nodes, yearly_map)
        x = np.stack(
            [
                cumulative_degree,
                recent_degree,
                current_year_degree,
                recency_gap,
                np.full((num_nodes,), float(year), dtype=np.float32),
            ],
            axis=1,
        )
        snapshots.append(
            {
                "year": np.asarray([year], dtype=np.int64),
                "x": x,
                "edge_index": edge_index,
                "edge_weight": edge_weight,
            }
        )

    existing_pairs = set(cumulative_edges)
    year_to_order = {year: idx for idx, year in enumerate(years)}
    return snapshots, nodes, existing_pairs, yearly_edges, yearly_positive_pairs, degree_history, year_to_order


def _sample_negative_pairs(
    *,
    num_nodes: int,
    forbidden: set[Tuple[int, int]],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    all_pairs = [p for p in combinations(range(num_nodes), 2) if p not in forbidden]
    if not all_pairs or num_samples <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    num = min(int(num_samples), len(all_pairs))
    if num <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    indexes = rng.choice(len(all_pairs), size=num, replace=False)
    return np.asarray([all_pairs[int(i)] for i in np.atleast_1d(indexes)], dtype=np.int64)


def _torch_tensor(array: np.ndarray, *, device: str, dtype=None):
    if dtype is None:
        return torch.as_tensor(array, device=device)
    return torch.as_tensor(array, device=device, dtype=dtype)


def pygt_temporal_link_prediction(
    events: Sequence[TemporalEvent],
    *,
    top_k: int = 30,
    config: Optional[PyGTemporalLinkPredConfig] = None,
) -> List[Tuple[str, str, float]]:
    """Predict temporal missing links using snapshot recurrent message passing.

    The model is intentionally small and CPU-friendly because this backend is optional and
    primarily meant for scientific workflows where chronological ordering matters more than
    raw benchmark performance.
    """

    if not pygt_temporal_available():  # pragma: no cover
        raise PyGTemporalUnavailableError("torch-geometric-temporal is not installed")

    cfg = config or PyGTemporalLinkPredConfig()
    snapshots, nodes, existing_pairs, yearly_edges, yearly_positive_pairs, _degree_history, year_to_order = _make_snapshots(
        events,
        recent_window_years=int(cfg.recent_window_years or 3),
    )
    if len(snapshots) < 2 or len(nodes) < 2:
        return []

    rng = np.random.default_rng(int(cfg.seed or 7))
    torch.manual_seed(int(cfg.seed or 7))
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(int(cfg.seed or 7))

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = int(snapshots[0]["x"].shape[1])
    model = _TemporalEdgeScorer(in_channels=in_channels, hidden_dim=int(cfg.hidden_dim or 48), k_hops=int(cfg.k_hops or 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr or 0.01), weight_decay=float(cfg.weight_decay or 0.0))
    criterion = nn.BCEWithLogitsLoss()

    ordered_snapshot_years = [int(snap["year"][0]) for snap in snapshots]
    train_windows = list(zip(ordered_snapshot_years[:-1], ordered_snapshot_years[1:]))
    if not train_windows:
        return []

    for _epoch in range(max(1, int(cfg.epochs or 1))):
        model.train()
        optimizer.zero_grad()
        hidden = None
        losses = []

        for src_year, target_year in train_windows:
            snap = next(s for s in snapshots if int(s["year"][0]) == src_year)
            x = _torch_tensor(snap["x"], device=device, dtype=torch.float32)
            edge_index = _torch_tensor(snap["edge_index"], device=device, dtype=torch.long)
            edge_weight = _torch_tensor(snap["edge_weight"], device=device, dtype=torch.float32)
            hidden = model.encode(x, edge_index, edge_weight=edge_weight, hidden=hidden)
            hidden = F.relu(hidden)

            positives = np.asarray(list(yearly_positive_pairs.get(target_year) or []), dtype=np.int64)
            if positives.size == 0:
                continue
            pos_pairs = torch.as_tensor(positives, device=device, dtype=torch.long)

            forbidden = set(yearly_edges.get(src_year, {})) | set(yearly_edges.get(target_year, {}))
            neg_count = max(1, int(round(len(positives) * max(float(cfg.negative_ratio or 1.0), 0.25))))
            negatives = _sample_negative_pairs(num_nodes=len(nodes), forbidden=forbidden, num_samples=neg_count, rng=rng)
            neg_pairs = torch.as_tensor(negatives, device=device, dtype=torch.long) if negatives.size else torch.zeros((0, 2), device=device, dtype=torch.long)

            pos_scores = model.pair_scores(hidden, pos_pairs)
            neg_scores = model.pair_scores(hidden, neg_pairs)
            logits = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat(
                [
                    torch.ones((pos_scores.shape[0],), device=device, dtype=torch.float32),
                    torch.zeros((neg_scores.shape[0],), device=device, dtype=torch.float32),
                ],
                dim=0,
            )
            losses.append(criterion(logits, labels))

        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        hidden = None
        final_z = None
        last_year = ordered_snapshot_years[-1]
        for snap in snapshots:
            x = _torch_tensor(snap["x"], device=device, dtype=torch.float32)
            edge_index = _torch_tensor(snap["edge_index"], device=device, dtype=torch.long)
            edge_weight = _torch_tensor(snap["edge_weight"], device=device, dtype=torch.float32)
            hidden = model.encode(x, edge_index, edge_weight=edge_weight, hidden=hidden)
            hidden = F.relu(hidden)
            final_z = hidden
        if final_z is None:
            return []
        final_z = F.normalize(final_z, p=2.0, dim=-1)

        pair_candidates = [p for p in combinations(range(len(nodes)), 2) if p not in existing_pairs]
        if not pair_candidates:
            return []
        pair_tensor = torch.as_tensor(np.asarray(pair_candidates, dtype=np.int64), device=device, dtype=torch.long)
        logits = model.pair_scores(final_z, pair_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    # Recency prior from latest active year.
    latest_incident_year: DefaultDict[int, int] = defaultdict(lambda: ordered_snapshot_years[-1])
    for year, pairs in yearly_positive_pairs.items():
        for u, v in pairs:
            latest_incident_year[u] = max(latest_incident_year[u], year)
            latest_incident_year[v] = max(latest_incident_year[v], year)

    scored: List[Tuple[str, str, float]] = []
    for (u, v), prob in zip(pair_candidates, probs.tolist()):
        age_u = max(0, ordered_snapshot_years[-1] - latest_incident_year[u])
        age_v = max(0, ordered_snapshot_years[-1] - latest_incident_year[v])
        recency = float(np.exp(-(age_u + age_v) / (2.0 * max(1.0, float(cfg.recent_window_years or 3)))))
        score = 0.85 * float(prob) + 0.15 * recency
        if score < float(cfg.min_candidate_score or 0.0):
            continue
        scored.append((nodes[u], nodes[v], score))

    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[: int(top_k)]
