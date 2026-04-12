from __future__ import annotations

"""Typed temporal link prediction using PyTorch Geometric Temporal.

This backend keeps the snapshot-based GConvGRU encoder but upgrades the prediction layer
from pair-only scoring to typed quadruples (u, r, v, t). The encoder operates over the
historical interaction graph while a relation-aware scorer conditions on:
- source node state
- target node state
- relation embedding
- time-step embedding
- element-wise interaction between node states

The implementation is intentionally CPU-friendly and best-effort. If the optional PyG
Temporal dependency is unavailable or the model cannot train, callers should fall back to
`scireason.tgnn.tgn_link_prediction`.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..temporal.schemas import TemporalEvent
from .prediction_types import LinkPredictionRecord, build_semantic_heuristic_predictions, collect_relation_prediction_stats, normalize_predicate, normalize_term, relation_direction, semantic_candidate_support

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
    relation_dim: int = 24
    time_dim: int = 12
    k_hops: int = 2
    epochs: int = 20
    lr: float = 0.01
    weight_decay: float = 1e-4
    recent_window_years: int = 3
    min_candidate_score: float = 0.05
    negative_ratio: float = 1.0
    max_relation_candidates: int = 8
    seed: int = 7
    device: str = "cpu"


class PyGTemporalUnavailableError(RuntimeError):
    pass


class _TypedTemporalEdgeScorer(nn.Module):  # type: ignore[misc]
    def __init__(self, in_channels: int, hidden_dim: int, relation_dim: int, time_dim: int, k_hops: int, num_relations: int, num_times: int) -> None:
        super().__init__()
        self.encoder = GConvGRU(in_channels, hidden_dim, k_hops)
        self.project = nn.Linear(hidden_dim, hidden_dim)
        self.relation_emb = nn.Embedding(max(1, num_relations), relation_dim)
        self.time_emb = nn.Embedding(max(1, num_times), time_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3 + relation_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 2 else 1),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 2 else 1, 1),
        )

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

    def quadruple_scores(self, z, quadruples):
        if quadruples.numel() == 0:
            return z.new_zeros((0,))
        lhs = z[quadruples[:, 0]]
        rel = self.relation_emb(quadruples[:, 1])
        rhs = z[quadruples[:, 2]]
        tim = self.time_emb(quadruples[:, 3])
        interaction = lhs * rhs
        feats = torch.cat([lhs, rhs, interaction, rel, tim], dim=-1)
        return self.scorer(feats).squeeze(-1)


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
    nodes = sorted({normalize_term(ev.subject) for ev in events if ev.subject} | {normalize_term(ev.object) for ev in events if ev.object})
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    by_year: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for ev in events:
        year = _safe_year(ev.ts_start or ev.ts_end)
        if year <= 0:
            continue
        u = normalize_term(ev.subject)
        v = normalize_term(ev.object)
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
) -> tuple[List[Dict[str, np.ndarray]], List[str], Dict[int, Dict[Tuple[int, int], float]], Dict[int, int]]:
    years, nodes, by_year = _build_year_aggregates(events)
    if len(years) < 2:
        return [], nodes, {}, {}

    num_nodes = len(nodes)
    cumulative_edges: Dict[Tuple[int, int], float] = {}
    yearly_edges: Dict[int, Dict[Tuple[int, int], float]] = {}
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
        snapshots.append({"year": np.asarray([year], dtype=np.int64), "x": x, "edge_index": edge_index, "edge_weight": edge_weight})

    year_to_order = {year: idx for idx, year in enumerate(years)}
    return snapshots, nodes, yearly_edges, year_to_order


def _build_typed_targets(
    events: Sequence[TemporalEvent],
    *,
    node_to_idx: Dict[str, int],
    predicate_to_idx: Dict[str, int],
    year_to_order: Dict[int, int],
) -> tuple[Dict[int, List[Tuple[int, int, int, int]]], Dict[str, set[Tuple[str, str]]], Dict[str, set[Tuple[str, str]]]]:
    by_year: DefaultDict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    directed_existing: DefaultDict[str, set[Tuple[str, str]]] = defaultdict(set)
    undirected_existing: DefaultDict[str, set[Tuple[str, str]]] = defaultdict(set)
    for ev in events:
        year = _safe_year(ev.ts_start or ev.ts_end)
        if year not in year_to_order:
            continue
        src = normalize_term(ev.subject)
        dst = normalize_term(ev.object)
        pred = normalize_predicate(ev.predicate)
        if not src or not dst or not pred or src == dst:
            continue
        u = node_to_idx[src]
        v = node_to_idx[dst]
        r = predicate_to_idx[pred]
        t = year_to_order[year]
        by_year[year].append((u, r, v, t))
        if relation_direction(pred) == 'undirected':
            undirected_existing[pred].add(tuple(sorted((src, dst))))
        else:
            directed_existing[pred].add((src, dst))
    return dict(by_year), dict(directed_existing), dict(undirected_existing)


def _sample_negative_quadruples(
    *,
    positives: Sequence[Tuple[int, int, int, int]],
    num_nodes: int,
    num_relations: int,
    existing_year: set[Tuple[int, int, int]],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out: list[Tuple[int, int, int, int]] = []
    if not positives or num_samples <= 0:
        return np.zeros((0, 4), dtype=np.int64)
    attempts = 0
    while len(out) < num_samples and attempts < num_samples * 20:
        attempts += 1
        u, r, v, t = positives[int(rng.integers(0, len(positives)))]
        if bool(rng.integers(0, 2)):
            v = int(rng.integers(0, num_nodes))
        else:
            r = int(rng.integers(0, num_relations))
        if u == v:
            continue
        if (u, r, v) in existing_year:
            continue
        out.append((u, r, v, t))
    if not out:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def _torch_tensor(array: np.ndarray, *, device: str, dtype=None):
    if dtype is None:
        return torch.as_tensor(array, device=device)
    return torch.as_tensor(array, device=device, dtype=dtype)


def pygt_temporal_link_prediction(
    events: Sequence[TemporalEvent],
    *,
    top_k: int = 30,
    config: Optional[PyGTemporalLinkPredConfig] = None,
) -> List[LinkPredictionRecord]:
    if not pygt_temporal_available():  # pragma: no cover
        raise PyGTemporalUnavailableError('torch-geometric-temporal is not installed')

    cfg = config or PyGTemporalLinkPredConfig()
    stats = collect_relation_prediction_stats(events)
    predicate_candidates = sorted(
        stats.predicate_recent_weight,
        key=lambda pred: (stats.predicate_recent_weight.get(pred, 0.0), pred),
        reverse=True,
    )
    if not predicate_candidates:
        predicate_candidates = ['may_relate_to']
    predicate_candidates = predicate_candidates[: max(2, int(cfg.max_relation_candidates or 8))]

    snapshots, nodes, yearly_edges, year_to_order = _make_snapshots(events, recent_window_years=int(cfg.recent_window_years or 3))
    if len(snapshots) < 2 or len(nodes) < 2:
        return []

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    predicate_to_idx = {pred: idx for idx, pred in enumerate(predicate_candidates)}
    by_year_typed, directed_existing, undirected_existing = _build_typed_targets(
        [ev for ev in events if normalize_predicate(ev.predicate) in predicate_to_idx],
        node_to_idx=node_to_idx,
        predicate_to_idx=predicate_to_idx,
        year_to_order=year_to_order,
    )
    if not by_year_typed:
        return build_semantic_heuristic_predictions(
            events,
            top_k=top_k,
            min_candidate_score=float(cfg.min_candidate_score or 0.0),
            recent_window_years=int(cfg.recent_window_years or 3),
            recency_half_life_years=float(cfg.recent_window_years or 3),
            backend='pygt_temporal_typed_fallback',
        )

    rng = np.random.default_rng(int(cfg.seed or 7))
    torch.manual_seed(int(cfg.seed or 7))
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(int(cfg.seed or 7))

    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = int(snapshots[0]['x'].shape[1])
    model = _TypedTemporalEdgeScorer(
        in_channels=in_channels,
        hidden_dim=int(cfg.hidden_dim or 48),
        relation_dim=int(cfg.relation_dim or 24),
        time_dim=int(cfg.time_dim or 12),
        k_hops=int(cfg.k_hops or 2),
        num_relations=len(predicate_to_idx),
        num_times=len(year_to_order),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr or 0.01), weight_decay=float(cfg.weight_decay or 0.0))
    criterion = nn.BCEWithLogitsLoss()

    ordered_snapshot_years = [int(snap['year'][0]) for snap in snapshots]
    train_windows = list(zip(ordered_snapshot_years[:-1], ordered_snapshot_years[1:]))
    if not train_windows:
        return []

    for _epoch in range(max(1, int(cfg.epochs or 1))):
        model.train()
        optimizer.zero_grad()
        hidden = None
        losses = []
        for src_year, target_year in train_windows:
            snap = next(s for s in snapshots if int(s['year'][0]) == src_year)
            x = _torch_tensor(snap['x'], device=device, dtype=torch.float32)
            edge_index = _torch_tensor(snap['edge_index'], device=device, dtype=torch.long)
            edge_weight = _torch_tensor(snap['edge_weight'], device=device, dtype=torch.float32)
            hidden = model.encode(x, edge_index, edge_weight=edge_weight, hidden=hidden)
            hidden = F.relu(hidden)

            positives = np.asarray(list(by_year_typed.get(target_year) or []), dtype=np.int64)
            if positives.size == 0:
                continue
            pos_quads = torch.as_tensor(positives, device=device, dtype=torch.long)
            existing_year = {(u, r, v) for (u, r, v, _t) in by_year_typed.get(target_year, [])}
            neg_count = max(1, int(round(len(positives) * max(float(cfg.negative_ratio or 1.0), 0.25))))
            negatives = _sample_negative_quadruples(
                positives=by_year_typed.get(target_year, []),
                num_nodes=len(nodes),
                num_relations=len(predicate_to_idx),
                existing_year=existing_year,
                num_samples=neg_count,
                rng=rng,
            )
            neg_quads = torch.as_tensor(negatives, device=device, dtype=torch.long) if negatives.size else torch.zeros((0, 4), device=device, dtype=torch.long)
            pos_scores = model.quadruple_scores(hidden, pos_quads)
            neg_scores = model.quadruple_scores(hidden, neg_quads)
            logits = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([
                torch.ones((pos_scores.shape[0],), device=device, dtype=torch.float32),
                torch.zeros((neg_scores.shape[0],), device=device, dtype=torch.float32),
            ], dim=0)
            losses.append(criterion(logits, labels))
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        hidden = None
        final_z = None
        for snap in snapshots:
            x = _torch_tensor(snap['x'], device=device, dtype=torch.float32)
            edge_index = _torch_tensor(snap['edge_index'], device=device, dtype=torch.long)
            edge_weight = _torch_tensor(snap['edge_weight'], device=device, dtype=torch.float32)
            hidden = model.encode(x, edge_index, edge_weight=edge_weight, hidden=hidden)
            hidden = F.relu(hidden)
            final_z = hidden
        if final_z is None:
            return []
        final_z = F.normalize(final_z, p=2.0, dim=-1)

        latest_year = ordered_snapshot_years[-1]
        next_time_idx = len(ordered_snapshot_years) - 1
        max_pred_weight = max([stats.predicate_recent_weight.get(pred, 0.0) for pred in predicate_candidates] or [1.0])
        predictions: list[LinkPredictionRecord] = []
        seen = set()
        pair_candidates = list(combinations(range(len(nodes)), 2))
        for pred in predicate_candidates:
            pred_idx = predicate_to_idx[pred]
            direction = relation_direction(pred)
            relation_prior = float(stats.predicate_recent_weight.get(pred, 0.0)) / float(max_pred_weight or 1.0)
            rel_quads: list[tuple[int, int, int, int]] = []
            rel_meta: list[tuple[str, str]] = []
            for u_idx, v_idx in pair_candidates:
                u = nodes[u_idx]
                v = nodes[v_idx]
                variants = [(u, v, u_idx, v_idx)] if direction == 'undirected' else [(u, v, u_idx, v_idx), (v, u, v_idx, u_idx)]
                for src, dst, su, sv in variants:
                    if direction == 'undirected':
                        if tuple(sorted((src, dst))) in undirected_existing.get(pred, set()):
                            continue
                    else:
                        if (src, dst) in directed_existing.get(pred, set()):
                            continue
                    rel_quads.append((su, pred_idx, sv, next_time_idx))
                    rel_meta.append((src, dst))
            if not rel_quads:
                continue
            quad_tensor = torch.as_tensor(np.asarray(rel_quads, dtype=np.int64), device=device, dtype=torch.long)
            probs = torch.sigmoid(model.quadruple_scores(final_z, quad_tensor)).detach().cpu().numpy().tolist()
            for (src, dst), prob in zip(rel_meta, probs):
                support = semantic_candidate_support(stats, source=src, predicate=pred, target=dst, half_life_years=float(cfg.recent_window_years or 3))
                score = 0.68 * float(prob) + 0.22 * float(support) + 0.10 * relation_prior
                if score < float(cfg.min_candidate_score or 0.0):
                    continue
                record = LinkPredictionRecord(
                    source=src,
                    predicate=pred,
                    target=dst,
                    score=float(score),
                    backend='pygt_temporal_typed',
                    polarity=stats.predicate_polarity.get(pred, 'unknown'),
                    direction=direction,
                    ts_pred=str(latest_year + 1),
                    relation_family=pred,
                    aux={
                        'typed_probability': float(prob),
                        'predicate_support': float(support),
                        'predicate_prior': float(relation_prior),
                        'scoring': 'typed_(u,r,v,t)',
                    },
                )
                key = record.edge_key()
                if key in seen:
                    continue
                seen.add(key)
                predictions.append(record)

    predictions.sort(key=lambda row: row.score, reverse=True)
    if not predictions:
        return build_semantic_heuristic_predictions(
            events,
            top_k=top_k,
            min_candidate_score=float(cfg.min_candidate_score or 0.0),
            recent_window_years=int(cfg.recent_window_years or 3),
            recency_half_life_years=float(cfg.recent_window_years or 3),
            backend='pygt_temporal_typed_fallback',
        )
    return predictions[: int(top_k)]
