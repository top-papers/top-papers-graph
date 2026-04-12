from __future__ import annotations

"""Temporal link prediction with an optional real PyG TGN memory backend.

Design goals:
- keep the repo usable in the base installation without heavy temporal-GNN dependencies;
- expose a stable TGNN/TGN-oriented API for hypothesis generation;
- prefer a real PyTorch Geometric TGN memory model when available/configured;
- fall back to a deterministic heuristic scorer when PyG is absent.
- preserve temporal semantics at the (subject, predicate, object, time) level.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

from ..config import settings
from ..temporal.schemas import TemporalEvent
from .prediction_types import (
    LinkPredictionRecord,
    build_semantic_heuristic_predictions,
    collect_relation_prediction_stats,
    lift_pair_scores_to_semantic_predictions,
    normalize_predicate,
)

try:  # pragma: no cover
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    IdentityMessage = None  # type: ignore[assignment]
    LastAggregator = None  # type: ignore[assignment]
    TGNMemory = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TGNLinkPredConfig:
    recent_window_years: int = 3
    recency_half_life_years: float = 2.0
    pair_repeat_weight: float = 0.35
    common_neighbor_weight: float = 0.40
    node_memory_weight: float = 0.25
    min_candidate_score: float = 0.05
    seed: int = 7
    memory_dim: int = 64
    time_dim: int = 16
    relation_dim: int = 8
    backend: str = "auto"  # auto|heuristic|pyg


def pyg_tgn_available() -> bool:
    return all(x is not None for x in (torch, F, IdentityMessage, LastAggregator, TGNMemory))


def tgnn_available() -> bool:
    """A TGNN-style predictor is always available thanks to the heuristic fallback."""
    return True


def _safe_year(ts: Optional[str]) -> int:
    try:
        return int(str(ts)[:4])
    except Exception:
        return 0


def _safe_time_number(ts: Optional[str], fallback: int) -> int:
    try:
        s = str(ts or "")
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return int(s[:4]) * 372 + int(s[5:7]) * 31 + int(s[8:10])
        if len(s) >= 7 and s[4] == "-":
            return int(s[:4]) * 12 + int(s[5:7])
        if len(s) >= 4:
            return int(s[:4])
    except Exception:
        pass
    return int(fallback)


def _hash_relation_features(predicate: str, dim: int) -> list[float]:
    pred = normalize_predicate(predicate)
    dim = max(1, int(dim or 1))
    if not pred:
        return [0.0] * dim
    buckets = [0.0] * dim
    for idx, ch in enumerate(pred.encode("utf-8", errors="ignore")):
        buckets[(idx + ch) % dim] += 1.0
    total = sum(buckets) or 1.0
    return [value / total for value in buckets]


def _heuristic_tgn_prediction(
    events: Sequence[TemporalEvent],
    *,
    top_k: int,
    config: TGNLinkPredConfig,
) -> List[LinkPredictionRecord]:
    return build_semantic_heuristic_predictions(
        events,
        top_k=top_k,
        min_candidate_score=float(config.min_candidate_score),
        recent_window_years=int(config.recent_window_years),
        recency_half_life_years=float(config.recency_half_life_years),
        backend="heuristic",
    )


def _pyg_tgn_memory_prediction(
    events: Sequence[TemporalEvent],
    *,
    top_k: int,
    config: TGNLinkPredConfig,
) -> List[LinkPredictionRecord]:
    if not pyg_tgn_available():  # pragma: no cover
        raise RuntimeError("PyG TGN backend is not available")

    ordered = sorted(list(events), key=lambda e: e.sort_key())
    if len(ordered) < 2:
        return []

    nodes = sorted({str(ev.subject) for ev in ordered if ev.subject} | {str(ev.object) for ev in ordered if ev.object})
    if len(nodes) < 2:
        return []
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    memory_dim = int(config.memory_dim or getattr(settings, 'hyp_tgnn_memory_dim', 64) or 64)
    time_dim = int(config.time_dim or getattr(settings, 'hyp_tgnn_time_dim', 16) or 16)
    relation_dim = max(2, int(config.relation_dim or 8))
    raw_msg_dim = 2 + relation_dim + 2

    device = torch.device('cpu')
    memory = TGNMemory(
        num_nodes=len(nodes),
        raw_msg_dim=raw_msg_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    memory.reset_state()
    memory.train()

    seen_pairs = set()
    last_t = 0
    latest_year = 0
    for seq, ev in enumerate(ordered, start=1):
        u = str(ev.subject)
        v = str(ev.object)
        if not u or not v or u == v:
            continue
        src = torch.tensor([node_to_idx[u]], dtype=torch.long, device=device)
        dst = torch.tensor([node_to_idx[v]], dtype=torch.long, device=device)
        ts_num = _safe_time_number(ev.ts_start or ev.ts_end, seq)
        last_t = max(last_t, ts_num)
        latest_year = max(latest_year, _safe_year(ev.ts_start or ev.ts_end))
        t = torch.tensor([ts_num], dtype=torch.long, device=device)
        relation_features = _hash_relation_features(str(ev.predicate or ""), relation_dim)
        polarity = str(getattr(ev, 'polarity', 'unknown') or 'unknown').lower()
        polarity_value = -1.0 if polarity == 'contradicts' else 1.0 if polarity == 'supports' else 0.0
        granularity_value = {
            'year': 0.25,
            'month': 0.5,
            'interval': 0.75,
            'day': 1.0,
        }.get(str(getattr(ev, 'granularity', 'year') or 'year').lower(), 0.25)
        raw_msg = torch.tensor(
            [[float(ev.confidence), float(max(ev.weight, 1.0)), *relation_features, polarity_value, granularity_value]],
            dtype=torch.float32,
            device=device,
        )
        memory.update_state(src, dst, t, raw_msg)
        seen_pairs.add(tuple(sorted((u, v))))

    memory.eval()  # flush message store into memory
    n_id = torch.arange(len(nodes), device=device)
    z, last_update = memory(n_id)
    z = F.normalize(z, p=2.0, dim=-1)

    sim = torch.sigmoid(z @ z.t())
    sim.fill_diagonal_(0.0)

    for a, b in seen_pairs:
        ia, ib = node_to_idx[a], node_to_idx[b]
        sim[ia, ib] = 0.0
        sim[ib, ia] = 0.0

    if last_t > 0:
        age = (float(last_t) - last_update.float()).clamp(min=0.0)
        denom = max(1.0, float(config.recent_window_years))
        recency = torch.exp(-age / denom)
        recency_pair = (recency.view(-1, 1) + recency.view(1, -1)) / 2.0
        sim = 0.85 * sim + 0.15 * recency_pair

    tri = torch.triu_indices(len(nodes), len(nodes), offset=1)
    tri_scores = sim[tri[0], tri[1]]
    if tri_scores.numel() == 0:
        return []

    pair_cap = min(max(int(top_k) * 4, int(top_k)), int(tri_scores.numel()))
    vals, best = torch.topk(tri_scores, k=pair_cap)

    pair_scores: List[Tuple[str, str, float]] = []
    for score, idx_flat in zip(vals.tolist(), best.tolist()):
        iu = int(tri[0][idx_flat])
        iv = int(tri[1][idx_flat])
        if score < float(config.min_candidate_score) * 0.5:
            continue
        pair_scores.append((nodes[iu], nodes[iv], float(score)))

    if not pair_scores:
        return []

    records = lift_pair_scores_to_semantic_predictions(
        pair_scores,
        events=ordered,
        backend="pyg_tgn",
        min_candidate_score=float(config.min_candidate_score),
        top_k=top_k,
        half_life_years=float(config.recency_half_life_years),
    )

    # If semantic lifting produced nothing, fall back to a relation-aware heuristic directly.
    if not records:
        return build_semantic_heuristic_predictions(
            ordered,
            top_k=top_k,
            min_candidate_score=float(config.min_candidate_score),
            recent_window_years=int(config.recent_window_years),
            recency_half_life_years=float(config.recency_half_life_years),
            backend="pyg_tgn_fallback",
        )

    return records


def tgn_link_prediction(
    events: Sequence[TemporalEvent],
    *,
    top_k: int = 30,
    config: Optional[TGNLinkPredConfig] = None,
) -> List[LinkPredictionRecord]:
    """Predict future semantic links from a chronological stream of temporal KG events.

    Backend policy:
    - `settings.hyp_tgnn_backend=pyg`  -> require/use PyG TGN memory backend;
    - `settings.hyp_tgnn_backend=auto` -> prefer PyG TGN, fallback to heuristic;
    - `settings.hyp_tgnn_backend=heuristic` -> deterministic semantic fallback only.
    """

    cfg = config or TGNLinkPredConfig(
        memory_dim=int(getattr(settings, 'hyp_tgnn_memory_dim', 64) or 64),
        time_dim=int(getattr(settings, 'hyp_tgnn_time_dim', 16) or 16),
        relation_dim=int(getattr(settings, 'hyp_tgnn_relation_dim', 8) or 8),
        backend=str(getattr(settings, 'hyp_tgnn_backend', 'auto') or 'auto'),
    )
    backend = str(cfg.backend or getattr(settings, 'hyp_tgnn_backend', 'auto') or 'auto').lower()

    if backend in {'auto', 'pyg'} and pyg_tgn_available():
        try:
            return _pyg_tgn_memory_prediction(events, top_k=top_k, config=cfg)
        except Exception:
            if backend == 'pyg':
                raise

    return _heuristic_tgn_prediction(events, top_k=top_k, config=cfg)
