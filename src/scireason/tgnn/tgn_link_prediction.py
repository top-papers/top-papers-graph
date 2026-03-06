from __future__ import annotations

"""Lightweight temporal link prediction with a TGNN/TGN-oriented interface.

The goal here is pragmatic:
- prefer event-stream reasoning over static GraphSAGE-style link prediction
- remain usable in the base installation without heavy temporal-GNN dependencies
- expose a stable API that can later be swapped for a fuller TGN implementation

The current scorer uses recency-aware node memory, temporal common-neighbor signals, and
pair recurrence. This is not a full research-grade TGN implementation, but it follows the same
continuous-time intuition: predictions are derived from the ordered stream of timestamped events.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import exp
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from ..temporal.schemas import TemporalEvent


@dataclass(frozen=True)
class TGNLinkPredConfig:
    recent_window_years: int = 3
    recency_half_life_years: float = 2.0
    pair_repeat_weight: float = 0.35
    common_neighbor_weight: float = 0.40
    node_memory_weight: float = 0.25
    min_candidate_score: float = 0.05
    seed: int = 7


def tgnn_available() -> bool:
    """A lightweight TGNN-style predictor is always available in base installation."""
    return True


def _safe_year(ts: Optional[str]) -> int:
    try:
        return int(str(ts)[:4])
    except Exception:
        return 0


def _decay(delta_years: int, half_life_years: float) -> float:
    if delta_years <= 0:
        return 1.0
    hl = max(0.1, float(half_life_years))
    return exp(-0.6931471805599453 * float(delta_years) / hl)


def tgn_link_prediction(
    events: Sequence[TemporalEvent],
    *,
    top_k: int = 30,
    config: Optional[TGNLinkPredConfig] = None,
) -> List[Tuple[str, str, float]]:
    """Predict future links from a chronological stream of temporal KG events."""

    cfg = config or TGNLinkPredConfig()
    ordered = sorted(list(events), key=lambda e: e.sort_key())
    if len(ordered) < 2:
        return []

    # --- temporal memories ---
    pair_last_year: Dict[Tuple[str, str], int] = {}
    pair_count: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    neighbors: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
    node_strength: DefaultDict[str, float] = defaultdict(float)

    current_year = 0
    for ev in ordered:
        year = _safe_year(ev.ts_start)
        current_year = max(current_year, year)
        u = str(ev.subject)
        v = str(ev.object)
        if not u or not v or u == v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        pair_last_year[(a, b)] = year
        pair_count[(a, b)] += 1
        neighbors[u][v] = year
        neighbors[v][u] = year
        strength = float(max(ev.weight, 1.0)) * float(ev.confidence)
        node_strength[u] += strength
        node_strength[v] += strength

    nodes = sorted({ev.subject for ev in ordered} | {ev.object for ev in ordered})
    scored: List[Tuple[str, str, float]] = []

    for u, v in combinations(nodes, 2):
        if u == v:
            continue
        pair = (u, v) if u <= v else (v, u)
        if pair in pair_count:
            continue

        # recency-aware common neighbors
        common = set(neighbors.get(u, {})).intersection(neighbors.get(v, {}))
        cn_score = 0.0
        for n in common:
            y1 = neighbors[u].get(n, current_year)
            y2 = neighbors[v].get(n, current_year)
            age = current_year - max(y1, y2)
            cn_score += _decay(age, cfg.recency_half_life_years)

        # node memory = how active nodes have been recently
        u_recent = 0.0
        for _, y in neighbors.get(u, {}).items():
            u_recent += _decay(current_year - y, cfg.recency_half_life_years)
        v_recent = 0.0
        for _, y in neighbors.get(v, {}).items():
            v_recent += _decay(current_year - y, cfg.recency_half_life_years)
        node_memory = (u_recent + v_recent) / 2.0

        # pair recurrence via two-hop motifs: if both nodes repeatedly appear in same recent window,
        # increase the chance that a direct edge emerges.
        recent_threshold = current_year - max(1, int(cfg.recent_window_years)) + 1
        repeat_score = 0.0
        for y in neighbors.get(u, {}).values():
            if y >= recent_threshold:
                repeat_score += 1.0
        for y in neighbors.get(v, {}).values():
            if y >= recent_threshold:
                repeat_score += 1.0
        repeat_score /= 2.0

        raw = (
            cfg.common_neighbor_weight * cn_score
            + cfg.node_memory_weight * node_memory
            + cfg.pair_repeat_weight * repeat_score
        )
        norm = 1.0 + float(len(common)) + node_strength.get(u, 0.0) + node_strength.get(v, 0.0)
        score = raw / norm
        if score >= float(cfg.min_candidate_score):
            scored.append((u, v, float(score)))

    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[: int(top_k)]
