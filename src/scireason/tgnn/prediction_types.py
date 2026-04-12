from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import math
import re
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..temporal.schemas import TemporalEvent


_PREDICATE_FAMILY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("causal_positive", ("lead", "cause", "drive", "induce", "promote", "increase", "boost", "enhance", "improve", "result_in", "trigger")),
    ("causal_negative", ("inhibit", "suppress", "reduce", "decrease", "block", "attenuate", "prevent")),
    ("association", ("associate", "correl", "relate", "link", "connected", "depend")),
    ("composition", ("contain", "consist", "part_of", "composed", "include")),
    ("comparison", ("outperform", "exceed", "compare", "better_than", "worse_than")),
    ("cooccurrence", ("cooccur", "co_occurs", "co-occurs", "cooccurs_with")),
]


def normalize_term(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_predicate(value: Any) -> str:
    text = normalize_term(value)
    return re.sub(r"\s+", "_", text)


def predicate_family(value: Any) -> str:
    pred = normalize_predicate(value)
    if not pred:
        return "unknown"
    for family, markers in _PREDICATE_FAMILY_RULES:
        if any(marker in pred for marker in markers):
            return family
    return pred


def same_predicate_family(left: Any, right: Any) -> bool:
    l = predicate_family(left)
    r = predicate_family(right)
    return bool(l and r and l == r)


def relation_direction(predicate: Any) -> str:
    pred = normalize_predicate(predicate)
    fam = predicate_family(pred)
    return "undirected" if fam == "cooccurrence" or pred == "cooccurs_with" else "directed"


@dataclass(frozen=True)
class SemanticEdgeKey:
    source: str
    predicate: str
    target: str
    direction: str = "directed"
    polarity: str | None = None
    time_bucket: str | None = None

    def normalized(self) -> "SemanticEdgeKey":
        return SemanticEdgeKey(
            source=normalize_term(self.source),
            predicate=normalize_predicate(self.predicate),
            target=normalize_term(self.target),
            direction=(self.direction or relation_direction(self.predicate)).strip().lower(),
            polarity=(str(self.polarity).strip().lower() if self.polarity else None),
            time_bucket=(str(self.time_bucket).strip() if self.time_bucket else None),
        )


@dataclass(frozen=True)
class LinkPredictionRecord:
    source: str
    predicate: str
    target: str
    score: float
    backend: str
    polarity: str | None = None
    direction: str = "directed"
    ts_pred: str | None = None
    relation_family: str | None = None
    evidence_path: list[str] = field(default_factory=list)
    aux: Dict[str, Any] = field(default_factory=dict)

    def edge_key(self, *, include_time: bool = False) -> SemanticEdgeKey:
        time_bucket = self.ts_pred if include_time else None
        return SemanticEdgeKey(
            source=self.source,
            predicate=self.predicate,
            target=self.target,
            direction=self.direction,
            polarity=self.polarity,
            time_bucket=time_bucket,
        ).normalized()

    def to_json_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source": self.source,
            "predicate": self.predicate,
            "target": self.target,
            "score": float(self.score),
            "backend": self.backend,
            "direction": self.direction,
            "relation_family": self.relation_family or predicate_family(self.predicate),
        }
        if self.polarity is not None:
            payload["polarity"] = self.polarity
        if self.ts_pred is not None:
            payload["ts_pred"] = self.ts_pred
        if self.evidence_path:
            payload["evidence_path"] = list(self.evidence_path)
        if self.aux:
            payload["aux"] = dict(self.aux)
        return payload

    def __iter__(self):
        # Backward-compatible tuple unpacking in older call sites/tests.
        yield self.source
        yield self.target
        yield float(self.score)


@dataclass
class RelationPredictionStats:
    latest_year: int
    nodes: List[str]
    predicate_pairs: Dict[str, Set[Tuple[str, str]]] = field(default_factory=dict)
    predicate_pairs_undirected: Dict[str, Set[Tuple[str, str]]] = field(default_factory=dict)
    predicate_out: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    predicate_in: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    predicate_recent_weight: Dict[str, float] = field(default_factory=dict)
    predicate_polarity: Dict[str, str] = field(default_factory=dict)


def _safe_year(ts: Optional[str]) -> int:
    try:
        return int(str(ts or "")[:4])
    except Exception:
        return 0


def _recency_decay(delta_years: int, half_life_years: float = 2.0) -> float:
    if delta_years <= 0:
        return 1.0
    hl = max(0.25, float(half_life_years))
    return math.exp(-0.6931471805599453 * float(delta_years) / hl)


def collect_relation_prediction_stats(events: Sequence[TemporalEvent]) -> RelationPredictionStats:
    latest_year = 0
    nodes = sorted({normalize_term(ev.subject) for ev in events if ev.subject} | {normalize_term(ev.object) for ev in events if ev.object})
    predicate_pairs: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)
    predicate_pairs_undirected: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)
    predicate_out: DefaultDict[str, DefaultDict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    predicate_in: DefaultDict[str, DefaultDict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    predicate_recent_weight: DefaultDict[str, float] = defaultdict(float)
    polarity_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    for ev in events:
        year = _safe_year(ev.ts_start or ev.ts_end)
        latest_year = max(latest_year, year)
        src = normalize_term(ev.subject)
        dst = normalize_term(ev.object)
        pred = normalize_predicate(ev.predicate)
        if not src or not dst or not pred or src == dst:
            continue
        predicate_pairs[pred].add((src, dst))
        predicate_pairs_undirected[pred].add(tuple(sorted((src, dst))))
        current_out = predicate_out[pred][src].get(dst, 0)
        current_in = predicate_in[pred][dst].get(src, 0)
        predicate_out[pred][src][dst] = max(current_out, year)
        predicate_in[pred][dst][src] = max(current_in, year)
        predicate_recent_weight[pred] += float(max(getattr(ev, "weight", 1.0), 1.0)) * _recency_decay(max(0, latest_year - year if latest_year else 0))
        polarity_counts[pred][str(getattr(ev, "polarity", "unknown") or "unknown")] += 1

    predicate_polarity = {
        pred: max(counts.items(), key=lambda kv: kv[1])[0] if counts else "unknown"
        for pred, counts in polarity_counts.items()
    }
    return RelationPredictionStats(
        latest_year=latest_year,
        nodes=nodes,
        predicate_pairs=dict(predicate_pairs),
        predicate_pairs_undirected=dict(predicate_pairs_undirected),
        predicate_out={pred: {node: dict(neigh) for node, neigh in rows.items()} for pred, rows in predicate_out.items()},
        predicate_in={pred: {node: dict(neigh) for node, neigh in rows.items()} for pred, rows in predicate_in.items()},
        predicate_recent_weight=dict(predicate_recent_weight),
        predicate_polarity=predicate_polarity,
    )


def semantic_candidate_support(
    stats: RelationPredictionStats,
    *,
    source: str,
    predicate: str,
    target: str,
    half_life_years: float = 2.0,
) -> float:
    src = normalize_term(source)
    dst = normalize_term(target)
    pred = normalize_predicate(predicate)
    direction = relation_direction(pred)
    latest_year = stats.latest_year or 0

    if direction == "undirected":
        left = stats.predicate_out.get(pred, {}).get(src, {})
        right = stats.predicate_out.get(pred, {}).get(dst, {})
        common = set(left).intersection(right)
        if not common:
            return 0.0
        score = 0.0
        for node in common:
            y = max(left.get(node, latest_year), right.get(node, latest_year))
            score += _recency_decay(max(0, latest_year - y), half_life_years)
        return score / float(max(1, len(common)))

    src_out = stats.predicate_out.get(pred, {}).get(src, {})
    dst_in = stats.predicate_in.get(pred, {}).get(dst, {})
    bridge = set(src_out).intersection(dst_in)
    support = 0.0
    for node in bridge:
        y = max(src_out.get(node, latest_year), dst_in.get(node, latest_year))
        support += _recency_decay(max(0, latest_year - y), half_life_years)

    src_in = stats.predicate_in.get(pred, {}).get(src, {})
    dst_out = stats.predicate_out.get(pred, {}).get(dst, {})
    alt = set(src_in).intersection(dst_out)
    for node in alt:
        y = max(src_in.get(node, latest_year), dst_out.get(node, latest_year))
        support += 0.5 * _recency_decay(max(0, latest_year - y), half_life_years)

    return support / float(max(1, len(bridge) + len(alt))) if (bridge or alt) else 0.0


def lift_pair_scores_to_semantic_predictions(
    pair_scores: Sequence[Tuple[str, str, float]],
    *,
    events: Sequence[TemporalEvent],
    backend: str,
    min_candidate_score: float,
    top_k: int,
    half_life_years: float = 2.0,
) -> List[LinkPredictionRecord]:
    stats = collect_relation_prediction_stats(events)
    if not stats.nodes:
        return []

    predicates = sorted(
        stats.predicate_recent_weight,
        key=lambda pred: (stats.predicate_recent_weight.get(pred, 0.0), pred),
        reverse=True,
    )
    if not predicates:
        predicates = ["may_relate_to"]

    max_pred_weight = max([stats.predicate_recent_weight.get(pred, 0.0) for pred in predicates] or [1.0])
    out: List[LinkPredictionRecord] = []
    seen: Set[SemanticEdgeKey] = set()

    for raw_source, raw_target, pair_score in pair_scores:
        base_u = normalize_term(raw_source)
        base_v = normalize_term(raw_target)
        if not base_u or not base_v or base_u == base_v:
            continue
        for pred in predicates:
            direction = relation_direction(pred)
            variants = [(base_u, base_v)] if direction == "undirected" else [(base_u, base_v), (base_v, base_u)]
            for source, target in variants:
                if direction == "undirected":
                    if tuple(sorted((source, target))) in stats.predicate_pairs_undirected.get(pred, set()):
                        continue
                else:
                    if (source, target) in stats.predicate_pairs.get(pred, set()):
                        continue
                support = semantic_candidate_support(stats, source=source, predicate=pred, target=target, half_life_years=half_life_years)
                pred_prior = float(stats.predicate_recent_weight.get(pred, 0.0)) / float(max_pred_weight or 1.0)
                semantic_score = 0.72 * float(pair_score) + 0.18 * float(support) + 0.10 * float(pred_prior)
                if semantic_score < float(min_candidate_score):
                    continue
                record = LinkPredictionRecord(
                    source=source,
                    predicate=pred,
                    target=target,
                    score=float(semantic_score),
                    backend=backend,
                    polarity=stats.predicate_polarity.get(pred, "unknown"),
                    direction=direction,
                    ts_pred=str((stats.latest_year + 1) if stats.latest_year else ""),
                    relation_family=predicate_family(pred),
                    aux={
                        "pair_score": float(pair_score),
                        "predicate_support": float(support),
                        "predicate_prior": float(pred_prior),
                    },
                )
                edge_key = record.edge_key()
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                out.append(record)

    out.sort(key=lambda row: row.score, reverse=True)
    return out[: int(top_k)]


def build_semantic_heuristic_predictions(
    events: Sequence[TemporalEvent],
    *,
    top_k: int,
    min_candidate_score: float,
    recent_window_years: int,
    recency_half_life_years: float,
    backend: str,
) -> List[LinkPredictionRecord]:
    stats = collect_relation_prediction_stats(events)
    if len(stats.nodes) < 2:
        return []

    scored_pairs: List[Tuple[str, str, float]] = []
    latest_year = stats.latest_year or 0
    for u, v in combinations(stats.nodes, 2):
        agg = 0.0
        for pred in stats.predicate_recent_weight:
            direction = relation_direction(pred)
            if direction == "undirected" and tuple(sorted((u, v))) in stats.predicate_pairs_undirected.get(pred, set()):
                continue
            if direction == "directed" and ((u, v) in stats.predicate_pairs.get(pred, set()) and (v, u) in stats.predicate_pairs.get(pred, set())):
                continue
            support_uv = semantic_candidate_support(stats, source=u, predicate=pred, target=v, half_life_years=recency_half_life_years)
            support_vu = semantic_candidate_support(stats, source=v, predicate=pred, target=u, half_life_years=recency_half_life_years)
            agg = max(agg, support_uv, support_vu)
        if agg <= 0.0:
            continue
        scored_pairs.append((u, v, agg))

    if not scored_pairs:
        return []

    # Add a small temporal prior favoring nodes that stayed active in the recent window.
    recent_threshold = latest_year - max(1, int(recent_window_years)) + 1 if latest_year else 0
    recent_node_counts: DefaultDict[str, int] = defaultdict(int)
    for ev in events:
        year = _safe_year(ev.ts_start or ev.ts_end)
        if recent_threshold and year and year < recent_threshold:
            continue
        recent_node_counts[normalize_term(ev.subject)] += 1
        recent_node_counts[normalize_term(ev.object)] += 1

    reweighted: List[Tuple[str, str, float]] = []
    max_recent = max(recent_node_counts.values() or [1])
    for u, v, score in scored_pairs:
        recency = (recent_node_counts.get(u, 0) + recent_node_counts.get(v, 0)) / float(2 * max_recent)
        reweighted.append((u, v, 0.85 * float(score) + 0.15 * float(recency)))

    reweighted.sort(key=lambda row: row[2], reverse=True)
    return lift_pair_scores_to_semantic_predictions(
        reweighted[: max(int(top_k) * 2, int(top_k))],
        events=events,
        backend=backend,
        min_candidate_score=min_candidate_score,
        top_k=top_k,
        half_life_years=recency_half_life_years,
    )
