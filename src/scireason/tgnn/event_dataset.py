from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from ..temporal.schemas import TemporalEvent
from ..temporal.temporal_kg_builder import PaperRecord, TemporalKnowledgeGraph


def _safe_year(value: object) -> int | None:
    try:
        if value is None:
            return None
        year = int(str(value)[:4])
        if 1800 <= year <= 2100:
            return year
    except Exception:
        return None
    return None


def build_event_stream(
    kg: TemporalKnowledgeGraph,
    *,
    papers: Sequence[PaperRecord] | None = None,
    default_predicate: str = "may_relate_to",
) -> List[TemporalEvent]:
    """Convert an aggregated temporal KG into a chronological event stream.

    We intentionally keep one event per (edge, year) bucket. That preserves temporal order and
    remains lightweight enough for classroom-sized corpora.
    """

    paper_years: Dict[str, int] = {}
    if papers is not None:
        paper_years = {p.paper_id: int(p.year) for p in papers if p.year is not None}

    events: List[TemporalEvent] = []
    seq = 0
    for edge in kg.edges:
        if edge.yearly_count:
            yearly_pairs = sorted(edge.yearly_count.items())
        else:
            fallback_year = None
            for pid in sorted(edge.papers):
                fallback_year = paper_years.get(pid)
                if fallback_year is not None:
                    break
            yearly_pairs = [] if fallback_year is None else [(fallback_year, max(1, int(edge.total_count or 1)))]

        for year, count in yearly_pairs:
            weight = float(count or 1)
            evidence_quote = None
            if edge.evidence_quotes:
                evidence_quote = str(edge.evidence_quotes[0].get("quote") or "") or None
            paper_id = None
            if edge.papers:
                paper_id = sorted(edge.papers)[0]
            for _ in range(max(1, int(count or 1))):
                seq += 1
                ev = TemporalEvent(
                    event_id=f"ev_{seq:08d}",
                    paper_id=str(paper_id or f"synthetic:{edge.source}:{edge.target}:{year}"),
                    subject=edge.source,
                    predicate=edge.predicate or default_predicate,
                    object=edge.target,
                    ts_start=str(year),
                    ts_end=str(year),
                    granularity="year",
                    confidence=float(edge.mean_confidence or edge.features.get("mean_conf", 0.6) or 0.6),
                    polarity=max(edge.polarity_counts.items(), key=lambda kv: kv[1])[0] if edge.polarity_counts else "unknown",
                    weight=weight,
                    evidence_quote=evidence_quote,
                )
                events.append(ev)

    events.sort(key=lambda e: e.sort_key())
    return events


def chronological_split(
    events: Sequence[TemporalEvent],
    *,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> Tuple[List[TemporalEvent], List[TemporalEvent], List[TemporalEvent]]:
    """Chronological split required for temporal prediction tasks."""

    ordered = sorted(list(events), key=lambda e: e.sort_key())
    n = len(ordered)
    if n == 0:
        return [], [], []

    train_end = max(1, int(n * train_ratio))
    valid_end = max(train_end, int(n * (train_ratio + valid_ratio)))

    train = ordered[:train_end]
    valid = ordered[train_end:valid_end]
    test = ordered[valid_end:]

    for seq in (train, valid, test):
        split = "train" if seq is train else "valid" if seq is valid else "test"
        for ev in seq:
            ev.split = split  # pydantic model is mutable by default

    return train, valid, test


def event_stats(events: Sequence[TemporalEvent]) -> dict:
    pair_counts = defaultdict(int)
    nodes = set()
    for ev in events:
        pair_counts[ev.pair_key()] += 1
        nodes.add(ev.subject)
        nodes.add(ev.object)
    return {
        "n_events": len(events),
        "n_nodes": len(nodes),
        "n_pairs": len(pair_counts),
    }
