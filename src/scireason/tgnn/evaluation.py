from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def hits_at_k(predictions: Sequence[tuple[str, str, float]], gold_pairs: Iterable[tuple[str, str]], *, k: int = 10) -> float:
    preds = {(min(u, v), max(u, v)) for u, v, _ in list(predictions)[:k]}
    gold = {(min(u, v), max(u, v)) for u, v in gold_pairs}
    if not gold:
        return 0.0
    return sum(1 for pair in gold if pair in preds) / float(len(gold))


def mean_reciprocal_rank(predictions: Sequence[tuple[str, str, float]], gold_pairs: Iterable[tuple[str, str]]) -> float:
    gold = {(min(u, v), max(u, v)) for u, v in gold_pairs}
    if not gold:
        return 0.0
    rr = []
    for rank, (u, v, _) in enumerate(predictions, start=1):
        if (min(u, v), max(u, v)) in gold:
            rr.append(1.0 / float(rank))
    return sum(rr) / float(len(gold)) if rr else 0.0
