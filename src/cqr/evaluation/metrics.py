from __future__ import annotations

import math
from typing import Iterable, Sequence


def deduplicate_ranking(ranking: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(map(str, ranking)))


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], cutoff: int) -> float:
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    if not relevant:
        return 0.0
    unique = deduplicate_ranking(ranking)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(unique[:cutoff], start=1)
        if str(doc_id) in relevant
    )
    ideal_hits = min(cutoff, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def mrr_at_k(ranking: Sequence[str], relevant: set[str], cutoff: int) -> float:
    for rank, doc_id in enumerate(deduplicate_ranking(ranking)[:cutoff], start=1):
        if str(doc_id) in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranking: Sequence[str], relevant: set[str], cutoff: int) -> float:
    return float(
        any(str(doc_id) in relevant for doc_id in deduplicate_ranking(ranking)[:cutoff])
    )


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("Cannot average an empty sequence")
    return sum(items) / len(items)
