"""Parameter-free complementary fusion."""

from __future__ import annotations

import math
from typing import Sequence


def rank_agreement(dense: Sequence[str], sparse: Sequence[str]) -> float:
    if len(dense) != len(set(dense)) or len(sparse) != len(set(sparse)):
        raise ValueError("Component rankings must contain unique document IDs")
    a = {str(d): 1 / (i + 1) for i, d in enumerate(dense)}
    b = {str(d): 1 / (i + 1) for i, d in enumerate(sparse)}
    documents = sorted(a.keys() | b.keys())
    denominator = math.fsum(max(a.get(d, 0.0), b.get(d, 0.0)) for d in documents)
    return (
        math.fsum(min(a.get(d, 0.0), b.get(d, 0.0)) for d in documents) / denominator
        if denominator
        else 0.0
    )


def memory_weight(
    similarity: float,
    rl_dense: Sequence[str],
    rl_sparse: Sequence[str],
    memory_dense: Sequence[str],
    memory_sparse: Sequence[str],
) -> float:
    if not math.isfinite(similarity):
        raise ValueError("Similarity must be finite")
    prior = min(1.0, max(0.0, similarity))
    left = (1 - prior) * rank_agreement(rl_dense, rl_sparse)
    right = prior * rank_agreement(memory_dense, memory_sparse)
    return right / (left + right) if left + right > 0 else prior


def fuse_paths(
    policy: Sequence[str],
    memory: Sequence[str],
    alpha: float,
    rank_constant: float = 1.0,
    top_k: int = 25,
) -> list[str]:
    if not 0 <= alpha <= 1 or not math.isfinite(rank_constant) or rank_constant <= 0:
        raise ValueError("Invalid fusion weight or rank constant")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in ((policy, 1 - alpha), (memory, alpha)):
        if len(ranking) != len(set(ranking)):
            raise ValueError("Path rankings must contain unique document IDs")
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + weight / (rank_constant + rank)
            best_rank[doc] = min(best_rank.get(doc, rank), rank)
    return sorted(scores, key=lambda doc: (-scores[doc], best_rank[doc], str(doc)))[:top_k]
