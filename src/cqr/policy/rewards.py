from __future__ import annotations

from typing import Sequence

from cqr.evaluation.metrics import ndcg_at_k
from cqr.schemas import normalize_query


def rewritten_query(completion: str) -> str:
    text = completion.replace("<|im_end|>", "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    for prefix in ("Revised query:", "Rewritten query:"):
        if text.casefold().startswith(prefix.casefold()):
            text = text[len(prefix) :].strip()
            break
    return text


def retrieval_reward(
    original_ranking: Sequence[str],
    rewritten_ranking: Sequence[str],
    relevant_doc_ids: Sequence[str],
    cutoff: int,
) -> float:
    relevant = set(map(str, relevant_doc_ids))
    return ndcg_at_k(rewritten_ranking, relevant, cutoff) - ndcg_at_k(
        original_ranking, relevant, cutoff
    )


def format_reward(original_query: str, completion: str) -> float:
    rewrite = rewritten_query(completion)
    invalid = (
        not rewrite
        or "\n" in rewrite
        or normalize_query(rewrite) == normalize_query(original_query)
        or "<think>" in completion
        or "</think>" in completion
    )
    return -1.0 if invalid else 0.0


def length_reward(token_count: int, soft_limit: int = 64, hard_limit: int = 128) -> float:
    if token_count < 0 or soft_limit < 1 or hard_limit <= soft_limit:
        raise ValueError("Invalid token budget")
    return -min(1.0, max(0.0, (token_count - soft_limit) / (hard_limit - soft_limit)))


def composite_reward(
    retrieval: float,
    format_constraint: float,
    length_penalty: float,
    format_weight: float = 0.1,
    length_weight: float = 0.1,
) -> float:
    return retrieval + format_weight * format_constraint + length_weight * length_penalty
