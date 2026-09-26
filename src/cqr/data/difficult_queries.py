from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from cqr.evaluation.metrics import ndcg_at_k
from cqr.schemas import QueryExample


@dataclass(frozen=True)
class DifficultQueryCriterion:
    metric: Literal["ndcg", "best_relevant_rank"]
    cutoff: int
    maximum_ndcg: float | None = None
    minimum_rank: int | None = None

    def accepts(self, example: QueryExample) -> bool:
        if not example.relevant_doc_ids or not example.original_ranking_doc_ids:
            return False
        if self.metric == "ndcg":
            if self.maximum_ndcg is None:
                raise ValueError("maximum_ndcg is required for an nDCG criterion")
            return (
                ndcg_at_k(
                    example.original_ranking_doc_ids,
                    set(example.relevant_doc_ids),
                    self.cutoff,
                )
                <= self.maximum_ndcg
            )
        if self.minimum_rank is None:
            raise ValueError("minimum_rank is required for a rank criterion")
        relevant = set(example.relevant_doc_ids)
        best_rank = next(
            (
                rank
                for rank, doc_id in enumerate(example.original_ranking_doc_ids, start=1)
                if doc_id in relevant
            ),
            len(example.original_ranking_doc_ids) + 1,
        )
        return best_rank >= self.minimum_rank


def select_difficult_queries(
    examples: Iterable[QueryExample], criterion: DifficultQueryCriterion
) -> list[QueryExample]:
    selected = [example for example in examples if criterion.accepts(example)]
    if not selected:
        raise ValueError("The difficult-query criterion selected no examples")
    return selected
