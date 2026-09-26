from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cqr.memory.bank import MemoryBank
from cqr.policy.generate import Rewriter
from cqr.retrieval.hybrid import HybridRetriever
from cqr.retrieval.fusion import fuse_paths, memory_weight


@dataclass(frozen=True)
class InferenceResult:
    query: str
    policy_rewrites: tuple[str, ...]
    memory_rewrites: tuple[str, ...]
    policy_rankings: tuple[tuple[str, ...], ...]
    memory_rankings: tuple[tuple[str, ...], ...]
    final_rankings: tuple[tuple[str, ...], ...]
    memory_entry_ids: tuple[str, ...]


class ComplementaryInference:
    def __init__(
        self,
        policy_rewriter: Rewriter,
        memory_rewriter: Rewriter,
        retriever: HybridRetriever,
        memory_bank: MemoryBank,
        memory_top_k: int = 3,
        retrieval_top_k: int = 25,
        rank_constant: float = 1.0,
    ) -> None:
        self.policy_rewriter = policy_rewriter
        self.memory_rewriter = memory_rewriter
        self.retriever = retriever
        self.memory_bank = memory_bank
        self.memory_top_k = memory_top_k
        self.retrieval_top_k = retrieval_top_k
        self.rank_constant = rank_constant

    def run(
        self,
        query: str,
        source_id: str | None = None,
        mode: Literal["rl", "full"] = "full",
        samples: int = 1,
    ) -> InferenceResult:
        policy_rewrites = self.policy_rewriter.rewrite(query, samples=samples)
        policy_components = self.retriever.retrieve_components(
            policy_rewrites, top_k=self.retrieval_top_k
        )
        policy_rankings = [parts["hybrid"] for parts in policy_components]
        if mode == "rl":
            return InferenceResult(
                query=query,
                policy_rewrites=tuple(policy_rewrites),
                memory_rewrites=(),
                policy_rankings=tuple(tuple(ranking) for ranking in policy_rankings),
                memory_rankings=(),
                final_rankings=tuple(tuple(ranking) for ranking in policy_rankings),
                memory_entry_ids=(),
            )

        matches = self.memory_bank.retrieve_scored(query, self.memory_top_k, source_id=source_id)
        memory = [entry for entry, _ in matches]
        similarity = matches[0][1] if matches else 0.0
        memory_rewrites = self.memory_rewriter.rewrite(query, memory=memory, samples=samples)
        memory_components = self.retriever.retrieve_components(
            memory_rewrites, top_k=self.retrieval_top_k
        )
        memory_rankings = [parts["hybrid"] for parts in memory_components]
        final_rankings = [
            fuse_paths(
                left["hybrid"],
                right["hybrid"],
                memory_weight(
                    similarity, left["dense"], left["sparse"], right["dense"], right["sparse"]
                ),
                self.rank_constant,
                self.retrieval_top_k,
            )
            for left, right in zip(policy_components, memory_components, strict=True)
        ]
        return InferenceResult(
            query=query,
            policy_rewrites=tuple(policy_rewrites),
            memory_rewrites=tuple(memory_rewrites),
            policy_rankings=tuple(tuple(ranking) for ranking in policy_rankings),
            memory_rankings=tuple(tuple(ranking) for ranking in memory_rankings),
            final_rankings=tuple(tuple(ranking) for ranking in final_rankings),
            memory_entry_ids=tuple(entry.entry_id for entry in memory),
        )
