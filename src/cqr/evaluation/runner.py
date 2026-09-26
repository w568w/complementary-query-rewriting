from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cqr.evaluation.metrics import hit_at_k, mean, mrr_at_k, ndcg_at_k


@dataclass(frozen=True)
class RankingRecord:
    query_id: str
    source_id: str
    method: str
    sample_index: int
    relevant_doc_ids: tuple[str, ...]
    original_ranking_doc_ids: tuple[str, ...]
    ranking_doc_ids: tuple[str, ...]
    query_generation_seconds: float | None = None


def _query_bootstrap(values: Mapping[str, float], draws: int, seed: int) -> list[float]:
    items = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(items), size=(draws, len(items)))
    return [float(value) for value in np.percentile(items[indices].mean(axis=1), [2.5, 97.5])]


def _cluster_bootstrap(
    values: Mapping[str, float],
    sources: Mapping[str, str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for query_id, source_id in sources.items():
        clusters[source_id].append(query_id)
    source_ids = sorted(clusters)
    cluster_sums = np.asarray(
        [sum(values[query_id] for query_id in clusters[source_id]) for source_id in source_ids],
        dtype=np.float64,
    )
    cluster_sizes = np.asarray(
        [len(clusters[source_id]) for source_id in source_ids], dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(source_ids), size=(draws, len(source_ids)))
    means = cluster_sums[indices].sum(axis=1) / cluster_sizes[indices].sum(axis=1)
    return {
        "clusters": len(source_ids),
        "interval": [float(value) for value in np.percentile(means, [2.5, 97.5])],
    }


def _method_summary(
    records: Sequence[RankingRecord],
    cutoffs: Sequence[int],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_query: dict[str, list[RankingRecord]] = defaultdict(list)
    for record in records:
        by_query[record.query_id].append(record)
    sources: dict[str, str] = {}
    generation_seconds: dict[str, float] = {}
    for query_id, rows in by_query.items():
        rows.sort(key=lambda row: row.sample_index)
        if [row.sample_index for row in rows] != list(range(len(rows))):
            raise ValueError(f"Non-contiguous sample indices for {query_id}")
        if len({row.source_id for row in rows}) != 1:
            raise ValueError(f"Query {query_id} maps to multiple source IDs")
        if len({row.relevant_doc_ids for row in rows}) != 1:
            raise ValueError(f"Query {query_id} has inconsistent relevance labels")
        if len({row.original_ranking_doc_ids for row in rows}) != 1:
            raise ValueError(f"Query {query_id} has inconsistent original rankings")
        sources[query_id] = rows[0].source_id
        timing = {row.query_generation_seconds for row in rows}
        if len(timing) > 1:
            raise ValueError(f"Query {query_id} has inconsistent generation timing")
        if None not in timing:
            generation_seconds[query_id] = float(next(iter(timing)))

    metrics: dict[str, Any] = {}
    for offset, cutoff in enumerate(cutoffs):
        rewritten = {}
        original = {}
        reciprocal_ranks = {}
        hits = {}
        for query_id, rows in by_query.items():
            relevant = set(rows[0].relevant_doc_ids)
            rewritten[query_id] = mean(
                ndcg_at_k(row.ranking_doc_ids, relevant, cutoff) for row in rows
            )
            original[query_id] = ndcg_at_k(
                rows[0].original_ranking_doc_ids, relevant, cutoff
            )
            reciprocal_ranks[query_id] = mean(
                mrr_at_k(row.ranking_doc_ids, relevant, cutoff) for row in rows
            )
            hits[query_id] = mean(
                hit_at_k(row.ranking_doc_ids, relevant, cutoff) for row in rows
            )
        delta = {query_id: rewritten[query_id] - original[query_id] for query_id in rewritten}
        epsilon = 1e-12
        metrics[f"at_{cutoff}"] = {
            "original_ndcg": mean(original.values()),
            "rewritten_ndcg": mean(rewritten.values()),
            "delta_ndcg": mean(delta.values()),
            "mrr": mean(reciprocal_ranks.values()),
            "hit_rate": mean(hits.values()),
            "win_rate": mean(float(value > epsilon) for value in delta.values()),
            "tie_rate": mean(float(abs(value) <= epsilon) for value in delta.values()),
            "harm_rate": mean(float(value < -epsilon) for value in delta.values()),
            "query_bootstrap_delta_95_percent_interval": _query_bootstrap(
                delta, draws, seed + offset
            ),
            "cluster_bootstrap_delta": _cluster_bootstrap(
                delta, sources, draws, seed + offset
            ),
        }
    return {
        "queries": len(by_query),
        "samples": len(records),
        "samples_per_query": sorted({len(rows) for rows in by_query.values()}),
        "metrics": metrics,
        "mean_generation_seconds_per_query": (
            mean(generation_seconds.values()) if generation_seconds else None
        ),
    }


def evaluate_rankings(
    records: Iterable[RankingRecord],
    cutoffs: Sequence[int] = (10, 25),
    draws: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    rows = list(records)
    if not rows or not cutoffs or draws < 1 or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("Invalid evaluation input")
    grouped: dict[str, list[RankingRecord]] = defaultdict(list)
    for record in rows:
        grouped[record.method].append(record)
    rosters = [
        {record.query_id for record in method_rows} for method_rows in grouped.values()
    ]
    if any(roster != rosters[0] for roster in rosters[1:]):
        raise ValueError("Methods do not cover the same query roster")
    return {
        method: _method_summary(method_rows, cutoffs, draws, seed + offset * 100)
        for offset, (method, method_rows) in enumerate(sorted(grouped.items()))
    }


def records_from_dicts(rows: Iterable[Mapping[str, object]]) -> list[RankingRecord]:
    return [
        RankingRecord(
            query_id=str(row["query_id"]),
            source_id=str(row["source_id"]),
            method=str(row["method"]),
            sample_index=int(row.get("sample_index", 0)),
            relevant_doc_ids=tuple(map(str, row["relevant_doc_ids"])),
            original_ranking_doc_ids=tuple(
                map(str, row["original_ranking_doc_ids"])
            ),
            ranking_doc_ids=tuple(map(str, row["ranking_doc_ids"])),
            query_generation_seconds=(
                float(row["query_generation_seconds"])
                if row.get("query_generation_seconds") is not None
                else None
            ),
        )
        for row in rows
    ]
