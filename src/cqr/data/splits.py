from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from cqr.schemas import QueryExample, normalize_query, stable_id


def _components(examples: list[QueryExample]) -> dict[str, str]:
    source_ids = {example.source_id for example in examples}
    parent = {source_id: source_id for source_id in source_ids}

    def find(source_id: str) -> str:
        while parent[source_id] != source_id:
            parent[source_id] = parent[parent[source_id]]
            source_id = parent[source_id]
        return source_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    query_sources: dict[str, str] = {}
    for example in examples:
        query = normalize_query(example.query)
        if query in query_sources:
            union(example.source_id, query_sources[query])
        else:
            query_sources[query] = example.source_id
    return {source_id: find(source_id) for source_id in source_ids}


def grouped_split(
    examples: Iterable[QueryExample],
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[QueryExample]]:
    rows = list(examples)
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    if min(validation_size, test_size) < 0:
        raise ValueError("Split sizes must be non-negative")
    if validation_size + test_size >= len(rows):
        raise ValueError("Held-out splits leave no training examples")

    component_by_source = _components(rows)
    groups: dict[str, list[QueryExample]] = defaultdict(list)
    for example in rows:
        groups[component_by_source[example.source_id]].append(example)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    def take_prefix(candidates: list[str], target: int) -> tuple[set[str], list[str]]:
        if target == 0:
            return set(), candidates
        cumulative = 0
        choices: list[tuple[int, int]] = []
        for index, group_id in enumerate(candidates[:-1], start=1):
            cumulative += len(groups[group_id])
            choices.append((abs(cumulative - target), index))
        if not choices:
            raise ValueError("Not enough source groups for the requested splits")
        _, cutoff = min(choices)
        return set(candidates[:cutoff]), candidates[cutoff:]

    test_groups, remaining = take_prefix(group_ids, test_size)
    validation_groups, remaining = take_prefix(remaining, validation_size)
    split_groups = {
        "train": set(remaining),
        "validation": validation_groups,
        "test": test_groups,
    }
    result = {
        split: [
            example
            for example in rows
            if component_by_source[example.source_id] in selected_groups
        ]
        for split, selected_groups in split_groups.items()
    }
    validate_splits(result)
    return result


def validate_splits(splits: dict[str, list[QueryExample]]) -> None:
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits")
    sources = {
        split: {example.source_id for example in examples}
        for split, examples in splits.items()
    }
    queries = {
        split: {normalize_query(example.query) for example in examples}
        for split, examples in splits.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if sources[left] & sources[right]:
            raise ValueError(f"Source overlap between {left} and {right}")
        if queries[left] & queries[right]:
            raise ValueError(f"Exact-query overlap between {left} and {right}")


def split_manifest(splits: dict[str, list[QueryExample]]) -> dict[str, object]:
    validate_splits(splits)
    return {
        "protocol": "source-and-exact-query-disjoint-v1",
        "splits": {
            split: {
                "examples": len(examples),
                "sources": len({example.source_id for example in examples}),
                "query_ids_sha256": stable_id(sorted(example.query_id for example in examples)),
            }
            for split, examples in splits.items()
        },
    }
