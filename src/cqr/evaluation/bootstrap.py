from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable, Iterable


def percentile_interval(
    values: Iterable[float],
    draws: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    samples = list(values)
    if not samples or draws < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap input")
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(samples) for _ in samples) / len(samples) for _ in range(draws)
    )
    tail = (1.0 - confidence) / 2.0
    lower = means[int(tail * (draws - 1))]
    upper = means[int((1.0 - tail) * (draws - 1))]
    return lower, upper


def clustered_interval(
    effects: Iterable[tuple[Hashable, float]],
    draws: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    clusters: dict[Hashable, list[float]] = defaultdict(list)
    for cluster_id, effect in effects:
        clusters[cluster_id].append(float(effect))
    if not clusters:
        raise ValueError("Cannot bootstrap an empty set of clusters")
    cluster_ids = list(clusters)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [effect for cluster_id in sampled for effect in clusters[cluster_id]]
        means.append(sum(values) / len(values))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return means[int(tail * (draws - 1))], means[int((1.0 - tail) * (draws - 1))]
