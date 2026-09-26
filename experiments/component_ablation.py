"""Query-averaged nDCG and paired source-bootstrap gaps to Full."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cqr.data.io import load_jsonl, write_json
from cqr.evaluation.metrics import mean, ndcg_at_k


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--rankings", type=Path)
    inputs.add_argument("--query-scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="fusion_memory")
    parser.add_argument("--cutoff", type=int, default=25)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.draws < 1 or args.cutoff < 1:
        raise ValueError("draws and cutoff must be positive")
    samples = defaultdict(lambda: defaultdict(dict))
    sources = {}
    for row in load_jsonl(args.rankings or args.query_scores):
        method, query = str(row["method"]), str(row["query_id"])
        source = str(row["source_id"])
        if query in sources and sources[query] != source:
            raise ValueError("Query/source mapping differs across conditions")
        sources[query] = source
        sample = int(row["sample_index"]) if args.rankings else 0
        if sample in samples[method][query]:
            raise ValueError("Duplicate method/query/sample record")
        score = (
            ndcg_at_k(row["ranking_doc_ids"], set(row["relevant_doc_ids"]), args.cutoff)
            if args.rankings
            else float(row["ndcg"])
        )
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Expected finite nDCG on the 0-1 scale")
        samples[method][query][sample] = score
    if args.reference not in samples:
        raise ValueError("Reference method is missing")
    ids = sorted(samples[args.reference])
    if not ids or any(set(rows) != set(ids) for rows in samples.values()):
        raise ValueError("All conditions must cover exactly the same queries")
    if args.rankings:
        expected = set(samples[args.reference][ids[0]])
        if any(set(values) != expected for rows in samples.values() for values in rows.values()):
            raise ValueError("Sample indices must match across all queries and methods")
    scores = {
        method: np.array([mean(rows[q].values()) for q in ids]) * 100
        for method, rows in samples.items()
    }
    unique, inverse = np.unique([sources[q] for q in ids], return_inverse=True)
    counts = np.bincount(inverse)
    totals = {
        method: np.bincount(inverse, weights=values - scores[args.reference])
        for method, values in scores.items()
    }
    rng = np.random.default_rng(args.seed)
    distributions = {method: [] for method in scores}
    # Batching bounds memory while preserving the frozen NumPy draw sequence.
    for start in range(0, args.draws, 1000):
        draws = rng.integers(len(unique), size=(min(1000, args.draws - start), len(unique)))
        denominator = counts[draws].sum(axis=1)
        for method, sums in totals.items():
            distributions[method].extend(sums[draws].sum(axis=1) / denominator)
    report = {
        "queries": len(ids),
        "source_groups": len(unique),
        "draws": args.draws,
        "seed": args.seed,
        "reference": args.reference,
        "scale": "nDCG times 100",
        "direction": "configuration minus reference",
        "interval": "paired source-cluster percentile bootstrap, unadjusted",
        "results": {
            method: {
                "mean": float(values.mean()),
                "delta": float((values - scores[args.reference]).mean()),
                "ci95": np.quantile(distributions[method], [0.025, 0.975]).tolist(),
            }
            for method, values in scores.items()
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
