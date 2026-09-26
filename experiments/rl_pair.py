"""Same-budget cyclic RL+RL control from existing per-sample retrievals."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from cqr.data.io import load_jsonl, write_jsonl
from cqr.retrieval.fusion import fuse_paths, memory_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    groups = defaultdict(dict)
    for row in load_jsonl(args.rankings):
        sample = int(row["sample_index"])
        if sample in groups[row["query_id"]]:
            raise ValueError("Duplicate query/sample key")
        groups[row["query_id"]][sample] = row
    output = []
    for query in sorted(groups):
        rows = groups[query]
        if set(rows) != set(range(8)):
            raise ValueError("The published RL+RL control requires eight samples per query")
        for i in range(8):
            a, b = rows[i], rows[(i + 1) % 8]
            left, right = a["policy_components"], b["policy_components"]
            alpha = memory_weight(
                0.5, left["dense"], left["sparse"], right["dense"], right["sparse"]
            )
            ranking = fuse_paths(a["ranking_doc_ids"], b["ranking_doc_ids"], alpha)
            output.append(
                {
                    **a,
                    "method": "rl_rl_corroboration",
                    "ranking_doc_ids": ranking,
                    "partner_sample_index": (i + 1) % 8,
                    "memory_weight": alpha,
                    "query_generation_seconds": float(a["query_generation_seconds"])
                    + float(b["query_generation_seconds"]),
                }
            )
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
