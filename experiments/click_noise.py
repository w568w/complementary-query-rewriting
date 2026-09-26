from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cqr.data.io import load_jsonl, write_json
from cqr.evaluation.bootstrap import clustered_interval
from cqr.evaluation.metrics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-effects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    grouped: dict[tuple[str, int], list[tuple[str, float]]] = defaultdict(list)
    for row in load_jsonl(args.paired_effects):
        key = (str(row["noise_type"]), int(row["noise_percent"]))
        grouped[key].append((str(row["cluster_id"]), float(row["effect"])))
    conditions = []
    for offset, ((noise_type, noise_percent), effects) in enumerate(sorted(grouped.items())):
        conditions.append(
            {
                "noise_type": noise_type,
                "noise_percent": noise_percent,
                "mean_effect": mean(effect for _, effect in effects),
                "clustered_95_percent_interval": list(
                    clustered_interval(effects, args.draws, args.seed + offset)
                ),
            }
        )
    report = {"protocol": "paired-click-noise-robustness-v1", "conditions": conditions}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
