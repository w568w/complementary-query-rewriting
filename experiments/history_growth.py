from __future__ import annotations

import argparse
import json
from pathlib import Path

from cqr.data.io import load_jsonl, write_json
from cqr.evaluation.metrics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.measurements)
    grouped: dict[tuple[float, str], list[float]] = {}
    for row in rows:
        key = (float(row["history_fraction"]), str(row["method"]))
        grouped.setdefault(key, []).append(float(row["score"]))
    curve = [
        {
            "history_fraction": fraction,
            "method": method,
            "queries": len(scores),
            "mean_score": mean(scores),
        }
        for (fraction, method), scores in sorted(grouped.items())
    ]
    report = {"protocol": "fixed-roster-chronological-history-growth-v1", "curve": curve}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
