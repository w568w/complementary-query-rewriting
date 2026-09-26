from __future__ import annotations

import argparse
import json
from pathlib import Path

from cqr.data.io import load_jsonl, write_json
from cqr.evaluation.runner import evaluate_rankings, records_from_dicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=(10, 25))
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = records_from_dicts(load_jsonl(args.rankings))
    report = {
        "protocol": "fixed-roster-main-effectiveness-v1",
        "cutoffs": args.cutoffs,
        "methods": evaluate_rankings(records, args.cutoffs, args.draws, args.seed),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
