from __future__ import annotations

import argparse
import json
from pathlib import Path

from cqr.data.io import write_json, write_jsonl
from cqr.manifests import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("primary", "base", "fusion", "intervals"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {
        name: json.loads(getattr(args, name).read_text())
        for name in ("primary", "base", "fusion", "intervals")
    }
    provenance = {name: file_sha256(getattr(args, name)) for name in inputs}
    args.output.mkdir(parents=True)
    for dataset, expected in inputs["intervals"]["datasets"].items():
        primary = inputs["primary"]["datasets"][dataset]["per_query"]
        base = inputs["base"]["datasets"][dataset]["per_query"]
        fusion = inputs["fusion"]["datasets"][dataset]["per_query"]
        if primary["ids"] != base["ids"] or primary["ids"] != fusion["ids"]:
            raise ValueError("Input query rosters differ")
        if list(map(str, primary["sources"])) != list(map(str, base["sources"])) or list(
            map(str, primary["sources"])
        ) != list(map(str, fusion["sources"])):
            raise ValueError("Input source rosters differ")
        records = []
        for method, summary in expected.items():
            source = (
                base
                if method in ("base", "base_memory")
                else fusion
                if method == "equal"
                else primary
            )
            scores = source["scores"][method]
            if abs(sum(scores) / len(scores) - summary["mean"]) > 1e-9:
                raise ValueError("Score mean differs from frozen table")
            for query, group, score in zip(primary["ids"], primary["sources"], scores, strict=True):
                records.append(
                    {
                        "query_id": str(query),
                        "source_id": str(group),
                        "method": method,
                        "ndcg": score / 100,
                    }
                )
        write_jsonl(args.output / f"{dataset}_query_scores.jsonl", records)
    write_json(args.output / "intervals.json", inputs["intervals"])
    write_json(
        args.output / "manifest.json",
        {
            "metric": "binary nDCG@25",
            "query_score_scale": "0-1",
            "interval_scale": "0-100",
            "samples_per_path": 8,
            "source_sha256": provenance,
            "output_sha256": {
                path.name: file_sha256(path) for path in sorted(args.output.iterdir())
            },
        },
    )


if __name__ == "__main__":
    main()
