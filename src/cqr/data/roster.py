from __future__ import annotations

import json
from pathlib import Path

from cqr.data.adapters import load_queries
from cqr.data.io import write_json, write_jsonl
from cqr.manifests import file_sha256
from cqr.schemas import stable_id


def freeze_roster(queries: Path, query_ids: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    roster = json.loads(query_ids.read_text(encoding="utf-8"))
    if not isinstance(roster, list) or any(not isinstance(value, str) for value in roster):
        raise ValueError("Roster must be a JSON array of query IDs")
    if len(set(roster)) != len(roster):
        raise ValueError("Roster contains duplicate query IDs")
    rows = {query.query_id: query for query in load_queries(queries)}
    missing = [query_id for query_id in roster if query_id not in rows]
    if missing:
        raise ValueError(f"Roster contains unknown query IDs: {missing[:5]}")
    selected = [rows[query_id] for query_id in roster]
    output.mkdir(parents=True)
    selected_path = output / "queries.jsonl"
    write_jsonl(selected_path, (query.to_dict() for query in selected))
    write_json(
        output / "manifest.json",
        {
            "protocol": "cqr-frozen-roster-v1",
            "queries": len(selected),
            "source_groups": len({query.source_id for query in selected}),
            "ordered_query_ids_sha256": stable_id(roster),
            "source_queries_sha256": file_sha256(queries),
            "source_roster_sha256": file_sha256(query_ids),
            "output_sha256": file_sha256(selected_path),
        },
    )
