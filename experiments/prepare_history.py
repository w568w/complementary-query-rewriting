"""AOL4PS chronological event prefixes."""

from __future__ import annotations

import argparse
import csv
import itertools
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from cqr.data.io import write_json, write_jsonl
from cqr.data.prepare import _lookup, history_feedback
from cqr.manifests import file_sha256
from cqr.schemas import stable_id


def build_history_database(data: Path, database: Path) -> dict[str, int]:
    if database.exists():
        raise FileExistsError(database)
    csv.field_size_limit(16 * 1024 * 1024)
    connection = sqlite3.connect(database)
    try:
        connection.executescript("""
            CREATE TABLE events (event_id TEXT PRIMARY KEY, user_id TEXT,
                                 query_time TEXT, session_id TEXT, query_id TEXT);
            CREATE TABLE clicks (row_id INTEGER PRIMARY KEY, event_id TEXT,
                                 doc_id TEXT, click_pos INTEGER);
        """)
        click_rows = 0
        with data.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["DataType"] != "0":
                    continue
                key = {
                    "user_id": row["AnonID"],
                    "query_time": row["QueryTime"],
                    "session_id": row["SessionNo"],
                    "query_id": row["QueryIndex"],
                    "candidate_doc_ids": row["CandiList"].split(),
                }
                event = stable_id(key)
                connection.execute(
                    "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?)",
                    (event, key["user_id"], key["query_time"], key["session_id"], key["query_id"]),
                )
                connection.execute(
                    "INSERT INTO clicks(event_id, doc_id, click_pos) VALUES (?, ?, ?)",
                    (event, row["DocIndex"], int(row["ClickPos"])),
                )
                click_rows += 1
                if click_rows % 100_000 == 0:
                    connection.commit()
        connection.executescript("""
            CREATE INDEX events_time_idx ON events(query_time, event_id);
            CREATE INDEX clicks_event_idx ON clicks(event_id, row_id);
        """)
        connection.commit()
        return {
            "history_events": connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "history_click_rows": click_rows,
            "history_users": connection.execute(
                "SELECT COUNT(DISTINCT user_id) FROM events"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("interactions", "queries", "documents", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0 <= f <= 1 for f in args.fractions):
        raise ValueError("Fractions must be in [0,1]")
    if args.minimum_support < 1 or not 0 <= args.minimum_confidence <= 1:
        raise ValueError("Invalid support/confidence thresholds")
    args.output.mkdir(parents=True)
    database = args.output / "history.sqlite"
    counts = build_history_database(args.interactions, database)
    queries = _lookup(args.queries, "QueryIndex", "Query")
    titles = _lookup(args.documents, "DocIndex", "Title")
    cutoffs = {round(counts["history_events"] * f) for f in args.fractions}
    clicks = defaultdict(Counter)
    prefixes = []

    def snapshot(observed: int) -> None:
        rows = list(
            history_feedback(queries, titles, clicks, args.minimum_support, args.minimum_confidence)
        )
        path = args.output / f"history_{observed}.jsonl"
        write_jsonl(path, rows)
        prefixes.append(
            {
                "observed_history_events": observed,
                "entries": len(rows),
                "file": path.name,
                "sha256": file_sha256(path),
            }
        )

    if 0 in cutoffs:
        snapshot(0)
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute("""
            SELECT e.event_id, e.query_id, c.doc_id FROM events e
            JOIN clicks c ON c.event_id=e.event_id
            ORDER BY e.query_time, e.event_id, c.row_id
        """)
        for observed, (_, event) in enumerate(itertools.groupby(rows, key=lambda r: r[0]), 1):
            for _, query_id, doc_id in event:
                clicks[query_id][doc_id] += 1
            if observed in cutoffs:
                snapshot(observed)
    finally:
        connection.close()
    write_json(
        args.output / "manifest.json",
        {
            **counts,
            "history_partition": "DataType=0",
            "prefixes": prefixes,
            "event_key": ["user", "time", "session", "query ID", "candidate document IDs"],
            "source_sha256": {
                name: file_sha256(getattr(args, name))
                for name in ("interactions", "queries", "documents")
            },
        },
    )


if __name__ == "__main__":
    main()
