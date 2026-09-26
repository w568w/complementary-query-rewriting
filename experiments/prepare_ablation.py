"""Freeze matched control prompts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from cqr.data.adapters import load_feedback, load_queries
from cqr.data.io import write_jsonl
from cqr.memory.bank import BGEM3MemoryEncoder, MemoryBank, _normalize
from cqr.policy.prompts import build_expansion_messages, build_rewrite_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("queries", "memory", "feedback", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=("nfcorpus", "marco"))
    parser.add_argument("--encoder", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--condition",
        required=True,
        choices=("rl", "memory", "random", "no_scope", "full_documents", "direct_documents"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    encoder = BGEM3MemoryEncoder(args.encoder, device=args.device)
    bank = MemoryBank.load(args.memory, encoder)
    by_entry = {entry.entry_id: entry for entry in bank.entries if entry.entities}
    bank = MemoryBank(encoder, by_entry.values())
    documents = {}
    for feedback in load_feedback(args.feedback):
        for doc, text in feedback.positive_documents:
            if doc in documents and documents[doc] != text:
                raise ValueError("Historical document ID maps to inconsistent text")
            documents[doc] = text
    doc_ids = sorted({doc for entry in by_entry.values() for doc in entry.positive_doc_ids})
    vectors = (
        _normalize(encoder.encode_documents([documents[d] for d in doc_ids]))
        if args.condition == "direct_documents"
        else None
    )
    rows = []
    for query in load_queries(args.queries):
        match = bank.retrieve_scored(query.query, 1)
        if not match:
            raise ValueError("No eligible history for a fixed-cohort query")
        selected, similarity = match[0]
        if args.condition == "random":
            choices = sorted(
                entry
                for entry in by_entry
                if entry != selected.entry_id
                and by_entry[entry].source_query.strip().casefold()
                != query.query.strip().casefold()
            )
            if not choices:
                raise ValueError("No eligible random-memory donor")
            key = f"topk-random-48:{args.dataset}:{query.query_id}"
            selected = by_entry[
                choices[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(choices)]
            ]
            # The fusion prior belongs to the supplied donor, not the original nearest memory.
            keys = [selected.source_query, selected.successful_rewrite or selected.source_query]
            similarity = float(
                (
                    _normalize(encoder.encode_queries(keys))
                    @ _normalize(encoder.encode_queries([query.query]))[0]
                ).max()
            )
        entries = [] if args.condition == "rl" else [selected]
        style = "expansion" if args.condition == "rl" else "scope"
        messages = build_rewrite_messages(query.query, entries, style, args.condition != "no_scope")
        document_id = None
        if args.condition in {"full_documents", "direct_documents"}:
            if args.condition == "direct_documents":
                eligible = {
                    doc
                    for entry in by_entry.values()
                    if entry.source_query.strip().casefold() != query.query.strip().casefold()
                    for doc in entry.positive_doc_ids
                }
                qv = _normalize(encoder.encode_queries([query.query]))[0]
                assert vectors is not None
                scores = vectors @ qv
                index = max(
                    (i for i, doc in enumerate(doc_ids) if doc in eligible),
                    key=lambda i: float(scores[i]),
                )
                document_id = doc_ids[index]
                providers = sorted(
                    e.entry_id
                    for e in by_entry.values()
                    if document_id in e.positive_doc_ids
                    and e.source_query.strip().casefold() != query.query.strip().casefold()
                )
                selected = by_entry[providers[0]]
                keys = [selected.source_query, selected.successful_rewrite or selected.source_query]
                similarity = float((_normalize(encoder.encode_queries(keys)) @ qv).max())
            else:
                if len(selected.positive_doc_ids) != 1:
                    raise ValueError(
                        "This matched control requires one feedback document per entry"
                    )
                document_id = selected.positive_doc_ids[0]
            messages = build_expansion_messages(query.query, documents[document_id])
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "condition": args.condition,
                "messages": messages,
                "memory_entry_ids": [selected.entry_id] if entries else [],
                "memory_similarity": similarity,
                "document_id": document_id,
            }
        )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
