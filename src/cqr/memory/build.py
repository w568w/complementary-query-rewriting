from __future__ import annotations

import os
from pathlib import Path

from cqr.data.adapters import load_feedback, load_queries
from cqr.data.io import write_json, write_jsonl
from cqr.manifests import file_sha256
from cqr.memory.writer import (
    MemoryWriter,
    OpenAIJsonCompletionClient,
    TransformersJsonCompletionClient,
)


def build_memory(
    feedback_path: Path,
    source_queries_path: Path | None,
    output: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    max_positive_documents: int,
    max_retrieved_documents: int,
    max_document_characters: int,
    exclude_targets: tuple[Path, ...] = (),
    writer_format: str = "entities",
    refine_scope: bool = False,
    backend: str = "transformers",
    device: str = "cuda",
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    api_key = os.environ.get(api_key_env)
    if backend == "openai" and not api_key:
        raise ValueError(f"Environment variable {api_key_env} is not set")
    if backend not in {"openai", "transformers"}:
        raise ValueError("Unknown writer backend")
    all_feedback = load_feedback(feedback_path)
    if source_queries_path is None:
        feedback = all_feedback
        eligibility_policy = "all records from the normalized historical-feedback partition"
    else:
        eligible_sources = {query.source_id for query in load_queries(source_queries_path)}
        feedback = [record for record in all_feedback if record.source_id in eligible_sources]
        eligibility_policy = "source IDs from supplied split only"
    if not feedback:
        raise ValueError("No feedback records belong to the supplied source split")
    blocked_sources = {query.source_id for path in exclude_targets for query in load_queries(path)}
    overlap_excluded = [record for record in feedback if record.source_id in blocked_sources]
    feedback = [record for record in feedback if record not in overlap_excluded]
    if not feedback:
        raise ValueError("Evaluation-source exclusion removed all feedback")

    writer = MemoryWriter(
        client=(
            OpenAIJsonCompletionClient.create(api_key, base_url)
            if backend == "openai"
            else TransformersJsonCompletionClient(model, device)
        ),
        model=model,
        max_positive_documents=max_positive_documents,
        max_retrieved_documents=max_retrieved_documents,
        max_document_characters=max_document_characters,
        writer_format=writer_format,
        refine_scope=refine_scope,
    )
    entries = []
    excluded = []
    for record in feedback:
        entry = writer.extract(record)
        if entry is None:
            excluded.append({"query_id": record.query_id, "source_id": record.source_id})
        else:
            entries.append(entry)
    if not entries:
        raise ValueError("Memory writer produced no reusable entries")

    output.mkdir(parents=True)
    memory_path = output / "memory.jsonl"
    excluded_path = output / "excluded.jsonl"
    write_jsonl(memory_path, (entry.to_dict() for entry in entries))
    write_jsonl(excluded_path, excluded)
    write_json(
        output / "manifest.json",
        {
            "protocol": "cqr-memory-build-v2",
            "feedback_records": len(feedback),
            "usable_entries": len(entries),
            "excluded_entries": len(excluded),
            "source_overlap_excluded_before_writing": len(overlap_excluded),
            "shared_target_documents_allowed": True,
            "writer_model": model,
            "writer_format": writer_format,
            "writer_backend": backend,
            "refine_scope": refine_scope,
            "source_split_sha256": (
                file_sha256(source_queries_path) if source_queries_path else None
            ),
            "feedback_sha256": file_sha256(feedback_path),
            "excluded_target_rosters_sha256": [file_sha256(path) for path in exclude_targets],
            "memory_sha256": file_sha256(memory_path),
            "current_query_qrels_used": False,
            "eligible_source_policy": eligibility_policy,
        },
    )
