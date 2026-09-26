from __future__ import annotations

import time
from pathlib import Path

from cqr.data.adapters import load_queries
from cqr.data.io import load_jsonl, write_json, write_jsonl
from cqr.manifests import file_sha256
from cqr.memory.bank import BGEM3MemoryEncoder, MemoryBank
from cqr.policy.generate import TransformersRewriter


def artifact_name(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return path.name if path.exists() else value


def generate_candidates(
    queries_path: Path,
    output: Path,
    model: str,
    adapter: str | None,
    condition: str,
    samples: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    memory_path: Path | None = None,
    memory_embedding_model: str | None = None,
    memory_top_k: int = 3,
    device: str = "cuda",
    batch_size: int = 4,
    prompt_style: str = "scope",
    include_scope: bool = True,
    contexts_path: Path | None = None,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if condition not in {"policy", "memory"}:
        raise ValueError("condition must be policy or memory")
    if (
        condition == "memory"
        and contexts_path is None
        and (memory_path is None or memory_embedding_model is None)
    ):
        raise ValueError("Memory generation requires a bank and embedding model")
    if condition == "policy" and memory_path is not None:
        raise ValueError("Policy generation must not receive a memory bank")
    if contexts_path is not None and (memory_path is not None or not include_scope):
        raise ValueError("Frozen contexts cannot be combined with memory or --without-scope")

    from transformers import set_seed

    set_seed(seed)
    rewriter = TransformersRewriter.load(
        model,
        adapter_path=adapter,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        prompt_style=prompt_style,
        include_scope=include_scope,
    )
    memory_bank = None
    if memory_path is not None and memory_embedding_model is not None:
        encoder = BGEM3MemoryEncoder(memory_embedding_model, device=device)
        memory_bank = MemoryBank.load(memory_path, encoder)

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    queries = load_queries(queries_path)
    contexts = {}
    if contexts_path is not None:
        for row in load_jsonl(contexts_path):
            key = str(row["query_id"])
            if key in contexts:
                raise ValueError("Duplicate context query ID")
            contexts[key] = row
        if set(contexts) != {query.query_id for query in queries}:
            raise ValueError("Contexts and queries cover different rosters")
        for query in queries:
            if contexts[query.query_id]["query"] != query.query:
                raise ValueError("Context query text mismatch")
            if condition == "policy" and contexts[query.query_id]["memory_entry_ids"]:
                raise ValueError("Policy context must not contain memory entries")
    rows = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        matches = [
            (
                memory_bank.retrieve_scored(query.query, memory_top_k)
                if memory_bank is not None
                else []
            )
            for query in batch
        ]
        memories = [[entry for entry, _ in match] for match in matches]
        similarities = {
            query.query_id: match[0][1] if match else 0.0
            for query, match in zip(batch, matches, strict=True)
        }
        started = time.monotonic()
        rewrites = (
            rewriter.rewrite_messages_batch(
                [contexts[query.query_id]["messages"] for query in batch], samples=samples
            )
            if contexts
            else rewriter.rewrite_batch([query.query for query in batch], memories, samples=samples)
        )
        query_seconds = (time.monotonic() - started) / len(batch)
        for query, query_memory, query_rewrites in zip(batch, memories, rewrites, strict=True):
            for sample_index, rewrite in enumerate(query_rewrites):
                rows.append(
                    {
                        "query_id": query.query_id,
                        "source_id": query.source_id,
                        "query": query.query,
                        "condition": condition,
                        "sample_index": sample_index,
                        "rewrite": rewrite,
                        "memory_entry_ids": (
                            contexts[query.query_id]["memory_entry_ids"]
                            if contexts
                            else [e.entry_id for e in query_memory]
                        ),
                        "memory_similarity": (
                            contexts[query.query_id]["memory_similarity"]
                            if contexts
                            else similarities[query.query_id]
                        ),
                        "query_generation_seconds": query_seconds,
                    }
                )
    output.mkdir(parents=True)
    candidates_path = output / "candidates.jsonl"
    write_jsonl(candidates_path, rows)
    write_json(
        output / "manifest.json",
        {
            "protocol": "cqr-candidates-v1",
            "condition": condition,
            "queries": len({row["query_id"] for row in rows}),
            "samples_per_query": samples,
            "model": artifact_name(model),
            "adapter": artifact_name(adapter),
            "generation": {
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "enable_thinking": False,
                "microbatch_size": batch_size,
            },
            "memory_bank_sha256": file_sha256(memory_path) if memory_path else None,
            "contexts_sha256": file_sha256(contexts_path) if contexts_path else None,
            "memory_top_k": memory_top_k if memory_path else 0,
            "prompt_style": "frozen_contexts" if contexts_path else prompt_style,
            "include_scope": None if contexts_path else include_scope,
            "memory_retrieval": "max cosine over query/rewrite keys; exact query excluded",
            "queries_sha256": file_sha256(queries_path),
            "candidates_sha256": file_sha256(candidates_path),
            "relevance_labels_passed_to_rewriter": False,
            "relevance_labels_used_for_memory_retrieval": False,
        },
    )
