from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cqr.data.adapters import load_queries
from cqr.data.io import load_jsonl, write_json, write_jsonl
from cqr.manifests import file_sha256
from cqr.retrieval.fusion import fuse_paths, memory_weight
from cqr.retrieval.index import load_index


def _candidates(path: Path, expected_condition: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = load_jsonl(path / "candidates.jsonl")
    if manifest.get("protocol") != "cqr-candidates-v1":
        raise ValueError("Unsupported candidate protocol")
    if manifest.get("condition") != expected_condition:
        raise ValueError(f"Expected {expected_condition} candidates")
    if file_sha256(path / "candidates.jsonl") != manifest["candidates_sha256"]:
        raise ValueError("Candidate content does not match its manifest")
    keys = [(str(row["query_id"]), int(row["sample_index"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Candidate file contains duplicate query/sample keys")
    return rows, manifest


def infer_rankings(
    queries_path: Path,
    index: Path,
    embedding_model: str,
    policy_candidates: Path,
    output: Path,
    mode: str,
    method: str,
    cutoff: int,
    device: str,
    memory_candidates: Path | None = None,
    fusion_method: str = "adaptive",
    fusion_rank_constant: float = 1.0,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if mode not in {"rl", "full"}:
        raise ValueError("mode must be rl or full")
    if fusion_method not in {"adaptive", "equal", "similarity"}:
        raise ValueError("Unknown fusion method")
    if (mode == "full") != (memory_candidates is not None):
        raise ValueError("Full inference requires memory candidates, and RL inference forbids them")

    queries = {query.query_id: query for query in load_queries(queries_path)}
    policy_rows, policy_manifest = _candidates(policy_candidates, "policy")
    policy = {(str(row["query_id"]), int(row["sample_index"])): row for row in policy_rows}
    memory: dict[tuple[str, int], dict[str, Any]] = {}
    memory_manifest = None
    if memory_candidates is not None:
        memory_rows, memory_manifest = _candidates(memory_candidates, "memory")
        memory = {(str(row["query_id"]), int(row["sample_index"])): row for row in memory_rows}
        if set(memory) != set(policy):
            raise ValueError("Policy and memory experts do not cover identical query/sample keys")
    if {query_id for query_id, _ in policy} != set(queries):
        raise ValueError("Candidates and query roster do not contain identical query IDs")

    retriever = load_index(index, embedding_model, device)
    ordered_keys = sorted(policy)
    policy_components = retriever.retrieve_components(
        (str(policy[key]["rewrite"]) for key in ordered_keys), top_k=cutoff
    )
    memory_components = (
        retriever.retrieve_components(
            (str(memory[key]["rewrite"]) for key in ordered_keys), top_k=cutoff
        )
        if memory
        else []
    )
    missing_original = [query for query in queries.values() if not query.original_ranking_doc_ids]
    scored_original = (
        retriever.score_original(missing_original, top_k=cutoff) if missing_original else []
    )
    original_rankings = {
        query.query_id: list(query.original_ranking_doc_ids) for query in queries.values()
    }
    original_rankings.update(
        {query.query_id: list(query.original_ranking_doc_ids) for query in scored_original}
    )

    output_rows = []
    for index_value, key in enumerate(ordered_keys):
        query_id, sample_index = key
        query = queries[query_id]
        policy_parts = policy_components[index_value]
        policy_ranking = policy_parts["hybrid"]
        alpha = None
        if mode == "full":
            memory_parts = memory_components[index_value]
            memory_ranking = memory_parts["hybrid"]
            if fusion_method == "equal":
                alpha = 0.5
            else:
                if "memory_similarity" not in memory[key]:
                    raise ValueError("Regenerate legacy candidates: memory_similarity is required")
                similarity = float(memory[key]["memory_similarity"])
                alpha = memory_weight(
                    similarity,
                    policy_parts["dense"],
                    policy_parts["sparse"],
                    memory_parts["dense"],
                    memory_parts["sparse"],
                )
                if fusion_method == "similarity":
                    alpha = min(1.0, max(0.0, similarity))
            final_ranking = fuse_paths(
                policy_ranking, memory_ranking, alpha, fusion_rank_constant, cutoff
            )
            generation_seconds = float(policy[key]["query_generation_seconds"]) + float(
                memory[key]["query_generation_seconds"]
            )
        else:
            memory_ranking = []
            final_ranking = policy_ranking
            generation_seconds = float(policy[key]["query_generation_seconds"])
        output_rows.append(
            {
                "query_id": query_id,
                "source_id": query.source_id,
                "method": method,
                "sample_index": sample_index,
                "relevant_doc_ids": list(query.relevant_doc_ids),
                "original_ranking_doc_ids": original_rankings[query_id][:cutoff],
                "policy_ranking_doc_ids": policy_ranking,
                "memory_ranking_doc_ids": memory_ranking,
                "ranking_doc_ids": final_ranking,
                "memory_weight": alpha,
                "policy_components": policy_parts,
                "memory_components": memory_components[index_value] if memory else None,
                "query_generation_seconds": generation_seconds,
            }
        )
    output.mkdir(parents=True)
    rankings_path = output / "rankings.jsonl"
    write_jsonl(rankings_path, output_rows)
    write_json(
        output / "manifest.json",
        {
            "protocol": "cqr-ranked-samples-v1",
            "mode": mode,
            "method": method,
            "queries": len(queries),
            "samples_per_query": int(policy_manifest["samples_per_query"]),
            "cutoff": cutoff,
            "fusion": (
                {
                    "method": fusion_method,
                    "rank_constant": fusion_rank_constant,
                    "rank_origin": 1,
                    "agreement": "reciprocal-rank-weighted-jaccard",
                }
                if mode == "full"
                else None
            ),
            "policy_candidates_sha256": file_sha256(policy_candidates / "candidates.jsonl"),
            "memory_candidates_sha256": (
                file_sha256(memory_candidates / "candidates.jsonl") if memory_candidates else None
            ),
            "policy_model": policy_manifest["model"],
            "policy_adapter": policy_manifest["adapter"],
            "memory_model": memory_manifest["model"] if memory_manifest else None,
            "memory_adapter": memory_manifest["adapter"] if memory_manifest else None,
            "qrels_used_for_fusion": False,
            "rankings_sha256": file_sha256(rankings_path),
        },
    )
