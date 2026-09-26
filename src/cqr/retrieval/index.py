from __future__ import annotations

import json
from pathlib import Path

from cqr.data.adapters import load_documents
from cqr.data.io import write_json, write_jsonl
from cqr.manifests import file_sha256
from cqr.retrieval.hybrid import HybridRetriever


def build_index(
    documents_path: Path,
    output: Path,
    embedding_model: str,
    device: str,
    batch_size: int,
    rank_constant: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    documents = load_documents(documents_path)
    retriever = HybridRetriever.build(
        [(document.doc_id, document.text) for document in documents],
        embedding_model=embedding_model,
        device=device,
        rank_constant=rank_constant,
        batch_size=batch_size,
    )
    output.mkdir(parents=True)
    dense_path = output / "dense.safetensors"
    sparse_path = output / "bm25"
    document_ids_path = output / "document_ids.json"
    retriever.save(dense_path, sparse_path)
    retriever.save_document_ids(document_ids_path)
    write_json(
        output / "manifest.json",
        {
            "protocol": "cqr-hybrid-index-v1",
            "documents": len(documents),
            "embedding_model": (
                Path(embedding_model).name
                if Path(embedding_model).exists()
                else embedding_model
            ),
            "sparse_model": "bm25s-with-english-stemming",
            "fusion": "reciprocal-rank-fusion",
            "rank_constant": rank_constant,
            "source_documents_sha256": file_sha256(documents_path),
            "dense_index_sha256": file_sha256(dense_path),
            "document_ids_sha256": file_sha256(document_ids_path),
        },
    )


def load_index(index: Path, embedding_model: str, device: str) -> HybridRetriever:
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "cqr-hybrid-index-v1":
        raise ValueError("Unsupported retrieval index protocol")
    document_ids_path = index / "document_ids.json"
    if file_sha256(document_ids_path) != manifest["document_ids_sha256"]:
        raise ValueError("Document-ID index does not match its manifest")
    dense_path = index / "dense.safetensors"
    if file_sha256(dense_path) != manifest["dense_index_sha256"]:
        raise ValueError("Dense index does not match its manifest")
    return HybridRetriever.load(
        document_ids=HybridRetriever.load_document_ids(document_ids_path),
        dense_index=dense_path,
        sparse_index=index / "bm25",
        embedding_model=embedding_model,
        device=device,
        rank_constant=float(manifest["rank_constant"]),
    )


def score_original_queries(
    queries_path: Path,
    index: Path,
    output: Path,
    embedding_model: str,
    device: str,
    cutoff: int,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    from cqr.data.adapters import load_queries

    queries = load_queries(queries_path)
    retriever = load_index(index, embedding_model, device)
    scored = retriever.score_original(queries, top_k=cutoff)
    write_jsonl(output, (query.to_dict() for query in scored))
