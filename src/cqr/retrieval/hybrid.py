from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    rank_constant: float = 60.0,
    weights: Sequence[float] | None = None,
) -> list[str]:
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    ranking_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(ranking_weights) != len(rankings):
        raise ValueError("Each ranking requires one weight")
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, ranking_weights, strict=True):
        for rank, doc_id in enumerate(ranking):
            scores[str(doc_id)] += float(weight) / (rank_constant + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: -item[1])]


class HybridRetriever:
    """BGE-M3 dense retrieval fused with stemmed BM25 through RRF."""

    def __init__(
        self,
        document_ids: Sequence[str],
        dense_embeddings: np.ndarray,
        dense_encoder: object,
        bm25: object,
        stemmer: object,
        rank_constant: float = 60.0,
    ) -> None:
        embeddings = np.asarray(dense_embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(document_ids):
            raise ValueError("Dense embeddings must align with document IDs")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Dense index contains zero vectors")
        self.document_ids = tuple(map(str, document_ids))
        self.max_document_multiplicity = max(Counter(self.document_ids).values())
        self.dense_embeddings = embeddings / norms
        self.dense_encoder = dense_encoder
        self.bm25 = bm25
        self.stemmer = stemmer
        self.rank_constant = rank_constant

    @classmethod
    def build(
        cls,
        documents: Sequence[tuple[str, str]],
        embedding_model: str,
        device: str = "cuda",
        rank_constant: float = 60.0,
        batch_size: int = 32,
    ) -> HybridRetriever:
        if not documents:
            raise ValueError("Cannot build a retriever from an empty corpus")
        import bm25s
        import Stemmer
        from FlagEmbedding import BGEM3FlagModel

        document_ids = [str(doc_id) for doc_id, _ in documents]
        texts = [text for _, text in documents]
        encoder = BGEM3FlagModel(
            embedding_model,
            use_fp16=device != "cpu",
            devices=device,
        )
        embeddings = encoder.encode(texts, batch_size=batch_size)["dense_vecs"]
        stemmer = Stemmer.Stemmer("english")
        bm25 = bm25s.BM25()
        bm25.index(bm25s.tokenize(texts, stemmer=stemmer))
        return cls(document_ids, embeddings, encoder, bm25, stemmer, rank_constant)

    @classmethod
    def load(
        cls,
        document_ids: Sequence[str],
        dense_index: Path,
        sparse_index: Path,
        embedding_model: str,
        device: str = "cuda",
        rank_constant: float = 60.0,
    ) -> HybridRetriever:
        import bm25s
        import Stemmer
        from FlagEmbedding import BGEM3FlagModel
        from safetensors.numpy import load_file

        tensors = load_file(str(dense_index))
        if "embeddings" not in tensors:
            raise ValueError("Dense index must contain an 'embeddings' tensor")
        encoder = BGEM3FlagModel(
            embedding_model,
            use_fp16=device != "cpu",
            devices=device,
        )
        return cls(
            document_ids=document_ids,
            dense_embeddings=tensors["embeddings"],
            dense_encoder=encoder,
            bm25=bm25s.BM25.load(str(sparse_index)),
            stemmer=Stemmer.Stemmer("english"),
            rank_constant=rank_constant,
        )

    def save(self, dense_index: Path, sparse_index: Path) -> None:
        from safetensors.numpy import save_file

        dense_index.parent.mkdir(parents=True, exist_ok=True)
        sparse_index.parent.mkdir(parents=True, exist_ok=True)
        save_file({"embeddings": self.dense_embeddings}, str(dense_index))
        self.bm25.save(str(sparse_index))

    def save_document_ids(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(self.document_ids), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load_document_ids(path: Path) -> list[str]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("Document-ID index must be an array of strings")
        return value

    def _dense_rankings(self, queries: Sequence[str], top_k: int) -> list[list[int]]:
        query_vectors = np.asarray(
            self.dense_encoder.encode_queries(list(queries))["dense_vecs"],
            dtype=np.float32,
        )
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Dense encoder returned a zero query vector")
        scores = query_vectors / norms @ self.dense_embeddings.T
        width = min(top_k, len(self.document_ids))
        candidates = np.argpartition(scores, -width, axis=1)[:, -width:]
        rankings = []
        for row_scores, row_candidates in zip(scores, candidates, strict=True):
            ordered = row_candidates[np.argsort(-row_scores[row_candidates])]
            rankings.append([int(index) for index in ordered])
        return rankings

    def _sparse_rankings(self, queries: Sequence[str], top_k: int) -> list[list[int]]:
        import bm25s

        tokens = bm25s.tokenize(list(queries), stemmer=self.stemmer)
        indices, _ = self.bm25.retrieve(tokens, k=min(top_k, len(self.document_ids)))
        return [[int(index) for index in row] for row in indices.tolist()]

    def retrieve(self, queries: Iterable[str], top_k: int = 25) -> list[list[str]]:
        return [row["hybrid"] for row in self.retrieve_components(queries, top_k)]

    def retrieve_components(
        self, queries: Iterable[str], top_k: int = 25,
    ) -> list[dict[str, list[str]]]:
        query_list = list(queries)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        retrieval_width = min(
            len(self.document_ids), top_k * self.max_document_multiplicity
        )
        dense = self._dense_rankings(query_list, retrieval_width)
        sparse = self._sparse_rankings(query_list, retrieval_width)
        output = []
        for dense_ranking, sparse_ranking in zip(dense, sparse, strict=True):
            fused_indices = reciprocal_rank_fusion(
                (
                    [str(index) for index in dense_ranking],
                    [str(index) for index in sparse_ranking],
                ),
                rank_constant=self.rank_constant,
            )
            ranking = []
            seen = set()
            for raw_index in fused_indices:
                doc_id = self.document_ids[int(raw_index)]
                if doc_id in seen:
                    continue
                ranking.append(doc_id)
                seen.add(doc_id)
                if len(ranking) == top_k:
                    break
            output.append({
                "hybrid": ranking,
                "dense": list(dict.fromkeys(self.document_ids[i] for i in dense_ranking))[:top_k],
                "sparse": list(dict.fromkeys(self.document_ids[i] for i in sparse_ranking))[:top_k],
            })
        return output

    def score_original(self, examples: Iterable[object], top_k: int = 25) -> list[object]:
        from cqr.schemas import QueryExample

        rows = list(examples)
        rankings = self.retrieve((str(row.query) for row in rows), top_k=top_k)
        return [
            QueryExample(
                query_id=row.query_id,
                source_id=row.source_id,
                query=row.query,
                relevant_doc_ids=row.relevant_doc_ids,
                original_ranking_doc_ids=tuple(ranking),
            )
            for row, ranking in zip(rows, rankings, strict=True)
        ]
