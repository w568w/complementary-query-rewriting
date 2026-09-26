from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from cqr.data.io import write_jsonl
from cqr.schemas import MemoryEntry, normalize_query


class MemoryEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class BGEM3MemoryEncoder:
    def __init__(
        self, model: str, device: str = "cuda", batch_size: int = 32, max_length: int = 256
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(model, use_fp16=device != "cpu", devices=device)
        self.batch_size = batch_size
        self.max_length = max_length

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), batch_size=self.batch_size)["dense_vecs"],
            dtype=np.float32,
        )

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode_queries(
                list(texts), batch_size=self.batch_size, max_length=self.max_length
            )["dense_vecs"],
            dtype=np.float32,
        )


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Memory encoder returned a zero vector")
    return vectors / norms


class MemoryBank:
    """Thread-safe storage with an asynchronously refreshable dense index."""

    def __init__(
        self,
        encoder: MemoryEncoder,
        entries: Iterable[MemoryEntry] = (),
    ) -> None:
        self.encoder = encoder
        self._entries = list(entries)
        self._entry_ids = {entry.entry_id for entry in self._entries}
        if len(self._entry_ids) != len(self._entries):
            raise ValueError("Memory entries must have unique IDs")
        self._vectors: np.ndarray | None = None
        self._lock = threading.RLock()

    @classmethod
    def load(cls, path: Path, encoder: MemoryEncoder) -> MemoryBank:
        entries = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    entries.append(MemoryEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: invalid memory entry") from error
        return cls(encoder, entries)

    def save(self, path: Path) -> None:
        with self._lock:
            write_jsonl(path, (entry.to_dict() for entry in self._entries))

    def append(self, entries: Iterable[MemoryEntry]) -> int:
        admitted = 0
        with self._lock:
            for entry in entries:
                if entry.entry_id in self._entry_ids:
                    continue
                self._entries.append(entry)
                self._entry_ids.add(entry.entry_id)
                admitted += 1
            if admitted:
                self._vectors = None
        return admitted

    def rebuild_index(self) -> None:
        with self._lock:
            if not self._entries:
                self._vectors = np.empty((0, 0), dtype=np.float32)
                return
            texts = [
                text
                for entry in self._entries
                for text in (entry.source_query, entry.successful_rewrite or entry.source_query)
            ]
            self._vectors = _normalize(self.encoder.encode_queries(texts))

    def retrieve(
        self,
        query: str,
        top_k: int,
        source_id: str | None = None,
    ) -> list[MemoryEntry]:
        return [entry for entry, _ in self.retrieve_scored(query, top_k, source_id)]

    def retrieve_scored(
        self,
        query: str,
        top_k: int,
        source_id: str | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        with self._lock:
            if self._vectors is None:
                self.rebuild_index()
            if not self._entries:
                return []
            assert self._vectors is not None
            query_vector = _normalize(self.encoder.encode_queries([query]))[0]
            scores = (self._vectors @ query_vector).reshape(-1, 2).max(axis=1)
            candidates = [
                index
                for index, entry in enumerate(self._entries)
                if (source_id is None or entry.source_id != source_id)
                and normalize_query(entry.source_query) != normalize_query(query)
            ]
            ordered = sorted(
                candidates, key=lambda index: (-float(scores[index]), self._entries[index].entry_id)
            )
            return [(self._entries[index], float(scores[index])) for index in ordered[:top_k]]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        with self._lock:
            return tuple(self._entries)
