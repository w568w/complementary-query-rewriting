from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    source_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    original_ranking_doc_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QueryExample:
        return cls(
            query_id=str(value["query_id"]),
            source_id=str(value["source_id"]),
            query=str(value["query"]),
            relevant_doc_ids=tuple(map(str, value.get("relevant_doc_ids") or ())),
            original_ranking_doc_ids=tuple(map(str, value.get("original_ranking_doc_ids") or ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Document:
        raw_doc_id = next(
            (value[key] for key in ("doc_id", "id", "passage_id") if key in value),
            "",
        )
        raw_text = next(
            (value[key] for key in ("text", "contents", "passage") if key in value),
            "",
        )
        doc_id = str(raw_doc_id)
        text = str(raw_text).strip()
        if not doc_id or not text:
            raise ValueError("Document ID and text must be non-empty")
        return cls(doc_id=doc_id, text=text)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackRecord:
    query_id: str
    source_id: str
    query: str
    successful_rewrite: str
    positive_documents: tuple[tuple[str, str], ...]
    retrieved_documents: tuple[tuple[str, str], ...] = ()
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeedbackRecord:
        return cls(
            query_id=str(value["query_id"]),
            source_id=str(value["source_id"]),
            query=str(value["query"]),
            successful_rewrite=str(value["successful_rewrite"]),
            positive_documents=tuple(
                (str(row["doc_id"]), str(row["text"]))
                for row in value.get("positive_documents") or ()
            ),
            retrieved_documents=tuple(
                (str(row["doc_id"]), str(row["text"]))
                for row in value.get("retrieved_documents") or ()
            ),
            timestamp=str(value["timestamp"]) if value.get("timestamp") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["positive_documents"] = [
            {"doc_id": doc_id, "text": text} for doc_id, text in self.positive_documents
        ]
        value["retrieved_documents"] = [
            {"doc_id": doc_id, "text": text} for doc_id, text in self.retrieved_documents
        ]
        return value


@dataclass(frozen=True)
class EntityDefinition:
    name: str
    description: str
    scope: str = ""
    synonyms: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EntityDefinition:
        name = str(value.get("entity") or value.get("name") or "").strip()
        description = str(value.get("definition") or value.get("description") or "").strip()
        if not name or not description:
            raise ValueError("Entity name and description must be non-empty")
        return cls(
            name=name,
            description=description,
            scope=str(value.get("scope") or "").strip(),
            synonyms=tuple(map(str, value.get("synonyms") or ())),
        )

    def context_dict(self, include_scope: bool = True) -> dict[str, Any]:
        value = {
            "entity": self.name,
            "definition": self.description,
            "synonyms": list(self.synonyms),
        }
        if include_scope:
            value["scope"] = self.scope
        return value


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    source_id: str
    source_query: str
    successful_rewrite: str
    entities: tuple[EntityDefinition, ...]
    constraints: tuple[str, ...]
    positive_doc_ids: tuple[str, ...]
    writer_model: str | None = None
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryEntry:
        constraints = tuple(str(item).strip() for item in value.get("constraints") or ())
        if any(not item for item in constraints):
            raise ValueError("Memory constraints must be non-empty")
        return cls(
            entry_id=str(value["entry_id"]),
            source_id=str(value["source_id"]),
            source_query=str(value["source_query"]),
            successful_rewrite=str(value.get("successful_rewrite") or ""),
            entities=tuple(
                EntityDefinition.from_dict(item) for item in value.get("entities") or ()
            ),
            constraints=constraints,
            positive_doc_ids=tuple(map(str, value.get("positive_doc_ids") or ())),
            writer_model=str(value["writer_model"]) if value.get("writer_model") else None,
            timestamp=str(value["timestamp"]) if value.get("timestamp") else None,
        )

    @classmethod
    def from_feedback(
        cls,
        feedback: FeedbackRecord,
        entities: Iterable[EntityDefinition],
        constraints: Iterable[str],
        writer_model: str,
    ) -> MemoryEntry:
        content = {
            "query_id": feedback.query_id,
            "source_id": feedback.source_id,
            "query": feedback.query,
            "successful_rewrite": feedback.successful_rewrite,
            "positive_doc_ids": [doc_id for doc_id, _ in feedback.positive_documents],
        }
        return cls(
            entry_id=stable_id(content),
            source_id=feedback.source_id,
            source_query=feedback.query,
            successful_rewrite=feedback.successful_rewrite,
            entities=tuple(entities),
            constraints=tuple(constraints),
            positive_doc_ids=tuple(doc_id for doc_id, _ in feedback.positive_documents),
            writer_model=writer_model,
            timestamp=feedback.timestamp,
        )

    def retrieval_text(self) -> str:
        entities = "; ".join(f"{entity.name}: {entity.description}" for entity in self.entities)
        constraints = "; ".join(self.constraints)
        return "\n".join(
            part
            for part in (
                f"Source query: {self.source_query}",
                f"Successful rewrite: {self.successful_rewrite}",
                f"Entities: {entities}" if entities else "",
                f"Constraints: {constraints}" if constraints else "",
            )
            if part
        )

    def context_text(self) -> str:
        if self.constraints:
            return canonical_json(
                {
                    "entities": [
                        {"name": e.name, "description": e.description} for e in self.entities
                    ],
                    "constraints": list(self.constraints),
                }
            )
        return canonical_json([entity.context_dict() for entity in self.entities])

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["entities"] = [asdict(entity) for entity in self.entities]
        return value
