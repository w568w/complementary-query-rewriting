from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from cqr.data.io import iter_jsonl, load_jsonl, write_json, write_jsonl
from cqr.manifests import file_sha256
from cqr.schemas import Document, FeedbackRecord, QueryExample, stable_id


def _json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain an array of objects")
    return value


def _write_dataset(
    output: Path,
    dataset: str,
    documents: Iterable[Document],
    queries: Iterable[QueryExample],
    feedback: Iterable[FeedbackRecord],
    metadata: Mapping[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    document_rows = list(documents)
    query_rows = list(queries)
    feedback_rows = list(feedback)
    if not document_rows or not query_rows:
        raise ValueError("Normalized data requires documents and queries")
    if len({row.query_id for row in query_rows}) != len(query_rows):
        raise ValueError("Normalized data contains duplicate query IDs")
    corpus_ids = {row.doc_id for row in document_rows}
    missing = sorted(
        {
            doc_id
            for row in query_rows
            for doc_id in row.relevant_doc_ids
            if doc_id not in corpus_ids
        }
    )
    if missing:
        raise ValueError(f"Relevant documents missing from corpus: {missing[:5]}")

    output.mkdir(parents=True)
    documents_path = output / "documents.jsonl"
    queries_path = output / "queries.jsonl"
    feedback_path = output / "feedback.jsonl"
    write_jsonl(documents_path, (row.to_dict() for row in document_rows))
    write_jsonl(queries_path, (row.to_dict() for row in query_rows))
    write_jsonl(feedback_path, (row.to_dict() for row in feedback_rows))
    write_json(
        output / "manifest.json",
        {
            "protocol": "cqr-normalized-data-v1",
            "dataset": dataset,
            "documents": len(document_rows),
            "queries": len(query_rows),
            "source_groups": len({row.source_id for row in query_rows}),
            "feedback_records": len(feedback_rows),
            "output_sha256": {
                "documents": file_sha256(documents_path),
                "queries": file_sha256(queries_path),
                "feedback": file_sha256(feedback_path),
            },
            **dict(metadata),
        },
    )


def _retrieved_texts(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    texts: list[str] = []
    for serialized in row.get("retrieval_context") or ():
        value = json.loads(serialized) if isinstance(serialized, str) else serialized
        if not isinstance(value, dict):
            continue
        documents = value.get("documents retrieved by user query")
        if isinstance(documents, list):
            texts.extend(str(document).strip() for document in documents if str(document).strip())
    return tuple((f"retrieved-{stable_id(text)}", text) for text in texts)


def prepare_nfcorpus(
    augmented_queries: Path,
    corpus: Path,
    canonical_queries: Path,
    output: Path,
) -> None:
    corpus_rows = load_jsonl(corpus)
    documents = [
        Document(
            doc_id=str(row.get("_id") or row.get("doc_id")),
            text=f"{row.get('title') or ''}\n\n{row.get('text') or ''}".strip(),
        )
        for row in corpus_rows
    ]
    canonical = {
        str(row.get("_id") or row.get("query_id")): str(row.get("text") or row.get("query"))
        for row in load_jsonl(canonical_queries)
    }
    _prepare_augmented(
        "nfcorpus", documents, canonical, augmented_queries, corpus, canonical_queries, output
    )


def prepare_msmarco(
    augmented_queries: Path,
    corpus: Path,
    canonical_queries: Path,
    output: Path,
) -> None:
    import datasets

    corpus_dataset = datasets.load_dataset("parquet", data_files=str(corpus))["train"]
    documents = [
        Document(str(row["doc_id"]), str(row.get("doc") or row.get("text")))
        for row in corpus_dataset
    ]
    canonical_dataset = datasets.load_dataset(
        "parquet", data_files=str(canonical_queries)
    )["train"]
    canonical = {
        str(row.get("query_id") if row.get("query_id") is not None else row["id"]): str(
            row["query"]
        )
        for row in canonical_dataset
    }
    _prepare_augmented(
        "msmarco", documents, canonical, augmented_queries, corpus, canonical_queries, output
    )


def _prepare_augmented(
    dataset: str,
    documents: list[Document],
    canonical: Mapping[str, str],
    augmented_queries: Path,
    corpus: Path,
    canonical_queries: Path,
    output: Path,
) -> None:
    documents_by_id = {row.doc_id: row.text for row in documents}
    queries, feedback = [], []
    for row in load_jsonl(augmented_queries):
        if dataset == "nfcorpus":
            source_id = str(row["id"])
        else:
            source_id = str(row.get("id") if row.get("id") is not None else row["query_id"])
        relevant = tuple(map(str, row.get("doc_ids") or ()))
        query = str(row["query"])
        query_id = stable_id({"dataset": dataset, "source_id": source_id, "query": query})
        queries.append(QueryExample(query_id, source_id, query, relevant))
        feedback.append(
            FeedbackRecord(
                query_id=query_id,
                source_id=source_id,
                query=query,
                successful_rewrite=canonical.get(source_id, query),
                positive_documents=tuple((doc_id, documents_by_id[doc_id]) for doc_id in relevant),
                retrieved_documents=_retrieved_texts(row),
            )
        )
    _write_dataset(
        output,
        dataset,
        documents,
        queries,
        feedback,
        {
            "group_key": "source_id",
            "source_sha256": {
                "augmented_queries": file_sha256(augmented_queries),
                "corpus": file_sha256(corpus),
                "canonical_queries": file_sha256(canonical_queries),
            },
        },
    )


def _tsv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def _lookup(path: Path, key: str, value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _tsv_rows(path):
        row_key = str(row[key])
        row_value = str(row[value])
        if row_key in result and result[row_key] != row_value:
            raise ValueError(f"Conflicting {key}: {row_key}")
        result[row_key] = row_value
    return result


def history_feedback(
    query_texts: Mapping[str, str],
    document_titles: Mapping[str, str],
    query_clicks: Mapping[str, Counter[str]],
    minimum_support: int,
    minimum_confidence: float,
) -> Iterable[dict[str, Any]]:
    for query_id in sorted(query_clicks):
        clicks = query_clicks[query_id]
        total = sum(clicks.values())
        doc_id, support = min(clicks.items(), key=lambda item: (-item[1], item[0]))
        confidence = support / total
        query = query_texts.get(query_id, "").strip()
        title = document_titles.get(doc_id, "").strip()
        if support < minimum_support or confidence < minimum_confidence or not query or not title:
            continue
        yield {
            "query_id": f"history-{query_id}",
            "source_id": f"history-query-{query_id}",
            "query": query,
            "successful_rewrite": query,
            "positive_documents": [{"doc_id": doc_id, "text": title}],
            "support": support,
            "total_clicks": total,
            "confidence": confidence,
        }


def prepare_aol4ps(
    interactions: Path,
    query_table: Path,
    document_table: Path,
    split: str,
    minimum_support: int,
    minimum_confidence: float,
    output: Path,
) -> None:
    split_codes = {"validation": "2", "test": "3"}
    if split not in split_codes:
        raise ValueError("AOL4PS split must be validation or test")
    query_text = _lookup(query_table, "QueryIndex", "Query")
    document_text = _lookup(document_table, "DocIndex", "Title")
    documents = [Document(doc_id, text) for doc_id, text in sorted(document_text.items()) if text]
    history_clicks: dict[str, Counter[str]] = defaultdict(Counter)
    interactions_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in _tsv_rows(interactions):
        data_type = str(row["DataType"])
        if data_type == "0":
            history_clicks[str(row["QueryIndex"])][str(row["DocIndex"])] += 1
            continue
        if data_type != split_codes[split]:
            continue
        key = (
            str(row["AnonID"]),
            str(row["QueryTime"]),
            str(row["SessionNo"]),
            str(row["QueryIndex"]),
            str(row["CandiList"]),
        )
        record = interactions_by_key.setdefault(
            key,
            {
                "user_id": str(row["AnonID"]),
                "timestamp": str(row["QueryTime"]),
                "session_id": str(row["SessionNo"]),
                "query_id": str(row["QueryIndex"]),
                "candidate_doc_ids": str(row["CandiList"]).split(),
                "relevant_doc_ids": set(),
            },
        )
        record["relevant_doc_ids"].add(str(row["DocIndex"]))

    history_query_ids = set(history_clicks)
    queries = []
    for key, row in sorted(interactions_by_key.items()):
        if row["query_id"] in history_query_ids:
            continue
        relevant = tuple(sorted(row["relevant_doc_ids"] & set(row["candidate_doc_ids"])))
        query = query_text.get(row["query_id"], "").strip()
        if not relevant or not query:
            continue
        queries.append(
            QueryExample(
                query_id=stable_id({"dataset": "aol4ps", "interaction": key}),
                source_id=row["user_id"],
                query=query,
                relevant_doc_ids=relevant,
                original_ranking_doc_ids=tuple(row["candidate_doc_ids"]),
            )
        )

    feedback = [
        FeedbackRecord.from_dict(row)
        for row in history_feedback(
            query_text, document_text, history_clicks, minimum_support, minimum_confidence
        )
    ]
    _write_dataset(
        output,
        "aol4ps",
        documents,
        queries,
        feedback,
        {
            "split": split,
            "group_key": "user_id",
            "history_scope": "DataType=0",
            "history_query_overlap": 0,
            "source_sha256": {
                "interactions": file_sha256(interactions),
                "queries": file_sha256(query_table),
                "documents": file_sha256(document_table),
            },
        },
    )


def prepare_qrecc(
    records: Path,
    passages: Path,
    official_split: str,
    output: Path,
    truth: Path | None = None,
) -> None:
    rows = _json_array(records)
    if truth is not None:
        labels = _json_array(truth)
        if len(rows) != len(labels):
            raise ValueError("QReCC records and truth have different lengths")
        merged = []
        for index, (row, label) in enumerate(zip(rows, labels, strict=True)):
            if str(row["Conversation_no"]) != str(label["Conversation_no"]):
                raise ValueError(f"QReCC conversation mismatch at row {index}")
            if str(row.get("Rewrite") or "").strip() != str(
                label.get("Truth_rewrite") or ""
            ).strip():
                raise ValueError(f"QReCC rewrite mismatch at row {index}")
            merged.append(
                {**row, "Passages": list(label.get("Truth_passages") or ())}
            )
        rows = merged
    relevant_doc_ids = {
        str(doc_id) for row in rows for doc_id in row.get("Passages") or ()
    }
    documents_by_id = {}
    for raw_document in iter_jsonl(passages):
        document = Document.from_dict(raw_document)
        if document.doc_id in relevant_doc_ids:
            documents_by_id[document.doc_id] = document.text
    documents = [
        Document(doc_id, documents_by_id[doc_id]) for doc_id in sorted(documents_by_id)
    ]
    queries = []
    feedback = []
    for row in rows:
        conversation = str(row["Conversation_no"])
        turn = int(row["Turn_no"])
        query = str(row.get("Question") or "").strip()
        rewrite = str(row.get("Rewrite") or query).strip()
        relevant = tuple(map(str, row.get("Passages") or ()))
        if not query or not relevant:
            continue
        missing = [doc_id for doc_id in relevant if doc_id not in documents_by_id]
        if missing:
            raise ValueError(f"QReCC passages missing from collection: {missing[:5]}")
        query_id = f"{official_split}-{conversation}-{turn}"
        source_id = f"{official_split}-{conversation}"
        queries.append(QueryExample(query_id, source_id, query, relevant))
        feedback.append(
            FeedbackRecord(
                query_id=query_id,
                source_id=source_id,
                query=query,
                successful_rewrite=rewrite,
                positive_documents=tuple(
                    (doc_id, documents_by_id[doc_id]) for doc_id in relevant
                ),
                timestamp=f"{turn:08d}",
            )
        )
    _write_dataset(
        output,
        "qrecc",
        documents,
        queries,
        feedback,
        {
            "official_split": official_split,
            "group_key": "conversation_id",
            "retrieval_collection": "external full QReCC passage index",
            "source_sha256": {
                "records": file_sha256(records),
                "passages": file_sha256(passages),
                "truth": file_sha256(truth) if truth else None,
            },
        },
    )
