from pathlib import Path

from cqr.data.io import load_jsonl
from cqr.schemas import Document, FeedbackRecord, QueryExample


def load_queries(path: Path) -> list[QueryExample]:
    return [QueryExample.from_dict(row) for row in load_jsonl(path)]


def load_documents(path: Path) -> list[Document]:
    return [Document.from_dict(row) for row in load_jsonl(path)]


def load_feedback(path: Path) -> list[FeedbackRecord]:
    records = [FeedbackRecord.from_dict(row) for row in load_jsonl(path)]
    if any(not record.positive_documents for record in records):
        raise ValueError("Every reusable feedback record requires a positive document")
    return records
