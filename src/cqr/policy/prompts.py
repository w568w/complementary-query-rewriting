from __future__ import annotations

from typing import Iterable

from cqr.schemas import MemoryEntry, canonical_json


REWRITE_RULES = """1. Basic Norms
- Correct spelling errors and standardize terminology.

2. Semantic Enhancement
- Convert colloquial expressions to formal terms.
- Expand abbreviations only when supported by the query or supplied memory.
- Add missing intent or action terms when needed.

3. Search Optimization
- Use concise corpus-aligned terminology.
- Return one revised query, without explanation."""

EXPANSION_INSTRUCTION = (
    "Use at most 40 words and at most five distinct expansion terms. Do not repeat synonymous phrases or list every possible association.\n\n"
    "Rewrite the query to retrieve relevant documents. Normalize spelling and phrasing, then enrich it with useful "
    "technical terms from the historical knowledge: canonical names, abbreviations, categories, examples, or mechanisms "
    "relevant to the information sought. Retain specific technical vocabulary rather than paraphrasing it into generic words. "
    "Broader and narrower related terms may be added as search expansion; they need not be exact synonyms. "
    "Use only relevant knowledge. Return one compact expanded search query, not an answer or explanation."
)
SCOPE_GUIDANCE = (
    "Scope describes where an entity meaning applies and may guide compatible query expansion. "
    "Use it as positive contextual evidence; do not treat it as a prohibition or replace the current topic.\n\n"
)


def build_expansion_messages(
    query: str,
    context: str,
    use_scope: bool = True,
) -> list[dict[str, str]]:
    guidance = SCOPE_GUIDANCE if use_scope else ""
    sections = [
        EXPANSION_INSTRUCTION,
        f"{guidance}Historical knowledge:\n{context}",
        f"Original query: {query}",
        "Expanded search query:",
    ]
    return [{"role": "user", "content": "\n\n".join(sections)}]


def build_rewrite_messages(
    query: str,
    memory: Iterable[MemoryEntry] = (),
    style: str = "scope",
    include_scope: bool = True,
) -> list[dict[str, str]]:
    entries = list(memory)
    if style in {"expansion", "scope"}:
        context = canonical_json(
            [entity.context_dict(include_scope) for entry in entries for entity in entry.entities]
        )
        return build_expansion_messages(query, context, use_scope=style == "scope")
    if style not in {"legacy", "factorized", "recurrent"}:
        raise ValueError(f"Unknown prompt style: {style}")
    sections = [
        "You are a query optimization module. Rewrite the original query for document retrieval.",
        REWRITE_RULES,
    ]
    if entries:
        memory_text = "\n".join(entry.context_text() for entry in entries if entry.context_text())
        if memory_text:
            sections.extend(
                [
                    (
                        "The following history comes only from an earlier query and its "
                        "observed feedback. It may be irrelevant. Preserve the current query's "
                        "subject and intent, and ignore the history when it does not apply."
                        if style == "recurrent"
                        else "The following domain memory was extracted only from training-query "
                        "failures. It may be irrelevant; use only supported entries."
                    ),
                    memory_text,
                ]
            )
    sections.extend([f"Original query: {query}", "Revised query:"])
    return [{"role": "user", "content": "\n\n".join(sections)}]
