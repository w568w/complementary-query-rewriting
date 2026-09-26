from __future__ import annotations

import json
import re


def q2e_prompt(query: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"Write a list of keywords for the following query: {query}"}]


def q2e_retrieval_query(query: str, output: str, maximum_keywords: int = 20) -> str:
    keywords = []
    for part in re.split(r"[,\n]", output):
        keyword = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", part.strip()).strip("\"'`.,;:")
        if keyword and keyword.casefold() not in {value.casefold() for value in keywords}:
            keywords.append(keyword)
        if len(keywords) == maximum_keywords:
            break
    return " ".join([query] * 5 + keywords) if keywords else query


def spqe_prompt(query: str, dataset: str) -> list[dict[str, str]]:
    if dataset == "nfcorpus":
        instruction = "Write a short biomedical passage to answer this question."
        label = "Question"
    elif dataset == "msmarco":
        instruction = "Write a short web passage that directly answers this search query."
        label = "Search query"
    else:
        raise ValueError(f"Unsupported SPQE dataset: {dataset}")
    return [{"role": "user", "content": f"{instruction}\n{label}: {query}\nPassage:"}]


def sage_sgp_prompt(query: str) -> list[dict[str, str]]:
    content = f"""Rewrite the query for document retrieval by selecting one strategy:
1. Semantic expansion
2. Entity disambiguation
3. Sub-question decomposition
4. Concise rewriting
5. Neutralized claim reformulation

Return JSON inside <answer> tags:
<answer>{{"query": "...", "strategy": 1}}</answer>

Input: {query}"""
    return [{"role": "user", "content": content}]


def parse_sage_sgp(output: str) -> tuple[str, int]:
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", output, flags=re.DOTALL)
    if match is None:
        raise ValueError("SAGE-SGP output has no answer block")
    value = json.loads(match.group(1))
    query = " ".join(str(value.get("query") or "").split())
    strategy = value.get("strategy")
    if not query or not isinstance(strategy, int) or not 1 <= strategy <= 5:
        raise ValueError("SAGE-SGP output has an invalid schema")
    return query, strategy
