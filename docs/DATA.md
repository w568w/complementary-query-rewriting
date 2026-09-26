# Data

This repository does not download or redistribute datasets. Obtain each dataset
from its official source and pass the extracted files to `cqr prepare-data`.

| Dataset | Source | Expected inputs | Use |
| --- | --- | --- | --- |
| NFCorpus | [BEIR](https://github.com/beir-cellar/beir) | `corpus.jsonl`, `queries.jsonl` | Main and recurrent evaluation |
| MS MARCO | [Microsoft](https://github.com/microsoft/MSMARCO-Passage-Ranking) | normalized passage/query parquet | Main and recurrent evaluation |
| AOL4PS | [Science Data Bank](https://doi.org/10.11922/sciencedb.j00104.00093) | interaction, query, and document TSV files | Chronological memory and click-noise analysis |

## Normalized records

Documents use:

```json
{"doc_id": "d1", "text": "document text"}
```

Queries use:

```json
{
  "query_id": "q1",
  "source_id": "source, user, or conversation ID",
  "query": "original query",
  "relevant_doc_ids": ["d1"],
  "original_ranking_doc_ids": ["d4", "d1"]
}
```

Feedback records contain the original query, successful rewrite, positive
documents, optional retrieved documents, and an optional timestamp. Memory is
built from historical or training sources.

NFCorpus and MS MARCO use the local BGE-M3+BM25 hybrid index. AOL4PS uses its
provided candidate rankings.
