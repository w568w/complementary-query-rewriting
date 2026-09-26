# Reproduce

## 1. Replay the table without a GPU

From the repository root:

```bash
uv run python experiments/component_ablation.py \
  --query-scores artifacts/results/ablation/nfcorpus_query_scores.jsonl \
  --output outputs/ablation_nf.json

uv run python experiments/component_ablation.py \
  --query-scores artifacts/results/ablation/marco_query_scores.jsonl \
  --output outputs/ablation_ms.json
```

## 2. New runs and original-run dependencies

Acquire datasets yourself as described in DATA.md. The full normalized pipeline
is `prepare-data -> build-index -> score-original -> prepare-difficult ->
build-memory -> train -> generate -> infer -> evaluate`.
Use `cqr COMMAND --help` for input schemas and arguments.

A main-table memory bank uses training sources only; validation/test sources
are excluded. For simulated-history ablations,
supply prior-feedback memory from the held-out-source history partition.

Example new memory build:

```bash
uv run cqr build-memory \
  --common-config configs/common.toml --dataset-config configs/nfcorpus/dataset.toml \
  --override-config configs/ablation.toml \
  --feedback data/history.jsonl --writer-model Qwen/Qwen3-8B \
  --writer-backend transformers --output outputs/memory
```

This uses greedy local extraction, entity-level scopes and offline
evidence-supported scope refinement. Missing or unsupported scope updates
retain the previous scope.

## 3. Matched ablation generation

Freeze contexts first. Replace memory with random, no_scope, full_documents,
direct_documents, or rl for the corresponding arm:

```bash
uv run python experiments/prepare_ablation.py \
  --queries data/ablation_queries.jsonl --memory outputs/memory/memory.jsonl \
  --feedback data/history.jsonl --dataset nfcorpus --condition memory \
  --output outputs/memory_contexts.jsonl

uv run cqr generate \
  --common-config configs/common.toml --dataset-config configs/nfcorpus/dataset.toml \
  --override-config configs/ablation.toml \
  --queries data/ablation_queries.jsonl --condition memory \
  --contexts outputs/memory_contexts.jsonl --adapter checkpoints/policy \
  --output outputs/memory_candidates
```

`full_documents` supplies the complete positive document of the selected
history entry. This is the control in the released table. `direct_documents`
instead searches the deduplicated historical document content and is a separate
control.
`no_scope` removes only entity scope fields; `random` selects a different
eligible donor deterministically and recomputes that donor's key similarity.

Generate policy candidates with `--condition policy` and the RL contexts.
Then fuse the matched candidates:

```bash
uv run cqr infer \
  --common-config configs/common.toml --dataset-config configs/nfcorpus/dataset.toml \
  --override-config configs/ablation.toml \
  --queries data/ablation_queries.jsonl --index outputs/index \
  --policy-candidates outputs/rl_candidates --memory-candidates outputs/memory_candidates \
  --mode full --method fusion_memory --output outputs/full
```

Use `--fusion equal` for equal-weight RRF and `--fusion similarity` for the
similarity-only control.

## 4. AOL4PS history prefixes

```bash
uv run python experiments/prepare_history.py \
  --interactions data/aol/data.csv --queries data/aol/query.csv \
  --documents data/aol/doc.csv --output outputs/aol_prefixes
```

Events are grouped before prefixes are
counted and ordered by (query_time, event_id). Default fractions are
0, 0.25, 0.5, 0.75 and 1.
Use one fixed evaluation roster for all prefixes, then aggregate
the measured scores with `history_growth.py`.
