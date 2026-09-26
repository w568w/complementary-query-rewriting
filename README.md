# Complementary Query Rewriting

This repository implements difficult-query rewriting through two complementary
learning paths: retrieval-oriented policy optimization and reusable semantic
memory extracted from historical retrieval feedback.

## Method pipeline

```text
data -> difficult queries -> feedback memory -> GRPO policy
                                      |              |
                                      +-> memory expert
                                                     |
                                  policy expert -----+-> adaptive RRF -> ranking
```

## Installation

Python 3.11 and `uv` are recommended.

```bash
uv sync --extra data --extra retrieval --extra training --extra writer
```

GPU-specific PyTorch wheels must match the target CUDA environment.

## Reproduction

Download the datasets from their official sources and prepare the expected
inputs described in [DATA.md](docs/DATA.md). The complete NFCorpus/MS MARCO
workflow is documented in [REPRODUCE.md](docs/REPRODUCE.md).

```text
prepare-data -> build-index -> score-original -> prepare-difficult
             -> build-memory -> train -> generate -> infer -> evaluate
```

## Layout

```text
src/cqr/      Core method and evaluation implementation
configs/      Dataset-independent and dataset-specific settings
baselines/    Baseline registry and shared prompt adapters
experiments/  Paper-level aggregation scripts
artifacts/    Sanitized aggregate results and release manifests
examples/     Synthetic input examples
```
