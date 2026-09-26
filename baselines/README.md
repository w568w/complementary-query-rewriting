# Baselines

Baseline adapters produce the same candidate and ranking schemas as the main
method.

| Family | Methods |
| --- | --- |
| Prompt-based | Qwen3-8B zero-shot, Query2Doc, HyDE, SAGE-SGP, Q2E, SPQE |
| Trained | OPQE, DeepRetrieval, SAGE-SCS |
| Iterative | ADoRE |

`registry.py` records upstream sources; `prompts.py` contains shared prompt and
output adapters.
