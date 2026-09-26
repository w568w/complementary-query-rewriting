from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    name: str
    family: str
    upstream: str
    release_component: str


BASELINES = (
    Baseline("Qwen3-8B zero-shot", "prompt", "Qwen/Qwen3-8B", "shared rewriter"),
    Baseline("Query2Doc", "prompt", "https://aclanthology.org/2023.emnlp-main.585/", "prompt adapter"),
    Baseline("HyDE", "prompt", "https://github.com/texttron/hyde", "embedding adapter"),
    Baseline("SAGE-SGP", "prompt", "https://arxiv.org/abs/2506.19783", "prompt and parser"),
    Baseline("Q2E", "prompt", "https://arxiv.org/abs/2305.03653", "prompt and parser"),
    Baseline("SPQE", "prompt", "https://github.com/BaleChen/RethinkQAug", "prompt adapter"),
    Baseline("OPQE", "trained", "https://github.com/BaleChen/RethinkQAug", "training adapter"),
    Baseline("DeepRetrieval", "trained", "https://github.com/pat-jj/DeepRetrieval", "training adapter"),
    Baseline("SAGE-SCS", "trained", "https://arxiv.org/abs/2506.19783", "training reproduction"),
    Baseline("ADoRE", "iterative", "https://github.com/aminbigdeli/ADORE", "iterative adapter"),
)
