from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from cqr.data.adapters import load_queries
from cqr.data.io import write_json
from cqr.manifests import file_sha256
from cqr.memory.bank import BGEM3MemoryEncoder, MemoryBank
from cqr.policy.prompts import build_rewrite_messages
from cqr.policy.rewards import format_reward, length_reward, retrieval_reward, rewritten_query
from cqr.retrieval.index import load_index


@dataclass(frozen=True)
class GRPOTrainingConfig:
    output_dir: Path
    learning_rate: float = 1e-5
    max_steps: int = 300
    num_generations: int = 8
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 64
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_completion_length: int = 128
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    seed: int = 42
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 5
    use_vllm: bool = True
    vllm_mode: str = "server"
    loss_type: str = "bnpo"
    prompt_style: str = "factorized"
    max_prompt_length: int = 512

    def validate(self) -> None:
        if self.max_steps < 1 or self.num_generations < 2:
            raise ValueError("GRPO requires positive steps and at least two generations")
        effective_batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if effective_batch % self.num_generations:
            raise ValueError("Effective batch size must be divisible by num_generations")
        if self.lora_rank < 1 or self.lora_alpha < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid LoRA configuration")

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        del value["output_dir"]
        return value


def train_grpo(
    model_name_or_path: str,
    tokenizer: object,
    train_dataset: object,
    validation_dataset: object,
    reward_functions: Sequence[Callable[..., list[float]]],
    reward_weights: Sequence[float],
    config: GRPOTrainingConfig,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Run the shared LoRA-GRPO policy optimization used by public experiments."""

    config.validate()
    if len(reward_functions) != len(reward_weights):
        raise ValueError("Each reward function requires one weight")

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        lr_scheduler_type="constant",
        max_steps=config.max_steps,
        num_generations=config.num_generations,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        beta=0.0,
        loss_type=config.loss_type,
        epsilon=0.2,
        reward_weights=list(reward_weights),
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        seed=config.seed,
        data_seed=config.seed,
        bf16=True,
        bf16_full_eval=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        log_completions=True,
        report_to="tensorboard",
    )
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=model_name_or_path,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        reward_funcs=list(reward_functions),
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content") or "").strip()
    raise ValueError(f"Unsupported completion shape: {type(completion).__name__}")


class RetrievalRewardCallbacks:
    def __init__(
        self,
        retriever: object,
        cutoff: int = 25,
        soft_token_limit: int = 64,
        hard_token_limit: int = 128,
    ) -> None:
        self.retriever = retriever
        self.cutoff = cutoff
        self.soft_token_limit = soft_token_limit
        self.hard_token_limit = hard_token_limit

    def retrieval(self, completions: list[Any], **kwargs: Any) -> list[float]:
        rewrites = [
            rewritten_query(completion_text(completion)) or "__invalid_empty_query__"
            for completion in completions
        ]
        rankings = self.retriever.retrieve(rewrites, top_k=self.cutoff)
        return [
            retrieval_reward(tuple(map(str, original)), ranking, relevant, self.cutoff)
            for ranking, original, relevant in zip(
                rankings, kwargs["original_ranking_doc_ids"], kwargs["relevant_doc_ids"],
                strict=True,
            )
        ]

    def format(self, completions: list[Any], **kwargs: Any) -> list[float]:
        return [
            format_reward(str(query), completion_text(completion))
            for query, completion in zip(kwargs["query"], completions, strict=True)
        ]

    def length(
        self,
        completions: list[Any],
        completion_ids: list[list[int]],
        **kwargs: Any,
    ) -> list[float]:
        del completions, kwargs
        return [
            length_reward(
                len(token_ids),
                soft_limit=self.soft_token_limit,
                hard_limit=self.hard_token_limit,
            )
            for token_ids in completion_ids
        ]


def training_rows(
    queries_path: Path,
    memory: MemoryBank,
    memory_top_k: int,
    prompt_style: str = "legacy",
) -> list[dict[str, Any]]:
    queries = load_queries(queries_path)
    rows = []
    for query in queries:
        if not query.original_ranking_doc_ids:
            raise ValueError(f"Training query {query.query_id} has no original ranking")
        retrieved = memory.retrieve(query.query, memory_top_k, source_id=query.source_id)
        for condition, entries in (("policy", ()), ("memory", retrieved)):
            rows.append(
                {
                    "pair_id": query.query_id,
                    "condition": condition,
                    "query": query.query,
                    "relevant_doc_ids": list(query.relevant_doc_ids),
                    "original_ranking_doc_ids": list(query.original_ranking_doc_ids),
                    "memory_entry_ids": [entry.entry_id for entry in entries],
                    "prompt_messages": build_rewrite_messages(query.query, entries, prompt_style),
                }
            )
    return rows


def run_training(
    train_queries: Path,
    validation_queries: Path,
    memory_path: Path,
    retrieval_index: Path,
    model: str,
    embedding_model: str,
    config: GRPOTrainingConfig,
    memory_top_k: int,
    cutoff: int,
    format_weight: float,
    length_weight: float,
    soft_token_limit: int,
    hard_token_limit: int,
    device: str,
    resume_from_checkpoint: str | None = None,
) -> None:
    import datasets
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    memory_bank = MemoryBank.load(
        memory_path,
        BGEM3MemoryEncoder(embedding_model, device=device),
    )

    def dataset(path: Path) -> object:
        rows = training_rows(path, memory_bank, memory_top_k, config.prompt_style)
        for row in rows:
            row["prompt"] = tokenizer.apply_chat_template(
                row.pop("prompt_messages"),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return datasets.Dataset.from_list(rows)

    retriever = load_index(retrieval_index, embedding_model, device)
    rewards = RetrievalRewardCallbacks(
        retriever,
        cutoff=cutoff,
        soft_token_limit=soft_token_limit,
        hard_token_limit=hard_token_limit,
    )
    manifest = {
        "protocol": "cqr-grpo-training-v1",
        "model": Path(model).name if Path(model).exists() else model,
        "training": config.manifest(),
        "train_queries_sha256": file_sha256(train_queries),
        "validation_queries_sha256": file_sha256(validation_queries),
        "memory_sha256": file_sha256(memory_path),
        "retrieval_index_manifest_sha256": file_sha256(
            retrieval_index / "manifest.json"
        ),
        "memory_top_k": memory_top_k,
        "retrieval_cutoff": cutoff,
        "reward_weights": [1.0, format_weight, length_weight],
        "soft_token_limit": soft_token_limit,
        "hard_token_limit": hard_token_limit,
        "conditions": ["policy", "memory"],
        "memory_condition_probability": 0.5,
    }
    manifest_path = config.output_dir / "run_manifest.json"
    if config.output_dir.exists():
        if not resume_from_checkpoint:
            raise FileExistsError(f"Refusing to overwrite {config.output_dir}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Resume configuration differs from the recorded manifest")
    else:
        config.output_dir.mkdir(parents=True)
        write_json(manifest_path, manifest)
    train_grpo(
        model_name_or_path=model,
        tokenizer=tokenizer,
        train_dataset=dataset(train_queries).shuffle(seed=config.seed),
        validation_dataset=dataset(validation_queries),
        reward_functions=(rewards.retrieval, rewards.format, rewards.length),
        reward_weights=(1.0, format_weight, length_weight),
        config=config,
        resume_from_checkpoint=resume_from_checkpoint,
    )
