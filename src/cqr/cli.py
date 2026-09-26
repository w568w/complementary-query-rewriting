from __future__ import annotations

import argparse
import json
from pathlib import Path

from cqr.config import load_config, require
from cqr.data.adapters import load_queries
from cqr.data.difficult_queries import DifficultQueryCriterion, select_difficult_queries
from cqr.data.io import load_jsonl, write_json, write_jsonl
from cqr.data.prepare import prepare_aol4ps, prepare_msmarco, prepare_nfcorpus, prepare_qrecc
from cqr.data.roster import freeze_roster
from cqr.data.splits import grouped_split, split_manifest
from cqr.evaluation.runner import evaluate_rankings, records_from_dicts
from cqr.infer import infer_rankings
from cqr.memory.build import build_memory
from cqr.policy.candidates import generate_candidates
from cqr.policy.train import GRPOTrainingConfig, run_training
from cqr.retrieval.index import build_index, score_original_queries


def prepare_data(args: argparse.Namespace) -> None:
    if args.dataset == "nfcorpus":
        prepare_nfcorpus(args.augmented_queries, args.corpus, args.queries, args.output)
    elif args.dataset == "msmarco":
        prepare_msmarco(args.augmented_queries, args.corpus, args.queries, args.output)
    elif args.dataset == "aol4ps":
        prepare_aol4ps(
            args.interactions,
            args.queries,
            args.documents,
            args.split,
            args.minimum_support,
            args.minimum_confidence,
            args.output,
        )
    elif args.dataset == "qrecc":
        prepare_qrecc(args.records, args.passages, args.split, args.output, args.truth)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def create_index(args: argparse.Namespace) -> None:
    config = load_config(args.common_config, args.dataset_config, tuple(args.override_config))
    if require(config, "retrieval", "backend") != "hybrid":
        raise ValueError("build-index only supports datasets configured for hybrid retrieval")
    build_index(
        args.documents,
        args.output,
        str(require(config, "retrieval", "dense_encoder")),
        args.device,
        args.batch_size,
        float(require(config, "retrieval", "rank_constant")),
    )


def score_original(args: argparse.Namespace) -> None:
    config = load_config(args.common_config, args.dataset_config, tuple(args.override_config))
    if require(config, "retrieval", "backend") != "hybrid":
        raise ValueError("score-original requires the configured hybrid retrieval backend")
    score_original_queries(
        args.queries,
        args.index,
        args.output,
        str(require(config, "retrieval", "dense_encoder")),
        args.device,
        int(require(config, "retrieval", "cutoff")),
    )


def prepare_difficult(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    config = load_config(args.common_config, args.dataset_config, tuple(args.override_config))
    difficulty = require(config, "difficulty")
    metric = str(difficulty["metric"])
    if metric not in {"ndcg", "best_relevant_rank"}:
        raise ValueError("Dataset uses a preconstructed difficult-query roster")
    default_cutoff = int(require(config, "retrieval", "cutoff"))
    criterion = DifficultQueryCriterion(
        metric=metric,
        cutoff=int(difficulty.get("cutoff", default_cutoff)),
        maximum_ndcg=(
            float(difficulty["maximum_ndcg"])
            if difficulty.get("maximum_ndcg") is not None
            else None
        ),
        minimum_rank=(
            int(difficulty["minimum_rank"])
            if difficulty.get("minimum_rank") is not None
            else None
        ),
    )
    selected = select_difficult_queries(load_queries(args.input), criterion)
    splits = grouped_split(selected, args.validation_size, args.test_size, args.seed)
    for split, rows in splits.items():
        write_jsonl(args.output / f"{split}.jsonl", (row.to_dict() for row in rows))
    manifest = split_manifest(splits)
    manifest["difficulty"] = dict(difficulty)
    write_json(args.output / "manifest.json", manifest)


def create_memory(args: argparse.Namespace) -> None:
    config = load_config(args.common_config, args.dataset_config, tuple(args.override_config))
    memory = require(config, "memory")
    build_memory(
        args.feedback,
        args.source_queries,
        args.output,
        args.writer_model,
        args.api_key_env,
        args.base_url,
        int(memory["max_positive_documents"]),
        int(memory["max_retrieved_documents"]),
        int(memory["max_document_characters"]),
        tuple(args.exclude_targets),
        str(memory.get("writer_format", "entities")),
        bool(memory.get("refine_scope", False)),
        args.writer_backend,
        args.device,
    )


def create_roster(args: argparse.Namespace) -> None:
    freeze_roster(args.queries, args.query_ids, args.output)


def train(args: argparse.Namespace) -> None:
    config = load_config(args.common_config, args.dataset_config, tuple(args.override_config))
    training = require(config, "training")
    model = require(config, "model")
    retrieval = require(config, "retrieval")
    memory = require(config, "memory")
    reward = require(config, "reward")
    run_training(
        train_queries=args.train_queries,
        validation_queries=args.validation_queries,
        memory_path=args.memory,
        retrieval_index=args.index,
        model=args.model or str(model["base"]),
        embedding_model=str(retrieval["dense_encoder"]),
        config=GRPOTrainingConfig(
            output_dir=args.output,
            learning_rate=float(training["learning_rate"]),
            max_steps=int(training["max_steps"]),
            num_generations=int(training["num_generations"]),
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            temperature=float(model["temperature"]),
            top_p=float(model["top_p"]),
            top_k=int(model["top_k"]),
            max_completion_length=int(model["max_new_tokens"]),
            lora_rank=int(training["lora_rank"]),
            lora_alpha=int(training["lora_alpha"]),
            lora_dropout=float(training["lora_dropout"]),
            seed=int(training.get("seed", require(config, "evaluation", "seed"))),
            use_vllm=not args.no_vllm,
            vllm_mode=args.vllm_mode,
            loss_type=str(training["loss_type"]),
            prompt_style=str(training["prompt_style"]),
            max_prompt_length=int(training["max_prompt_length"]),
        ),
        memory_top_k=int(memory["top_k"]),
        cutoff=int(retrieval["cutoff"]),
        format_weight=float(reward["format_weight"]),
        length_weight=float(reward["length_weight"]),
        soft_token_limit=int(reward["soft_token_limit"]),
        hard_token_limit=int(reward["hard_token_limit"]),
        device=args.device,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


def generate(args: argparse.Namespace) -> None:
    config = load_config(args.common_config, args.dataset_config, tuple(args.override_config))
    model = require(config, "model")
    generate_candidates(
        queries_path=args.queries,
        output=args.output,
        model=args.model or str(model["base"]),
        adapter=args.adapter,
        condition=args.condition,
        samples=args.samples or int(require(config, "evaluation", "generation_repetitions")),
        seed=int(require(config, "evaluation", "seed")),
        temperature=float(model["temperature"]),
        top_p=float(model["top_p"]),
        top_k=int(model["top_k"]),
        max_new_tokens=int(model["max_new_tokens"]),
        memory_path=args.memory,
        memory_embedding_model=(
            str(require(config, "retrieval", "dense_encoder")) if args.memory else None
        ),
        memory_top_k=int(require(config, "memory", "top_k")),
        device=args.device,
        batch_size=args.batch_size,
        prompt_style=str(model.get("prompt_style", "scope")),
        include_scope=not args.without_scope,
        contexts_path=args.contexts,
    )


def infer(args: argparse.Namespace) -> None:
    config = load_config(args.common_config, args.dataset_config, tuple(args.override_config))
    infer_rankings(
        queries_path=args.queries,
        index=args.index,
        embedding_model=str(require(config, "retrieval", "dense_encoder")),
        policy_candidates=args.policy_candidates,
        memory_candidates=args.memory_candidates,
        output=args.output,
        mode=args.mode,
        method=args.method,
        fusion_method=args.fusion or str(require(config, "fusion", "method")),
        fusion_rank_constant=float(require(config, "fusion", "rank_constant")),
        cutoff=int(require(config, "retrieval", "cutoff")),
        device=args.device,
    )


def evaluate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    config = load_config(args.common_config, args.dataset_config, tuple(args.override_config))
    rows = [row for path in args.rankings for row in load_jsonl(path)]
    cutoffs = tuple(map(int, require(config, "evaluation", "cutoffs")))
    report = {
        "protocol": "cqr-query-averaged-evaluation-v1",
        "aggregation": "samples within query, then queries",
        "document_id_deduplication": True,
        "cutoffs": list(cutoffs),
        "methods": evaluate_rankings(
            records_from_dicts(rows),
            cutoffs=cutoffs,
            draws=int(require(config, "evaluation", "bootstrap_draws")),
            seed=int(config["evaluation"].get("bootstrap_seed", config["evaluation"]["seed"])),
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def _configs(command: argparse.ArgumentParser) -> None:
    command.add_argument("--common-config", type=Path, required=True)
    command.add_argument("--dataset-config", type=Path, required=True)
    command.add_argument("--override-config", type=Path, action="append", default=[])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="cqr")
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data")
    datasets = prepare.add_subparsers(dest="dataset", required=True)
    for name in ("nfcorpus", "msmarco"):
        dataset = datasets.add_parser(name)
        dataset.add_argument("--augmented-queries", type=Path, required=True)
        dataset.add_argument("--corpus", type=Path, required=True)
        dataset.add_argument("--queries", type=Path, required=True)
        dataset.add_argument("--output", type=Path, required=True)
        dataset.set_defaults(handler=prepare_data)
    aol4ps = datasets.add_parser("aol4ps")
    aol4ps.add_argument("--interactions", type=Path, required=True)
    aol4ps.add_argument("--queries", type=Path, required=True)
    aol4ps.add_argument("--documents", type=Path, required=True)
    aol4ps.add_argument("--split", choices=("validation", "test"), required=True)
    aol4ps.add_argument("--minimum-support", type=int, default=2)
    aol4ps.add_argument("--minimum-confidence", type=float, default=0.5)
    aol4ps.add_argument("--output", type=Path, required=True)
    aol4ps.set_defaults(handler=prepare_data)
    qrecc = datasets.add_parser("qrecc")
    qrecc.add_argument("--records", type=Path, required=True)
    qrecc.add_argument("--truth", type=Path)
    qrecc.add_argument("--passages", type=Path, required=True)
    qrecc.add_argument("--split", choices=("train", "test"), required=True)
    qrecc.add_argument("--output", type=Path, required=True)
    qrecc.set_defaults(handler=prepare_data)

    index = commands.add_parser("build-index")
    _configs(index)
    index.add_argument("--documents", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--device", default="cuda")
    index.add_argument("--batch-size", type=int, default=32)
    index.set_defaults(handler=create_index)

    original = commands.add_parser("score-original")
    _configs(original)
    original.add_argument("--queries", type=Path, required=True)
    original.add_argument("--index", type=Path, required=True)
    original.add_argument("--output", type=Path, required=True)
    original.add_argument("--device", default="cuda")
    original.set_defaults(handler=score_original)

    difficult = commands.add_parser("prepare-difficult")
    _configs(difficult)
    difficult.add_argument("--input", type=Path, required=True)
    difficult.add_argument("--output", type=Path, required=True)
    difficult.add_argument("--validation-size", type=int, required=True)
    difficult.add_argument("--test-size", type=int, required=True)
    difficult.add_argument("--seed", type=int, default=42)
    difficult.set_defaults(handler=prepare_difficult)

    roster = commands.add_parser("freeze-roster")
    roster.add_argument("--queries", type=Path, required=True)
    roster.add_argument("--query-ids", type=Path, required=True)
    roster.add_argument("--output", type=Path, required=True)
    roster.set_defaults(handler=create_roster)

    memory = commands.add_parser("build-memory")
    _configs(memory)
    memory.add_argument("--feedback", type=Path, required=True)
    memory.add_argument("--source-queries", type=Path)
    memory.add_argument("--exclude-targets", type=Path, action="append", default=[],
                        help="Exclude these source IDs")
    memory.add_argument("--writer-model", required=True)
    memory.add_argument("--writer-backend", choices=("transformers", "openai"), default="transformers")
    memory.add_argument("--device", default="cuda")
    memory.add_argument("--api-key-env", default="OPENAI_API_KEY")
    memory.add_argument("--base-url")
    memory.add_argument("--output", type=Path, required=True)
    memory.set_defaults(handler=create_memory)

    training = commands.add_parser("train")
    _configs(training)
    training.add_argument("--train-queries", type=Path, required=True)
    training.add_argument("--validation-queries", type=Path, required=True)
    training.add_argument("--memory", type=Path, required=True)
    training.add_argument("--index", type=Path, required=True)
    training.add_argument("--model")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--device", default="cuda")
    training.add_argument("--per-device-batch-size", type=int, default=2)
    training.add_argument("--gradient-accumulation-steps", type=int, default=64)
    training.add_argument("--vllm-mode", choices=("server", "colocate"), default="server")
    training.add_argument("--no-vllm", action="store_true")
    training.add_argument("--resume-from-checkpoint")
    training.set_defaults(handler=train)

    generation = commands.add_parser("generate")
    _configs(generation)
    generation.add_argument("--queries", type=Path, required=True)
    generation.add_argument("--condition", choices=("policy", "memory"), required=True)
    generation.add_argument("--model")
    generation.add_argument("--adapter")
    generation.add_argument("--memory", type=Path)
    generation.add_argument("--contexts", type=Path,
                            help="Frozen query/messages/similarity records for matched controls")
    generation.add_argument("--samples", type=int)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--device", default="cuda")
    generation.add_argument("--batch-size", type=int, default=4)
    generation.add_argument("--without-scope", action="store_true")
    generation.set_defaults(handler=generate)

    inference = commands.add_parser("infer")
    _configs(inference)
    inference.add_argument("--queries", type=Path, required=True)
    inference.add_argument("--index", type=Path, required=True)
    inference.add_argument("--policy-candidates", type=Path, required=True)
    inference.add_argument("--memory-candidates", type=Path)
    inference.add_argument("--mode", choices=("rl", "full"), required=True)
    inference.add_argument("--method", required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--device", default="cuda")
    inference.add_argument("--fusion", choices=("adaptive", "equal", "similarity"))
    inference.set_defaults(handler=infer)

    evaluation = commands.add_parser("evaluate")
    _configs(evaluation)
    evaluation.add_argument("--rankings", type=Path, nargs="+", required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.set_defaults(handler=evaluate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
