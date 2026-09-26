from __future__ import annotations

from typing import Iterable, Protocol, Sequence

from cqr.policy.prompts import build_rewrite_messages
from cqr.schemas import MemoryEntry


class Rewriter(Protocol):
    def rewrite(
        self,
        query: str,
        memory: Sequence[MemoryEntry] = (),
        samples: int = 1,
    ) -> list[str]: ...

    def rewrite_batch(
        self,
        queries: Sequence[str],
        memories: Sequence[Sequence[MemoryEntry]],
        samples: int = 1,
    ) -> list[list[str]]: ...


class TransformersRewriter:
    def __init__(
        self,
        model: object,
        tokenizer: object,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 128,
        prompt_style: str = "scope",
        include_scope: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.prompt_style = prompt_style
        self.include_scope = include_scope

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        adapter_path: str | None = None,
        **generation: object,
    ) -> TransformersRewriter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype="auto",
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        return cls(model=model, tokenizer=tokenizer, **generation)

    def rewrite(
        self,
        query: str,
        memory: Sequence[MemoryEntry] = (),
        samples: int = 1,
    ) -> list[str]:
        return self.rewrite_batch([query], [memory], samples=samples)[0]

    def rewrite_batch(
        self,
        queries: Sequence[str],
        memories: Sequence[Sequence[MemoryEntry]],
        samples: int = 1,
    ) -> list[list[str]]:
        if samples < 1 or not queries or len(queries) != len(memories):
            raise ValueError("Invalid batch or sample count")
        messages = [
            build_rewrite_messages(query, memory, self.prompt_style, self.include_scope)
            for query, memory in zip(queries, memories, strict=True)
        ]
        return self.rewrite_messages_batch(messages, samples)

    def rewrite_messages_batch(
        self,
        messages: Sequence[list[dict[str, str]]],
        samples: int = 1,
    ) -> list[list[str]]:
        if samples < 1 or not messages:
            raise ValueError("Invalid batch or sample count")
        prompts = [
            self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for message in messages
        ]
        inputs = self.tokenizer(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=samples,
        )
        prompt_length = inputs["input_ids"].shape[1]
        decoded = [
            self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
            for output in outputs
        ]
        return [decoded[index * samples : (index + 1) * samples] for index in range(len(messages))]


def generate_rewrites(
    rewriter: Rewriter,
    queries: Iterable[str],
    samples: int,
) -> list[list[str]]:
    return [rewriter.rewrite(query, samples=samples) for query in queries]
