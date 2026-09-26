from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Protocol

from cqr.schemas import EntityDefinition, FeedbackRecord, MemoryEntry

ENTITY_INSTRUCTION = (
    "Extract reusable terminology from the historical retrieval evidence. Return only JSON: "
    '{"entities":[{"entity":"term","definition":"brief document-supported meaning","synonyms":["supported alternate term"],"scope":"context where that meaning applies"}]}. '
    "Use at most four entities. A synonym or definition must be supported by the historical document, not guessed from general knowledge. "
    "Scope describes the meaning of that entity; it must not instruct future users to exclude terms or change their questions. "
    "Do not output a rewritten query or an instruction for future rewriting. Return an empty entities array when nothing is supported."
)
SCOPE_INSTRUCTION = (
    "Refine the scope of each existing memory entity using the historical feedback document. "
    "Do not change, remove, or add entities, definitions, or synonyms. "
    "A scope records the concrete documented context in which an entity is discussed: its role, "
    "application, related object or intervention, population, setting, or outcome when stated. "
    "Retain information-bearing terminology from the document title and main findings when it is "
    "actually relevant to that entity. Broad labels such as health or research alone are insufficient. "
    "Do not invent relationships, turn scope into rewriting instructions, or impose prohibitions on future queries. "
    "The historical query is context, not a requirement to repeat its wording. "
    'Return JSON {"updates":[{"index":0,"scope":"at most 32 words","evidence":"a short exact contiguous quotation from the document"}]}. '
    "Only propose updates supported by the document. Omit an entity when no supported refinement is available. "
    "Evidence must be copied exactly; scope should be a concise factual description, not an answer to the query."
)


MEMORY_WRITER_PROMPT = """Convert retrieval feedback into reusable query-rewriting memory.

Compare the original query, its successful rewrite, relevant documents, and
documents retrieved by the original query. Return one JSON object with exactly
these fields:
- "reusable": a boolean indicating whether the feedback supports knowledge that
  can help queries beyond this exact instance;
- "entities": a list of objects with non-empty "name" and "description" fields;
- "constraints": a list of concise strings delimiting the applicable semantic
  scope, document type, or terminology.

Extract only document-supported knowledge. Do not copy document passages and do
not produce another rewritten query.

Original query:
{query}

Successful rewrite:
{rewrite}

Relevant documents:
{positive_documents}

Documents retrieved by the original query:
{retrieved_documents}
"""


class JsonCompletionClient(Protocol):
    def complete_json(self, prompt: str, model: str) -> dict[str, Any]: ...

    def complete_messages(
        self, messages: list[dict[str, str]], model: str, max_new_tokens: int = 512
    ) -> dict[str, Any]: ...


@dataclass
class OpenAIJsonCompletionClient:
    client: Any

    @classmethod
    def create(cls, api_key: str, base_url: str | None = None) -> OpenAIJsonCompletionClient:
        from openai import OpenAI

        return cls(OpenAI(api_key=api_key, base_url=base_url))

    def complete_json(self, prompt: str, model: str) -> dict[str, Any]:
        return self.complete_messages([{"role": "user", "content": prompt}], model)

    def complete_messages(
        self, messages: list[dict[str, str]], model: str, max_new_tokens: int = 512
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_new_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Memory writer returned empty content")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Memory writer output must be a JSON object")
        return value


class TransformersJsonCompletionClient:
    def __init__(self, model: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device,
        ).eval()

    def complete_json(self, prompt: str, model: str) -> dict[str, Any]:
        return self.complete_messages([{"role": "user", "content": prompt}], model)

    def complete_messages(
        self, messages: list[dict[str, str]], model: str, max_new_tokens: int = 512
    ) -> dict[str, Any]:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Memory writer output must be a JSON object")
        return value


@dataclass(frozen=True)
class MemoryWriter:
    client: JsonCompletionClient
    model: str
    max_positive_documents: int = 3
    max_retrieved_documents: int = 3
    max_document_characters: int = 1_500
    writer_format: str = "entities"
    refine_scope: bool = False

    def _documents(self, documents: tuple[tuple[str, str], ...], limit: int) -> str:
        values = [
            {"doc_id": doc_id, "text": text[: self.max_document_characters]}
            for doc_id, text in documents[:limit]
        ]
        return json.dumps(values, ensure_ascii=False)

    def extract(self, feedback: FeedbackRecord) -> MemoryEntry | None:
        if not feedback.positive_documents:
            raise ValueError("Positive feedback is required to construct memory")
        if self.writer_format == "entities":
            return self.extract_entities(feedback)
        if self.writer_format != "legacy":
            raise ValueError("Unknown writer format")
        prompt = MEMORY_WRITER_PROMPT.format(
            query=feedback.query,
            rewrite=feedback.successful_rewrite,
            positive_documents=self._documents(
                feedback.positive_documents, self.max_positive_documents
            ),
            retrieved_documents=self._documents(
                feedback.retrieved_documents, self.max_retrieved_documents
            ),
        )
        output = self.client.complete_json(prompt, self.model)
        if output.get("reusable") is not True:
            return None
        entities = tuple(
            EntityDefinition.from_dict(value) for value in output.get("entities") or ()
        )
        constraints = tuple(str(value).strip() for value in output.get("constraints") or ())
        if any(not value for value in constraints):
            raise ValueError("Memory constraints must be non-empty")
        if not entities and not constraints:
            raise ValueError("Reusable memory must contain entities or constraints")
        return MemoryEntry.from_feedback(
            feedback=feedback,
            entities=entities,
            constraints=constraints,
            writer_model=self.model,
        )

    def extract_entities(self, feedback: FeedbackRecord) -> MemoryEntry | None:
        document = "\n\n".join(
            text for _, text in feedback.positive_documents[: self.max_positive_documents]
        )
        payload = {
            "history_query": feedback.query,
            "best_rewrite": feedback.successful_rewrite,
            "clicked_document": document,
        }
        output = self.client.complete_messages(
            [
                {"role": "system", "content": ENTITY_INSTRUCTION},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            self.model,
        )
        entities = tuple(EntityDefinition.from_dict(value) for value in output["entities"])
        if len(entities) > 4:
            raise ValueError("Entity extractor returned more than four entries")
        if not entities:
            return None
        if self.refine_scope:
            payload = {
                "history_query": feedback.query,
                "history_rewrite": feedback.successful_rewrite,
                "entities": [entity.context_dict() for entity in entities],
                "feedback_document": document,
            }
            try:
                response = self.client.complete_messages(
                    [
                        {"role": "system", "content": SCOPE_INSTRUCTION},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    self.model,
                    max_new_tokens=768,
                )
                updated = list(entities)
                seen = set()
                for update in response["updates"]:
                    index, scope, quote = update["index"], update["scope"], update["evidence"]
                    if type(index) is not int or not 0 <= index < len(entities) or index in seen:
                        raise ValueError("Invalid scope update index")
                    seen.add(index)
                    if not isinstance(scope, str) or not isinstance(quote, str):
                        continue
                    normalized = " ".join(quote.split()).casefold()
                    if (
                        scope.strip()
                        and len(scope.split()) <= 32
                        and normalized
                        and normalized in " ".join(document.split()).casefold()
                    ):
                        updated[index] = replace(entities[index], scope=scope.strip())
                entities = tuple(updated)
            except (ValueError, KeyError, TypeError):
                pass  # Malformed scope updates keep the previously extracted scopes.
        return MemoryEntry.from_feedback(feedback, entities, (), self.model)
