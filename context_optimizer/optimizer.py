"""Prompt sandwich construction and token-budget-aware context pruning."""
from __future__ import annotations
import re
from .tokenizer import TokenCodec, count_tokens, get_codec

_STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "does", "are", "is", "of", "to", "in", "a", "an", "on", "it", "be", "by", "as"}


def relevance_score(chunk: str, query: str) -> float:
    terms = {word.lower() for word in re.findall(r"[a-zA-Z0-9_-]+", query) if word.lower() not in _STOPWORDS}
    words = [word.lower() for word in re.findall(r"[a-zA-Z0-9_-]+", chunk)]
    if not terms or not words:
        return 0.0
    return sum(1 for word in words if word in terms) / max(len(terms), 1)


class ContextOptimizer:
    def __init__(self, max_tokens: int = 4000, tokenizer: TokenCodec | None = None) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero")
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer or get_codec()

    def count_tokens(self, text: str) -> int:
        return count_tokens(text, self.tokenizer)

    def construct_sandwich_prompt(
        self, system_instruction: str, dynamic_context: str, user_query: str,
        critical_fact: str | None = None,
    ) -> str:
        priority = critical_fact if critical_fact is not None else system_instruction
        return (
            f"{system_instruction}\n\n"
            f"=== HIGH PRIORITY CONTEXT (BEGINNING) ===\n{priority}\n\n"
            f"=== SUPPORTING CONTEXT ===\n{dynamic_context}\n\n"
            f"=== HIGH PRIORITY REINFORCEMENT ===\n{priority}\n\n"
            f"=== USER QUERY ===\n{user_query}"
        )

    def prune_context(self, context_chunks: list[str], target_max_tokens: int,
                      query: str = "") -> list[str]:
        if target_max_tokens < 0:
            raise ValueError("target_max_tokens must be non-negative")
        kept = list(enumerate(context_chunks))
        def size(items: list[tuple[int, str]]) -> int:
            return self.count_tokens("\n".join(chunk for _, chunk in items))
        while kept and size(kept) > target_max_tokens:
            middle = len(kept) // 2
            # Preserve the original edge chunks when possible; otherwise drop lowest relevance.
            if len(kept) > 2:
                removable = range(1, len(kept) - 1)
                victim = min(removable, key=lambda idx: (relevance_score(kept[idx][1], query), abs(idx - middle)))
            else:
                victim = min(range(len(kept)), key=lambda idx: relevance_score(kept[idx][1], query))
            kept.pop(victim)
        return [chunk for _, chunk in kept]

    def optimize_payload(self, system_prompt: str, critical_fact: str,
                         context_chunks: list[str], user_query: str,
                         budget_tokens: int | None = None) -> tuple[str, list[str], list[int]]:
        budget = budget_tokens if budget_tokens is not None else self.max_tokens
        if budget < 1:
            raise ValueError("budget_tokens must be greater than zero")
        original = list(context_chunks)
        kept = list(enumerate(original))
        def render(items: list[tuple[int, str]]) -> str:
            context = "\n\n".join(chunk for _, chunk in items)
            return self.construct_sandwich_prompt(system_prompt, context, user_query, critical_fact)
        prompt = render(kept)
        if self.count_tokens(render([])) > budget:
            raise ValueError("System instruction, critical fact, and query alone exceed the token budget.")
        while kept and self.count_tokens(prompt) > budget:
            middle = len(kept) // 2
            choices = range(1, len(kept) - 1) if len(kept) > 2 else range(len(kept))
            victim = min(choices, key=lambda idx: (relevance_score(kept[idx][1], user_query + " " + critical_fact), abs(idx - middle)))
            kept.pop(victim)
            prompt = render(kept)
        if self.count_tokens(prompt) > budget:
            raise ValueError("Prompt cannot fit the budget without removing required system/query content.")
        dropped = sorted(set(range(len(original))) - {index for index, _ in kept})
        return prompt, [chunk for _, chunk in kept], dropped
