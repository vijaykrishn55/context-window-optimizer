"""Token accounting and context-region diagnostics."""
from __future__ import annotations

from .schemas import AllocationReport
from .tokenizer import TokenCodec, count_tokens, get_codec


class ContextWindowAnalyzer:
    def __init__(self, model_name: str = "gpt-4o", tokenizer: TokenCodec | None = None) -> None:
        self.model_name = model_name
        self.tokenizer = tokenizer or get_codec(model_name)

    def count_tokens(self, text: str) -> int:
        return count_tokens(text, self.tokenizer)

    def audit_allocation(
        self, system_prompt: str, context_chunks: list[str], user_query: str,
        max_limit: int = 4000,
    ) -> dict:
        if max_limit < 1:
            raise ValueError("max_limit must be greater than zero")
        chunks = [self.count_tokens(chunk) for chunk in context_chunks]
        system_count = self.count_tokens(system_prompt)
        query_count = self.count_tokens(user_query)
        context_count = sum(chunks)
        total = system_count + context_count + query_count
        # Chunk center is flagged for inspection, not asserted to be ignored.
        total_context = max(context_count, 1)
        high_risk = []
        cursor = 0
        for index, count in enumerate(chunks):
            midpoint = (cursor + count / 2) / total_context
            if 0.2 <= midpoint <= 0.8:
                high_risk.append(index)
            cursor += count
        report = AllocationReport(
            tokenizer=self.tokenizer.name, max_tokens=max_limit,
            system_tokens=system_count, context_chunk_tokens=chunks,
            total_context_tokens=context_count, query_tokens=query_count,
            total_tokens=total, remaining_tokens=max_limit - total,
            utilization_percent=round(total / max_limit * 100, 2),
            status="within_budget" if total <= max_limit else "exceeds_budget",
            high_risk_chunk_indexes=high_risk,
        )
        return report.model_dump()
