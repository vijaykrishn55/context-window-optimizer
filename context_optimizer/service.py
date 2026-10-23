"""Application services used by CLI commands."""
from __future__ import annotations
import json
from pathlib import Path
from .analyzer import ContextWindowAnalyzer
from .chunking import sliding_window_chunks
from .optimizer import ContextOptimizer
from .schemas import OptimizationReport
from .simulation import run_position_simulation
from .tokenizer import get_codec


def read_request(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("system_prompt", "context_chunks", "user_query"):
        if key not in data:
            raise ValueError(f"Input JSON must include '{key}'.")
    if not isinstance(data["context_chunks"], list):
        raise ValueError("context_chunks must be a JSON array of strings.")
    return data


def analyze_request(data: dict, budget: int) -> dict:
    analyzer = ContextWindowAnalyzer(model_name=data.get("tokenizer_model", "gpt-4o"))
    return analyzer.audit_allocation(data["system_prompt"], data["context_chunks"], data["user_query"], budget)


def optimize_request(data: dict, budget: int) -> tuple[str, dict]:
    analyzer = ContextWindowAnalyzer(model_name=data.get("tokenizer_model", "gpt-4o"))
    optimizer = ContextOptimizer(max_tokens=budget, tokenizer=analyzer.tokenizer)
    before_prompt = optimizer.construct_sandwich_prompt(
        data["system_prompt"], "\n\n".join(data["context_chunks"]), data["user_query"], data.get("critical_fact", data["system_prompt"]))
    before = analyzer.count_tokens(before_prompt)
    prompt, kept, dropped = optimizer.optimize_payload(
        data["system_prompt"], data.get("critical_fact", data["system_prompt"]),
        data["context_chunks"], data["user_query"], budget)
    after = analyzer.count_tokens(prompt)
    report = OptimizationReport(
        tokenizer=analyzer.tokenizer.name, budget_tokens=budget, input_tokens=before,
        output_tokens=after, retained_chunks=len(kept), dropped_chunks=len(dropped),
        dropped_chunk_indexes=dropped, status="within_budget" if after <= budget else "exceeds_budget",
        notes=["System instructions, critical fact, and final query are preserved.",
               f"Dropped context chunk indexes: {dropped}.",
               "Counts are estimates for the selected tiktoken encoding; provider token accounting can differ."])
    return prompt, report.model_dump()


def split_document(text: str, chunk_tokens: int, overlap_ratio: float) -> list[str]:
    return sliding_window_chunks(text, chunk_tokens=chunk_tokens, overlap_ratio=overlap_ratio, tokenizer=get_codec())


def simulate_document(text: str, needle: str, query: str, **kwargs) -> dict:
    return run_position_simulation(text, needle, query, **kwargs)
