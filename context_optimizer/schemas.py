"""Typed report contracts used by the CLI and JSON exports."""
from typing import Any
from pydantic import BaseModel, Field


class AllocationReport(BaseModel):
    tokenizer: str
    max_tokens: int
    system_tokens: int
    context_chunk_tokens: list[int]
    total_context_tokens: int
    query_tokens: int
    total_tokens: int
    remaining_tokens: int
    utilization_percent: float
    status: str
    high_risk_chunk_indexes: list[int] = Field(default_factory=list)


class PositionResult(BaseModel):
    position_percent: int
    retrieved: bool
    answer: str = ""
    error: str | None = None


class SimulationReport(BaseModel):
    model: str
    mode: str
    needle: str
    query: str
    positions: list[PositionResult]
    successful_runs: int
    total_runs: int
    accuracy_percent: float
    note: str


class OptimizationReport(BaseModel):
    tokenizer: str
    budget_tokens: int
    input_tokens: int
    output_tokens: int
    retained_chunks: int
    dropped_chunks: int
    dropped_chunk_indexes: list[int] = Field(default_factory=list)
    status: str
    notes: list[str] = Field(default_factory=list)
