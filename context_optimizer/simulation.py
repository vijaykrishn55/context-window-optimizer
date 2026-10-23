"""Needle-in-a-haystack testing across context-depth intervals."""
from __future__ import annotations
from .gateway import LLMGateway
from .schemas import PositionResult, SimulationReport
from .tokenizer import TokenCodec, get_codec

DEFAULT_DEPTHS = (0, 25, 50, 75, 100)


def _insert_at_depth(base_tokens: list[int], needle_tokens: list[int], depth: int) -> list[int]:
    index = round(len(base_tokens) * (depth / 100))
    return base_tokens[:index] + needle_tokens + base_tokens[index:]


def run_position_simulation(base_text: str, needle: str, query: str, *, mode: str = "mock",
                            model: str = "openai/gpt-oss-20b", provider: str = "groq",
                            max_context_tokens: int = 12000, positions: tuple[int, ...] = DEFAULT_DEPTHS,
                            tokenizer: TokenCodec | None = None,
                            gateway: LLMGateway | None = None) -> dict:
    if not needle.strip():
        raise ValueError("needle must not be empty")
    if mode not in {"mock", "api"}:
        raise ValueError("mode must be 'mock' or 'api'")
    if not positions or any(depth < 0 or depth > 100 for depth in positions):
        raise ValueError("positions must be percentages between 0 and 100")
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be greater than zero")
    codec = tokenizer or get_codec()
    base_tokens = codec.encode(base_text)
    needle_tokens = codec.encode(needle)
    system = "Find the exact identifier in the supplied context. Return only the identifier if present; otherwise return NOT_FOUND."
    fixed_overhead = len(codec.encode("CONTEXT START\n\nCONTEXT END\nQUESTION: " + query)) + len(codec.encode(system))
    available_base = max_context_tokens - len(needle_tokens) - fixed_overhead
    if available_base < 0:
        raise ValueError("Needle, query, and system instruction exceed max_context_tokens.")
    base_tokens = base_tokens[:available_base]
    client = gateway or (LLMGateway(provider=provider, model=model) if mode == "api" else None)
    results = []
    for depth in positions:
        context = codec.decode(_insert_at_depth(base_tokens, needle_tokens, depth))
        prompt = f"CONTEXT START\n{context}\nCONTEXT END\nQUESTION: {query}"
        if len(codec.encode(prompt)) + len(codec.encode(system)) > max_context_tokens:
            results.append(PositionResult(position_percent=depth, retrieved=False, error="Prompt exceeds configured token cap."))
            continue
        try:
            if mode == "mock":
                answer = needle if needle in context else "NOT_FOUND"
            else:
                assert client is not None
                answer = client.complete(prompt, system_prompt=system, max_output_tokens=128)
            results.append(PositionResult(position_percent=depth, retrieved=needle in answer, answer=answer))
        except Exception as exc:
            results.append(PositionResult(position_percent=depth, retrieved=False, error=str(exc)))
    successes = sum(result.retrieved for result in results)
    accuracy = round(successes / len(results) * 100, 2) if results else 0.0
    note = ("Mock mode validates placement/report plumbing only; it is not empirical model evidence."
            if mode == "mock" else "API mode records actual responses; results depend on model, prompt, and provider limits.")
    report = SimulationReport(model=model, mode=mode, needle=needle, query=query,
                              positions=results, successful_runs=successes,
                              total_runs=len(results), accuracy_percent=accuracy, note=note)
    return report.model_dump()
