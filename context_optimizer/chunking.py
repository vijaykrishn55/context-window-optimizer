"""Token-based sliding-window chunking with configurable overlap."""
from .tokenizer import TokenCodec, get_codec


def sliding_window_chunks(text: str, chunk_tokens: int = 2000, overlap_ratio: float = 0.2,
                          tokenizer: TokenCodec | None = None) -> list[str]:
    if chunk_tokens < 2:
        raise ValueError("chunk_tokens must be at least 2")
    if not 0.15 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be at least 0.15 and less than 1.0")
    if not text:
        return []
    codec = tokenizer or get_codec()
    tokens = codec.encode(text)
    overlap = max(1, int(chunk_tokens * overlap_ratio))
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be smaller than chunk_tokens")
    step = chunk_tokens - overlap
    starts = [0]
    while starts[-1] + chunk_tokens < len(tokens):
        candidate = starts[-1] + step
        if candidate + chunk_tokens >= len(tokens):
            final_start = max(0, len(tokens) - chunk_tokens)
            if final_start > starts[-1]:
                starts.append(final_start)
            break
        starts.append(candidate)
    return [codec.decode(tokens[start:start + chunk_tokens]) for start in starts]
