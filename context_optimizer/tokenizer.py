"""Token codec with tiktoken-first behavior and an explicit offline fallback."""
from __future__ import annotations

import re
from typing import Protocol


class TokenCodec(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class ApproximateCodec:
    """Character-level fallback for offline tests; counts are estimates, not model tokens."""
    name = "character-fallback-estimate"

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class TiktokenCodec:
    """Wrap tiktoken and avoid returning malformed UTF-8 at truncated boundaries."""
    def __init__(self, model_name: str = "gpt-4o") -> None:
        import tiktoken
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.name = f"tiktoken:{self.encoding.name}"

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        # Incomplete final multi-byte characters are dropped instead of replaced.
        return self.encoding.decode_bytes(tokens).decode("utf-8", errors="ignore")


def get_codec(model_name: str = "gpt-4o", allow_fallback: bool = True) -> TokenCodec:
    """Return tiktoken when installed; fallback is visible in reports and CLI output."""
    try:
        return TiktokenCodec(model_name)
    except (ImportError, OSError, RuntimeError) as exc:
        if not allow_fallback:
            raise RuntimeError("tiktoken is required. Install requirements.txt first.") from exc
        return ApproximateCodec()


def count_tokens(text: str, codec: TokenCodec) -> int:
    return len(codec.encode(text or ""))


def truncate_tokens(text: str, max_tokens: int, codec: TokenCodec) -> str:
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if not text or max_tokens == 0:
        return ""
    tokens = codec.encode(text)
    if len(tokens) <= max_tokens:
        return text
    # Trim cautiously because decoders may split a multibyte character.
    return codec.decode(tokens[:max_tokens])
