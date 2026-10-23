"""Minimal LLM gateway for Groq, OpenAI-compatible endpoints, and Anthropic."""
from __future__ import annotations
import os

DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}
DEFAULT_MODELS = {"groq": "openai/gpt-oss-20b", "openai": "gpt-4o-mini",
                  "ollama": "llama3.2", "vllm": "local-model", "anthropic": "claude-3-5-sonnet-latest"}


class ProviderError(RuntimeError):
    """Wrap provider configuration and request failures with actionable messages."""


class LLMGateway:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 timeout: float | None = None) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODELS.get(self.provider, ""))
        self.timeout = timeout or float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
        if self.provider not in {*DEFAULT_BASE_URLS, "anthropic"}:
            raise ProviderError(f"Unsupported provider '{self.provider}'. Choose groq/openai/ollama/vllm/anthropic.")
        if not self.model:
            raise ProviderError("Model is required. Set LLM_MODEL or pass --model.")

    def complete(self, prompt: str, system_prompt: str = "Answer exactly and concisely.",
                 max_output_tokens: int = 128) -> str:
        if self.provider == "anthropic":
            return self._anthropic(prompt, system_prompt, max_output_tokens)
        return self._openai_compatible(prompt, system_prompt, max_output_tokens)

    def _openai_compatible(self, prompt: str, system_prompt: str, max_output_tokens: int) -> str:
        from openai import OpenAI
        api_key_name = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}.get(self.provider)
        api_key = os.getenv(api_key_name, "") if api_key_name else os.getenv("LOCAL_LLM_API_KEY", "local")
        if api_key_name and not api_key:
            raise ProviderError(f"{api_key_name} is not set. Copy .env.example to .env and configure it.")
        base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URLS[self.provider])
        try:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                max_completion_tokens=max_output_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise ProviderError(f"{self.provider} request failed: {exc}") from exc

    def _anthropic(self, prompt: str, system_prompt: str, max_output_tokens: int) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)
            response = client.messages.create(
                model=self.model, max_tokens=max_output_tokens, system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
        except Exception as exc:
            raise ProviderError(f"anthropic request failed: {exc}") from exc
