import pytest
from context_optimizer.gateway import LLMGateway, ProviderError


def test_groq_defaults_to_gpt_oss_20b():
    gateway = LLMGateway(provider="groq")
    assert gateway.model == "openai/gpt-oss-20b"


def test_local_provider_does_not_require_remote_api_key():
    gateway = LLMGateway(provider="ollama", model="llama3.2")
    assert gateway.provider == "ollama"
    assert gateway.model == "llama3.2"


def test_unknown_provider_is_rejected():
    with pytest.raises(ProviderError, match="Unsupported provider"):
        LLMGateway(provider="unknown")
