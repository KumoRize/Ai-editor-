"""Provider selection from server configuration."""
from __future__ import annotations

from ...config import Settings
from .base import LLMError, LLMProvider


def make_provider(s: Settings) -> LLMProvider | None:
    """Return the configured LLM provider, or None for the rule-based planner."""
    choice = s.llm_provider
    if choice == "auto":
        if s.anthropic_api_key:
            choice = "anthropic"
        elif s.openai_api_key and s.llm_model:
            choice = "openai"
        else:
            choice = "rules"
    if choice == "anthropic":
        if not s.anthropic_api_key:
            raise LLMError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.", retryable=False)
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(s.anthropic_api_key, s.llm_model, s.llm_effort, s.llm_timeout_s)
    if choice == "openai":
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(s.openai_api_key, s.openai_base_url, s.llm_model, s.llm_timeout_s)
    return None


def planner_name(s: Settings) -> str:
    try:
        p = make_provider(s)
    except LLMError:
        return "misconfigured"
    return p.name if p else "built-in rules"


__all__ = ["LLMError", "LLMProvider", "make_provider", "planner_name"]
