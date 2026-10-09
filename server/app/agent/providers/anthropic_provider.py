"""Claude adapter (Anthropic Python SDK). Optional: only imported when configured."""
from __future__ import annotations

import json
import logging

from .base import LLMError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "", effort: str = "medium", timeout_s: int = 120):
        import anthropic  # local import keeps the SDK optional

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model = model or DEFAULT_MODEL
        self.effort = effort
        self.name = f"anthropic:{self.model}"

    def generate_json(self, *, system: str, user: str, schema: dict, schema_name: str) -> dict:
        a = self._anthropic
        try:
            # Structured output guarantees schema-valid JSON in the first text block.
            # Server-side fallback reroutes the request if the model declines it.
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (a.AuthenticationError, a.PermissionDeniedError) as e:
            raise LLMError("The AI planner is not configured correctly (API key rejected).",
                           retryable=False, detail=str(e)) from e
        except a.BadRequestError as e:
            raise LLMError("The AI planner rejected the request.", retryable=False, detail=str(e)) from e
        except a.RateLimitError as e:
            raise LLMError("The AI planner is busy (rate limited).", retryable=True, detail=str(e)) from e
        except a.APIStatusError as e:
            raise LLMError("The AI planner failed.", retryable=e.status_code >= 500, detail=str(e)) from e
        except a.APIConnectionError as e:
            raise LLMError("Could not reach the AI planner.", retryable=True, detail=str(e)) from e

        if resp.stop_reason == "refusal":
            raise LLMError("The AI planner declined this request.", retryable=False)
        if resp.stop_reason == "max_tokens":
            raise LLMError("The AI planner's answer was cut off.", retryable=True)
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)
        if not text:
            raise LLMError("The AI planner returned no plan.", retryable=True)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("The AI planner returned malformed JSON.", retryable=True, detail=text[:500]) from e
        log.info("anthropic plan usage: in=%s out=%s", resp.usage.input_tokens, resp.usage.output_tokens)
        return data
