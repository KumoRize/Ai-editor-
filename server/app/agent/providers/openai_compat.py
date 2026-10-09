"""Adapter for any OpenAI-compatible Chat Completions endpoint
(OpenAI, OpenRouter, Together, Groq, local Ollama / vLLM / LM Studio ...)."""
from __future__ import annotations

import json

import httpx

from .base import LLMError


class OpenAICompatProvider:
    def __init__(self, api_key: str, base_url: str, model: str, timeout_s: int = 120):
        if not model:
            raise LLMError("LLM_MODEL must be set for the OpenAI-compatible provider.", retryable=False)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.name = f"openai-compatible:{model}"

    def generate_json(self, *, system: str, user: str, schema: dict, schema_name: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": schema_name, "schema": schema, "strict": True}},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                           timeout=self.timeout_s)
        except httpx.HTTPError as e:
            raise LLMError("Could not reach the AI planner.", retryable=True, detail=str(e)) from e
        if r.status_code in (401, 403):
            raise LLMError("The AI planner is not configured correctly (API key rejected).", retryable=False)
        if r.status_code == 429 or r.status_code >= 500:
            raise LLMError("The AI planner is busy or failed.", retryable=True, detail=r.text[:500])
        if r.status_code >= 400:
            raise LLMError("The AI planner rejected the request.", retryable=False, detail=r.text[:500])
        try:
            choice = r.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMError("The AI planner's answer was cut off.", retryable=True)
            return json.loads(choice["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            raise LLMError("The AI planner returned malformed output.", retryable=True,
                           detail=r.text[:500]) from e
