"""Provider-neutral LLM interface.

The agent depends only on `LLMProvider`. Adapters translate to a vendor API.
Phase 1 needs one capability: return JSON that matches a JSON Schema (used for
the edit plan). Tool-calling loops are added to this interface in Phase 5.
"""
from __future__ import annotations

from typing import Protocol

from ...errors import EditorError


class LLMError(EditorError):
    def __init__(self, message: str, *, retryable: bool, detail: str = ""):
        super().__init__("PLANNER_FAILED", message, retryable=retryable, detail=detail)


class LLMProvider(Protocol):
    name: str

    def generate_json(self, *, system: str, user: str, schema: dict, schema_name: str) -> dict:
        """Return a dict that conforms to `schema`, or raise LLMError."""
        ...
