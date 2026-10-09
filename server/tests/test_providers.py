"""Provider adapters, exercised offline with mocked transports."""
import json
from types import SimpleNamespace

import httpx
import pytest

from app import config
from app.agent.plan import PLAN_JSON_SCHEMA
from app.agent.providers import make_provider, planner_name
from app.agent.providers.anthropic_provider import AnthropicProvider
from app.agent.providers.base import LLMError
from app.agent.providers.openai_compat import OpenAICompatProvider

PLAN = {"summary": "s", "segments": [], "unsupported_requests": []}


def _anthropic(monkeypatch, response=None, exc=None):
    p = AnthropicProvider(api_key="sk-test", model="", effort="low")
    seen = {}

    def create(**kw):
        seen.update(kw)
        if exc:
            raise exc
        return response

    monkeypatch.setattr(p.client.beta.messages, "create", create)
    return p, seen


def resp(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def test_anthropic_request_shape_and_parse(monkeypatch):
    p, seen = _anthropic(monkeypatch, resp(json.dumps(PLAN)))
    assert p.generate_json(system="sys", user="u", schema=PLAN_JSON_SCHEMA, schema_name="edit_plan") == PLAN
    assert seen["model"] == "claude-opus-5-5"
    assert seen["output_config"]["format"] == {"type": "json_schema", "schema": PLAN_JSON_SCHEMA}
    assert seen["output_config"]["effort"] == "low"
    assert "tool_choice" not in seen  # forced tool use is rejected by current models


@pytest.mark.parametrize("r,retryable", [(resp("", "refusal"), False), (resp("{", "end_turn"), True),
                                          (resp("x", "max_tokens"), True)])
def test_anthropic_bad_responses(monkeypatch, r, retryable):
    p, _ = _anthropic(monkeypatch, r)
    with pytest.raises(LLMError) as ei:
        p.generate_json(system="s", user="u", schema={}, schema_name="x")
    assert ei.value.retryable is retryable


def test_anthropic_api_errors(monkeypatch):
    import anthropic
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
    p, _ = _anthropic(monkeypatch, exc=err)
    with pytest.raises(LLMError) as ei:
        p.generate_json(system="s", user="u", schema={}, schema_name="x")
    assert not ei.value.retryable
    err = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)
    p, _ = _anthropic(monkeypatch, exc=err)
    with pytest.raises(LLMError) as ei:
        p.generate_json(system="s", user="u", schema={}, schema_name="x")
    assert ei.value.retryable


def test_openai_compat(monkeypatch):
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.update(url=url, body=json, headers=headers)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                      "message": {"content": __import__("json").dumps(PLAN)}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    p = OpenAICompatProvider("k", "http://localhost:11434/v1/", "llama3")
    assert p.generate_json(system="s", user="u", schema=PLAN_JSON_SCHEMA, schema_name="edit_plan") == PLAN
    assert calls["url"] == "http://localhost:11434/v1/chat/completions"
    assert calls["body"]["response_format"]["json_schema"]["strict"] is True
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(503, text="down"))
    with pytest.raises(LLMError) as ei:
        p.generate_json(system="s", user="u", schema={}, schema_name="x")
    assert ei.value.retryable


def test_provider_selection(settings):
    s = config.get_settings()
    s.llm_provider = "auto"
    assert make_provider(s) is None and planner_name(s) == "built-in rules"
    s.anthropic_api_key = "sk-x"
    assert make_provider(s).name == "anthropic:claude-opus-5-5"
    s.llm_provider, s.anthropic_api_key = "anthropic", ""
    assert planner_name(s) == "misconfigured"
    s.llm_provider, s.openai_api_key, s.llm_model = "openai", "k", "gpt-x"
    assert make_provider(s).name == "openai-compatible:gpt-x"


def test_orchestrator_falls_back_honestly(settings, media, monkeypatch, tmp_path):
    """If the LLM fails, the job still completes with the built-in planner and says so."""
    from app.agent import orchestrator
    from tests.conftest import make_asset

    class Broken:
        name = "broken"

        def generate_json(self, **kw):
            raise LLMError("The AI planner failed.", retryable=False)

    monkeypatch.setattr(orchestrator, "make_provider", lambda s: Broken())
    steps = orchestrator.initial_steps()
    rep = orchestrator.Reporter(steps, lambda st, p: None)
    req = {"preset": "square", "resolution": "720p", "duration": 3, "prompt": "", "fit": "cover"}
    res = orchestrator.run_edit_job("j_test", req, {"a_1": make_asset(media["land"])}, tmp_path, rep)
    assert res["planner"] == "built-in rules"
    assert any("built-in planner" in n for n in res["notes"])
    assert (tmp_path / res["output_file"]).is_file()
