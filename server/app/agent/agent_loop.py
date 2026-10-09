"""Tool-using agent loop: an LLM revises an existing timeline by calling the same
validated tools the rest of the system uses (used for "Edit again")."""
from __future__ import annotations

import json
import logging

from ..engine.timeline import Timeline
from ..tools.registry import TOOLS, ToolContext, call_tool, tool_specs
from .providers.base import LLMError

log = logging.getLogger(__name__)

AGENT_SYSTEM = """You are a video editor working on an existing edit through tools. The user asks \
for changes to the edit described in the first message. Make exactly the changes they ask for — \
nothing else — by calling the tools. Inspect with get_timeline when you need ids or times. Every \
tool validates its input; if a call fails, read the error and fix the call. Times on the timeline \
are output seconds; clip src_in/src_out are seconds of the source file. Reference videos are style \
inspiration only and can never be placed on the timeline. When you are done, call finish_edit with \
one plain sentence for the user describing what changed. If something cannot be done with the tools, \
say so in that sentence instead of pretending."""

FINISH = {"name": "finish_edit", "description": "Call when the requested changes are complete.",
          "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                           "required": ["summary"]}}


def agent_tool_specs() -> list[dict]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tool_specs(groups=("inspect", "edit")) if t["name"] not in ("create_timeline",)] + [FINISH]


def revise_timeline(provider, instruction: str, previous_timeline: dict, ctx: ToolContext,
                    max_turns: int = 12) -> tuple[Timeline, str]:
    """Return (revised timeline, summary). Raises LLMError if the agent can't finish validly."""
    try:
        ctx.editor.timeline = Timeline.model_validate(previous_timeline)
    except Exception as e:  # noqa: BLE001
        raise LLMError("The previous edit could not be loaded.", retryable=False, detail=str(e)) from e
    original = ctx.editor.timeline.model_copy(deep=True)
    assets = [{"asset_id": a, "role": v["role"], "name": v.get("original_name", a),
               "duration": v["meta"].get("duration"), "has_audio": v["meta"].get("has_audio")}
              for a, v in ctx.assets.items()]
    first = {"role": "user", "content": json.dumps({
        "request": instruction, "current_edit": ctx.editor.summary(), "files": assets}, separators=(",", ":"))}
    messages: list[dict] = [first]
    tools = agent_tool_specs()
    summary = ""
    for _turn in range(max_turns):
        resp = provider.chat(system=AGENT_SYSTEM, messages=messages, tools=tools)
        messages.append({"role": "assistant", "raw": resp["raw"]})
        if not resp["tool_calls"]:
            summary = resp["text"] or "Updated the edit."
            break
        results = []
        done = False
        for call in resp["tool_calls"]:
            if call["name"] == "finish_edit":
                summary = str((call.get("input") or {}).get("summary") or "Updated the edit.")[:300]
                v = call_tool(ctx, "validate_timeline")
                if v["ok"]:
                    done = True
                    results.append({"id": call["id"], "content": "ok"})
                else:
                    results.append({"id": call["id"], "is_error": True,
                                    "content": "The timeline is not valid yet: " + v["error"]["message"]})
                continue
            if call["name"] not in TOOLS or call["name"] not in {t["name"] for t in tools}:
                results.append({"id": call["id"], "is_error": True, "content": f"Unknown tool {call['name']}."})
                continue
            res = call_tool(ctx, call["name"], call.get("input") or {})
            log.info("agent tool %s -> %s", call["name"], "ok" if res["ok"] else res["error"]["code"])
            results.append({"id": call["id"], "is_error": not res["ok"],
                            "content": json.dumps(res["result"] if res["ok"] else res["error"], default=str)[:6000]})
        messages.append({"role": "tool", "results": results})
        if done:
            break
    else:
        if call_tool(ctx, "validate_timeline")["ok"] and ctx.editor.timeline != original:
            return ctx.editor.timeline, "Updated the edit (the agent ran out of steps before summarising)."
        raise LLMError("The AI agent could not finish the changes.", retryable=True)
    v = call_tool(ctx, "validate_timeline")
    if not v["ok"]:
        raise LLMError("The AI agent left the edit in an invalid state.", retryable=True, detail=v["error"]["message"])
    return ctx.editor.timeline, summary
