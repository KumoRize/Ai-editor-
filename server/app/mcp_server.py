"""MCP server (streamable HTTP at /mcp/): lets external agents (Claude Code, other MCP
clients) drive the editor with the same validated tools as the built-in agent.

Workflow for a client: list_assets → create_project → edit tools (project_id + args)
→ render_project → get_job (poll) → download from the returned URL.
Authentication: the editor's ACCESS_TOKEN as a Bearer token (checked by the app).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.transport_security import TransportSecuritySettings

from .agent import projects
from .errors import EditorError
from .storage import db
from .tools.registry import TOOLS

INSTRUCTIONS = """AI Video Editor. Files are uploaded through the web app (or POST /api/uploads); \
use list_assets to see them. create_project gives a project_id with an empty timeline; every editing \
tool takes project_id plus its own arguments and is validated (errors are returned, never faked). \
render_project starts a render job; poll get_job until status is complete, then download video_url."""


def _project_tool_schema(name: str) -> dict:
    schema = TOOLS[name].args.model_json_schema()
    props = {"project_id": {"type": "string", "description": "Project to edit"}, **schema.get("properties", {})}
    req = ["project_id", *schema.get("required", [])]
    out = {"type": "object", "properties": props, "required": req}
    if "$defs" in schema:
        out["$defs"] = schema["$defs"]
    return out


MANAGEMENT = {
    "list_assets": ("Uploaded files (clips, music, references) with ids and durations.",
                    {"type": "object", "properties": {"limit": {"type": "integer"}}}),
    "create_project": ("Create an editing project (empty timeline) from uploaded files.",
                       {"type": "object", "properties": {
                           "asset_ids": {"type": "array", "items": {"type": "string"}},
                           "preset": {"type": "string", "enum": ["instagram_reels", "youtube_shorts", "tiktok",
                                                                 "youtube_landscape", "square"]},
                           "aspect_ratio": {"type": "string", "enum": ["9:16", "16:9", "1:1"]},
                           "resolution": {"type": "string", "enum": ["720p", "1080p"]},
                           "fit": {"type": "string", "enum": ["cover", "contain", "blur"]}},
                        "required": ["asset_ids"]}),
    "add_project_assets": ("Make more uploaded files usable in a project.",
                           {"type": "object", "properties": {"project_id": {"type": "string"},
                                                             "asset_ids": {"type": "array", "items": {"type": "string"}}},
                            "required": ["project_id", "asset_ids"]}),
    "render_project": ("Render a project to MP4 (background job; poll get_job).",
                       {"type": "object", "properties": {"project_id": {"type": "string"}},
                        "required": ["project_id"]}),
    "get_job": ("Status, progress, errors and the video URL of a render/edit job.",
                {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}),
}


def _list_tools() -> list[types.Tool]:
    out = [types.Tool(name=n, description=d, inputSchema=s) for n, (d, s) in MANAGEMENT.items()]
    for name in projects.PROJECT_TOOLS:
        t = TOOLS[name]
        out.append(types.Tool(name=name, description=t.description, inputSchema=_project_tool_schema(name)))
    return out


def _call(name: str, args: dict[str, Any]) -> dict:
    """Synchronous dispatcher; returns {"ok": ..., "result"/"error": ...}."""
    try:
        if name == "list_assets":
            rows = db.list_assets(min(200, int(args.get("limit") or 50)))
            return {"ok": True, "result": [{"asset_id": a["asset_id"], "role": a["role"], "name": a["original_name"],
                                            "duration": a["meta"].get("duration"), "width": a["meta"].get("width"),
                                            "height": a["meta"].get("height"),
                                            "has_audio": a["meta"].get("has_audio")} for a in rows]}
        if name == "create_project":
            return {"ok": True, "result": projects.create_project(
                list(args.get("asset_ids") or []), args.get("preset") or "instagram_reels", args.get("aspect_ratio"),
                args.get("resolution") or "1080p", args.get("fit") or "cover")}
        if name == "add_project_assets":
            return {"ok": True, "result": projects.add_assets(str(args.get("project_id")),
                                                              list(args.get("asset_ids") or []))}
        if name == "render_project":
            return {"ok": True, "result": projects.render_project(str(args.get("project_id")))}
        if name == "get_job":
            return {"ok": True, "result": projects.job_status(str(args.get("job_id")))}
        if name in projects.PROJECT_TOOLS:
            a = dict(args)
            pid = str(a.pop("project_id", ""))
            return projects.project_tool(pid, name, a)
        return {"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": f"Unknown tool '{name}'.", "retryable": False}}
    except EditorError as e:
        return {"ok": False, "error": e.to_dict()}


async def _on_list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=_list_tools())


async def _on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    res = await asyncio.to_thread(_call, params.name, dict(params.arguments or {}))
    payload = res["result"] if res["ok"] else {"error": res["error"]}
    text = json.dumps(payload, default=str)
    return types.CallToolResult(content=[types.TextContent(text=text)],
                                structuredContent=payload if isinstance(payload, dict) else {"items": payload},
                                isError=not res["ok"])


def build_mcp_app() -> dict:
    server = Server("ai-video-editor", version="1.0", instructions=INSTRUCTIONS,
                    on_list_tools=_on_list_tools, on_call_tool=_on_call_tool)
    app = server.streamable_http_app(
        streamable_http_path="/", stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    return {"app": app, "lifespan": server.session_manager.run, "server": server}
