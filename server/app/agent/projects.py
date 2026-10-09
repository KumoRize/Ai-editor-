"""Editing projects: a persistent timeline that an external agent (via MCP) or the
API edits step by step with the same validated tools, then renders as a job."""
from __future__ import annotations

import threading
from pathlib import Path

from ..engine.timeline import Timeline, TimelineEditor
from ..errors import EditorError
from ..presets import PRESETS, output_size
from ..storage import db
from ..storage.files import job_dir, new_id
from ..tools.registry import TOOLS, ToolContext, call_tool
from .orchestrator import Reporter, initial_steps

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
# Tools an outside agent may call on a project (rendering goes through render_project as a job).
PROJECT_TOOLS = [n for n, t in TOOLS.items() if t.group in ("inspect", "edit") or n == "validate_timeline"]


def _lock(project_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(project_id, threading.Lock())


def _assets(asset_ids: list[str]) -> dict[str, dict]:
    out = {}
    for aid in asset_ids:
        a = db.get_asset(aid)
        if a is None or not Path(a["path"]).is_file():
            raise EditorError("ASSET_MISSING", f"File '{aid}' is no longer available.")
        out[aid] = a
    return out


def create_project(asset_ids: list[str], preset: str = "instagram_reels", aspect_ratio: str | None = None,
                   resolution: str = "1080p", fit: str = "cover") -> dict:
    if not asset_ids:
        raise EditorError("BAD_REQUEST", "Give at least one uploaded file.")
    if preset not in PRESETS:
        raise EditorError("BAD_REQUEST", f"Unknown preset '{preset}'.")
    assets = _assets(list(dict.fromkeys(asset_ids)))
    width, height = output_size(aspect_ratio or PRESETS[preset]["aspect_ratio"], resolution)
    ed = TimelineEditor(None, assets)
    ed.create_timeline(width, height, 30, fit, float(PRESETS[preset]["max_duration"]))
    pid = new_id("p")
    db.insert_project(pid, list(assets))
    db.update_project(pid, timeline=ed.timeline.model_dump())
    return {"project_id": pid, "width": width, "height": height, "assets": list(assets)}


def add_assets(project_id: str, asset_ids: list[str]) -> dict:
    with _lock(project_id):
        p = _project(project_id)
        ids = list(dict.fromkeys(p["asset_ids"] + asset_ids))
        _assets(ids)
        db.update_project(project_id, asset_ids=ids)
        return {"project_id": project_id, "assets": ids}


def _project(project_id: str) -> dict:
    p = db.get_project(project_id)
    if p is None:
        raise EditorError("NOT_FOUND", f"Project '{project_id}' does not exist.")
    return p


def project_tool(project_id: str, tool: str, args: dict | None = None) -> dict:
    """Run one tool on a project's timeline; mutations are saved only if they succeed."""
    if tool not in PROJECT_TOOLS:
        return {"ok": False, "error": {"code": "NOT_ALLOWED", "retryable": False,
                                       "message": f"'{tool}' can't be used on a project (use render_project)."}}
    with _lock(project_id):
        p = _project(project_id)
        assets = _assets(p["asset_ids"])
        ed = TimelineEditor(Timeline.model_validate(p["timeline"]), assets)
        ctx = ToolContext(editor=ed, assets=assets)
        res = call_tool(ctx, tool, args or {})
        if res["ok"] and TOOLS[tool].mutates:
            db.update_project(project_id, timeline=ed.timeline.model_dump())
        return res


def render_project(project_id: str) -> dict:
    p = _project(project_id)
    ed = TimelineEditor(Timeline.model_validate(p["timeline"]), _assets(p["asset_ids"]))
    ed.validate_timeline()  # fail fast, before queuing
    job_id = new_id("j")
    db.insert_job(job_id, {"kind": "render_timeline", "project_id": project_id, "asset_ids": p["asset_ids"],
                           "prompt": "", "mode": "settings"}, initial_steps())
    from .. import jobs
    jobs.enqueue(job_id)
    return {"job_id": job_id, "status": "queued"}


def render_project_job(job_id: str, req: dict, assets: dict[str, dict], work_dir: Path, reporter: Reporter) -> dict:
    p = _project(req["project_id"])
    tl = Timeline.model_validate(p["timeline"])
    ed = TimelineEditor(tl, assets)
    ctx = ToolContext(editor=ed, assets=assets, work_dir=work_dir)
    for key in ("analyze_media", "analyze_reference", "plan"):
        reporter.set(key, "skipped", "Edited directly with tools")
    v = call_tool(ctx, "validate_timeline")
    if not v["ok"]:
        raise EditorError(v["error"]["code"], v["error"]["message"])
    reporter.set("build_timeline", "done", f"{v['result']['clips']} clip(s), {v['result']['duration']:.1f}s")
    reporter.set("apply_effects", "done", "From the project")
    reporter.set("render", "running", frac=0.0)
    ctx.on_progress = lambda f: reporter.set("render", "running", f"{int(f * 100)}%", frac=f)
    res = call_tool(ctx, "export_video")
    if not res["ok"]:
        raise EditorError(res["error"]["code"], res["error"]["message"], retryable=res["error"].get("retryable", False))
    out = res["result"]
    reporter.set("render", "done")
    q = call_tool(ctx, "quality_check")
    quality = q["result"] if q["ok"] else {"passed": False, "issues": [], "measurements": {}}
    reporter.set("verify", "done", f"H.264/AAC {out['width']}x{out['height']}, {out['duration']:.1f}s")
    (work_dir / "timeline.json").write_text(tl.model_dump_json(indent=2))
    return {"output_file": out["file"], "duration": out["duration"], "width": out["width"], "height": out["height"],
            "size_bytes": out["size_bytes"], "summary": f"Rendered project {req['project_id']}.",
            "planner": "tools", "segments": [], "unsupported_requests": [],
            "notes": [i["message"] for i in quality.get("issues", [])], "mode": "project", "quality": quality,
            "applied": [], "timeline": {"clips": len(tl.video_track().clips)}}


def job_status(job_id: str) -> dict:
    j = db.get_job(job_id)
    if j is None:
        raise EditorError("NOT_FOUND", f"Job '{job_id}' does not exist.")
    out = {"job_id": job_id, "status": j["status"], "progress": j["progress"], "error": j["error"],
           "steps": [{"label": s["label"], "status": s["status"], "detail": s["detail"]} for s in j["steps"]]}
    if j["status"] == "complete" and j["result"]:
        out["video_url"] = f"/api/jobs/{job_id}/video"
        out["download_url"] = f"/api/jobs/{job_id}/video?download=1"
        out["duration"] = j["result"].get("duration")
        out["quality"] = j["result"].get("quality")
    return out


__all__ = ["create_project", "add_assets", "project_tool", "render_project", "render_project_job", "job_status",
           "PROJECT_TOOLS", "job_dir"]
