"""The tool layer: the only bridge between the agent and analysis/engine/render.

Each tool has a JSON-Schema input (from a Pydantic model), a phase, and a handler.
Calls return {"ok": true, "result": ...} or {"ok": false, "error": {...}} — they
never raise to the caller and never report success for a failed operation.
Unavailable (future-phase) tools are listed with their phase but cannot run.

The same registry will back the MCP server in Phase 5.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from ..analysis import media
from ..config import get_settings
from ..engine.compiler import compile_timeline
from ..engine.timeline import TimelineEditor
from ..errors import EditorError
from ..render.runner import render, verify_output

log = logging.getLogger(__name__)


@dataclass
class ToolContext:
    editor: TimelineEditor
    assets: dict[str, dict]
    analyses: dict[str, dict] = field(default_factory=dict)
    work_dir: Path | None = None
    on_progress: Callable[[float], None] | None = None


# ------------------------------------------------------------------ inputs
class AssetArg(BaseModel):
    asset_id: str


class CreateTimelineArgs(BaseModel):
    width: int
    height: int
    fps: int = 30
    fit: Literal["cover", "contain", "blur"] = "cover"
    max_duration: float = 600.0


class SelectClipArgs(BaseModel):
    asset_id: str
    src_in: float = Field(description="Start time in the source (s)")
    src_out: float = Field(description="End time in the source (s)")
    track_id: str = "V1"
    index: int | None = Field(default=None, description="Insert position; default appends")
    note: str = ""


class TrimClipArgs(BaseModel):
    clip_id: str
    src_in: float | None = None
    src_out: float | None = None


class SplitClipArgs(BaseModel):
    clip_id: str
    at: float = Field(description="Source time to split at (s)")


class MoveClipArgs(BaseModel):
    clip_id: str
    index: int


class ClipArg(BaseModel):
    clip_id: str


class ResizeArgs(BaseModel):
    width: int
    height: int
    fit: Literal["cover", "contain", "blur"] = "cover"


class VolumeArgs(BaseModel):
    clip_id: str
    volume: float


class NoArgs(BaseModel):
    pass


# ------------------------------------------------------------------ handlers
def _asset(ctx: ToolContext, asset_id: str) -> dict:
    a = ctx.assets.get(asset_id)
    if a is None:
        raise EditorError("ASSET_NOT_FOUND", f"Media '{asset_id}' does not exist.")
    return a


def _get_metadata(ctx: ToolContext, a: AssetArg):
    return _asset(ctx, a.asset_id)["meta"]


def _analyze(ctx: ToolContext, a: AssetArg):
    asset = _asset(ctx, a.asset_id)
    if a.asset_id not in ctx.analyses:
        ctx.analyses[a.asset_id] = media.analyze_video(Path(asset["path"]), asset["sha256"], asset["meta"])
    return ctx.analyses[a.asset_id]


def _detect_scenes(ctx: ToolContext, a: AssetArg):
    return _analyze(ctx, a)["scenes"]


def _analyze_audio(ctx: ToolContext, a: AssetArg):
    return _analyze(ctx, a)["audio"]


def _render(ctx: ToolContext, preview: bool):
    if ctx.work_dir is None:
        raise EditorError("NO_WORKDIR", "Rendering is not available in this context.")
    ctx.editor.validate_timeline()
    s = get_settings()
    safe_name = "preview.mp4" if preview else "final.mp4"  # fixed names: no caller-controlled paths
    out = ctx.work_dir / safe_name
    spec = compile_timeline(ctx.editor.timeline, ctx.assets, out, segment_dir=s.data_dir / "cache" / "segments",
                            ffmpeg_bin=s.ffmpeg_bin, preview=preview, x264_preset=s.x264_preset)
    stats = render(spec, ctx.on_progress)
    check = verify_output(out, width=spec.width, height=spec.height, duration=spec.duration, fps=spec.fps)
    check.update(stats)
    return {"path": str(out), **check}


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    phase: int
    handler: Callable[[ToolContext, Any], Any] | None
    mutates: bool = False


TOOLS: dict[str, Tool] = {}


def _reg(name, description, args, phase, handler=None, mutates=False):
    TOOLS[name] = Tool(name, description, args, phase, handler, mutates)


E = lambda fn: (lambda ctx, a: fn(ctx.editor, a))  # noqa: E731  (editor-bound helper)

_reg("get_video_metadata", "Duration, size, fps, rotation and codecs of an uploaded clip.", AssetArg, 1, _get_metadata)
_reg("analyze_video", "Scene cuts and audio loudness of a clip (cached).", AssetArg, 1, _analyze)
_reg("detect_scenes", "Scene-change timestamps of a clip.", AssetArg, 1, _detect_scenes)
_reg("analyze_audio", "Loudness envelope and silent ranges of a clip.", AssetArg, 1, _analyze_audio)
_reg("create_timeline", "Create an empty timeline with output size and framing.", CreateTimelineArgs, 1,
     E(lambda ed, a: ed.create_timeline(a.width, a.height, a.fps, a.fit, a.max_duration)), True)
_reg("add_video_track", "Add a video track.", NoArgs, 2, E(lambda ed, a: ed.add_video_track()), True)
_reg("select_clip", "Place a segment of a clip on a track.", SelectClipArgs, 1,
     E(lambda ed, a: ed.select_clip(a.asset_id, a.src_in, a.src_out, a.track_id, a.index, a.note)), True)
_reg("trim_clip", "Change a clip's source in/out points.", TrimClipArgs, 1,
     E(lambda ed, a: ed.trim_clip(a.clip_id, a.src_in, a.src_out)), True)
_reg("split_clip", "Split a clip at a source time.", SplitClipArgs, 1,
     E(lambda ed, a: ed.split_clip(a.clip_id, a.at)), True)
_reg("move_clip", "Move a clip to a new position.", MoveClipArgs, 1,
     E(lambda ed, a: ed.move_clip(a.clip_id, a.index)), True)
_reg("delete_clip", "Remove a clip from the timeline.", ClipArg, 1,
     E(lambda ed, a: ed.delete_clip(a.clip_id)), True)
_reg("resize_video", "Set output size and framing (cover / contain / blur).", ResizeArgs, 1,
     E(lambda ed, a: ed.resize_video(a.width, a.height, a.fit)), True)
_reg("set_volume", "Set a clip's volume (0–2).", VolumeArgs, 1,
     E(lambda ed, a: ed.set_volume(a.clip_id, a.volume)), True)
_reg("validate_timeline", "Check the whole timeline for errors.", NoArgs, 1,
     E(lambda ed, a: ed.validate_timeline()))
_reg("render_preview", "Render and verify a low-resolution preview.", NoArgs, 1,
     lambda ctx, a: _render(ctx, True))
_reg("export_video", "Render and verify the final MP4 (H.264/AAC).", NoArgs, 1,
     lambda ctx, a: _render(ctx, False))

# Planned tools: advertised with their phase, never executable before then.
for _name, _phase, _desc in [
    ("extract_frames", 3, "Extract thumbnails at given times."),
    ("analyze_reference", 3, "Pacing/style profile of a reference video."),
    ("detect_beats", 4, "Beat grid of a music track."),
    ("add_audio_track", 2, "Add an audio track."),
    ("add_music", 2, "Add background music."),
    ("add_caption", 2, "Add a caption."),
    ("add_subtitles", 2, "Auto-generate subtitles from speech."),
    ("add_text", 2, "Add a text overlay."),
    ("add_transition", 4, "Add a transition between clips."),
    ("add_zoom", 4, "Add a zoom."),
    ("add_keyframe", 4, "Add a transform keyframe."),
    ("change_speed", 4, "Change clip speed / speed ramp."),
    ("crop_video", 4, "Crop a clip."),
    ("rotate_video", 4, "Rotate a clip."),
    ("apply_effect", 4, "Apply a visual effect."),
    ("apply_color_adjustment", 4, "Adjust brightness/contrast/saturation."),
    ("add_sound_effect", 4, "Add a sound effect."),
]:
    _reg(_name, _desc, NoArgs, _phase, None)

CURRENT_PHASE = 1


def tool_specs(available_only: bool = True) -> list[dict]:
    """Provider-neutral tool specs (name, description, JSON Schema)."""
    out = []
    for t in TOOLS.values():
        available = t.handler is not None and t.phase <= CURRENT_PHASE
        if available_only and not available:
            continue
        out.append({"name": t.name, "description": t.description, "input_schema": t.args.model_json_schema(),
                    "available": available, "phase": t.phase})
    return out


def call_tool(ctx: ToolContext, name: str, args: dict | None = None) -> dict:
    t = TOOLS.get(name)
    if t is None:
        return {"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": f"Unknown tool '{name}'.", "retryable": False}}
    if t.handler is None or t.phase > CURRENT_PHASE:
        return {"ok": False, "error": {"code": "NOT_AVAILABLE",
                                       "message": f"'{name}' is not available yet (Phase {t.phase}).",
                                       "retryable": False}}
    try:
        parsed = t.args.model_validate(args or {})
    except ValidationError as e:
        return {"ok": False, "error": {"code": "BAD_ARGUMENTS", "message": str(e.errors()[0].get("msg")),
                                       "retryable": False}}
    try:
        if t.mutates:
            with ctx.editor.transaction():
                result = t.handler(ctx, parsed)
        else:
            result = t.handler(ctx, parsed)
        return {"ok": True, "result": result}
    except EditorError as e:
        log.info("tool %s failed: %s %s", name, e.code, e.detail or e.message)
        return {"ok": False, "error": e.to_dict()}
