"""The tool layer: the only bridge between the agent and analysis/engine/render.

Each tool has a JSON-Schema input (from a Pydantic model) and a handler. Calls
return {"ok": true, "result": ...} or {"ok": false, "error": {...}} — they never
raise to the caller and never report success for a failed operation. Mutating
tools run in a transaction, so a failed call leaves the timeline unchanged.

The same registry backs the built-in agent, the LLM tool loop and the MCP server.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from ..analysis import media
from ..analysis import transcribe as speech
from ..analysis.reference import analyze_reference
from ..analysis.sfx import sfx_path
from ..config import get_settings
from ..engine.compiler import compile_timeline
from ..engine.timeline import TimelineEditor
from ..errors import EditorError
from ..render.quality import quality_check
from ..render.runner import render, verify_output

log = logging.getLogger(__name__)
CURRENT_PHASE = 6


@dataclass
class ToolContext:
    editor: TimelineEditor
    assets: dict[str, dict]
    analyses: dict[str, dict] = field(default_factory=dict)      # clip analysis by asset_id
    music: dict[str, dict] = field(default_factory=dict)         # beat analysis by asset_id
    transcripts: dict[str, dict] = field(default_factory=dict)
    references: dict[str, dict] = field(default_factory=dict)
    work_dir: Path | None = None
    on_progress: Callable[[float], None] | None = None


# ------------------------------------------------------------------ inputs
class NoArgs(BaseModel):
    pass


class AssetArg(BaseModel):
    asset_id: str


class FramesArgs(BaseModel):
    asset_id: str
    times: list[float] = Field(description="Source times in seconds (max 24)")
    width: int = 256


class CreateTimelineArgs(BaseModel):
    width: int
    height: int
    fps: Literal[24, 25, 30, 50, 60] = 30
    fit: Literal["cover", "contain", "blur"] = "cover"
    max_duration: float = 600.0


class SelectClipArgs(BaseModel):
    asset_id: str
    src_in: float = Field(description="Start time in the source (s)")
    src_out: float = Field(description="End time in the source (s)")
    track_id: str = Field("V1", description="V1 = main track; V2+ = picture-in-picture overlay track")
    index: int | None = Field(None, description="Main track: insert position (default: append)")
    start: float | None = Field(None, description="Overlay track: timeline start (s)")
    x: float | None = Field(None, description="Overlay: left edge, fraction of the frame")
    y: float | None = Field(None, description="Overlay: top edge, fraction of the frame")
    width: float | None = Field(None, description="Overlay: width, fraction of the frame")
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
    index: int | None = Field(None, description="Main-track clips: new position")
    start: float | None = Field(None, description="Overlay clips: new timeline start (s)")


class ItemArg(BaseModel):
    clip_id: str = Field(description="Any item id: clip, overlay, caption/text, music or sound effect")


class SpeedArgs(BaseModel):
    clip_id: str
    speed: float = Field(description="0.25-4")


class RampArgs(BaseModel):
    clip_id: str
    pattern: Literal["slow_middle", "fast_middle", "slow_start", "slow_end", "speed_up_end"]
    keep_duration: bool = True


class ZoomArgs(BaseModel):
    clip_id: str = "all"
    kind: Literal["in", "out", "punch_in", "pan_left", "pan_right", "ken_burns"] = "in"
    amount: float = Field(0.15, description="0.02-1, e.g. 0.15 = 15% closer")


class KeyframeArgs(BaseModel):
    clip_id: str
    t: float = Field(description="Seconds from the start of the clip")
    scale: float = 1.0
    x: float = 0.5
    y: float = 0.5


class ColorArgs(BaseModel):
    clip_id: str = "all"
    preset: Literal["none", "warm", "cool", "cinematic", "bw", "vintage", "vibrant", "moody"] = "none"
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    warmth: float = 0.0


class EffectArgs(BaseModel):
    clip_id: str = "all"
    effect: Literal["vignette", "grain", "blur", "sharpen", "bw", "sepia"]
    remove: bool = False


class CropArgs(BaseModel):
    clip_id: str = "all"
    x: float
    y: float
    w: float
    h: float


class RotateArgs(BaseModel):
    clip_id: str = "all"
    degrees: Literal[0, 90, 180, 270]


class TransitionArgs(BaseModel):
    clip_id: str | None = Field(None, description="Clip that receives the transition from the previous clip")
    type: str = "fade"
    duration: float = 0.5
    position: Literal["before", "start", "end"] = "before"


class ResizeArgs(BaseModel):
    width: int
    height: int
    fit: Literal["cover", "contain", "blur"] = "cover"


class TextArgs(BaseModel):
    text: str
    start: float
    end: float
    style: Literal["subtitle", "bold", "minimal", "title"] = "title"
    position: Literal["bottom", "center", "top"] = "center"


class CaptionArgs(TextArgs):
    style: Literal["subtitle", "bold", "minimal", "title"] = "subtitle"
    position: Literal["bottom", "center", "top"] = "bottom"


class SubtitlesArgs(BaseModel):
    style: Literal["subtitle", "bold", "minimal"] = "subtitle"
    position: Literal["bottom", "center", "top"] = "bottom"
    language: str | None = None


class MusicArgs(BaseModel):
    asset_id: str
    start: float = 0.0
    src_in: float = 0.0
    duration: float | None = None
    volume: float = 0.8
    fade_in: float = 0.5
    fade_out: float = 1.5
    loop: bool = True
    duck: bool = True
    track_id: str | None = None


class VolumeArgs(BaseModel):
    clip_id: str = Field(description="Clip id, 'all', or a music/sound-effect id")
    volume: float


class SfxArgs(BaseModel):
    at: float
    sfx: Literal["whoosh", "swoosh", "impact", "pop", "click", "riser", "shutter"] | None = None
    asset_id: str | None = None
    volume: float = 0.8


class AudioSettingsArgs(BaseModel):
    normalize: bool = True


class SnapArgs(BaseModel):
    asset_id: str = Field(description="Music asset whose beats to use")
    offset: float = Field(0.0, description="Song time at timeline 0 (the music's src_in)")


# ------------------------------------------------------------------ handlers
def _asset(ctx: ToolContext, asset_id: str) -> dict:
    a = ctx.assets.get(asset_id)
    if a is None:
        raise EditorError("ASSET_NOT_FOUND", f"Media '{asset_id}' does not exist.")
    return a


def _analyze(ctx: ToolContext, a: AssetArg):
    asset = _asset(ctx, a.asset_id)
    if asset["role"] != "clip":
        raise EditorError("BAD_ASSET", "analyze_video works on video clips.")
    if a.asset_id not in ctx.analyses:
        ctx.analyses[a.asset_id] = media.analyze_video(Path(asset["path"]), asset["sha256"], asset["meta"])
    return ctx.analyses[a.asset_id]


def _beats(ctx: ToolContext, a: AssetArg):
    asset = _asset(ctx, a.asset_id)
    if not asset["meta"].get("has_audio"):
        raise EditorError("NO_AUDIO", "This file has no sound to find beats in.")
    if a.asset_id not in ctx.music:
        ctx.music[a.asset_id] = media.analyze_music(Path(asset["path"]), asset["sha256"], asset["meta"])
    return ctx.music[a.asset_id]


def _transcribe(ctx: ToolContext, a: AssetArg, language: str | None = None):
    asset = _asset(ctx, a.asset_id)
    if not asset["meta"].get("has_audio"):
        return {"language": None, "segments": []}
    if a.asset_id not in ctx.transcripts:
        ctx.transcripts[a.asset_id] = speech.transcribe(Path(asset["path"]), asset["sha256"], language)
    return ctx.transcripts[a.asset_id]


def _reference(ctx: ToolContext, a: AssetArg):
    asset = _asset(ctx, a.asset_id)
    if asset["role"] != "reference":
        raise EditorError("BAD_ASSET", "analyze_reference needs a file uploaded as a reference video.")
    if a.asset_id not in ctx.references:
        ctx.references[a.asset_id] = analyze_reference(Path(asset["path"]), asset["sha256"], asset["meta"])
    return ctx.references[a.asset_id]


def _frames(ctx: ToolContext, a: FramesArgs):
    asset = _asset(ctx, a.asset_id)
    if not asset["meta"].get("has_video"):
        raise EditorError("NO_VIDEO", "This file has no picture.")
    width = max(64, min(640, a.width))
    times = [t for t in a.times if 0 <= t <= asset["meta"]["duration"]][:24]
    media.extract_frames(Path(asset["path"]), asset["sha256"], times, width)
    return {"frames": [{"t": t, "url": f"/api/assets/{a.asset_id}/frame?t={t:g}&w={width}"} for t in times]}


def _subtitles(ctx: ToolContext, a: SubtitlesArgs):
    tl = ctx.editor.timeline
    if tl is None:
        raise EditorError("NO_TIMELINE", "No timeline exists yet.")
    clips = tl.video_track().clips
    vertical = tl.settings.height > tl.settings.width
    from ..analysis.transcribe import caption_chunks
    chunks: list[dict] = []
    for aid in dict.fromkeys(c.asset_id for c in clips):
        tr = _transcribe(ctx, AssetArg(asset_id=aid), a.language)
        chunks += caption_chunks(tr, [c for c in clips if c.asset_id == aid], a.style, vertical)
    chunks.sort(key=lambda c: c["start"])
    ctx.editor.clear_captions("caption")
    added = 0
    for c in chunks:
        if c["start"] < tl.duration - 0.05:
            ctx.editor.add_caption(c["text"], c["start"], min(c["end"], tl.duration), a.style, a.position)
            added += 1
    return {"captions": added, "words_found": bool(chunks)}


def _snap(ctx: ToolContext, a: SnapArgs):
    beats = _beats(ctx, AssetArg(asset_id=a.asset_id))["beats"]
    shifted = [b - a.offset for b in beats if b - a.offset >= 0]
    return ctx.editor.snap_to_beats(shifted)


def _sfx_paths(ctx: ToolContext) -> dict[str, Path]:
    names = {it.sfx for it in ctx.editor.timeline.audio if it.kind == "sfx" and it.sfx}
    return {n: sfx_path(n) for n in names}


def _render(ctx: ToolContext, preview: bool):
    if ctx.work_dir is None:
        raise EditorError("NO_WORKDIR", "Rendering is not available in this context.")
    ctx.editor.validate_timeline()
    s = get_settings()
    out = ctx.work_dir / ("preview.mp4" if preview else "final.mp4")  # fixed names: no caller-controlled paths
    spec = compile_timeline(ctx.editor.timeline, ctx.assets, out, segment_dir=s.data_dir / "cache" / "segments",
                            ffmpeg_bin=s.ffmpeg_bin, preview=preview, x264_preset=s.x264_preset,
                            sfx_paths=_sfx_paths(ctx))
    stats = render(spec, ctx.on_progress)
    check = verify_output(out, width=spec.width, height=spec.height, duration=spec.duration, fps=spec.fps)
    check.update(stats)
    return {"path": str(out), "file": out.name, **check}


def _quality(ctx: ToolContext, a: NoArgs):
    if ctx.work_dir is None or not (ctx.work_dir / "final.mp4").is_file():
        raise EditorError("NO_OUTPUT", "Export the video before checking its quality.")
    return quality_check(ctx.work_dir / "final.mp4", ctx.editor.timeline, ctx.assets)


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    phase: int
    handler: Callable[[ToolContext, Any], Any] | None
    mutates: bool = False
    group: str = "edit"


TOOLS: dict[str, Tool] = {}


def _reg(name, description, args, phase, handler=None, mutates=False, group="edit"):
    TOOLS[name] = Tool(name, description, args, phase, handler, mutates, group)


def E(fn):  # editor-bound helper
    return lambda ctx, a: fn(ctx.editor, a)


# Inspect
_reg("get_video_metadata", "Duration, size, fps, rotation, HDR and codecs of an uploaded file.", AssetArg, 1,
     lambda ctx, a: _asset(ctx, a.asset_id)["meta"], group="inspect")
_reg("analyze_video", "Scene cuts and audio loudness of a clip (cached).", AssetArg, 1, _analyze, group="inspect")
_reg("detect_scenes", "Scene-change timestamps of a clip.", AssetArg, 1,
     lambda ctx, a: _analyze(ctx, a)["scenes"], group="inspect")
_reg("analyze_audio", "Loudness envelope and silent ranges of a clip.", AssetArg, 1,
     lambda ctx, a: _analyze(ctx, a)["audio"], group="inspect")
_reg("extract_frames", "Small JPEG frames at given source times (returns URLs).", FramesArgs, 3, _frames,
     group="inspect")
_reg("detect_beats", "Tempo, beat and downbeat times of a music file.", AssetArg, 4, _beats, group="inspect")
_reg("transcribe", "Speech-to-text with word timings for a clip.", AssetArg, 2,
     lambda ctx, a: _transcribe(ctx, a), group="inspect")
_reg("analyze_reference", "Style/pacing profile of a reference video (style only, footage never used).",
     AssetArg, 3, _reference, group="inspect")
_reg("get_timeline", "Compact description of the current timeline.", NoArgs, 1,
     E(lambda ed, a: ed.summary()), group="inspect")
# Timeline
_reg("create_timeline", "Create an empty timeline with output size and framing.", CreateTimelineArgs, 1,
     E(lambda ed, a: ed.create_timeline(a.width, a.height, a.fps, a.fit, a.max_duration)), True)
_reg("add_video_track", "Add a picture-in-picture overlay track (V2, V3...).", NoArgs, 2,
     E(lambda ed, a: ed.add_video_track()), True)
_reg("add_audio_track", "Add an audio track for music/sound effects.", NoArgs, 2,
     E(lambda ed, a: ed.add_audio_track()), True)
_reg("select_clip", "Place a segment of a clip on the main track (append/insert) or an overlay track.",
     SelectClipArgs, 1, E(lambda ed, a: ed.select_clip(a.asset_id, a.src_in, a.src_out, a.track_id, a.index,
                                                        a.note, a.start, a.x, a.y, a.width)), True)
_reg("trim_clip", "Change a clip's source in/out points.", TrimClipArgs, 1,
     E(lambda ed, a: ed.trim_clip(a.clip_id, a.src_in, a.src_out)), True)
_reg("split_clip", "Split a clip at a source time.", SplitClipArgs, 1,
     E(lambda ed, a: ed.split_clip(a.clip_id, a.at)), True)
_reg("move_clip", "Move a main-track clip to a new position, or an overlay to a new start time.", MoveClipArgs, 1,
     E(lambda ed, a: ed.move_clip(a.clip_id, a.index, a.start)), True)
_reg("delete_clip", "Delete any item (clip, overlay, caption/text, music, sound effect).", ItemArg, 1,
     E(lambda ed, a: ed.delete_clip(a.clip_id)), True)
# Visual
_reg("change_speed", "Change a clip's playback speed.", SpeedArgs, 4,
     E(lambda ed, a: ed.change_speed(a.clip_id, a.speed)), True)
_reg("speed_ramp", "Stepped speed ramp on a clip (keeps its length by default).", RampArgs, 4,
     E(lambda ed, a: ed.speed_ramp(a.clip_id, a.pattern, a.keep_duration)), True)
_reg("add_zoom", "Zoom/pan movement on a clip or all clips.", ZoomArgs, 4,
     E(lambda ed, a: ed.add_zoom(a.clip_id, a.kind, a.amount)), True)
_reg("add_keyframe", "Custom zoom/position keyframe on a clip.", KeyframeArgs, 4,
     E(lambda ed, a: ed.add_keyframe(a.clip_id, a.t, a.scale, a.x, a.y)), True)
_reg("apply_color_adjustment", "Colour preset and/or brightness, contrast, saturation, gamma, warmth.",
     ColorArgs, 4, E(lambda ed, a: ed.apply_color_adjustment(a.clip_id, a.preset, a.brightness, a.contrast,
                                                              a.saturation, a.gamma, a.warmth)), True)
_reg("apply_effect", "Visual effect (vignette, grain, blur, sharpen, bw, sepia).", EffectArgs, 4,
     E(lambda ed, a: ed.apply_effect(a.clip_id, a.effect, a.remove)), True)
_reg("crop_video", "Crop a clip to a region (fractions of the picture).", CropArgs, 4,
     E(lambda ed, a: ed.crop_video(a.clip_id, a.x, a.y, a.w, a.h)), True)
_reg("rotate_video", "Rotate a clip by 90/180/270 degrees.", RotateArgs, 4,
     E(lambda ed, a: ed.rotate_video(a.clip_id, a.degrees)), True)
_reg("resize_video", "Set output size and framing (cover / contain / blur).", ResizeArgs, 1,
     E(lambda ed, a: ed.resize_video(a.width, a.height, a.fit)), True)
_reg("add_transition", "Transition into a clip, or a fade at the start/end of the video.", TransitionArgs, 4,
     E(lambda ed, a: ed.add_transition(a.clip_id, a.type, a.duration, a.position)), True)
_reg("remove_transition", "Remove the transition into a clip.", ItemArg, 4,
     E(lambda ed, a: ed.remove_transition(a.clip_id)), True)
# Text & audio
_reg("add_text", "Title / text overlay at timeline times.", TextArgs, 2,
     E(lambda ed, a: ed.add_text(a.text, a.start, a.end, a.style, a.position)), True)
_reg("add_caption", "One caption at timeline times.", CaptionArgs, 2,
     E(lambda ed, a: ed.add_caption(a.text, a.start, a.end, a.style, a.position)), True)
_reg("add_subtitles", "Subtitles from the speech in the clips (speech-to-text).", SubtitlesArgs, 2, _subtitles, True)
_reg("add_music", "Add an uploaded song as background music (loop, fades, ducking).", MusicArgs, 2,
     E(lambda ed, a: ed.add_music(a.asset_id, a.start, a.src_in, a.duration, a.volume, a.fade_in, a.fade_out,
                                  a.loop, a.duck, a.track_id)), True)
_reg("set_volume", "Volume (0-2) of a clip, all clips, or a music/sound-effect item.", VolumeArgs, 1,
     E(lambda ed, a: ed.set_volume(a.clip_id, a.volume)), True)
_reg("add_sound_effect", "Built-in (whoosh, swoosh, impact, pop, click, riser, shutter) or uploaded sound.",
     SfxArgs, 4, E(lambda ed, a: ed.add_sound_effect(a.at, a.sfx, a.asset_id, a.volume)), True)
_reg("set_audio_settings", "Loudness normalisation of the final mix on/off.", AudioSettingsArgs, 4,
     E(lambda ed, a: ed.set_audio_settings(a.normalize)), True)
_reg("snap_to_beats", "Move every cut onto the nearest beat of a song.", SnapArgs, 4, _snap, True)
# Output
_reg("validate_timeline", "Check the whole timeline for errors.", NoArgs, 1,
     E(lambda ed, a: ed.validate_timeline()), group="output")
_reg("render_preview", "Render and verify a low-resolution preview.", NoArgs, 1,
     lambda ctx, a: _render(ctx, True), group="output")
_reg("export_video", "Render and verify the final MP4 (H.264/AAC).", NoArgs, 1,
     lambda ctx, a: _render(ctx, False), group="output")
_reg("quality_check", "Automatic checks of the exported video (black frames, loudness, clipping, frozen picture).",
     NoArgs, 5, _quality, group="output")


def tool_specs(available_only: bool = True, groups: tuple[str, ...] | None = None) -> list[dict]:
    """Provider-neutral tool specs (name, description, JSON Schema)."""
    out = []
    for t in TOOLS.values():
        available = t.handler is not None and t.phase <= CURRENT_PHASE
        if available_only and not available:
            continue
        if groups and t.group not in groups:
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
        err = e.errors()[0]
        loc = ".".join(str(x) for x in err.get("loc", ()))
        return {"ok": False, "error": {"code": "BAD_ARGUMENTS", "message": f"{loc}: {err.get('msg')}".strip(": "),
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
    except ValueError as e:  # defensive: malformed values that slipped through
        return {"ok": False, "error": {"code": "BAD_ARGUMENTS", "message": str(e)[:300], "retryable": False}}
