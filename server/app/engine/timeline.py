"""Editing engine (component C): timeline model, validated operations, transactions.

The timeline is the single source of truth for rendering. Every mutation goes
through `TimelineEditor`, which validates inputs against real asset metadata and
applies changes transactionally (copy → mutate → validate → commit, or rollback).

Model
-----
* main video track (V1) is "magnetic": clips play back to back; a clip's
  `transition_in` overlaps it with the previous clip by the transition length.
  `timeline_start` is always derived.
* overlay tracks (V2+): picture-in-picture clips placed at explicit times.
* audio tracks (A1+): music beds and sound effects placed at explicit times.
* text track (T1): captions/subtitles and text overlays in timeline time.

Per-clip effects: speed, volume, zoom keyframes (as fractions of the clip so they
survive trims and speed changes), colour adjustment, visual effects, crop, rotate.
"""
from __future__ import annotations

import copy
import re
import secrets
from contextlib import contextmanager
from typing import Literal

from pydantic import BaseModel, Field

from ..errors import EditorError, invalid_timeline

FRAME_TOL = 1 / 30  # one frame of tolerance when comparing against source duration
MIN_CLIP_S = 0.2
SPEED_MIN, SPEED_MAX = 0.25, 4.0
VOLUME_MIN, VOLUME_MAX = 0.0, 2.0
ALLOWED_FPS = (24, 25, 30, 50, 60)  # 48000/fps is an integer: audio/video lengths match exactly
MAX_TEXT_ITEMS, MAX_AUDIO_ITEMS, MAX_OVERLAYS, MAX_CLIPS = 600, 200, 20, 300

Fit = Literal["cover", "contain", "blur"]

TRANSITIONS = ("fade", "dissolve", "fadeblack", "fadewhite", "wipeleft", "wiperight", "wipeup", "wipedown",
               "slideleft", "slideright", "slideup", "slidedown", "circleopen", "circleclose", "zoomin",
               "smoothleft", "smoothright", "radial", "pixelize", "hblur", "squeezeh")
INTRO_OUTRO = ("fade", "fadeblack", "fadewhite")
TRANSITION_LABELS = {"fade": "fade", "dissolve": "dissolve", "fadeblack": "fade-to-black", "fadewhite": "fade-to-white",
                     "wipeleft": "wipe", "wiperight": "wipe", "wipeup": "wipe", "wipedown": "wipe",
                     "slideleft": "slide", "slideright": "slide", "slideup": "slide", "slidedown": "slide",
                     "circleopen": "circle", "circleclose": "circle", "zoomin": "zoom", "smoothleft": "smooth slide",
                     "smoothright": "smooth slide", "radial": "radial", "pixelize": "pixelate", "hblur": "blur",
                     "squeezeh": "squeeze"}  # allowed at the very start / end of the video
COLOR_PRESETS = ("none", "warm", "cool", "cinematic", "bw", "vintage", "vibrant", "moody")
EFFECTS = ("vignette", "grain", "blur", "sharpen", "bw", "sepia")
ZOOM_KINDS = ("in", "out", "punch_in", "pan_left", "pan_right", "ken_burns")
SPEED_RAMPS: dict[str, list[tuple[float, float]]] = {
    # (fraction of the source range, speed)
    "slow_middle": [(0.4, 1.0), (0.2, 0.5), (0.4, 1.0)],
    "fast_middle": [(0.3, 1.0), (0.4, 2.0), (0.3, 1.0)],
    "slow_start": [(0.4, 0.5), (0.6, 1.0)],
    "slow_end": [(0.6, 1.0), (0.4, 0.5)],
    "speed_up_end": [(0.6, 1.0), (0.4, 2.0)],
}
TEXT_STYLES = ("subtitle", "bold", "minimal", "title")
TEXT_POSITIONS = ("bottom", "center", "top")
SFX_NAMES = ("whoosh", "swoosh", "impact", "pop", "click", "riser", "shutter")


class Keyframe(BaseModel):
    at: float          # fraction of the clip's duration, 0..1
    scale: float = 1.0
    x: float = 0.5     # centre of the visible area, fraction of the frame
    y: float = 0.5


class ColorAdjust(BaseModel):
    preset: str = "none"
    brightness: float = 0.0   # -0.5 .. 0.5
    contrast: float = 1.0     # 0.5 .. 2
    saturation: float = 1.0   # 0 .. 3
    gamma: float = 1.0        # 0.5 .. 2
    warmth: float = 0.0       # -1 (cool) .. 1 (warm)


class Transition(BaseModel):
    type: str
    duration: float


class Crop(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Clip(BaseModel):
    clip_id: str
    asset_id: str
    src_in: float
    src_out: float
    speed: float = 1.0
    volume: float = 1.0
    timeline_start: float = 0.0
    note: str = ""
    keyframes: list[Keyframe] = Field(default_factory=list)
    ease: Literal["linear", "inout"] = "inout"
    color: ColorAdjust | None = None
    effects: list[str] = Field(default_factory=list)
    transition_in: Transition | None = None
    crop: Crop | None = None
    rotate: int = 0

    @property
    def duration(self) -> float:
        return (self.src_out - self.src_in) / self.speed


class TextItem(BaseModel):
    item_id: str
    kind: Literal["caption", "text"] = "caption"
    text: str
    start: float
    end: float
    style: str = "subtitle"
    position: str = "bottom"


class AudioItem(BaseModel):
    item_id: str
    track_id: str
    kind: Literal["music", "sfx"]
    asset_id: str | None = None
    sfx: str | None = None
    start: float = 0.0
    src_in: float = 0.0
    duration: float | None = None   # None: music runs to the end of the video; sfx plays fully
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    loop: bool = False
    duck: bool = False              # lower under the original sound (speech)


class OverlayItem(BaseModel):
    item_id: str
    track_id: str
    asset_id: str
    src_in: float
    src_out: float
    start: float
    x: float = 0.62      # top-left corner, fraction of the frame
    y: float = 0.05
    width: float = 0.33  # fraction of the frame width

    @property
    def duration(self) -> float:
        return self.src_out - self.src_in


class Track(BaseModel):
    track_id: str
    kind: Literal["video", "overlay", "audio", "caption"]
    clips: list[Clip] = Field(default_factory=list)  # only for the main video track


class TimelineSettings(BaseModel):
    width: int
    height: int
    fps: int = 30
    fit: Fit = "cover"
    bg_color: str = "black"
    max_duration: float = 600.0
    intro: Transition | None = None   # fade in from black/white at the start
    outro: Transition | None = None   # fade out at the end
    normalize_audio: bool = True      # loudness-normalise the final mix (-14 LUFS)


class Timeline(BaseModel):
    timeline_id: str
    revision: int = 0
    settings: TimelineSettings
    tracks: list[Track] = Field(default_factory=list)
    texts: list[TextItem] = Field(default_factory=list)
    audio: list[AudioItem] = Field(default_factory=list)
    overlays: list[OverlayItem] = Field(default_factory=list)
    oplog: list[dict] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        v = self.video_track(required=False)
        if v is None or not v.clips:
            return 0.0
        last = v.clips[-1]
        return round(last.timeline_start + last.duration, 6)

    def video_track(self, required: bool = True) -> Track | None:
        for t in self.tracks:
            if t.kind == "video":
                return t
        if required:
            raise EditorError("TRACK_NOT_FOUND", "The timeline has no video track.")
        return None


def atomic(method):
    """Make an editor operation all-or-nothing: if it raises, the timeline is restored."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        snapshot = copy.deepcopy(self.timeline)
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.timeline = snapshot
            raise
    return wrapper


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def _finite(name: str, *vals: float) -> None:
    for v in vals:
        if v is None:
            continue
        if not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            raise invalid_timeline(f"{name} is not a number")


def _check_range(name: str, v: float, lo: float, hi: float) -> None:
    _finite(name, v)
    if not (lo <= v <= hi):
        raise invalid_timeline(f"{name} {v:g} must be between {lo:g} and {hi:g}")


def clean_text(text: str, limit: int = 200) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", str(text)).strip()
    if not text:
        raise invalid_timeline("text cannot be empty")
    return text[:limit]


def zoom_keyframes(kind: str, amount: float = 0.15) -> list[Keyframe]:
    a = amount
    return {
        "in": [Keyframe(at=0, scale=1.0), Keyframe(at=1, scale=1 + a)],
        "out": [Keyframe(at=0, scale=1 + a), Keyframe(at=1, scale=1.0)],
        "punch_in": [Keyframe(at=0, scale=1 + a), Keyframe(at=1, scale=1 + a)],
        "pan_left": [Keyframe(at=0, scale=1 + a, x=0.5 + a / 3), Keyframe(at=1, scale=1 + a, x=0.5 - a / 3)],
        "pan_right": [Keyframe(at=0, scale=1 + a, x=0.5 - a / 3), Keyframe(at=1, scale=1 + a, x=0.5 + a / 3)],
        "ken_burns": [Keyframe(at=0, scale=1.0, x=0.45, y=0.52), Keyframe(at=1, scale=1 + a, x=0.55, y=0.48)],
    }[kind]


def sample_zoom(kfs: list[Keyframe], at: float, ease: str = "linear") -> Keyframe:
    """Interpolate the zoom curve at a fraction of the clip."""
    if not kfs:
        return Keyframe(at=at)
    pts = sorted(kfs, key=lambda k: k.at)
    if at <= pts[0].at:
        return pts[0].model_copy(update={"at": at})
    for k0, k1 in zip(pts, pts[1:]):
        if k0.at <= at <= k1.at:
            u = 0.0 if k1.at == k0.at else (at - k0.at) / (k1.at - k0.at)
            if ease == "inout":
                u = u * u * (3 - 2 * u)
            return Keyframe(at=at, scale=k0.scale + (k1.scale - k0.scale) * u,
                            x=k0.x + (k1.x - k0.x) * u, y=k0.y + (k1.y - k0.y) * u)
    return pts[-1].model_copy(update={"at": at})


class TimelineEditor:
    """Validated, transactional operations on a Timeline.

    `assets` maps asset_id -> asset record (with `role` and `meta`). Only assets
    passed in here can be referenced, and reference videos can never be placed.
    """

    def __init__(self, timeline: Timeline | None, assets: dict[str, dict]):
        self.timeline = timeline
        self.assets = assets

    # ------------------------------------------------------------ helpers
    def _tl(self) -> Timeline:
        if self.timeline is None:
            raise EditorError("NO_TIMELINE", "No timeline exists yet. Call create_timeline first.")
        return self.timeline

    def _asset(self, asset_id: str) -> dict:
        a = self.assets.get(asset_id)
        if a is None:
            raise invalid_timeline(f"media '{asset_id}' does not exist")
        return a

    def _video_asset(self, asset_id: str) -> dict:
        a = self._asset(asset_id)
        if a.get("role") != "clip":
            raise invalid_timeline("reference and music files cannot be placed on a video track")
        if not a["meta"].get("has_video"):
            raise invalid_timeline(f"media '{asset_id}' has no video")
        return a

    def _find(self, clip_id: str) -> tuple[Track, int]:
        for t in self._tl().tracks:
            for i, c in enumerate(t.clips):
                if c.clip_id == clip_id:
                    return t, i
        raise EditorError("CLIP_NOT_FOUND", f"Clip '{clip_id}' does not exist.")

    def _clip(self, clip_id: str) -> Clip:
        t, i = self._find(clip_id)
        return t.clips[i]

    def _clips_for(self, clip_id: str | None) -> list[Clip]:
        """A single clip, or every main-track clip when clip_id is 'all' / None."""
        if clip_id in (None, "", "all"):
            clips = self._tl().video_track().clips
            if not clips:
                raise invalid_timeline("the timeline is empty")
            return clips
        return [self._clip(clip_id)]

    def _track(self, track_id: str) -> Track:
        for t in self._tl().tracks:
            if t.track_id == track_id:
                return t
        raise EditorError("TRACK_NOT_FOUND", f"Track '{track_id}' does not exist.")

    def _check_src(self, asset: dict, src_in: float, src_out: float, speed: float = 1.0) -> None:
        dur = float(asset["meta"]["duration"])
        _finite("start", src_in)
        _finite("end", src_out)
        _finite("speed", speed)
        if src_in < 0:
            raise invalid_timeline(f"start time cannot be negative ({src_in:.2f}s)")
        if src_out <= src_in:
            raise invalid_timeline(f"end time ({src_out:.2f}s) must be after start time ({src_in:.2f}s)")
        if src_out > dur + FRAME_TOL:
            raise invalid_timeline(f"end time {src_out:.2f}s is beyond the end of the clip ({dur:.2f}s)")
        if not (SPEED_MIN <= speed <= SPEED_MAX):
            raise invalid_timeline(f"speed {speed} must be between {SPEED_MIN} and {SPEED_MAX}")
        if (src_out - src_in) / speed < MIN_CLIP_S:
            raise invalid_timeline(f"clip is shorter than {MIN_CLIP_S}s")

    def _ripple(self) -> None:
        for t in self._tl().tracks:
            if t.kind != "video":
                continue
            pos = 0.0
            for i, c in enumerate(t.clips):
                if i > 0 and c.transition_in:
                    pos -= c.transition_in.duration
                elif i == 0:
                    c.transition_in = None  # the first clip can't overlap anything (use intro instead)
                c.timeline_start = round(pos, 6)
                pos += c.duration

    def _commit(self, op: str, args: dict) -> None:
        tl = self._tl()
        self._ripple()
        tl.revision += 1
        tl.oplog.append({"rev": tl.revision, "op": op, "args": args})
        if len(tl.oplog) > 2000:
            tl.oplog = tl.oplog[-2000:]

    @contextmanager
    def transaction(self):
        """All-or-nothing batch: on any exception the timeline is restored."""
        snapshot = copy.deepcopy(self.timeline)
        try:
            yield self
        except BaseException:
            self.timeline = snapshot
            raise

    # ------------------------------------------------------------ timeline & tracks
    @atomic
    def create_timeline(self, width: int, height: int, fps: int = 30, fit: Fit = "cover",
                        max_duration: float = 600.0) -> dict:
        if width % 2 or height % 2 or not (144 <= width <= 3840) or not (144 <= height <= 3840):
            raise invalid_timeline(f"output size {width}x{height} is not allowed")
        if fps not in ALLOWED_FPS:
            raise invalid_timeline(f"fps {fps} is not allowed")
        self.timeline = Timeline(
            timeline_id=f"t_{secrets.token_hex(4)}",
            settings=TimelineSettings(width=width, height=height, fps=fps, fit=fit, max_duration=max_duration),
            tracks=[Track(track_id="V1", kind="video"), Track(track_id="T1", kind="caption")],
        )
        self._commit("create_timeline", {"width": width, "height": height, "fps": fps, "fit": fit})
        return {"timeline_id": self.timeline.timeline_id}

    @atomic
    def add_video_track(self) -> dict:
        """Add an overlay (picture-in-picture) video track."""
        tl = self._tl()
        n = sum(1 for t in tl.tracks if t.kind in ("video", "overlay")) + 1
        if n > 4:
            raise invalid_timeline("at most 3 overlay tracks are supported")
        tid = f"V{n}"
        tl.tracks.append(Track(track_id=tid, kind="overlay"))
        self._commit("add_video_track", {})
        return {"track_id": tid}

    @atomic
    def add_audio_track(self) -> dict:
        tl = self._tl()
        n = sum(1 for t in tl.tracks if t.kind == "audio") + 1
        if n > 8:
            raise invalid_timeline("at most 8 audio tracks are supported")
        tid = f"A{n}"
        tl.tracks.append(Track(track_id=tid, kind="audio"))
        self._commit("add_audio_track", {})
        return {"track_id": tid}

    def _audio_track_id(self, track_id: str | None) -> str:
        tl = self._tl()
        if track_id:
            if self._track(track_id).kind != "audio":
                raise invalid_timeline(f"track '{track_id}' is not an audio track")
            return track_id
        for t in tl.tracks:
            if t.kind == "audio":
                return t.track_id
        return self.add_audio_track()["track_id"]

    # ------------------------------------------------------------ clips (main track)
    @atomic
    def select_clip(self, asset_id: str, src_in: float, src_out: float, track_id: str = "V1",
                    index: int | None = None, note: str = "", start: float | None = None,
                    x: float | None = None, y: float | None = None, width: float | None = None) -> dict:
        """Place a segment of an asset on a track (main track: insert/append; overlay: at `start`)."""
        asset = self._video_asset(asset_id)
        src_in, src_out = float(src_in), float(src_out)
        self._check_src(asset, src_in, src_out)
        src_out = min(src_out, float(asset["meta"]["duration"]))
        track = self._track(track_id)
        if track.kind == "overlay":
            return self._add_overlay(track_id, asset_id, src_in, src_out, start, x, y, width)
        if track.kind != "video":
            raise invalid_timeline(f"track '{track_id}' is not a video track")
        if len(track.clips) >= MAX_CLIPS:
            raise invalid_timeline(f"at most {MAX_CLIPS} clips are supported")
        clip = Clip(clip_id=_new_id("c"), asset_id=asset_id, src_in=round(src_in, 3),
                    src_out=round(src_out, 3), note=str(note)[:200])
        if index is None:
            track.clips.append(clip)
        else:
            if not (0 <= index <= len(track.clips)):
                raise invalid_timeline(f"position {index} is outside the track")
            track.clips.insert(index, clip)
        self._commit("select_clip", {"asset_id": asset_id, "src_in": src_in, "src_out": src_out,
                                     "track_id": track_id, "index": index})
        self._check_duration()
        return {"clip_id": clip.clip_id, "timeline_duration": self._tl().duration}

    def _add_overlay(self, track_id, asset_id, src_in, src_out, start, x, y, width) -> dict:
        tl = self._tl()
        if len(tl.overlays) >= MAX_OVERLAYS:
            raise invalid_timeline(f"at most {MAX_OVERLAYS} overlay clips are supported")
        start = 0.0 if start is None else float(start)
        _check_range("overlay start", start, 0, max(0.0, tl.duration - MIN_CLIP_S))
        item = OverlayItem(item_id=_new_id("o"), track_id=track_id, asset_id=asset_id, src_in=round(src_in, 3),
                           src_out=round(src_out, 3), start=round(start, 3))
        if width is not None:
            _check_range("overlay width", width, 0.1, 1.0)
            item.width = width
        for name, v in (("overlay x", x), ("overlay y", y)):
            if v is not None:
                _check_range(name, v, 0.0, 0.95)
        item.x = item.x if x is None else x
        item.y = item.y if y is None else y
        tl.overlays.append(item)
        self._commit("select_clip", {"overlay": item.item_id})
        return {"item_id": item.item_id}

    @atomic
    def trim_clip(self, clip_id: str, src_in: float | None = None, src_out: float | None = None) -> dict:
        tl = self._tl()
        for o in tl.overlays:
            if o.item_id == clip_id:
                new_in = o.src_in if src_in is None else float(src_in)
                new_out = o.src_out if src_out is None else float(src_out)
                self._check_src(self._asset(o.asset_id), new_in, new_out)
                o.src_in, o.src_out = round(new_in, 3), round(new_out, 3)
                self._commit("trim_clip", {"clip_id": clip_id})
                return {"clip_id": clip_id, "duration": round(o.duration, 3)}
        c = self._clip(clip_id)
        new_in = c.src_in if src_in is None else float(src_in)
        new_out = c.src_out if src_out is None else float(src_out)
        self._check_src(self._asset(c.asset_id), new_in, new_out, c.speed)
        c.src_in, c.src_out = round(new_in, 3), round(new_out, 3)
        self._commit("trim_clip", {"clip_id": clip_id, "src_in": src_in, "src_out": src_out})
        self._check_transitions()
        return {"clip_id": clip_id, "duration": round(c.duration, 3)}

    @atomic
    def split_clip(self, clip_id: str, at: float) -> dict:
        """Split at a source time strictly inside the clip. Zoom continues smoothly across the split."""
        track, i = self._find(clip_id)
        c = track.clips[i]
        _finite("split point", at)
        at = float(at)
        if not (c.src_in + MIN_CLIP_S * c.speed <= at <= c.src_out - MIN_CLIP_S * c.speed):
            raise invalid_timeline(f"split point {at:.2f}s must be inside the clip ({c.src_in:.2f}-{c.src_out:.2f}s)")
        f = (at - c.src_in) / (c.src_out - c.src_in)
        left_kf = right_kf = []
        if c.keyframes:
            k0, kf, k1 = sample_zoom(c.keyframes, 0, c.ease), sample_zoom(c.keyframes, f, c.ease), \
                sample_zoom(c.keyframes, 1, c.ease)
            left_kf = [k0.model_copy(update={"at": 0}), kf.model_copy(update={"at": 1})]
            right_kf = [kf.model_copy(update={"at": 0}), k1.model_copy(update={"at": 1})]
        right = c.model_copy(deep=True, update={"clip_id": _new_id("c"), "src_in": round(at, 3),
                                                "transition_in": None, "keyframes": right_kf, "ease": "linear"})
        c.src_out = round(at, 3)
        if c.keyframes:
            c.keyframes, c.ease = left_kf, "linear"
        track.clips.insert(i + 1, right)
        self._commit("split_clip", {"clip_id": clip_id, "at": at})
        self._check_transitions()
        return {"left": c.clip_id, "right": right.clip_id}

    @atomic
    def move_clip(self, clip_id: str, index: int | None = None, start: float | None = None) -> dict:
        tl = self._tl()
        for o in tl.overlays:
            if o.item_id == clip_id:
                if start is None:
                    raise invalid_timeline("overlay clips are moved with a new start time")
                _check_range("overlay start", start, 0, max(0.0, tl.duration - MIN_CLIP_S))
                o.start = round(float(start), 3)
                self._commit("move_clip", {"clip_id": clip_id, "start": start})
                return {"clip_id": clip_id, "start": o.start}
        if index is None:
            raise invalid_timeline("give the new position (index) for a main-track clip")
        track, i = self._find(clip_id)
        if not (0 <= index < len(track.clips)):
            raise invalid_timeline(f"position {index} is outside the track")
        track.clips.insert(index, track.clips.pop(i))
        self._commit("move_clip", {"clip_id": clip_id, "index": index})
        self._check_transitions()
        return {"clip_id": clip_id, "index": index}

    @atomic
    def delete_clip(self, clip_id: str) -> dict:
        """Delete any item: main clip, overlay clip, text/caption, music or sound effect."""
        tl = self._tl()
        for lst in (tl.overlays, tl.texts, tl.audio):
            for i, it in enumerate(lst):
                if it.item_id == clip_id:
                    lst.pop(i)
                    self._commit("delete_clip", {"clip_id": clip_id})
                    return {"deleted": clip_id}
        track, i = self._find(clip_id)
        track.clips.pop(i)
        self._commit("delete_clip", {"clip_id": clip_id})
        self._check_transitions()
        return {"deleted": clip_id, "timeline_duration": tl.duration}

    @atomic
    def change_speed(self, clip_id: str, speed: float) -> dict:
        c = self._clip(clip_id)
        _finite("speed", speed)
        self._check_src(self._asset(c.asset_id), c.src_in, c.src_out, float(speed))
        c.speed = float(speed)
        self._commit("change_speed", {"clip_id": clip_id, "speed": speed})
        self._check_duration()
        self._check_transitions()
        return {"clip_id": clip_id, "speed": c.speed, "duration": round(c.duration, 3)}

    @atomic
    def speed_ramp(self, clip_id: str, pattern: str, keep_duration: bool = True) -> dict:
        """Stepped speed ramp: split the clip and give each part its own speed."""
        if pattern not in SPEED_RAMPS:
            raise invalid_timeline(f"unknown speed ramp '{pattern}' (use one of {', '.join(SPEED_RAMPS)})")
        track, i = self._find(clip_id)
        c = track.clips[i]
        if c.speed != 1.0:
            raise invalid_timeline("speed ramps can only be applied to clips at normal speed")
        steps = SPEED_RAMPS[pattern]
        asset = self._asset(c.asset_id)
        if keep_duration:  # choose a source range so the ramped clip keeps its length
            factor = sum(f / s for f, s in steps)
            out_len = c.duration
            need = out_len / factor
            dur = float(asset["meta"]["duration"])
            if need <= dur:
                mid = (c.src_in + c.src_out) / 2
                a = min(max(0.0, mid - need / 2), dur - need)
                c.src_in, c.src_out = round(a, 3), round(a + need, 3)
        total = c.src_out - c.src_in
        parts: list[Clip] = []
        pos, frac = c.src_in, 0.0
        out_total = sum(f * total / s for f, s in steps)
        out_pos = 0.0
        for n, (f, s) in enumerate(steps):
            seg_src = total * f
            seg_out = seg_src / s
            if seg_out < MIN_CLIP_S:
                raise invalid_timeline("the clip is too short for this speed ramp")
            p = c.model_copy(deep=True, update={
                "clip_id": c.clip_id if n == 0 else _new_id("c"), "src_in": round(pos, 3),
                "src_out": round(c.src_out if n == len(steps) - 1 else pos + seg_src, 3), "speed": s,
                "transition_in": c.transition_in if n == 0 else None})
            if c.keyframes:  # continue the zoom curve across the parts
                a0, a1 = out_pos / out_total, (out_pos + seg_out) / out_total
                k0, k1 = sample_zoom(c.keyframes, a0, c.ease), sample_zoom(c.keyframes, a1, c.ease)
                p.keyframes, p.ease = [k0.model_copy(update={"at": 0}), k1.model_copy(update={"at": 1})], "linear"
            parts.append(p)
            pos += seg_src
            out_pos += seg_out
            frac += f
        track.clips[i:i + 1] = parts
        self._commit("speed_ramp", {"clip_id": clip_id, "pattern": pattern})
        self._check_duration()
        self._check_transitions()
        return {"clips": [p.clip_id for p in parts], "duration": round(sum(p.duration for p in parts), 3)}

    @atomic
    def set_volume(self, clip_id: str, volume: float) -> dict:
        _check_range("volume", volume, VOLUME_MIN, VOLUME_MAX)
        for it in self._tl().audio:
            if it.item_id == clip_id:
                it.volume = float(volume)
                self._commit("set_volume", {"clip_id": clip_id, "volume": volume})
                return {"item_id": clip_id, "volume": volume}
        for c in self._clips_for(clip_id):
            c.volume = float(volume)
        self._commit("set_volume", {"clip_id": clip_id, "volume": volume})
        return {"clip_id": clip_id, "volume": volume}

    # ------------------------------------------------------------ visual effects
    @atomic
    def add_zoom(self, clip_id: str, kind: str = "in", amount: float = 0.15) -> dict:
        if kind not in ZOOM_KINDS:
            raise invalid_timeline(f"unknown zoom '{kind}' (use one of {', '.join(ZOOM_KINDS)})")
        _check_range("zoom amount", amount, 0.02, 1.0)
        for c in self._clips_for(clip_id):
            c.keyframes = zoom_keyframes(kind, amount)
            c.ease = "linear" if kind == "punch_in" else "inout"
        self._commit("add_zoom", {"clip_id": clip_id, "kind": kind, "amount": amount})
        return {"clip_id": clip_id, "zoom": kind}

    @atomic
    def add_keyframe(self, clip_id: str, t: float, scale: float = 1.0, x: float = 0.5, y: float = 0.5) -> dict:
        c = self._clip(clip_id)
        _check_range("keyframe time", t, 0, c.duration + 1e-6)
        _check_range("scale", scale, 1.0, 3.0)
        _check_range("x", x, 0.0, 1.0)
        _check_range("y", y, 0.0, 1.0)
        at = round(t / c.duration, 6) if c.duration else 0.0
        c.keyframes = [k for k in c.keyframes if abs(k.at - at) > 1e-6] + [Keyframe(at=at, scale=scale, x=x, y=y)]
        c.keyframes.sort(key=lambda k: k.at)
        if len(c.keyframes) > 50:
            raise invalid_timeline("at most 50 keyframes per clip")
        self._commit("add_keyframe", {"clip_id": clip_id, "t": t, "scale": scale, "x": x, "y": y})
        return {"clip_id": clip_id, "keyframes": len(c.keyframes)}

    @atomic
    def apply_color_adjustment(self, clip_id: str | None = "all", preset: str = "none", brightness: float = 0.0,
                               contrast: float = 1.0, saturation: float = 1.0, gamma: float = 1.0,
                               warmth: float = 0.0) -> dict:
        if preset not in COLOR_PRESETS:
            raise invalid_timeline(f"unknown colour preset '{preset}' (use one of {', '.join(COLOR_PRESETS)})")
        _check_range("brightness", brightness, -0.5, 0.5)
        _check_range("contrast", contrast, 0.5, 2.0)
        _check_range("saturation", saturation, 0.0, 3.0)
        _check_range("gamma", gamma, 0.5, 2.0)
        _check_range("warmth", warmth, -1.0, 1.0)
        adj = ColorAdjust(preset=preset, brightness=brightness, contrast=contrast, saturation=saturation,
                          gamma=gamma, warmth=warmth)
        for c in self._clips_for(clip_id):
            c.color = None if adj == ColorAdjust() else adj.model_copy()
        self._commit("apply_color_adjustment", {"clip_id": clip_id, **adj.model_dump()})
        return {"clip_id": clip_id, "color": adj.model_dump()}

    @atomic
    def apply_effect(self, clip_id: str | None, effect: str, remove: bool = False) -> dict:
        if effect not in EFFECTS:
            raise invalid_timeline(f"unknown effect '{effect}' (use one of {', '.join(EFFECTS)})")
        for c in self._clips_for(clip_id):
            if remove:
                c.effects = [e for e in c.effects if e != effect]
            elif effect not in c.effects:
                c.effects.append(effect)
        self._commit("apply_effect", {"clip_id": clip_id, "effect": effect, "remove": remove})
        return {"clip_id": clip_id, "effect": effect, "removed": remove}

    @atomic
    def crop_video(self, clip_id: str, x: float, y: float, w: float, h: float) -> dict:
        for name, v in (("x", x), ("y", y)):
            _check_range(f"crop {name}", v, 0.0, 0.9)
        _check_range("crop width", w, 0.1, 1.0)
        _check_range("crop height", h, 0.1, 1.0)
        if x + w > 1.0 + 1e-6 or y + h > 1.0 + 1e-6:
            raise invalid_timeline("the crop area goes outside the picture")
        for c in self._clips_for(clip_id):
            c.crop = Crop(x=x, y=y, w=w, h=h)
        self._commit("crop_video", {"clip_id": clip_id, "x": x, "y": y, "w": w, "h": h})
        return {"clip_id": clip_id}

    @atomic
    def rotate_video(self, clip_id: str, degrees: int) -> dict:
        if degrees not in (0, 90, 180, 270, -90):
            raise invalid_timeline("rotation must be 0, 90, 180 or 270 degrees")
        for c in self._clips_for(clip_id):
            c.rotate = degrees % 360
        self._commit("rotate_video", {"clip_id": clip_id, "degrees": degrees})
        return {"clip_id": clip_id, "rotate": degrees % 360}

    @atomic
    def add_transition(self, clip_id: str | None = None, type: str = "fade", duration: float = 0.5,
                       position: Literal["before", "start", "end"] = "before") -> dict:
        """position=before: between this clip and the previous one; start/end: fade in/out of the video."""
        tl = self._tl()
        _check_range("transition duration", duration, 0.1, 3.0)
        if position in ("start", "end"):
            if type not in INTRO_OUTRO:
                raise invalid_timeline(f"the start/end of the video can only use {', '.join(INTRO_OUTRO)}")
            if duration > tl.duration / 2:
                raise invalid_timeline("the fade is longer than half the video")
            setattr(tl.settings, "intro" if position == "start" else "outro", Transition(type=type, duration=duration))
            self._commit("add_transition", {"position": position, "type": type, "duration": duration})
            return {"position": position, "type": type}
        if type not in TRANSITIONS:
            raise invalid_timeline(f"unknown transition '{type}' (use one of {', '.join(TRANSITIONS)})")
        track, i = self._find(clip_id)
        if i == 0:
            raise invalid_timeline("the first clip has nothing before it; use position 'start' for a fade-in")
        track.clips[i].transition_in = Transition(type=type, duration=round(duration, 3))
        self._commit("add_transition", {"clip_id": clip_id, "type": type, "duration": duration})
        self._check_transitions()
        return {"clip_id": clip_id, "type": type, "duration": duration}

    @atomic
    def remove_transition(self, clip_id: str) -> dict:
        self._clip(clip_id).transition_in = None
        self._commit("remove_transition", {"clip_id": clip_id})
        return {"clip_id": clip_id}

    @atomic
    def resize_video(self, width: int, height: int, fit: Fit = "cover") -> dict:
        tl = self._tl()
        if width % 2 or height % 2 or not (144 <= width <= 3840) or not (144 <= height <= 3840):
            raise invalid_timeline(f"output size {width}x{height} is not allowed")
        if fit not in ("cover", "contain", "blur"):
            raise invalid_timeline(f"unknown framing '{fit}'")
        tl.settings.width, tl.settings.height, tl.settings.fit = width, height, fit
        self._commit("resize_video", {"width": width, "height": height, "fit": fit})
        return {"width": width, "height": height, "fit": fit}

    # ------------------------------------------------------------ text
    @atomic
    def add_caption(self, text: str, start: float, end: float, style: str = "subtitle",
                    position: str = "bottom", kind: Literal["caption", "text"] = "caption") -> dict:
        tl = self._tl()
        if style not in TEXT_STYLES:
            raise invalid_timeline(f"unknown text style '{style}' (use one of {', '.join(TEXT_STYLES)})")
        if position not in TEXT_POSITIONS:
            raise invalid_timeline(f"unknown position '{position}' (use one of {', '.join(TEXT_POSITIONS)})")
        _finite("start", start)
        _finite("end", end)
        if start < 0 or end <= start:
            raise invalid_timeline(f"text times {start:.2f}-{end:.2f}s are invalid")
        if tl.duration and start >= tl.duration:
            raise invalid_timeline(f"text starts at {start:.2f}s, after the end of the video ({tl.duration:.2f}s)")
        if len(tl.texts) >= MAX_TEXT_ITEMS:
            raise invalid_timeline(f"at most {MAX_TEXT_ITEMS} captions/texts are supported")
        item = TextItem(item_id=_new_id("x"), kind=kind, text=clean_text(text), start=round(start, 3),
                        end=round(min(end, tl.duration) if tl.duration else end, 3), style=style, position=position)
        tl.texts.append(item)
        tl.texts.sort(key=lambda t: t.start)
        self._commit("add_caption" if kind == "caption" else "add_text", {"text": item.text, "start": start, "end": end})
        return {"item_id": item.item_id}

    def add_text(self, text: str, start: float, end: float, style: str = "title", position: str = "center") -> dict:
        return self.add_caption(text, start, end, style, position, kind="text")

    @atomic
    def clear_captions(self, kind: str = "caption") -> dict:
        tl = self._tl()
        n = len(tl.texts)
        tl.texts = [t for t in tl.texts if t.kind != kind]
        self._commit("clear_captions", {"kind": kind})
        return {"removed": n - len(tl.texts)}

    # ------------------------------------------------------------ audio
    @atomic
    def add_music(self, asset_id: str, start: float = 0.0, src_in: float = 0.0, duration: float | None = None,
                  volume: float = 0.8, fade_in: float = 0.5, fade_out: float = 1.5, loop: bool = True,
                  duck: bool = True, track_id: str | None = None) -> dict:
        tl = self._tl()
        a = self._asset(asset_id)
        if a.get("role") != "music" or not a["meta"].get("has_audio"):
            raise invalid_timeline("only uploaded music/audio files can be used as music")
        mdur = float(a["meta"]["duration"])
        _check_range("music start", start, 0, max(0.0, tl.duration - 0.1))
        _check_range("music offset", src_in, 0, max(0.0, mdur - 0.1))
        if duration is not None:
            _check_range("music duration", duration, 0.1, tl.settings.max_duration)
        _check_range("music volume", volume, VOLUME_MIN, VOLUME_MAX)
        _check_range("fade in", fade_in, 0, 10)
        _check_range("fade out", fade_out, 0, 10)
        if len(tl.audio) >= MAX_AUDIO_ITEMS:
            raise invalid_timeline("too many audio items")
        item = AudioItem(item_id=_new_id("m"), track_id=self._audio_track_id(track_id), kind="music",
                         asset_id=asset_id, start=round(start, 3), src_in=round(src_in, 3), duration=duration,
                         volume=volume, fade_in=fade_in, fade_out=fade_out, loop=loop, duck=duck)
        tl.audio.append(item)
        self._commit("add_music", {"asset_id": asset_id, "start": start})
        return {"item_id": item.item_id, "track_id": item.track_id}

    @atomic
    def add_sound_effect(self, at: float, sfx: str | None = None, asset_id: str | None = None,
                         volume: float = 0.8, track_id: str | None = None) -> dict:
        tl = self._tl()
        if (sfx is None) == (asset_id is None):
            raise invalid_timeline("give either a built-in sound effect name or an uploaded sound")
        if sfx is not None and sfx not in SFX_NAMES:
            raise invalid_timeline(f"unknown sound effect '{sfx}' (use one of {', '.join(SFX_NAMES)})")
        if asset_id is not None:
            a = self._asset(asset_id)
            if a.get("role") != "music" or not a["meta"].get("has_audio"):
                raise invalid_timeline("sound effects must be audio files")
        _check_range("sound effect time", at, 0, max(0.0, tl.duration - 0.05))
        _check_range("sound effect volume", volume, VOLUME_MIN, VOLUME_MAX)
        if len(tl.audio) >= MAX_AUDIO_ITEMS:
            raise invalid_timeline("too many audio items")
        item = AudioItem(item_id=_new_id("s"), track_id=self._audio_track_id(track_id), kind="sfx", sfx=sfx,
                         asset_id=asset_id, start=round(at, 3), volume=volume)
        tl.audio.append(item)
        self._commit("add_sound_effect", {"sfx": sfx, "asset_id": asset_id, "at": at})
        return {"item_id": item.item_id}

    @atomic
    def set_audio_settings(self, normalize: bool = True) -> dict:
        self._tl().settings.normalize_audio = bool(normalize)
        self._commit("set_audio_settings", {"normalize": normalize})
        return {"normalize": normalize}

    # ------------------------------------------------------------ beats
    @atomic
    def snap_to_beats(self, beats: list[float], max_shift: float | None = None) -> dict:
        """Move each cut (and the end) onto the nearest beat by trimming clip ends."""
        tl = self._tl()
        clips = tl.video_track().clips
        beats = sorted(b for b in beats if b >= 0)
        if len(beats) < 2 or not clips:
            return {"snapped": 0, "total": len(clips)}
        interval = (beats[-1] - beats[0]) / (len(beats) - 1)
        max_shift = interval * 0.5 if max_shift is None else max_shift
        snapped = 0
        for i, c in enumerate(clips):
            self._ripple()
            nxt = clips[i + 1] if i + 1 < len(clips) else None
            # the visible cut is in the middle of a transition
            half = nxt.transition_in.duration / 2 if nxt and nxt.transition_in else 0.0
            cut = c.timeline_start + c.duration - half
            b = min(beats, key=lambda t: abs(t - cut))
            delta = b - cut
            if abs(delta) < 1e-3 or abs(delta) > max_shift:
                continue
            new_out = c.src_out + delta * c.speed
            dur = float(self._asset(c.asset_id)["meta"]["duration"])
            if new_out > dur + FRAME_TOL or (new_out - c.src_in) / c.speed < MIN_CLIP_S:
                continue
            c.src_out = round(min(new_out, dur), 3)
            snapped += 1
        self._commit("snap_to_beats", {"beats": len(beats)})
        self._check_transitions()
        return {"snapped": snapped, "total": len(clips)}

    # ------------------------------------------------------------ validation
    def _check_duration(self) -> None:
        tl = self._tl()
        if tl.duration > tl.settings.max_duration + 0.01:
            raise invalid_timeline(
                f"total length {tl.duration:.1f}s exceeds the {tl.settings.max_duration:.0f}s limit for this preset")

    def _check_transitions(self) -> None:
        clips = self._tl().video_track().clips
        for i, c in enumerate(clips):
            d_in = c.transition_in.duration if c.transition_in else 0.0
            d_out = clips[i + 1].transition_in.duration if i + 1 < len(clips) and clips[i + 1].transition_in else 0.0
            if d_in + d_out > c.duration - 0.1 + 1e-6:
                raise invalid_timeline(
                    f"clip {i + 1} ({c.duration:.2f}s) is too short for its transitions ({d_in + d_out:.2f}s)")

    def validate_timeline(self) -> dict:
        """Full consistency check. Raises on the first error; returns a summary."""
        tl = self._tl()
        self._ripple()
        v = tl.video_track()
        if not v.clips:
            raise invalid_timeline("the timeline is empty")
        pos = 0.0
        ids = set()
        for i, c in enumerate(v.clips):
            if c.clip_id in ids:
                raise invalid_timeline(f"duplicate clip id {c.clip_id}")
            ids.add(c.clip_id)
            asset = self._video_asset(c.asset_id)
            self._check_src(asset, c.src_in, c.src_out, c.speed)
            _check_range("volume", c.volume, VOLUME_MIN, VOLUME_MAX)
            if c.transition_in:
                if c.transition_in.type not in TRANSITIONS:
                    raise invalid_timeline(f"unknown transition {c.transition_in.type}")
                pos -= c.transition_in.duration
            if abs(c.timeline_start - pos) > 1e-3:
                raise invalid_timeline(f"clip {c.clip_id} has a gap or overlap at {pos:.2f}s")
            for k in c.keyframes:
                _check_range("keyframe", k.at, 0, 1)
                _check_range("zoom scale", k.scale, 1.0, 3.0)
            if c.color and c.color.preset not in COLOR_PRESETS:
                raise invalid_timeline(f"unknown colour preset {c.color.preset}")
            for e in c.effects:
                if e not in EFFECTS:
                    raise invalid_timeline(f"unknown effect {e}")
            pos += c.duration
        self._check_transitions()
        if tl.duration <= 0:
            raise invalid_timeline("the timeline has zero length")
        self._check_duration()
        for fade in (tl.settings.intro, tl.settings.outro):
            if fade and fade.duration > tl.duration / 2:
                raise invalid_timeline("a fade is longer than half the video")
        for t in tl.texts:
            if t.start >= tl.duration - 0.05:
                raise invalid_timeline(f"text '{t.text[:20]}' starts after the end of the video")
            t.end = min(t.end, tl.duration)
        for a in tl.audio:
            if a.start >= tl.duration:
                raise invalid_timeline(f"audio item {a.item_id} starts after the end of the video")
            if a.asset_id:
                self._asset(a.asset_id)
        for o in tl.overlays:
            self._check_src(self._video_asset(o.asset_id), o.src_in, o.src_out)
            if o.start >= tl.duration:
                raise invalid_timeline(f"overlay {o.item_id} starts after the end of the video")
        return {"ok": True, "clips": len(v.clips), "duration": round(tl.duration, 3),
                "width": tl.settings.width, "height": tl.settings.height, "revision": tl.revision,
                "texts": len(tl.texts), "audio_items": len(tl.audio), "overlays": len(tl.overlays),
                "transitions": sum(1 for c in v.clips if c.transition_in)}

    def summary(self) -> dict:
        """Compact description of the timeline for an agent."""
        tl = self._tl()
        self._ripple()
        return {
            "duration": round(tl.duration, 3),
            "settings": tl.settings.model_dump(),
            "clips": [{"clip_id": c.clip_id, "asset_id": c.asset_id, "src_in": c.src_in, "src_out": c.src_out,
                       "start": round(c.timeline_start, 3), "duration": round(c.duration, 3), "speed": c.speed,
                       "volume": c.volume, "zoom": bool(c.keyframes),
                       "transition_in": c.transition_in.model_dump() if c.transition_in else None,
                       "color": c.color.preset if c.color else None, "effects": c.effects}
                      for c in tl.video_track().clips],
            "texts": [t.model_dump() for t in tl.texts[:100]],
            "audio": [a.model_dump() for a in tl.audio],
            "overlays": [o.model_dump() for o in tl.overlays],
            "tracks": [{"track_id": t.track_id, "kind": t.kind} for t in tl.tracks],
        }
