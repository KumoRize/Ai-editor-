"""Editing engine (component C): timeline model, validated operations, transactions.

The timeline is the single source of truth for rendering. Every mutation goes
through `TimelineEditor`, which validates inputs against real asset metadata and
applies changes transactionally (copy → mutate → validate → commit, or rollback).

Phase 1 timeline model is "magnetic": clips on the main video track play back to
back, so `timeline_start` is always derived (no gaps, no overlaps).
"""
from __future__ import annotations

import copy
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

Fit = Literal["cover", "contain", "blur"]


class Clip(BaseModel):
    clip_id: str
    asset_id: str
    src_in: float
    src_out: float
    speed: float = 1.0
    volume: float = 1.0
    timeline_start: float = 0.0
    note: str = ""

    @property
    def duration(self) -> float:
        return (self.src_out - self.src_in) / self.speed


class Track(BaseModel):
    track_id: str
    kind: Literal["video", "audio", "caption"]
    clips: list[Clip] = Field(default_factory=list)


class TimelineSettings(BaseModel):
    width: int
    height: int
    fps: int = 30
    fit: Fit = "cover"
    bg_color: str = "black"
    max_duration: float = 600.0


class Timeline(BaseModel):
    timeline_id: str
    revision: int = 0
    settings: TimelineSettings
    tracks: list[Track] = Field(default_factory=list)
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


def _new_clip_id() -> str:
    return f"c_{secrets.token_hex(4)}"


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

    def _find(self, clip_id: str) -> tuple[Track, int]:
        for t in self._tl().tracks:
            for i, c in enumerate(t.clips):
                if c.clip_id == clip_id:
                    return t, i
        raise EditorError("CLIP_NOT_FOUND", f"Clip '{clip_id}' does not exist.")

    def _track(self, track_id: str) -> Track:
        for t in self._tl().tracks:
            if t.track_id == track_id:
                return t
        raise EditorError("TRACK_NOT_FOUND", f"Track '{track_id}' does not exist.")

    def _check_range(self, asset: dict, src_in: float, src_out: float, speed: float = 1.0) -> None:
        dur = float(asset["meta"]["duration"])
        for name, v in (("start", src_in), ("end", src_out), ("speed", speed)):
            if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
                raise invalid_timeline(f"{name} is not a number")
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
            for c in t.clips:
                c.timeline_start = round(pos, 6)
                pos += c.duration

    def _commit(self, op: str, args: dict) -> None:
        tl = self._tl()
        self._ripple()
        tl.revision += 1
        tl.oplog.append({"rev": tl.revision, "op": op, "args": args})

    @contextmanager
    def transaction(self):
        """All-or-nothing batch: on any exception the timeline is restored."""
        snapshot = copy.deepcopy(self.timeline)
        try:
            yield self
        except BaseException:
            self.timeline = snapshot
            raise

    # ------------------------------------------------------------ operations
    def create_timeline(self, width: int, height: int, fps: int = 30, fit: Fit = "cover",
                        max_duration: float = 600.0) -> dict:
        if width % 2 or height % 2 or not (144 <= width <= 3840) or not (144 <= height <= 3840):
            raise invalid_timeline(f"output size {width}x{height} is not allowed")
        if fps not in ALLOWED_FPS:
            raise invalid_timeline(f"fps {fps} is not allowed")
        self.timeline = Timeline(
            timeline_id=f"t_{secrets.token_hex(4)}",
            settings=TimelineSettings(width=width, height=height, fps=fps, fit=fit, max_duration=max_duration),
            tracks=[Track(track_id="V1", kind="video")],
        )
        self._commit("create_timeline", {"width": width, "height": height, "fps": fps, "fit": fit})
        return {"timeline_id": self.timeline.timeline_id}

    def add_video_track(self) -> dict:
        tl = self._tl()
        n = sum(1 for t in tl.tracks if t.kind == "video") + 1
        if n > 1:
            raise EditorError("NOT_AVAILABLE", "Multiple video tracks arrive in Phase 2.")
        tl.tracks.append(Track(track_id=f"V{n}", kind="video"))
        self._commit("add_video_track", {})
        return {"track_id": f"V{n}"}

    def select_clip(self, asset_id: str, src_in: float, src_out: float, track_id: str = "V1",
                    index: int | None = None, note: str = "") -> dict:
        """Insert a segment of an asset onto a track (append by default)."""
        asset = self._asset(asset_id)
        if asset.get("role") != "clip":
            raise invalid_timeline("reference and music files cannot be placed on the video track")
        if not asset["meta"].get("has_video"):
            raise invalid_timeline(f"media '{asset_id}' has no video")
        src_in, src_out = float(src_in), float(src_out)
        self._check_range(asset, src_in, src_out)
        src_out = min(src_out, float(asset["meta"]["duration"]))
        track = self._track(track_id)
        if track.kind != "video":
            raise invalid_timeline(f"track '{track_id}' is not a video track")
        clip = Clip(clip_id=_new_clip_id(), asset_id=asset_id, src_in=round(src_in, 3),
                    src_out=round(src_out, 3), note=note[:200])
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

    def trim_clip(self, clip_id: str, src_in: float | None = None, src_out: float | None = None) -> dict:
        track, i = self._find(clip_id)
        c = track.clips[i]
        new_in = c.src_in if src_in is None else float(src_in)
        new_out = c.src_out if src_out is None else float(src_out)
        self._check_range(self._asset(c.asset_id), new_in, new_out, c.speed)
        c.src_in, c.src_out = round(new_in, 3), round(new_out, 3)
        self._commit("trim_clip", {"clip_id": clip_id, "src_in": src_in, "src_out": src_out})
        return {"clip_id": clip_id, "duration": round(c.duration, 3)}

    def split_clip(self, clip_id: str, at: float) -> dict:
        """Split at a source time strictly inside the clip."""
        track, i = self._find(clip_id)
        c = track.clips[i]
        at = float(at)
        if not (c.src_in + MIN_CLIP_S <= at <= c.src_out - MIN_CLIP_S):
            raise invalid_timeline(f"split point {at:.2f}s must be inside the clip ({c.src_in:.2f}-{c.src_out:.2f}s)")
        right = c.model_copy(update={"clip_id": _new_clip_id(), "src_in": round(at, 3)})
        c.src_out = round(at, 3)
        track.clips.insert(i + 1, right)
        self._commit("split_clip", {"clip_id": clip_id, "at": at})
        return {"left": c.clip_id, "right": right.clip_id}

    def move_clip(self, clip_id: str, index: int) -> dict:
        track, i = self._find(clip_id)
        if not (0 <= index < len(track.clips)):
            raise invalid_timeline(f"position {index} is outside the track")
        track.clips.insert(index, track.clips.pop(i))
        self._commit("move_clip", {"clip_id": clip_id, "index": index})
        return {"clip_id": clip_id, "index": index}

    def delete_clip(self, clip_id: str) -> dict:
        track, i = self._find(clip_id)
        track.clips.pop(i)
        self._commit("delete_clip", {"clip_id": clip_id})
        return {"deleted": clip_id, "timeline_duration": self._tl().duration}

    def resize_video(self, width: int, height: int, fit: Fit = "cover") -> dict:
        tl = self._tl()
        if width % 2 or height % 2 or not (144 <= width <= 3840) or not (144 <= height <= 3840):
            raise invalid_timeline(f"output size {width}x{height} is not allowed")
        if fit not in ("cover", "contain", "blur"):
            raise invalid_timeline(f"unknown framing '{fit}'")
        tl.settings.width, tl.settings.height, tl.settings.fit = width, height, fit
        self._commit("resize_video", {"width": width, "height": height, "fit": fit})
        return {"width": width, "height": height, "fit": fit}

    def set_volume(self, clip_id: str, volume: float) -> dict:
        track, i = self._find(clip_id)
        volume = float(volume)
        if not (VOLUME_MIN <= volume <= VOLUME_MAX):
            raise invalid_timeline(f"volume {volume} must be between {VOLUME_MIN} and {VOLUME_MAX}")
        track.clips[i].volume = volume
        self._commit("set_volume", {"clip_id": clip_id, "volume": volume})
        return {"clip_id": clip_id, "volume": volume}

    # ------------------------------------------------------------ validation
    def _check_duration(self) -> None:
        tl = self._tl()
        if tl.duration > tl.settings.max_duration + 0.01:
            raise invalid_timeline(
                f"total length {tl.duration:.1f}s exceeds the {tl.settings.max_duration:.0f}s limit for this preset")

    def validate_timeline(self) -> dict:
        """Full consistency check. Raises on the first error; returns a summary."""
        tl = self._tl()
        v = tl.video_track()
        if not v.clips:
            raise invalid_timeline("the timeline is empty")
        pos = 0.0
        ids = set()
        for c in v.clips:
            if c.clip_id in ids:
                raise invalid_timeline(f"duplicate clip id {c.clip_id}")
            ids.add(c.clip_id)
            asset = self._asset(c.asset_id)
            if asset.get("role") != "clip":
                raise invalid_timeline("reference and music files cannot be placed on the video track")
            self._check_range(asset, c.src_in, c.src_out, c.speed)
            if not (VOLUME_MIN <= c.volume <= VOLUME_MAX):
                raise invalid_timeline(f"clip {c.clip_id} volume out of range")
            if abs(c.timeline_start - pos) > 1e-3:
                raise invalid_timeline(f"clip {c.clip_id} has a gap or overlap at {pos:.2f}s")
            pos += c.duration
        if tl.duration <= 0:
            raise invalid_timeline("the timeline has zero length")
        self._check_duration()
        return {"ok": True, "clips": len(v.clips), "duration": round(tl.duration, 3),
                "width": tl.settings.width, "height": tl.settings.height, "revision": tl.revision}
