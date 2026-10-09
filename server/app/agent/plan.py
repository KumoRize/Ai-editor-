"""The structured edit plan the agent produces before touching the timeline,
and the validator that checks it against real media metadata."""
from __future__ import annotations

import copy

from pydantic import BaseModel, Field

from ..engine.timeline import MIN_CLIP_S
from ..presets import FITS, PRESETS, SIZES

DEFAULT_PRESET_FOR_ASPECT = {"9:16": "instagram_reels", "16:9": "youtube_landscape", "1:1": "square"}


class ProjectSettings(BaseModel):
    preset: str
    aspect_ratio: str
    resolution: str
    duration: float | None  # None = whatever the chosen segments add up to
    fit: str

    def describe(self) -> str:
        length = f"{self.duration:g}s" if self.duration else "length from your cuts"
        return (f"{PRESETS[self.preset]['label']} · {self.aspect_ratio} · {self.resolution} · {length} · "
                f"{FITS[self.fit].lower()}")


def resolve_settings(preset: str | None = None, aspect_ratio: str | None = None, resolution: str | None = None,
                     duration: float | None = None, fit: str | None = None,
                     auto_length: bool = False) -> tuple[ProjectSettings, list[str]]:
    """Fill gaps with sensible defaults and clamp to what the preset allows."""
    notes: list[str] = []
    if preset not in PRESETS:
        preset = DEFAULT_PRESET_FOR_ASPECT.get(aspect_ratio or "", "instagram_reels")
    aspect_ratio = aspect_ratio if aspect_ratio in DEFAULT_PRESET_FOR_ASPECT else PRESETS[preset]["aspect_ratio"]
    resolution = resolution if (aspect_ratio, resolution) in SIZES else "1080p"
    fit = fit if fit in FITS else "cover"
    max_d = float(PRESETS[preset]["max_duration"])
    if duration is not None and duration <= 0:
        duration = None
    if duration is None and not auto_length:
        duration = float(PRESETS[preset]["default_duration"])
    if duration is not None and duration > max_d:
        notes.append(f"{PRESETS[preset]['label']} allows at most {max_d:g}s, so the edit is {max_d:g}s.")
        duration = max_d
    return ProjectSettings(preset=preset, aspect_ratio=aspect_ratio, resolution=resolution,
                           duration=duration, fit=fit), notes


class PlanSegment(BaseModel):
    asset_id: str
    src_in: float
    src_out: float
    reason: str = ""


class EditPlan(BaseModel):
    summary: str = ""                     # one short user-facing sentence
    segments: list[PlanSegment] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)  # parts of the prompt we can't do yet
    original_audio_volume: float = 1.0    # 0 = mute the clip's own sound
    settings: dict | None = None          # output settings chosen from the prompt ("Full prompt" mode)
    notes: list[str] = Field(default_factory=list)  # user-facing notes about how the prompt was followed
    planner: str = ""                     # filled in by the orchestrator

    @property
    def duration(self) -> float:
        return sum(max(0.0, s.src_out - s.src_in) for s in self.segments)


# JSON Schema used for structured output from any LLM provider. Kept to the
# subset every provider supports (no numeric bounds) - bounds are enforced by
# validate_plan() and again by the engine.
PLAN_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One short sentence describing the edit for the user."},
        "segments": {
            "type": "array",
            "description": "Ordered list of source segments to play back to back.",
            "items": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "src_in": {"type": "number", "description": "Start time in the source, seconds."},
                    "src_out": {"type": "number", "description": "End time in the source, seconds."},
                    "reason": {"type": "string", "description": "Very short reason (max 10 words)."},
                },
                "required": ["asset_id", "src_in", "src_out", "reason"],
                "additionalProperties": False,
            },
        },
        "unsupported_requests": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Parts of the user's request that cannot be done with the available operations.",
        },
        "original_audio_volume": {
            "type": "number",
            "description": "Volume of the clip's own sound: 1 = unchanged, 0 = muted, up to 2 = louder.",
        },
    },
    "required": ["summary", "segments", "unsupported_requests", "original_audio_volume"],
    "additionalProperties": False,
}

# "Full prompt" mode: the model also chooses the output settings from the prompt.
PROMPT_MODE_SCHEMA: dict = copy.deepcopy(PLAN_JSON_SCHEMA)
PROMPT_MODE_SCHEMA["properties"]["settings"] = {
    "type": "object",
    "description": "Output settings inferred from the user's prompt.",
    "properties": {
        "preset": {"type": "string", "enum": list(PRESETS)},
        "aspect_ratio": {"type": "string", "enum": ["9:16", "16:9", "1:1"]},
        "resolution": {"type": "string", "enum": ["720p", "1080p"]},
        "duration_s": {"type": "number",
                       "description": "Target output length in seconds; 0 = whatever the segments add up to."},
        "fit": {"type": "string", "enum": list(FITS)},
    },
    "required": ["preset", "aspect_ratio", "resolution", "duration_s", "fit"],
    "additionalProperties": False,
}
PROMPT_MODE_SCHEMA["required"] = PLAN_JSON_SCHEMA["required"] + ["settings"]


def validate_plan(plan: EditPlan, assets: dict[str, dict], target_duration: float | None,
                  max_duration: float) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors make the plan unusable."""
    errors: list[str] = []
    warnings: list[str] = []
    if not plan.segments:
        errors.append("the plan has no segments")
    if not (0.0 <= plan.original_audio_volume <= 2.0):
        errors.append(f"original_audio_volume {plan.original_audio_volume} must be between 0 and 2")
    if len(plan.segments) > 120:
        errors.append(f"too many segments ({len(plan.segments)}, max 120)")
    for i, s in enumerate(plan.segments):
        a = assets.get(s.asset_id)
        tag = f"segment {i + 1}"
        if a is None:
            errors.append(f"{tag}: unknown asset_id '{s.asset_id}'")
            continue
        if a.get("role") != "clip":
            errors.append(f"{tag}: asset '{s.asset_id}' is not a usable clip (reference/music cannot be placed)")
            continue
        dur = float(a["meta"]["duration"])
        if s.src_in < 0:
            errors.append(f"{tag}: src_in {s.src_in} is negative")
        if s.src_out <= s.src_in:
            errors.append(f"{tag}: src_out {s.src_out} must be greater than src_in {s.src_in}")
        if s.src_out > dur + 1 / 30:
            errors.append(f"{tag}: src_out {s.src_out} is beyond the clip length {dur:.3f}")
        if 0 < s.src_out - s.src_in < MIN_CLIP_S:
            errors.append(f"{tag}: shorter than {MIN_CLIP_S}s")
    total = plan.duration
    if total > max_duration + 0.01:
        errors.append(f"total length {total:.1f}s exceeds the {max_duration:.0f}s limit")
    available = sum(float(a["meta"]["duration"]) for a in assets.values() if a.get("role") == "clip")
    if not errors and target_duration:
        expected = min(target_duration, available)
        if abs(total - expected) > max(1.0, 0.15 * expected):
            warnings.append(f"edit length {total:.1f}s differs from the requested {target_duration:.0f}s")
    return errors, warnings
