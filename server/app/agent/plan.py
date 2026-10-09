"""The structured edit plan the agent produces before touching the timeline,
and the validator that checks it against real media metadata."""
from __future__ import annotations

import copy

from pydantic import BaseModel, Field

from ..engine.timeline import (COLOR_PRESETS, EFFECTS, MIN_CLIP_S, SFX_NAMES, SPEED_RAMPS, TEXT_POSITIONS,
                               TEXT_STYLES, TRANSITIONS, ZOOM_KINDS)
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
    speed: float = 1.0
    speed_ramp: str = "none"          # a SPEED_RAMPS pattern; keeps the segment's length
    zoom: str = "none"                # a ZOOM_KINDS value
    zoom_amount: float = 0.15
    transition_in: str = "none"       # transition from the previous segment
    transition_duration: float = 0.5

    @property
    def out_len(self) -> float:
        return max(0.0, self.src_out - self.src_in) / (self.speed or 1.0)


class PlanText(BaseModel):
    text: str
    start: float
    end: float
    style: str = "title"
    position: str = "center"


class PlanCaptions(BaseModel):
    mode: str = "none"                # "none" | "speech" (subtitles from what is said)
    style: str = "subtitle"
    position: str = "bottom"


class PlanMusic(BaseModel):
    use: bool = False
    volume: float = 0.8
    duck: bool = True                 # lower the music under speech
    beat_sync: bool = False           # put cuts on the beat
    start_at: float | None = None     # offset into the song (None = first beat)


class PlanSfx(BaseModel):
    name: str
    at: float


class EditPlan(BaseModel):
    summary: str = ""                     # one short user-facing sentence
    segments: list[PlanSegment] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)  # parts of the prompt we can't do
    original_audio_volume: float = 1.0    # 0 = mute the clip's own sound
    color_preset: str = "none"
    effects: list[str] = Field(default_factory=list)
    captions: PlanCaptions = Field(default_factory=PlanCaptions)
    texts: list[PlanText] = Field(default_factory=list)
    music: PlanMusic = Field(default_factory=PlanMusic)
    sound_effects: list[PlanSfx] = Field(default_factory=list)
    sfx_on_transitions: bool = False
    intro: str = "none"                   # "none" | fade | fadeblack | fadewhite
    outro: str = "none"
    settings: dict | None = None          # output settings chosen from the prompt ("Full prompt" mode)
    notes: list[str] = Field(default_factory=list)  # user-facing notes about how the prompt was followed
    planner: str = ""                     # filled in by the orchestrator

    @property
    def duration(self) -> float:
        total = sum(s.out_len for s in self.segments)
        total -= sum(s.transition_duration for s in self.segments[1:] if s.transition_in != "none")
        return max(0.0, total)


def _enum(values, desc: str = "") -> dict:
    d = {"type": "string", "enum": list(values)}
    if desc:
        d["description"] = desc
    return d


# JSON Schema used for structured output from any LLM provider. Kept to the subset every provider
# supports (enums, no numeric bounds) — bounds are enforced by validate_plan() and again by the engine.
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
                    "speed": {"type": "number", "description": "Playback speed 0.25-4; 1 = normal."},
                    "speed_ramp": _enum(["none", *SPEED_RAMPS], "Stepped speed ramp; keeps the segment length."),
                    "zoom": _enum(["none", *ZOOM_KINDS]),
                    "zoom_amount": {"type": "number", "description": "0.05-0.5, e.g. 0.15 = 15% closer."},
                    "transition_in": _enum(["none", *TRANSITIONS], "Transition from the previous segment."),
                    "transition_duration": {"type": "number", "description": "Seconds, 0.2-1.0 typical."},
                },
                "required": ["asset_id", "src_in", "src_out", "reason", "speed", "speed_ramp", "zoom",
                             "zoom_amount", "transition_in", "transition_duration"],
                "additionalProperties": False,
            },
        },
        "unsupported_requests": {"type": "array", "items": {"type": "string"},
                                 "description": "Parts of the request that cannot be done with these operations."},
        "original_audio_volume": {"type": "number", "description": "Clip sound: 1 unchanged, 0 muted, up to 2."},
        "color_preset": _enum(COLOR_PRESETS),
        "effects": {"type": "array", "items": _enum(EFFECTS), "description": "Applied to the whole video."},
        "captions": {
            "type": "object",
            "properties": {"mode": _enum(["none", "speech"], "speech = subtitles of what is said."),
                           "style": _enum(TEXT_STYLES), "position": _enum(TEXT_POSITIONS)},
            "required": ["mode", "style", "position"], "additionalProperties": False,
        },
        "texts": {
            "type": "array", "description": "Text overlays (titles) in output-timeline seconds.",
            "items": {"type": "object",
                      "properties": {"text": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"},
                                     "style": _enum(TEXT_STYLES), "position": _enum(TEXT_POSITIONS)},
                      "required": ["text", "start", "end", "style", "position"], "additionalProperties": False},
        },
        "music": {
            "type": "object",
            "properties": {"use": {"type": "boolean"}, "volume": {"type": "number"}, "duck": {"type": "boolean"},
                           "beat_sync": {"type": "boolean"},
                           "start_at": {"type": "number", "description": "Offset into the song (s); -1 = first beat."}},
            "required": ["use", "volume", "duck", "beat_sync", "start_at"], "additionalProperties": False,
        },
        "sound_effects": {
            "type": "array", "description": "Sound effects at output-timeline seconds.",
            "items": {"type": "object", "properties": {"name": _enum(SFX_NAMES), "at": {"type": "number"}},
                      "required": ["name", "at"], "additionalProperties": False},
        },
        "sfx_on_transitions": {"type": "boolean", "description": "Add a whoosh at every transition."},
        "intro": _enum(["none", "fade", "fadeblack", "fadewhite"], "Fade in at the start."),
        "outro": _enum(["none", "fade", "fadeblack", "fadewhite"], "Fade out at the end."),
    },
    "required": ["summary", "segments", "unsupported_requests", "original_audio_volume", "color_preset", "effects",
                 "captions", "texts", "music", "sound_effects", "sfx_on_transitions", "intro", "outro"],
    "additionalProperties": False,
}

# "Full prompt" mode: the model also chooses the output settings from the prompt.
PROMPT_MODE_SCHEMA: dict = copy.deepcopy(PLAN_JSON_SCHEMA)
PROMPT_MODE_SCHEMA["properties"]["settings"] = {
    "type": "object",
    "description": "Output settings inferred from the user's prompt.",
    "properties": {
        "preset": _enum(PRESETS),
        "aspect_ratio": _enum(["9:16", "16:9", "1:1"]),
        "resolution": _enum(["720p", "1080p"]),
        "duration_s": {"type": "number",
                       "description": "Target output length in seconds; 0 = whatever the segments add up to."},
        "fit": _enum(FITS),
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
    if len(plan.segments) > 200:
        errors.append(f"too many segments ({len(plan.segments)}, max 200)")
    if not (0.0 <= plan.original_audio_volume <= 2.0):
        errors.append(f"original_audio_volume {plan.original_audio_volume} must be between 0 and 2")
    if plan.color_preset not in COLOR_PRESETS:
        errors.append(f"unknown colour preset {plan.color_preset}")
    for e in plan.effects:
        if e not in EFFECTS:
            errors.append(f"unknown effect {e}")
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
        if not (0.25 <= s.speed <= 4.0):
            errors.append(f"{tag}: speed {s.speed} must be between 0.25 and 4")
        elif 0 < s.out_len < MIN_CLIP_S:
            errors.append(f"{tag}: shorter than {MIN_CLIP_S}s")
        if s.speed_ramp not in ("none", *SPEED_RAMPS):
            errors.append(f"{tag}: unknown speed ramp {s.speed_ramp}")
        if s.speed_ramp != "none" and s.speed != 1.0:
            errors.append(f"{tag}: a speed ramp needs speed 1")
        if s.zoom not in ("none", *ZOOM_KINDS):
            errors.append(f"{tag}: unknown zoom {s.zoom}")
        if not (0.02 <= s.zoom_amount <= 1.0):
            errors.append(f"{tag}: zoom_amount {s.zoom_amount} must be between 0.02 and 1")
        if s.transition_in not in ("none", *TRANSITIONS):
            errors.append(f"{tag}: unknown transition {s.transition_in}")
        if s.transition_in != "none":
            if i == 0:
                errors.append(f"{tag}: the first segment cannot have a transition (use intro)")
            elif not (0.1 <= s.transition_duration <= 3.0):
                errors.append(f"{tag}: transition_duration must be between 0.1 and 3")
            else:
                prev = plan.segments[i - 1]
                if s.transition_duration > min(prev.out_len, s.out_len) / 2:
                    errors.append(f"{tag}: transition_duration {s.transition_duration} is too long for its segments")
    total = plan.duration
    if total > max_duration + 0.01:
        errors.append(f"total length {total:.1f}s exceeds the {max_duration:.0f}s limit")
    for t in plan.texts:
        if not t.text.strip() or t.start < 0 or t.end <= t.start or t.start >= max(total, 0.1):
            errors.append(f"text '{t.text[:20]}' has invalid times ({t.start}-{t.end}s)")
        if t.style not in TEXT_STYLES or t.position not in TEXT_POSITIONS:
            errors.append(f"text '{t.text[:20]}' has an unknown style/position")
    if plan.captions.mode not in ("none", "speech") or plan.captions.style not in TEXT_STYLES \
            or plan.captions.position not in TEXT_POSITIONS:
        errors.append("captions settings are invalid")
    if len(plan.sound_effects) > 60:
        errors.append("too many sound effects (max 60)")
    for s in plan.sound_effects:
        if s.name not in SFX_NAMES or s.at < 0 or s.at >= max(total, 0.1):
            errors.append(f"sound effect {s.name} at {s.at}s is invalid")
    if not (0 <= plan.music.volume <= 2):
        errors.append("music volume must be between 0 and 2")
    for f in (plan.intro, plan.outro):
        if f not in ("none", "fade", "fadeblack", "fadewhite"):
            errors.append(f"unknown fade {f}")
    available = sum(float(a["meta"]["duration"]) for a in assets.values() if a.get("role") == "clip")
    if not errors and target_duration:
        expected = min(target_duration, available)
        if abs(total - expected) > max(1.0, 0.15 * expected):
            warnings.append(f"edit length {total:.1f}s differs from the requested {target_duration:.0f}s")
    return errors, warnings
