"""LLM-backed edit planner. Sends compact JSON summaries (never media) and asks
for a schema-constrained EditPlan. One repair round if validation fails."""
from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from ..presets import FITS, PRESETS
from .plan import PLAN_JSON_SCHEMA, PROMPT_MODE_SCHEMA, EditPlan, resolve_settings, validate_plan
from .providers.base import LLMError, LLMProvider

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the planning component of a video editing system. You receive a user's \
editing request and a JSON summary of their uploaded footage (durations, detected scene cuts, an \
audio loudness envelope, silent ranges). You return an edit plan: an ordered list of source \
segments that will be played back to back, plus a short summary for the user.

Operations available right now: choosing segments (cuts/trims), ordering them, removing silent \
parts (by leaving them out), and the volume of the clip's own sound (original_audio_volume: 0 mutes \
it, 1 keeps it, up to 2 is louder). Nothing else is available yet. List every part of the request \
that needs an unavailable operation (for example subtitles, music, text, zooms, speed changes, \
transitions, colour, sound effects, reference matching) in unsupported_requests, using a few words \
each. Never pretend those will be applied.

Rules:
- Use only the asset_id values provided. Times are in seconds of the source file.
- If the user gives exact times or ranges (e.g. "0:05-0:12", "first 10 seconds"), use exactly \
those, in the order given, clamped to the clip length; then the target duration does not apply.
- 0 <= src_in < src_out <= asset duration. Each segment at least 0.4 s long.
- The sum of (src_out - src_in) should equal the target duration (within 5%), unless the footage \
is shorter, in which case use the footage you have.
- Prefer starting segments on scene cuts, avoid silent ranges when audio matters, prefer louder / \
more active moments, and keep chronological order unless the user asks otherwise.
- Fast-paced requests: segments around 0.8–1.8 s. Calm requests: 3–6 s. Otherwise ~2–3 s.
- Avoid using the same source moment twice.
- The summary is one plain sentence for a non-technical user; no internal reasoning."""


def _downsample(values: list[float], max_len: int) -> list[float]:
    if len(values) <= max_len:
        return values
    step = len(values) / max_len
    return [round(max(values[int(i * step): int((i + 1) * step)] or [-90.0]), 1) for i in range(max_len)]


PROMPT_MODE_ADDENDUM = """

FULL PROMPT MODE: the user did not fill in any settings. Infer the output settings from the \
prompt and return them in `settings`:
- preset: the platform named or implied (instagram_reels, youtube_shorts, tiktok, \
youtube_landscape, square); default instagram_reels.
- aspect_ratio: as stated, else the preset's; resolution: 720p only if asked, else 1080p.
- duration_s: the length the user asked for; if they gave exact time ranges or asked to keep \
everything / only remove silence, use 0 (= whatever the segments add up to); otherwise the \
preset's default. Never exceed the preset's maximum.
- fit: "blur" if they mention a blurred background, "contain" if they don't want cropping / want \
black bars, otherwise "cover"."""


def build_context(prompt: str, assets: dict[str, dict], analyses: dict[str, dict],
                  target_duration: float | None, previous_plan: dict | None, prompt_mode: bool = False) -> str:
    media = []
    for aid, a in assets.items():
        if a.get("role") != "clip":
            continue
        m = a["meta"]
        an = analyses.get(aid, {})
        audio = an.get("audio", {})
        energy = audio.get("energy_db", [])
        win = audio.get("window_s", 0.5) or 0.5
        # ~1 value per second, max 240 values
        per_s = _downsample(energy, min(240, max(1, int(len(energy) * win))))
        media.append({
            "asset_id": aid,
            "duration": m["duration"],
            "display_size": f"{m['width']}x{m['height']}",
            "fps": m.get("fps"),
            "has_audio": m.get("has_audio", False),
            "scene_cuts": [c["t"] for c in an.get("scenes", [])][:150],
            "loudness_db_envelope": {"seconds_per_value": round(len(energy) * win / max(1, len(per_s)), 2),
                                     "values": per_s},
            "silent_ranges": audio.get("silent_ranges", [])[:50],
        })
    ctx = {"user_request": prompt, "media": media}
    if prompt_mode:
        ctx["presets"] = {k: {"aspect_ratio": v["aspect_ratio"], "default_duration_s": v["default_duration"],
                              "max_duration_s": v["max_duration"]} for k, v in PRESETS.items()}
        ctx["framing_options"] = FITS
    else:
        ctx["target_duration_s"] = target_duration
    if previous_plan:
        ctx["previous_edit"] = {"note": "The user is revising this earlier edit. Apply their new request to it.",
                                "segments": previous_plan.get("segments", [])[:120],
                                "summary": previous_plan.get("summary", "")}
    return json.dumps(ctx, separators=(",", ":"))


def plan_with_llm(provider: LLMProvider, prompt: str, assets: dict[str, dict], analyses: dict[str, dict],
                  target_duration: float | None, max_duration: float, previous_plan: dict | None,
                  max_calls: int, prompt_mode: bool = False) -> tuple[EditPlan, list[str], int]:
    """Return (plan, warnings, calls_used). Raises LLMError if no valid plan is produced.

    In prompt mode the model also picks the output settings; they are resolved
    (defaults + preset limits) into plan.settings and validated like everything else.
    """
    user = build_context(prompt, assets, analyses, target_duration, previous_plan, prompt_mode)
    system = SYSTEM_PROMPT + (PROMPT_MODE_ADDENDUM if prompt_mode else "")
    schema = PROMPT_MODE_SCHEMA if prompt_mode else PLAN_JSON_SCHEMA
    calls = 0
    last_errors: list[str] = []
    retried_transient = False
    while calls < max_calls:
        msg = user
        if last_errors:
            msg = user + "\n\nYour previous plan was invalid:\n- " + "\n- ".join(last_errors[:20]) + \
                  "\nReturn a corrected plan."
        calls += 1
        try:
            data = provider.generate_json(system=system, user=msg, schema=schema, schema_name="edit_plan")
        except LLMError as e:
            if e.retryable and not retried_transient and calls < max_calls:
                retried_transient = True
                log.warning("planner transient error, retrying once: %s", e.detail or e.message)
                continue
            raise
        try:
            plan = EditPlan.model_validate({k: v for k, v in data.items() if k != "settings"})
            target, max_d = target_duration, max_duration
            if prompt_mode:
                raw = data.get("settings") or {}
                settings, notes = resolve_settings(raw.get("preset"), raw.get("aspect_ratio"), raw.get("resolution"),
                                                   raw.get("duration_s"), raw.get("fit"), auto_length=True)
                plan.settings, plan.notes = settings.model_dump(), notes
                target, max_d = settings.duration, float(PRESETS[settings.preset]["max_duration"])
        except (ValidationError, AttributeError, TypeError) as e:
            last_errors = [f"plan does not match the schema: {e}"[:300]]
            continue
        errors, warnings = validate_plan(plan, assets, target, max_d)
        if not errors:
            return plan, warnings, calls
        last_errors = errors
        log.info("plan invalid (%d errors), asking for a repair", len(errors))
    raise LLMError("The AI planner could not produce a valid edit plan.", retryable=True,
                   detail="; ".join(last_errors))
