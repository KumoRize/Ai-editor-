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
audio loudness envelope, silent ranges, optional speech transcript), optional music (tempo and beat \
times), and an optional style profile measured from a reference video. You return an edit plan that \
the system executes exactly.

Available operations:
- segments: ordered source ranges played back to back (cuts/trims; removing silent parts = leaving \
them out). Per segment: speed (0.25-4), speed_ramp (stepped ramps that keep the segment's length), \
zoom (in, out, punch_in, pan_left, pan_right, ken_burns) with zoom_amount, and transition_in from \
the previous segment (fade, dissolve, fadeblack, fadewhite, wipes, slides, circleopen, zoomin, ...). \
A transition overlaps the two segments by transition_duration, which shortens the video.
- original_audio_volume (0 mutes the clips' own sound), color_preset, effects for the whole video.
- captions.mode "speech" burns in subtitles of what is said (only if speech_recognition is available).
- texts: title overlays at output-timeline seconds. sound_effects: built-in effects at output seconds; \
sfx_on_transitions adds a whoosh at every transition. intro/outro: fade in/out of the whole video.
- music (only if a music file was uploaded): volume, duck (lower it under speech), beat_sync (cuts are \
snapped to the beat afterwards), start_at (seconds into the song, -1 = first beat).
List every part of the request that none of these can do in unsupported_requests (a few words each). \
Never pretend those will be applied.

Rules:
- Use only the asset_id values provided. Times in segments are seconds of the source file.
- If the user gives exact times or ranges (e.g. "0:05-0:12", "first 10 seconds"), use exactly \
those, in the order given, clamped to the clip length; then the target duration does not apply.
- 0 <= src_in < src_out <= asset duration. Each segment at least 0.4 s of output.
- The output length is sum((src_out - src_in) / speed) minus transition durations; it should equal \
the target duration (within 5%) unless the footage is shorter.
- Prefer starting segments on scene cuts, avoid silent ranges when audio matters, prefer louder / \
more active moments, keep chronological order unless asked otherwise, never reuse a moment.
- Fast-paced requests: segments around 0.8-1.8 s. Calm requests: 3-6 s. Otherwise ~2-3 s. With \
beat_sync, make segment lengths whole numbers of beats.
- With a reference style profile, match its pacing (target_shot_length), transitions, zoom frequency, \
colour and caption placement — it is style inspiration only; its footage is never used.
- Use effects only when asked for or implied by the reference; keep it tasteful.
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
                  target_duration: float | None, previous_plan: dict | None, prompt_mode: bool = False,
                  extra: dict | None = None) -> str:
    extra = extra or {}
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
        tr = (extra.get("transcripts") or {}).get(aid)
        if tr:
            media[-1]["speech"] = [{"start": sg["start"], "end": sg["end"], "text": sg["text"][:200]}
                                   for sg in tr.get("segments", [])[:120]]
    ctx = {"user_request": prompt, "media": media}
    if prompt_mode:
        ctx["presets"] = {k: {"aspect_ratio": v["aspect_ratio"], "default_duration_s": v["default_duration"],
                              "max_duration_s": v["max_duration"]} for k, v in PRESETS.items()}
        ctx["framing_options"] = FITS
    else:
        ctx["target_duration_s"] = target_duration
    ctx["speech_recognition_available"] = bool(extra.get("speech_available"))
    mus = extra.get("music")
    if mus:
        an = mus["analysis"]
        ctx["music"] = {"duration": an.get("duration"), "tempo_bpm": an.get("tempo"),
                        "beat_times": an.get("beats", [])[:240], "first_beat": (an.get("beats") or [0])[0],
                        "energy_db_per_second": _downsample(an.get("energy_db", []), 240)}
    else:
        ctx["music"] = None
    if extra.get("reference"):
        ref = extra["reference"]
        ctx["reference_style"] = {k: ref.get(k) for k in ("aspect_ratio", "shots", "transitions", "zoom", "captions",
                                                           "color", "style")}
        ctx["reference_style"]["audio"] = {k: v for k, v in (ref.get("audio") or {}).items() if k != "beats"}
    if previous_plan:
        ctx["previous_edit"] = {"note": "The user is revising this earlier edit. Apply their new request to it.",
                                "segments": previous_plan.get("segments", [])[:120],
                                "summary": previous_plan.get("summary", "")}
    return json.dumps(ctx, separators=(",", ":"))


def plan_with_llm(provider: LLMProvider, prompt: str, assets: dict[str, dict], analyses: dict[str, dict],
                  target_duration: float | None, max_duration: float, previous_plan: dict | None,
                  max_calls: int, prompt_mode: bool = False,
                  extra: dict | None = None) -> tuple[EditPlan, list[str], int]:
    """Return (plan, warnings, calls_used). Raises LLMError if no valid plan is produced.

    In prompt mode the model also picks the output settings; they are resolved
    (defaults + preset limits) into plan.settings and validated like everything else.
    """
    user = build_context(prompt, assets, analyses, target_duration, previous_plan, prompt_mode, extra)
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
            if plan.music.start_at is not None and plan.music.start_at < 0:
                plan.music.start_at = None
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
