"""LLM-backed edit planner. Sends compact JSON summaries (never media) and asks
for a schema-constrained EditPlan. One repair round if validation fails."""
from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from .plan import PLAN_JSON_SCHEMA, EditPlan, validate_plan
from .providers.base import LLMError, LLMProvider

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the planning component of a video editing system. You receive a user's \
editing request and a JSON summary of their uploaded footage (durations, detected scene cuts, an \
audio loudness envelope, silent ranges). You return an edit plan: an ordered list of source \
segments that will be played back to back, plus a short summary for the user.

Operations available right now: choosing segments (cuts/trims), ordering them, and the output \
format/framing (already chosen by the user — you do not control it). Nothing else is available yet. \
List every part of the request that needs an unavailable operation (for example subtitles, \
music, text, zooms, speed changes, transitions, colour, sound effects, reference matching) in \
unsupported_requests, using a few words each. Never pretend those will be applied.

Rules:
- Use only the asset_id values provided. Times are in seconds of the source file.
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


def build_context(prompt: str, assets: dict[str, dict], analyses: dict[str, dict],
                  target_duration: float, previous_plan: dict | None) -> str:
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
    ctx = {"user_request": prompt, "target_duration_s": target_duration, "media": media}
    if previous_plan:
        ctx["previous_edit"] = {"note": "The user is revising this earlier edit. Apply their new request to it.",
                                "segments": previous_plan.get("segments", [])[:120],
                                "summary": previous_plan.get("summary", "")}
    return json.dumps(ctx, separators=(",", ":"))


def plan_with_llm(provider: LLMProvider, prompt: str, assets: dict[str, dict], analyses: dict[str, dict],
                  target_duration: float, max_duration: float, previous_plan: dict | None,
                  max_calls: int) -> tuple[EditPlan, list[str], int]:
    """Return (plan, warnings, calls_used). Raises LLMError if no valid plan is produced."""
    user = build_context(prompt, assets, analyses, target_duration, previous_plan)
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
            data = provider.generate_json(system=SYSTEM_PROMPT, user=msg, schema=PLAN_JSON_SCHEMA,
                                          schema_name="edit_plan")
        except LLMError as e:
            if e.retryable and not retried_transient and calls < max_calls:
                retried_transient = True
                log.warning("planner transient error, retrying once: %s", e.detail or e.message)
                continue
            raise
        try:
            plan = EditPlan.model_validate(data)
        except ValidationError as e:
            last_errors = [f"plan does not match the schema: {e.errors()[0].get('msg', '')}"]
            continue
        errors, warnings = validate_plan(plan, assets, target_duration, max_duration)
        if not errors:
            return plan, warnings, calls
        last_errors = errors
        log.info("plan invalid (%d errors), asking for a repair", len(errors))
    raise LLMError("The AI planner could not produce a valid edit plan.", retryable=True,
                   detail="; ".join(last_errors))
