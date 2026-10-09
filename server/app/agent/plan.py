"""The structured edit plan the agent produces before touching the timeline,
and the validator that checks it against real media metadata."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..engine.timeline import MIN_CLIP_S


class PlanSegment(BaseModel):
    asset_id: str
    src_in: float
    src_out: float
    reason: str = ""


class EditPlan(BaseModel):
    summary: str = ""                     # one short user-facing sentence
    segments: list[PlanSegment] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)  # parts of the prompt we can't do yet
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
    },
    "required": ["summary", "segments", "unsupported_requests"],
    "additionalProperties": False,
}


def validate_plan(plan: EditPlan, assets: dict[str, dict], target_duration: float,
                  max_duration: float) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors make the plan unusable."""
    errors: list[str] = []
    warnings: list[str] = []
    if not plan.segments:
        errors.append("the plan has no segments")
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
    if not errors and target_duration > 0:
        expected = min(target_duration, available)
        if abs(total - expected) > max(1.0, 0.15 * expected):
            warnings.append(f"edit length {total:.1f}s differs from the requested {target_duration:.0f}s")
    return errors, warnings
