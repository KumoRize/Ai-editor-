"""Agent orchestrator (component A): runs one edit job end to end.

inspect → plan → validate plan → execute (one transaction) → validate timeline
→ apply framing → render → verify → complete.

The orchestrator only talks to the rest of the system through the tool layer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from ..config import get_settings
from ..engine.timeline import TimelineEditor
from ..errors import EditorError
from ..presets import PRESETS, output_size
from ..tools.registry import ToolContext, call_tool
from .llm_planner import plan_with_llm
from .plan import EditPlan, validate_plan
from .providers import LLMError, make_provider
from .rule_planner import detect_unsupported, plan_rules

log = logging.getLogger(__name__)

STEPS = [
    ("analyze_media", "Analyzing media..."),
    ("analyze_reference", "Analyzing reference..."),
    ("plan", "Creating edit plan..."),
    ("build_timeline", "Building timeline..."),
    ("apply_effects", "Applying effects..."),
    ("render", "Rendering..."),
    ("verify", "Checking final video..."),
]
# Share of the overall progress bar each step represents.
WEIGHTS = {"analyze_media": 0.15, "analyze_reference": 0.0, "plan": 0.1, "build_timeline": 0.05,
           "apply_effects": 0.02, "render": 0.63, "verify": 0.05}


def initial_steps() -> list[dict]:
    return [{"key": k, "label": label, "status": "pending", "detail": ""} for k, label in STEPS]


class Reporter:
    """Collects step state and pushes it to the job store via a callback."""

    def __init__(self, steps: list[dict], push: Callable[[list[dict], float], None]):
        self.steps = steps
        self.push = push
        self._frac: dict[str, float] = {}

    def _progress(self) -> float:
        return round(min(0.99, sum(WEIGHTS[k] * f for k, f in self._frac.items())), 4)

    def set(self, key: str, status: str, detail: str = "", frac: float | None = None) -> None:
        for s in self.steps:
            if s["key"] == key:
                s["status"] = status
                if detail:
                    s["detail"] = detail
        if frac is not None:
            self._frac[key] = frac
        elif status in ("done", "skipped"):
            self._frac[key] = 1.0
        self.push(self.steps, self._progress())


def _must(res: dict) -> dict:
    """Unwrap a tool result or raise its error — never continue after a failure."""
    if not res.get("ok"):
        err = res.get("error", {})
        raise EditorError(err.get("code", "TOOL_FAILED"), err.get("message", "An editing step failed."),
                          retryable=err.get("retryable", False))
    return res["result"]


def run_edit_job(job_id: str, request: dict, assets: dict[str, dict], work_dir: Path,
                 reporter: Reporter, previous_plan: dict | None = None) -> dict:
    s = get_settings()
    preset = PRESETS[request["preset"]]
    aspect = request.get("aspect_ratio") or preset["aspect_ratio"]
    width, height = output_size(aspect, request["resolution"])
    target = float(request.get("duration") or preset["default_duration"])
    max_dur = float(preset["max_duration"])
    prompt = request.get("prompt", "")
    notes: list[str] = []

    editor = TimelineEditor(None, assets)
    ctx = ToolContext(editor=editor, assets=assets, work_dir=work_dir)

    # 1. Inspect media (cached by content hash).
    reporter.set("analyze_media", "running")
    clip_ids = [a for a, v in assets.items() if v["role"] == "clip"]
    for i, aid in enumerate(clip_ids):
        res = call_tool(ctx, "analyze_video", {"asset_id": aid})
        if not res["ok"]:
            raise EditorError(res["error"]["code"],
                              f"We couldn't analyze {assets[aid]['original_name']}: {res['error']['message']}",
                              retryable=res["error"].get("retryable", False))
        reporter.set("analyze_media", "running", frac=(i + 1) / len(clip_ids))
    total_scenes = sum(len(ctx.analyses[a]["scenes"]) for a in clip_ids)
    reporter.set("analyze_media", "done",
                 f"{len(clip_ids)} clip(s), {sum(assets[a]['meta']['duration'] for a in clip_ids):.1f}s, "
                 f"{total_scenes} scene cut(s)")

    # 2. Reference (Phase 3).
    reporter.set("analyze_reference", "skipped", "No reference video (reference matching arrives in Phase 3)")

    # 3. Plan.
    reporter.set("plan", "running")
    plan: EditPlan | None = None
    warnings: list[str] = []
    try:
        provider = make_provider(s)
    except LLMError as e:
        provider = None
        notes.append(f"AI planner unavailable ({e.message}); used the built-in planner.")
    if provider is not None:
        try:
            plan, warnings, calls = plan_with_llm(provider, prompt, assets, ctx.analyses, target, max_dur,
                                                  previous_plan, s.llm_max_calls_per_job)
            plan.planner = provider.name
            log.info("job %s: LLM plan in %d call(s)", job_id, calls)
        except LLMError as e:
            log.warning("job %s: LLM planner failed: %s %s", job_id, e.message, e.detail)
            notes.append(f"{e.message} The built-in planner was used instead.")
    if plan is None:
        plan = plan_rules(prompt, assets, ctx.analyses, target)
        plan.planner = "built-in rules"
        errors, warnings = validate_plan(plan, assets, target, max_dur)
        if errors:
            raise EditorError("PLAN_INVALID", "Could not create a valid edit plan: " + errors[0])
    # Make sure unsupported asks are reported even if the planner missed them.
    generic = {"video", "cuts", "changes", "matching", "a", "the", "background", "text"}
    for item in detect_unsupported(prompt):
        words = {w for w in item.split(" (")[0].replace("/", " ").split() if w not in generic}
        if not any(w in u.lower() for u in plan.unsupported_requests for w in words):
            plan.unsupported_requests.append(item)
    (work_dir / "plan.json").write_text(plan.model_dump_json(indent=2))
    reporter.set("plan", "done", f"{len(plan.segments)} segment(s), {plan.duration:.1f}s")

    # 4. Build timeline — one transaction for the whole plan.
    reporter.set("build_timeline", "running")
    _must(call_tool(ctx, "create_timeline", {"width": width, "height": height, "fps": 30,
                                             "fit": "cover", "max_duration": max_dur}))
    with editor.transaction():
        for seg in plan.segments:
            _must(call_tool(ctx, "select_clip", {"asset_id": seg.asset_id, "src_in": seg.src_in,
                                                 "src_out": seg.src_out, "note": seg.reason}))
    summary = _must(call_tool(ctx, "validate_timeline"))
    reporter.set("build_timeline", "done", f"{summary['clips']} clip(s) on the timeline, {summary['duration']:.1f}s")

    # 5. Effects (Phase 1: output framing only).
    reporter.set("apply_effects", "running")
    fit = request.get("fit", "cover")
    _must(call_tool(ctx, "resize_video", {"width": width, "height": height, "fit": fit}))
    _must(call_tool(ctx, "validate_timeline"))
    (work_dir / "timeline.json").write_text(editor.timeline.model_dump_json(indent=2))
    reporter.set("apply_effects", "done", f"{aspect} {width}x{height}, framing: {fit}")

    # 6. Render, 7. verify (export_video renders then independently verifies).
    reporter.set("render", "running", frac=0.0)
    ctx.on_progress = lambda f: reporter.set("render", "running", f"{int(f * 100)}%", frac=f)
    res = call_tool(ctx, "export_video")
    if not res["ok"]:
        code = res["error"]["code"]
        if code.startswith("OUTPUT_"):
            reporter.set("render", "done")
            reporter.set("verify", "failed", res["error"]["message"])
        raise EditorError(code, res["error"]["message"], retryable=res["error"].get("retryable", False))
    out = res["result"]
    reporter.set("render", "done")
    reporter.set("verify", "done", f"H.264/AAC {out['width']}x{out['height']}, {out['duration']:.1f}s, "
                                   f"{out['size_bytes'] / 1e6:.1f} MB")
    if not Path(out["path"]).is_file():  # belt and braces: never report success without a file
        raise EditorError("OUTPUT_MISSING", "Rendering produced no output file.", retryable=True)

    return {
        "output_file": Path(out["path"]).name,
        "duration": out["duration"], "width": out["width"], "height": out["height"],
        "size_bytes": out["size_bytes"],
        "summary": plan.summary, "planner": plan.planner,
        "segments": [sg.model_dump() for sg in plan.segments],
        "unsupported_requests": plan.unsupported_requests,
        "notes": notes + warnings,
    }


def load_previous_plan(prev_job_dir: Path) -> dict | None:
    try:
        return json.loads((prev_job_dir / "plan.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
