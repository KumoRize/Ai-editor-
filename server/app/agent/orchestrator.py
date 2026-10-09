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
from .plan import EditPlan, ProjectSettings, resolve_settings, validate_plan
from .prompt_parser import parse_prompt
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
    prompt = request.get("prompt", "")
    prompt_mode = request.get("mode") == "prompt"
    notes: list[str] = []
    directives = parse_prompt(prompt)
    explicit_cuts = bool(directives.ranges or directives.first_s or directives.last_s or directives.remove_silence)
    if prompt_mode:  # "Full prompt": every setting comes from the user's text
        settings, snotes = resolve_settings(directives.preset, directives.aspect_ratio, directives.resolution,
                                            directives.duration, directives.fit, auto_length=explicit_cuts)
    else:
        settings, snotes = resolve_settings(request.get("preset"), request.get("aspect_ratio"),
                                            request.get("resolution"), request.get("duration"), request.get("fit"))
    notes += snotes

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
            plan, warnings, calls = plan_with_llm(
                provider, prompt, assets, ctx.analyses, settings.duration,
                float(PRESETS[settings.preset]["max_duration"]), previous_plan, s.llm_max_calls_per_job,
                prompt_mode=prompt_mode)
            plan.planner = provider.name
            if prompt_mode and plan.settings:
                settings = ProjectSettings(**plan.settings)
            log.info("job %s: LLM plan in %d call(s)", job_id, calls)
        except LLMError as e:
            log.warning("job %s: LLM planner failed: %s %s", job_id, e.message, e.detail)
            notes.append(f"{e.message} The built-in planner was used instead.")
    max_dur = float(PRESETS[settings.preset]["max_duration"])
    if plan is None:
        plan = plan_rules(prompt, assets, ctx.analyses, settings.duration, directives, max_dur)
        plan.planner = "built-in rules"
        errors, warnings = validate_plan(plan, assets, None if explicit_cuts else settings.duration, max_dur)
        if errors:
            raise EditorError("PLAN_INVALID", "Could not create a valid edit plan: " + errors[0])
        if prompt_mode:
            plan.settings = settings.model_dump()
    notes += plan.notes
    if prompt_mode:
        understood = [settings.describe()] + (directives.describe() if plan.planner == "built-in rules" else [])
        notes.insert(0, "Understood your prompt as: " + "; ".join(understood) + ".")
    width, height = output_size(settings.aspect_ratio, settings.resolution)
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
            clip = _must(call_tool(ctx, "select_clip", {"asset_id": seg.asset_id, "src_in": seg.src_in,
                                                        "src_out": seg.src_out, "note": seg.reason}))
            if plan.original_audio_volume != 1.0:
                _must(call_tool(ctx, "set_volume", {"clip_id": clip["clip_id"],
                                                    "volume": plan.original_audio_volume}))
    summary = _must(call_tool(ctx, "validate_timeline"))
    reporter.set("build_timeline", "done", f"{summary['clips']} clip(s) on the timeline, {summary['duration']:.1f}s")

    # 5. Effects (Phase 1: output framing only).
    reporter.set("apply_effects", "running")
    fit = settings.fit
    _must(call_tool(ctx, "resize_video", {"width": width, "height": height, "fit": fit}))
    _must(call_tool(ctx, "validate_timeline"))
    (work_dir / "timeline.json").write_text(editor.timeline.model_dump_json(indent=2))
    vol = plan.original_audio_volume
    sound = "" if vol == 1.0 else ", sound muted" if vol == 0 else f", sound {int(vol * 100)}%"
    reporter.set("apply_effects", "done", f"{settings.aspect_ratio} {width}x{height}, framing: {fit}{sound}")

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
        "mode": "prompt" if prompt_mode else "settings",
        "settings": settings.model_dump(),
    }


def load_previous_plan(prev_job_dir: Path) -> dict | None:
    try:
        return json.loads((prev_job_dir / "plan.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
