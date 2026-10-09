"""Agent orchestrator (component A): runs one edit job end to end.

inspect media / music / reference → plan → validate plan → build the timeline
(one transaction) → apply effects, captions, music, sound effects → validate →
render → verify the file → automatic quality check (+ one repair round) → done.

The orchestrator only talks to the rest of the system through the tool layer.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from ..analysis import transcribe as speech
from ..config import get_settings
from ..engine.timeline import TRANSITION_LABELS, Timeline, TimelineEditor
from ..errors import EditorError
from ..presets import PRESETS, output_size
from ..tools.registry import ToolContext, call_tool
from .llm_planner import plan_with_llm
from .plan import EditPlan, ProjectSettings, resolve_settings, validate_plan
from .prompt_parser import parse_prompt
from .providers import LLMError, make_provider
from .rule_planner import plan_rules

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
WEIGHTS = {"analyze_media": 0.12, "analyze_reference": 0.04, "plan": 0.08, "build_timeline": 0.03,
           "apply_effects": 0.05, "render": 0.6, "verify": 0.07}
SPEECH_WORDS = r"\b(say|says|said|talk|talks|talking|speech|speaking|quote|words|interview|podcast|explain)"


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


def _try(ctx: ToolContext, notes: list[str], what: str, name: str, args: dict) -> dict | None:
    """Optional creative step: on failure keep going, but tell the user it was not applied."""
    res = call_tool(ctx, name, args)
    if not res["ok"]:
        notes.append(f"Not applied — {what}: {res['error']['message']}")
        return None
    return res["result"]


def _merge_unsupported(*lists: list[str]) -> list[str]:
    out: list[str] = []
    for lst in lists:
        for item in lst:
            key = item.lower().split(" (")[0]
            if not any(key in o.lower() or o.lower().split(" (")[0] in key for o in out):
                out.append(item)
    return out


def run_edit_job(job_id: str, request: dict, assets: dict[str, dict], work_dir: Path,
                 reporter: Reporter, previous_plan: dict | None = None,
                 previous_timeline: dict | None = None) -> dict:
    s = get_settings()
    prompt = request.get("prompt", "")
    prompt_mode = request.get("mode") == "prompt"
    notes: list[str] = []
    directives = parse_prompt(prompt)
    explicit_cuts = bool(directives.ranges or directives.first_s or directives.last_s or directives.remove_silence)
    clip_ids = [a for a, v in assets.items() if v["role"] == "clip"]
    music_id = request.get("music_asset_id")
    ref_id = request.get("reference_asset_id")

    editor = TimelineEditor(None, assets)
    ctx = ToolContext(editor=editor, assets=assets, work_dir=work_dir)

    # 1. Inspect media (cached by content hash).
    reporter.set("analyze_media", "running")
    for i, aid in enumerate(clip_ids):
        res = call_tool(ctx, "analyze_video", {"asset_id": aid})
        if not res["ok"]:
            raise EditorError(res["error"]["code"],
                              f"We couldn't analyze {assets[aid]['original_name']}: {res['error']['message']}",
                              retryable=res["error"].get("retryable", False))
        reporter.set("analyze_media", "running", frac=(i + 1) / (len(clip_ids) + (1 if music_id else 0)))
    music = None
    if music_id:
        res = call_tool(ctx, "detect_beats", {"asset_id": music_id})
        if res["ok"]:
            music = {"asset_id": music_id, "analysis": res["result"]}
        else:
            notes.append(f"The music could not be analysed ({res['error']['message']}); it was not used.")
    total_scenes = sum(len(ctx.analyses[a]["scenes"]) for a in clip_ids)
    detail = (f"{len(clip_ids)} clip(s), {sum(assets[a]['meta']['duration'] for a in clip_ids):.1f}s, "
              f"{total_scenes} scene cut(s)")
    if music:
        detail += f"; music {music['analysis']['tempo']:.0f} BPM"
    reporter.set("analyze_media", "done", detail)

    # 2. Reference: style/pacing profile only.
    reference = None
    if ref_id:
        reporter.set("analyze_reference", "running")
        res = call_tool(ctx, "analyze_reference", {"asset_id": ref_id})
        if res["ok"]:
            reference = res["result"]
            st = reference["style"]
            reporter.set("analyze_reference", "done",
                         f"{reference['shots']['count']} shots, ~{st['target_shot_length']:.1f}s each, "
                         f"{reference['aspect_ratio']}, {st['energy']} energy")
        else:
            reporter.set("analyze_reference", "failed", "The reference video could not be analyzed.")
            notes.append("The reference video could not be analyzed, so the edit was made without it.")
    else:
        reporter.set("analyze_reference", "skipped", "No reference video")

    if prompt_mode:  # "Full prompt": every setting comes from the user's text (or the reference)
        aspect = directives.aspect_ratio or (reference["aspect_ratio"] if reference and
                                             reference["aspect_ratio"] in ("9:16", "16:9", "1:1") else None)
        dur = directives.duration
        if dur is None and reference and not explicit_cuts and not directives.preset:
            dur = reference["duration"]
        settings, snotes = resolve_settings(directives.preset, aspect, directives.resolution, dur,
                                            directives.fit, auto_length=explicit_cuts)
    else:
        settings, snotes = resolve_settings(request.get("preset"), request.get("aspect_ratio"),
                                            request.get("resolution"), request.get("duration"), request.get("fit"))
    notes += snotes
    speech_ok, speech_reason = speech.availability()

    # 3. Plan.
    reporter.set("plan", "running")
    plan: EditPlan | None = None
    warnings: list[str] = []
    agent_timeline: Timeline | None = None
    try:
        provider = make_provider(s)
    except LLMError as e:
        provider = None
        notes.append(f"AI planner unavailable ({e.message}); used the built-in planner.")
    if provider is not None and previous_timeline and hasattr(provider, "chat"):
        # Edit again with an AI model: let the agent edit the previous timeline with tools.
        from .agent_loop import revise_timeline
        try:
            agent_timeline, agent_summary = revise_timeline(provider, prompt, previous_timeline, ctx,
                                                            s.llm_max_calls_per_job * 4)
            plan = EditPlan.model_validate(previous_plan or {})
            plan.summary, plan.planner = agent_summary, f"{provider.name} (agent)"
        except LLMError as e:
            notes.append(f"{e.message} The edit was re-planned instead.")
            agent_timeline = None
    transcripts: dict[str, dict] = {}
    if provider is not None and plan is None and speech_ok and (
            directives.captions or re.search(SPEECH_WORDS, prompt.lower())):
        for aid in clip_ids:
            res = call_tool(ctx, "transcribe", {"asset_id": aid})
            if res["ok"]:
                transcripts[aid] = res["result"]
    if provider is not None and plan is None:
        try:
            plan, warnings, calls = plan_with_llm(
                provider, prompt, assets, ctx.analyses, settings.duration,
                float(PRESETS[settings.preset]["max_duration"]), previous_plan, s.llm_max_calls_per_job,
                prompt_mode=prompt_mode,
                extra={"music": music, "reference": reference, "speech_available": speech_ok,
                       "transcripts": transcripts})
            plan.planner = provider.name
            if prompt_mode and plan.settings:
                settings = ProjectSettings(**plan.settings)
            log.info("job %s: LLM plan in %d call(s)", job_id, calls)
        except LLMError as e:
            log.warning("job %s: LLM planner failed: %s %s", job_id, e.message, e.detail)
            notes.append(f"{e.message} The built-in planner was used instead.")
    max_dur = float(PRESETS[settings.preset]["max_duration"])
    if plan is None:
        plan = plan_rules(prompt, assets, ctx.analyses, settings.duration, directives, max_dur,
                          reference=reference, music=music)
        plan.planner = "built-in rules"
        errors, warnings = validate_plan(plan, assets, None if explicit_cuts else settings.duration, max_dur)
        if errors:
            raise EditorError("PLAN_INVALID", "Could not create a valid edit plan: " + errors[0])
        if prompt_mode:
            plan.settings = settings.model_dump()
    notes += plan.notes
    if plan.captions.mode == "speech" and not speech_ok:
        plan.captions.mode = "none"
        plan.unsupported_requests.append(f"subtitles from speech ({speech_reason} on this server)")
    if plan.music.use and not music:
        plan.music.use = False
    plan.unsupported_requests = _merge_unsupported(plan.unsupported_requests, directives.unsupported)
    if prompt_mode:
        understood = [settings.describe()] + (directives.describe() if plan.planner == "built-in rules" else [])
        notes.insert(0, "Understood your prompt as: " + "; ".join(understood) + ".")
    width, height = output_size(settings.aspect_ratio, settings.resolution)
    (work_dir / "plan.json").write_text(plan.model_dump_json(indent=2))
    reporter.set("plan", "done", plan.summary[:160] if agent_timeline is not None else
                 f"{len(plan.segments)} segment(s), {plan.duration:.1f}s")

    # 4. Build the timeline — core cuts in one transaction (all or nothing).
    reporter.set("build_timeline", "running")
    if agent_timeline is not None:
        editor.timeline = agent_timeline
        _must(call_tool(ctx, "resize_video", {"width": width, "height": height, "fit": settings.fit}))
    else:
        _must(call_tool(ctx, "create_timeline", {"width": width, "height": height, "fps": 30,
                                                 "fit": settings.fit, "max_duration": max_dur}))
        clip_of: list[str] = []
        with editor.transaction():
            for seg in plan.segments:
                clip = _must(call_tool(ctx, "select_clip", {"asset_id": seg.asset_id, "src_in": seg.src_in,
                                                            "src_out": seg.src_out, "note": seg.reason}))
                clip_of.append(clip["clip_id"])
                if seg.speed != 1.0:
                    _must(call_tool(ctx, "change_speed", {"clip_id": clip["clip_id"], "speed": seg.speed}))
        # Creative per-segment edits: applied if valid, reported if not.
        for seg, cid in zip(plan.segments, clip_of):
            if seg.zoom != "none":
                _try(ctx, notes, f"a {seg.zoom} zoom", "add_zoom", {"clip_id": cid, "kind": seg.zoom,
                                                                      "amount": seg.zoom_amount})
        for seg, cid in zip(plan.segments, clip_of):
            if seg.speed_ramp != "none":
                _try(ctx, notes, "a speed ramp", "speed_ramp", {"clip_id": cid, "pattern": seg.speed_ramp})
        for i, (seg, cid) in enumerate(zip(plan.segments, clip_of)):
            if i > 0 and seg.transition_in != "none":
                _try(ctx, notes, f"a {seg.transition_in} transition", "add_transition",
                     {"clip_id": cid, "type": seg.transition_in, "duration": seg.transition_duration})
    summary = _must(call_tool(ctx, "validate_timeline"))
    reporter.set("build_timeline", "done", f"{summary['clips']} clip(s), {summary['transitions']} transition(s), "
                                           f"{summary['duration']:.1f}s")

    # 5. Effects, sound and text.
    reporter.set("apply_effects", "running")
    applied: list[str] = []
    if agent_timeline is None:
        if plan.color_preset != "none" and _try(ctx, notes, "the colour look", "apply_color_adjustment",
                                                {"clip_id": "all", "preset": plan.color_preset}):
            applied.append(f"{plan.color_preset} colour")
        for eff in plan.effects:
            if _try(ctx, notes, f"the {eff} effect", "apply_effect", {"clip_id": "all", "effect": eff}):
                applied.append(eff)
        if plan.original_audio_volume != 1.0:
            _must(call_tool(ctx, "set_volume", {"clip_id": "all", "volume": plan.original_audio_volume}))
            applied.append("sound muted" if plan.original_audio_volume == 0 else
                           f"sound {int(plan.original_audio_volume * 100)}%")
        if plan.music.use and music:
            beats = music["analysis"].get("beats") or [0.0]
            offset = plan.music.start_at if plan.music.start_at is not None else max(0.0, beats[0] - 0.02)
            offset = min(offset, max(0.0, music["analysis"]["duration"] - 1.0))
            if _try(ctx, notes, "the music", "add_music",
                    {"asset_id": music_id, "src_in": round(offset, 3), "volume": plan.music.volume,
                     "duck": plan.music.duck, "loop": True}):
                applied.append("music")
                if plan.music.beat_sync:
                    r = _try(ctx, notes, "beat sync", "snap_to_beats", {"asset_id": music_id, "offset": offset})
                    if r:
                        applied.append(f"{r['snapped']} cut(s) on the beat")
        reporter.set("apply_effects", "running", frac=0.4)
        if plan.captions.mode == "speech":
            r = _try(ctx, notes, "subtitles", "add_subtitles", {"style": plan.captions.style,
                                                                "position": plan.captions.position})
            if r is not None:
                applied.append(f"{r['captions']} subtitle line(s)" if r["captions"] else "no speech found for subtitles")
        dur = editor.timeline.duration
        for t in plan.texts:
            if t.start < dur - 0.1:
                if _try(ctx, notes, f"the text “{t.text[:30]}”", "add_text",
                        {"text": t.text, "start": t.start, "end": min(t.end, dur), "style": t.style,
                         "position": t.position}):
                    applied.append("text")
            else:
                notes.append(f"The text “{t.text[:30]}” was skipped: it starts after the end of the video.")
        sfx_times: list[tuple[str, float]] = [(x.name, x.at) for x in plan.sound_effects]
        if plan.sfx_on_transitions:
            for c in editor.timeline.video_track().clips:
                if c.transition_in:
                    sfx_times.append(("whoosh", max(0.0, c.timeline_start + c.transition_in.duration / 2 - 0.4)))
        n_sfx = 0
        for name, at in sorted(sfx_times, key=lambda x: x[1])[:60]:
            if at < dur - 0.1 and _try(ctx, notes, f"the {name} sound", "add_sound_effect",
                                       {"at": round(at, 3), "sfx": name, "volume": 0.7}):
                n_sfx += 1
        if n_sfx:
            applied.append(f"{n_sfx} sound effect(s)")
        for pos, kind in (("start", plan.intro), ("end", plan.outro)):
            if kind != "none":
                d = min(0.8, editor.timeline.duration / 4)
                if _try(ctx, notes, f"the fade at the {pos}", "add_transition",
                        {"type": kind, "duration": round(d, 2), "position": pos}):
                    applied.append(f"fade {'in' if pos == 'start' else 'out'}")
    if any(c.keyframes for c in editor.timeline.video_track().clips):
        applied.insert(0, "zooms")
    _must(call_tool(ctx, "validate_timeline"))
    (work_dir / "timeline.json").write_text(editor.timeline.model_dump_json(indent=2))
    reporter.set("apply_effects", "done", f"{settings.aspect_ratio} {width}x{height}, {settings.fit}" +
                 (f"; {', '.join(dict.fromkeys(applied))}" if applied else ""))

    # 6. Render, 7. verify the file, then check its quality (one repair round).
    reporter.set("render", "running", frac=0.0)
    ctx.on_progress = lambda f: reporter.set("render", "running", f"{int(f * 100)}%", frac=f)
    out = _export(ctx, reporter)
    reporter.set("render", "done")
    reporter.set("verify", "running", "Checking the file and the picture/sound quality…")
    quality = _must(call_tool(ctx, "quality_check"))
    fixes = []
    for issue in quality["issues"]:
        if issue["code"] == "CLIPPING" and issue["fixable"]:
            for it in editor.timeline.audio:
                it.volume = round(it.volume * 0.7, 3)
            editor.timeline.settings.normalize_audio = True
            fixes.append("lowered the music/effects to stop distortion")
        elif issue["code"] == "TOO_QUIET" and issue["fixable"]:
            editor.timeline.settings.normalize_audio = True
            fixes.append("raised the overall loudness")
    if fixes:
        reporter.set("verify", "running", "Fixing: " + "; ".join(fixes))
        out = _export(ctx, reporter)  # pieces are cached: only the final mix is redone
        quality = _must(call_tool(ctx, "quality_check"))
        quality["repaired"] = fixes
        (work_dir / "timeline.json").write_text(editor.timeline.model_dump_json(indent=2))
    reporter.set("verify", "done", f"H.264/AAC {out['width']}x{out['height']}, {out['duration']:.1f}s, "
                                   f"{out['size_bytes'] / 1e6:.1f} MB; quality "
                                   f"{'OK' if quality['passed'] else 'has warnings'}")
    if not Path(out["path"]).is_file():  # belt and braces: never report success without a file
        raise EditorError("OUTPUT_MISSING", "Rendering produced no output file.", retryable=True)
    for issue in quality["issues"]:
        notes.append(issue["message"])

    tl = editor.timeline
    summary_text = _describe_result(tl, applied, reference is not None) if plan.planner == "built-in rules" \
        else plan.summary + (" Some requested parts could not be applied — see the notes."
                             if any(n.startswith("Not applied") for n in notes) else "")
    return {
        "output_file": Path(out["path"]).name,
        "duration": out["duration"], "width": out["width"], "height": out["height"],
        "size_bytes": out["size_bytes"],
        "summary": summary_text, "planner": plan.planner,
        "segments": [sg.model_dump() for sg in plan.segments],
        "unsupported_requests": plan.unsupported_requests,
        "notes": list(dict.fromkeys(notes + warnings)),
        "mode": "prompt" if prompt_mode else "settings",
        "settings": settings.model_dump(),
        "applied": list(dict.fromkeys(applied)),
        "quality": quality,
        "reference_used": reference is not None,
        "music_used": any(a.kind == "music" for a in tl.audio),
        "timeline": {"clips": [{"start": round(c.timeline_start, 2), "duration": round(c.duration, 2),
                                "asset_id": c.asset_id, "src_in": c.src_in, "speed": c.speed,
                                "zoom": bool(c.keyframes),
                                "transition": c.transition_in.type if c.transition_in else None}
                               for c in tl.video_track().clips],
                     "texts": len(tl.texts), "audio_items": len(tl.audio)},
    }


def _describe_result(tl: Timeline, applied: list[str], used_reference: bool) -> str:
    """One sentence built only from what the final timeline really contains."""
    clips = tl.video_track().clips
    parts = [f"{len(clips)} cut{'s' if len(clips) != 1 else ''}"]
    trans = [TRANSITION_LABELS.get(c.transition_in.type, c.transition_in.type) for c in clips if c.transition_in]
    if trans:
        parts.append(f"{len(trans)} {max(set(trans), key=trans.count)} transition{'s' if len(trans) != 1 else ''}")
    if any(c.keyframes for c in clips):
        parts.append("zooms")
    if any(c.speed != 1.0 for c in clips):
        parts.append("speed changes")
    colors = {c.color.preset for c in clips if c.color and c.color.preset != "none"}
    if colors:
        parts.append(f"{'/'.join(sorted(colors))} colour")
    effects = sorted({e for c in clips for e in c.effects})
    if effects:
        parts.append(", ".join(effects))
    n_sub = sum(1 for t in tl.texts if t.kind == "caption")
    if n_sub:
        parts.append(f"{n_sub} subtitle line{'s' if n_sub != 1 else ''}")
    if any(t.kind == "text" for t in tl.texts):
        parts.append("titles")
    if any(a.kind == "music" for a in tl.audio):
        parts.append("music" + (" with cuts on the beat" if any("on the beat" in x for x in applied) else ""))
    n_sfx = sum(1 for a in tl.audio if a.kind == "sfx")
    if n_sfx:
        parts.append(f"{n_sfx} sound effect{'s' if n_sfx != 1 else ''}")
    if tl.settings.intro or tl.settings.outro:
        parts.append("fades")
    text = ", ".join(parts) + f" — {tl.duration:.0f}s total"
    if used_reference:
        text += ", paced like your reference"
    return text[0].upper() + text[1:] + "."


def _export(ctx: ToolContext, reporter: Reporter) -> dict:
    res = call_tool(ctx, "export_video")
    if not res["ok"]:
        code = res["error"]["code"]
        if code.startswith("OUTPUT_"):
            reporter.set("render", "done")
            reporter.set("verify", "failed", res["error"]["message"])
        raise EditorError(code, res["error"]["message"], retryable=res["error"].get("retryable", False))
    return res["result"]


def load_previous(prev_job_dir: Path) -> tuple[dict | None, dict | None]:
    """(plan, timeline) of an earlier job, if still on disk."""
    out = []
    for name in ("plan.json", "timeline.json"):
        try:
            out.append(json.loads((prev_job_dir / name).read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            out.append(None)
    return out[0], out[1]


def load_previous_plan(prev_job_dir: Path) -> dict | None:
    return load_previous(prev_job_dir)[0]
