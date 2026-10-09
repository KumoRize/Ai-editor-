"""Deterministic planner: works with no API key, used for tests, offline mode,
and as an honest fallback when the LLM planner is unavailable.

Strategy: decide a shot length from the prompt, the reference video's pacing and
the music's beat grid; split the source into equal zones (keeps the story
chronological and covers the whole video) and in each zone pick the window with
the most audio energy, preferring scene starts and avoiding silence. Then add the
creative layer the prompt or reference asks for: transitions, zooms, speed
ramps, colour, effects, subtitles, titles, music and sound effects.
"""
from __future__ import annotations

import math
import re

from ..errors import EditorError
from .plan import EditPlan, PlanMusic, PlanSegment, PlanSfx, PlanText
from .prompt_parser import Directives, fmt_t, parse_prompt

MIN_PIECE = 0.3   # shortest piece kept when following explicit instructions (s)
PAD = 0.15        # silence kept around speech when removing silent parts (s)

FAST_WORDS = r"\b(fast|quick|rapid|energetic|energy|hype|dynamic|snappy|punchy|tiktok style)\b"
SLOW_WORDS = r"\b(slow|calm|relax(ed|ing)?|chill|gentle|long takes?|vlog)\b"


def detect_unsupported(prompt: str) -> list[str]:
    """Parts of a prompt that no available operation can do."""
    return parse_prompt(prompt).unsupported


def shot_length_for(prompt: str) -> float:
    p = prompt.lower()
    if re.search(FAST_WORDS, p):
        return 1.3
    if re.search(SLOW_WORDS, p):
        return 3.5
    return 2.2


def _window_score(start: float, length: float, energy: list[float], win: float,
                  cuts: list[float], silent: list[list[float]]) -> float:
    score = 0.0
    if energy:
        i0 = int(start / win)
        i1 = max(i0 + 1, int(math.ceil((start + length) / win)))
        vals = energy[i0:i1] or [-90.0]
        score += sum(vals) / len(vals) / 10.0  # dB/10: louder → higher
    if any(abs(start - c) < 0.15 for c in cuts):
        score += 1.5
    if any(c - 0.1 > start and c + 0.1 < start + length for c in cuts):
        score -= 0.5  # a cut inside the window makes a jarring micro-shot
    for s0, s1 in silent:
        overlap = max(0.0, min(s1, start + length) - max(s0, start))
        score -= 3.0 * overlap / length
    return score


def _silent_fraction(start: float, length: float, silent: list[list[float]]) -> float:
    return sum(max(0.0, min(s1, start + length) - max(s0, start)) for s0, s1 in silent) / length


def _pick_windows(d: float, n: int, seg_len: float, energy: list[float], win: float,
                  cuts: list[float], silent: list[list[float]]) -> list[float]:
    """Choose n non-overlapping windows of seg_len in [0, d], spread across the clip.

    The clip is split into n zones; every zone that isn't mostly silent gets one
    window, then remaining picks go to the zones with the best leftover windows.
    """
    zone = d / n
    zones: list[list[tuple[float, float]]] = []  # per zone: [(score, start)] best first
    for k in range(n):
        z0, z1 = k * zone, min(d, (k + 1) * zone)
        last = max(z0, min(z1, d - seg_len))
        cands = {round(z0 + j * 0.25, 2) for j in range(int((last - z0) / 0.25) + 1)}
        cands |= {c for c in cuts if z0 <= c <= last}
        scored = sorted(((_window_score(st, seg_len, energy, win, cuts, silent), round(st, 3))
                         for st in cands if st + seg_len <= d + 1e-6), reverse=True)
        zones.append(scored)

    def usable(z: list[tuple[float, float]]) -> bool:
        return bool(z) and _silent_fraction(z[0][1], seg_len, silent) < 0.5

    active = [i for i, z in enumerate(zones) if usable(z)] or [i for i, z in enumerate(zones) if z]
    picks: list[float] = []

    def overlaps(st: float) -> bool:
        return any(st < p + seg_len - 1e-6 and p < st + seg_len - 1e-6 for p in picks)

    for i in active:
        for _, st in zones[i]:
            if not overlaps(st):
                picks.append(st)
                break
    while len(picks) < n:  # redistribute picks from skipped (silent) zones
        best = None
        for i in active:
            for sc, st in zones[i]:
                if not overlaps(st) and _silent_fraction(st, seg_len, silent) < 0.5:
                    if best is None or sc > best[0]:
                        best = (sc, st)
                    break
        if best is None:  # nothing non-silent left: accept any free window
            free = [st for z in zones for _, st in z if not overlaps(st)]
            if not free:
                break
            best = (0.0, free[0])
        picks.append(best[1])
    return sorted(picks)


def _explicit_segments(d: Directives, assets: dict[str, dict], analyses: dict[str, dict],
                       clip_ids: list[str], notes: list[str]) -> list[PlanSegment] | None:
    """Segments the prompt asked for literally (time ranges, first/last N s, remove silence)."""
    if not (d.ranges or d.first_s or d.last_s or d.remove_silence):
        return None
    aid = clip_ids[0]  # times in a prompt refer to the first clip
    if len(clip_ids) > 1 and (d.ranges or d.first_s or d.last_s):
        notes.append("Times in your prompt were applied to the first clip.")
    dur = float(assets[aid]["meta"]["duration"])
    pieces: list[tuple[float, float, str]] = []
    for a, b in d.ranges:
        if a >= dur - MIN_PIECE:
            notes.append(f"Skipped {fmt_t(a)}–{fmt_t(b)}: your video is only {dur:.1f}s long.")
            continue
        if b > dur:
            notes.append(f"{fmt_t(a)}–{fmt_t(b)} goes past the end of the video, so it stops at {fmt_t(dur)}.")
        pieces.append((a, min(b, dur), "time range from your prompt"))
    if d.first_s:
        pieces.append((0.0, min(d.first_s, dur), f"first {d.first_s:g}s"))
    if d.last_s:
        pieces.append((max(0.0, dur - d.last_s), dur, f"last {d.last_s:g}s"))
    if (d.ranges or d.first_s or d.last_s) and not pieces:
        raise EditorError("PROMPT_TIMES_INVALID",
                          f"The times in your prompt are outside the video (it is {dur:.1f}s long).")
    if not pieces:
        pieces = [(0.0, dur, "")]

    if d.remove_silence:
        audio = analyses.get(aid, {}).get("audio", {})
        if not audio.get("has_audio"):
            notes.append("Your video has no sound, so there were no silent parts to remove.")
        else:
            silent = audio.get("silent_ranges", [])
            kept: list[tuple[float, float, str]] = []
            for a, b, why in pieces:
                cur = a
                for s0, s1 in silent:
                    s0, s1 = s0 + PAD, s1 - PAD  # keep a little air around speech
                    if s1 <= cur or s0 >= b or s1 - s0 <= 0:
                        continue
                    if s0 > cur:
                        kept.append((cur, s0, why or "non-silent part"))
                    cur = max(cur, s1)
                if b > cur:
                    kept.append((cur, b, why or "non-silent part"))
            removed = sum(b - a for a, b, _ in pieces) - sum(b - a for a, b, _ in kept)
            notes.append(f"Removed {removed:.1f}s of silence." if removed > 0.05 else "No silent parts were found.")
            pieces = kept
    return [PlanSegment(asset_id=aid, src_in=round(a, 3), src_out=round(b, 3), reason=why)
            for a, b, why in pieces if b - a >= MIN_PIECE]


def _median_interval(beats: list[float]) -> float:
    if len(beats) < 3:
        return 0.0
    diffs = sorted(b - a for a, b in zip(beats, beats[1:]))
    return diffs[len(diffs) // 2]


def plan_rules(prompt: str, assets: dict[str, dict], analyses: dict[str, dict],
               target_duration: float | None, directives: Directives | None = None,
               max_duration: float | None = None, reference: dict | None = None,
               music: dict | None = None) -> EditPlan:
    """target_duration None = keep everything the prompt selected (no length target).

    reference: style profile of a reference video (style only, footage never used).
    music: {"asset_id", "analysis": {"beats", "tempo", ...}} for an uploaded song.
    """
    d = directives or parse_prompt(prompt)
    style = (reference or {}).get("style", {})
    notes: list[str] = list(d.notes)
    unsupported = list(d.unsupported)
    clip_ids = [aid for aid, a in assets.items() if a.get("role") == "clip"]
    total_src = sum(float(assets[a]["meta"]["duration"]) for a in clip_ids)
    volume = 1.0 if d.volume is None else d.volume
    speed = d.speed or 1.0
    beats = (music or {}).get("analysis", {}).get("beats", []) if music else []
    beat_iv = _median_interval(beats)
    beat_sync = bool((d.beat_sync or style.get("beat_sync")) and beat_iv)
    if d.beat_sync and not music:
        unsupported.append("beat-synced cuts (upload a music file to sync to)")
    if d.music and not music:
        unsupported.append("background music (no music file was uploaded)")

    # ---- pacing
    if d.pacing == "fast":
        shot = 1.3
    elif d.pacing == "slow":
        shot = 3.5
    elif style.get("target_shot_length"):
        shot = min(8.0, max(0.5, float(style["target_shot_length"])))
    else:
        shot = 2.2
    if beat_sync:
        k = max(1, round(shot / beat_iv))
        if beat_iv * k < 0.45:
            k += 1
        shot = beat_iv * k

    # ---- transitions
    trans_type = d.transition
    trans_every = 1
    if not trans_type and style.get("transition_type") and style.get("transition_ratio", 0) >= 0.25:
        trans_type = style["transition_type"]
        trans_every = max(1, round(1 / style["transition_ratio"]))
    trans_d = min(0.5, round(0.35 * shot, 2)) if trans_type else 0.0

    # ---- segments
    explicit = _explicit_segments(d, assets, analyses, clip_ids, notes)
    if explicit is not None:
        if not explicit:
            raise EditorError("PROMPT_TIMES_INVALID", "Nothing was left to use after following your prompt.")
        limit = min(x for x in (target_duration, max_duration) if x) if (target_duration or max_duration) else None
        if speed != 1.0:
            explicit = [sg.model_copy(update={"speed": speed}) for sg in explicit]
        total = sum(sg.out_len for sg in explicit)
        if limit and total > limit + 0.05:  # shorten from the end to fit the requested/allowed length
            kept, acc = [], 0.0
            for sg in explicit:
                room = limit - acc
                if room < MIN_PIECE:
                    break
                if sg.out_len > room:
                    sg = sg.model_copy(update={"src_out": round(sg.src_in + room * sg.speed, 3)})
                kept.append(sg)
                acc += sg.out_len
            notes.append(f"Your selection was {total:.1f}s, so it was shortened to {limit:g}s.")
            explicit = kept
        segments = explicit
        summary_cuts = f"{len(segments)} part(s) from your prompt"
    else:
        target = min(target_duration or total_src / speed, total_src / speed)
        n_total = max(1, round(target / shot))
        n_trans = len(range(1, n_total, trans_every)) if trans_type and n_total > 1 else 0
        segments = []
        if target >= total_src / speed - 0.05 and not trans_type:
            for aid in clip_ids:
                dur = float(assets[aid]["meta"]["duration"])
                segments.append(PlanSegment(asset_id=aid, src_in=0.0, src_out=round(dur, 3), reason="full clip",
                                            speed=speed))
            summary_cuts = f"the full {total_src / speed:.0f}s of footage"
        else:
            out_total = target + n_trans * trans_d  # transitions overlap, so pick a little more
            for aid in clip_ids:
                dur = float(assets[aid]["meta"]["duration"])
                share = out_total * dur / total_src
                n = max(1, round(n_total * dur / total_src))
                seg_out = share / n
                src_len = min(seg_out * speed, dur / n)
                an = analyses.get(aid, {})
                audio = an.get("audio", {})
                for st in _pick_windows(dur, n, src_len, audio.get("energy_db", []), audio.get("window_s", 0.5) or 0.5,
                                        [c["t"] for c in an.get("scenes", [])], audio.get("silent_ranges", [])):
                    segments.append(PlanSegment(asset_id=aid, src_in=st, src_out=round(min(dur, st + src_len), 3),
                                                reason="most active moment in this part", speed=speed))
            summary_cuts = f"{len(segments)} cuts of about {target / max(1, len(segments)):.1f}s"

    # ---- per-segment creative layer
    for i, sg in enumerate(segments):
        if trans_type and i > 0 and i % trans_every == 0:
            limit = min(segments[i - 1].out_len, sg.out_len) / 2 - 0.05
            if limit >= 0.15:
                sg.transition_in, sg.transition_duration = trans_type, round(min(trans_d or 0.5, limit), 2)
    zoom_kind, zoom_every = d.zoom, 1
    if not zoom_kind and style.get("zoom_ratio", 0) >= 0.25:
        zoom_kind, zoom_every = style.get("zoom_kind", "in"), max(1, round(1 / style["zoom_ratio"]))
    if zoom_kind:
        for i, sg in enumerate(segments):
            if i % zoom_every:
                continue
            kind = ("in" if (i // zoom_every) % 2 == 0 else "out") if zoom_kind == "alternate" else zoom_kind
            if kind == "pan_left" and i % 2:
                kind = "pan_right"
            sg.zoom, sg.zoom_amount = kind, 0.12 if d.pacing == "fast" else 0.15
    if (d.ramps or d.slowmo) and speed == 1.0:
        long_enough = [sg for sg in segments if sg.out_len >= 1.0]
        if d.slowmo and not d.ramps:
            chosen = long_enough[1::3][:3] or long_enough[:1]
            patterns = ["slow_middle"] * len(chosen)
        else:
            chosen = long_enough[::3]
            patterns = ["slow_middle" if j % 2 == 0 else "fast_middle" for j in range(len(chosen))]
        for sg, pat in zip(chosen, patterns):
            sg.speed_ramp = pat
        if not chosen:
            notes.append("The cuts are too short for speed ramps.")

    plan = EditPlan(segments=segments, original_audio_volume=volume, unsupported_requests=unsupported, notes=notes)
    total = plan.duration
    plan.color_preset = d.color or (style.get("color_preset") if style.get("color_preset") not in (None, "none") else "none")
    plan.effects = list(d.effects)
    any_audio = any(assets[a]["meta"].get("has_audio") for a in clip_ids)
    if d.captions or (style.get("captions") and any_audio and volume > 0):
        plan.captions.mode = "speech"
        plan.captions.style = d.caption_style if d.captions else ("bold" if style.get("caption_position") == "center"
                                                                   else "subtitle")
        plan.captions.position = d.caption_position if d.captions else (style.get("caption_position") or "bottom")
    # titles: quoted text, one after another from the start; "... at the end" goes last
    t_pos = "top" if plan.captions.mode == "speech" and plan.captions.position == "center" else "center"
    span = min(2.5, max(1.0, total / (len(d.texts) + (1 if d.end_text else 0) + 1)))
    for k, text in enumerate(d.texts):
        start = k * span
        if start < total - 0.3:
            plan.texts.append(PlanText(text=text, start=round(start, 2), end=round(min(total, start + span), 2),
                                       style="title", position=t_pos))
    if d.end_text and total > 1.0:
        plan.texts.append(PlanText(text=d.end_text, start=round(max(0.0, total - span), 2), end=round(total, 2),
                                   style="title", position=t_pos))
    if music:
        plan.music = PlanMusic(use=True, beat_sync=beat_sync,
                               volume=d.music_volume or (0.55 if any_audio and volume > 0 else 0.9),
                               duck=bool(any_audio and volume > 0))
    if d.sfx:
        plan.sfx_on_transitions = any(sg.transition_in != "none" for sg in segments)
        pos = 0.0
        for i, sg in enumerate(segments):
            if i > 0 and sg.transition_in == "none" and i % 4 == 0:
                plan.sound_effects.append(PlanSfx(name="swoosh", at=round(max(0.0, pos - 0.15), 2)))
            pos += sg.out_len - (sg.transition_duration if i > 0 and sg.transition_in != "none" else 0)
        for t in plan.texts[:3]:
            plan.sound_effects.append(PlanSfx(name="pop", at=t.start))
        if not plan.sound_effects and not plan.sfx_on_transitions and len(segments) > 1:
            pos = segments[0].out_len
            plan.sound_effects.append(PlanSfx(name="whoosh", at=round(max(0.0, pos - 0.3), 2)))
    plan.intro = d.intro or "none"
    plan.outro = d.outro or "none"

    parts = [summary_cuts]
    if beat_sync:
        parts.append("on the beat")
    if any(sg.transition_in != "none" for sg in segments):
        parts.append(f"{trans_type} transitions")
    if any(sg.zoom != "none" for sg in segments):
        parts.append("zooms")
    if any(sg.speed_ramp != "none" for sg in segments):
        parts.append("speed ramps")
    if plan.color_preset != "none":
        parts.append(f"{plan.color_preset} colour")
    if plan.captions.mode == "speech":
        parts.append("subtitles")
    if plan.texts:
        parts.append("titles")
    if plan.music.use:
        parts.append("music")
    if plan.sound_effects or plan.sfx_on_transitions:
        parts.append("sound effects")
    if reference:
        parts.append("paced like your reference")
    plan.summary = (", ".join(parts) + f" — {total:.0f}s total.").capitalize()
    return plan
