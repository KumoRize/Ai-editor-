"""Deterministic planner: works with no API key, used for tests, offline mode,
and as an honest fallback when the LLM planner is unavailable.

Strategy: decide a shot length from pacing words in the prompt, split the source
into equal zones (keeps the story chronological and covers the whole video), and
in each zone pick the window with the most audio energy, preferring windows that
start on a scene cut and avoiding silence.
"""
from __future__ import annotations

import math
import re

from .plan import EditPlan, PlanSegment

# Requested features that are not implemented yet, mapped to the phase that adds them.
UNSUPPORTED_KEYWORDS: list[tuple[str, str]] = [
    (r"\b(subtitles?|captions?)\b", "subtitles/captions (Phase 2)"),
    (r"\b(text overlay|title card|add text|lower third)\b", "text overlays (Phase 2)"),
    (r"\b(music|song|soundtrack)\b", "background music (Phase 2)"),
    (r"\b(reference|similar to|like this video|same style as)\b", "matching a reference video (Phase 3)"),
    (r"\b(zoom|zooms|punch[- ]?in)\b", "zooms (Phase 4)"),
    (r"\b(speed ramps?|slow[- ]?mo(tion)?|time[- ]?lapse|speed up|slow down)\b", "speed changes (Phase 4)"),
    (r"\b(transitions?|crossfade|dissolve|wipe)\b", "transitions (Phase 4)"),
    (r"\b(beat[- ]?sync(ed)?|on the beat|beats?)\b", "beat-synced cuts (Phase 4)"),
    (r"\b(sound effects?|sfx|whoosh)\b", "sound effects (Phase 4)"),
    (r"\b(colou?r grad(e|ing)|filter|lut|warm(er)? look|cinematic look)\b", "color grading (Phase 4)"),
]

FAST_WORDS = r"\b(fast|quick|rapid|energetic|energy|hype|dynamic|snappy|punchy|tiktok style)\b"
SLOW_WORDS = r"\b(slow|calm|relax(ed|ing)?|chill|gentle|long takes?|vlog)\b"


def detect_unsupported(prompt: str) -> list[str]:
    p = prompt.lower()
    found = []
    for pattern, label in UNSUPPORTED_KEYWORDS:
        if re.search(pattern, p) and label not in found:
            found.append(label)
    return found


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


def plan_rules(prompt: str, assets: dict[str, dict], analyses: dict[str, dict],
               target_duration: float) -> EditPlan:
    clip_ids = [aid for aid, a in assets.items() if a.get("role") == "clip"]
    total_src = sum(float(assets[a]["meta"]["duration"]) for a in clip_ids)
    target = min(target_duration, total_src)
    shot = shot_length_for(prompt)
    segments: list[PlanSegment] = []

    if target >= total_src - 0.05:
        for aid in clip_ids:
            d = float(assets[aid]["meta"]["duration"])
            segments.append(PlanSegment(asset_id=aid, src_in=0.0, src_out=round(d, 3), reason="full clip"))
        summary = f"Used the full {total_src:.0f}s of footage (it is not longer than the requested length)."
    else:
        n_total = max(1, round(target / shot))
        for aid in clip_ids:
            d = float(assets[aid]["meta"]["duration"])
            share = target * d / total_src           # this clip's share of the output
            n = max(1, round(n_total * d / total_src))
            seg_len = share / n
            an = analyses.get(aid, {})
            audio = an.get("audio", {})
            energy = audio.get("energy_db", [])
            win = audio.get("window_s", 0.5) or 0.5
            cuts = [c["t"] for c in an.get("scenes", [])]
            silent = audio.get("silent_ranges", [])
            for st in _pick_windows(d, n, seg_len, energy, win, cuts, silent):
                segments.append(PlanSegment(asset_id=aid, src_in=st, src_out=round(min(d, st + seg_len), 3),
                                            reason="most active moment in this part"))
        summary = (f"{len(segments)} cuts of about {target / max(1, len(segments)):.1f}s each, "
                   f"picked from the liveliest moments, {target:.0f}s total.")

    return EditPlan(summary=summary, segments=segments, unsupported_requests=detect_unsupported(prompt))
