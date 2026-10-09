"""Automatic quality check of an exported video (agent self-verification).

One decoding pass measures: black sections, frozen picture, integrated
loudness and true peak. Findings are compared with what the timeline intends
(fades to black, muted sound...) so intended effects are not flagged.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..config import get_settings
from ..engine.timeline import Timeline
from ..proc import run


def _expected_black(tl: Timeline) -> list[tuple[float, float]]:
    win: list[tuple[float, float]] = []
    st = tl.settings
    if st.intro:
        win.append((0.0, st.intro.duration + 0.3))
    if st.outro:
        win.append((tl.duration - st.outro.duration - 0.3, tl.duration + 1))
    for c in tl.video_track().clips:
        if c.transition_in and c.transition_in.type in ("fadeblack", "fade"):
            win.append((c.timeline_start - 0.3, c.timeline_start + c.transition_in.duration + 0.3))
    return win


def _expects_sound(tl: Timeline, assets: dict[str, dict]) -> bool:
    if tl.audio:
        return True
    return any(c.volume > 0 and assets.get(c.asset_id, {}).get("meta", {}).get("has_audio")
               for c in tl.video_track().clips)


def quality_check(path: Path, tl: Timeline, assets: dict[str, dict]) -> dict:
    s = get_settings()
    cp = run([s.ffmpeg_bin, "-nostdin", "-hide_banner", "-v", "info", "-i", str(path),
              "-filter_complex", "[0:v:0]blackdetect=d=0.5:pix_th=0.10,freezedetect=n=0.002:d=3[v];"
                                 "[0:a:0]ebur128=peak=true[a]",
              "-map", "[v]", "-map", "[a]", "-f", "null", "-"], timeout=s.render_timeout_s)
    log = (cp.stderr or b"").decode(errors="replace")
    issues: list[dict] = []
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)]
    expected = _expected_black(tl)
    for a, b in black:
        inside = sum(max(0.0, min(b, e1) - max(a, e0)) for e0, e1 in expected)
        if (b - a) - inside > 0.4:
            issues.append({"code": "BLACK_FRAMES", "severity": "warning", "fixable": False,
                           "message": f"The picture is black from {a:.1f}s to {b:.1f}s (it is black in the source)."})
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", log)]
    for k, a in enumerate(starts):
        b = ends[k] if k < len(ends) else tl.duration
        if b - a >= 3:
            issues.append({"code": "FROZEN", "severity": "info", "fixable": False,
                           "message": f"The picture doesn't move from {a:.1f}s to {b:.1f}s."})
    m = re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", log)
    loud = float(m[-1]) if m and m[-1] != "-inf" else -120.0
    pk = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", log)
    peak = float(pk[-1]) if pk and pk[-1] != "-inf" else -120.0
    wants_sound = _expects_sound(tl, assets)
    if wants_sound and loud < -35:
        issues.append({"code": "TOO_QUIET", "severity": "warning", "fixable": not tl.settings.normalize_audio,
                       "message": f"The sound is very quiet ({loud:.0f} LUFS)."})
    if peak > -0.2:
        issues.append({"code": "CLIPPING", "severity": "warning", "fixable": True,
                       "message": f"The sound peaks at {peak:.1f} dBFS and may distort."})
    if not wants_sound and loud > -60:
        issues.append({"code": "UNEXPECTED_SOUND", "severity": "info", "fixable": False,
                       "message": "There is sound although all clips are muted."})
    return {"passed": not any(i["severity"] == "warning" for i in issues), "issues": issues,
            "measurements": {"loudness_lufs": loud, "true_peak_dbfs": peak, "black_sections": len(black),
                             "frozen_sections": len(starts)},
            "decoded": cp.returncode == 0}
