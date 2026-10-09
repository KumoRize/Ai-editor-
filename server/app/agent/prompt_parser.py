"""Deterministic reading of a free-form editing prompt (used by the built-in planner).

Extracts output settings (platform, aspect, resolution, length, framing) and
explicit editing directives (exact time ranges, first/last N seconds, remove
silence, mute / volume). The LLM planner does the same job more flexibly; this
keeps "Full prompt" mode working with no API key.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUM = r"\d+(?:\.\d+)?"
_CLOCK = r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?"
_UNIT_S = r"(?:s|secs?|seconds?)\b"
_UNIT_M = r"(?:m|mins?|minutes?)\b"


@dataclass
class Directives:
    preset: str | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    duration: float | None = None          # requested output length
    fit: str | None = None
    volume: float | None = None            # original audio volume (0 = mute)
    ranges: list[tuple[float, float]] = field(default_factory=list)   # explicit source ranges, in order
    first_s: float | None = None
    last_s: float | None = None
    remove_silence: bool = False
    notes: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        out = []
        if self.ranges:
            out.append("exact time ranges: " + ", ".join(f"{fmt_t(a)}–{fmt_t(b)}" for a, b in self.ranges))
        if self.first_s:
            out.append(f"first {self.first_s:g}s")
        if self.last_s:
            out.append(f"last {self.last_s:g}s")
        if self.remove_silence:
            out.append("silent parts removed")
        if self.volume is not None:
            out.append("original sound muted" if self.volume == 0 else f"original sound at {int(self.volume * 100)}%")
        return out


def fmt_t(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}" if m else f"{s:g}s"


def _clock_to_s(tok: str) -> float:
    tok = tok.strip().lower()
    if ":" in tok:
        parts = [float(p) for p in tok.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(re.sub(r"[a-z\s]+$", "", tok))


def parse_prompt(prompt: str) -> Directives:
    d = Directives()
    p = " " + prompt.lower().replace("–", "-").replace("—", "-") + " "

    # Aspect tokens first (and remove them so "9:16" is never read as a time).
    for tok, ar in (("9:16", "9:16"), ("16:9", "16:9"), ("1:1", "1:1"), ("4:5", None)):
        pat = rf"(?<![\d:.]){re.escape(tok)}(?![\d:.])"  # standalone only: "1:10" is a time, not 1:1
        if re.search(pat, p):
            if ar:
                d.aspect_ratio = d.aspect_ratio or ar
            else:
                d.notes.append("4:5 isn't offered yet; using 9:16.")
                d.aspect_ratio = d.aspect_ratio or "9:16"
            p = re.sub(pat, " ", p)

    # Platform presets.
    if re.search(r"\b(youtube shorts?|yt shorts?|shorts)\b", p):
        d.preset = "youtube_shorts"
    elif re.search(r"\btik ?tok\b", p):
        d.preset = "tiktok"
    elif re.search(r"\b(insta(gram)?|reels?|ig)\b", p):
        d.preset = "instagram_reels"
    elif re.search(r"\byoutube\b", p):
        d.preset = "youtube_landscape"

    if not d.aspect_ratio:
        if re.search(r"\b(vertical|portrait|story|stories)\b", p):
            d.aspect_ratio = "9:16"
        elif re.search(r"\b(landscape|horizontal|widescreen)\b", p):
            d.aspect_ratio = "16:9"
        elif re.search(r"\bsquare\b", p):
            d.aspect_ratio = "1:1"
            d.preset = d.preset or "square"

    if re.search(r"\b(4k|2160p|uhd)\b", p):
        d.resolution = "1080p"
        d.notes.append("4K output isn't offered; rendering at 1080p.")
    elif re.search(r"\b(720p?|hd ready)\b", p):
        d.resolution = "720p"
    elif re.search(r"\b(1080p?|full ?hd|fhd)\b", p):
        d.resolution = "1080p"

    # Framing. Negative forms ("don't crop") before positive ones ("crop").
    if re.search(r"\bblur(red)?\b", p):
        d.fit = "blur"
    elif re.search(r"\b(don'?t|do not|without|no) crop|black bars|letterbox|pillarbox|whole frame|show everything\b", p):
        d.fit = "contain"
    elif re.search(r"\b(crop|fill( the)? (frame|screen)|full ?screen)\b", p):
        d.fit = "cover"

    # Audio.
    if re.search(r"\b(mute|muted|no (sound|audio)|without (sound|audio)|remove (the )?(sound|audio)|silent video)\b", p):
        d.volume = 0.0
    elif re.search(r"\b(louder|boost (the )?(volume|audio|sound))\b", p):
        d.volume = 1.5
    elif re.search(r"\b(quieter|lower (the )?(volume|audio|sound)|softer)\b", p):
        d.volume = 0.6
    if re.search(r"\b(remove|cut|skip|trim|delete|no)( out)?( the| all)? (silen(ce|t parts?)|pauses?|dead air|gaps)\b", p):
        d.remove_silence = True

    # Explicit ranges: "0:05-0:12", "from 1:02 to 1:10", "5s to 12s", "from 5 to 12 seconds".
    spans: list[tuple[int, float, float]] = []
    t = rf"({_CLOCK}|{_NUM}\s*{_UNIT_S})"
    for m in re.finditer(rf"(?:from\s+)?{t}\s*(?:-|to|until|till)\s*{t}", p):
        spans.append((m.start(), _clock_to_s(m.group(1)), _clock_to_s(m.group(2))))
        p = p[:m.start()] + " " * (m.end() - m.start()) + p[m.end():]
    for m in re.finditer(rf"(?:from\s+)?({_NUM})\s*(?:-|to)\s*({_NUM})\s*{_UNIT_S}", p):
        spans.append((m.start(), float(m.group(1)), float(m.group(2))))
        p = p[:m.start()] + " " * (m.end() - m.start()) + p[m.end():]
    for _, a, b in sorted(spans):
        if b > a:
            d.ranges.append((round(a, 3), round(b, 3)))
        else:
            d.notes.append(f"Ignored the range {fmt_t(a)}–{fmt_t(b)} (the end is not after the start).")

    for m in re.finditer(rf"\b(first|opening|beginning)\s+({_NUM})\s*({_UNIT_S}|{_UNIT_M})", p):
        d.first_s = float(m.group(2)) * (60 if m.group(3).startswith("m") else 1)
        p = p.replace(m.group(0), " ")
    for m in re.finditer(rf"\b(last|final|ending)\s+({_NUM})\s*({_UNIT_S}|{_UNIT_M})", p):
        d.last_s = float(m.group(2)) * (60 if m.group(3).startswith("m") else 1)
        p = p.replace(m.group(0), " ")

    # Output length: "30-second reel", "30s", "1 minute", "under 45 seconds".
    m = re.search(rf"({_NUM})\s*-?\s*{_UNIT_M}", p)
    if m:
        d.duration = float(m.group(1)) * 60
    m = re.search(rf"({_NUM})\s*-?\s*{_UNIT_S}", p)
    if m:
        d.duration = float(m.group(1)) if d.duration is None else d.duration
    if d.duration is not None and d.duration <= 0:
        d.duration = None
    return d
