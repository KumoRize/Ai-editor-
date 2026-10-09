"""Build an ASS subtitle file (rendered by libass) from timeline text items.

User text never reaches the FFmpeg command line: it is written into this file
with ASS override characters neutralised.
"""
from __future__ import annotations

from .timeline import TextItem

FONT = "DejaVu Sans"
ALIGN = {"bottom": 2, "center": 5, "top": 8}


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape_ass(text: str) -> str:
    text = text.replace("\\", "/").replace("{", "(").replace("}", ")")
    return text.replace("\r", "").replace("\n", "\\N")


def _styles(W: int, H: int) -> list[str]:
    base = min(W, H)
    mv_bottom = int(H * (0.14 if H > W else 0.08))  # stay above platform UI on vertical video
    mv_top = int(H * (0.10 if H > W else 0.06))
    mh = int(W * 0.06)
    # Name, Fontname, Fontsize, Primary, Secondary, Outline, Back, Bold, Italic, Underline, StrikeOut,
    # ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    rows = {
        "subtitle": (int(base * 0.055), "&H00FFFFFF", "&H00000000", "&H80000000", -1, 1, max(2, base // 300), 1),
        "bold": (int(base * 0.085), "&H00FFFFFF", "&H00000000", "&H00000000", -1, 1, max(4, base // 160), 0),
        "minimal": (int(base * 0.045), "&H00FFFFFF", "&H00000000", "&H96000000", 0, 3, max(6, base // 120), 0),
        "title": (int(base * 0.10), "&H00FFFFFF", "&H00000000", "&H64000000", -1, 1, max(4, base // 180), 2),
    }
    out = []
    for name, (size, prim, outline_c, back, bold, border_style, outline, shadow) in rows.items():
        for pos, align in ALIGN.items():
            mv = mv_bottom if pos == "bottom" else mv_top if pos == "top" else 0
            out.append(f"Style: {name}_{pos},{FONT},{size},{prim},&H000000FF,{outline_c},{back},{bold},0,0,0,"
                       f"100,100,0,0,{border_style},{outline},{shadow},{align},{mh},{mh},{mv},1")
    return out


def build_ass(texts: list[TextItem], W: int, H: int) -> str:
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0",
             "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
             "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
             "Alignment, MarginL, MarginR, MarginV, Encoding", *_styles(W, H), "", "[Events]",
             "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for t in sorted(texts, key=lambda x: x.start):
        text = escape_ass(t.text.upper() if t.style == "bold" else t.text)
        fade = "{\\fad(150,150)}" if t.kind == "text" else ""
        layer = 1 if t.kind == "text" else 0
        lines.append(f"Dialogue: {layer},{_ts(t.start)},{_ts(t.end)},{t.style}_{t.position},,0,0,0,,{fade}{text}")
    return "\n".join(lines) + "\n"
