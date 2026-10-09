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
    # creative directives
    pacing: str | None = None              # "fast" | "slow"
    texts: list[str] = field(default_factory=list)      # quoted text → title overlays
    end_text: str | None = None
    captions: bool = False
    caption_style: str = "subtitle"
    caption_position: str = "bottom"
    zoom: str | None = None                # "alternate" | ZOOM_KINDS value
    speed: float | None = None             # whole-video speed
    slowmo: bool = False
    ramps: bool = False
    transition: str | None = None
    intro: str | None = None
    outro: str | None = None
    beat_sync: bool = False
    sfx: bool = False
    color: str | None = None
    effects: list[str] = field(default_factory=list)
    music: bool = False                    # music explicitly asked for
    music_volume: float | None = None
    unsupported: list[str] = field(default_factory=list)

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
        if self.captions:
            out.append(f"{self.caption_style} subtitles ({self.caption_position})")
        if self.texts or self.end_text:
            out.append("text: " + ", ".join(f"“{t}”" for t in self.texts + ([self.end_text] if self.end_text else [])))
        if self.zoom:
            out.append("zooms" if self.zoom == "alternate" else
                       {"in": "zoom-ins", "out": "zoom-outs", "punch_in": "punch-in zooms", "ken_burns": "Ken Burns zooms",
                        "pan_left": "pans"}.get(self.zoom, "zooms"))
        if self.speed:
            out.append(f"{self.speed:g}x speed")
        if self.slowmo or self.ramps:
            out.append("speed ramps / slow motion")
        if self.transition:
            from ..engine.timeline import TRANSITION_LABELS
            out.append(f"{TRANSITION_LABELS.get(self.transition, self.transition)} transitions")
        if self.intro:
            out.append("fade in")
        if self.outro:
            out.append("fade out")
        if self.beat_sync:
            out.append("cuts on the beat")
        if self.sfx:
            out.append("sound effects")
        if self.color:
            out.append(f"{self.color} colour")
        if self.effects:
            out.append(", ".join(self.effects))
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


UNSUPPORTED: list[tuple[str, str]] = [
    (r"\bstabili[sz]", "video stabilisation"),
    (r"\b(green ?screen|chroma ?key|remove (the )?background)\b", "background removal / green screen"),
    (r"\b(voice ?-?over|narration|ai voice|text to speech|tts)\b", "voice-over"),
    (r"\btranslat", "translated subtitles"),
    (r"\b(stickers?|emojis?|gifs?)\b", "stickers / emojis"),
    (r"\b(blur (the )?faces?|face blur|track(ing)? (the )?(face|object|person))\b", "face blur / object tracking"),
    (r"\b(lip ?sync)\b", "lip sync"),
    (r"\b(generate|ai[- ]generated) (images?|video|footage|b-?roll)\b", "generated footage"),
]
_QUOTES = r"[\"“”„«»]([^\"“”„«»]{1,80})[\"“”„«»]|(?<![a-z])'([^']{1,80})'(?![a-z])"


def _parse_creative(d: Directives, original: str, p: str) -> str:
    """Creative directives. Returns the prompt with quoted text removed (so it isn't parsed as settings)."""
    # Quoted text → text overlays ("... at the end" puts it at the end).
    for m in list(re.finditer(_QUOTES, original)):
        text = (m.group(1) or m.group(2) or "").strip()
        if not text:
            continue
        # Only look at this title's own clause: up to the next quote or punctuation on each side.
        after = re.split(r"[\"“”„«»,.;!?]|\band\b", original[m.end(): m.end() + 40].lower())[0]
        before = re.split(r"[\"“”„«»,.;!?]", original[max(0, m.start() - 40): m.start()].lower())[-1]
        if re.search(r"\b(at|in) the (very )?end\b|\bending\b|\boutro\b", after) or \
                re.search(r"\b(end|ending|outro|closing|final)\b", before):
            d.end_text = text
        else:
            d.texts.append(text)
    p = re.sub(_QUOTES, " ", p)

    if re.search(r"\b(fast|quick|rapid|energetic|energy|hype|dynamic|snappy|punchy)\b", p):
        d.pacing = "fast"
    elif re.search(r"\b(slow[- ]paced|calm|relax(ed|ing)?|chill|gentle|long takes?)\b", p):
        d.pacing = "slow"

    if re.search(r"\b(subtitles?|captions?|caption it|transcri(be|ption))\b", p):
        d.captions = True
        if re.search(r"\b(bold|big|tiktok[- ]style|word by word|animated|large)\b", p):
            d.caption_style = "bold"
        elif re.search(r"\b(minimal|simple|small|clean)\b (subtitles?|captions?)", p):
            d.caption_style = "minimal"
        if re.search(r"(subtitles?|captions?)[^.]{0,30}\b(top)\b", p):
            d.caption_position = "top"
        elif re.search(r"(subtitles?|captions?)[^.]{0,30}\b(center|centre|middle)\b", p) or d.caption_style == "bold":
            d.caption_position = "center"

    if re.search(r"\bken ?burns\b", p):
        d.zoom = "ken_burns"
    elif re.search(r"\bpunch[- ]?ins?\b", p):
        d.zoom = "punch_in"
    elif re.search(r"\bzoom(s|ing)? out\b", p):
        d.zoom = "out"
    elif re.search(r"\b(slow|cinematic|smooth) zoom", p) or re.search(r"\bzoom(s|ing)? in\b", p):
        d.zoom = "in"
    elif re.search(r"\bpan(ning|s)?\b", p) and not re.search(r"\bpan(cake|el)", p):
        d.zoom = "pan_left"
    elif re.search(r"\bzoom(s|ing|ed)?\b(?! transitions?)", p):
        d.zoom = "alternate"

    m = re.search(r"\b(\d(?:\.\d)?)x\b", p)
    if re.search(r"\b(time ?-?lapse|speed (it|everything|the video) up|faster playback)\b", p) or \
            (m and re.search(r"\b(speed|faster|fast forward)\b", p)):
        d.speed = min(4.0, max(0.25, float(m.group(1)))) if m else 2.0
    if re.search(r"\bslow[- ]?mo(tion)?\b", p):
        d.slowmo = True
    if re.search(r"\bspeed[- ]?ramps?(ing)?\b", p):
        d.ramps = True

    if re.search(r"\b(dissolves?|cross ?-?fades?|smooth transitions?)\b", p):
        d.transition = "dissolve"
    elif re.search(r"\bwipes?\b", p):
        d.transition = "wipeleft"
    elif re.search(r"\bslides?\b", p):
        d.transition = "slideleft"
    elif re.search(r"\bzoom transitions?\b", p):
        d.transition = "zoomin"
    elif re.search(r"\b(fade|dip) (to|through) (black|white)\b|\bfade transitions?\b", p):
        d.transition = "fadewhite" if "white" in p else "fadeblack"
    elif re.search(r"\btransitions?\b", p) and not re.search(r"\bno transitions?\b", p):
        d.transition = "dissolve"
    if re.search(r"\bfade[sd]? in\b|\bfade from black\b", p):
        d.intro = "fadeblack"
    if re.search(r"\bfade[sd]? out\b|\bfade to black at the end\b|\bend with a fade\b", p):
        d.outro = "fadeblack"

    if re.search(r"\b(beat[- ]?sync(ed)?|on the beats?|to the beats?|with the beats?|beat drops?|rhythm)\b", p):
        d.beat_sync = True
    if re.search(r"\b(sound effects?|sfx|whooshe?s?|swooshe?s?)\b", p):
        d.sfx = True

    for pat, name in ((r"\b(black and white|black & white|b&w|monochrome|greyscale|grayscale)\b", "bw"),
                      (r"\b(vintage|retro|old film|film look)\b", "vintage"),
                      (r"\b(warm|golden|sunset look)\b", "warm"),
                      (r"\b(cool|cold|blue tones?)\b(?! transitions?)", "cool"),
                      (r"\b(vibrant|colou?rful|saturated|punchy colou?rs?)\b", "vibrant"),
                      (r"\b(moody|dark tone|dramatic)\b", "moody"),
                      (r"\bcinematic\b(?! zoom)", "cinematic")):
        if re.search(pat, p):
            d.color = name
            break
    for pat, name in ((r"\bvignette\b", "vignette"), (r"\b(film )?grain\b", "grain"),
                      (r"\bsharpen|\bsharper\b", "sharpen"), (r"\bsepia\b", "sepia")):
        if re.search(pat, p):
            d.effects.append(name)

    if re.search(r"\b(music|song|soundtrack|track)\b", p):
        d.music = True
        if re.search(r"\b(quiet|soft|low|background) (music|song)", p):
            d.music_volume = 0.5
        elif re.search(r"\b(loud|full) (music|song)", p):
            d.music_volume = 1.0
        if re.search(r"\b(only (the )?(music|song)|music only|replace the (sound|audio))\b", p):
            d.volume = 0.0
    for pat, label in UNSUPPORTED:
        if re.search(pat, p) and label not in d.unsupported:
            d.unsupported.append(label)
    return p


def parse_prompt(prompt: str) -> Directives:
    d = Directives()
    p = " " + prompt.lower().replace("–", "-").replace("—", "-") + " "
    p = _parse_creative(d, prompt.replace("–", "-").replace("—", "-"), p)

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
