"""Built-in sound effects, synthesised with FFmpeg (no downloads, no licensing issues)."""
from __future__ import annotations

import threading
from pathlib import Path

from ..config import get_settings
from ..errors import EditorError
from ..proc import run

SFX_VERSION = 1
# name -> (lavfi source, extra audio filters)
RECIPES: dict[str, tuple[str, str]] = {
    "whoosh": ("anoisesrc=color=pink:duration=0.8:sample_rate=48000:amplitude=0.9",
               "highpass=f=250,lowpass=f=3500,volume='pow(sin(PI*t/0.8),3)':eval=frame"),
    "swoosh": ("anoisesrc=color=white:duration=0.4:sample_rate=48000:amplitude=0.6",
               "highpass=f=900,lowpass=f=7000,volume='pow(sin(PI*t/0.4),2)':eval=frame"),
    "impact": ("aevalsrc='0.9*sin(2*PI*52*t)*exp(-3.5*t)+0.35*(random(0)*2-1)*exp(-30*t)':d=1.4:s=48000", ""),
    "pop": ("aevalsrc='0.8*sin(2*PI*(1400-2600*t)*t)*exp(-28*t)':d=0.18:s=48000", ""),
    "click": ("aevalsrc='0.9*(random(0)*2-1)*exp(-220*t)':d=0.06:s=48000", "highpass=f=1500"),
    "riser": ("aevalsrc='0.45*sin(2*PI*(180*t+260*t*t))*(t/2)+0.12*(random(0)*2-1)*(t/2)':d=2:s=48000",
              "afade=t=out:st=1.9:d=0.1"),
    "shutter": ("aevalsrc='0.9*(random(0)*2-1)*(exp(-160*t)+0.8*exp(-160*abs(t-0.11))*gte(t,0.11))':d=0.3:s=48000",
                "highpass=f=800"),
}
_lock = threading.Lock()


def sfx_dir() -> Path:
    d = get_settings().data_dir / "cache" / f"sfx_v{SFX_VERSION}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sfx_path(name: str) -> Path:
    if name not in RECIPES:
        raise EditorError("UNKNOWN_SFX", f"Unknown sound effect '{name}'.")
    out = sfx_dir() / f"{name}.wav"
    with _lock:
        if not out.is_file() or out.stat().st_size < 1000:
            src, af = RECIPES[name]
            s = get_settings()
            tmp = out.with_suffix(".tmp.wav")
            cmd = [s.ffmpeg_bin, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", src]
            if af:
                cmd += ["-af", af]
            cmd += ["-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(tmp)]
            cp = run(cmd, timeout=60)
            if cp.returncode != 0:
                raise EditorError("SFX_FAILED", "A built-in sound effect could not be created.",
                                  detail=(cp.stderr or b"").decode(errors="replace")[-500:])
            tmp.replace(out)
    return out


def all_sfx_paths(names: set[str] | None = None) -> dict[str, Path]:
    return {n: sfx_path(n) for n in (names if names is not None else RECIPES)}
