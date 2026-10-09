"""Reference-video analysis: a style/pacing profile, never footage to reuse.

Everything runs locally on a small proxy (4 frames/s at 96 px wide), cached by
content hash. Heuristic measurements carry a confidence label instead of being
presented as facts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..errors import EditorError
from ..proc import run
from ..storage.files import cache_dir
from .beats import analyze_beats
from .media import detect_scenes

REF_VERSION = 1
SAMPLE_FPS = 4
PROXY_W = 96
MAX_SECONDS = 180


def _frames(path: Path, meta: dict) -> tuple[np.ndarray, int, int]:
    s = get_settings()
    w = PROXY_W
    h = max(2, int(round(PROXY_W * meta["height"] / meta["width"] / 2)) * 2)
    cp = run([s.ffmpeg_bin, "-nostdin", "-v", "error", "-t", str(MAX_SECONDS), "-i", str(path), "-map", "0:v:0",
              "-vf", f"fps={SAMPLE_FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
             timeout=s.analysis_timeout_s)
    if cp.returncode != 0 or not cp.stdout:
        raise EditorError("REFERENCE_FAILED", "The reference video could not be analyzed.")
    arr = np.frombuffer(cp.stdout, dtype=np.uint8)
    n = len(arr) // (w * h * 3)
    return arr[: n * w * h * 3].reshape(n, h, w, 3).astype(np.float32) / 255.0, w, h


def _gray(f: np.ndarray) -> np.ndarray:
    return f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114


def _blur(a: np.ndarray) -> np.ndarray:
    k = np.array([1, 2, 1], dtype=np.float32) / 4
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, a)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, a)


def _zoom_of(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Best scale s such that b ≈ a zoomed by s (s>1 zoom in). Returns (s, error ratio vs. s=1)."""
    a, b = _blur(a), _blur(b)
    a = (a - a.mean()) / (a.std() + 1e-4)
    b = (b - b.mean()) / (b.std() + 1e-4)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    m = 0.18  # ignore borders
    sl = (slice(int(h * m), int(h * (1 - m))), slice(int(w * m), int(w * (1 - m))))

    def err(s: float) -> float:
        fy, fx = cy + (yy - cy) / s, cx + (xx - cx) / s  # bilinear sample of a at the scaled position
        y0, x0 = np.clip(np.floor(fy).astype(int), 0, h - 2), np.clip(np.floor(fx).astype(int), 0, w - 2)
        wy, wx = np.clip(fy - y0, 0, 1), np.clip(fx - x0, 0, 1)
        v = (a[y0, x0] * (1 - wy) * (1 - wx) + a[y0 + 1, x0] * wy * (1 - wx) +
             a[y0, x0 + 1] * (1 - wy) * wx + a[y0 + 1, x0 + 1] * wy * wx)
        return float(np.mean((v[sl] - b[sl]) ** 2))

    base = err(1.0)
    best_s, best_e = 1.0, base
    for s in (0.8, 0.84, 0.88, 0.92, 0.95, 1.05, 1.08, 1.12, 1.16, 1.2, 1.25):
        e = err(s)
        if e < best_e:
            best_s, best_e = s, e
    return best_s, (best_e / base if base > 1e-6 else 1.0)


def _shot_zoom(frames: list[np.ndarray]) -> float:
    """Zoom over a shot (1 = none). Needs both halves to agree on the direction."""
    if len(frames) < 3:
        s, r = _zoom_of(frames[0], frames[-1])
        return s if r < 0.75 else 1.0
    mid = len(frames) // 2
    s1, r1 = _zoom_of(frames[0], frames[mid])
    s2, r2 = _zoom_of(frames[mid], frames[-1])
    s, r = _zoom_of(frames[0], frames[-1])
    same_dir = (s1 - 1) * (s2 - 1) > 0 or abs(s1 - 1) < 0.03 or abs(s2 - 1) < 0.03
    return s if (r < 0.75 and same_dir and abs(s - 1) >= 0.05) else 1.0


def _is_blend(a: np.ndarray, b: np.ndarray, mid: np.ndarray) -> bool:
    """True if `mid` looks like a cross-fade of two clearly different pictures a and b."""
    apart = np.abs(a - b).mean()
    if apart < 0.12:  # different shots differ a lot; slow zooms/pans inside a shot differ a little
        return False
    near = min(np.abs(mid - a).mean(), np.abs(mid - b).mean())
    # a blend differs from both neighbours (a hard cut matches one of them)
    return bool(near > 0.3 * apart and np.abs(mid - (a + b) / 2).mean() < 0.7 * near)


def _aspect_name(w: int, h: int) -> str:
    r = w / h
    return min({"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:5": 0.8}.items(), key=lambda kv: abs(kv[1] - r))[0]


def analyze_reference(path: Path, sha256: str, meta: dict) -> dict:
    cfile = cache_dir(sha256) / f"reference_v{REF_VERSION}.json"
    if cfile.exists():
        try:
            return json.loads(cfile.read_text())
        except json.JSONDecodeError:
            cfile.unlink(missing_ok=True)
    try:
        prof = _analyze(path, meta)
    except EditorError:
        raise
    except Exception as e:  # noqa: BLE001
        raise EditorError("REFERENCE_FAILED", "The reference video could not be analyzed.", detail=str(e)) from e
    cfile.write_text(json.dumps(prof))
    return prof


def _analyze(path: Path, meta: dict) -> dict:
    dur = min(float(meta["duration"]), MAX_SECONDS)
    frames, w, h = _frames(path, meta)
    n = len(frames)
    gray = _gray(frames)
    luma = gray.mean(axis=(1, 2))

    # ---- shots & pacing
    cuts = [c["t"] for c in detect_scenes(path, dur, threshold=0.25) if c["t"] < dur]
    # Gradual transitions (dissolves/fades) change the picture over several frames, so the
    # scene-cut detector can miss them: look for a sampled frame that is a blend of its neighbours.
    soft_cuts = []
    for k in range(1, n - 1):
        if _is_blend(gray[k - 1], gray[k + 1], gray[k]):
            t = k / SAMPLE_FPS
            if all(abs(t - c) > 0.6 for c in cuts + soft_cuts):
                soft_cuts.append(t)
    cuts = sorted(cuts + soft_cuts)
    bounds = [0.0] + cuts + [dur]
    shots = np.diff(bounds)
    shots = shots[shots > 0.05]
    spans = [((a + b) / 2, b - a) for a, b in zip(bounds, bounds[1:])]
    quarters = []
    for q in range(4):
        a, b = dur * q / 4, dur * (q + 1) / 4
        lens = [ln for mid, ln in spans if a <= mid < b]
        quarters.append(round(float(np.mean(lens)) if lens else b - a, 2))
    shots_info = {"count": int(len(shots)), "avg": round(float(shots.mean()), 2), "median": round(float(np.median(shots)), 2),
                  "min": round(float(shots.min()), 2), "max": round(float(shots.max()), 2),
                  "cuts_per_minute": round(len(cuts) / dur * 60, 1), "pacing_by_quarter": quarters}

    # ---- transitions at each cut
    kinds = {"hard_cut": 0, "dissolve": 0, "fade_black": 0, "fade_white": 0}
    for c in cuts:
        i = int(round(c * SAMPLE_FPS))
        lo, hi = max(0, i - 3), min(n - 1, i + 3)
        if hi - lo < 4:
            kinds["hard_cut"] += 1
            continue
        win = luma[lo:hi + 1]
        side = min(luma[lo], luma[hi])
        if win.min() < 0.06 and side > 0.12:
            kinds["fade_black"] += 1
        elif win.max() > 0.93 and max(luma[lo], luma[hi]) < 0.85:
            kinds["fade_white"] += 1
        else:
            dissolve = any(_is_blend(gray[k - 1], gray[k + 1], gray[k]) for k in range(max(1, i - 2), min(n - 1, i + 2)))
            kinds["dissolve" if dissolve else "hard_cut"] += 1
    soft = len(cuts) - kinds["hard_cut"]
    trans = {**kinds, "soft_ratio": round(soft / len(cuts), 2) if cuts else 0.0,
             "main_soft_type": max(("dissolve", "fade_black", "fade_white"), key=lambda k: kinds[k]) if soft else None}

    # ---- zoom & motion per shot
    zin = zout = 0
    amounts: list[float] = []
    motion: list[float] = []
    speed_flags = 0
    for s, e in zip(bounds, bounds[1:]):
        i0, i1 = int(math.ceil(s * SAMPLE_FPS)), min(n - 1, int(math.ceil(e * SAMPLE_FPS)) - 1)
        if i1 - i0 < 2:
            continue
        diffs = np.abs(np.diff(gray[i0:i1 + 1], axis=0)).mean(axis=(1, 2))
        motion.extend(diffs.tolist())
        half = len(diffs) // 2
        if half >= 2:
            m1, m2 = diffs[:half].mean() + 1e-4, diffs[half:].mean() + 1e-4
            if max(m1, m2) / min(m1, m2) > 3 and max(m1, m2) > 0.02:
                speed_flags += 1
        sc = _shot_zoom([gray[k] for k in range(i0, i1 + 1)])
        if sc != 1.0:
            if sc > 1:
                zin += 1
            else:
                zout += 1
            amounts.append(abs(sc - 1))
    analysed = max(1, len(shots))
    zoom = {"zoom_in_shots": zin, "zoom_out_shots": zout, "ratio": round((zin + zout) / analysed, 2),
            "avg_amount": round(float(np.mean(amounts)), 2) if amounts else 0.0, "confidence": "medium"}
    mot = np.array(motion) if motion else np.zeros(1)
    motion_info = {"mean": round(float(mot.mean()), 4), "variability": round(float(mot.std() / (mot.mean() + 1e-6)), 2),
                   "possible_speed_change_shots": speed_flags, "confidence": "low"}

    # ---- on-screen captions (bright strokes with strong edges that persist)
    bands = {"top": (0.04, 0.28), "center": (0.36, 0.64), "bottom": (0.66, 0.94)}
    gy = np.abs(np.diff(gray, axis=1))[:, :, :-1]
    gx = np.abs(np.diff(gray, axis=2))[:, :-1, :]
    edges = (gx + gy) > 0.25
    bright = gray[:, :-1, :-1] > 0.82
    text_mask = edges & np.roll(bright, 1, axis=2)
    stable = text_mask[1:] & text_mask[:-1]
    per_band = {}
    for name, (a, b) in bands.items():
        ya, yb = int(a * (h - 1)), int(b * (h - 1))
        frac = stable[:, ya:yb].mean(axis=(1, 2))
        per_band[name] = frac
    best = max(per_band, key=lambda k: float((per_band[k] > 0.01).mean()))
    on = per_band[best] > 0.01
    runs, cur = [], 0
    for v in on:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    coverage = float(on.mean()) if len(on) else 0.0
    captions = {"detected": coverage > 0.15, "position": best if coverage > 0.15 else None,
                "coverage": round(coverage, 2),
                "avg_duration": round(float(np.mean(runs)) / SAMPLE_FPS, 2) if runs else 0.0,
                "confidence": "low"}

    # ---- audio / beats
    audio = {"has_audio": bool(meta.get("has_audio"))}
    if meta.get("has_audio"):
        try:
            b = analyze_beats(path, MAX_SECONDS)
            beats = np.array(b["beats"])
            audio.update({"tempo": b["tempo"], "beats": len(beats)})
            if len(beats) > 4 and cuts:
                interval = float(np.median(np.diff(beats)))
                tol = 0.08
                near = [min(abs(beats - c)) <= tol for c in cuts]
                ratio = float(np.mean(near))
                chance = min(1.0, 2 * tol / interval)
                audio.update({"beat_aligned_cuts": round(ratio, 2), "chance_level": round(chance, 2),
                              "beat_synced": bool(ratio > max(0.5, 1.8 * chance))})
        except EditorError:
            audio["note"] = "audio could not be analysed"

    # ---- colour
    mx, mn = frames.max(axis=3), frames.min(axis=3)
    sat = float(((mx - mn) / (mx + 1e-4)).mean())
    warmth = float((frames[..., 0] - frames[..., 2]).mean())
    bright = float(luma.mean())
    contrast = float(gray.std(axis=(1, 2)).mean())
    if sat < 0.06:
        preset = "bw"
    elif warmth > 0.06:
        preset = "warm"
    elif warmth < -0.06:
        preset = "cool"
    elif sat > 0.5:
        preset = "vibrant"
    elif contrast > 0.26 and bright < 0.4:
        preset = "moody"
    elif contrast > 0.22:
        preset = "cinematic"
    else:
        preset = "none"
    color = {"brightness": round(bright, 3), "contrast": round(contrast, 3), "saturation": round(sat, 3),
             "warmth": round(warmth, 3), "suggested_preset": preset}

    style = {
        "target_shot_length": round(float(np.median(shots)), 2),
        "transition_ratio": trans["soft_ratio"],
        "transition_type": {"dissolve": "dissolve", "fade_black": "fadeblack",
                            "fade_white": "fadewhite"}.get(trans["main_soft_type"] or "", None),
        "zoom_ratio": zoom["ratio"],
        "zoom_kind": "in" if zin >= zout else "out",
        "color_preset": preset,
        "captions": captions["detected"],
        "caption_position": captions["position"],
        "beat_sync": bool(audio.get("beat_synced")),
        "energy": "high" if shots_info["median"] < 1.2 else "medium" if shots_info["median"] < 2.5 else "calm",
    }
    return {"version": REF_VERSION, "duration": round(dur, 2), "width": meta["width"], "height": meta["height"],
            "aspect_ratio": _aspect_name(meta["width"], meta["height"]), "fps": meta.get("fps"),
            "shots": shots_info, "transitions": trans, "zoom": zoom, "motion": motion_info, "captions": captions,
            "audio": audio, "color": color, "style": style,
            "analysed_seconds": round(dur, 2), "note": "Style measurements only; the reference footage is never used."}
