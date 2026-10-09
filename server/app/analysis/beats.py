"""Beat tracking in plain numpy (no heavy dependencies).

1. Onset strength: spectral flux of a log-magnitude STFT.
2. Tempo: autocorrelation of the onset envelope, weighted towards ~120 BPM.
3. Beats: dynamic-programming beat tracker (Ellis 2007).
4. Downbeats: the beat phase (of 4) with the strongest onsets.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..errors import EditorError
from ..proc import run

SR = 22050
N_FFT = 2048
HOP = 512
FPS = SR / HOP  # onset frames per second (~43)


def decode_mono(path: Path, max_seconds: float = 600.0, sr: int = SR) -> np.ndarray:
    s = get_settings()
    cp = run([s.ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-t", str(max_seconds),
              "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], timeout=s.analysis_timeout_s)
    if cp.returncode != 0:
        raise EditorError("ANALYSIS_FAILED", "The audio could not be decoded for beat detection.")
    return np.frombuffer(cp.stdout or b"", dtype=np.int16).astype(np.float32) / 32768.0


# Flux at frame t is measured on a window starting at t*HOP; the attack it reacts to sits
# about half a window plus a hop later (measured: ~70 ms). Beat times are shifted by this.
ONSET_DELAY = (N_FFT / 2 + HOP) / SR
LOW_BAND_HZ = 180


def _norm(flux: np.ndarray) -> np.ndarray:
    w = max(1, int(FPS * 0.5))  # remove slow trends (local mean over ~0.5 s) and normalise
    env = np.maximum(0.0, flux - np.convolve(flux, np.ones(w) / w, mode="same"))
    sd = env.std()
    return (env / sd) if sd > 0 else env


def onset_bands(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(full-band, low-band) onset envelopes. Kick drums live in the low band."""
    if len(y) < N_FFT:
        z = np.zeros(1, dtype=np.float32)
        return z, z
    n = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = y[idx] * np.hanning(N_FFT).astype(np.float32)[None, :]
    logm = np.log1p(100.0 * np.abs(np.fft.rfft(frames, axis=1)))
    d = np.maximum(0.0, np.diff(logm, axis=0))
    k = int(LOW_BAND_HZ * N_FFT / SR) + 1
    full = np.concatenate([[0.0], d.sum(axis=1)])
    low = np.concatenate([[0.0], d[:, :k].sum(axis=1)])
    return _norm(full), _norm(low)


def onset_envelope(y: np.ndarray) -> np.ndarray:
    full, low = onset_bands(y)
    return 0.5 * full + 0.5 * low


def estimate_tempo(env: np.ndarray, lo_bpm: float = 60, hi_bpm: float = 200) -> tuple[float, float]:
    """Return (bpm, confidence 0..1)."""
    if len(env) < FPS * 4 or env.max() <= 0:
        return 0.0, 0.0
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lags = np.arange(len(ac))
    min_lag, max_lag = int(FPS * 60 / hi_bpm), int(FPS * 60 / lo_bpm)
    cand = lags[min_lag:max_lag + 1]
    bpm = 60 * FPS / np.maximum(cand, 1)
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.9) ** 2)
    score = ac[min_lag:max_lag + 1] * prior
    best = int(np.argmax(score))
    lag = cand[best]
    # parabolic refinement
    if 0 < best < len(score) - 1:
        a, b, c = score[best - 1], score[best], score[best + 1]
        denom = a - 2 * b + c
        if denom != 0:
            lag = lag + 0.5 * (a - c) / denom
    conf = float(max(0.0, min(1.0, ac[int(round(lag))] if int(round(lag)) < len(ac) else 0.0)))
    return float(60 * FPS / lag), conf


def track_beats(env: np.ndarray, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Dynamic-programming beat tracking. Returns beat times in seconds."""
    if bpm <= 0 or len(env) < 4:
        return np.zeros(0)
    period = 60 * FPS / bpm
    n = len(env)
    score = env.astype(np.float64).copy()
    back = -np.ones(n, dtype=np.int64)
    lo, hi = int(round(period / 2)), int(round(2 * period))
    offsets = np.arange(lo, hi + 1)
    penalty = -tightness * np.log(offsets / period) ** 2
    for t in range(lo, n):
        prev = t - offsets
        ok = prev >= 0
        if not ok.any():
            continue
        cand = score[prev[ok]] + penalty[ok]
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[t] = env[t] + cand[j]
            back[t] = prev[ok][j]
    # start from the best score in the last period
    tail = max(0, n - int(period) - 1)
    t = tail + int(np.argmax(score[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    beats = np.array(beats[::-1], dtype=np.float64)
    # drop weak beats at the very start/end (silence)
    strong = env[beats.astype(int)] > 0.1 * env.max()
    if strong.any():
        first, last = np.argmax(strong), len(strong) - 1 - np.argmax(strong[::-1])
        beats = beats[first:last + 1]
    return beats / FPS + ONSET_DELAY


def _fix_offbeat(beats: np.ndarray, low: np.ndarray, period_s: float) -> np.ndarray:
    """If the low band (kick) is clearly stronger half a beat later, we locked onto the off-beat."""
    if len(beats) < 4:
        return beats

    def strength(times: np.ndarray) -> float:
        fr = np.clip(((times - ONSET_DELAY) * FPS).round().astype(int), 0, len(low) - 1)
        return float(low[fr].mean())

    shifted = beats + period_s / 2
    shifted = shifted[shifted < (len(low) / FPS + ONSET_DELAY)]
    if len(shifted) and strength(shifted) > 1.3 * strength(beats):
        return shifted
    return beats


def analyze_beats(path: Path, max_seconds: float = 600.0) -> dict:
    y = decode_mono(path, max_seconds)
    duration = len(y) / SR
    full, low = onset_bands(y)
    env = 0.5 * full + 0.5 * low
    bpm, conf = estimate_tempo(env)
    beats = track_beats(env, bpm)
    if bpm > 0:
        beats = _fix_offbeat(beats, low, 60 / bpm)
    downbeats: list[float] = []
    if len(beats) >= 8:
        fr = np.clip(((beats - ONSET_DELAY) * FPS).round().astype(int), 0, len(env) - 1)
        strengths = [env[fr[k::4]].mean() for k in range(4)]
        phase = int(np.argmax(strengths))
        downbeats = [round(float(b), 3) for b in beats[phase::4]]
    # 1-second energy envelope (for choosing a lively start point)
    sec = max(1, int(math.ceil(duration)))
    rms = [float(np.sqrt(np.mean(y[i * SR:(i + 1) * SR] ** 2) + 1e-12)) for i in range(sec)]
    energy_db = [round(max(-90.0, 20 * math.log10(r)), 1) for r in rms]
    return {"duration": round(duration, 3), "tempo": round(bpm, 2), "confidence": round(conf, 3),
            "beats": [round(float(b), 3) for b in beats], "downbeats": downbeats, "energy_db": energy_db}
