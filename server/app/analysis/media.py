"""Video analysis (component B), Phase 1 subset.

* get_video_metadata  – ffprobe, rotation-aware display size
* validate_upload     – container/codec checks + a real decode test
* detect_scenes       – FFmpeg scene score on a downscaled stream
* analyze_audio       – loudness envelope (RMS dB per 0.5 s) + silent ranges

Everything is cached under cache/<sha256>/ so the same file is never analysed twice.
Analysis decodes a downscaled/mono version of the stream (an in-memory proxy);
the original is only used again for the final render.
"""
from __future__ import annotations

import json
import logging
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..errors import EditorError, unsupported_format
from ..proc import run
from ..storage.files import cache_dir

log = logging.getLogger(__name__)

ANALYSIS_VERSION = 1
ALLOWED_CONTAINERS = ("mov", "mp4", "m4a", "3gp", "matroska", "webm")
ENERGY_WINDOW_S = 0.5


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        v = stream.get(key) or "0/0"
        try:
            f = Fraction(v)
            if 0 < f < 1000:
                return float(f)
        except (ValueError, ZeroDivisionError):
            continue
    return 0.0


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    try:
        return int(stream.get("tags", {}).get("rotate", 0)) % 360
    except (TypeError, ValueError):
        return 0


def get_video_metadata(path: Path) -> dict:
    s = get_settings()
    cp = run([s.ffprobe_bin, "-v", "error", "-print_format", "json",
              "-show_format", "-show_streams", str(path)], timeout=60)
    if cp.returncode != 0:
        raise unsupported_format("Your video format isn't supported (the file could not be read).")
    try:
        info = json.loads(cp.stdout or b"{}")
    except json.JSONDecodeError as e:
        raise unsupported_format() from e
    fmt = info.get("format", {})
    streams = info.get("streams", [])
    v = next((st for st in streams if st.get("codec_type") == "video"
              and not st.get("disposition", {}).get("attached_pic")), None)
    a = next((st for st in streams if st.get("codec_type") == "audio"), None)

    def num(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    duration = num(fmt.get("duration")) or num((v or {}).get("duration")) or num((a or {}).get("duration"))
    meta = {
        "format_name": fmt.get("format_name", ""),
        "duration": round(duration, 3),
        "size_bytes": int(num(fmt.get("size"))),
        "has_video": v is not None,
        "has_audio": a is not None,
    }
    if v is not None:
        w, h = int(v.get("width") or 0), int(v.get("height") or 0)
        rot = _rotation(v)
        dw, dh = (h, w) if rot in (90, 270) else (w, h)
        meta.update({
            "vcodec": v.get("codec_name", ""),
            "coded_width": w, "coded_height": h,
            "rotation": rot,
            "width": dw, "height": dh,  # display size after rotation
            "fps": round(_fps(v), 3),
            "pix_fmt": v.get("pix_fmt", ""),
            "color_transfer": v.get("color_transfer", ""),
            "color_primaries": v.get("color_primaries", ""),
            # HDR (PQ / HLG) footage, e.g. iPhone video, must be tone-mapped for SDR H.264 output.
            "hdr": v.get("color_transfer", "") in ("smpte2084", "arib-std-b67"),
            "aspect": round(dw / dh, 4) if dw and dh else 0,
        })
    if a is not None:
        meta.update({
            "acodec": a.get("codec_name", ""),
            "sample_rate": int(num(a.get("sample_rate"))),
            "channels": int(num(a.get("channels"))),
        })
    return meta


def validate_video(path: Path, meta: dict) -> None:
    """Raise a clear EditorError unless this is a usable video."""
    s = get_settings()
    names = set((meta.get("format_name") or "").split(","))
    if not names.intersection(ALLOWED_CONTAINERS):
        raise unsupported_format("Your video format isn't supported. Use MP4, MOV or WEBM.")
    if not meta.get("has_video") or not meta.get("width") or not meta.get("height"):
        raise unsupported_format("This file has no video track.")
    if meta["duration"] < 0.3:
        raise EditorError("TOO_SHORT", "This video is too short to edit (under 0.3 s).")
    if meta["duration"] > s.max_source_duration_s:
        raise EditorError("TOO_LONG", f"This video is too long (limit {int(s.max_source_duration_s // 60)} minutes).")
    if max(meta["width"], meta["height"]) > 8192:
        raise unsupported_format("This video's resolution is too large to process.")
    # Real decode test: first video frame (and audio, if any) must decode.
    cp = run([s.ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
              "-frames:v", "1", "-f", "null", "-"], timeout=60)
    if cp.returncode != 0:
        raise unsupported_format("Your video format isn't supported (the video could not be decoded).")
    if meta.get("has_audio"):
        cp = run([s.ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
                  "-t", "1", "-f", "null", "-"], timeout=60)
        if cp.returncode != 0:
            log.warning("audio not decodable in %s; treating as silent", path)
            meta["has_audio"] = False
            meta["audio_note"] = "audio track could not be decoded and will be ignored"


def detect_scenes(path: Path, duration: float, threshold: float = 0.3) -> list[dict]:
    """Return scene cuts [{t, score}] using FFmpeg's scene-change score on a 160px stream."""
    s = get_settings()
    vf = f"scale=160:-2,select='gt(scene\\,{threshold})',metadata=print:file=-"
    cp = run([s.ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-sn",
              "-vf", vf, "-f", "null", "-"], timeout=s.analysis_timeout_s)
    if cp.returncode != 0:
        raise EditorError("ANALYSIS_FAILED", "Scene detection failed for this video.",
                          detail=(cp.stderr or b"")[-500:].decode(errors="replace"))
    cuts: list[dict] = []
    cur_t = None
    for line in (cp.stdout or b"").decode(errors="replace").splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            cur_t = float(m.group(1))
            continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m and cur_t is not None:
            if 0.2 < cur_t < duration - 0.2:
                cuts.append({"t": round(cur_t, 3), "score": round(float(m.group(1)), 3)})
            cur_t = None
    # Merge cuts closer than 0.4 s (flashes), keep the stronger one.
    merged: list[dict] = []
    for c in cuts:
        if merged and c["t"] - merged[-1]["t"] < 0.4:
            if c["score"] > merged[-1]["score"]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged[:1000]


def analyze_audio(path: Path, has_audio: bool, duration: float) -> dict:
    """Loudness envelope in dBFS per ENERGY_WINDOW_S and silent ranges."""
    if not has_audio:
        return {"has_audio": False, "window_s": ENERGY_WINDOW_S, "energy_db": [], "silent_ranges": []}
    s = get_settings()
    sr = 8000
    cp = run([s.ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
              "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], timeout=s.analysis_timeout_s)
    if cp.returncode != 0:
        raise EditorError("ANALYSIS_FAILED", "Audio analysis failed for this video.")
    pcm = np.frombuffer(cp.stdout or b"", dtype=np.int16).astype(np.float32) / 32768.0
    win = int(sr * ENERGY_WINDOW_S)
    n = len(pcm) // win
    energy: list[float] = []
    if n:
        frames = pcm[: n * win].reshape(n, win)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        energy = [round(max(-90.0, 20 * math.log10(float(r))), 1) for r in rms]
    silent: list[list[float]] = []
    start = None
    for i, e in enumerate(energy + [0.0]):
        if e < -45 and start is None:
            start = i
        elif e >= -45 and start is not None:
            if i - start >= 2:  # >= 1 s
                silent.append([round(start * ENERGY_WINDOW_S, 2), round(i * ENERGY_WINDOW_S, 2)])
            start = None
    return {"has_audio": True, "window_s": ENERGY_WINDOW_S, "energy_db": energy, "silent_ranges": silent}


def analyze_video(path: Path, sha256: str, meta: dict) -> dict:
    """Full Phase 1 analysis, cached by content hash."""
    cfile = cache_dir(sha256) / f"analysis_v{ANALYSIS_VERSION}.json"
    if cfile.exists():
        try:
            return json.loads(cfile.read_text())
        except json.JSONDecodeError:
            cfile.unlink(missing_ok=True)
    scenes = detect_scenes(path, meta["duration"])
    audio = analyze_audio(path, meta.get("has_audio", False), meta["duration"])
    result = {"version": ANALYSIS_VERSION, "duration": meta["duration"], "scenes": scenes, "audio": audio}
    tmp = cfile.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(cfile)
    return result


# ----------------------------------------------------------------------------- audio (music / sound effects)
AUDIO_CONTAINERS = ("mp3", "mov", "mp4", "m4a", "wav", "ogg", "matroska", "webm", "aac", "flac")


def validate_audio(path: Path, meta: dict) -> None:
    s = get_settings()
    names = set((meta.get("format_name") or "").split(","))
    if not names.intersection(AUDIO_CONTAINERS):
        raise unsupported_format("That audio format isn't supported. Use MP3, M4A, WAV, AAC, OGG or FLAC.")
    if not meta.get("has_audio"):
        raise unsupported_format("This file has no sound.")
    if meta["duration"] < 1.0:
        raise EditorError("TOO_SHORT", "This audio is too short (under 1 s).")
    if meta["duration"] > 30 * 60:
        raise EditorError("TOO_LONG", "This audio is too long (limit 30 minutes).")
    cp = run([s.ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", "2",
              "-f", "null", "-"], timeout=60)
    if cp.returncode != 0:
        raise unsupported_format("That audio format isn't supported (it could not be decoded).")


def analyze_music(path: Path, sha256: str, meta: dict) -> dict:
    """Tempo, beats, downbeats and a 1 s energy envelope; cached."""
    from .beats import analyze_beats

    cfile = cache_dir(sha256) / "music_v2.json"
    if cfile.exists():
        try:
            return json.loads(cfile.read_text())
        except json.JSONDecodeError:
            cfile.unlink(missing_ok=True)
    res = analyze_beats(path)
    tmp = cfile.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(cfile)
    return res


# ----------------------------------------------------------------------------- frames / thumbnails
def extract_frames(path: Path, sha256: str, times: list[float], width: int = 256) -> list[Path]:
    """JPEG frames at the given source times (cached)."""
    s = get_settings()
    out: list[Path] = []
    d = cache_dir(sha256) / "frames"
    d.mkdir(exist_ok=True)
    for t in times[:24]:
        f = d / f"{int(round(t * 1000))}_{width}.jpg"
        if not f.exists():
            cp = run([s.ffmpeg_bin, "-nostdin", "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
                      "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(f)], timeout=60)
            if cp.returncode != 0 or not f.exists():
                raise EditorError("FRAME_FAILED", f"Could not extract a frame at {t:.2f}s.")
        out.append(f)
    return out


def thumbnail(path: Path, sha256: str, duration: float) -> Path:
    return extract_frames(path, sha256, [min(1.0, duration * 0.1)], width=240)[0]
