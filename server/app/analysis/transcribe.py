"""Speech-to-text for subtitles, using faster-whisper (open source, runs on CPU).

The model (default "base", ~140 MB) is downloaded once on first use into
DATA_DIR/models. Results are cached by content hash. If the model can't be
loaded (no internet on first use, package missing) subtitles are reported as
unavailable — never faked.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from ..config import get_settings
from ..errors import EditorError
from ..storage.files import cache_dir

log = logging.getLogger(__name__)
_model = None
_model_error: str | None = None
_lock = threading.Lock()

# Tests (or another engine) can replace this: fn(path) -> {"language": str, "segments": [...]}
_override = None


def set_transcriber(fn) -> None:
    global _override
    _override = fn


def model_name() -> str:
    return os.environ.get("WHISPER_MODEL", "base")


def _load():
    global _model, _model_error
    with _lock:
        if _model is not None or _model_error is not None:
            return _model
        if os.environ.get("TRANSCRIBE", "1") == "0":
            _model_error = "speech recognition is disabled on this server"
            return None
        try:
            from faster_whisper import WhisperModel
            root = get_settings().data_dir / "models"
            root.mkdir(parents=True, exist_ok=True)
            _model = WhisperModel(model_name(), device="cpu", compute_type="int8", download_root=str(root))
        except Exception as e:  # noqa: BLE001 - any failure means "unavailable", reported honestly
            _model_error = f"the speech model could not be loaded ({type(e).__name__})"
            log.warning("whisper unavailable: %s", e)
        return _model


def availability() -> tuple[bool, str]:
    if _override is not None:
        return True, ""
    if os.environ.get("TRANSCRIBE", "1") == "0":
        return False, "speech recognition is disabled"
    if _model is not None:
        return True, ""
    if _model_error:
        return False, _model_error
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False, "the speech recognition package is not installed"
    return True, "model loads on first use"


def transcribe(path: Path, sha256: str, language: str | None = None) -> dict:
    """Return {"language", "segments": [{start, end, text, words: [{start, end, word}]}]} in source seconds."""
    cfile = cache_dir(sha256) / f"transcript_{model_name()}_{language or 'auto'}.json"
    if cfile.exists():
        try:
            return json.loads(cfile.read_text())
        except json.JSONDecodeError:
            cfile.unlink(missing_ok=True)
    if _override is not None:
        result = _override(path)
    else:
        model = _load()
        if model is None:
            raise EditorError("SUBTITLES_UNAVAILABLE",
                              f"Subtitles from speech aren't available on this server: {_model_error}.")
        try:
            segs, info = model.transcribe(str(path), language=language, word_timestamps=True, vad_filter=True,
                                          beam_size=1)
            out = []
            for sg in segs:
                words = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()}
                         for w in (sg.words or []) if w.word.strip()]
                out.append({"start": round(sg.start, 3), "end": round(sg.end, 3), "text": sg.text.strip(),
                            "words": words})
            result = {"language": info.language, "segments": out}
        except Exception as e:  # noqa: BLE001
            raise EditorError("SUBTITLES_FAILED", "Speech recognition failed for this video.", retryable=True,
                              detail=str(e)) from e
    cfile.write_text(json.dumps(result))
    return result


def caption_chunks(transcript: dict, clips: list, style: str = "subtitle", vertical: bool = True) -> list[dict]:
    """Map words (source time) onto the timeline and group them into readable caption chunks.

    clips: main-track Clip objects with asset_id, src_in, src_out, speed, timeline_start, duration.
    Returns [{"text", "start", "end"}] in timeline seconds.
    """
    words = [w for sg in transcript.get("segments", []) for w in (sg.get("words") or
             [{"start": sg["start"], "end": sg["end"], "word": sg["text"]}])]
    max_chars = 14 if style == "bold" else (32 if vertical else 42)
    max_words = 3 if style == "bold" else 12
    placed: list[dict] = []
    for c in clips:
        for w in words:
            mid = (w["start"] + w["end"]) / 2
            if c.src_in <= mid < c.src_out:
                s = c.timeline_start + (max(w["start"], c.src_in) - c.src_in) / c.speed
                e = c.timeline_start + (min(w["end"], c.src_out) - c.src_in) / c.speed
                placed.append({"start": s, "end": max(e, s + 0.08), "word": w["word"]})
    placed.sort(key=lambda x: x["start"])
    chunks: list[dict] = []
    cur: list[dict] = []

    def flush():
        if cur:
            text = " ".join(x["word"] for x in cur).replace(" ,", ",").replace(" .", ".").strip()
            chunks.append({"text": text, "start": cur[0]["start"], "end": max(cur[-1]["end"], cur[0]["start"] + 0.6)})
            cur.clear()

    for w in placed:
        if cur:
            text_len = len(" ".join(x["word"] for x in cur)) + 1 + len(w["word"])
            gap = w["start"] - cur[-1]["end"]
            if (text_len > max_chars or len(cur) >= max_words or gap > 0.6 or w["start"] - cur[0]["start"] > 3.0
                    or cur[-1]["word"].endswith((".", "?", "!"))):
                flush()
        cur.append(w)
    flush()
    for a, b in zip(chunks, chunks[1:]):  # no overlaps
        a["end"] = min(a["end"], b["start"])
    return [{**c, "start": round(c["start"], 3), "end": round(c["end"], 3)} for c in chunks if c["end"] - c["start"] >= 0.15]
