"""Beats, reference-style analysis, transcription mapping, audio validation, frames."""
import subprocess

import numpy as np
import pytest

from app.analysis import media as M
from app.analysis import transcribe as T
from app.analysis.beats import analyze_beats
from app.analysis.reference import analyze_reference
from app.engine.compiler import compile_timeline
from app.engine.timeline import TimelineEditor
from app.errors import EditorError
from app.render.runner import render
from tests.conftest import ff, make_asset


@pytest.fixture(scope="session")
def audio(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    out = {}
    for bpm in (100, 128):
        p = d / f"kick{bpm}.m4a"
        ff("-f", "lavfi", "-i", f"aevalsrc='0.9*sin(2*PI*80*t)*exp(-30*mod(t,60/{bpm}))"
                               f"+0.4*(random(0)*2-1)*exp(-200*mod(t,60/{bpm}))':d=16:s=44100", str(p))
        out[bpm] = p
    out["mp3"] = d / "song.mp3"
    ff("-f", "lavfi", "-i", "sine=f=330:d=5", str(out["mp3"]))
    out["tiny"] = d / "tiny.wav"
    ff("-f", "lavfi", "-i", "sine=d=0.3", str(out["tiny"]))
    return out


@pytest.mark.parametrize("bpm", [100, 128])
def test_beats(audio, settings, bpm):
    r = analyze_beats(audio[bpm])
    assert r["tempo"] == pytest.approx(bpm, rel=0.02)
    b = np.array(r["beats"])
    period = 60 / bpm
    ph = np.mod(b, period)
    ph = np.minimum(ph, period - ph)
    assert ph.mean() < 0.02  # beats land on the kicks (within 20 ms)
    assert len(b) >= 16 / period * 0.85
    assert r["downbeats"] and len(r["energy_db"]) in (16, 17)  # AAC padding may add a fraction of a second


def test_audio_validation(audio, settings, media):
    m = M.get_video_metadata(audio["mp3"])
    M.validate_audio(audio["mp3"], m)
    with pytest.raises(EditorError, match="too short"):
        M.validate_audio(audio["tiny"], M.get_video_metadata(audio["tiny"]))
    with pytest.raises(EditorError, match="no sound"):
        M.validate_audio(media["silent"], M.get_video_metadata(media["silent"]))


def test_frames_and_thumbnail(media, settings):
    a = make_asset(media["land"])
    fs = M.extract_frames(media["land"], a["sha256"], [0.5, 5.0])
    assert len(fs) == 2 and all(f.stat().st_size > 500 for f in fs)
    assert M.thumbnail(media["land"], a["sha256"], 12).exists()


@pytest.fixture(scope="session")
def reference_video(tmp_path_factory, audio):
    """A reference with known style: 1 s shots on a 120-ish grid, zooms on 3 of 6 shots,
    one dissolve, bold captions at the bottom, a beat track."""
    d = tmp_path_factory.mktemp("ref")
    gens = ["mandelbrot=s=640x360", "cellauto=s=640x360:rule=30:start_full=1",
            "mandelbrot=s=640x360:start_x=-0.743:start_y=0.131:start_scale=0.2",
            "life=s=640x360:ratio=0.3:mold=5", "nullsrc=s=640x360,geq=lum='128+110*sin(X/9)*sin(Y/13)':cb=90:cr=170",
            "cellauto=s=640x360:rule=110:start_full=1"]
    stills = []
    for k, g in enumerate(gens):
        p = d / f"s{k}.png"
        ff("-f", "lavfi", "-i", g, "-frames:v", "1", "-vf", "format=rgb24,gblur=sigma=1", str(p))
        stills.append(p)
    src = d / "src.mp4"
    inputs = []
    for p in stills:
        inputs += ["-loop", "1", "-t", "3", "-i", str(p)]
    ff(*inputs, "-filter_complex", "".join(f"[{i}:v]fps=30,format=yuv420p[v{i}];" for i in range(6)) +
       "".join(f"[v{i}]" for i in range(6)) + "concat=n=6:v=1:a=0", "-c:v", "libx264", "-preset", "ultrafast",
       str(src))
    from app import config
    config.reset_settings(data_dir=d / "data")
    A = {"a": {"role": "clip", "path": str(src), "meta": M.get_video_metadata(src), "sha256": "a"},
         "m": {"role": "music", "path": str(audio[128]), "meta": M.get_video_metadata(audio[128]), "sha256": "m"}}
    e = TimelineEditor(None, A)
    e.create_timeline(540, 960)
    ids = [e.select_clip("a", k * 3 + 0.5, k * 3 + 1.5 + (0.5 if k == 3 else 0))["clip_id"] for k in range(6)]
    for k in (0, 2, 4):
        e.add_zoom(ids[k], "in", 0.3)
    e.add_transition(ids[4], "dissolve", 0.5)
    e.add_caption("A CAPTION ON SCREEN", 0, e.timeline.duration - 0.05, "bold", "bottom")
    e.add_music("m", volume=1.0, duck=False, fade_in=0, fade_out=0)
    out = d / "ref.mp4"
    spec = compile_timeline(e.timeline, A, out, segment_dir=d / "seg", x264_preset="ultrafast", sfx_paths={})
    render(spec)
    return out


def test_reference_profile(reference_video, settings):
    meta = M.get_video_metadata(reference_video)
    p = analyze_reference(reference_video, "e" * 64, meta)
    assert p["aspect_ratio"] == "9:16"
    assert p["shots"]["count"] == 6
    assert p["shots"]["median"] == pytest.approx(1.0, abs=0.15)
    assert p["zoom"]["zoom_in_shots"] >= 2 and p["zoom"]["zoom_out_shots"] == 0
    assert p["transitions"]["dissolve"] >= 1
    assert p["captions"]["detected"] and p["captions"]["position"] == "bottom"
    assert p["audio"]["tempo"] == pytest.approx(128, rel=0.03)
    assert p["style"]["energy"] == "high" and p["style"]["zoom_kind"] == "in"
    assert "never used" in p["note"]
    # cached
    assert analyze_reference(reference_video, "e" * 64, meta) == p


def test_transcript_mapping_to_timeline():
    from app.engine.timeline import Clip
    transcript = {"segments": [{"start": 0, "end": 6, "text": "x", "words": [
        {"start": 1.0, "end": 1.4, "word": "Hello"}, {"start": 1.5, "end": 1.9, "word": "there."},
        {"start": 2.0, "end": 2.4, "word": "This"}, {"start": 2.5, "end": 2.9, "word": "is"},
        {"start": 4.1, "end": 4.5, "word": "later"}, {"start": 8.0, "end": 8.5, "word": "cut"}]}]}
    clips = [Clip(clip_id="c1", asset_id="a", src_in=0.5, src_out=3.0, timeline_start=0.0),
             Clip(clip_id="c2", asset_id="a", src_in=4.0, src_out=5.0, timeline_start=2.5, speed=1.0)]
    chunks = T.caption_chunks(transcript, clips, "subtitle")
    texts = [c["text"] for c in chunks]
    assert texts == ["Hello there.", "This is later"]  # sentence break; "cut" was cut out of the edit
    assert chunks[0]["start"] == pytest.approx(0.5)     # 1.0 in the source → 0.5 on the timeline
    assert chunks[1]["end"] == pytest.approx(3.0)       # "later" 4.1-4.5 in clip 2 → 2.6-3.0
    bold = T.caption_chunks(transcript, clips, "bold")
    assert all(len(c["text"].split()) <= 3 for c in bold)


def test_transcriber_override_and_cache(media, settings):
    calls = []

    def fake(path):
        calls.append(path)
        return {"language": "en", "segments": [{"start": 0, "end": 1, "text": "hi",
                                                "words": [{"start": 0.1, "end": 0.5, "word": "hi"}]}]}
    T.set_transcriber(fake)
    try:
        a = make_asset(media["land"])
        r1 = T.transcribe(media["land"], a["sha256"])
        r2 = T.transcribe(media["land"], a["sha256"])
        assert r1 == r2 and len(calls) == 1 and T.availability()[0]
    finally:
        T.set_transcriber(None)
