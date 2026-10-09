"""Engine v2 operations (effects, transitions, text, audio, overlays) and their rendering."""
import json
import subprocess

import pytest

from app.analysis.sfx import RECIPES, all_sfx_paths
from app.engine.captions import build_ass, escape_ass
from app.engine.compiler import compile_timeline, plan_pieces
from app.engine.timeline import TextItem, TimelineEditor, sample_zoom
from app.errors import EditorError
from app.render.runner import render, verify_output
from tests.conftest import make_asset

A = {"a_1": {"asset_id": "a_1", "role": "clip", "meta": {"duration": 30.0, "has_video": True}},
     "m_1": {"asset_id": "m_1", "role": "music", "meta": {"duration": 60.0, "has_audio": True}},
     "r_1": {"asset_id": "r_1", "role": "reference", "meta": {"duration": 10.0, "has_video": True}}}


def ed(n=3, length=4.0):
    e = TimelineEditor(None, A)
    e.create_timeline(720, 1280)
    ids = [e.select_clip("a_1", i * 5, i * 5 + length)["clip_id"] for i in range(n)]
    return e, ids


def test_transitions_overlap_and_limits():
    e, ids = ed()
    e.add_transition(ids[1], "dissolve", 0.5)
    e.add_transition(ids[2], "wipeleft", 1.0)
    assert e.timeline.duration == pytest.approx(12 - 1.5)
    assert e.timeline.video_track().clips[2].timeline_start == pytest.approx(6.5)
    with pytest.raises(EditorError, match="first clip"):
        e.add_transition(ids[0], "fade", 0.5)
    with pytest.raises(EditorError, match="unknown transition"):
        e.add_transition(ids[1], "explode", 0.5)
    with pytest.raises(EditorError, match="too short"):
        e.add_transition(ids[1], "fade", 3.0)  # 3.0 in + 1.0 out > clip 2 length (4 s)
    pieces = plan_pieces(e.timeline)
    assert [p.kind for p in pieces] == ["body", "transition", "body", "transition", "body"]
    assert sum(p.t1 - p.t0 for p in pieces) == pytest.approx(e.timeline.duration)


def test_zoom_speed_ramp_and_split_keep_curves():
    e, ids = ed(1, 8.0)
    e.add_zoom(ids[0], "in", 0.2)
    before_end = sample_zoom(e.timeline.video_track().clips[0].keyframes, 1, "inout").scale
    r = e.speed_ramp(ids[0], "slow_middle")
    clips = e.timeline.video_track().clips
    assert [c.speed for c in clips] == [1.0, 0.5, 1.0]
    assert sum(c.duration for c in clips) == pytest.approx(8.0, abs=0.01)  # keep_duration
    assert clips[-1].keyframes[-1].scale == pytest.approx(before_end)  # zoom continues to the same end
    assert clips[0].keyframes[0].scale == pytest.approx(1.0)
    with pytest.raises(EditorError):
        e.speed_ramp(r["clips"][1], "fast_middle")  # already at 0.5x


def test_effect_validation():
    e, ids = ed()
    with pytest.raises(EditorError):
        e.add_zoom(ids[0], "spin")
    with pytest.raises(EditorError):
        e.apply_color_adjustment("all", "neon")
    with pytest.raises(EditorError):
        e.apply_color_adjustment("all", "warm", contrast=5)
    with pytest.raises(EditorError):
        e.apply_effect(ids[0], "explosion")
    with pytest.raises(EditorError, match="outside"):
        e.crop_video(ids[0], 0.5, 0.5, 0.8, 0.8)
    with pytest.raises(EditorError):
        e.rotate_video(ids[0], 45)
    with pytest.raises(EditorError):
        e.change_speed(ids[0], 10)
    e.apply_color_adjustment("all", "warm")
    assert all(c.color.preset == "warm" for c in e.timeline.video_track().clips)


def test_text_and_audio_validation():
    e, ids = ed()
    with pytest.raises(EditorError, match="after the end"):
        e.add_caption("hi", 20, 21)
    with pytest.raises(EditorError):
        e.add_caption("   ", 1, 2)
    with pytest.raises(EditorError):
        e.add_caption("hi", 2, 1)
    item = e.add_caption("a" * 500, 1, 2)
    assert len(next(t for t in e.timeline.texts if t.item_id == item["item_id"]).text) == 200
    with pytest.raises(EditorError, match="only uploaded music"):
        e.add_music("a_1")
    with pytest.raises(EditorError):
        e.add_sound_effect(1.0, sfx="laser")
    with pytest.raises(EditorError):
        e.add_sound_effect(1.0)
    m = e.add_music("m_1", volume=0.5)
    s = e.add_sound_effect(2.0, sfx="whoosh")
    assert e.delete_clip(s["item_id"])["deleted"] == s["item_id"]
    assert e.set_volume(m["item_id"], 0.2)["volume"] == 0.2
    with pytest.raises(EditorError, match="reference"):
        e.select_clip("r_1", 0, 2)


def test_snap_to_beats():
    e, ids = ed(3, 3.1)
    beats = [i * 0.5 for i in range(40)]
    res = e.snap_to_beats(beats)
    assert res["snapped"] == 3
    clips = e.timeline.video_track().clips
    for c in clips:
        end = c.timeline_start + c.duration
        assert min(abs(end - b) for b in beats) < 1e-6


def test_ass_escaping_and_styles():
    assert escape_ass("a{\\b1}b\nc") == "a(/b1)b\\Nc"
    ass = build_ass([TextItem(item_id="x", text="hi {x}", start=1, end=2, style="bold", position="top")], 1080, 1920)
    assert "Style: bold_top" in ass and "HI (X)" in ass and "0:00:01.00" in ass


def test_sfx_library(settings):
    paths = all_sfx_paths()
    assert set(paths) == set(RECIPES)
    assert all(p.stat().st_size > 1000 for p in paths.values())


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, check=True).stdout
    return json.loads(out)["streams"]


def test_render_everything(media, settings, tmp_path):
    clip = make_asset(media["land"], asset_id="a_1")
    music = make_asset(media["land"], role="music", asset_id="m_1")  # its tone works as "music"
    assets = {"a_1": clip, "m_1": music}
    e = TimelineEditor(None, assets)
    e.create_timeline(480, 854, fit="blur")
    ids = [e.select_clip("a_1", a, b)["clip_id"] for a, b in [(0, 2.5), (3, 5.5), (8, 11)]]
    e.add_zoom(ids[0], "in", 0.2)
    e.apply_color_adjustment("all", "cinematic", warmth=0.3)
    e.apply_effect(ids[1], "vignette")
    e.crop_video(ids[2], 0.1, 0.1, 0.8, 0.8)
    e.rotate_video(ids[1], 180)
    e.add_transition(ids[1], "dissolve", 0.5)
    e.add_transition(ids[2], "slideleft", 0.4)
    e.speed_ramp(ids[2], "slow_end")
    e.add_transition(None, "fadeblack", 0.4, "start")
    e.add_transition(None, "fadewhite", 0.4, "end")
    e.add_caption("Hello", 0.2, 1.5, "bold", "center")
    e.add_text("Title", 2, 3)
    e.add_video_track()
    e.select_clip("a_1", 4, 6, track_id="V2", start=1.0)
    e.add_music("m_1", volume=0.4, duck=True)
    e.add_sound_effect(2.2, sfx="whoosh")
    e.validate_timeline()
    out = tmp_path / "all.mp4"
    spec = compile_timeline(e.timeline, assets, out, segment_dir=tmp_path / "seg", x264_preset="ultrafast",
                            sfx_paths=all_sfx_paths())
    stats = render(spec, parallel=2)
    assert stats["segments"] == len(plan_pieces(e.timeline))
    v = verify_output(out, width=480, height=854, duration=spec.duration, fps=30)
    assert spec.duration == pytest.approx(e.timeline.duration, abs=1 / 30)
    streams = _probe(out)
    vs = next(s for s in streams if s["codec_type"] == "video")
    au = next(s for s in streams if s["codec_type"] == "audio")
    assert int(vs["nb_read_frames"]) == round(spec.duration * 30)
    assert abs(float(vs["duration"]) - float(au["duration"])) < 0.05
    assert v["acodec"] == "aac"
    # Re-render after changing only the caption: only pieces under it change.
    e.timeline.texts[0].text = "Changed"
    spec2 = compile_timeline(e.timeline, assets, out, segment_dir=tmp_path / "seg", x264_preset="ultrafast",
                             sfx_paths=all_sfx_paths())
    assert render(spec2)["segments_reused"] >= len(spec2.segments) - 2
