import pytest

from app.analysis import media as M
from app.engine.compiler import compile_timeline
from app.engine.timeline import TimelineEditor
from app.errors import EditorError
from app.render.runner import render, verify_output
from tests.conftest import make_asset


def test_metadata_rotation_and_audio(media, settings):
    rot = M.get_video_metadata(media["rotated"])
    assert rot["rotation"] in (90, 270)
    assert (rot["width"], rot["height"]) == (360, 640)  # display size is portrait
    silent = M.get_video_metadata(media["silent"])
    assert silent["has_audio"] is False and silent["fps"] == 25
    webm = M.get_video_metadata(media["webm"])
    M.validate_video(media["webm"], webm)


def test_fake_video_rejected(media, settings):
    with pytest.raises(EditorError) as ei:
        meta = M.get_video_metadata(media["fake"])
        M.validate_video(media["fake"], meta)
    assert ei.value.code == "UNSUPPORTED_FORMAT"


def test_analysis_finds_scenes_and_silence_and_caches(media, settings):
    a = make_asset(media["land"])
    res = M.analyze_video(media["land"], a["sha256"], a["meta"])
    cuts = [c["t"] for c in res["scenes"]]
    assert any(abs(t - 4) < 0.1 for t in cuts) and any(abs(t - 8) < 0.1 for t in cuts)
    assert any(s0 <= 5.5 and s1 >= 6.5 for s0, s1 in res["audio"]["silent_ranges"])
    # Second call is served from cache (file exists, identical result).
    assert M.analyze_video(media["land"], a["sha256"], a["meta"]) == res


@pytest.mark.parametrize("key,fit,size", [
    ("land", "blur", (720, 1280)),
    ("rotated", "cover", (1280, 720)),
    ("silent", "contain", (720, 720)),
    ("webm", "cover", (720, 1280)),
])
def test_render_and_verify(media, settings, tmp_path, key, fit, size):
    a = make_asset(media[key])
    assets = {"a_1": a}
    e = TimelineEditor(None, assets)
    e.create_timeline(*size, fit=fit)
    d = a["meta"]["duration"]
    e.select_clip("a_1", 0.5, min(d, 2.0))
    e.select_clip("a_1", max(0.0, d - 1.5), d)
    out = tmp_path / "out.mp4"
    spec = compile_timeline(e.timeline, assets, out, segment_dir=tmp_path / "seg", x264_preset="ultrafast")
    progress = []
    stats = render(spec, progress.append)
    assert progress and progress[-1] == 1.0 and progress == sorted(progress)
    assert stats == {"segments": 2, "segments_reused": 0}
    v = verify_output(out, width=size[0], height=size[1], duration=spec.duration, fps=30)
    assert v["vcodec"] == "h264" and v["acodec"] == "aac"
    # Wrong expectations must be reported, never silently accepted.
    with pytest.raises(EditorError):
        verify_output(out, width=size[0] + 2, height=size[1], duration=spec.duration, fps=30)
    with pytest.raises(EditorError):
        verify_output(tmp_path / "missing.mp4", width=size[0], height=size[1], duration=1, fps=30)


def test_render_failure_is_reported(media, settings, tmp_path):
    a = make_asset(media["land"])
    a["path"] = str(tmp_path / "deleted.mov")  # source vanished after planning
    e = TimelineEditor(None, {"a_1": a})
    e.create_timeline(720, 1280)
    e.select_clip("a_1", 0, 2)
    spec = compile_timeline(e.timeline, {"a_1": a}, tmp_path / "o.mp4", segment_dir=tmp_path / "seg")
    with pytest.raises(EditorError) as ei:
        render(spec)
    assert ei.value.code == "RENDER_FAILED"
    assert not (tmp_path / "o.mp4").exists()


def test_segment_cache_reused_and_av_in_sync(media, settings, tmp_path):
    a = make_asset(media["land"])
    assets = {"a_1": a}
    e = TimelineEditor(None, assets)
    e.create_timeline(720, 1280)
    for i in range(12):  # many short cuts with awkward lengths
        e.select_clip("a_1", i * 0.95, i * 0.95 + 0.53)
    out = tmp_path / "many.mp4"
    spec = compile_timeline(e.timeline, assets, out, segment_dir=tmp_path / "seg", x264_preset="ultrafast")
    assert render(spec)["segments_reused"] == 0
    verify_output(out, width=720, height=1280, duration=spec.duration, fps=30)
    import json, subprocess
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json",
                                       str(out)], capture_output=True, check=True).stdout)
    v = next(st for st in probe["streams"] if st["codec_type"] == "video")
    au = next(st for st in probe["streams"] if st["codec_type"] == "audio")
    assert int(v["nb_read_frames"]) == round(spec.duration * 30)
    assert abs(float(v["duration"]) - float(au["duration"])) < 0.05  # no drift
    # Edit again: change one clip → only that segment is re-rendered.
    e.trim_clip(e.timeline.video_track().clips[3].clip_id, src_out=3.4)
    spec2 = compile_timeline(e.timeline, assets, out, segment_dir=tmp_path / "seg", x264_preset="ultrafast")
    reused = render(spec2)["segments_reused"]
    assert 8 <= reused <= 11  # changed clip (and any clip whose frame count shifted) re-rendered
    verify_output(out, width=720, height=1280, duration=spec2.duration, fps=30)
