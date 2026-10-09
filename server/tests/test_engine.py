import pytest

from app.engine.timeline import TimelineEditor
from app.errors import EditorError
from app.storage.files import safe_path, sanitize_filename
from app.tools.registry import ToolContext, call_tool, tool_specs

ASSETS = {
    "a_1": {"asset_id": "a_1", "role": "clip", "meta": {"duration": 10.0, "has_video": True}},
    "r_1": {"asset_id": "r_1", "role": "reference", "meta": {"duration": 10.0, "has_video": True}},
}


def editor() -> TimelineEditor:
    e = TimelineEditor(None, ASSETS)
    e.create_timeline(1080, 1920)
    return e


def test_sanitize_and_paths(tmp_path):
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("my vid<script>.MOV") == "my vid_script_.MOV"
    assert sanitize_filename("\x00\x01") == "video"
    assert sanitize_filename("a" * 300 + ".mp4").endswith(".mp4")
    with pytest.raises(EditorError):
        safe_path(tmp_path, "../outside")
    with pytest.raises(EditorError):
        safe_path(tmp_path, "/etc/passwd")
    assert safe_path(tmp_path, "x.mp4") == (tmp_path / "x.mp4").resolve()


@pytest.mark.parametrize("src_in,src_out,fragment", [
    (-1, 2, "negative"),
    (3, 2, "must be after"),
    (2, 2, "must be after"),
    (5, 10.5, "beyond the end"),
    (1, 1.1, "shorter than"),
    (float("nan"), 2, "not a number"),
])
def test_select_clip_validation(src_in, src_out, fragment):
    e = editor()
    with pytest.raises(EditorError) as ei:
        e.select_clip("a_1", src_in, src_out)
    assert fragment in ei.value.message


def test_reference_and_unknown_assets_rejected():
    e = editor()
    with pytest.raises(EditorError, match="reference"):
        e.select_clip("r_1", 0, 2)
    with pytest.raises(EditorError, match="does not exist"):
        e.select_clip("nope", 0, 2)
    with pytest.raises(EditorError) as ei:
        e.delete_clip("c_missing")
    assert ei.value.code == "CLIP_NOT_FOUND"
    with pytest.raises(EditorError) as ei:
        e.select_clip("a_1", 0, 2, track_id="V9")
    assert ei.value.code == "TRACK_NOT_FOUND"


def test_ripple_split_move_trim():
    e = editor()
    c1 = e.select_clip("a_1", 0, 2)["clip_id"]
    c2 = e.select_clip("a_1", 5, 8)["clip_id"]
    assert e.timeline.duration == 5
    parts = e.split_clip(c2, 6.5)
    clips = e.timeline.video_track().clips
    assert [c.clip_id for c in clips] == [c1, parts["left"], parts["right"]]
    assert [c.timeline_start for c in clips] == [0, 2, 3.5]
    e.move_clip(parts["right"], 0)
    assert e.timeline.video_track().clips[0].clip_id == parts["right"]
    e.trim_clip(c1, src_out=1.0)
    assert e.timeline.duration == pytest.approx(4.0)
    assert e.validate_timeline()["clips"] == 3
    with pytest.raises(EditorError, match="inside the clip"):
        e.split_clip(c1, 5)


def test_transaction_rolls_back_whole_batch():
    e = editor()
    e.select_clip("a_1", 0, 2)
    rev = e.timeline.revision
    with pytest.raises(EditorError):
        with e.transaction():
            e.select_clip("a_1", 2, 4)
            e.select_clip("a_1", 4, 99)  # invalid -> everything in the batch is undone
    assert len(e.timeline.video_track().clips) == 1
    assert e.timeline.revision == rev


def test_max_duration_enforced_and_rolled_back_via_tool():
    e = TimelineEditor(None, ASSETS)
    e.create_timeline(1080, 1920, max_duration=5)
    ctx = ToolContext(editor=e, assets=ASSETS)
    assert call_tool(ctx, "select_clip", {"asset_id": "a_1", "src_in": 0, "src_out": 4})["ok"]
    res = call_tool(ctx, "select_clip", {"asset_id": "a_1", "src_in": 4, "src_out": 8})
    assert not res["ok"] and "exceeds" in res["error"]["message"]
    assert len(e.timeline.video_track().clips) == 1  # failed op did not leave a clip behind


def test_empty_timeline_invalid():
    e = editor()
    with pytest.raises(EditorError, match="empty"):
        e.validate_timeline()


def test_tool_layer_errors_and_unavailable_tools():
    ctx = ToolContext(editor=editor(), assets=ASSETS)
    assert call_tool(ctx, "nope")["error"]["code"] == "UNKNOWN_TOOL"
    assert call_tool(ctx, "add_zoom", {"clip_id": "nope"})["error"]["code"] == "CLIP_NOT_FOUND"
    assert call_tool(ctx, "select_clip", {"asset_id": "a_1"})["error"]["code"] == "BAD_ARGUMENTS"
    names = {t["name"] for t in tool_specs()}
    required = {"analyze_video", "get_video_metadata", "extract_frames", "detect_scenes", "analyze_audio", "detect_beats",
                "select_clip", "trim_clip", "split_clip", "move_clip", "delete_clip", "add_video_track",
                "add_audio_track", "add_music", "set_volume", "add_caption", "add_subtitles", "add_text",
                "add_transition", "add_zoom", "add_keyframe", "change_speed", "crop_video", "resize_video",
                "rotate_video", "apply_effect", "apply_color_adjustment", "add_sound_effect", "render_preview",
                "validate_timeline", "export_video"}
    assert required <= names  # every tool from the original specification is implemented
    assert all(t["input_schema"]["type"] == "object" for t in tool_specs())


def test_output_size_validation():
    e = TimelineEditor(None, ASSETS)
    with pytest.raises(EditorError):
        e.create_timeline(1081, 1920)
    e = editor()
    with pytest.raises(EditorError):
        e.resize_video(1080, 1920, "stretch")
