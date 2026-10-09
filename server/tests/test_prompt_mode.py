"""'Full prompt' mode: settings and explicit instructions come from the user's text."""
import pytest

from app import jobs
from app.agent.llm_planner import plan_with_llm
from app.agent.plan import resolve_settings
from app.agent.prompt_parser import parse_prompt
from app.agent.rule_planner import plan_rules
from app.errors import EditorError

ASSETS = {"a_1": {"asset_id": "a_1", "role": "clip",
                  "meta": {"duration": 60.0, "has_video": True, "has_audio": True, "width": 1920, "height": 1080}}}
ANALYSIS = {"a_1": {"scenes": [], "audio": {"has_audio": True, "window_s": 0.5, "energy_db": [-20.0] * 120,
                                            "silent_ranges": [[10.0, 14.0], [30.0, 31.5]]}}}


@pytest.mark.parametrize("prompt,expect", [
    ("Make a 15-second vertical TikTok in 720p with a blurred background",
     dict(preset="tiktok", aspect_ratio="9:16", resolution="720p", duration=15.0, fit="blur")),
    ("landscape YouTube video, 1 minute, don't crop", dict(preset="youtube_landscape", aspect_ratio="16:9",
                                                          duration=60.0, fit="contain")),
    ("square post 1:1, about 30 secs", dict(aspect_ratio="1:1", duration=30.0)),
    ("YouTube Shorts please", dict(preset="youtube_shorts")),
    ("render it in 4k", dict(resolution="1080p")),
])
def test_parse_settings(prompt, expect):
    d = parse_prompt(prompt)
    for k, v in expect.items():
        assert getattr(d, k) == v, (k, getattr(d, k))


def test_parse_directives_and_traps():
    d = parse_prompt("Use 0:05-0:12 and then 1:02 to 1:10, then from 5 to 7 seconds. Mute the sound. 9:16")
    assert d.ranges == [(5.0, 12.0), (62.0, 70.0), (5.0, 7.0)]
    assert d.volume == 0.0 and d.aspect_ratio == "9:16"
    assert d.duration is None  # range seconds are not mistaken for the output length
    d = parse_prompt("first 20 seconds, remove the silent parts, louder")
    assert d.first_s == 20 and d.remove_silence and d.volume == 1.5
    assert parse_prompt("make it shorter").preset is None  # "shorter" is not "Shorts"
    assert parse_prompt("cut 1:10 to 1:20").aspect_ratio is None  # "1:10" is a time, not 1:1
    assert parse_prompt("12 to 5 seconds").notes  # backwards range is reported, not used


def test_resolve_defaults_and_limits():
    st, notes = resolve_settings(aspect_ratio="16:9")
    assert st.preset == "youtube_landscape" and st.duration == 60 and not notes
    st, notes = resolve_settings("instagram_reels", duration=300)
    assert st.duration == 90 and notes
    st, _ = resolve_settings(auto_length=True)
    assert st.duration is None and st.aspect_ratio == "9:16" and st.resolution == "1080p"


def test_rules_follow_explicit_ranges():
    plan = plan_rules("use 0:05-0:12 then 0:40-0:45 and 1:30-1:40", ASSETS, ANALYSIS, None)
    assert [(s.src_in, s.src_out) for s in plan.segments] == [(5.0, 12.0), (40.0, 45.0)]
    assert any("only 60.0s long" in n for n in plan.notes)
    with pytest.raises(EditorError):
        plan_rules("use 2:00-2:10", ASSETS, ANALYSIS, None)


def test_rules_remove_silence_and_mute():
    plan = plan_rules("remove the silent parts and mute it", ASSETS, ANALYSIS, None)
    assert plan.original_audio_volume == 0.0
    for s in plan.segments:  # nothing inside the silent stretches (minus padding)
        assert not (10.2 < s.src_in < 13.8) and not (s.src_in < 12 < s.src_out)
    assert plan.duration == pytest.approx(60 - (4 - 0.3) - (1.5 - 0.3), abs=0.01)
    assert any("Removed" in n for n in plan.notes)


def test_rules_cap_explicit_selection_to_length():
    plan = plan_rules("first 50 seconds", ASSETS, ANALYSIS, 20.0)
    assert plan.duration == pytest.approx(20.0) and any("shortened" in n for n in plan.notes)


class FakeProvider:
    name = "fake"

    def __init__(self, data):
        self.data, self.system = data, None

    def generate_json(self, *, system, user, schema, schema_name):
        self.system, self.schema = system, schema
        return self.data


def test_llm_prompt_mode_returns_settings():
    data = {"summary": "s", "unsupported_requests": [], "original_audio_volume": 0,
            "segments": [{"asset_id": "a_1", "src_in": 0, "src_out": 12, "reason": ""}],
            "settings": {"preset": "tiktok", "aspect_ratio": "16:9", "resolution": "720p", "duration_s": 0,
                         "fit": "blur"}}
    p = FakeProvider(data)
    plan, _, _ = plan_with_llm(p, "x", ASSETS, ANALYSIS, None, 600, None, 3, prompt_mode=True)
    assert plan.settings["preset"] == "tiktok" and plan.settings["duration"] is None
    assert plan.original_audio_volume == 0
    assert "FULL PROMPT MODE" in p.system and "settings" in p.schema["required"]


@pytest.fixture()
def client(settings):
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c


def test_prompt_mode_end_to_end(client, media):
    with open(media["land"], "rb") as f:
        a = client.post("/api/uploads?filename=land.mov", content=f.read()).json()
    # Form fields say 16:9/1080p/60s; in prompt mode they must be ignored in favour of the prompt.
    body = {"mode": "prompt", "asset_ids": [a["asset_id"]], "preset": "youtube_landscape", "resolution": "1080p",
            "duration": 60, "fit": "cover",
            "prompt": "Square 720p video with black bars. Use 0:01-0:03 then 0:08-0:11, mute it, add subtitles."}
    job = client.post("/api/jobs", json=body).json()
    done = jobs.wait_for(job["job_id"], 180)
    assert done["status"] == "complete", done["error"]
    r = done["result"]
    assert (r["width"], r["height"]) == (720, 720) and r["duration"] == pytest.approx(5.0, abs=0.05)
    assert r["settings"]["fit"] == "contain" and r["mode"] == "prompt"
    assert r["notes"][0].startswith("Understood your prompt as: Square post · 1:1 · 720p")
    assert any("subtitles" in u for u in r["unsupported_requests"])
    # The output really is silent (original sound muted).
    import subprocess
    from pathlib import Path

    from app.storage.files import job_dir
    out = Path(job_dir(job["job_id"])) / "final.mp4"
    vol = subprocess.run(["ffmpeg", "-v", "info", "-i", str(out), "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    assert "max_volume: -91" in vol or "max_volume: -inf" in vol


def test_prompt_mode_needs_a_prompt(client, media):
    with open(media["silent"], "rb") as f:
        a = client.post("/api/uploads?filename=s.mp4", content=f.read()).json()
    r = client.post("/api/jobs", json={"mode": "prompt", "asset_ids": [a["asset_id"]], "prompt": " "})
    assert r.status_code == 400
