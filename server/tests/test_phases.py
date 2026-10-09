"""Phases 2-6: creative planning, music/beat sync, reference-driven pacing, subtitles, quality checks,
the agent tool loop, projects + MCP, chunked uploads."""
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.agent import orchestrator
from app.agent.agent_loop import revise_timeline
from app.agent.plan import validate_plan
from app.agent.prompt_parser import parse_prompt
from app.agent.rule_planner import plan_rules
from app.analysis import transcribe as T
from app.engine.timeline import TimelineEditor
from app.render.quality import quality_check
from app.tools.registry import ToolContext
from tests.conftest import ff, make_asset

ASSETS = {"a_1": {"asset_id": "a_1", "role": "clip",
                  "meta": {"duration": 60.0, "has_video": True, "has_audio": True, "width": 1920, "height": 1080}}}
ANALYSIS = {"a_1": {"scenes": [], "audio": {"has_audio": True, "window_s": 0.5, "energy_db": [-20.0] * 120,
                                            "silent_ranges": []}}}


def test_rules_creative_layer():
    prompt = ('Fast 20 second reel: zooms, speed ramps, dissolve transitions, vintage look, vignette, grain, '
              'bold subtitles, the title "HELLO" and "BYE" at the end, whooshes, fade in, fade out')
    plan = plan_rules(prompt, ASSETS, ANALYSIS, 20.0)
    assert validate_plan(plan, ASSETS, 20.0, 90)[0] == []
    assert plan.duration == pytest.approx(20.0, abs=0.6)
    segs = plan.segments
    assert all(s.transition_in == "dissolve" for s in segs[1:])
    assert any(s.zoom in ("in", "out") for s in segs)
    assert any(s.speed_ramp != "none" for s in segs)
    assert plan.color_preset == "vintage" and plan.effects == ["vignette", "grain"]
    assert plan.captions.mode == "speech" and plan.captions.style == "bold"
    assert [t.text for t in plan.texts] == ["HELLO", "BYE"] and plan.texts[-1].end == pytest.approx(plan.duration, abs=0.05)
    assert plan.sfx_on_transitions and plan.intro == "fadeblack" and plan.outro == "fadeblack"


def test_rules_beat_sync_and_reference_pacing():
    music = {"asset_id": "m", "analysis": {"beats": [0.25 + i * 0.5 for i in range(200)], "tempo": 120.0,
                                           "duration": 100.0}}
    plan = plan_rules("cuts on the beat", ASSETS, ANALYSIS, 20.0, music=music)
    assert plan.music.use and plan.music.beat_sync
    for s in plan.segments:  # whole numbers of beats
        assert (s.out_len / 0.5) == pytest.approx(round(s.out_len / 0.5), abs=0.02)
    ref = {"style": {"target_shot_length": 3.0, "transition_ratio": 0.5, "transition_type": "fadeblack",
                     "zoom_ratio": 0.5, "zoom_kind": "out", "color_preset": "cool", "captions": False,
                     "beat_sync": False, "energy": "medium"}}
    plan = plan_rules("", ASSETS, ANALYSIS, 30.0, reference=ref)
    assert 8 <= len(plan.segments) <= 12  # ~3 s shots like the reference
    assert any(s.transition_in == "fadeblack" for s in plan.segments)
    assert any(s.zoom == "out" for s in plan.segments)
    assert plan.color_preset == "cool"


def test_missing_music_is_reported():
    plan = plan_rules("beat-synced with background music", ASSETS, ANALYSIS, 10.0)
    joined = " ".join(plan.unsupported_requests)
    assert "beat-synced" in joined and "background music" in joined


def test_speed_directive_and_multi_clip():
    two = {**ASSETS, "a_2": {**ASSETS["a_1"], "asset_id": "a_2", "meta": {**ASSETS["a_1"]["meta"], "duration": 30.0}}}
    an = {**ANALYSIS, "a_2": ANALYSIS["a_1"]}
    plan = plan_rules("2x speed timelapse", two, an, 15.0)
    assert all(s.speed == 2.0 for s in plan.segments)
    assert {s.asset_id for s in plan.segments} == {"a_1", "a_2"}
    assert plan.duration == pytest.approx(15.0, abs=0.5)
    assert parse_prompt("mute the original, only music").volume == 0.0


def fake_transcriber(path):
    return {"language": "en", "segments": [{"start": 0.0, "end": 12.0, "text": "x", "words": [
        {"start": t, "end": t + 0.4, "word": w} for t, w in
        [(0.5, "Hello"), (1.0, "and"), (1.5, "welcome."), (3.0, "This"), (3.5, "is"), (4.0, "a"), (4.5, "test."),
         (6.0, "Music"), (6.5, "and"), (7.0, "cuts."), (9.0, "Goodbye.")]]}]}


def test_full_job_all_features(media, settings, tmp_path):
    """Music + reference + subtitles + effects through the orchestrator, rendered and quality-checked."""
    T.set_transcriber(fake_transcriber)
    music_path = tmp_path / "beat.m4a"
    ff("-f", "lavfi", "-i", "aevalsrc='0.9*sin(2*PI*80*t)*exp(-30*mod(t,0.5))':d=30:s=44100", str(music_path))
    assets = {"a_1": make_asset(media["land"], asset_id="a_1"),
              "m_1": make_asset(music_path, role="music", asset_id="m_1"),
              "r_1": make_asset(media["land"], role="reference", asset_id="r_1")}
    steps = orchestrator.initial_steps()
    rep = orchestrator.Reporter(steps, lambda st, p: None)
    req = {"mode": "prompt", "music_asset_id": "m_1", "reference_asset_id": "r_1",
           "prompt": 'Square 720p, 8 seconds, fast cuts on the beat, subtitles, zooms, slide transitions, '
                     'warm colour, title "TEST", whooshes, fade out'}
    res = orchestrator.run_edit_job("j_test", req, assets, tmp_path, rep)
    assert (res["width"], res["height"]) == (720, 720)
    assert res["duration"] == pytest.approx(8.0, abs=0.6)
    assert res["music_used"] and res["reference_used"]
    applied = " ".join(res["applied"])
    for word in ("zooms", "warm", "music", "on the beat", "subtitle", "text", "sound effect", "fade out"):
        assert word in applied, word
    assert res["quality"]["passed"], res["quality"]
    assert all(s["status"] in ("done", "skipped") for s in steps)
    tl = json.loads((tmp_path / "timeline.json").read_text())
    assert any(t["kind"] == "caption" for t in tl["texts"]) and any(t["text"] == "TEST" for t in tl["texts"])
    assert (tmp_path / res["output_file"]).is_file()


def test_quality_check_detects_problems(media, settings, tmp_path):
    loud = tmp_path / "loud.mp4"
    ff("-f", "lavfi", "-i", "color=black:s=320x240:d=4:r=30", "-f", "lavfi", "-i", "sine=f=200:d=4",
       "-filter_complex", "[1:a]volume=12[a]", "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-c:a", "aac", str(loud))
    asset = make_asset(loud, asset_id="a_1")
    ed = TimelineEditor(None, {"a_1": asset})
    ed.create_timeline(320, 240)
    ed.select_clip("a_1", 0, 4)
    q = quality_check(loud, ed.timeline, {"a_1": asset})
    codes = {i["code"] for i in q["issues"]}
    assert "BLACK_FRAMES" in codes and "CLIPPING" in codes and not q["passed"]
    assert "FROZEN" in codes  # a black frame doesn't move


class ChatFake:
    """Scripted tool-using model."""
    name = "fake-agent"

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def chat(self, *, system, messages, tools):
        self.seen.append(messages[-1])
        calls = self.script.pop(0)
        return {"text": "", "raw": [], "stop": "tool_use",
                "tool_calls": [{"id": f"t{i}", "name": n, "input": a} for i, (n, a) in enumerate(calls)]}


def _tl_with_three():
    ed = TimelineEditor(None, ASSETS)
    ed.create_timeline(720, 1280)
    ids = [ed.select_clip("a_1", i * 10, i * 10 + 3)["clip_id"] for i in range(3)]
    return ed.timeline.model_dump(), ids


def test_agent_loop_revises_timeline():
    prev, ids = _tl_with_three()
    agent = ChatFake([
        [("get_timeline", {})],
        [("delete_clip", {"clip_id": ids[1]}), ("add_text", {"text": "Hi", "start": 0, "end": 1}),
         ("trim_clip", {"clip_id": ids[0], "src_out": 999})],   # last call is invalid → error returned
        [("finish_edit", {"summary": "Removed the middle clip and added a title."})],
    ])
    ctx = ToolContext(editor=TimelineEditor(None, ASSETS), assets=ASSETS)
    tl, summary = revise_timeline(agent, "remove the middle clip, add Hi", prev, ctx)
    assert [c.clip_id for c in tl.video_track().clips] == [ids[0], ids[2]]
    assert tl.texts[0].text == "Hi" and tl.video_track().clips[0].src_out == 3  # invalid trim had no effect
    results = agent.seen[2]["results"]
    assert results[2]["is_error"] and "beyond the end" in results[2]["content"]
    assert summary.startswith("Removed")


def test_agent_loop_gives_up_honestly():
    prev, _ = _tl_with_three()
    agent = ChatFake([[("get_timeline", {})]] * 3)
    ctx = ToolContext(editor=TimelineEditor(None, ASSETS), assets=ASSETS)
    from app.agent.providers.base import LLMError
    with pytest.raises(LLMError):
        revise_timeline(agent, "do something", prev, ctx, max_turns=3)  # unchanged timeline after all turns


@pytest.fixture()
def client(settings):
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_chunked_resumable_upload(client, media):
    data = media["land"].read_bytes()
    init = client.post("/api/uploads/init", json={"filename": "big.mov", "size": len(data), "role": "clip"}).json()
    uid, chunk = init["upload_id"], 1024 * 1024
    assert client.put(f"/api/uploads/{uid}?offset=0", content=data[:chunk]).json()["received"] == chunk
    # a retried / out-of-order chunk is rejected with the position to continue from
    r = client.put(f"/api/uploads/{uid}?offset=0", content=data[:chunk])
    assert r.status_code == 409 and r.json()["received"] == chunk
    assert client.post(f"/api/uploads/{uid}/complete").status_code == 400  # incomplete
    assert client.get(f"/api/uploads/{uid}").json()["received"] == chunk  # resume point after a reload
    off = chunk
    while off < len(data):
        off = client.put(f"/api/uploads/{uid}?offset={off}", content=data[off:off + chunk]).json()["received"]
    asset = client.post(f"/api/uploads/{uid}/complete").json()
    assert asset["duration"] == pytest.approx(12, abs=0.1) and asset["thumb_url"]
    assert client.get(asset["thumb_url"]).headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/assets/{asset['asset_id']}/frame?t=2.5&w=128").status_code == 200
    assert client.get("/api/uploads/u_0000000000000000").status_code == 404
    assert client.post("/api/uploads/init", json={"filename": "x.mov", "size": 10**12, "role": "clip"}).status_code == 413
    assert client.post("/api/uploads/init", json={"filename": "x.exe", "size": 10, "role": "clip"}).status_code == 415


def test_music_upload_and_multi_clip_job(client, media, tmp_path):
    song = tmp_path / "song.mp3"
    ff("-f", "lavfi", "-i", "aevalsrc='0.8*sin(2*PI*70*t)*exp(-25*mod(t,0.5))':d=20:s=44100", str(song))
    up = lambda p, role: client.post(f"/api/uploads?filename={p.name}&role={role}", content=p.read_bytes()).json()
    a1, a2, m = up(media["land"], "clip"), up(media["rotated"], "clip"), up(song, "music")
    assert m["role"] == "music" and m["has_audio"] and "thumb_url" not in m
    job = client.post("/api/jobs", json={"asset_ids": [a1["asset_id"], a2["asset_id"]], "music_asset_id": m["asset_id"],
                                         "prompt": "fast cuts on the beat with zooms", "preset": "youtube_shorts",
                                         "resolution": "720p", "duration": 8}).json()
    done = jobs.wait_for(job["job_id"], 300)
    assert done["status"] == "complete", done["error"]
    r = done["result"]
    assert r["music_used"] and {c["asset_id"] for c in r["timeline"]["clips"]} == {a1["asset_id"], a2["asset_id"]}


def test_projects_and_mcp_dispatch(client, media):
    from app import mcp_server
    a = client.post(f"/api/uploads?filename=c.mov", content=media["land"].read_bytes()).json()
    tools = {t.name for t in mcp_server._list_tools()}
    assert {"create_project", "render_project", "get_job", "select_clip", "add_transition", "add_music"} <= tools
    assert "export_video" not in tools  # rendering only through render_project jobs
    p = mcp_server._call("create_project", {"asset_ids": [a["asset_id"]], "preset": "square", "resolution": "720p"})
    pid = p["result"]["project_id"]
    for s0 in (0.5, 4.5, 8.5):
        assert mcp_server._call("select_clip", {"project_id": pid, "asset_id": a["asset_id"], "src_in": s0,
                                                "src_out": s0 + 2})["ok"]
    bad = mcp_server._call("add_transition", {"project_id": pid, "clip_id": "c_nope", "type": "fade"})
    assert not bad["ok"] and bad["error"]["code"] == "CLIP_NOT_FOUND"
    tl = mcp_server._call("get_timeline", {"project_id": pid})["result"]
    second = tl["clips"][1]["clip_id"]
    assert mcp_server._call("add_transition", {"project_id": pid, "clip_id": second, "type": "circleopen",
                                               "duration": 0.5})["ok"]
    job = mcp_server._call("render_project", {"project_id": pid})["result"]
    done = jobs.wait_for(job["job_id"], 120)
    assert done["status"] == "complete", done["error"]
    st = mcp_server._call("get_job", {"job_id": job["job_id"]})["result"]
    assert st["video_url"] and client.get(st["video_url"]).status_code == 200
    assert st["duration"] == pytest.approx(5.5, abs=0.1)
    assert not mcp_server._call("render_project", {"project_id": "p_missing"})["ok"]


def test_mcp_http_requires_token(settings, media):
    from app import config
    from app.main import app
    config.get_settings().access_token = "secret"
    with TestClient(app) as c:
        assert c.post("/mcp/", json={}).status_code == 401
        r = c.post("/mcp/", headers={"Authorization": "Bearer secret", "Accept": "application/json, text/event-stream",
                                     "Content-Type": "application/json"},
                   json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "t", "version": "1"}}})
        assert r.status_code == 200 and "ai-video-editor" in r.text


def test_edit_again_uses_agent_tools(client, media, monkeypatch):
    a = client.post("/api/uploads?filename=c.mov", content=media["land"].read_bytes()).json()
    first = client.post("/api/jobs", json={"asset_ids": [a["asset_id"]], "prompt": "fast cuts", "resolution": "720p",
                                           "duration": 6}).json()
    done = jobs.wait_for(first["job_id"], 120)
    n_before = len(done["result"]["timeline"]["clips"])

    class Agent(ChatFake):
        name = "fake-agent"

        def generate_json(self, **kw):  # not used on the agent path
            raise AssertionError("planner should not be called")

    def script():
        return Agent([[("get_timeline", {})],
                      [("add_text", {"text": "Revised", "start": 0.2, "end": 1.5}),
                       ("apply_color_adjustment", {"clip_id": "all", "preset": "bw"})],
                      [("finish_edit", {"summary": "Made it black and white with a title."})]])
    monkeypatch.setattr(orchestrator, "make_provider", lambda s: script())
    second = client.post("/api/jobs", json={"asset_ids": [a["asset_id"]], "prompt": "black and white, add a title",
                                            "resolution": "720p", "duration": 6,
                                            "parent_job_id": first["job_id"]}).json()
    done2 = jobs.wait_for(second["job_id"], 120)
    assert done2["status"] == "complete", done2["error"]
    r = done2["result"]
    assert r["planner"] == "fake-agent (agent)" and r["summary"].startswith("Made it black and white")
    assert len(r["timeline"]["clips"]) == n_before  # cuts kept; only the requested changes were made


def test_chunk_size_is_bounded(client):
    init = client.post("/api/uploads/init", json={"filename": "v.mp4", "size": 100, "role": "clip"}).json()
    r = client.put(f"/api/uploads/{init['upload_id']}?offset=0", content=b"x" * 101)
    assert r.status_code == 400 and client.get(f"/api/uploads/{init['upload_id']}").json()["received"] == 0
