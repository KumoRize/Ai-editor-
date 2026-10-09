"""End-to-end tests through the HTTP API (upload → job → render → verify → download)."""
import pytest
from fastapi.testclient import TestClient

from app import config, jobs
from app.main import app


@pytest.fixture()
def client(settings):
    with TestClient(app) as c:
        yield c


def upload(c, path, name=None):
    with open(path, "rb") as f:
        return c.post(f"/api/uploads?filename={name or path.name}", content=f.read())


def test_full_workflow(client, media):
    cfg = client.get("/api/config").json()
    assert cfg["planner"] == "built-in rules" and "instagram_reels" in cfg["presets"]

    r = upload(client, media["land"], "My Clip (1).MOV")
    assert r.status_code == 200, r.text
    asset = r.json()
    assert asset["duration"] == pytest.approx(12, abs=0.1) and "path" not in asset
    # Same file again → deduplicated, analysis cache reused.
    assert upload(client, media["land"]).json()["asset_id"] == asset["asset_id"]

    job = client.post("/api/jobs", json={"asset_ids": [asset["asset_id"]], "prompt": "fast reel with subtitles",
                                         "preset": "instagram_reels", "resolution": "720p", "duration": 6,
                                         "fit": "blur"}).json()
    assert job["status"] == "queued" and len(job["steps"]) == 7
    done = jobs.wait_for(job["job_id"], 180)
    assert done["status"] == "complete", done["error"]
    j = client.get(f"/api/jobs/{job['job_id']}").json()
    res = j["result"]
    assert (res["width"], res["height"]) == (720, 1280)
    assert res["duration"] == pytest.approx(6, abs=0.3)
    # speech recognition is disabled in tests, so subtitles are honestly reported as not possible
    assert any("subtitles" in u for u in res["unsupported_requests"])
    assert all(s["status"] in ("done", "skipped") for s in j["steps"])
    assert "output_file" not in res

    v = client.get(res["video_url"])
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4" and len(v.content) > 1000
    part = client.get(res["video_url"], headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100   # seeking works on phones
    d = client.get(res["download_url"])
    assert "attachment" in d.headers["content-disposition"]

    # Edit again: new job referencing the previous one.
    job2 = client.post("/api/jobs", json={"asset_ids": [asset["asset_id"]], "prompt": "shorter",
                                          "preset": "square", "resolution": "720p", "duration": 4,
                                          "parent_job_id": job["job_id"]}).json()
    done2 = jobs.wait_for(job2["job_id"], 180)
    assert done2["status"] == "complete" and done2["result"]["width"] == 720 == done2["result"]["height"]


def test_upload_rejections(client, media, tmp_path):
    r = upload(client, media["fake"])
    assert r.status_code == 415 and "isn't supported" in r.json()["error"]["message"]
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert upload(client, txt).status_code == 415
    config.get_settings().max_upload_mb = 1
    r = upload(client, media["land"])
    assert r.status_code == 413 and "too large" in r.json()["error"]["message"]
    assert not any((config.get_settings().data_dir / "uploads").iterdir())  # nothing left behind
    assert client.post("/api/uploads?filename=x.mp4", content=b"").status_code == 400
    assert client.post("/api/uploads?filename=x.mp4&role=bogus", content=b"x").status_code == 400
    assert client.post("/api/uploads?filename=x.txt&role=music", content=b"x").status_code == 415


def test_job_validation(client, media):
    a = upload(client, media["silent"]).json()
    bad = [
        {"asset_ids": ["a_doesnotexist"]},
        {"asset_ids": [a["asset_id"]], "preset": "nope"},
        {"asset_ids": [a["asset_id"]], "resolution": "4k"},
        {"asset_ids": [a["asset_id"]], "preset": "instagram_reels", "duration": 500},
        {"asset_ids": [a["asset_id"], "a_other"]},
        {"asset_ids": [a["asset_id"]], "music_asset_id": a["asset_id"]},      # a clip is not a music file
        {"asset_ids": [a["asset_id"]], "reference_asset_id": "a_missing"},
        {"asset_ids": [a["asset_id"]], "fit": "stretch"},
    ]
    for body in bad:
        assert client.post("/api/jobs", json=body).status_code in (400, 422), body
    assert client.get("/api/jobs/j_0000000000000000").status_code == 404
    assert client.get("/api/jobs/../../etc/passwd").status_code == 404


def test_failed_job_reports_error_and_can_retry(client, media):
    from pathlib import Path

    from app.storage import db
    a = upload(client, media["silent"]).json()
    p = Path(db.get_asset(a["asset_id"])["path"])
    data = p.read_bytes()
    jobs.stop_workers()  # hold the queue so the failure is deterministic
    job = client.post("/api/jobs", json={"asset_ids": [a["asset_id"]], "resolution": "720p", "duration": 3}).json()
    p.unlink()  # upload lost after the job was queued
    jobs.start_workers()
    done = jobs.wait_for(job["job_id"], 60)
    assert done["status"] == "failed" and "upload it again" in done["error"]["message"]
    assert any(s["status"] == "failed" for s in done["steps"]) or done["steps"]
    assert client.get(f"/api/jobs/{job['job_id']}/video").status_code == 409  # no fake video
    p.write_bytes(data)
    assert client.post(f"/api/jobs/{job['job_id']}/retry").status_code == 200
    assert jobs.wait_for(job["job_id"], 120)["status"] == "complete"
    assert client.post(f"/api/jobs/{job['job_id']}/retry").status_code == 400  # only failed jobs


def test_busy_limit(client, media):
    a = upload(client, media["silent"]).json()
    jobs.stop_workers()
    body = {"asset_ids": [a["asset_id"]], "resolution": "720p", "duration": 2}
    codes = [client.post("/api/jobs", json=body).status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]
    jobs.start_workers()


def test_access_token(settings, media):
    config.get_settings().access_token = "s3cret"
    with TestClient(app) as c:
        assert c.get("/api/config").status_code == 401
        assert c.get("/api/health").status_code == 200
        assert c.get("/").status_code == 200  # the UI itself loads to show the login form
        assert c.post("/api/session", json={"token": "wrong"}).status_code == 401
        assert c.post("/api/session", json={"token": "s3cret"}).status_code == 200
        assert c.get("/api/config").status_code == 200  # cookie set
