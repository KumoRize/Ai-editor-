"""HTTP API + static mobile web UI."""
from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import jobs
from .agent.orchestrator import initial_steps
from .agent.providers import planner_name
from .analysis.media import get_video_metadata, validate_video
from .config import get_settings
from .errors import EditorError
from .presets import FITS, PRESETS, SIZES
from .storage import db
from .storage.files import (VIDEO_EXTENSIONS, check_extension, job_dir, new_id, safe_path, sanitize_filename,
                            save_stream, uploads_dir)
from .tools.registry import CURRENT_PHASE, tool_specs

log = logging.getLogger("editor")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
MAX_CLIPS = 1  # Phase 1: one clip per edit; multi-clip timelines arrive in Phase 2.
MAX_ACTIVE_JOBS = 5  # simple protection for a small server
COOKIE = "ve_token"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if not s.access_token:
        log.warning("ACCESS_TOKEN is not set: the editor is open to anyone who can reach it.")
    db.conn()
    jobs.start_workers()
    yield
    jobs.stop_workers()


app = FastAPI(title="AI Video Editor", lifespan=lifespan, docs_url=None, redoc_url=None)


# ------------------------------------------------------------------ errors & auth
@app.exception_handler(EditorError)
async def editor_error_handler(_: Request, e: EditorError):
    status = {"FILE_TOO_LARGE": 413, "UNSUPPORTED_FORMAT": 415, "BUSY": 429}.get(e.code, 400)
    return JSONResponse(status_code=status, content={"error": e.to_dict()})


def _authorized(request: Request) -> bool:
    token = get_settings().access_token
    if not token:
        return True
    given = request.cookies.get(COOKIE) or request.headers.get("x-access-token") or ""
    return hmac.compare_digest(given.encode(), token.encode())


@app.middleware("http")
async def auth_and_headers(request: Request, call_next):
    path = request.url.path
    public = (not path.startswith("/api/")) or path in ("/api/session", "/api/health")
    if not public and not _authorized(request):
        return JSONResponse(status_code=401, content={"error": {
            "code": "UNAUTHORIZED", "message": "Please enter the access code.", "retryable": False}})
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    return resp


class SessionIn(BaseModel):
    token: str = Field(max_length=500)


@app.post("/api/session")
async def create_session(body: SessionIn, request: Request):
    token = get_settings().access_token
    if token and not hmac.compare_digest(body.token.encode(), token.encode()):
        await asyncio.sleep(1.0)  # slow down guessing
        raise HTTPException(401, detail="Wrong access code.")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, body.token, httponly=True, samesite="strict",
                    secure=request.url.scheme == "https", max_age=60 * 60 * 24 * 30)
    return resp


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/config")
async def public_config():
    s = get_settings()
    return {
        "phase": CURRENT_PHASE,
        "max_upload_mb": s.max_upload_mb,
        "max_clips": MAX_CLIPS,
        "accept": sorted(VIDEO_EXTENSIONS),
        "presets": PRESETS,
        "resolutions": ["720p", "1080p"],
        "aspect_ratios": sorted({a for a, _ in SIZES}),
        "fits": FITS,
        "planner": planner_name(s),
        "features": {
            "reference_video": {"available": False, "phase": 3},
            "music": {"available": False, "phase": 2},
            "multiple_clips": {"available": False, "phase": 2},
            "captions": {"available": False, "phase": 2},
        },
        "tools": [{"name": t["name"], "available": t["available"], "phase": t["phase"]}
                  for t in tool_specs(available_only=False)],
    }


# ------------------------------------------------------------------ uploads
def _public_asset(a: dict) -> dict:
    m = a["meta"]
    return {"asset_id": a["asset_id"], "role": a["role"], "name": a["original_name"], "size_bytes": a["size"],
            "duration": m.get("duration"), "width": m.get("width"), "height": m.get("height"),
            "has_audio": m.get("has_audio"), "fps": m.get("fps")}


@app.post("/api/uploads")
async def upload(request: Request, filename: str, role: str = "clip"):
    """Raw-body upload (the phone sends the file as the request body)."""
    s = get_settings()
    if role != "clip":
        raise EditorError("NOT_AVAILABLE", "Only video clips can be uploaded in this version "
                                           "(music and reference videos arrive in Phases 2 and 3).")
    display = sanitize_filename(filename)
    ext = check_extension(display, VIDEO_EXTENSIONS)
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > s.max_upload_bytes:
        raise EditorError("FILE_TOO_LARGE", f"The uploaded file is too large (limit {s.max_upload_mb} MB).")
    asset_id = new_id("a")
    dest = safe_path(uploads_dir(), f"{asset_id}{ext}")
    size, sha = await save_stream(request.stream(), dest, s.max_upload_bytes)

    existing = db.find_asset_by_sha(sha, role)
    if existing and Path(existing["path"]).is_file():
        dest.unlink(missing_ok=True)  # identical file already stored: reuse it and its cached analysis
        Path(existing["path"]).touch()
        return _public_asset(existing)
    try:
        meta = await asyncio.to_thread(get_video_metadata, dest)
        await asyncio.to_thread(validate_video, dest, meta)
    except EditorError:
        dest.unlink(missing_ok=True)
        raise
    asset = {"asset_id": asset_id, "role": role, "original_name": display, "path": str(dest),
             "sha256": sha, "size": size, "meta": meta}
    db.insert_asset(asset)
    return _public_asset(asset)


# ------------------------------------------------------------------ jobs
class JobIn(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=10)
    prompt: str = Field(default="", max_length=2000)
    preset: str = "instagram_reels"
    aspect_ratio: str | None = None
    resolution: str = "1080p"
    duration: float | None = Field(default=None, gt=0, le=900)
    fit: str = "cover"
    parent_job_id: str | None = None


def _public_job(j: dict) -> dict:
    out = {"job_id": j["job_id"], "status": j["status"], "progress": j["progress"], "steps": j["steps"],
           "error": j["error"], "request": {k: j["request"].get(k) for k in
                                            ("asset_ids", "prompt", "preset", "aspect_ratio", "resolution",
                                             "duration", "fit")}}
    if j["status"] == "complete" and j["result"]:
        r = dict(j["result"])
        r.pop("output_file", None)
        r["video_url"] = f"/api/jobs/{j['job_id']}/video"
        r["download_url"] = f"/api/jobs/{j['job_id']}/video?download=1"
        out["result"] = r
    return out


@app.post("/api/jobs")
async def create_job(body: JobIn):
    if body.preset not in PRESETS:
        raise EditorError("BAD_REQUEST", "Unknown preset.")
    aspect = body.aspect_ratio or PRESETS[body.preset]["aspect_ratio"]
    if (aspect, body.resolution) not in SIZES:
        raise EditorError("BAD_REQUEST", f"Unsupported output format {aspect} at {body.resolution}.")
    if body.fit not in FITS:
        raise EditorError("BAD_REQUEST", "Unknown framing option.")
    if body.duration and body.duration > PRESETS[body.preset]["max_duration"]:
        raise EditorError("BAD_REQUEST", f"{PRESETS[body.preset]['label']} videos can be at most "
                                         f"{PRESETS[body.preset]['max_duration']} seconds.")
    if len(db.jobs_with_status("queued", "running")) >= MAX_ACTIVE_JOBS:
        raise EditorError("BUSY", "Too many edits are in progress. Please wait for one to finish.", retryable=True)
    ids = list(dict.fromkeys(body.asset_ids))
    if len(ids) > MAX_CLIPS:
        raise EditorError("NOT_AVAILABLE", "Editing several clips together arrives in Phase 2. Please use one clip.")
    for aid in ids:
        a = db.get_asset(aid)
        if a is None or a["role"] != "clip" or not Path(a["path"]).is_file():
            raise EditorError("ASSET_MISSING", "An uploaded file is no longer available. Please upload it again.")
    if body.parent_job_id and db.get_job(body.parent_job_id) is None:
        raise EditorError("BAD_REQUEST", "The edit you are revising no longer exists.")
    req = body.model_dump()
    req["asset_ids"] = ids
    req["aspect_ratio"] = aspect
    req["prompt"] = body.prompt.strip()
    job_id = new_id("j")
    db.insert_job(job_id, req, initial_steps())
    jobs.enqueue(job_id)
    return _public_job(db.get_job(job_id))


def _job_or_404(job_id: str) -> dict:
    j = db.get_job(job_id)
    if j is None:
        raise HTTPException(404, detail="Edit not found.")
    return j


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    return _public_job(_job_or_404(job_id))


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    j = _job_or_404(job_id)
    if j["status"] != "failed":
        raise EditorError("BAD_REQUEST", "Only failed edits can be retried.")
    db.update_job(job_id, status="queued", progress=0.0, error=None, result=None, steps=initial_steps())
    jobs.enqueue(job_id)
    return _public_job(db.get_job(job_id))


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str, download: int = 0):
    j = _job_or_404(job_id)
    if j["status"] != "complete" or not j["result"]:
        raise HTTPException(409, detail="This edit has no finished video.")
    path = safe_path(job_dir(job_id), j["result"]["output_file"])
    if not path.is_file():
        raise HTTPException(410, detail="This video has expired. Please run the edit again.")
    headers = {"Cache-Control": "private, max-age=3600"}
    if download:
        return FileResponse(path, media_type="video/mp4", filename=f"edit_{job_id[2:10]}.mp4", headers=headers)
    return FileResponse(path, media_type="video/mp4", headers=headers)


# ------------------------------------------------------------------ web UI
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
