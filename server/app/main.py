"""HTTP API + static mobile web UI (+ MCP endpoint at /mcp)."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import jobs
from .agent.orchestrator import initial_steps
from .agent.providers import planner_name
from .analysis import transcribe as speech
from .analysis.media import extract_frames, get_video_metadata, thumbnail, validate_audio, validate_video
from .config import get_settings
from .errors import EditorError, too_large
from .presets import FITS, PRESETS, SIZES
from .storage import db
from .storage.files import (AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, check_extension, job_dir, new_id, safe_path,
                            sanitize_filename, save_stream, uploads_dir)
from .tools.registry import CURRENT_PHASE, tool_specs

log = logging.getLogger("editor")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
MAX_CLIPS = 10
MAX_ACTIVE_JOBS = 5  # simple protection for a small server
CHUNK_SIZE = 4 * 1024 * 1024
MAX_PENDING_UPLOADS = 30
COOKIE = "ve_token"
ROLES = {"clip": VIDEO_EXTENSIONS, "reference": VIDEO_EXTENSIONS, "music": AUDIO_EXTENSIONS | VIDEO_EXTENSIONS}


class _MCPMount:
    """Stable ASGI mount for /mcp; a fresh MCP app (and session manager) is built on every startup."""

    def __init__(self):
        self.inner = None

    async def __call__(self, scope, receive, send):
        if self.inner is None:
            resp = JSONResponse(status_code=503, content={"error": {"code": "MCP_UNAVAILABLE", "retryable": True,
                                                                    "message": "The MCP server is not running."}})
            await resp(scope, receive, send)
            return
        await self.inner(scope, receive, send)


def _build_mcp():
    try:
        from .mcp_server import build_mcp_app
        return build_mcp_app()
    except Exception:  # noqa: BLE001 - the editor works without MCP
        log.exception("MCP server unavailable")
        return None


_mcp = _MCPMount()


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if not s.access_token:
        log.warning("ACCESS_TOKEN is not set: the editor is open to anyone who can reach it.")
    db.conn()
    jobs.start_workers()
    built = _build_mcp()
    try:
        if built is not None:
            _mcp.inner = built["app"]
            async with built["lifespan"]():
                yield
        else:
            yield
    finally:
        _mcp.inner = None
        jobs.stop_workers()


app = FastAPI(title="AI Video Editor", lifespan=lifespan, docs_url=None, redoc_url=None)


# ------------------------------------------------------------------ errors & auth
@app.exception_handler(EditorError)
async def editor_error_handler(_: Request, e: EditorError):
    status = {"FILE_TOO_LARGE": 413, "UNSUPPORTED_FORMAT": 415, "BUSY": 429, "CONFLICT": 409}.get(e.code, 400)
    return JSONResponse(status_code=status, content={"error": e.to_dict()})


def _authorized(request: Request) -> bool:
    token = get_settings().access_token
    if not token:
        return True
    auth = request.headers.get("authorization", "")
    bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
    given = request.cookies.get(COOKIE) or request.headers.get("x-access-token") or bearer or ""
    return hmac.compare_digest(given.encode(), token.encode())


@app.middleware("http")
async def auth_and_headers(request: Request, call_next):
    path = request.url.path
    protected = path.startswith("/api/") or path.startswith("/mcp")
    public = path in ("/api/session", "/api/health")
    if protected and not public and not _authorized(request):
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
    speech_ok, speech_reason = speech.availability()
    return {
        "phase": CURRENT_PHASE,
        "max_upload_mb": s.max_upload_mb,
        "max_clips": MAX_CLIPS,
        "chunk_size": CHUNK_SIZE,
        "accept": {"clip": sorted(VIDEO_EXTENSIONS), "reference": sorted(VIDEO_EXTENSIONS),
                   "music": sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)},
        "presets": PRESETS,
        "resolutions": ["720p", "1080p"],
        "aspect_ratios": sorted({a for a, _ in SIZES}),
        "fits": FITS,
        "planner": planner_name(s),
        "features": {
            "multiple_clips": {"available": True},
            "reference_video": {"available": True},
            "music": {"available": True},
            "subtitles_from_speech": {"available": speech_ok, "note": speech_reason},
            "mcp": {"available": _mcp.inner is not None, "path": "/mcp/"},
        },
        "tools": [{"name": t["name"], "available": t["available"], "phase": t["phase"]}
                  for t in tool_specs(available_only=False)],
    }


# ------------------------------------------------------------------ uploads
def _public_asset(a: dict) -> dict:
    m = a["meta"]
    out = {"asset_id": a["asset_id"], "role": a["role"], "name": a["original_name"], "size_bytes": a["size"],
           "duration": m.get("duration"), "width": m.get("width"), "height": m.get("height"),
           "has_audio": m.get("has_audio"), "has_video": m.get("has_video"), "fps": m.get("fps"),
           "hdr": m.get("hdr", False)}
    if m.get("has_video"):
        out["thumb_url"] = f"/api/assets/{a['asset_id']}/thumb"
    return out


def _check_role(role: str, filename: str) -> tuple[str, str]:
    if role not in ROLES:
        raise EditorError("BAD_REQUEST", "Unknown upload type.")
    display = sanitize_filename(filename)
    if role == "music":
        try:
            ext = check_extension(display, ROLES[role])
        except EditorError:
            raise EditorError("UNSUPPORTED_FORMAT", "That audio format isn't supported. "
                                                    "Use MP3, M4A, WAV or AAC (or a video with sound).") from None
    else:
        ext = check_extension(display, ROLES[role])
    return display, ext


async def _finalize(dest: Path, display: str, role: str, size: int, sha: str, asset_id: str) -> dict:
    """Validate a fully received file and register it (or reuse an identical earlier upload)."""
    existing = db.find_asset_by_sha(sha, role)
    if existing and Path(existing["path"]).is_file():
        dest.unlink(missing_ok=True)  # identical file already stored: reuse it and its cached analysis
        Path(existing["path"]).touch()
        return _public_asset(existing)
    try:
        meta = await asyncio.to_thread(get_video_metadata, dest)
        if role == "music":
            await asyncio.to_thread(validate_audio, dest, meta)
        else:
            await asyncio.to_thread(validate_video, dest, meta)
    except EditorError:
        dest.unlink(missing_ok=True)
        raise
    asset = {"asset_id": asset_id, "role": role, "original_name": display, "path": str(dest),
             "sha256": sha, "size": size, "meta": meta}
    db.insert_asset(asset)
    return _public_asset(asset)


@app.post("/api/uploads")
async def upload(request: Request, filename: str, role: str = "clip"):
    """Simple upload: the whole file is the request body."""
    s = get_settings()
    display, ext = _check_role(role, filename)
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > s.max_upload_bytes:
        raise too_large(s.max_upload_mb)
    asset_id = new_id("a")
    dest = safe_path(uploads_dir(), f"{asset_id}{ext}")
    size, sha = await save_stream(request.stream(), dest, s.max_upload_bytes)
    return await _finalize(dest, display, role, size, sha, asset_id)


# Resumable chunked uploads (survive network drops on mobile data).
class UploadInit(BaseModel):
    filename: str = Field(max_length=300)
    size: int = Field(gt=0)
    role: str = "clip"


def _chunk_paths(upload_id: str) -> tuple[Path, Path]:
    if not (upload_id.startswith("u_") and len(upload_id) == 18 and upload_id[2:].isalnum()):
        raise HTTPException(404, detail="Upload not found.")
    d = get_settings().data_dir / "tmp" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return safe_path(d, f"{upload_id}.part"), safe_path(d, f"{upload_id}.json")


def _chunk_state(upload_id: str) -> tuple[dict, Path, Path]:
    part, meta = _chunk_paths(upload_id)
    if not meta.exists():
        raise HTTPException(404, detail="Upload not found or expired. Please start again.")
    st = json.loads(meta.read_text())
    st["received"] = part.stat().st_size if part.exists() else 0
    return st, part, meta


@app.post("/api/uploads/init")
async def upload_init(body: UploadInit):
    s = get_settings()
    display, ext = _check_role(body.role, body.filename)
    if body.size > s.max_upload_bytes:
        raise too_large(s.max_upload_mb)
    pending = list((s.data_dir / "tmp" / "uploads").glob("u_*.json"))
    if len(pending) >= MAX_PENDING_UPLOADS:
        raise EditorError("BUSY", "Too many unfinished uploads. Finish or wait for them to expire.", retryable=True)
    upload_id = new_id("u")
    part, meta = _chunk_paths(upload_id)
    part.write_bytes(b"")
    meta.write_text(json.dumps({"filename": display, "ext": ext, "size": body.size, "role": body.role,
                                "created": time.time()}))
    return {"upload_id": upload_id, "chunk_size": CHUNK_SIZE, "received": 0, "size": body.size}


@app.get("/api/uploads/{upload_id}")
async def upload_status(upload_id: str):
    st, _, _ = _chunk_state(upload_id)
    return {"upload_id": upload_id, "received": st["received"], "size": st["size"]}


@app.put("/api/uploads/{upload_id}")
async def upload_chunk(upload_id: str, offset: int, request: Request):
    st, part, _ = _chunk_state(upload_id)
    if offset != st["received"]:
        # Client and server disagree (e.g. a retried chunk): tell the client where to continue.
        return JSONResponse(status_code=409, content={"error": {"code": "CONFLICT", "retryable": True,
                                                                "message": "Upload position mismatch."},
                                                      "received": st["received"]})
    limit = min(CHUNK_SIZE + 1024, st["size"] - offset)
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit:
        raise EditorError("BAD_REQUEST", "Chunk too large.")
    body = bytearray()
    async for piece in request.stream():  # bounded read, even without a Content-Length header
        body += piece
        if len(body) > limit:
            raise EditorError("BAD_REQUEST", "Chunk too large or more data than the announced file size.")
    with open(part, "ab") as f:
        f.write(body)
    return {"received": offset + len(body), "size": st["size"]}


@app.post("/api/uploads/{upload_id}/complete")
async def upload_complete(upload_id: str):
    st, part, meta = _chunk_state(upload_id)
    if st["received"] != st["size"]:
        raise EditorError("INCOMPLETE", f"The upload is incomplete ({st['received']} of {st['size']} bytes).",
                          retryable=True)
    h = hashlib.sha256()
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    asset_id = new_id("a")
    dest = safe_path(uploads_dir(), f"{asset_id}{st['ext']}")
    part.replace(dest)
    meta.unlink(missing_ok=True)
    return await _finalize(dest, st["filename"], st["role"], st["size"], h.hexdigest(), asset_id)


def _asset_or_404(asset_id: str) -> dict:
    a = db.get_asset(asset_id)
    if a is None or not Path(a["path"]).is_file():
        raise HTTPException(404, detail="File not found.")
    return a


@app.get("/api/assets/{asset_id}")
async def get_asset(asset_id: str):
    return _public_asset(_asset_or_404(asset_id))


@app.get("/api/assets/{asset_id}/thumb")
async def asset_thumb(asset_id: str):
    a = _asset_or_404(asset_id)
    if not a["meta"].get("has_video"):
        raise HTTPException(404, detail="No picture.")
    p = await asyncio.to_thread(thumbnail, Path(a["path"]), a["sha256"], a["meta"]["duration"])
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/assets/{asset_id}/frame")
async def asset_frame(asset_id: str, t: float, w: int = 256):
    a = _asset_or_404(asset_id)
    if not a["meta"].get("has_video") or not (0 <= t <= a["meta"]["duration"]):
        raise HTTPException(404, detail="No frame there.")
    p = (await asyncio.to_thread(extract_frames, Path(a["path"]), a["sha256"], [t], max(64, min(640, w))))[0]
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


# ------------------------------------------------------------------ jobs
class JobIn(BaseModel):
    # "settings": format/length/framing come from the form. "prompt": the agent reads them from the prompt.
    mode: Literal["settings", "prompt"] = "settings"
    asset_ids: list[str] = Field(min_length=1, max_length=MAX_CLIPS * 2)
    music_asset_id: str | None = None
    reference_asset_id: str | None = None
    prompt: str = Field(default="", max_length=4000)
    preset: str = "instagram_reels"
    aspect_ratio: str | None = None
    resolution: str = "1080p"
    duration: float | None = Field(default=None, gt=0, le=900)
    fit: str = "cover"
    parent_job_id: str | None = None


def _public_job(j: dict) -> dict:
    out = {"job_id": j["job_id"], "status": j["status"], "progress": j["progress"], "steps": j["steps"],
           "error": j["error"], "request": {k: j["request"].get(k) for k in
                                            ("mode", "asset_ids", "music_asset_id", "reference_asset_id", "prompt",
                                             "preset", "aspect_ratio", "resolution", "duration", "fit")}}
    if j["status"] == "complete" and j["result"]:
        r = dict(j["result"])
        r.pop("output_file", None)
        r["video_url"] = f"/api/jobs/{j['job_id']}/video"
        r["download_url"] = f"/api/jobs/{j['job_id']}/video?download=1"
        out["result"] = r
    return out


def _require_asset(aid: str, role: str, what: str) -> None:
    a = db.get_asset(aid)
    if a is None or not Path(a["path"]).is_file():
        raise EditorError("ASSET_MISSING", f"The {what} is no longer available. Please upload it again.")
    if a["role"] != role:
        raise EditorError("BAD_REQUEST", f"That file was not uploaded as a {what}.")


@app.post("/api/jobs")
async def create_job(body: JobIn):
    if body.mode == "prompt" and len(body.prompt.strip()) < 3:
        raise EditorError("BAD_REQUEST", "Describe the edit you want in the prompt.")
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
        raise EditorError("BAD_REQUEST", f"At most {MAX_CLIPS} clips per edit.")
    for aid in ids:
        _require_asset(aid, "clip", "video clip")
    if body.music_asset_id:
        _require_asset(body.music_asset_id, "music", "music file")
    if body.reference_asset_id:
        _require_asset(body.reference_asset_id, "reference", "reference video")
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


# ------------------------------------------------------------------ MCP + web UI
app.mount("/mcp", _mcp)
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
