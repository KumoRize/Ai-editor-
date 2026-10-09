"""Tiny SQLite metadata store for assets and jobs (WAL mode, thread-safe)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

from ..config import get_settings

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    original_name TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    meta TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    request TEXT NOT NULL,
    steps TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    result TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    asset_ids TEXT NOT NULL,
    timeline TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


def conn() -> sqlite3.Connection:
    global _conn, _conn_path
    path = str(get_settings().data_dir / "editor.db")
    with _lock:
        if _conn is None or _conn_path != path:
            _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.executescript(SCHEMA)
            _conn_path = path
        return _conn


def _row(r: sqlite3.Row | None, json_fields: tuple[str, ...]) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in json_fields:
        if d.get(k) is not None:
            d[k] = json.loads(d[k])
    return d


# ---------------------------------------------------------------- assets
def insert_asset(a: dict) -> None:
    with _lock:
        conn().execute(
            "INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)",
            (a["asset_id"], a["role"], a["original_name"], a["path"], a["sha256"],
             a["size"], json.dumps(a["meta"]), time.time()),
        )


def get_asset(asset_id: str) -> dict | None:
    with _lock:
        r = conn().execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
    return _row(r, ("meta",))


def find_asset_by_sha(sha256: str, role: str) -> dict | None:
    with _lock:
        r = conn().execute(
            "SELECT * FROM assets WHERE sha256=? AND role=? ORDER BY created_at DESC LIMIT 1",
            (sha256, role),
        ).fetchone()
    return _row(r, ("meta",))


# ---------------------------------------------------------------- jobs
def insert_job(job_id: str, request: dict, steps: list[dict]) -> None:
    now = time.time()
    with _lock:
        conn().execute(
            "INSERT INTO jobs (job_id,status,request,steps,progress,created_at,updated_at) "
            "VALUES (?,?,?,?,0,?,?)",
            (job_id, "queued", json.dumps(request), json.dumps(steps), now, now),
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols, vals = [], []
    for k, v in fields.items():
        if k in ("request", "steps", "error", "result") and v is not None:
            v = json.dumps(v)
        cols.append(f"{k}=?")
        vals.append(v)
    cols.append("updated_at=?")
    vals.append(time.time())
    vals.append(job_id)
    with _lock:
        conn().execute(f"UPDATE jobs SET {', '.join(cols)} WHERE job_id=?", vals)


def get_job(job_id: str) -> dict | None:
    with _lock:
        r = conn().execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _row(r, ("request", "steps", "error", "result"))


def jobs_with_status(*statuses: str) -> list[dict]:
    q = ",".join("?" * len(statuses))
    with _lock:
        rows = conn().execute(
            f"SELECT * FROM jobs WHERE status IN ({q}) ORDER BY created_at", statuses
        ).fetchall()
    return [_row(r, ("request", "steps", "error", "result")) for r in rows]


def claim_next_queued() -> dict | None:
    """Atomically move the oldest queued job to running and return it."""
    with _lock:
        c = conn()
        r = c.execute("SELECT job_id FROM jobs WHERE status='queued' ORDER BY updated_at LIMIT 1").fetchone()
        if r is None:
            return None
        c.execute(
            "UPDATE jobs SET status='running', attempts=attempts+1, updated_at=? WHERE job_id=? AND status='queued'",
            (time.time(), r["job_id"]),
        )
        return get_job(r["job_id"])


def list_assets(limit: int = 50) -> list[dict]:
    with _lock:
        rows = conn().execute("SELECT * FROM assets ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r, ("meta",)) for r in rows]


# ---------------------------------------------------------------- projects (MCP / agent editing sessions)
def insert_project(project_id: str, asset_ids: list[str]) -> None:
    now = time.time()
    with _lock:
        conn().execute("INSERT INTO projects VALUES (?,?,?,?,?)", (project_id, json.dumps(asset_ids), None, now, now))


def get_project(project_id: str) -> dict | None:
    with _lock:
        r = conn().execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
    return _row(r, ("asset_ids", "timeline"))


def update_project(project_id: str, *, timeline: dict | None = None, asset_ids: list[str] | None = None) -> None:
    sets, vals = ["updated_at=?"], [time.time()]
    if timeline is not None:
        sets.append("timeline=?")
        vals.append(json.dumps(timeline))
    if asset_ids is not None:
        sets.append("asset_ids=?")
        vals.append(json.dumps(asset_ids))
    vals.append(project_id)
    with _lock:
        conn().execute(f"UPDATE projects SET {', '.join(sets)} WHERE project_id=?", vals)
