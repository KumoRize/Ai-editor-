"""Background job worker. HTTP requests never block on rendering."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .agent.orchestrator import Reporter, initial_steps, load_previous_plan, run_edit_job
from .config import get_settings
from .errors import EditorError
from .storage import db
from .storage.files import cleanup_expired, job_dir

log = logging.getLogger(__name__)

_stop = threading.Event()
_wake = threading.Event()
_threads: list[threading.Thread] = []


def enqueue(job_id: str) -> None:
    _wake.set()


def _run_one(job: dict) -> None:
    job_id = job["job_id"]
    req = job["request"]
    steps = initial_steps()

    def push(st: list[dict], progress: float) -> None:
        db.update_job(job_id, steps=st, progress=progress)

    reporter = Reporter(steps, push)
    push(steps, 0.0)
    try:
        assets = {}
        for aid in req["asset_ids"]:
            a = db.get_asset(aid)
            if a is None or not Path(a["path"]).is_file():
                raise EditorError("ASSET_MISSING", "An uploaded file is no longer available. Please upload it again.")
            assets[aid] = a
        previous = None
        if req.get("parent_job_id"):
            try:
                previous = load_previous_plan(job_dir(req["parent_job_id"]))
            except EditorError:
                previous = None
        result = run_edit_job(job_id, req, assets, job_dir(job_id), reporter, previous)
        db.update_job(job_id, status="complete", progress=1.0, result=result, error=None, steps=reporter.steps)
        log.info("job %s complete", job_id)
    except EditorError as e:
        log.warning("job %s failed: %s %s %s", job_id, e.code, e.message, e.detail[:500] if e.detail else "")
        _mark_running_step_failed(reporter.steps, e.message)
        db.update_job(job_id, status="failed", error=e.to_dict(), steps=reporter.steps)
    except Exception:  # noqa: BLE001 - last line of defence, never leave a job "running"
        log.exception("job %s crashed", job_id)
        _mark_running_step_failed(reporter.steps, "Unexpected error")
        db.update_job(job_id, status="failed", steps=reporter.steps,
                      error={"code": "INTERNAL", "message": "Something went wrong on the server.", "retryable": True})


def _mark_running_step_failed(steps: list[dict], msg: str) -> None:
    for st in steps:
        if st["status"] == "running":
            st["status"] = "failed"
            st["detail"] = msg


def _worker() -> None:
    while not _stop.is_set():
        job = db.claim_next_queued()
        if job is None:
            _wake.wait(timeout=2.0)
            _wake.clear()
            continue
        _run_one(job)


def _janitor() -> None:
    while not _stop.wait(timeout=3600):
        try:
            protected: set[Path] = set()
            for j in db.jobs_with_status("queued", "running"):
                protected.add(job_dir(j["job_id"]))
                for aid in j["request"].get("asset_ids", []):
                    a = db.get_asset(aid)
                    if a:
                        protected.add(Path(a["path"]))
            log.info("cleanup: %s", cleanup_expired(protected))
        except Exception:  # noqa: BLE001
            log.exception("cleanup failed")


def recover_interrupted() -> None:
    """Jobs left 'running' by a crash/restart are failed honestly (and can be retried)."""
    for j in db.jobs_with_status("running"):
        db.update_job(j["job_id"], status="failed", error={
            "code": "INTERRUPTED", "message": "The server restarted while this edit was running.", "retryable": True})


def start_workers() -> None:
    if _threads:
        return
    _stop.clear()
    recover_interrupted()
    for i in range(max(1, get_settings().worker_threads)):
        t = threading.Thread(target=_worker, name=f"worker-{i}", daemon=True)
        t.start()
        _threads.append(t)
    t = threading.Thread(target=_janitor, name="janitor", daemon=True)
    t.start()
    _threads.append(t)


def stop_workers() -> None:
    _stop.set()
    _wake.set()
    for t in _threads:
        t.join(timeout=5)
    _threads.clear()


def wait_for(job_id: str, timeout: float = 120) -> dict:
    """Test helper: block until a job leaves queued/running."""
    end = time.time() + timeout
    while time.time() < end:
        j = db.get_job(job_id)
        if j and j["status"] in ("complete", "failed"):
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)
