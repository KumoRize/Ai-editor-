"""Safe subprocess helpers. Commands are always argument lists (never a shell)."""
from __future__ import annotations

import logging
import subprocess
from typing import Callable, Sequence

from .errors import EditorError

log = logging.getLogger(__name__)


def run(cmd: Sequence[str], *, timeout: float, input_bytes: bytes | None = None,
        capture_stdout: bool = True) -> subprocess.CompletedProcess:
    if not cmd or not isinstance(cmd, (list, tuple)):
        raise ValueError("command must be a non-empty argument list")
    log.debug("run: %s", " ".join(cmd))
    try:
        return subprocess.run(
            list(cmd),
            input=input_bytes,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            stdin=None if input_bytes is not None else subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        raise EditorError("TIMEOUT", "Processing took too long and was stopped.", retryable=True,
                          detail=f"timeout after {timeout}s: {cmd[0]}") from e
    except FileNotFoundError as e:
        raise EditorError("TOOL_MISSING", "The processing server is misconfigured (missing FFmpeg).",
                          detail=str(e)) from e


def run_streaming(cmd: Sequence[str], *, timeout: float,
                  on_line: Callable[[str], None]) -> tuple[int, str]:
    """Run a command, feeding each stdout line to on_line. Returns (returncode, stderr_tail)."""
    import threading
    import time
    from collections import deque

    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, text=True, bufsize=1)
    tail: deque[str] = deque(maxlen=40)

    def drain_stderr():
        assert proc.stderr is not None
        for line in proc.stderr:
            tail.append(line.rstrip())

    t = threading.Thread(target=drain_stderr, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            on_line(line.rstrip())
            if time.monotonic() > deadline:
                proc.kill()
                raise EditorError("TIMEOUT", "Rendering took too long and was stopped.", retryable=True)
        rc = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as e:
        proc.kill()
        raise EditorError("TIMEOUT", "Rendering took too long and was stopped.", retryable=True) from e
    finally:
        if proc.poll() is None:
            proc.kill()
        t.join(timeout=2)
    return rc, "\n".join(tail)
