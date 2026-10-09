"""Typed, user-presentable errors shared by every component."""
from __future__ import annotations


class EditorError(Exception):
    """An error with a stable code and a message that is safe to show the user."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, detail: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.detail = detail  # internal detail for logs, never shown verbatim to users

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


# Convenience constructors for the messages promised in the architecture doc.
def unsupported_format(what: str = "Your video format isn't supported.") -> EditorError:
    return EditorError("UNSUPPORTED_FORMAT", what)


def too_large(limit_mb: int) -> EditorError:
    return EditorError("FILE_TOO_LARGE", f"The uploaded file is too large (limit {limit_mb} MB).")


def invalid_timeline(msg: str) -> EditorError:
    return EditorError("INVALID_TIMELINE", f"The timeline contains an invalid clip: {msg}")


def render_failed(detail: str = "") -> EditorError:
    return EditorError("RENDER_FAILED", "The rendering server failed.", retryable=True, detail=detail)
