"""Server settings, read once from environment variables.

Secrets (API keys, access token) live only here, on the server. Nothing in this
module is ever sent to the browser except the explicitly public fields returned
by `public_config()` in main.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "./data")).resolve())
    access_token: str = field(default_factory=lambda: os.environ.get("ACCESS_TOKEN", ""))

    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 500))
    max_source_duration_s: float = field(default_factory=lambda: _float("MAX_SOURCE_DURATION_S", 20 * 60))

    # Retention (hours)
    upload_retention_h: float = field(default_factory=lambda: _float("UPLOAD_RETENTION_HOURS", 48))
    render_retention_h: float = field(default_factory=lambda: _float("RENDER_RETENTION_HOURS", 72))
    tmp_retention_h: float = field(default_factory=lambda: _float("TMP_RETENTION_HOURS", 6))

    # Processing
    ffmpeg_bin: str = field(default_factory=lambda: os.environ.get("FFMPEG_BIN", "ffmpeg"))
    ffprobe_bin: str = field(default_factory=lambda: os.environ.get("FFPROBE_BIN", "ffprobe"))
    render_timeout_s: int = field(default_factory=lambda: _int("RENDER_TIMEOUT_S", 30 * 60))
    analysis_timeout_s: int = field(default_factory=lambda: _int("ANALYSIS_TIMEOUT_S", 10 * 60))
    worker_threads: int = field(default_factory=lambda: _int("WORKER_THREADS", 1))
    x264_preset: str = field(default_factory=lambda: os.environ.get("X264_PRESET", "veryfast"))

    # AI planner. LLM_PROVIDER = auto | anthropic | openai | rules
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "auto").lower())
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", ""))
    llm_effort: str = field(default_factory=lambda: os.environ.get("LLM_EFFORT", "medium"))
    llm_timeout_s: int = field(default_factory=lambda: _int("LLM_TIMEOUT_S", 120))
    llm_max_calls_per_job: int = field(default_factory=lambda: _int("LLM_MAX_CALLS_PER_JOB", 3))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for sub in ("uploads", "cache", "jobs", "tmp"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings(**overrides) -> Settings:
    """Used by tests to point the app at a temporary data dir."""
    global _settings
    _settings = Settings()
    for k, v in overrides.items():
        setattr(_settings, k, v)
    _settings.ensure_dirs()
    return _settings
