"""Social-media output presets and output-size resolution."""
from __future__ import annotations

from .errors import EditorError

PRESETS: dict[str, dict] = {
    "instagram_reels": {"label": "Instagram Reels", "aspect_ratio": "9:16", "default_duration": 30, "max_duration": 90},
    "youtube_shorts": {"label": "YouTube Shorts", "aspect_ratio": "9:16", "default_duration": 30, "max_duration": 180},
    "tiktok": {"label": "TikTok", "aspect_ratio": "9:16", "default_duration": 30, "max_duration": 600},
    "youtube_landscape": {"label": "YouTube (landscape)", "aspect_ratio": "16:9", "default_duration": 60, "max_duration": 900},
    "square": {"label": "Square post", "aspect_ratio": "1:1", "default_duration": 30, "max_duration": 600},
}

SIZES = {
    ("9:16", "1080p"): (1080, 1920), ("9:16", "720p"): (720, 1280),
    ("16:9", "1080p"): (1920, 1080), ("16:9", "720p"): (1280, 720),
    ("1:1", "1080p"): (1080, 1080), ("1:1", "720p"): (720, 720),
}

FITS = {
    "cover": "Fill frame (crop edges)",
    "blur": "Fit with blurred background",
    "contain": "Fit with black bars",
}


def output_size(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    try:
        return SIZES[(aspect_ratio, resolution)]
    except KeyError:
        raise EditorError("BAD_REQUEST", f"Unsupported output format {aspect_ratio} at {resolution}.") from None
