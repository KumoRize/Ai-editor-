import subprocess
from pathlib import Path

import pytest

from app import config


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("media")
    out = {}
    # 12 s landscape, 3 scenes, tone with a silent stretch from 5-7 s.
    out["land"] = d / "land.mov"
    ff("-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=4", "-f", "lavfi", "-i", "smptebars=s=1280x720:r=30:d=4",
       "-f", "lavfi", "-i", "mandelbrot=s=1280x720:r=30", "-f", "lavfi", "-i", "sine=f=440:d=12",
       "-filter_complex", "[2:v]trim=0:4,setpts=PTS-STARTPTS[m];[0:v][1:v][m]concat=n=3:v=1:a=0[v];"
       "[3:a]volume='if(between(t,5,7),0,1)':eval=frame[a]",
       "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
       str(out["land"]))
    # 6 s portrait phone clip stored landscape with a 90° rotation flag.
    out["rotated"] = d / "rotated.mov"
    ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=6", "-f", "lavfi", "-i", "sine=f=300:d=6",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "unrotated.mov"))
    ff("-display_rotation", "90", "-i", str(d / "unrotated.mov"), "-c", "copy", str(out["rotated"]))
    # 5 s clip with no audio track.
    out["silent"] = d / "silent.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=s=640x480:r=25:d=5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       str(out["silent"]))
    # 4 s WEBM.
    out["webm"] = d / "clip.webm"
    ff("-f", "lavfi", "-i", "testsrc2=s=480x270:r=30:d=4", "-f", "lavfi", "-i", "sine=d=4",
       "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus", str(out["webm"]))
    # Garbage with a video extension.
    out["fake"] = d / "fake.mp4"
    out["fake"].write_bytes(b"this is not a video" * 1000)
    return out


@pytest.fixture(autouse=True)
def no_speech_model(monkeypatch):
    """Tests never download a speech model; tests that need speech install a fake transcriber."""
    from app.analysis import transcribe
    monkeypatch.setenv("TRANSCRIBE", "0")
    monkeypatch.setattr(transcribe, "_model", None)
    monkeypatch.setattr(transcribe, "_model_error", None)
    monkeypatch.setattr(transcribe, "_override", None)


@pytest.fixture()
def settings(tmp_path):
    s = config.reset_settings(data_dir=tmp_path / "data", access_token="", llm_provider="rules")
    from app.storage import db
    db._conn = None
    yield s
    db._conn = None


def make_asset(path: Path, role: str = "clip", asset_id: str = "a_1") -> dict:
    import hashlib

    from app.analysis.media import get_video_metadata

    return {"asset_id": asset_id, "role": role, "original_name": path.name, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size,
            "meta": get_video_metadata(path)}
