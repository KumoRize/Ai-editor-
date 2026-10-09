"""Compile a validated Timeline into FFmpeg jobs.

Rendering is segment-based to keep memory bounded on small servers:

1. every timeline clip is rendered on its own to a normalised intermediate
   (H.264 at the output size/fps + PCM audio, exact frame count), cached by a
   hash of everything that affects its pixels — so "Edit again" only re-renders
   segments that changed;
2. the intermediates are joined with the concat demuxer (video stream copied,
   audio encoded to AAC once, so there are no per-segment AAC priming gaps).

No shell is involved and no user-provided text reaches a filter graph: all
numbers are formatted here and file paths are passed as separate arguments.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .timeline import ALLOWED_FPS, Clip, Timeline

RENDER_VERSION = 3


@dataclass
class SegmentJob:
    key: str
    out_path: Path
    args: list[str]
    duration: float


@dataclass
class RenderSpec:
    segments: list[SegmentJob]
    concat_list: Path
    concat_args: list[str]
    out_path: Path
    duration: float
    width: int
    height: int
    fps: int
    extra: dict = field(default_factory=dict)


def _f(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".") or "0"


def _atempo_chain(speed: float) -> list[str]:
    parts, s = [], speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-6:
        parts.append(f"atempo={_f(s)}")
    return parts


# HDR (PQ/HLG, BT.2020) → SDR BT.709 tone mapping, so phone HDR footage looks right in H.264 SDR output.
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
           "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def _video_chain(fit: str, W: int, H: int, fps: int, speed: float, n_frames: int, bg: str,
                 hdr: bool = False) -> str:
    speed_v = f"setpts=(PTS-STARTPTS)/{_f(speed)}" if speed != 1.0 else "setpts=PTS-STARTPTS"
    base = f"[0:v:0]{speed_v},fps={fps}" + (f",{TONEMAP}" if hdr else "")
    tail = f"setsar=1,format=yuv420p,tpad=stop_mode=clone:stop_duration=2,trim=end_frame={n_frames},setpts=PTS-STARTPTS[v]"
    if fit == "cover":
        return f"{base},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{tail}"
    if fit == "contain":
        return (f"{base},scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={bg},{tail}")
    # blur: blurred, cropped background + whole picture on top
    bw, bh = W // 4 * 2, H // 4 * 2
    return (f"{base},split=2[bg][fg];"
            f"[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=10:2,"
            f"scale={W}:{H},setsar=1[bgb];"
            f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,{tail}")


def segment_key(clip: Clip, asset: dict, W: int, H: int, fps: int, fit: str, bg: str,
                preset: str, crf: str, n_frames: int) -> str:
    material = {"v": RENDER_VERSION, "sha": asset.get("sha256") or str(asset["path"]),
                "in": clip.src_in, "out": clip.src_out, "speed": clip.speed, "vol": clip.volume, "n": n_frames,
                "W": W, "H": H, "fps": fps, "fit": fit, "bg": bg, "preset": preset, "crf": crf}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:24]


def compile_timeline(tl: Timeline, assets: dict[str, dict], out_path: Path, *, segment_dir: Path,
                     ffmpeg_bin: str = "ffmpeg", preview: bool = False, x264_preset: str = "veryfast",
                     threads: int = 0) -> RenderSpec:
    st = tl.settings
    W, H, fps = st.width, st.height, st.fps
    if fps not in ALLOWED_FPS:
        raise ValueError(f"fps {fps} not supported")
    if preview:  # low-res draft, shorter side 360 px
        scale = 360 / min(W, H)
        W, H = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    crf = "30" if preview else "20"
    preset = "ultrafast" if preview else x264_preset
    segment_dir.mkdir(parents=True, exist_ok=True)

    segments: list[SegmentJob] = []
    total = 0.0
    for c in tl.video_track().clips:
        asset = assets[c.asset_id]
        src_len = c.src_out - c.src_in
        # Whole number of frames per clip → audio and video stay in sync across any number of clips.
        # Frames are assigned from cumulative timeline positions so rounding never accumulates.
        n_frames = max(1, round((c.timeline_start + c.duration) * fps) - round(c.timeline_start * fps))
        out_len = n_frames / fps
        total += out_len
        key = segment_key(c, asset, W, H, fps, st.fit, st.bg_color, preset, crf, n_frames)
        seg_path = segment_dir / f"{key}.mov"

        args = [ffmpeg_bin, "-nostdin", "-hide_banner", "-y", "-v", "error",
                "-ss", _f(c.src_in), "-t", _f(src_len), "-i", str(asset["path"])]
        filters = [_video_chain(st.fit, W, H, fps, c.speed, n_frames, st.bg_color,
                                hdr=bool(asset["meta"].get("hdr")))]
        if asset["meta"].get("has_audio"):
            chain = ["asetpts=PTS-STARTPTS", "aresample=48000",
                     "aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo"]
            chain += _atempo_chain(c.speed)
            if c.volume != 1.0:
                chain.append(f"volume={_f(c.volume)}")
            chain += ["afade=t=in:d=0.02", "apad", f"atrim=end_sample={n_frames * 48000 // fps}",
                      f"afade=t=out:st={_f(max(0.0, out_len - 0.03))}:d=0.03", "asetpts=PTS-STARTPTS"]
            filters.append(f"[0:a:0]{','.join(chain)}[a]")
        else:
            args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            filters.append(f"[1:a:0]aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo,"
                           f"atrim=end_sample={n_frames * 48000 // fps},asetpts=PTS-STARTPTS[a]")
        args += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
                 "-c:v", "libx264", "-preset", preset, "-crf", crf, "-profile:v", "high",
                 "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps * 2),
                 "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                 "-c:a", "pcm_s16le", "-video_track_timescale", str(fps * 1000),
                 "-threads", str(threads), "-progress", "pipe:1", "-nostats",
                 str(seg_path.with_suffix(".part.mov"))]
        segments.append(SegmentJob(key=key, out_path=seg_path, args=args, duration=out_len))

    concat_list = out_path.parent / f".{out_path.stem}.concat.txt"
    concat_args = [ffmpeg_bin, "-nostdin", "-hide_banner", "-y", "-v", "error",
                   "-f", "concat", "-safe", "0", "-i", str(concat_list),
                   "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
                   "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                   "-movflags", "+faststart", "-progress", "pipe:1", "-nostats",
                   str(out_path.with_suffix(".part.mp4"))]
    return RenderSpec(segments=segments, concat_list=concat_list, concat_args=concat_args,
                      out_path=out_path, duration=round(total, 6), width=W, height=H, fps=fps)
