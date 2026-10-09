"""Compile a validated Timeline into FFmpeg jobs.

Rendering is piece-based to keep memory bounded on small servers:

1. The main track is cut into *pieces*: the body of each clip, plus one
   transition piece per transition (tail of clip A blended into the head of
   clip B with `xfade` / `acrossfade`). Each piece is rendered on its own to a
   normalised intermediate (H.264 at the output size/fps + PCM audio, exact frame
   count), with per-clip effects (speed, zoom keyframes, colour, effects, crop,
   rotate, HDR tone-mapping) and timeline-level layers that fall inside it
   (captions/text via libass, picture-in-picture overlays, intro/outro fades).
   Pieces are cached by a hash of their full recipe, so "Edit again" only
   re-renders what changed.
2. A final pass joins the pieces with the concat demuxer (video stream copied)
   and builds the soundtrack once: original audio + music (looped, faded, ducked
   under speech) + sound effects, loudness-normalised, encoded to AAC.

No shell is involved and no user-provided text reaches a filter graph: text goes
into an ASS file, numbers are formatted here, paths are separate arguments.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .captions import build_ass
from .timeline import ALLOWED_FPS, Clip, Keyframe, Timeline

RENDER_VERSION = 4
SR = 48000

# HDR (PQ/HLG, BT.2020) → SDR BT.709 tone mapping, so phone HDR footage looks right in H.264 SDR output.
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
           "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")

COLOR_PRESET_FILTERS = {
    "warm": "eq=saturation=1.08,colorbalance=rm=0.07:bm=-0.07:rh=0.05:bh=-0.05",
    "cool": "colorbalance=rm=-0.05:bm=0.07:rh=-0.03:bh=0.05",
    "cinematic": "eq=contrast=1.12:saturation=0.9,colorbalance=rs=-0.05:bs=0.08:rh=0.08:bh=-0.06",
    "bw": "hue=s=0,eq=contrast=1.1",
    "vintage": "curves=preset=vintage,eq=saturation=0.85",
    "vibrant": "eq=saturation=1.35:contrast=1.05",
    "moody": "eq=brightness=-0.04:contrast=1.15:saturation=0.75",
}
EFFECT_FILTERS = {
    "vignette": "vignette=angle=PI/5",
    "grain": "noise=alls=10:allf=t",
    "blur": "gblur=sigma=6",
    "sharpen": "unsharp=5:5:0.8",
    "bw": "hue=s=0",
    "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
}


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


def escape_filter_path(p: Path) -> str:
    s = str(p)
    if "'" in s or "\n" in s:
        raise ValueError(f"unsafe path for a filter graph: {s!r}")
    return "'" + s.replace("\\", "\\\\").replace(":", "\\:") + "'"


# ----------------------------------------------------------------------------- zoom
def _zoom_exprs(kfs: list[Keyframe], ease: str, local_a: float, clip_dur: float, fps: int) -> tuple[str, str, str]:
    """zoompan z/x/y expressions over output frame number `on` for a clip portion."""
    pts = sorted(kfs, key=lambda k: k.at)
    u = f"(({_f(local_a)}+on/{fps})/{_f(max(clip_dur, 1e-6))})"

    def curve(attr: str) -> str:
        expr = _f(getattr(pts[-1], attr))
        for k0, k1 in reversed(list(zip(pts, pts[1:]))):
            v0, v1 = getattr(k0, attr), getattr(k1, attr)
            span = max(k1.at - k0.at, 1e-6)
            p = f"clip(({u}-{_f(k0.at)})/{_f(span)},0,1)"
            if ease == "inout":
                p = f"({p}*{p}*(3-2*{p}))"
            seg = f"({_f(v0)}+({_f(v1 - v0)})*{p})"
            expr = f"if(lt({u},{_f(k1.at)}),{seg},{expr})"
        return f"if(lt({u},{_f(pts[0].at)}),{_f(getattr(pts[0], attr))},{expr})"

    z = curve("scale")
    x = f"clip(({curve('x')})*iw-iw/zoom/2,0,iw-iw/zoom)"
    y = f"clip(({curve('y')})*ih-ih/zoom/2,0,ih-ih/zoom)"
    return z, x, y


# ----------------------------------------------------------------------------- per-clip streams
def _portion(c: Clip, asset: dict, local_a: float, local_b: float, n_frames: int, in_v: int, in_a: int | None,
             tag: str, st, W: int, H: int, fps: int, declick_in: bool, declick_out: bool) -> list[str]:
    """Filter chains turning one input into [v{tag}] and [a{tag}] for clip-local time [local_a, local_b]."""
    meta = asset["meta"]
    speed_v = f"setpts=(PTS-STARTPTS)/{_f(c.speed)}" if c.speed != 1.0 else "setpts=PTS-STARTPTS"
    chain = [speed_v, f"fps={fps}"]
    if meta.get("hdr"):
        chain.append(TONEMAP)
    if c.rotate == 90:
        chain.append("transpose=1")
    elif c.rotate == 270:
        chain.append("transpose=2")
    elif c.rotate == 180:
        chain.append("hflip,vflip")
    if c.crop:
        k = c.crop
        chain.append(f"crop=trunc(iw*{_f(k.w)}/2)*2:trunc(ih*{_f(k.h)}/2)*2:iw*{_f(k.x)}:ih*{_f(k.y)}")
    filters: list[str] = []
    src = f"[{in_v}:v:0]"
    if st.fit == "cover":
        chain.append(f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}")
    elif st.fit == "contain":
        chain.append(f"scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                     f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={st.bg_color}")
    else:  # blur
        bw, bh = W // 4 * 2, H // 4 * 2
        filters.append(f"{src}{','.join(chain)},split=2[bg{tag}][fg{tag}]")
        filters.append(f"[bg{tag}]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                       f"boxblur=10:2,scale={W}:{H},setsar=1[bgb{tag}]")
        filters.append(f"[fg{tag}]scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                       f"setsar=1[fgs{tag}]")
        filters.append(f"[bgb{tag}][fgs{tag}]overlay=(W-w)/2:(H-h)/2[fit{tag}]")
        src, chain = f"[fit{tag}]", []
    chain.append("setsar=1")
    if c.keyframes:
        z, x, y = _zoom_exprs(c.keyframes, c.ease, local_a, c.duration, fps)
        chain.append(f"scale={W * 2}:{H * 2}:flags=bicubic,"
                     f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={fps}")
    if c.color:
        col = c.color
        if col.preset in COLOR_PRESET_FILTERS:
            chain.append(COLOR_PRESET_FILTERS[col.preset])
        if (col.brightness, col.contrast, col.saturation, col.gamma) != (0.0, 1.0, 1.0, 1.0):
            chain.append(f"eq=brightness={_f(col.brightness)}:contrast={_f(col.contrast)}:"
                         f"saturation={_f(col.saturation)}:gamma={_f(col.gamma)}")
        if col.warmth:
            w = col.warmth * 0.1
            chain.append(f"colorbalance=rm={_f(w)}:bm={_f(-w)}:rh={_f(w * 0.6)}:bh={_f(-w * 0.6)}")
    for e in c.effects:
        chain.append(EFFECT_FILTERS[e])
    chain += ["format=yuv420p", "tpad=stop_mode=clone:stop_duration=2", f"trim=end_frame={n_frames}",
              "setpts=PTS-STARTPTS"]
    filters.append(f"{src}{','.join(chain)}[v{tag}]")

    n_samples = n_frames * SR // fps
    out_len = n_frames / fps
    if in_a is None:
        achain = ["asetpts=PTS-STARTPTS", f"aresample={SR}",
                  f"aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo"]
        achain += _atempo_chain(c.speed)
        if c.volume != 1.0:
            achain.append(f"volume={_f(c.volume)}")
        if declick_in:
            achain.append("afade=t=in:d=0.02")
        achain += ["apad", f"atrim=end_sample={n_samples}"]
        if declick_out:
            achain.append(f"afade=t=out:st={_f(max(0.0, out_len - 0.03))}:d=0.03")
        achain.append("asetpts=PTS-STARTPTS")
        filters.append(f"[{in_v}:a:0]{','.join(achain)}[a{tag}]")
    else:
        filters.append(f"[{in_a}:a:0]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
                       f"atrim=end_sample={n_samples},asetpts=PTS-STARTPTS[a{tag}]")
    return filters


def _input_args(c: Clip, asset: dict, local_a: float, local_b: float) -> list[str]:
    dur = float(asset["meta"]["duration"])
    ss = c.src_in + local_a * c.speed
    t = min((local_b - local_a) * c.speed + 0.1, max(0.05, dur - ss + 0.05))  # small pad, trimmed exactly later
    return ["-ss", _f(ss), "-t", _f(t), "-i", str(asset["path"])]


# ----------------------------------------------------------------------------- pieces
@dataclass
class Piece:
    kind: str                     # "body" | "transition"
    t0: float                     # timeline start
    t1: float                     # timeline end
    clip: Clip
    local_a: float = 0.0
    local_b: float = 0.0
    prev: Clip | None = None      # transition: outgoing clip
    trans_type: str = ""
    trans_d: float = 0.0


def plan_pieces(tl: Timeline) -> list[Piece]:
    clips = tl.video_track().clips
    pieces: list[Piece] = []
    for i, c in enumerate(clips):
        d_in = c.transition_in.duration if i > 0 and c.transition_in else 0.0
        nxt = clips[i + 1] if i + 1 < len(clips) else None
        d_out = nxt.transition_in.duration if nxt and nxt.transition_in else 0.0
        if d_in:
            prev = clips[i - 1]
            pieces.append(Piece("transition", c.timeline_start, c.timeline_start + d_in, c, 0.0, d_in, prev,
                                c.transition_in.type, d_in))
        a, b = d_in, c.duration - d_out
        if b - a > 1e-6:
            pieces.append(Piece("body", c.timeline_start + a, c.timeline_start + b, c, a, b))
    return pieces


def _frames(t: float, fps: int) -> int:
    return round(t * fps + 1e-9)


def compile_timeline(tl: Timeline, assets: dict[str, dict], out_path: Path, *, segment_dir: Path,
                     ffmpeg_bin: str = "ffmpeg", preview: bool = False, x264_preset: str = "veryfast",
                     threads: int = 0, sfx_paths: dict[str, Path] | None = None) -> RenderSpec:
    st = tl.settings
    fps = st.fps
    if fps not in ALLOWED_FPS:
        raise ValueError(f"fps {fps} not supported")
    W, H = st.width, st.height
    if preview:  # low-res draft, shorter side 360 px
        scale = 360 / min(W, H)
        W, H = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    crf = "30" if preview else "20"
    preset = "ultrafast" if preview else x264_preset
    segment_dir.mkdir(parents=True, exist_ok=True)

    ass_path = out_path.parent / f".{out_path.stem}.captions.ass"
    if tl.texts:
        ass_path.write_text(build_ass(tl.texts, W, H), encoding="utf-8")

    total_frames = _frames(tl.duration, fps)
    pieces = plan_pieces(tl)
    segments: list[SegmentJob] = []
    for p in pieces:
        f0, f1 = _frames(p.t0, fps), _frames(p.t1, fps)
        n_frames = max(1, f1 - f0)
        T0 = f0 / fps
        length = n_frames / fps
        args = [ffmpeg_bin, "-nostdin", "-hide_banner", "-y", "-v", "error"]
        filters: list[str] = []
        recipe: dict = {"v": RENDER_VERSION, "kind": p.kind, "W": W, "H": H, "fps": fps, "n": n_frames,
                        "fit": st.fit, "bg": st.bg_color, "preset": preset, "crf": crf}
        idx = 0

        def add_portion(c: Clip, la: float, lb: float, tag: str, d_in: bool, d_out: bool) -> None:
            nonlocal idx
            asset = assets[c.asset_id]
            args.extend(_input_args(c, asset, la, lb))
            vi = idx
            idx += 1
            ai = None
            if not asset["meta"].get("has_audio"):
                args.extend(["-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo"])
                ai = idx
                idx += 1
            filters.extend(_portion(c, asset, la, lb, n_frames, vi, ai, tag, st, W, H, fps, d_in, d_out))
            recipe.setdefault("portions", []).append({
                "sha": asset.get("sha256") or str(asset["path"]), "la": round(la, 6), "lb": round(lb, 6),
                "clip": c.model_dump(exclude={"clip_id", "timeline_start", "note", "transition_in"}),
                "has_audio": bool(asset["meta"].get("has_audio")), "hdr": bool(asset["meta"].get("hdr")),
                "declick": [d_in, d_out]})

        clip_is_first = p.clip is tl.video_track().clips[0]
        if p.kind == "body":
            ends_clip = abs(p.local_b - p.clip.duration) < 1e-6
            add_portion(p.clip, p.local_a, p.local_b, "0", p.local_a == 0.0, ends_clip)
            vlab, alab = "[v0]", "[a0]"
        else:
            a = p.prev
            add_portion(a, a.duration - p.trans_d, a.duration, "0", False, False)
            add_portion(p.clip, 0.0, p.trans_d, "1", False, False)
            xt = "fade" if p.trans_type == "dissolve" else p.trans_type
            filters.append(f"[v0][v1]xfade=transition={xt}:duration={_f(length)}:offset=0,"
                           f"trim=end_frame={n_frames},setpts=PTS-STARTPTS[vx]")
            filters.append(f"[a0][a1]acrossfade=d={_f(length)}:c1=tri:c2=tri,apad,"
                           f"atrim=end_sample={n_frames * SR // fps},asetpts=PTS-STARTPTS[ax]")
            vlab, alab = "[vx]", "[ax]"
            recipe["transition"] = p.trans_type
        del clip_is_first

        # Timeline-level layers that fall inside this piece.
        post: list[str] = []
        ovs = [o for o in tl.overlays if o.start < T0 + length and o.start + o.duration > T0]
        for k, o in enumerate(ovs):
            oa = assets[o.asset_id]
            skip = max(0.0, T0 - o.start)
            offset = max(0.0, o.start - T0)
            dur = min(o.duration - skip, length - offset)
            if dur <= 0:
                continue
            args.extend(["-ss", _f(o.src_in + skip), "-t", _f(dur + 0.1), "-i", str(oa["path"])])
            oi = idx
            idx += 1
            ow = int(W * o.width) // 2 * 2
            filters.append(f"[{oi}:v:0]setpts=PTS-STARTPTS,fps={fps},"
                           f"{TONEMAP + ',' if oa['meta'].get('hdr') else ''}"
                           f"scale={ow}:-2,pad=iw+8:ih+8:4:4:color=white,format=yuv420p,"
                           f"setpts=PTS+{_f(offset)}/TB[ov{k}]")
            nv = f"[vo{k}]"
            filters.append(f"{vlab}[ov{k}]overlay=x={int(W * o.x)}:y={int(H * o.y)}:eof_action=pass:"
                           f"enable='between(t,{_f(offset)},{_f(offset + dur)})'{nv}")
            vlab = nv
            recipe.setdefault("overlays", []).append({**o.model_dump(exclude={"item_id", "track_id"}),
                                                      "sha": oa.get("sha256") or str(oa["path"])})
        texts = [t for t in tl.texts if t.start < T0 + length and t.end > T0]
        if texts:
            post.append(f"setpts=PTS+{_f(T0)}/TB,ass=filename={escape_filter_path(ass_path)},setpts=PTS-STARTPTS")
            recipe["texts"] = [t.model_dump(exclude={"item_id"}) for t in texts]
        if st.intro and f0 == 0:
            d = min(st.intro.duration, length)
            color = "white" if st.intro.type == "fadewhite" else "black"
            post.append(f"fade=t=in:st=0:d={_f(d)}:color={color}")
            recipe["intro"] = [st.intro.type, d]
        if st.outro and f1 >= total_frames:
            d = min(st.outro.duration, length)
            color = "white" if st.outro.type == "fadewhite" else "black"
            post.append(f"fade=t=out:st={_f(length - d)}:d={_f(d)}:color={color}")
            recipe["outro"] = [st.outro.type, d]
        if post:
            filters.append(f"{vlab}{','.join(post)}[vfinal]")
            vlab = "[vfinal]"
        key = hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()[:32]
        seg_path = segment_dir / f"{key}.mov"
        args += ["-filter_complex", ";".join(filters), "-map", vlab, "-map", alab,
                 "-c:v", "libx264", "-preset", preset, "-crf", crf, "-profile:v", "high",
                 "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps * 2),
                 "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                 "-c:a", "pcm_s16le", "-ar", str(SR), "-ac", "2", "-video_track_timescale", str(fps * 1000),
                 "-threads", str(threads), "-progress", "pipe:1", "-nostats",
                 str(seg_path.with_suffix(f".{out_path.stem}.part.mov"))]
        segments.append(SegmentJob(key=key, out_path=seg_path, args=args, duration=length))

    duration = sum(sg.duration for sg in segments)
    concat_list = out_path.parent / f".{out_path.stem}.concat.txt"
    concat_args = _final_pass_args(tl, assets, concat_list, out_path, duration, ffmpeg_bin, sfx_paths or {})
    return RenderSpec(segments=segments, concat_list=concat_list, concat_args=concat_args,
                      out_path=out_path, duration=round(duration, 6), width=W, height=H, fps=fps)


def _final_pass_args(tl: Timeline, assets: dict[str, dict], concat_list: Path, out_path: Path, total: float,
                     ffmpeg_bin: str, sfx_paths: dict[str, Path]) -> list[str]:
    """Join pieces (video copied) and build the final soundtrack."""
    st = tl.settings
    args = [ffmpeg_bin, "-nostdin", "-hide_banner", "-y", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list)]
    fmt = f"aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo"
    music = [a for a in tl.audio if a.kind == "music"]
    sfx = [a for a in tl.audio if a.kind == "sfx"]
    ducked = [m for m in music if m.duck]
    filters = []
    n_split = 1 + len(ducked)
    filters.append(f"[0:a:0]{fmt}" + (f",asplit={n_split}" if n_split > 1 else "") +
                   "".join(f"[main{i}]" for i in range(n_split)))
    mix = ["[main0]"]
    idx = 1
    for k, m in enumerate(music):
        a = assets[m.asset_id]
        if m.loop:
            args += ["-stream_loop", "-1"]
        args += ["-i", str(a["path"])]
        L = (m.duration if m.duration else total - m.start)
        L = max(0.05, min(L, total - m.start))
        ch = [fmt, f"atrim=start={_f(m.src_in)}", "asetpts=PTS-STARTPTS", f"atrim=duration={_f(L)}",
              f"volume={_f(m.volume)}"]
        if m.fade_in:
            ch.append(f"afade=t=in:d={_f(min(m.fade_in, L / 2))}")
        if m.fade_out:
            fo = min(m.fade_out, L / 2)
            ch.append(f"afade=t=out:st={_f(L - fo)}:d={_f(fo)}")
        if m.start > 0:
            ms = int(round(m.start * 1000))
            ch.append(f"adelay={ms}|{ms}")
        filters.append(f"[{idx}:a:0]{','.join(ch)}[mus{k}]")
        if m.duck:
            j = ducked.index(m) + 1
            filters.append(f"[mus{k}][main{j}]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=400:"
                           f"makeup=1[musd{k}]")
            mix.append(f"[musd{k}]")
        else:
            mix.append(f"[mus{k}]")
        idx += 1
    for k, s in enumerate(sfx):
        path = sfx_paths[s.sfx] if s.sfx else Path(assets[s.asset_id]["path"])
        args += ["-i", str(path)]
        ms = int(round(s.start * 1000))
        filters.append(f"[{idx}:a:0]{fmt},volume={_f(s.volume)},adelay={ms}|{ms}[sfx{k}]")
        mix.append(f"[sfx{k}]")
        idx += 1
    chain = []
    if len(mix) > 1:
        chain.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first:dropout_transition=0")
    else:
        chain.append("[main0]anull")
    tail = ["apad", f"atrim=duration={_f(total)}"]
    if st.intro:
        tail.append(f"afade=t=in:d={_f(min(st.intro.duration, total / 2))}")
    if st.outro:
        d = min(st.outro.duration, total / 2)
        tail.append(f"afade=t=out:st={_f(total - d)}:d={_f(d)}")
    if st.normalize_audio:
        tail += ["loudnorm=I=-14:TP=-1.5:LRA=11", f"aresample={SR}", "alimiter=limit=0.95:level=false"]
    tail.append(f"atrim=duration={_f(total)}")
    filters.append(chain[0] + "," + ",".join(tail) + "[aout]")
    args += ["-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2",
             "-movflags", "+faststart", "-progress", "pipe:1", "-nostats",
             str(out_path.with_suffix(".part.mp4"))]
    return args
