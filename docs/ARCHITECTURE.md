# AI Video Editor Agent — Architecture

Status: **design approved for Phase 1** · No application code yet.

This document fixes the architecture, the editing-engine choice, the data
contracts (timeline / edit plan), the tool surface the AI model uses, and the
phased delivery plan. Code written in later phases must conform to it; changes
to it should be made here first.

---

## 1. Requirements that drive the design

| Requirement | Consequence |
|---|---|
| User is on a **phone**, no powerful PC | All decoding, analysis and rendering run **server-side**. The phone only uploads, watches progress, previews and downloads. Nothing heavy runs in the mobile browser. |
| Agent must **actually edit** | The model never produces video. It produces **validated tool calls** that mutate a server-side timeline, which a real engine renders to an MP4 that is then probed and checked. |
| Model must be **replaceable** | The agent talks to an `LLMProvider` interface using a neutral JSON-Schema tool format. Claude is one adapter; OpenAI-compatible APIs (incl. local Ollama/vLLM) are another; a deterministic rule-based planner is a third (works with no API key). |
| **Free / open-source first**, cost control | FFmpeg + Python analysis libraries. LLM calls are few, small, and cached: the model sees compact JSON summaries and a handful of low-res thumbnails, never whole videos. |
| **Never fake success** | Every operation returns a typed `ok / error` result. "Export succeeded" is only reported after the output file exists, is non-empty, and `ffprobe` confirms codec, resolution, and duration. |
| Safety against bugs | Plan → validate → execute (transactional) → validate timeline → preview → verify → export. |

---

## 2. Editing-engine evaluation

The engine must be: (1) headless and scriptable from a server, (2) deterministic
and controllable by an external agent, (3) runnable on a cheap Linux CPU box,
(4) able to output H.264/AAC MP4 at 9:16, 16:9, 1:1, (5) licence-compatible with
a hosted service, (6) independent of what the user's phone can do.

| Candidate | What it is | Headless server control | Mobile fit | Licence / cost | Verdict |
|---|---|---|---|---|---|
| **FFmpeg** | The de-facto media toolkit (decode, filter graph, encode). | Excellent — a CLI / filter graph, fully scriptable, used by virtually every video service. | Runs on the server; phone just uses a browser. | LGPL/GPL, free. | **Selected as the render/execution engine.** |
| **MLT (melt)** | Timeline framework behind Kdenlive/Shotcut; XML project → render. | Good (CLI + XML). | Server-side, fine. | LGPL/GPL, free. | Strong runner-up. Kept as a possible **second engine adapter** if our own filter-graph compiler becomes too complex for transitions. |
| **Remotion** | React components rendered to video via headless Chromium. | Good, but every render boots Chromium — heavy RAM/CPU on a small VPS. | Server-side OK. | **Source-available, not OSS**: free only for individuals / companies ≤ 3 people; paid Company License otherwise. | Not the core engine. Possible later, optional, for fancy motion-graphics/caption templates only. |
| **OpenReel** (Augani/openreel-video) | CapCut-like editor running **in the browser** (React, WebCodecs, WebGPU). | No headless/server API; designed for a human at a UI. | Processing happens **on the user's device** — exactly what we must avoid on a phone (memory, thermal, iOS Safari WebCodecs/WebGPU limits, large files). | MIT. | **Rejected** as engine. Its web UI also isn't agent-controlled. Could inspire the "manual tweak" UI later. |
| **OpenChatCut** (0xsline/OpenChatCut) | Local-first AI-agent desktop video editor with a real timeline; exposes its project tools over MCP. | MCP tools exist, but it is a **desktop app** (Windows/macOS builds) meant to run on the user's computer. | Requires a desktop the user does not have; running a desktop GUI app on a server per user is fragile. | **AGPL** — hosting a modified version obliges releasing source to users. | **Rejected** as engine. Its design (agent and humans editing the *same* real timeline through shared tools) validates our approach and we adopt the same idea. |
| MoviePy / GES / others | Python wrapper around FFmpeg / GStreamer Editing Services. | OK. | Server-side. | Free. | MoviePy is slow (per-frame Python); GES has a smaller ecosystem. Not needed on top of FFmpeg. |

### Decision

**FFmpeg is the execution and rendering engine, driven by our own typed
timeline model.** The agent never writes FFmpeg commands. It edits a JSON
timeline through validated tools; a *timeline compiler* turns that timeline
into an FFmpeg filter graph with an **argument list** (no shell, no
user-supplied strings in filter expressions except escaped caption text written
to an `.ass` file).

Why this is the best fit:

1. **Mobile-first by construction** — all work is server-side; the phone needs
   only a browser.
2. **Covers every required operation natively**: `trim`, `setpts`/`atempo`
   (speed, speed ramps via segment splitting), `crop`/`scale`/`pad`
   (aspect ratios), `zoompan` or animated `crop`+`scale` (zooms/keyframes),
   `xfade` (transitions), `eq`/`lut3d` (colour), `ass`/`subtitles` via libass
   (styled captions), `drawtext`, `amix`/`volume`/`sidechaincompress` (music,
   ducking, SFX), `loudnorm`, `scdet` (scene detection), `libx264` + `aac`.
   All verified present in the target FFmpeg 6.1 build.
3. **Deterministic and verifiable** — same timeline → same output; `ffprobe`
   verifies results.
4. **Cheap** — runs on a 2–4 vCPU VPS; 720p/1080p H.264 renders of 30–60 s
   clips take seconds to a couple of minutes on CPU.
5. **Swappable** — the compiler sits behind an `EditingEngine` interface, so an
   MLT (or Remotion) adapter can be added without touching the agent.

---

## 3. System overview

```
 ┌──────────────────────────── PHONE (browser / PWA) ────────────────────────────┐
 │  F. UI: upload media · reference · music · prompt · preset · CREATE EDIT       │
 │        progress steps · preview player · Download MP4 · Edit again            │
 └───────────────▲───────────────────────────────┬───────────────────────────────┘
                 │ HTTPS (JSON + chunked upload)  │ SSE/polling for progress
 ┌───────────────┴───────────────────────────────▼───────────────────────────────┐
 │  API server (FastAPI)  — auth, upload validation, job creation, downloads     │
 └───────┬───────────────────────────┬───────────────────────────┬───────────────┘
         │ enqueue job               │ read/write                │ read/write
 ┌───────▼────────┐       ┌──────────▼──────────┐      ┌─────────▼─────────┐
 │ Job queue      │       │ D. Storage           │      │ Metadata DB       │
 │ (SQLite → Redis)│      │ originals / proxies  │      │ (SQLite→Postgres) │
 └───────┬────────┘       │ frames / analysis    │      │ jobs, assets,     │
         │                │ cache / renders      │      │ timelines, events │
 ┌───────▼──────────────────────────────────────┴──────┐└─────────▲─────────┘
 │  Worker process                                      │          │
 │  ┌──────────────────┐   tool calls   ┌─────────────┐ │          │
 │  │ A. Agent          │──────────────▶│ Tool layer  │ │──────────┘
 │  │ orchestrator      │◀──────────────│ (validated) │ │
 │  │  + LLMProvider    │  typed results└──┬───────┬──┘ │
 │  └──────────────────┘                   │       │    │
 │        ┌────────────────────────────────▼─┐  ┌──▼──────────────────────┐
 │        │ B. Analysis (ffprobe, scdet,      │  │ C. Editing engine        │
 │        │ librosa beats, whisper captions,  │  │ Timeline model + compiler│
 │        │ frame sampling, reference style)  │  │ → E. Renderer (FFmpeg)   │
 │        └───────────────────────────────────┘  └──────────────────────────┘
 └──────────────────────────────────────────────────────────────────────────┘
                      (same tool layer also exposed as an MCP server)
```

### Component boundaries (A–F)

| Component | Responsibility | Must NOT |
|---|---|---|
| **A. Agent** (`agent/`) | Turn prompt + analysis summaries into an `EditPlan`, then into tool calls; react to tool errors; decide retries. | Touch files, spawn processes, or know FFmpeg syntax. |
| **B. Analysis** (`analysis/`) | Metadata, proxies, scene cuts, frames, audio/beat analysis, transcripts, reference-style profile. All results cached by content hash. | Modify timelines. |
| **C. Editing engine** (`engine/`) | Timeline data model, operation validation, transactional apply/undo, compile timeline → render job. | Call the LLM. |
| **D. Storage** (`storage/`) | Content-addressed file store, sanitised paths, retention/cleanup. Local disk first; S3-compatible (MinIO/R2) adapter later. | Accept paths from users. |
| **E. Rendering** (`render/`) | Run FFmpeg with arg lists, timeouts, progress parsing; produce preview (low-res) and final; verify output with ffprobe. | Report success without verification. |
| **F. UI** (`web/`) | Mobile-first PWA: forms, chunked upload, progress, player, download, retry. | Hold secrets or call the LLM directly. |

The **tool layer** is the only bridge between A and B/C/E. It is defined once
(Pydantic models → JSON Schema) and exposed in two ways:

* in-process to the built-in agent (any `LLMProvider`), and
* as an **MCP server** (Phase 5), so external agents (Claude Code, other MCP
  clients) can drive the same editor with the same validation.

---

## 4. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Language (backend) | **Python 3.12+** | Best ecosystem for media analysis (numpy, librosa, faster-whisper, PySceneDetect). |
| API | **FastAPI** + Uvicorn | Typed, async, auto-docs, Pydantic models shared with the tool layer. |
| Jobs | Phase 1: in-process worker thread + SQLite job table. Later: Redis + RQ/Arq | Long renders never block HTTP requests. |
| DB | SQLite (WAL) → Postgres when multi-instance | Jobs, assets, timelines (JSON), event log. |
| Media | **FFmpeg / ffprobe** 6.x | System binary, invoked via `subprocess` arg list. |
| Scene detection | FFmpeg `scdet` (Phase 1), PySceneDetect optional | On proxies. |
| Beats / audio | `librosa` (onset + beat tracking), FFmpeg `ebur128`/`silencedetect` | On a 22 kHz mono proxy. |
| Speech-to-text | `faster-whisper` (CPU, `small`/`base` int8) | Open source, no API cost; optional per job. |
| LLM | `LLMProvider` interface: `AnthropicProvider`, `OpenAICompatibleProvider`, `RuleBasedPlanner` | Selected by server env vars. |
| Frontend | Plain HTML + CSS + small vanilla JS modules, served by FastAPI; installable **PWA** | No build step, fast on low-end phones. |
| Deployment | One Docker image (API + worker) on any small VPS / Railway / Render / Fly | User only needs the URL. |

---

## 5. Data contracts

### 5.1 Asset

```json
{
  "asset_id": "a_7f3c…",             // server-generated, never user-provided
  "role": "clip | reference | music | sfx",
  "original_name": "IMG_0042.MOV",   // display only, sanitised
  "sha256": "…",                     // cache key for all analysis
  "path": "store/originals/ab/abcd….mov",
  "proxy_path": "store/proxies/…_360p.mp4",
  "meta": { "duration": 12.43, "width": 1080, "height": 1920, "fps": 29.97,
            "rotation": 90, "vcodec": "hevc", "has_audio": true, "acodec": "aac" }
}
```

### 5.2 Edit plan (what the agent produces first)

The plan is *intent*. It is validated against asset metadata before any
timeline is touched.

```json
{
  "version": 1,
  "project": { "preset": "instagram_reels", "aspect_ratio": "9:16",
               "resolution": "1080p", "fps": 30, "target_duration": 30 },
  "style": { "pacing": "fast", "avg_shot": 1.2, "transitions": "rare",
             "zoom": "punch_in_on_beats", "captions": "bold_center_lower" },
  "segments": [
    { "asset_id": "a_1", "src_in": 3.20, "src_out": 5.30,
      "speed": 1.0, "effects": [ { "type": "zoom", "from": 1.0, "to": 1.15 } ],
      "transition_in": null, "reason": "subject enters frame" }
  ],
  "audio": { "music": { "asset_id": "a_m", "start": 0, "volume": 0.8,
                        "duck_under_speech": true },
             "keep_original_audio": true, "original_volume": 0.6 },
  "captions": { "mode": "transcript | text", "items": [] },
  "text_overlays": [],
  "sound_effects": []
}
```

### 5.3 Timeline (engine state, the single source of truth for rendering)

```json
{
  "timeline_id": "t_…", "revision": 7,
  "settings": { "width": 1080, "height": 1920, "fps": 30, "bg": "#000000" },
  "tracks": [
    { "track_id": "V1", "kind": "video", "clips": [
      { "clip_id": "c_1", "asset_id": "a_1",
        "src_in": 3.20, "src_out": 5.30, "timeline_start": 0.0,
        "speed": 1.0, "fit": "cover",
        "transform": { "keyframes": [ { "t": 0, "scale": 1.0, "x": 0.5, "y": 0.5 },
                                      { "t": 2.1, "scale": 1.15, "x": 0.5, "y": 0.45 } ] },
        "color": { "brightness": 0, "contrast": 1.05, "saturation": 1.1 },
        "transition_in": { "type": "fade", "duration": 0.25 },
        "volume": 0.6 } ] },
    { "track_id": "A1", "kind": "audio", "clips": [] },
    { "track_id": "T1", "kind": "caption", "items": [
      { "item_id": "cap_1", "start": 0.4, "end": 1.6, "text": "LET'S GO",
        "style": "bold_center_lower" } ] }
  ]
}
```

Clip duration on the timeline = `(src_out − src_in) / speed`.

---

## 6. Tool surface (provider-neutral)

Every tool: Pydantic input model → JSON Schema; returns
`{"ok": true, "result": …}` or
`{"ok": false, "error": {"code": "CLIP_NOT_FOUND", "message": "…", "retryable": false}}`.

| Group | Tools | Phase |
|---|---|---|
| Inspect | `get_video_metadata`, `analyze_video`, `extract_frames`, `detect_scenes`, `analyze_audio`, `detect_beats`, `transcribe` | 1–3 |
| Reference | `analyze_reference` (→ style profile, §8) | 3 |
| Timeline | `create_timeline`, `add_video_track`, `add_audio_track`, `select_clip` (insert segment), `trim_clip`, `split_clip`, `move_clip`, `delete_clip`, `get_timeline` | 1–2 |
| Visual | `crop_video`, `resize_video` (fit/cover), `rotate_video`, `add_zoom`, `add_keyframe`, `change_speed`, `apply_color_adjustment`, `apply_effect`, `add_transition` | 2–4 |
| Text/audio | `add_text`, `add_caption`, `add_subtitles`, `add_music`, `set_volume`, `add_sound_effect` | 2–4 |
| Output | `validate_timeline`, `render_preview`, `export_video`, `verify_output` | 1 |

Tools not yet implemented are **not offered** to the model and are shown in
the UI as "Coming in Phase N" — never simulated.

### Validation rules (enforced in the engine, not trusted from the model)

* `0 ≤ src_in < src_out ≤ asset.duration` (with 1-frame tolerance).
* `timeline_start ≥ 0`; clips on a video track may not overlap except within a
  declared transition window.
* Referenced `asset_id`, `track_id`, `clip_id` must exist; role must fit
  (no music asset on a video track, reference assets are **never** placeable).
* `0.25 ≤ speed ≤ 4`; zoom scale `1.0–3.0`; volume `0–2`; transition
  duration `< min(adjacent clip durations)/2`.
* Total duration `> 0` and `≤ preset max` (e.g. 90 s Reels, 180 s Shorts,
  10 min TikTok, 15 min landscape — configurable).
* Captions: `start < end`, inside the timeline, text length limited, rendered
  through an `.ass` file with escaping (no filter-string injection).
* Music shorter than the timeline → loop or end-fade (explicit choice);
  longer → trimmed with fade-out.

### Transactions

Every mutating tool runs against a **copy** of the timeline; on success the
revision increments and the operation is appended to an op-log (enables
undo / "edit again"). On failure the previous revision is untouched. A whole
plan is executed as one batch transaction: if any step fails non-retryably,
the batch rolls back and the agent receives the error to repair the plan
(max N repair rounds, then a clear user-facing error).

---

## 7. Job pipeline (what the user sees vs. what runs)

| UI step | Work | Failure → user message |
|---|---|---|
| Uploading… | Chunked upload, size limit, extension + **ffprobe sniff** of real container/codec | "Your video format isn't supported." / "The uploaded file is too large." |
| Analyzing media… | metadata, rotation normalisation, 360p proxy, scenes, audio loudness, (opt.) transcript; cached by sha256 | "We couldn't read clip X." |
| Analyzing reference… | §8 style profile, cached | "The reference video could not be analyzed." (job may continue without it, clearly flagged) |
| Creating edit plan… | 1 LLM call (or rule planner) with compact JSON summaries | "The AI planner failed — retry?" |
| Building timeline… | Validate plan → batch tool calls → `validate_timeline` | "The timeline contains an invalid clip." |
| Applying effects… | zoom/speed/colour/transitions/captions tools | per-op error |
| Rendering… | compile → FFmpeg final (preview first for long jobs) with progress % | "The rendering server failed." |
| Checking final video… | ffprobe: exists, size > 0, H.264/AAC, WxH matches, duration ≈ timeline ±0.2 s, decodable sample frames | "Rendering produced an invalid file." |
| Complete. | Only now: preview URL + Download MP4 + Edit Again | — |

Retries: transient failures (render timeout, LLM 5xx/429) are retried with
backoff up to 2 times; validation errors are never blindly retried.

---

## 8. Reference-video analysis (style, not content)

Computed on a 360p proxy, entirely with local tools, then summarised:

* duration, aspect ratio, fps (ffprobe)
* shot boundaries (`scdet`) → shot count, **average/median shot length**,
  cuts per second, shot-length distribution over time (pacing curve)
* motion/zoom hints: frame-to-frame scale change estimated from a few sampled
  frame pairs (OpenCV feature matching) → "zoom-ins per minute"
* speed changes: heuristic from motion magnitude spikes (low confidence, labelled)
* transitions: dissolve/fade detection from luminance blend around cuts
* captions: sampled frames → text-region position (top/centre/bottom) and
  on-screen timing; optional OCR later
* audio: tempo (BPM), beat grid, cut-to-beat alignment ratio, loudness
* colour: mean brightness, contrast, saturation, warm/cool bias per shot

Only **≤ 8 thumbnails** (one per representative shot, 256 px) plus the JSON
profile go to the model, and only if the provider supports images. The
reference asset is flagged `role=reference` and the engine **rejects** any
attempt to place it on the timeline — the agent can only borrow its numbers.

---

## 9. LLM abstraction & cost control

```python
class LLMProvider(Protocol):
    def complete(self, system: str, messages: list[Message],
                 tools: list[ToolSpec], images: list[ImageRef] = ...) -> LLMResponse: ...
```

* `ToolSpec` = name, description, JSON Schema — mapped by each adapter to its
  vendor format (Anthropic `tools`, OpenAI `functions`, MCP `tools/list`).
* `RuleBasedPlanner` implements the same interface deterministically (even
  spacing, beat snapping, preset defaults). Used for tests, for offline/no-key
  mode, and as fallback if the LLM is unavailable.
* Budget per job: default ≤ 3 LLM calls (plan, optional repair, optional
  review), hard cap configurable. Inputs are summaries, not media.
* Caching: metadata, proxies, frames, transcripts, beats, scene lists and
  reference profiles stored under `cache/<sha256>/…` — re-uploading the same
  file never re-analyses it.
* Proxies (360p, 22 kHz mono) for all analysis; originals only for final render.

---

## 10. Security

* API keys live only in server env vars; the browser never sees them and never
  calls the LLM.
* Uploads: max size (default 500 MB/file, configurable), allowed containers
  MP4/MOV/WEBM (+ MP3/M4A/WAV/AAC for audio), verified by ffprobe — not just
  by extension.
* Stored under server-generated IDs; original names kept only as sanitised
  display strings. All paths resolved and checked to stay inside the storage
  root (no path traversal).
* FFmpeg is always called with an **argument list** (`shell=False`), timeouts,
  and `-nostdin`; user text only reaches FFmpeg via generated `.ass` files with
  escaping. No tool accepts raw commands.
* Per-job temp directories; retention policy (default: temp 24 h, renders
  72 h) enforced by a cleanup task.
* Simple access token / passphrase for the single-user MVP; rate limit job
  creation.

---

## 11. Presets

| Preset | Aspect | 1080p | 720p | Default duration |
|---|---|---|---|---|
| Instagram Reels | 9:16 | 1080×1920 | 720×1280 | 30 s |
| YouTube Shorts | 9:16 | 1080×1920 | 720×1280 | 30 s |
| TikTok | 9:16 | 1080×1920 | 720×1280 | 30 s |
| YouTube landscape | 16:9 | 1920×1080 | 1280×720 | 60 s |
| Square | 1:1 | 1080×1080 | 720×720 | 30 s |

Output always MP4 (H.264 High, yuv420p, `+faststart`), AAC 48 kHz stereo,
30 fps default. No 4K.

---

## 12. Repository layout (target)

```
server/
  app/
    main.py            # FastAPI app, routes
    config.py          # env-based settings
    jobs.py            # job table + worker
    storage/           # D
    analysis/          # B
    engine/            # C: timeline model, ops, validation, compiler
    render/            # E: ffmpeg runner, verify
    agent/             # A: orchestrator, providers/, planner
    tools/             # tool registry (+ MCP server, Phase 5)
  tests/
web/                   # F: index.html, app.js, styles.css, manifest
docs/ARCHITECTURE.md
Dockerfile
```

---

## 13. Phased delivery

Each phase ends with: tests → bug hunt → fixes → end-to-end verification of
the main workflow on generated test media.

| Phase | Scope | Exit criteria |
|---|---|---|
| **1** | Upload 1 video → prompt → plan (LLM or rule planner) → simple cuts → aspect/resize → render MP4 → verify → download. Mobile UI with real progress. | A phone-size browser can produce a verified 9:16 MP4 from an uploaded MOV/MP4. |
| **2** | Multiple clips, multi-track timeline, captions (manual + whisper), music with volume/ducking, `move/split/delete`. | Multi-clip edit with captions + music renders and verifies. |
| **3** | Reference analysis + style profile → pacing/shot-length matching. | Shot-length distribution of output tracks the reference within tolerance. |
| **4** | Zoom/keyframes, speed ramps, transitions, colour, SFX, beat-synced cuts. | Cuts land on beats (±1 frame); effects render correctly. |
| **5** | MCP server over the same tool layer, agent self-verification (sample frames → review), repair loop. | External MCP client can complete an edit end-to-end. |
| **6** | Mobile UI polish, resumable uploads, PWA install, render performance. | Smooth on a mid-range phone over mobile data. |

---

## 14. Known limitations (stated up front)

* CPU rendering: a 30 s 1080p edit with effects may take ~1–3 min on a small VPS.
* Reference "speed change" and "zoom" detection are heuristic and reported
  with confidence levels, not as facts.
* Whisper transcription on CPU is the slowest analysis step; it is optional
  and cached.
* Single-user MVP auth; multi-user accounts are out of scope until after Phase 6.
