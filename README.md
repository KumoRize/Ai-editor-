# AI Video Editor Agent

A mobile-first AI agent that really edits video. Upload a clip from your phone,
describe the edit, and the server plans it, builds a real timeline, renders it
with FFmpeg, verifies the output file, and gives you an MP4 to download.

Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Status: Phase 1 complete

| Works now | Coming |
|---|---|
| Upload MP4 / MOV / WEBM from your phone (with progress, size + format checks) | Multiple clips, captions/subtitles, music (Phase 2) |
| Prompt → edit plan (Claude, any OpenAI-compatible model, or the built-in no-API-key planner) | Reference-video style matching (Phase 3) |
| **Two ways to ask:** *Quick setup* (pick format/length/framing, describe the edit) or *Full prompt* (one prompt, the agent works out everything) | |
| Prompt instructions it follows exactly: time ranges (“0:05–0:12”), first/last N seconds, remove silent parts, mute / louder / quieter | |
| Cuts chosen from scene changes and audio activity, avoiding silence | Zooms, speed ramps, transitions, beat sync, colour, SFX (Phase 4) |
| Presets: Instagram Reels, YouTube Shorts, TikTok, YouTube landscape, Square; 720p / 1080p | MCP server for external agents (Phase 5) |
| Framing: fill (crop), fit with blurred background, fit with bars | |
| iPhone HDR (HLG/PQ) footage tone-mapped to SDR; rotated phone video handled | |
| Render → independent verification (codec, size, duration, full decode) → preview + Download MP4 | |
| Edit again (revises the previous plan; unchanged segments are reused from cache) | |

Full prompt example: *"Make a 20-second vertical TikTok in 720p with a blurred background. Use 0:03–0:10,
then the last 8 seconds. Remove the silent parts and mute the sound."* The result screen shows how the
agent understood it (e.g. "TikTok · 9:16 · 720p · length from your cuts · blurred background; exact time
ranges …; original sound muted").

Anything in your prompt that isn't available yet is listed under **"Not applied yet"**
on the result screen. It is never faked.

## Run it

Requirements: Python 3.11+ and FFmpeg (with libx264, libzimg).

```bash
cd server
pip install -r requirements-dev.txt
ACCESS_TOKEN=pick-a-code uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://<server>:8000` on your phone, enter the access code, upload, prompt, **CREATE EDIT**.
On the phone you can "Add to Home Screen" to use it like an app.

### Deploy (so you only need your phone)

Any small Linux VPS or container host (Railway, Render, Fly.io, a $5–10 VPS) works:
`docker build -t ai-video-editor . && docker run -p 8000:8000 -v editor-data:/data --env-file .env ai-video-editor`.
Put it behind HTTPS (the host's TLS, or Caddy/nginx). See `.env.example` for settings.
2 vCPU / 2 GB RAM is enough: rendering is done one segment at a time
(≈0.4 GB peak for a 90 s, 70-cut 1080p reel).

### AI planner

| `LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `auto` (default) | — | Uses Anthropic if `ANTHROPIC_API_KEY` is set, else OpenAI-compatible if `OPENAI_API_KEY` + `LLM_MODEL` are set, else the built-in planner |
| `anthropic` | `ANTHROPIC_API_KEY` | Default model `claude-opus-5-5` (override with `LLM_MODEL`) |
| `openai` | `OPENAI_BASE_URL`, `LLM_MODEL`, (`OPENAI_API_KEY`) | OpenAI, OpenRouter, Groq, local Ollama/vLLM … |
| `rules` | — | Free, offline, deterministic |

One edit costs one planner call (two if the first plan fails validation). The model
gets a compact JSON summary of your footage, never the video. If the AI planner
fails, the built-in planner is used and the result says so.

## Tests

```bash
cd server && python -m pytest -q
```

The 61 tests generate their own media. They cover validation rules, transactions,
rendering in every framing mode, rotation, files without audio, WEBM, A/V sync over many cuts,
the segment cache, provider adapters (mocked), and the full HTTP workflow including failures and retries.
