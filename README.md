# AI Video Editor Agent

A mobile-first AI agent that really edits video. From your phone you upload clips (plus optional music
and a style-reference video), describe the edit, and the server plans it, builds a real timeline,
renders it with FFmpeg, verifies and quality-checks the file, and gives you an MP4 to download.

Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Status: Phases 1–6 implemented

| Phase | What works |
|---|---|
| 1 | Upload → prompt → planned cuts → verified H.264/AAC MP4 · presets (Reels, Shorts, TikTok, YouTube, Square) · 720p/1080p · crop / blurred-background / black-bar framing · iPhone HDR and rotated video handled |
| 2 | Up to 10 clips in order · subtitles from speech (faster-whisper) in 3 styles · text titles · music with loop, fades and automatic ducking under speech · overlay (picture-in-picture) and audio tracks |
| 3 | Reference video → style profile (shot length and pacing curve, cut rate, dissolves/fades, zoom frequency, motion, caption placement, beat alignment, colour) → applied to *your* footage. The reference footage is never used. |
| 4 | Zooms / pans / Ken Burns / punch-ins and keyframes · speed changes and speed ramps · 21 transitions · colour looks (warm, cool, cinematic, B&W, vintage, vibrant, moody) and manual adjustments · vignette / grain / blur / sharpen / sepia · crop, rotate · beat detection with cuts snapped to the beat · 7 built-in sound effects · fade in/out · loudness normalisation (−14 LUFS) |
| 5 | 40 validated tools · MCP server at `/mcp/` for external agents (Claude Code etc.) · AI agent revises the previous edit with tools on "Edit again" · automatic quality check (black frames, frozen picture, loudness, clipping) with one repair round |
| 6 | Resumable chunked uploads (survive network drops) · thumbnails · reorderable clip list · edit details (cut list, quality numbers) · share sheet · offline app shell (PWA) · parallel, cached rendering (only changed pieces re-render) |

Two ways to ask:
- **Quick setup**: pick the format, length and framing, and describe the edit.
- **Full prompt**: one prompt and the agent decides everything. Example: *"Make a 30-second vertical
  Reel like my reference: fast cuts on the beat, punch-in zooms, warm cinematic colour, bold subtitles,
  the title "Summer 2026" and "Follow for more" at the end, whooshes on transitions, fade out."*

Anything that can't be done (e.g. stabilisation, green screen, voice-over) is listed under **"Not
applied"**, never faked. The result summary lists only what is actually in the final video.

## Run it

Requirements: Python 3.11+ and FFmpeg (with libx264, libass, libzimg).

```bash
cd server
pip install -r requirements-dev.txt
ACCESS_TOKEN=pick-a-code uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://<server>:8000` on your phone, enter the access code, add clips, write a prompt, **CREATE EDIT**.
"Add to Home Screen" installs it like an app.

### Deploy (so you only need your phone)

Any small Linux VPS or container host works:
`docker build -t ai-video-editor . && docker run -p 8000:8000 -v editor-data:/data --env-file .env ai-video-editor`.
Put it behind HTTPS. Settings are in `.env.example`. 2 vCPU / 2 GB RAM is enough, because each render
piece uses about 0.4 GB at 1080p. Raise `RENDER_PARALLEL` on bigger machines.

### AI planner / agent

| `LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `auto` (default) | — | Anthropic if `ANTHROPIC_API_KEY` is set, else OpenAI-compatible if `OPENAI_API_KEY` + `LLM_MODEL`, else built-in |
| `anthropic` | `ANTHROPIC_API_KEY` | Default model `claude-opus-5-5` (override with `LLM_MODEL`) |
| `openai` | `OPENAI_BASE_URL`, `LLM_MODEL` (`OPENAI_API_KEY`) | OpenAI, OpenRouter, Groq, local Ollama/vLLM… |
| `rules` | — | Free, offline, deterministic; understands all features listed above from plain English |

A new edit costs one planner call (two if the first plan fails validation). "Edit again" with an AI
model runs a short tool-using agent loop. Models receive compact JSON summaries (scenes, loudness,
beats, transcript, reference profile), never the video. If the AI fails, the built-in planner takes
over and the result says so.

### MCP (external agents)

Endpoint `https://<server>/mcp/`, with header `Authorization: Bearer <ACCESS_TOKEN>`. Workflow:
`list_assets` → `create_project` → editing tools (each takes `project_id`) → `render_project` →
`get_job` → download `video_url`. Every call is validated and errors are returned, never faked.

## Tests

```bash
cd server && python -m pytest -q
```

92 tests generate their own media. They cover:
- every engine operation and its validation, transactions, and every framing/effect/transition path
- A/V sync, segment caching, HDR, rotation, WEBM
- beat tracking accuracy and reference-style measurements
- subtitle mapping (with a stand-in speech model)
- the quality check and the full orchestrator job
- the agent tool loop, the provider adapters (mocked), and the HTTP and MCP APIs, including auth, chunked
  uploads and failure/retry paths

Not verified in the development sandbox (network-blocked):
- downloading the Whisper model
- live calls to Claude or OpenAI
- the Docker image build

They are covered by mocked tests and run on a normal server.
