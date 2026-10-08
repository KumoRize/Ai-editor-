# AI Video Editor Agent

A mobile-first AI agent that really edits video: you upload clips (and optionally
music and a style reference) from your phone, describe the edit, and the server
plans it, executes it on a real timeline, renders it with FFmpeg, verifies the
output, and gives you an MP4 to download.

Status: architecture defined, Phase 1 next. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Key decisions:

* **Engine:** FFmpeg (server-side), driven by a typed JSON timeline and a
  validated tool layer. The model never writes FFmpeg commands.
* **AI:** provider-agnostic `LLMProvider` (Claude, OpenAI-compatible/local
  models, or a no-key rule-based planner). Tools are also exposed over MCP later.
* **Phone:** browser/PWA only; all heavy work runs on the server.
