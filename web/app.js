// Mobile web UI. Every action calls a real backend endpoint; nothing is simulated.
const $ = (id) => document.getElementById(id);
const state = {
  config: null, mode: "settings", clips: [], ref: null, music: null,
  jobId: null, lastJob: null, pollTimer: null, parentJobId: null,
};
const MODES = {
  settings: {
    hint: "Choose the format, length and framing below, and describe the edit in the prompt.",
    label: "Prompt",
    placeholder: "e.g. Fast cuts on the beat, zooms, warm colour and bold subtitles",
    rows: 4,
  },
  prompt: {
    hint: "Write everything in one prompt. The agent works out the platform, format, length, framing and the whole edit from it.",
    label: "Full prompt — describe the whole edit",
    placeholder: "e.g. Make a 30-second vertical Instagram Reel like my reference: fast cuts on the beat, " +
      "cinematic zooms, a slow-motion moment, dissolve transitions, bold subtitles, the title \"Summer 2026\", " +
      "whooshes on transitions and a fade out at the end.",
    rows: 7,
  },
};
const EXAMPLES = [
  "Make a 30-second vertical Instagram Reel: fast cuts on the beat, punch-in zooms, warm cinematic colour, " +
    "bold subtitles, the title \"Summer 2026\" at the start and \"Follow for more\" at the end, whooshes on transitions.",
  "Calm 45-second YouTube Short with long takes, slow zooms, dissolve transitions, vintage look, " +
    "remove the silent parts and fade out at the end.",
  "Landscape YouTube video, keep my clips in order, remove the silent parts, add subtitles at the bottom, " +
    "quiet background music that ducks under my voice.",
];
const STEP_ICONS = { pending: "○", running: '<span class="spin"></span>', done: "✓", failed: "✕", skipped: "–" };

function show(view) {
  for (const v of ["login", "form", "progress", "result"]) $(`view-${v}`).hidden = v !== view;
  window.scrollTo(0, 0);
}
function store(key, val) { try { if (val === null) localStorage.removeItem(key); else localStorage.setItem(key, val); } catch { /* no storage */ } }
function load(key) { try { return localStorage.getItem(key); } catch { return null; } }
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}
function errorMessage(body, fallback) {
  if (body && body.error && body.error.message) return body.error.message;
  if (body && typeof body.detail === "string") return body.detail;
  return fallback;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function fmtMB(b) { return `${(b / 1048576).toFixed(1)} MB`; }
function fmtDur(s) { return s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : `${s.toFixed(1)}s`; }

async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, { credentials: "same-origin", ...opts });
  } catch {
    throw Object.assign(new Error("No connection to the server. Check your internet and try again."), { network: true, retryable: true });
  }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (res.status === 401 && path !== "/api/session") { show("login"); throw new Error("Please enter the access code."); }
  if (!res.ok) throw Object.assign(new Error(errorMessage(body, `Server error (${res.status}).`)),
    { retryable: body?.error?.retryable, status: res.status, body });
  return body;
}

// ------------------------------------------------------------------ resumable chunked upload
async function uploadFile(file, role, onProgress) {
  const fp = `ve_up:${role}:${file.name}:${file.size}:${file.lastModified}`;
  let up = null;
  const saved = load(fp);
  if (saved) {
    try { up = await api(`/api/uploads/${saved}`); up.upload_id = saved; } catch { up = null; }
  }
  if (!up) {
    up = await api("/api/uploads/init", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, size: file.size, role }) });
    store(fp, up.upload_id);
  }
  const chunk = state.config.chunk_size || 4 * 1048576;
  let received = up.received || 0;
  let failures = 0;
  onProgress(received / file.size);
  while (received < file.size) {
    const part = file.slice(received, Math.min(file.size, received + chunk));
    try {
      const r = await api(`/api/uploads/${up.upload_id}?offset=${received}`, { method: "PUT",
        headers: { "Content-Type": "application/octet-stream" }, body: part });
      received = r.received;
      failures = 0;
      onProgress(received / file.size);
    } catch (e) {
      if (e.status === 409 && e.body && typeof e.body.received === "number") { received = e.body.received; continue; }
      if (e.status === 404) { store(fp, null); throw new Error("The upload expired. Please add the file again."); }
      if ((e.network || e.status >= 500) && failures < 8) {
        failures += 1;
        onProgress(received / file.size, `Connection problem — retrying (${failures})…`);
        await sleep(Math.min(15000, 1000 * 2 ** failures));
        if (!navigator.onLine) await new Promise((res) => window.addEventListener("online", res, { once: true }));
        try { received = (await api(`/api/uploads/${up.upload_id}`)).received; } catch { /* retry loop */ }
        continue;
      }
      throw e;
    }
  }
  onProgress(1, "Checking the file…");
  const asset = await api(`/api/uploads/${up.upload_id}/complete`, { method: "POST" });
  store(fp, null);
  return asset;
}

// ------------------------------------------------------------------ file lists
let nextKey = 1;
function accepted(file, role) {
  const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  return state.config.accept[role].includes(ext);
}

function addFile(file, role) {
  const item = { key: nextKey++, name: file.name, size: file.size, status: "uploading", progress: 0, note: "", asset: null, error: "" };
  if (!accepted(file, role)) {
    item.status = "error";
    item.error = role === "music" ? "That audio format isn't supported. Use MP3, M4A, WAV or AAC."
      : "Your video format isn't supported. Use MP4, MOV or WEBM.";
  } else if (file.size > state.config.max_upload_mb * 1048576) {
    item.status = "error";
    item.error = `The file is too large (${fmtMB(file.size)}; limit ${state.config.max_upload_mb} MB).`;
  }
  if (role === "clip") state.clips.push(item); else state[role === "reference" ? "ref" : "music"] = item;
  state.parentJobId = null; // new media starts a fresh edit
  renderFiles();
  if (item.status === "error") return;
  uploadFile(file, role, (p, note) => { item.progress = p; item.note = note || ""; renderFiles(); })
    .then((asset) => { item.asset = asset; item.status = "ready"; item.note = ""; renderFiles(); })
    .catch((e) => { item.status = "error"; item.error = e.message; renderFiles(); });
}

function fileRow(item, role, idx) {
  const a = item.asset;
  let info = "";
  if (item.status === "ready" && a) {
    info = `${fmtDur(a.duration)}${a.width ? ` · ${a.width}×${a.height}` : ""} · ${fmtMB(a.size_bytes)}` +
      (a.has_audio ? "" : " · no sound") + (a.hdr ? " · HDR" : "");
  } else if (item.status === "uploading") {
    info = item.note || `Uploading… ${Math.round(item.progress * 100)}%`;
  }
  const thumb = a && a.thumb_url ? `<img src="${a.thumb_url}" alt="" loading="lazy">` : `<span class="ph">${role === "music" ? "♪" : "▶"}</span>`;
  const controls = role === "clip"
    ? `<button type="button" data-act="up" data-i="${idx}" aria-label="Move up" ${idx === 0 ? "disabled" : ""}>↑</button>
       <button type="button" data-act="down" data-i="${idx}" aria-label="Move down" ${idx === state.clips.length - 1 ? "disabled" : ""}>↓</button>`
    : "";
  return `<li class="file ${item.status}">
    ${thumb}
    <div class="meta"><span class="ellipsis">${escapeHtml(item.name)}</span>
      <span class="muted small">${escapeHtml(info)}</span>
      ${item.status === "uploading" ? `<div class="bar"><div style="width:${Math.round(item.progress * 100)}%"></div></div>` : ""}
      ${item.status === "error" ? `<span class="error small">${escapeHtml(item.error)}</span>` : ""}</div>
    <div class="ctl">${controls}<button type="button" data-act="remove" data-role="${role}" data-i="${idx}" aria-label="Remove">✕</button></div>
  </li>`;
}

function renderFiles() {
  $("clip-list").innerHTML = state.clips.map((c, i) => fileRow(c, "clip", i)).join("");
  $("ref-list").innerHTML = state.ref ? fileRow(state.ref, "reference", 0) : "";
  $("music-list").innerHTML = state.music ? fileRow(state.music, "music", 0) : "";
  const ready = state.clips.filter((c) => c.status === "ready").length;
  $("clip-count").textContent = state.clips.length ? `(${ready} of ${state.clips.length} ready, max ${state.config.max_clips})` : "";
  updateCreateButton();
}

function onListClick(ev) {
  const b = ev.target.closest("button[data-act]");
  if (!b) return;
  const i = Number(b.dataset.i);
  if (b.dataset.act === "remove") {
    if (b.dataset.role === "clip") state.clips.splice(i, 1);
    else state[b.dataset.role === "reference" ? "ref" : "music"] = null;
  } else if (b.dataset.act === "up" && i > 0) {
    [state.clips[i - 1], state.clips[i]] = [state.clips[i], state.clips[i - 1]];
  } else if (b.dataset.act === "down" && i < state.clips.length - 1) {
    [state.clips[i + 1], state.clips[i]] = [state.clips[i], state.clips[i + 1]];
  }
  state.parentJobId = null;
  renderFiles();
}
for (const id of ["clip-list", "ref-list", "music-list"]) $(id).addEventListener("click", onListClick);

$("file-clip").addEventListener("change", () => {
  const files = [...$("file-clip").files];
  const room = state.config.max_clips - state.clips.length;
  if (files.length > room) {
    $("form-error").textContent = `At most ${state.config.max_clips} clips per edit.`; $("form-error").hidden = false;
  }
  files.slice(0, Math.max(0, room)).forEach((f) => addFile(f, "clip"));
  $("file-clip").value = "";
});
$("file-ref").addEventListener("change", () => { const f = $("file-ref").files[0]; if (f) addFile(f, "reference"); $("file-ref").value = ""; });
$("file-music").addEventListener("change", () => { const f = $("file-music").files[0]; if (f) addFile(f, "music"); $("file-music").value = ""; });

function uploading() {
  return [...state.clips, state.ref, state.music].some((x) => x && x.status === "uploading");
}
function updateCreateButton() {
  $("create-btn").disabled = !state.clips.some((c) => c.status === "ready") || uploading();
}

// ------------------------------------------------------------------ setup
async function init() {
  try {
    state.config = await api("/api/config");
  } catch (e) {
    if (!$("view-login").hidden) return;
    document.body.insertAdjacentHTML("beforeend", `<p class="error" style="padding:16px">${escapeHtml(e.message)}</p>`);
    return;
  }
  const c = state.config;
  $("planner").textContent = `Planner: ${c.planner}`;
  $("planner").hidden = false;
  const sp = c.features.subtitles_from_speech;
  $("speech-note").textContent = sp.available ? "" : ` (not available on this server: ${sp.note})`;
  $("preset").innerHTML = Object.entries(c.presets)
    .map(([k, p]) => `<option value="${k}">${p.label} (${p.aspect_ratio})</option>`).join("");
  $("aspect").innerHTML = ["9:16", "16:9", "1:1"].filter((a) => c.aspect_ratios.includes(a))
    .map((a) => `<option value="${a}">${a}</option>`).join("");
  $("fit").innerHTML = Object.entries(c.fits).map(([k, l]) => `<option value="${k}">${l}</option>`).join("");
  applyPreset();
  setMode(load("ve_mode") || "settings");
  renderFiles();
  const saved = load("ve_job");
  if (saved) resumeJob(saved); else show("form");
}

function applyPreset() {
  const p = state.config.presets[$("preset").value];
  $("aspect").value = p.aspect_ratio;
  $("duration").value = p.default_duration;
  $("duration").max = p.max_duration;
}
$("preset").addEventListener("change", applyPreset);

function setMode(mode) {
  if (!MODES[mode]) mode = "settings";
  state.mode = mode;
  store("ve_mode", mode);
  const m = MODES[mode];
  document.querySelectorAll(".seg-btn").forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle("active", on);
    b.setAttribute("aria-checked", String(on));
  });
  $("mode-hint").textContent = m.hint;
  $("prompt-label").textContent = m.label;
  if (!state.parentJobId) $("prompt").placeholder = m.placeholder;
  $("prompt").rows = m.rows;
  $("settings-card").hidden = mode === "prompt";
}
document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
let exampleIdx = 0;
$("example-btn").addEventListener("click", () => {
  $("prompt").value = EXAMPLES[exampleIdx++ % EXAMPLES.length];
  $("prompt").focus();
});

$("login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("login-error").hidden = true;
  try {
    await api("/api/session", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: $("login-token").value }) });
    show("form");
    init();
  } catch (e) {
    $("login-error").textContent = e.message; $("login-error").hidden = false;
  }
});

// ------------------------------------------------------------------ create
$("edit-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  await createJob(state.parentJobId || null); // set by "Edit again": the new edit revises the previous one
});

async function createJob(parentJobId) {
  $("form-error").hidden = true;
  const clips = state.clips.filter((c) => c.status === "ready");
  if (!clips.length) return;
  if (state.mode === "prompt" && $("prompt").value.trim().length < 3) {
    $("form-error").textContent = "Write your prompt first — in Full prompt mode the agent takes everything from it.";
    $("form-error").hidden = false;
    $("prompt").focus();
    return;
  }
  const dur = parseFloat($("duration").value);
  const payload = {
    mode: state.mode,
    asset_ids: clips.map((c) => c.asset.asset_id),
    music_asset_id: state.music && state.music.status === "ready" ? state.music.asset.asset_id : null,
    reference_asset_id: state.ref && state.ref.status === "ready" ? state.ref.asset.asset_id : null,
    prompt: $("prompt").value.trim(),
    preset: $("preset").value,
    aspect_ratio: $("aspect").value,
    resolution: $("resolution").value,
    duration: Number.isFinite(dur) && dur > 0 ? dur : null,
    fit: $("fit").value,
    parent_job_id: parentJobId,
  };
  $("create-btn").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) });
    startPolling(job);
  } catch (e) {
    $("form-error").textContent = e.message; $("form-error").hidden = false;
  } finally {
    updateCreateButton();
  }
}

// ------------------------------------------------------------------ progress
function startPolling(job) {
  state.jobId = job.job_id;
  store("ve_job", job.job_id);
  renderJob(job);
  clearTimeout(state.pollTimer);
  const tick = async () => {
    try {
      const j = await api(`/api/jobs/${state.jobId}`);
      renderJob(j);
      if (j.status === "queued" || j.status === "running") state.pollTimer = setTimeout(tick, 1000);
    } catch (e) {
      if (e.status === 404) { store("ve_job", null); show("form"); return; }
      state.pollTimer = setTimeout(tick, 3000); // keep trying through brief network drops
    }
  };
  state.pollTimer = setTimeout(tick, 700);
}

async function resumeJob(jobId) {
  try {
    const j = await api(`/api/jobs/${jobId}`);
    if (j.status === "queued" || j.status === "running") startPolling(j);
    else { state.jobId = jobId; renderJob(j); }
  } catch {
    store("ve_job", null);
    show("form");
  }
}

function renderJob(j) {
  state.lastJob = j;
  if (j.status === "complete") return renderResult(j);
  show("progress");
  $("job-bar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
  $("steps").innerHTML = j.steps.map((s) =>
    `<li class="${s.status}"><span class="icon">${STEP_ICONS[s.status] || ""}</span>
      <span>${escapeHtml(s.label)}${s.detail ? `<span class="detail">${escapeHtml(s.detail)}</span>` : ""}</span></li>`).join("");
  const failed = j.status === "failed";
  $("progress-title").textContent = failed ? "The edit could not be finished" :
    j.status === "queued" ? "Waiting for the editor…" : "Working on your edit…";
  $("job-error").hidden = !failed;
  if (failed) {
    $("job-error-msg").textContent = j.error?.message || "Something went wrong.";
    $("retry-btn").hidden = !j.error?.retryable;
  }
}

$("retry-btn").addEventListener("click", async () => {
  try { startPolling(await api(`/api/jobs/${state.jobId}/retry`, { method: "POST" })); }
  catch (e) { $("job-error-msg").textContent = e.message; }
});
$("back-btn").addEventListener("click", () => { store("ve_job", null); show("form"); });

// ------------------------------------------------------------------ result
function renderResult(j) {
  show("result");
  const r = j.result;
  store("ve_job", j.job_id);
  const player = $("player");
  if (player.getAttribute("src") !== r.video_url) {
    $("player-error").hidden = true;
    player.src = r.video_url;
  }
  $("download-btn").href = r.download_url;
  $("share-btn").hidden = !(navigator.canShare && navigator.share);
  $("result-summary").textContent = r.summary || "";
  $("result-meta").textContent =
    `${r.width}×${r.height} · ${r.duration.toFixed(1)}s · ${(r.size_bytes / 1048576).toFixed(1)} MB · planner: ${r.planner}`;
  const uns = r.unsupported_requests || [];
  $("result-unsupported").hidden = uns.length === 0;
  $("result-unsupported").innerHTML = uns.length
    ? `<strong>Not applied</strong> — these parts of your request can't be done:<ul>${
      uns.map((u) => `<li>${escapeHtml(u)}</li>`).join("")}</ul>` : "";
  const notes = r.notes || [];
  $("result-notes").hidden = notes.length === 0;
  $("result-notes").innerHTML = notes.map((n) => escapeHtml(n)).join("<br>");
  renderDetails(r);
}

function renderDetails(r) {
  const q = r.quality || {};
  const m = q.measurements || {};
  const tl = r.timeline || {};
  const clips = Array.isArray(tl.clips) ? tl.clips : [];
  const rows = clips.slice(0, 60).map((c, i) =>
    `<tr><td>${i + 1}</td><td>${fmtDur(c.start)}</td><td>${c.duration.toFixed(1)}s${c.speed !== 1 ? ` · ${c.speed}x` : ""}</td>
     <td>${[c.transition, c.zoom ? "zoom" : ""].filter(Boolean).map(escapeHtml).join(", ")}</td></tr>`).join("");
  $("details-body").innerHTML = `
    ${r.applied && r.applied.length ? `<p><strong>Applied:</strong> ${r.applied.map(escapeHtml).join(", ")}</p>` : ""}
    <p><strong>Quality check:</strong> ${q.passed ? "passed" : "has warnings"}${m.loudness_lufs !== undefined
      ? ` · loudness ${m.loudness_lufs.toFixed(1)} LUFS · peak ${m.true_peak_dbfs.toFixed(1)} dBFS` : ""}${
      q.repaired ? ` · fixed: ${q.repaired.map(escapeHtml).join(", ")}` : ""}</p>
    ${rows ? `<table class="cuts"><thead><tr><th>#</th><th>At</th><th>Length</th><th>Effects</th></tr></thead><tbody>${rows}</tbody></table>` : ""}`;
}

$("player").addEventListener("error", () => { $("player-error").hidden = false; });

$("share-btn").addEventListener("click", async () => {
  try {
    const blob = await (await fetch(state.lastJob.result.download_url, { credentials: "same-origin" })).blob();
    const file = new File([blob], "edit.mp4", { type: "video/mp4" });
    if (navigator.canShare({ files: [file] })) await navigator.share({ files: [file], title: "My edit" });
  } catch { /* user cancelled or sharing not possible */ }
});

async function restoreAssets(req) {
  // After a page reload the file list is gone: rebuild it from the edit's request.
  const fetchAsset = async (id) => { try { return await api(`/api/assets/${id}`); } catch { return null; } };
  if (!state.clips.length && req.asset_ids) {
    for (const id of req.asset_ids) {
      const a = await fetchAsset(id);
      if (a) state.clips.push({ key: nextKey++, name: a.name, size: a.size_bytes, status: "ready", progress: 1, asset: a });
    }
  }
  if (!state.music && req.music_asset_id) {
    const a = await fetchAsset(req.music_asset_id);
    if (a) state.music = { key: nextKey++, name: a.name, status: "ready", asset: a };
  }
  if (!state.ref && req.reference_asset_id) {
    const a = await fetchAsset(req.reference_asset_id);
    if (a) state.ref = { key: nextKey++, name: a.name, status: "ready", asset: a };
  }
  renderFiles();
}

$("again-btn").addEventListener("click", async () => {
  const req = state.lastJob?.request || {};
  await restoreAssets(req);
  if (req.preset) $("preset").value = req.preset;
  if (req.aspect_ratio) $("aspect").value = req.aspect_ratio;
  if (req.resolution) $("resolution").value = req.resolution;
  if (req.duration) $("duration").value = req.duration;
  if (req.fit) $("fit").value = req.fit;
  $("prompt").value = req.prompt || "";
  state.parentJobId = state.lastJob?.job_id || null;
  setMode(req.mode || state.mode);
  $("prompt").placeholder = "Describe what to change, e.g. make it shorter, slower zooms, move the title to the end";
  show("form");
  $("prompt").focus();
});

$("new-btn").addEventListener("click", () => {
  store("ve_job", null);
  state.clips = []; state.ref = null; state.music = null; state.parentJobId = null;
  $("edit-form").reset(); applyPreset();
  setMode(state.mode);
  renderFiles();
  show("form");
});

function updateOnline() { $("offline").hidden = navigator.onLine; }
window.addEventListener("online", updateOnline);
window.addEventListener("offline", updateOnline);
updateOnline();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => { /* optional */ });
}

init();
