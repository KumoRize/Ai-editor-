// Mobile web UI. Every action calls a real backend endpoint; nothing is simulated.
const $ = (id) => document.getElementById(id);
const state = { config: null, asset: null, uploading: false, jobId: null, lastJob: null, pollTimer: null, parentJobId: null };
const STEP_ICONS = { pending: "○", running: '<span class="spin"></span>', done: "✓", failed: "✕", skipped: "–" };

function show(view) {
  for (const v of ["login", "form", "progress", "result"]) $(`view-${v}`).hidden = v !== view;
  window.scrollTo(0, 0);
}

function errorMessage(body, fallback) {
  if (body && body.error && body.error.message) return body.error.message;
  if (body && typeof body.detail === "string") return body.detail;
  return fallback;
}

async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, { credentials: "same-origin", ...opts });
  } catch {
    throw Object.assign(new Error("No connection to the server. Check your internet and try again."), { retryable: true });
  }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (res.status === 401 && path !== "/api/session") { show("login"); throw new Error("Please enter the access code."); }
  if (!res.ok) throw Object.assign(new Error(errorMessage(body, `Server error (${res.status}).`)),
    { retryable: body?.error?.retryable, status: res.status });
  return body;
}

// ------------------------------------------------------------------ setup
async function init() {
  try {
    state.config = await api("/api/config");
  } catch (e) {
    if (!$("view-login").hidden) return;
    document.body.insertAdjacentHTML("beforeend", `<p class="error" style="padding:16px">${e.message}</p>`);
    return;
  }
  const c = state.config;
  $("planner").textContent = `Planner: ${c.planner}`;
  $("planner").hidden = false;
  $("preset").innerHTML = Object.entries(c.presets)
    .map(([k, p]) => `<option value="${k}">${p.label} (${p.aspect_ratio})</option>`).join("");
  $("aspect").innerHTML = ["9:16", "16:9", "1:1"].filter((a) => c.aspect_ratios.includes(a))
    .map((a) => `<option value="${a}">${a}</option>`).join("");
  $("fit").innerHTML = Object.entries(c.fits).map(([k, l]) => `<option value="${k}">${l}</option>`).join("");
  applyPreset();
  $("preset").addEventListener("change", applyPreset);

  const saved = localStorage.getItem("ve_job");
  if (saved) { resumeJob(saved); } else { show("form"); }
}

function applyPreset() {
  const p = state.config.presets[$("preset").value];
  $("aspect").value = p.aspect_ratio;
  $("duration").value = p.default_duration;
  $("duration").max = p.max_duration;
}

$("login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("login-error").hidden = true;
  try {
    await api("/api/session", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: $("login-token").value }) });
    show("form");
    $("player").addEventListener("error", () => { $("player-error").hidden = false; });

init();
  } catch (e) {
    $("login-error").textContent = e.message; $("login-error").hidden = false;
  }
});

// ------------------------------------------------------------------ upload
$("file-clip").addEventListener("change", () => {
  const f = $("file-clip").files[0];
  if (f) uploadFile(f);
});

function fmtMB(b) { return `${(b / 1048576).toFixed(1)} MB`; }

function uploadFile(file) {
  const c = state.config;
  state.asset = null;
  state.parentJobId = null; // a new clip starts a fresh edit
  updateCreateButton();
  $("upload-status").hidden = false;
  $("upload-error").hidden = true;
  $("upload-info").textContent = "";
  $("upload-name").textContent = file.name;
  $("upload-pct").textContent = "0%";
  $("upload-bar").style.width = "0%";

  const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  if (!c.accept.includes(ext)) return uploadFailed("Your video format isn't supported. Use MP4, MOV or WEBM.");
  if (file.size > c.max_upload_mb * 1048576)
    return uploadFailed(`The uploaded file is too large (${fmtMB(file.size)}; limit ${c.max_upload_mb} MB).`);

  state.uploading = true;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/uploads?role=clip&filename=${encodeURIComponent(file.name)}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    $("upload-bar").style.width = `${pct}%`;
    $("upload-pct").textContent = pct < 100 ? `${pct}%` : "Checking…";
  };
  xhr.onerror = () => uploadFailed("Upload failed — check your connection and try again.");
  xhr.onload = () => {
    state.uploading = false;
    let body = null;
    try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
    if (xhr.status === 401) { show("login"); return; }
    if (xhr.status !== 200) return uploadFailed(errorMessage(body, `Upload failed (${xhr.status}).`));
    state.asset = body;
    $("upload-pct").textContent = "Ready ✓";
    $("upload-bar").style.width = "100%";
    $("upload-info").textContent =
      `${body.duration.toFixed(1)}s · ${body.width}×${body.height} · ${fmtMB(body.size_bytes)}` +
      (body.has_audio ? "" : " · no sound");
    $("drop-text").textContent = "Tap to choose a different video";
    updateCreateButton();
  };
  xhr.send(file);
}

function uploadFailed(msg) {
  state.uploading = false;
  state.asset = null;
  $("upload-pct").textContent = "";
  $("upload-bar").style.width = "0%";
  $("upload-error").textContent = msg;
  $("upload-error").hidden = false;
  $("file-clip").value = "";
  updateCreateButton();
}

function updateCreateButton() { $("create-btn").disabled = !state.asset || state.uploading; }

// ------------------------------------------------------------------ create
$("edit-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  await createJob(state.parentJobId || null); // set by "Edit again": the new edit revises the previous one
});

async function createJob(parentJobId) {
  $("form-error").hidden = true;
  if (!state.asset) return;
  const dur = parseFloat($("duration").value);
  const payload = {
    asset_ids: [state.asset.asset_id],
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
  localStorage.setItem("ve_job", job.job_id);
  renderJob(job);
  clearTimeout(state.pollTimer);
  const tick = async () => {
    try {
      const j = await api(`/api/jobs/${state.jobId}`);
      renderJob(j);
      if (j.status === "queued" || j.status === "running") state.pollTimer = setTimeout(tick, 1000);
    } catch (e) {
      if (e.status === 404) { localStorage.removeItem("ve_job"); show("form"); return; }
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
    localStorage.removeItem("ve_job");
    show("form");
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
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
$("back-btn").addEventListener("click", () => { localStorage.removeItem("ve_job"); show("form"); });

// ------------------------------------------------------------------ result
function renderResult(j) {
  show("result");
  const r = j.result;
  localStorage.setItem("ve_job", j.job_id);
  const player = $("player");
  if (player.getAttribute("src") !== r.video_url) {
    $("player-error").hidden = true;
    player.src = r.video_url;
  }
  $("download-btn").href = r.download_url;
  $("result-summary").textContent = r.summary || "";
  $("result-meta").textContent =
    `${r.width}×${r.height} · ${r.duration.toFixed(1)}s · ${(r.size_bytes / 1048576).toFixed(1)} MB · ` +
    `${r.segments.length} segment(s) · planner: ${r.planner}`;
  const uns = r.unsupported_requests || [];
  $("result-unsupported").hidden = uns.length === 0;
  $("result-unsupported").innerHTML = uns.length
    ? `<strong>Not applied yet</strong> — these parts of your request aren't available in this version:<ul>${
      uns.map((u) => `<li>${escapeHtml(u)}</li>`).join("")}</ul>` : "";
  const notes = r.notes || [];
  $("result-notes").hidden = notes.length === 0;
  $("result-notes").innerHTML = notes.map((n) => escapeHtml(n)).join("<br>");
}

$("again-btn").addEventListener("click", () => {
  // Keep the same clip and settings; the next edit revises this one.
  const req = state.lastJob?.request || {};
  if (!state.asset && req.asset_ids) state.asset = { asset_id: req.asset_ids[0] };
  if (req.preset) $("preset").value = req.preset;
  if (req.aspect_ratio) $("aspect").value = req.aspect_ratio;
  if (req.resolution) $("resolution").value = req.resolution;
  if (req.duration) $("duration").value = req.duration;
  if (req.fit) $("fit").value = req.fit;
  $("prompt").value = req.prompt || "";
  $("prompt").placeholder = "Describe what to change, e.g. make it shorter and start with the action";
  state.parentJobId = state.lastJob?.job_id || null;
  $("upload-status").hidden = false;
  $("upload-name").textContent = "Same clip as the previous edit";
  $("upload-pct").textContent = "Ready ✓";
  $("upload-error").hidden = true;
  updateCreateButton();
  show("form");
  $("prompt").focus();
});

$("new-btn").addEventListener("click", () => {
  localStorage.removeItem("ve_job");
  state.asset = null; state.parentJobId = null;
  $("edit-form").reset(); applyPreset();
  $("upload-status").hidden = true;
  $("drop-text").textContent = "Tap to choose a video (MP4, MOV, WEBM)";
  $("prompt").placeholder = "e.g. Make a fast-paced 30 second reel with the most exciting moments";
  updateCreateButton();
  show("form");
});

$("player").addEventListener("error", () => { $("player-error").hidden = false; });

init();
