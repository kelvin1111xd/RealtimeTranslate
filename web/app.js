const state = { jobId: null, translationSocket: null, translationSocketKey: null };
const STAGES = [
  ["queued", "Queued"],
  ["downloading", "Download"],
  ["audio", "Audio"],
  ["transcribing", "ASR"],
  ["normalizing", "Clean"],
  ["translating_setup", "Prepare"],
  ["translating_segments", "Translate"],
  ["exporting", "Export"],
  ["done", "Done"],
];

const $ = (id) => document.getElementById(id);

function formatMs(ms) {
  const total = Math.floor(ms / 1000);
  const h = String(Math.floor(total / 3600)).padStart(2, "0");
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    $("serverStatus").textContent = response.ok ? "Ready" : "Unavailable";
  } catch {
    $("serverStatus").textContent = "Unavailable";
  }
}

async function createJob(event) {
  event.preventDefault();
  const targetLanguages = $("targetLanguages")
    .value.split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      youtubeUrl: $("youtubeUrl").value,
      sourceLanguage: $("sourceLanguage").value,
      targetLanguages,
      qualityMode: "quality",
    }),
  });
  const payload = await response.json();
  state.jobId = payload.job?.id;
  $("videoId").value = payload.job?.videoId || "";
  renderJob(payload);
  connectTranslationStream(payload.job);
  refreshJob();
}

async function refreshJob() {
  if (!state.jobId) return;
  const response = await fetch(`/api/jobs/${state.jobId}`);
  renderJob(await response.json());
}

function renderJob(payload) {
  renderProgress(payload);
  $("jobOutput").textContent = JSON.stringify(payload, null, 2);
  connectTranslationStream(payload?.job);
}

function renderProgress(payload) {
  const job = payload?.job;
  const links = payload?.links || {};
  if (!job) return;
  const percent = Number(job.progressPercent ?? statusFallbackPercent(job.status));
  const stage = job.progressStage || job.status;
  $("progressStatus").textContent = `${job.status}${links.cache ? ` · cache ${links.cache}` : ""}`;
  $("progressMessage").textContent = job.progressMessage || links.message || "";
  $("progressPercent").textContent = `${percent}%`;
  $("progressBar").style.width = `${percent}%`;
  $("progressBar").className = job.status === "failed" ? "failed" : job.status === "done" ? "done" : "";
  $("progressDetail").textContent = formatProgressDetail(job.progressDetail || {}, links);
  renderSegmentProgress(job);

  const activeIndex = STAGES.findIndex(([key]) => key === stage || key === job.status);
  $("stageGrid").innerHTML = "";
  STAGES.forEach(([key, label], index) => {
    const node = document.createElement("div");
    node.className = "stage";
    if (index < activeIndex || job.status === "done") node.classList.add("complete");
    if (index === activeIndex && job.status !== "done") node.classList.add("active");
    if (job.status === "failed" && index === activeIndex) node.classList.add("failed");
    node.textContent = label;
    $("stageGrid").appendChild(node);
  });
}

function renderSegmentProgress(job) {
  const detail = job.progressDetail || {};
  const visible = job.status === "translating" || detail.segmentTotal;
  $("segmentProgress").hidden = !visible;
  if (!visible) return;
  const completed = Number(detail.segmentCompleted || 0);
  const total = Number(detail.segmentTotal || detail.segments || 0);
  const percent = Number(detail.segmentPercent || 0);
  $("segmentProgressCount").textContent = `${completed} / ${total}`;
  $("segmentProgressBar").style.width = `${percent}%`;
}

function statusFallbackPercent(status) {
  return {
    queued: 0,
    downloading: 15,
    transcribing: 35,
    translating: 65,
    exporting: 92,
    done: 100,
    failed: 100,
  }[status] ?? 0;
}

function formatProgressDetail(detail, links) {
  const parts = [];
  if (links.message) parts.push(links.message);
  if (detail.language) parts.push(`Language ${detail.language}`);
  if (detail.languageIndex && detail.languageTotal) {
    parts.push(`${detail.languageIndex}/${detail.languageTotal} languages`);
  }
  if (detail.phase === "preparing") parts.push("Preparing translation context");
  if (detail.formats) parts.push(`Formats: ${detail.formats.join(", ")}`);
  return parts.join(" · ");
}

function connectTranslationStream(job) {
  if (!job?.id || !job.targetLanguages?.length) return;
  const language = $("subtitleLang").value.trim() || job.targetLanguages[0];
  const key = `${job.id}:${language}`;
  if (state.translationSocketKey === key) return;
  if (state.translationSocket) state.translationSocket.close();
  state.translationSocketKey = key;
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${protocol}//${location.host}/ws/jobs/${job.id}/translations/${encodeURIComponent(language)}`
  );
  state.translationSocket = socket;
  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.error) return;
    renderTranslationStream(payload);
  });
  socket.addEventListener("close", () => {
    if (state.translationSocket === socket) state.translationSocket = null;
  });
}

function renderTranslationStream(payload) {
  $("segmentProgress").hidden = false;
  $("segmentProgressCount").textContent = `${payload.completed} / ${payload.total}`;
  $("segmentProgressBar").style.width = `${payload.percent}%`;
  const container = $("subtitleList");
  container.innerHTML = "";
  for (const segment of payload.segments) {
    const node = document.createElement("div");
    node.className = `translation-segment${segment.translatedText ? " complete" : " pending"}`;
    node.innerHTML = `
      <div class="cue-time">${formatMs(segment.startMs)} - ${formatMs(segment.endMs)}</div>
      <div class="segment-source"></div>
      <div class="segment-translation"></div>
    `;
    node.querySelector(".segment-source").textContent = segment.sourceText;
    node.querySelector(".segment-translation").textContent =
      segment.translatedText || "Waiting for translation...";
    container.appendChild(node);
  }
  const latest = container.querySelector(".translation-segment.complete:last-of-type");
  if (latest) latest.scrollIntoView({ block: "nearest" });
}

async function loadSubtitles() {
  const videoId = $("videoId").value.trim();
  const lang = $("subtitleLang").value.trim() || "zh-TW";
  const response = await fetch(`/api/subtitles/${videoId}?lang=${encodeURIComponent(lang)}&format=json`);
  const container = $("subtitleList");
  if (!response.ok) {
    container.textContent = "No subtitles available.";
    return;
  }
  const payload = await response.json();
  container.innerHTML = "";
  for (const cue of payload.cues) {
    const node = document.createElement("div");
    node.className = "cue";
    node.innerHTML = `<div class="cue-time">${formatMs(cue.startMs)} - ${formatMs(cue.endMs)}</div><div class="cue-text"></div>`;
    node.querySelector(".cue-text").textContent = cue.text;
    container.appendChild(node);
  }
  await loadGlossary();
}

async function loadGlossary() {
  const videoId = $("videoId").value.trim();
  if (!videoId) return;
  const response = await fetch(`/api/videos/${videoId}/glossary`);
  if (!response.ok) return;
  const payload = await response.json();
  $("glossaryText").value = (payload.entries || [])
    .map((entry) => `${entry.source} => ${entry.target} | ${(entry.languages || []).join(",")}`)
    .join("\n");
}

async function saveGlossary() {
  const videoId = $("videoId").value.trim();
  if (!videoId) return;
  const entries = $("glossaryText")
    .value.split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [pair, languages = "zh-TW"] = line.split("|").map((part) => part.trim());
      const [source, target] = pair.split("=>").map((part) => part.trim());
      return { source, target, languages: languages.split(",").map((part) => part.trim()).filter(Boolean), caseSensitive: false };
    })
    .filter((entry) => entry.source && entry.target);
  await fetch(`/api/videos/${videoId}/glossary`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entries),
  });
  await loadGlossary();
}

const params = new URLSearchParams(location.search);
if (params.get("videoId")) {
  $("videoId").value = params.get("videoId");
}

$("jobForm").addEventListener("submit", createJob);
$("refreshJob").addEventListener("click", refreshJob);
$("loadSubtitles").addEventListener("click", loadSubtitles);
$("subtitleLang").addEventListener("change", () => {
  state.translationSocketKey = null;
  if (state.jobId) refreshJob();
});
$("saveGlossary").addEventListener("click", saveGlossary);
setInterval(refreshJob, 3000);
checkHealth();
