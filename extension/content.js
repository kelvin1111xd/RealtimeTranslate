const DEFAULTS = {
  enabled: true,
  apiBase: "http://127.0.0.1:8765",
  apiToken: "",
  language: "zh-TW",
  showSource: false,
  bilingual: false,
  order: "translation-first",
  displayMode: "normal",
  horizontalPosition: "center",
  fontSize: 28,
  verticalPosition: 12,
  stackMaxItems: 4,
  stackDirection: "up",
  stackPastOpacity: 0.55,
  stackPastScale: 0.82,
  background: true,
  shadow: true,
  backgroundOpacity: 0.62,
  autoLoad: true,
  autoCreateJob: false,
  toolbar: true,
  shortcutToggle: "Alt+S",
  shortcutMode: "Alt+B",
  shortcutFontUp: "Alt+=",
  shortcutFontDown: "Alt+-",
  shortcutUp: "Alt+ArrowUp",
  shortcutDown: "Alt+ArrowDown",
};

let translatedCues = [];
let sourceCues = [];
let settings = { ...DEFAULTS };
let activeVideoId = null;
let lastCueText = "";
let stackItems = [];
let loadInFlight = null;
let translationSocket = null;
let translationSocketRetry = null;
let apiStatus = "unknown";

function getVideoId() {
  const url = new URL(location.href);
  if (url.pathname.startsWith("/live/")) return url.pathname.split("/").filter(Boolean).pop();
  return url.searchParams.get("v");
}

function youtubeUrl() {
  const videoId = getVideoId();
  return videoId ? `https://www.youtube.com/watch?v=${videoId}` : location.href;
}

function ensureOverlay() {
  let overlay = document.getElementById("local-ai-subtitle-overlay");
  if (overlay) return overlay;

  overlay = document.createElement("div");
  overlay.id = "local-ai-subtitle-overlay";

  const toolbar = document.createElement("div");
  toolbar.id = "local-ai-subtitle-toolbar";
  toolbar.innerHTML = `
    <button type="button" data-action="toggle" title="Enable subtitles">字幕</button>
    <button type="button" data-action="stack" title="Toggle stacked subtitles">堆疊</button>
    <button type="button" data-action="mode" title="Toggle mono/bilingual">雙語</button>
    <select data-action="language" title="Target language">
      <option value="zh-TW">繁中</option>
      <option value="en">EN</option>
      <option value="ja">日本語</option>
    </select>
    <button type="button" data-action="up" title="Move subtitles up">↑</button>
    <button type="button" data-action="down" title="Move subtitles down">↓</button>
    <button type="button" data-action="fontDown" title="Smaller font">A-</button>
    <button type="button" data-action="fontUp" title="Larger font">A+</button>
    <button type="button" data-action="reload" title="Reload subtitles">↻</button>
  `;

  const lines = document.createElement("div");
  lines.className = "subtitle-lines";
  lines.innerHTML = `
    <div class="subtitle-line subtitle-source"></div>
    <div class="subtitle-line subtitle-translation"></div>
  `;

  const stack = document.createElement("div");
  stack.className = "subtitle-stack";

  overlay.appendChild(toolbar);
  overlay.appendChild(lines);
  overlay.appendChild(stack);
  const player = document.querySelector("#movie_player, .html5-video-player");
  if (player) {
    if (getComputedStyle(player).position === "static") player.style.position = "relative";
    player.appendChild(overlay);
    overlay.classList.remove("viewport-overlay");
  } else {
    document.body.appendChild(overlay);
    overlay.classList.add("viewport-overlay");
  }
  toolbar.addEventListener("click", handleToolbarClick);
  toolbar.addEventListener("change", handleToolbarChange);
  return overlay;
}

function applySettings() {
  const overlay = ensureOverlay();
  overlay.style.setProperty("--subtitle-font-size", `${settings.fontSize}px`);
  overlay.style.setProperty("--subtitle-bottom", `${settings.verticalPosition}%`);
  overlay.style.setProperty("--subtitle-bg-opacity", String(settings.backgroundOpacity));
  overlay.style.setProperty("--stack-past-opacity", String(settings.stackPastOpacity));
  overlay.style.setProperty("--stack-past-scale", String(settings.stackPastScale));
  overlay.classList.toggle("with-bg", Boolean(settings.background));
  overlay.classList.toggle("with-shadow", Boolean(settings.shadow));
  overlay.classList.toggle("disabled", !settings.enabled);
  overlay.classList.toggle("toolbar-hidden", !settings.toolbar);
  overlay.dataset.order = settings.order;
  overlay.dataset.mode = settings.displayMode;
  overlay.dataset.horizontal = settings.horizontalPosition;
  overlay.dataset.stackDirection = settings.stackDirection;

  const language = overlay.querySelector('[data-action="language"]');
  if (language) language.value = settings.language;
  const toggle = overlay.querySelector('[data-action="toggle"]');
  if (toggle) toggle.classList.toggle("active", settings.enabled);
  const mode = overlay.querySelector('[data-action="mode"]');
  if (mode) mode.classList.toggle("active", settings.bilingual);
  const stack = overlay.querySelector('[data-action="stack"]');
  if (stack) stack.classList.toggle("active", settings.displayMode === "stack");
}

async function loadSettings() {
  settings = await chrome.storage.sync.get(DEFAULTS);
  applySettings();
}

async function saveSettings(patch) {
  settings = { ...settings, ...patch };
  if (
    Object.hasOwn(patch, "displayMode") ||
    Object.hasOwn(patch, "stackMaxItems") ||
    Object.hasOwn(patch, "stackDirection")
  ) {
    stackItems = [];
    lastCueText = "";
  }
  await chrome.storage.sync.set(patch);
  applySettings();
}

async function fetchJson(url) {
  let response;
  try {
    response = await fetch(url, { headers: apiHeaders() });
  } catch {
    apiStatus = "offline";
    return null;
  }
  apiStatus = response.ok ? "online" : response.status === 401 ? "unauthorized" : "error";
  if (!response.ok) return null;
  return response.json();
}

function apiHeaders() {
  return settings.apiToken ? { Authorization: `Bearer ${settings.apiToken}` } : {};
}

async function loadCues() {
  if (loadInFlight) return loadInFlight;
  loadInFlight = (async () => {
    activeVideoId = getVideoId();
    if (!activeVideoId || !settings.autoLoad) return;

    const subtitlesUrl = `${settings.apiBase}/api/subtitles/${activeVideoId}?lang=${encodeURIComponent(settings.language)}&format=json`;
    const sourceUrl = `${settings.apiBase}/api/transcripts/${activeVideoId}`;
    const [translatedPayload, sourcePayload] = await Promise.all([
      fetchJson(subtitlesUrl),
      settings.showSource || settings.bilingual ? fetchJson(sourceUrl) : Promise.resolve(null),
    ]);

    translatedCues = translatedPayload?.cues || [];
    sourceCues = sourcePayload?.cues || [];
    connectSubtitleSocket();

    if (!translatedCues.length && settings.autoCreateJob) {
      await createTranslationJob();
    }
  })().finally(() => {
    loadInFlight = null;
  });
  return loadInFlight;
}

async function createTranslationJob() {
  try {
    const response = await fetch(`${settings.apiBase}/api/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...apiHeaders() },
      body: JSON.stringify({
        youtubeUrl: youtubeUrl(),
        sourceLanguage: "auto",
        targetLanguages: [settings.language],
        qualityMode: "quality",
      }),
    });
    apiStatus = response.ok ? "online" : response.status === 401 ? "unauthorized" : "error";
  } catch {
    apiStatus = "offline";
  }
}

function connectSubtitleSocket() {
  if (!activeVideoId || !settings.autoLoad) return;
  if (translationSocket) translationSocket.close();
  const protocol = settings.apiBase.startsWith("https:") ? "wss:" : "ws:";
  const base = settings.apiBase.replace(/^https?:/, "").replace(/\/$/, "");
  const query = new URLSearchParams({ lang: settings.language });
  if (settings.apiToken) query.set("token", settings.apiToken);
  const socket = new WebSocket(`${protocol}${base}/ws/subtitles/${activeVideoId}?${query}`);
  translationSocket = socket;
  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.cues) translatedCues = payload.cues;
  });
  socket.addEventListener("close", () => {
    if (translationSocket !== socket || !activeVideoId) return;
    translationSocket = null;
    clearTimeout(translationSocketRetry);
    translationSocketRetry = setTimeout(connectSubtitleSocket, 3000);
  });
  socket.addEventListener("error", () => socket.close());
}

function currentCue(cues, timeMs) {
  let low = 0;
  let high = cues.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const cue = cues[mid];
    if (timeMs < cue.startMs) high = mid - 1;
    else if (timeMs > cue.endMs) low = mid + 1;
    else return cue;
  }
  return null;
}

function renderSubtitle(timeMs) {
  const overlay = ensureOverlay();
  const sourceLine = overlay.querySelector(".subtitle-source");
  const translationLine = overlay.querySelector(".subtitle-translation");

  if (!settings.enabled) {
    sourceLine.textContent = "";
    translationLine.textContent = "";
    overlay.querySelector(".subtitle-stack").innerHTML = "";
    return;
  }

  const translated = currentCue(translatedCues, timeMs);
  const source = currentCue(sourceCues, timeMs);
  const showSource = settings.showSource || settings.bilingual;
  const showTranslation = !settings.showSource || settings.bilingual;
  const sourceText = showSource && source ? source.text : "";
  const translatedText = showTranslation && translated ? translated.text : "";
  const nextText = `${sourceText}\n---\n${translatedText}`;

  if (nextText === lastCueText) return;
  lastCueText = nextText;

  if (settings.displayMode === "stack") {
    sourceLine.textContent = "";
    translationLine.textContent = "";
    sourceLine.hidden = true;
    translationLine.hidden = true;
    renderStackSubtitle(sourceText, translatedText);
    return;
  }

  stackItems = [];
  overlay.querySelector(".subtitle-stack").innerHTML = "";
  sourceLine.textContent = sourceText;
  translationLine.textContent = translatedText;
  sourceLine.hidden = !sourceText;
  translationLine.hidden = !translatedText;
}

function renderStackSubtitle(sourceText, translatedText) {
  const overlay = ensureOverlay();
  const stack = overlay.querySelector(".subtitle-stack");
  const primaryText = translatedText || sourceText;
  if (!primaryText) return;

  const secondaryText = settings.bilingual && sourceText && translatedText ? sourceText : "";
  stackItems.unshift({
    primaryText,
    secondaryText,
    key: `${Date.now()}-${primaryText}`,
  });
  stackItems = stackItems.slice(0, Math.max(1, Number(settings.stackMaxItems) || 4));
  stack.innerHTML = "";

  stackItems.forEach((item, index) => {
    const node = document.createElement("div");
    node.className = "stack-subtitle-item";
    node.classList.toggle("current", index === 0);
    node.style.setProperty("--stack-index", String(index));
    node.style.setProperty(
      "--item-opacity",
      index === 0 ? "1" : String(Math.max(0.05, settings.stackPastOpacity ** index))
    );
    node.style.setProperty(
      "--item-scale",
      index === 0 ? "1" : String(Math.max(0.55, settings.stackPastScale ** index))
    );

    const primary = document.createElement("div");
    primary.className = "stack-primary";
    primary.textContent = item.primaryText;
    node.appendChild(primary);

    if (item.secondaryText) {
      const secondary = document.createElement("div");
      secondary.className = "stack-secondary";
      secondary.textContent = item.secondaryText;
      node.appendChild(secondary);
    }
    stack.appendChild(node);
  });
}

function tick() {
  const video = document.querySelector("video");
  if (video) renderSubtitle(video.currentTime * 1000);
  requestAnimationFrame(tick);
}

async function maybeReloadForNavigation() {
  const nextVideoId = getVideoId();
  if (nextVideoId && nextVideoId !== activeVideoId) {
    if (translationSocket) translationSocket.close();
    translatedCues = [];
    sourceCues = [];
    lastCueText = "";
    stackItems = [];
    await loadCues();
  }
}

async function handleToolbarClick(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const action = button.dataset.action;
  if (action === "toggle") await saveSettings({ enabled: !settings.enabled });
  if (action === "mode") await saveSettings({ bilingual: !settings.bilingual, showSource: false });
  if (action === "up") await saveSettings({ verticalPosition: Math.min(40, settings.verticalPosition + 2) });
  if (action === "down") await saveSettings({ verticalPosition: Math.max(4, settings.verticalPosition - 2) });
  if (action === "fontUp") await saveSettings({ fontSize: Math.min(72, settings.fontSize + 2) });
  if (action === "fontDown") await saveSettings({ fontSize: Math.max(16, settings.fontSize - 2) });
  if (action === "stack") {
    const displayMode = settings.displayMode === "stack" ? "normal" : "stack";
    stackItems = [];
    await saveSettings({ displayMode });
  }
  if (action === "reload") await loadCues();
}

async function handleToolbarChange(event) {
  const input = event.target.closest('[data-action="language"]');
  if (!input) return;
  await saveSettings({ language: input.value });
  translatedCues = [];
  await loadCues();
}

function shortcutMatches(event, shortcut) {
  if (!shortcut) return false;
  const parts = shortcut.split("+");
  const key = parts.pop();
  const wants = {
    alt: parts.includes("Alt"),
    ctrl: parts.includes("Ctrl"),
    shift: parts.includes("Shift"),
    meta: parts.includes("Meta"),
  };
  return (
    event.altKey === wants.alt &&
    event.ctrlKey === wants.ctrl &&
    event.shiftKey === wants.shift &&
    event.metaKey === wants.meta &&
    event.key.toLowerCase() === key.toLowerCase()
  );
}

async function handleShortcut(event) {
  const editable = event.target.closest("input, textarea, [contenteditable='true']");
  if (editable) return;
  if (shortcutMatches(event, settings.shortcutToggle)) {
    event.preventDefault();
    await saveSettings({ enabled: !settings.enabled });
  } else if (shortcutMatches(event, settings.shortcutMode)) {
    event.preventDefault();
    await saveSettings({ bilingual: !settings.bilingual, showSource: false });
  } else if (shortcutMatches(event, settings.shortcutFontUp)) {
    event.preventDefault();
    await saveSettings({ fontSize: Math.min(72, settings.fontSize + 2) });
  } else if (shortcutMatches(event, settings.shortcutFontDown)) {
    event.preventDefault();
    await saveSettings({ fontSize: Math.max(16, settings.fontSize - 2) });
  } else if (shortcutMatches(event, settings.shortcutUp)) {
    event.preventDefault();
    await saveSettings({ verticalPosition: Math.min(40, settings.verticalPosition + 2) });
  } else if (shortcutMatches(event, settings.shortcutDown)) {
    event.preventDefault();
    await saveSettings({ verticalPosition: Math.max(4, settings.verticalPosition - 2) });
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message.type === "get-state") {
      sendResponse({
        settings,
        videoId: getVideoId(),
        translatedCount: translatedCues.length,
        sourceCount: sourceCues.length,
        apiStatus,
      });
    }
    if (message.type === "set-settings") {
      await saveSettings(message.patch || {});
      if (message.reload) await loadCues();
      sendResponse({ ok: true });
    }
    if (message.type === "reload") {
      await loadCues();
      sendResponse({ ok: true });
    }
    if (message.type === "create-job") {
      await createTranslationJob();
      sendResponse({ ok: true });
    }
  })();
  return true;
});

loadSettings().then(loadCues).then(tick);
setInterval(maybeReloadForNavigation, 1000);
document.addEventListener("keydown", handleShortcut);
chrome.storage.onChanged.addListener(async () => {
  const previousLanguage = settings.language;
  const previousApiBase = settings.apiBase;
  const previousApiToken = settings.apiToken;
  const previousSourceMode = settings.showSource || settings.bilingual;
  await loadSettings();
  const nextSourceMode = settings.showSource || settings.bilingual;
  if (
    settings.language !== previousLanguage ||
    nextSourceMode !== previousSourceMode ||
    settings.apiBase !== previousApiBase ||
    settings.apiToken !== previousApiToken
  ) {
    await loadCues();
  }
});
