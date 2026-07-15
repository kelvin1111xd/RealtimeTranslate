const $ = (id) => document.getElementById(id);

let activeTabId = null;
let settings = { ...DEFAULTS };

function fillLanguages() {
  $("language").innerHTML = LANGUAGE_OPTIONS.map(
    ([value, label]) => `<option value="${value}">${label}</option>`
  ).join("");
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTabId = tab?.id || null;
  return tab;
}

function sendToTab(message) {
  if (!activeTabId) return Promise.resolve(null);
  return chrome.tabs.sendMessage(activeTabId, message).catch(() => null);
}

async function load() {
  fillLanguages();
  const stored = await chrome.storage.sync.get(DEFAULTS);
  settings = { ...DEFAULTS, ...stored };
  const tab = await getActiveTab();
  const state = await sendToTab({ type: "get-state" });
  if (state?.settings) settings = { ...settings, ...state.settings };
  $("videoState").textContent = state?.videoId
    ? `${state.videoId} · ${state.translatedCount || 0} cues`
    : tab?.url?.includes("youtube.com")
      ? "No video"
      : "Not YouTube";
  if (state?.apiStatus && state.apiStatus !== "online") {
    $("videoState").textContent += ` · API ${state.apiStatus}`;
  }
  render();
}

function render() {
  for (const key of ["enabled", "showSource", "bilingual"]) {
    $(key).checked = Boolean(settings[key]);
  }
  $("language").value = settings.language;
  $("displayMode").value = settings.displayMode;
  $("horizontalPosition").value = settings.horizontalPosition;
  $("verticalPosition").value = settings.verticalPosition;
  $("fontSize").value = settings.fontSize;
  $("stackMaxItems").value = settings.stackMaxItems;
}

async function patchSettings(patch, reload = false) {
  settings = { ...settings, ...patch };
  await chrome.storage.sync.set(patch);
  await sendToTab({ type: "set-settings", patch, reload });
  render();
}

for (const key of ["enabled", "showSource", "bilingual"]) {
  $(key).addEventListener("change", (event) => {
    const patch = { [key]: event.target.checked };
    if (key === "bilingual" && event.target.checked) patch.showSource = false;
    patchSettings(patch, key !== "enabled");
  });
}

$("language").addEventListener("change", (event) => {
  patchSettings({ language: event.target.value }, true);
});

$("displayMode").addEventListener("change", (event) => {
  patchSettings({ displayMode: event.target.value });
});

$("horizontalPosition").addEventListener("change", (event) => {
  patchSettings({ horizontalPosition: event.target.value });
});

$("verticalPosition").addEventListener("input", (event) => {
  patchSettings({ verticalPosition: Number(event.target.value) });
});

$("fontSize").addEventListener("input", (event) => {
  patchSettings({ fontSize: Number(event.target.value) });
});

$("stackMaxItems").addEventListener("input", (event) => {
  patchSettings({ stackMaxItems: Number(event.target.value) });
});

$("reload").addEventListener("click", () => sendToTab({ type: "reload" }));
$("createJob").addEventListener("click", () => sendToTab({ type: "create-job" }));

load();
