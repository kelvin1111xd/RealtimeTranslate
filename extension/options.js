const $ = (id) => document.getElementById(id);

function fillLanguages() {
  $("language").innerHTML = LANGUAGE_OPTIONS.map(
    ([value, label]) => `<option value="${value}">${label}</option>`
  ).join("");
}

async function load() {
  fillLanguages();
  const settings = await chrome.storage.sync.get(DEFAULTS);
  for (const [key, value] of Object.entries({ ...DEFAULTS, ...settings })) {
    const input = $(key);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value;
  }
}

function readSettings() {
  const payload = {};
  for (const key of Object.keys(DEFAULTS)) {
    const input = $(key);
    if (!input) continue;
    if (input.type === "checkbox") payload[key] = input.checked;
    else if (input.type === "number" || input.type === "range") payload[key] = Number(input.value);
    else payload[key] = input.value.trim();
  }
  if (payload.bilingual) payload.showSource = false;
  return payload;
}

async function save() {
  await chrome.storage.sync.set(readSettings());
  $("status").textContent = "Saved.";
  setTimeout(() => ($("status").textContent = ""), 1200);
}

$("save").addEventListener("click", save);
load();
