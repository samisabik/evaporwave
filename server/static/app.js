const $ = (sel) => document.querySelector(sel);

let cfg = null;
let live = {};
let selected = null;
let picking = false;
let draft = null;
let saveTimer = null;

const img = $("#stream");
const canvas = $("#overlay");
const ctx = canvas.getContext("2d");

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const noteName = (n) => `${NOTE_NAMES[n % 12]}${Math.floor(n / 12) - 1}`;

const getPath = (o, p) => p.split(".").reduce((a, k) => a[k], o);
function setPath(o, p, v) {
  const keys = p.split(".");
  const last = keys.pop();
  keys.reduce((a, k) => a[k], o)[last] = v;
}

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  Object.assign(el, props);
  el.append(...children);
  return el;
}

async function loadConfig() {
  cfg = await (await fetch("/api/config")).json();
  if (!cfg.tracks.find((t) => t.id === selected)) selected = cfg.tracks[0]?.id ?? null;
  renderTracks();
  renderEditor();
  renderSettings();
  showOsc();
}

function setSaveState(text, err = false) {
  const el = $("#save-state");
  el.textContent = text;
  el.classList.toggle("err", err);
}

function showOsc() {
  $("#osc").textContent = cfg.osc.enabled ? `${cfg.osc.host}:${cfg.osc.port}` : "Off";
}

function scheduleSave() {
  setSaveState("Saving…");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 400);
}

async function save() {
  clearTimeout(saveTimer);
  const res = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  if (res.ok) {
    setSaveState("Saved");
    showOsc();
  } else {
    const err = await res.json().catch(() => ({}));
    const detail = Array.isArray(err.detail)
      ? err.detail.map((d) => `${d.loc.slice(1).join(".")}: ${d.msg}`).join("; ")
      : err.detail ?? res.status;
    setSaveState("Save failed: " + detail, true);
  }
  return res.ok;
}

function selectedTrack() {
  return cfg?.tracks.find((t) => t.id === selected);
}

function select(id) {
  selected = id;
  for (const n of $("#tracks").children) n.classList.toggle("selected", n.dataset.id === selected);
  renderEditor();
}

// ---------- fields ----------

function fieldRow(root, obj, s, rerender) {
  if (s.show && !s.show(obj)) return;
  const val = getPath(obj, s.k);
  const clamp = (v) => Math.min(Math.max(v, s.min ?? -Infinity), s.max ?? Infinity);
  const commit = (v, structural = false) => {
    setPath(obj, s.k, v);
    scheduleSave();
    if (structural && rerender) rerender();
  };
  const num = (props = {}) => h("input", { type: "number", value: val, min: s.min ?? "", max: s.max ?? "", step: s.step ?? 1, ...props });
  let control;

  if (s.type === "select") {
    control = h("select");
    for (const o of s.options) {
      const [value, label] = Array.isArray(o) ? o : [o, o];
      control.add(new Option(label, value));
    }
    control.value = val;
    control.addEventListener("change", () => {
      const v = typeof val === "number" ? Number(control.value) : control.value;
      commit(v, true);
    });
  } else if (s.type === "check") {
    control = h("input", { type: "checkbox", checked: val });
    control.addEventListener("change", () => commit(control.checked, true));
  } else if (s.type === "text") {
    control = h("input", { type: "text", value: val });
    control.addEventListener("change", () => commit(control.value));
  } else if (s.type === "range") {
    const slider = h("input", { type: "range", min: s.min, max: s.max, step: s.step ?? 1, value: val });
    const box = num();
    slider.addEventListener("input", () => {
      box.value = slider.value;
      commit(Number(slider.value));
    });
    box.addEventListener("change", () => {
      const v = clamp(Number(box.value));
      box.value = slider.value = v;
      commit(v);
    });
    control = h("div", { className: "range" }, slider, box);
  } else if (s.type === "note") {
    const box = num();
    const name = h("span", { className: "note-name", textContent: noteName(val) });
    box.addEventListener("input", () => {
      const v = clamp(Math.round(Number(box.value)));
      name.textContent = noteName(v);
      commit(v);
    });
    control = h("div", { className: "with-note" }, box, name);
  } else {
    control = num();
    control.addEventListener("change", () => {
      const v = clamp(Number(control.value));
      control.value = v;
      commit(v);
    });
  }
  root.append(h("label", { textContent: s.label }), control);
}

function renderSections(root, obj, sections, rerender) {
  root.replaceChildren();
  for (const sec of sections) {
    const kv = h("div", { className: "kv" });
    for (const s of sec.fields) fieldRow(kv, obj, s, rerender);
    if (sec.title) root.append(h("h3", { textContent: sec.title }));
    root.append(kv);
  }
}

// ---------- track editor ----------

const FEATURES = [
  ["coverage", "Coverage"],
  ["motion", "Motion"],
  ["x", "X position"],
  ["y", "Y position"],
  ["brightness", "Brightness"],
  ["change", "Change"],
  ["blobs", "Blobs"],
];

const hsv = (t) => t.match.mode === "hsv";
const auto = (t) => t.signal.range === "auto";
const mode = (...m) => (t) => m.includes(t.midi.mode);
const notes = (t) => t.midi.mode === "note";
const pitched = (t) => notes(t) && t.midi.pitch !== "fixed";

const TRACK_SECTIONS = [
  { title: "Colour match", fields: [
    { k: "match.mode", label: "Method", type: "select", options: [["rgb", "RGB distance"], ["hsv", "Hue (HSV)"]] },
    { k: "match.spread", label: "Spread", type: "range", min: 0, max: 255, show: (t) => !hsv(t) },
    { k: "match.hue_tol", label: "Hue ± degrees", type: "range", min: 0, max: 90, show: hsv },
    { k: "match.sat_min", label: "Saturation min %", type: "range", min: 0, max: 100, show: hsv },
    { k: "match.sat_max", label: "Saturation max %", type: "range", min: 0, max: 100, show: hsv },
    { k: "match.val_min", label: "Brightness min %", type: "range", min: 0, max: 100, show: hsv },
    { k: "match.val_max", label: "Brightness max %", type: "range", min: 0, max: 100, show: hsv },
    { k: "match.blur", label: "Blur", type: "range", min: 0, max: 15 },
    { k: "match.open", label: "Remove specks", type: "range", min: 0, max: 10 },
    { k: "match.close", label: "Fill holes", type: "range", min: 0, max: 10 },
  ] },
  { title: "Motion", fields: [
    { k: "motion.threshold", label: "Threshold", type: "range", min: 1, max: 255 },
    { k: "motion.colour_only", label: "Coloured pixels only", type: "check" },
  ] },
  { title: "Signal", fields: [
    { k: "signal.feature", label: "Drive from", type: "select", options: FEATURES },
    { k: "signal.range", label: "Range", type: "select", options: [["auto", "Auto (rolling min/max)"], ["fixed", "Fixed"]] },
    { k: "signal.window_s", label: "Window (s)", type: "range", min: 0.5, max: 120, step: 0.5, show: auto },
    { k: "signal.in_min", label: "Input min", type: "number", step: 0.001, show: (t) => !auto(t) },
    { k: "signal.in_max", label: "Input max", type: "number", step: 0.001, show: (t) => !auto(t) },
    { k: "signal.attack", label: "Attack (s)", type: "range", min: 0, max: 5, step: 0.01 },
    { k: "signal.release", label: "Release (s)", type: "range", min: 0, max: 5, step: 0.01 },
    { k: "signal.curve", label: "Curve", type: "range", min: 0.1, max: 5, step: 0.05 },
    { k: "signal.invert", label: "Invert", type: "check" },
  ] },
  { title: "MIDI", fields: [
    { k: "midi.mode", label: "Output", type: "select", options: [["cc", "Control change"], ["note", "Notes (triggered)"], ["drone", "Drone (held note + CC)"]] },
    { k: "midi.channel", label: "Channel", type: "number", min: 1, max: 16 },
    { k: "midi.cc", label: "CC number", type: "number", min: 0, max: 127, show: mode("cc", "drone") },
    { k: "midi.out_min", label: "CC min", type: "range", min: 0, max: 127, show: mode("cc", "drone") },
    { k: "midi.out_max", label: "CC max", type: "range", min: 0, max: 127, show: mode("cc", "drone") },
    { k: "midi.pitch", label: "Pitch from", type: "select", options: [["fixed", "Fixed note"], ["level", "Level"], ["x", "X position"], ["y", "Y position"]], show: notes },
    { k: "midi.note", label: "Note", type: "note", min: 0, max: 127, show: (t) => t.midi.mode === "drone" || (notes(t) && !pitched(t)) },
    { k: "midi.note_low", label: "Lowest note", type: "note", min: 0, max: 127, show: pitched },
    { k: "midi.note_high", label: "Highest note", type: "note", min: 0, max: 127, show: pitched },
    { k: "midi.quantize", label: "Snap to scale", type: "check", show: pitched },
    { k: "midi.retrigger", label: "Retrigger on pitch change", type: "check", show: pitched },
    { k: "midi.threshold", label: "Trigger level", type: "range", min: 0, max: 1, step: 0.01, show: notes },
    { k: "midi.hysteresis", label: "Release gap", type: "range", min: 0, max: 0.5, step: 0.01, show: notes },
    { k: "midi.cooldown_s", label: "Min time between notes (s)", type: "range", min: 0, max: 2, step: 0.01, show: notes },
    { k: "midi.velocity_from", label: "Velocity from", type: "select", options: [["fixed", "Fixed"], ["level", "Level"]], show: notes },
    { k: "midi.velocity", label: "Velocity", type: "range", min: 1, max: 127, show: (t) => t.midi.mode === "drone" || (notes(t) && t.midi.velocity_from === "fixed") },
    { k: "midi.vel_min", label: "Velocity min", type: "range", min: 1, max: 127, show: (t) => notes(t) && t.midi.velocity_from === "level" },
    { k: "midi.vel_max", label: "Velocity max", type: "range", min: 1, max: 127, show: (t) => notes(t) && t.midi.velocity_from === "level" },
  ] },
];

const featureMax = {};

function renderEditor() {
  const t = selectedTrack();
  $("#editor").hidden = !t;
  if (!t) return;
  $("#editor-title").textContent = `Track ${t.id}`;

  const feats = $("#features");
  feats.replaceChildren();
  for (const [key, label] of FEATURES) {
    const row = h("div", { className: "f-row" },
      h("span", { className: "f-name", textContent: label }),
      h("div", { className: "meter" }, h("div", { className: "fill" })),
      h("span", { className: "f-val", textContent: "-" }));
    row.dataset.f = key;
    row.title = "Drive the output from this feature";
    row.classList.toggle("on", t.signal.feature === key);
    row.addEventListener("click", () => {
      t.signal.feature = key;
      scheduleSave();
      renderEditor();
    });
    feats.append(row);
  }
  renderSections($("#editor-fields"), t, TRACK_SECTIONS, renderEditor);
}

function updateFeatures(d) {
  for (const row of $("#features").children) {
    const f = row.dataset.f;
    const v = d?.features[f];
    if (v === undefined) continue;
    const bounded = ["coverage", "motion", "x", "y", "brightness"].includes(f);
    featureMax[f] = Math.max((featureMax[f] ?? 0) * 0.998, v, bounded ? 0 : 1e-6);
    const scale = bounded && f !== "coverage" && f !== "motion" ? 1 : featureMax[f] || 1;
    row.querySelector(".fill").style.width = `${Math.min(v / scale, 1) * 100}%`;
    row.querySelector(".f-val").textContent = f === "blobs" ? v : v < 10 ? v.toFixed(3) : v.toFixed(1);
  }
}

// ---------- tracks panel ----------

function renderTracks() {
  const root = $("#tracks");
  root.replaceChildren();
  for (const t of cfg.tracks) {
    const el = $("#track-tpl").content.firstElementChild.cloneNode(true);
    el.dataset.id = t.id;
    const idInput = el.querySelector(".t-id");
    idInput.value = t.id;
    idInput.addEventListener("input", () => {
      idInput.value = idInput.value.toLowerCase().replace(/[^a-z0-9_-]/g, "");
      const taken = cfg.tracks.some((x) => x !== t && x.id === idInput.value);
      idInput.classList.toggle("bad", !idInput.value || taken);
    });
    idInput.addEventListener("change", () => {
      const v = idInput.value;
      if (!v || cfg.tracks.some((x) => x !== t && x.id === v)) {
        idInput.value = t.id;
        idInput.classList.remove("bad");
        return;
      }
      if (selected === t.id) selected = v;
      t.id = v;
      el.dataset.id = v;
      renderEditor();
      scheduleSave();
    });
    idInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") idInput.blur();
      if (e.key === "Escape") { idInput.value = t.id; idInput.blur(); }
    });
    for (const input of el.querySelectorAll("[data-k]")) {
      const k = input.dataset.k;
      if (input.type === "checkbox") input.checked = t[k];
      else input.value = "#" + t[k];
      input.addEventListener("input", () => {
        if (input.type === "checkbox") t[k] = input.checked;
        else t[k] = input.value.slice(1);
        syncTrackEl(el, t);
        scheduleSave();
      });
    }
    el.querySelector(".del").addEventListener("click", (e) => {
      e.stopPropagation();
      cfg.tracks = cfg.tracks.filter((x) => x !== t);
      if (selected === t.id) selected = cfg.tracks[0]?.id ?? null;
      renderTracks();
      renderEditor();
      scheduleSave();
    });
    el.addEventListener("click", () => {
      if (selected !== t.id) select(t.id);
    });
    syncTrackEl(el, t);
    root.append(el);
  }
}

function syncTrackEl(el, t) {
  el.classList.toggle("selected", t.id === selected);
  el.classList.toggle("off", !t.enable);
  el.querySelector('[data-k="color"]').value = "#" + t.color;
}

function outputText(t, d) {
  if (!d || !t.enable) return "Off";
  const gate = `<i class="gate${d.gate ? " on" : ""}"></i>`;
  if (t.midi.mode === "cc") return `CC${t.midi.cc} <b>${d.value}</b>`;
  if (t.midi.mode === "drone") return `${gate}${noteName(t.midi.note)} <b>${d.value}</b>`;
  return `${gate}<b>${d.note != null ? noteName(d.note) : "-"}</b>`;
}

$("#add-track").addEventListener("click", async () => {
  let n = cfg.tracks.length + 1;
  while (cfg.tracks.some((t) => t.id === `track${n}`)) n++;
  cfg.tracks.push({ id: `track${n}`, color: "ffffff", roi: [0, 0, 1, 1], midi: { channel: Math.min(n, 16) } });
  selected = `track${n}`;
  if (await save()) await loadConfig();
});

$("#panic").addEventListener("click", () => fetch("/api/panic", { method: "POST" }));

// ---------- settings ----------

function renderSettings() {
  renderSections($("#music-form"), cfg.music, [{ fields: [
    { k: "root", label: "Root", type: "select", options: NOTE_NAMES.map((n, i) => [i, n]) },
    { k: "scale", label: "Scale", type: "select", options: [
      "chromatic", "major", "minor", "dorian", "phrygian", "lydian", "mixolydian",
      "pentatonic_major", "pentatonic_minor", "blues", "whole_tone",
    ].map((s) => [s, s.replace("_", " ").replace(/^./, (c) => c.toUpperCase())]) },
  ] }]);

  renderSections($("#source-form"), cfg.source, [{ fields: [
    { k: "type", label: "Type", type: "select", options: [["file", "Video file"], ["camera", "USB camera"], ["picamera", "Pi camera"]] },
    { k: "path", label: "File path", type: "text" },
    { k: "device", label: "Camera index", type: "number", min: 0 },
    { k: "width", label: "Width", type: "number", min: 1 },
    { k: "height", label: "Height", type: "number", min: 1 },
    { k: "fps", label: "FPS", type: "number", min: 1, max: 120 },
  ] }]);

  renderSections($("#osc-form"), cfg.osc, [{ fields: [
    { k: "enabled", label: "Enabled", type: "check" },
    { k: "host", label: "Host", type: "text" },
    { k: "port", label: "Port", type: "number", min: 1, max: 65535 },
  ] }]);

  const perf = $("#perf-form");
  renderSections(perf, cfg.processing, [{ fields: [
    { k: "width", label: "Analysis width", type: "number", min: 64, max: 1920 },
  ] }]);
  const stream = h("div");
  renderSections(stream, cfg.stream, [{ fields: [
    { k: "width", label: "Stream width", type: "number", min: 64, max: 1920 },
    { k: "fps", label: "Stream FPS", type: "number", min: 1, max: 60 },
    { k: "jpeg_quality", label: "JPEG quality", type: "range", min: 10, max: 100 },
  ] }]);
  perf.firstChild.append(...stream.firstChild.childNodes);
}

// ---------- viewer ----------

let view = "raw";
function setView(v) {
  view = v;
  img.src = `/stream.mjpg?view=${v}`;
  for (const b of document.querySelectorAll(".tools [data-view]")) b.classList.toggle("on", b.dataset.view === v);
}
for (const b of document.querySelectorAll(".tools [data-view]")) {
  b.addEventListener("click", () => setView(b.dataset.view));
}

$("#pick-btn").addEventListener("click", () => {
  picking = !picking;
  $("#pick-btn").classList.toggle("on", picking);
  $("#frame").classList.toggle("picking", picking);
});

$("#full-roi-btn").addEventListener("click", () => {
  const t = selectedTrack();
  if (!t) return;
  t.roi = [0, 0, 1, 1];
  scheduleSave();
});

function norm(e) {
  const r = canvas.getBoundingClientRect();
  return [
    Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1),
    Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1),
  ];
}

canvas.addEventListener("pointerdown", async (e) => {
  const [x, y] = norm(e);
  const t = selectedTrack();
  if (!t) return;
  if (picking) {
    const res = await fetch("/api/pick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y }),
    });
    if (res.ok) {
      t.color = (await res.json()).color;
      renderTracks();
      scheduleSave();
    }
    picking = false;
    $("#pick-btn").classList.remove("on");
    $("#frame").classList.remove("picking");
    return;
  }
  canvas.setPointerCapture(e.pointerId);
  draft = { x0: x, y0: y, x1: x, y1: y };
});

canvas.addEventListener("pointermove", (e) => {
  if (!draft) return;
  [draft.x1, draft.y1] = norm(e);
});

canvas.addEventListener("pointerup", () => {
  if (!draft) return;
  const t = selectedTrack();
  const x = Math.min(draft.x0, draft.x1);
  const y = Math.min(draft.y0, draft.y1);
  const w = Math.abs(draft.x1 - draft.x0);
  const h = Math.abs(draft.y1 - draft.y0);
  draft = null;
  if (t && w > 0.01 && h > 0.01) {
    t.roi = [x, y, w, h];
    scheduleSave();
  }
});

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "500 11px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textBaseline = "top";
  ctx.lineWidth = 1;
  const usedLabels = new Set();

  for (const t of cfg?.tracks ?? []) {
    const [rx, ry, rw, rh] = t.roi;
    const x = Math.round(rx * w) + 0.5;
    const y = Math.round(ry * h) + 0.5;
    const isSel = t.id === selected;
    ctx.globalAlpha = !t.enable ? 0.3 : isSel ? 1 : 0.6;
    ctx.strokeStyle = "#" + t.color;
    ctx.setLineDash(isSel ? [] : [2, 3]);
    ctx.strokeRect(x, y, Math.round(rw * w) - 1, Math.round(rh * h) - 1);

    const d = live[t.id];
    let label = t.id;
    if (d && t.enable) label += t.midi.mode === "note" ? `  ${d.note != null ? noteName(d.note) : "-"}` : `  ${d.value}`;
    let ly = y;
    while (usedLabels.has(`${x},${ly}`)) ly += 16;
    usedLabels.add(`${x},${ly}`);
    ctx.fillStyle = "rgba(0,0,0,0.75)";
    ctx.fillRect(x - 0.5, ly - 0.5, ctx.measureText(label).width + 10, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 4.5, ly + 2.5);

    if (d && d.present) {
      const cx = Math.round(d.features.x * w) + 0.5;
      const cy = Math.round(d.features.y * h) + 0.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
      ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);

  if (draft) {
    ctx.strokeStyle = "#fff";
    ctx.setLineDash([2, 2]);
    ctx.strokeRect(draft.x0 * w, draft.y0 * h, (draft.x1 - draft.x0) * w, (draft.y1 - draft.y0) * h);
  }
  requestAnimationFrame(draw);
}

// Scale the stream to the largest size that fits the stage, keeping aspect ratio,
// so the overlay canvas can sit exactly on top of it.
function fitFrame() {
  if (!img.naturalWidth) return;
  const stage = $(".stage");
  const cs = getComputedStyle(stage);
  const aw = stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const ah = stage.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  const ratio = img.naturalWidth / img.naturalHeight;
  const narrow = matchMedia("(max-width: 860px)").matches;
  const width = narrow ? aw : Math.min(aw, ah * ratio);
  img.style.width = `${Math.floor(width)}px`;
  img.style.height = `${Math.floor(width / ratio)}px`;
}
img.addEventListener("load", fitFrame);
window.addEventListener("resize", fitFrame);
setInterval(fitFrame, 1000);

// ---------- live data ----------

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (msg) => {
    const s = JSON.parse(msg.data);
    live = Object.fromEntries(s.tracks.map((t) => [t.id, t]));
    $("#fps").textContent = s.fps.toFixed(1);
    $("#frame-count").textContent = s.frame;
    $("#dot").className = s.error ? "err" : "live";
    $("#dot").title = s.error ?? "";
    for (const el of document.querySelectorAll(".track")) {
      const d = live[el.dataset.id];
      const t = cfg.tracks.find((x) => x.id === el.dataset.id);
      if (!t) continue;
      el.querySelector(".fill").style.width = d ? `${d.level * 100}%` : "0";
      el.querySelector(".t-out").innerHTML = outputText(t, d);
    }
    updateFeatures(live[selected]);
  };
  ws.onclose = () => {
    $("#dot").className = "err";
    $("#fps").textContent = "-";
    setTimeout(connect, 1000);
  };
}

loadConfig().then(() => {
  setView("raw");
  connect();
  requestAnimationFrame(draw);
});
