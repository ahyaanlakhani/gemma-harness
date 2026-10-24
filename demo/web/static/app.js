"use strict";
// Live race page: reads the server's event stream and fills the two lanes,
// the scoreboard, and the whole-pack table.
// All model and tool text goes in with textContent, never as HTML.

const $ = (sel) => document.querySelector(sel);

const FAULT_NAMES = {
  bad_names: "misspelled tool", bad_args: "wrong args", blank: "blank reply", crash_tools: "tool crash",
};
// Mirrors gemma_harness/trace.py RECOVERY_LABELS.
const RECOVERY = new Set([
  "PHANTOM_TOOL_LINTED", "RETRIED", "SILENT_FAIL_PREVENTED", "TOOL_NAME_FIXED", "UNKNOWN_TOOL",
  "ARGS_COERCED", "ARGS_MISSING", "TOOL_ERROR", "TOOL_TIMEOUT", "DEDUPED", "STEP_CAP",
  "BACKOFF_429", "BACKOFF_5XX", "TRUNCATED",
]);
// Plain-English names for the screen; the trace keeps the real labels (shown on hover).
const DISPLAY = {
  ARGS_COERCED: "ARGS FIXED", DEDUPED: "REPEAT SKIPPED", PHANTOM_TOOL_LINTED: "FAKE TOOL CAUGHT",
  SILENT_FAIL_PREVENTED: "BLANK CAUGHT", BACKOFF_429: "QUOTA WAIT", BACKOFF_5XX: "SERVER WAIT",
};
const VERDICT_TEXT = { SILENT_FAILURE: "SILENT FAILURE", FALLBACK: "SAFE FALLBACK", ANSWERED: "ANSWERED", ERROR: "API ERROR" };
const LANE_NAMES = { raw: "Raw loop: ", harness: "Harness: " };
const LANES = { raw: $("#lane-raw"), harness: $("#lane-harness") };
const SCORE_KEY = "gemma-harness-score";
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const prompts = new Map();
let info = { mode: null, model: null };
let busy = false;
let hasReplays = false;
let lastTrigger = null;   // the button that started the current stream, refocused after
// The run on screen: its mode, what each lane ended with, and the harness fixes seen.
let current = null;

// ---------- small helpers ----------
const short = (s, n = 140) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const faultList = (faults) => (faults.length ? faults.map((f) => FAULT_NAMES[f] || f).join(", ") : "no chaos");
const chaosText = (faults) => (faults.length ? `chaos: ${faultList(faults)}` : "no chaos");
const fmtArgs = (args) => Object.entries(args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
const pretty = (label) => DISPLAY[label] || label.replaceAll("_", " ");
const isFix = (label) => RECOVERY.has(label) && !label.startsWith("BACKOFF_");
const fixesOf = (labels) => labels.filter(isFix);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

// ---------- banner (keeps its space, so the page never jumps) ----------
function showBanner(text, kind = "") {
  const b = $("#banner");
  b.textContent = text;
  b.className = `banner ${kind}`.trim();
}
function hideBanner() {
  const b = $("#banner");
  b.textContent = "";
  b.className = "banner empty";
}

// ---------- controls ----------
function setBusy(value) {
  busy = value;
  for (const c of document.querySelectorAll("#run, #run-all, #prompt, #replays, .chaos input")) c.disabled = value;
  $("#replay").disabled = value || !hasReplays;
  if (!value && lastTrigger && (document.activeElement === document.body || document.activeElement === null)) {
    lastTrigger.focus({ preventScroll: true });   // keep the lanes in view
  }
}

function selectedFaults() {
  return [...document.querySelectorAll(".chaos input:checked")].map((c) => c.value);
}

function showPrompt(id) {
  const p = prompts.get(id);
  $("#prompt-text").textContent = p ? `“${p.prompt}”` : "";
  let expect = "";
  if (p && (info.mode === "fake" || info.source === "fake")) expect = "Fake model: scripted replies that always look up the weather, whatever the prompt.";
  else if (p && p.control) expect = "Control prompt: the harness should not need to step in.";
  else if (p && p.expect && p.expect.length) expect = `Even without chaos, watch the harness lane for: ${p.expect.map(pretty).join(", ")}`;
  $("#prompt-expect").textContent = expect;
}

async function loadInfo() {
  try {
    info = await getJSON("/api/info");
  } catch { return; }   // older server: no badge
  const badge = $("#mode-badge");
  badge.textContent = info.mode === "fake" ? "FAKE MODEL"
    : info.mode === "recorded" ? `RECORDED RUNS${info.source === "fake" ? " · FAKE MODEL" : ""}`
    : `LIVE · ${info.model}`;
  badge.className = `mode-badge ${info.mode === "live" ? "live" : "fake"}`;
  badge.hidden = false;
  showPrompt(Number($("#prompt").value));
}

async function loadPrompts() {
  const list = await getJSON("/api/prompts");
  const sel = $("#prompt");
  sel.replaceChildren();
  for (const p of list) {
    prompts.set(p.id, p);
    const opt = el("option", "", `#${p.id} · ${short(p.prompt, 56)}${p.control ? " (control)" : ""}`);
    opt.value = p.id;
    sel.append(opt);
  }
  $("#run-all").textContent = `RUN ALL ${list.length}`;
  showPrompt(Number(sel.value));
}

function outcomeText(r) {
  if (!r.raw && !r.harness) return "";
  const mark = (v) => ({ SILENT_FAILURE: "✕", ANSWERED: "✓", FALLBACK: "◐", ERROR: "!" }[v] || "?");
  return ` · raw ${mark(r.raw)} harness ${mark(r.harness)}`;
}

async function loadReplays({ selectNewest = false } = {}) {
  const list = await getJSON("/api/replays");
  const sel = $("#replays");
  const previous = sel.value;
  sel.replaceChildren();
  hasReplays = list.length > 0;
  if (!hasReplays) {
    const opt = el("option", "", "No saved runs yet");
    opt.value = "";
    sel.append(opt);
  }
  for (const r of list) {
    const fake = r.mode === "fake" ? " · fake model" : "";
    const opt = el("option", "", `${clock(r.saved_at)} · #${r.prompt_id} · ${faultList(r.faults)}${outcomeText(r)}${fake}`);
    opt.value = r.name;
    sel.append(opt);
  }
  if (!selectNewest && list.some((r) => r.name === previous)) sel.value = previous;
  $("#replay").disabled = busy || !hasReplays;
}

// ---------- lanes ----------
function setVerdict(side, cls, title, ...details) {
  const v = LANES[side].querySelector(".verdict");
  v.className = `verdict ${cls}`;
  const strong = el("strong");
  strong.append(el("span", "visually-hidden", LANE_NAMES[side]), document.createTextNode(title));
  v.replaceChildren(strong);
  for (const d of details) if (d) v.append(typeof d === "string" ? el("p", "", d) : d);
}

function idleLanes() {
  for (const side of Object.keys(LANES)) {
    LANES[side].querySelector(".feed").replaceChildren(el("li", "hint", "Pick a prompt, switch on some chaos, and press RUN."));
    setVerdict(side, "idle", "Waiting");
  }
}

function startLanes() {
  for (const side of Object.keys(LANES)) {
    LANES[side].querySelector(".feed").replaceChildren();
    setVerdict(side, "pending", "Running");
  }
}

function kindOf(side, ev) {
  if (ev.label === "CHAOS") return "chaos";
  if (ev.label === "CRASH" || ev.label === "ERROR") return "fail";
  if (side === "raw" && ev.label === "STEP_CAP") return "fail";
  if (side === "raw" && ev.label === "FINAL" && !(ev.text || "").trim()) return "fail";
  if (RECOVERY.has(ev.label)) return "fix";
  if (["TOOL_OK", "TOOL_RESULT", "FINAL"].includes(ev.label)) return "ok";
  return "neutral";
}

function describe(ev) {
  switch (ev.label) {
    case "CALL": {
      const bits = [`step ${ev.step}`];
      if (ev.ms != null) bits.push(`${ev.ms} ms`);
      if (ev.thinking) bits.push(`thinking ${ev.thinking}`);
      if (ev.think_tok) bits.push(`${ev.think_tok} think tokens`);
      return bits.join(" · ");
    }
    case "TOOL_CALL": return `${ev.name}(${fmtArgs(ev.args)})`;
    case "TOOL_RESULT": return `${ev.name} → ${short(JSON.stringify(ev.result), 90)}`;
    case "TOOL_OK": return `${ev.tool} ran`;
    case "FINAL":
      if (ev.text != null) return ev.text.trim() ? `“${short(ev.text)}”` : "(empty reply: the user sees nothing)";
      return `answer ready (${ev.chars} chars)`;
    case "CRASH": return ev.error;
    case "ERROR": return ev.message;
    case "CHAOS": return `injected: ${FAULT_NAMES[ev.fault] || ev.fault}`;
    case "PHANTOM_TOOL_LINTED": return `prompt names a tool that doesn't exist: ${ev.names.join(", ")}`;
    case "RETRIED": return `${ev.reason} → asked again with a hint`;
    case "SILENT_FAIL_PREVENTED": return "blank reply replaced by an honest fallback";
    case "TOOL_NAME_FIXED": return `${ev.got} → ${ev.used}`;
    case "UNKNOWN_TOOL": return `${ev.got} isn't a tool → told the model what exists`;
    case "ARGS_COERCED": return ev.fixes.join(", ");
    case "ARGS_MISSING": return `${ev.tool} needs ${ev.missing.join(", ")} → asked the model for it`;
    case "TOOL_ERROR": return `${ev.tool}: ${ev.err} → sent back so the model can adapt`;
    case "TOOL_TIMEOUT": return `${ev.tool} timed out → sent back so the model can adapt`;
    case "DEDUPED": return `${ev.tool} called again with the same args → reused the result`;
    case "STEP_CAP": return "step limit reached → stopped the loop";
    case "TRUNCATED": return `reply cut off at the token limit (${ev.chars} chars) → flagged, not hidden`;
    case "BACKOFF_429":
    case "BACKOFF_5XX": return `waiting ${ev.wait_s}s before retrying${ev.reason ? ` (${ev.reason})` : ""}`;
    default:
      return Object.entries(ev)
        .filter(([k]) => !["side", "label", "t", "step"].includes(k))
        .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(" ");
  }
}

function appendEvent(side, ev) {
  const feed = LANES[side].querySelector(".feed");
  const li = el("li", `ev ${kindOf(side, ev)}`);
  const tag = el("span", "tag", pretty(ev.label));
  if (DISPLAY[ev.label]) tag.title = ev.label;
  li.append(tag, el("span", "text", describe(ev)));
  feed.append(li);
  feed.scrollTop = feed.scrollHeight;
}

function fixList(side) {
  if (side !== "harness" || !current) return null;
  const fixes = [...new Set(fixesOf(current.labels))];
  return fixes.length ? el("p", "fix-list", fixes.map((l) => `✓ ${pretty(l)}`).join("  ")) : null;
}

function showVerdict(side, ev) {
  const text = (ev.text || "").trim();
  const quote = text ? `“${short(text, 160)}”` : "";
  if (ev.verdict === "SILENT_FAILURE") {
    setVerdict(side, "silent_failure", "SILENT FAILURE", "The user gets a blank reply or a crash, with no explanation.");
  } else if (ev.verdict === "FALLBACK") {
    setVerdict(side, "fallback", `SAFE FALLBACK · ${ev.fixes} FIX${ev.fixes === 1 ? "" : "ES"}`,
      text ? `Couldn't finish, so it told the user why: ${quote}` : "", fixList(side));
  } else {
    const fixes = ev.fixes ? ` · ${ev.fixes} FIX${ev.fixes === 1 ? "" : "ES"}` : "";
    setVerdict(side, "answered", `ANSWERED${fixes}`, quote, fixList(side));
  }
}

function showError(side, message) {
  const hint = hasReplays ? " Try a saved run with REPLAY." : "";
  setVerdict(side, "error", "API ERROR", `${message.replace(/\.+$/, "")}.${hint}`);
}

// ---------- scoreboard ----------
// Counts live and fake runs (never replays) for this browser tab.
const emptyScore = () => ({ runs: 0, fake: 0, raw_silent: 0, harness_silent: 0, fixes: 0, labels: {} });
let score = emptyScore();

function loadScore() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(SCORE_KEY) || "null");
    if (saved && typeof saved.runs === "number") score = { ...emptyScore(), ...saved };
  } catch { /* storage unavailable: start from zero */ }
}

function saveScore() {
  try { sessionStorage.setItem(SCORE_KEY, JSON.stringify(score)); } catch { /* ignore */ }
}

function renderScore() {
  $("#s-runs").textContent = String(score.runs);
  $("#s-raw").textContent = String(score.raw_silent);
  $("#s-raw").className = score.raw_silent ? "bad" : "";
  $("#s-harness").textContent = String(score.harness_silent);
  $("#s-harness").className = score.runs ? (score.harness_silent ? "bad" : "good") : "";
  $("#s-fixes").textContent = String(score.fixes);
  $("#score-note").textContent = score.runs === 0
    ? "Counts every run in this tab, not replays."
    : score.fake ? `${score.fake} of ${score.runs} on the fake model` : "all on live Gemma 4";
  const list = $("#s-labels");
  list.replaceChildren();
  for (const [label, n] of Object.entries(score.labels).sort((a, b) => b[1] - a[1])) {
    const li = el("li", "", `✓ ${pretty(label)}`);
    li.append(el("b", "", ` ×${n}`));
    list.append(li);
  }
}

function tally(run) {
  if (run.mode === "replay" || run.counted || !run.raw || !run.harness) return;
  if (run.raw === "ERROR" || run.harness === "ERROR") return;   // API trouble isn't a loop result
  run.counted = true;
  score.runs += 1;
  if (run.mode === "fake") score.fake += 1;
  if (run.raw === "SILENT_FAILURE") score.raw_silent += 1;
  if (run.harness === "SILENT_FAILURE") score.harness_silent += 1;
  for (const label of fixesOf(run.labels)) {
    score.fixes += 1;
    score.labels[label] = (score.labels[label] || 0) + 1;
  }
  saveScore();
  renderScore();
}

// ---------- whole pack ----------
function verdictPill(verdict, fixes) {
  const cls = (verdict || "error").toLowerCase();
  const extra = verdict === "ANSWERED" && fixes ? ` · ${fixes}` : "";
  return el("span", `pill ${cls}`, `${VERDICT_TEXT[verdict] || verdict}${extra}`);
}

function setLiveRegions(on) {
  for (const lane of Object.values(LANES)) lane.querySelector(".verdict").setAttribute("aria-live", on ? "polite" : "off");
}

function startSweep(ev) {
  $("#sweep").hidden = false;
  $("#sweep-rows").replaceChildren();
  $("#sweep-summary").replaceChildren();
  const fake = ev.mode === "fake" || ev.original_mode === "fake" ? " · fake model" : "";
  const recorded = ev.mode === "recorded" ? " · recorded" : "";
  $("#sweep-meta").textContent = `${ev.ids.length} prompts · ${chaosText(ev.faults)}${recorded}${fake}`;
  setLiveRegions(false);   // 15 prompts × 2 verdicts would flood a screen reader
}

function addSweepRow(ev) {
  const p = prompts.get(ev.prompt_id);
  const tr = el("tr");
  const cell = (child, cls) => { const td = el("td", cls); td.append(child); tr.append(td); };
  cell(document.createTextNode(`#${ev.prompt_id}`));
  cell(document.createTextNode(p ? short(p.prompt, 60) + (p.control ? " (control)" : "") : ""), "prompt-cell");
  cell(verdictPill(ev.raw.verdict));
  cell(verdictPill(ev.harness.verdict, ev.harness.fixes));
  const labels = fixesOf(ev.harness.labels || []);
  cell(document.createTextNode(labels.length ? labels.map(pretty).join(", ") : "—"), "labels-cell");
  cell(document.createTextNode(ev.raw.s != null ? ev.raw.s.toFixed(1) : "—"), "num");
  cell(document.createTextNode(ev.harness.s != null ? ev.harness.s.toFixed(1) : "—"), "num");
  $("#sweep-rows").append(tr);
  const wrap = $("#sweep-wrap");
  wrap.scrollTop = wrap.scrollHeight;   // scroll the table, never the page
}

function showSweepSummary(ev, faults) {
  const tr = el("tr");
  const td = el("td");
  td.colSpan = 7;
  const bits = [
    `Raw loop: ${ev.raw_silent}/${ev.total} silent failures`,
    `Harness: ${ev.harness_silent}/${ev.total} silent failures, ${ev.harness_fallback} safe fallbacks`,
    `${ev.fixes} fixes applied`,
  ];
  // Under chaos every prompt gets a fault, so control prompts are expected to show fixes.
  if (!faults.length) bits.push(`${ev.false_triggers} false triggers on control prompts`);
  td.textContent = bits.join(" · ");
  tr.append(td);
  $("#sweep-summary").replaceChildren(tr);
  showBanner(`WHOLE PACK DONE · ${bits.slice(0, 2).join(" · ")}`);
}

// ---------- stream ----------
let sweepFaults = [];

function syncControls(ev) {
  // Make the controls match what's on screen, so R after a replay runs the same thing.
  $("#prompt").value = String(ev.prompt_id);
  for (const c of document.querySelectorAll(".chaos input")) c.checked = ev.faults.includes(c.value);
}

function handle(ev) {
  if (ev.side === "sweep") {
    if (current && current.mode === "replay") return;   // a replayed pack prompt: lanes only
    if (ev.label === "ROW") addSweepRow(ev);
    else if (ev.label === "SUMMARY") showSweepSummary(ev, sweepFaults);
    return;
  }
  if (ev.side === "run") {
    if (ev.label === "SWEEP_START") {
      sweepFaults = ev.faults;
      startSweep(ev);
    } else if (ev.label === "START") {
      current = { mode: ev.mode, raw: null, harness: null, labels: [] };
      startLanes();
      syncControls(ev);
      showPrompt(ev.prompt_id);
      $(".prompt-box").scrollIntoView({ block: "start", behavior: REDUCED_MOTION ? "auto" : "smooth" });
      if (ev.mode === "replay") {
        const fake = ev.original_mode === "fake" ? " · fake model" : "";
        showBanner(`REPLAY · run from ${clock(ev.saved_at)}${fake} · ${chaosText(ev.faults)}`);
      } else {
        const pack = ev.sweep ? `WHOLE PACK ${ev.index}/${ev.total}` : `PROMPT #${ev.prompt_id}`;
        const recorded = ev.mode === "recorded"
          ? `RECORDED RUN${ev.original_mode === "fake" ? " · fake model" : ""} · ` : "";
        showBanner(`${recorded}${pack} · ${chaosText(ev.faults)}`);
      }
    } else if (ev.label === "ERROR") {
      for (const side of Object.keys(LANES)) showError(side, ev.message);
    }
    return;
  }
  if (!LANES[ev.side]) return;
  if (ev.label === "VERDICT") {
    if (current) current[ev.side] = ev.verdict;
    showVerdict(ev.side, ev);
    if (current) tally(current);
    return;
  }
  if (current && ev.side === "harness" && RECOVERY.has(ev.label)) current.labels.push(ev.label);
  appendEvent(ev.side, ev);
  if (ev.label === "ERROR") {
    showError(ev.side, ev.message);
    if (current) { current[ev.side] = "ERROR"; tally(current); }
  }
}

async function stream(url, trigger, { newestReplay = false } = {}) {
  lastTrigger = trigger;
  setBusy(true);
  try {
    const res = await fetch(url);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) handle(JSON.parse(line.slice(6)));
        }
      }
    }
  } catch (err) {
    showBanner(`Couldn't run: ${err.message}`, "error");
  } finally {
    for (const side of Object.keys(LANES)) {
      if (LANES[side].querySelector(".verdict.pending")) setVerdict(side, "error", "NO VERDICT", "The stream ended early.");
    }
    setLiveRegions(true);
    await loadReplays({ selectNewest: newestReplay }).catch(() => {});
    setBusy(false);
  }
}

// ---------- actions ----------
function runSelected() {
  if (busy) return;
  const faults = selectedFaults().join(",");
  stream(`/api/run?prompt_id=${encodeURIComponent($("#prompt").value)}&faults=${encodeURIComponent(faults)}`,
    $("#run"), { newestReplay: true });
}

function runAll() {
  if (busy) return;
  stream(`/api/sweep?faults=${encodeURIComponent(selectedFaults().join(","))}`, $("#run-all"));
}

function replaySelected() {
  if (busy || !$("#replays").value) return;
  stream(`/api/replay?name=${encodeURIComponent($("#replays").value)}`, $("#replay"));
}

function stepPrompt(delta) {
  const sel = $("#prompt");
  const next = sel.selectedIndex + delta;
  if (busy || next < 0 || next >= sel.options.length) return;
  sel.selectedIndex = next;
  showPrompt(Number(sel.value));
}

// ---------- presenter keys ----------
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const t = e.target;
  // Leave typing and native control keys alone.
  if (t.matches("select, textarea, input:not([type=checkbox])")) return;
  const onPage = t === document.body || t === document.documentElement;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (key === "r") runSelected();
  else if (key === "a") runAll();
  else if (["1", "2", "3", "4"].includes(key)) {
    const box = document.querySelectorAll(".chaos input")[Number(key) - 1];
    if (box && !busy) box.checked = !box.checked;
  } else if (key === "ArrowRight" && onPage) stepPrompt(1);
  else if (key === "ArrowLeft" && onPage) stepPrompt(-1);
  else if (key === " " && onPage) {
    e.preventDefault();
    replaySelected();
  }
});

// ---------- wiring ----------
$("#prompt").addEventListener("change", (e) => showPrompt(Number(e.target.value)));
$("#run").addEventListener("click", runSelected);
$("#run-all").addEventListener("click", runAll);
$("#replay").addEventListener("click", replaySelected);
$("#score-reset").addEventListener("click", () => {
  score = emptyScore();
  saveScore();
  renderScore();
});

loadScore();
renderScore();
idleLanes();
Promise.all([loadPrompts(), loadReplays()])
  .then(loadInfo)
  .catch((err) => showBanner(`Couldn't load: ${err.message}`, "error"));
