"use strict";
// Server-free mode for the hosted demo page: answers the page's /api/* requests from
// recordings baked into window.__DEMO_DATA__ by build_static.py. RUN and RUN ALL play
// the newest recording for the chosen prompt and chaos; nothing reaches a network.
(() => {
  const DATA = window.__DEMO_DATA__;
  const PACE_MIN_S = 0.22;   // recorded runs are fast; slow them to a watchable pace
  const PACE_MAX_S = 1.5;
  const FIX_LABELS = new Set([
    "PHANTOM_TOOL_LINTED", "RETRIED", "SILENT_FAIL_PREVENTED", "TOOL_NAME_FIXED", "UNKNOWN_TOOL",
    "ARGS_COERCED", "ARGS_MISSING", "TOOL_ERROR", "TOOL_TIMEOUT", "DEDUPED", "STEP_CAP", "TRUNCATED",
  ]);
  const prompts = new Map(DATA.prompts.map((p) => [p.id, p]));
  const byName = new Map(DATA.recordings.map((r) => [r.name, r]));
  const sleep = (s) => new Promise((done) => setTimeout(done, s * 1000));

  const json = (obj, status = 200) =>
    new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
  const sameFaults = (a, b) => a.length === b.length && a.every((f) => b.includes(f));
  const parseFaults = (raw) => (raw || "").split(",").filter(Boolean);
  // Recordings are newest first, so find() picks the latest take.
  const newestFor = (pid, faults) => DATA.recordings.find((r) => r.prompt_id === pid && sameFaults(r.faults, faults));

  function sse(events) {
    const enc = new TextEncoder();
    const body = new ReadableStream({
      async start(ctrl) {
        let prev = null;
        for (const ev of events) {
          if (prev !== null) await sleep(Math.min(Math.max(ev.t - prev, PACE_MIN_S), PACE_MAX_S));
          prev = ev.t;
          ctrl.enqueue(enc.encode(`data: ${JSON.stringify(ev)}\n\n`));
        }
        ctrl.close();
      },
    });
    return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
  }

  // A recording's events with its START relabelled for playback; never its END.
  function playback(rec, mode, extra = {}) {
    return rec.events
      .filter((ev) => !(ev.side === "run" && ev.label === "END"))
      .map((ev) => (ev.side === "run" && ev.label === "START"
        ? { ...ev, mode, original_mode: ev.mode, saved_at: ev.t, ...extra }
        : ev));
  }

  function rowFor(rec) {
    const recorded = rec.events.find((ev) => ev.side === "sweep" && ev.label === "ROW");
    if (recorded) return recorded;
    const start = rec.events[0].t;
    const lane = (side) => {
      const ev = rec.events.filter((e) => e.side === side && (e.label === "VERDICT" || e.label === "ERROR")).pop();
      return { verdict: ev && ev.label === "VERDICT" ? ev.verdict : "ERROR", s: ev ? Math.round((ev.t - start) * 10) / 10 : null };
    };
    const labels = rec.events.filter((e) => e.side === "harness" && FIX_LABELS.has(e.label)).map((e) => e.label);
    return { side: "sweep", label: "ROW", prompt_id: rec.prompt_id, t: rec.events[rec.events.length - 1].t,
      control: Boolean((prompts.get(rec.prompt_id) || {}).control), raw: lane("raw"),
      harness: { ...lane("harness"), fixes: labels.length, labels } };
  }

  function summary(rows, t) {
    const count = (side, v) => rows.filter((r) => r[side].verdict === v).length;
    return { side: "sweep", label: "SUMMARY", t, total: rows.length,
      raw_silent: count("raw", "SILENT_FAILURE"), harness_silent: count("harness", "SILENT_FAILURE"),
      harness_fallback: count("harness", "FALLBACK"), harness_answered: count("harness", "ANSWERED"),
      fixes: rows.reduce((n, r) => n + (r.harness.fixes || 0), 0),
      false_triggers: rows.filter((r) => r.control && r.harness.labels.some((l) => FIX_LABELS.has(l))).length };
  }

  function sweep(faults) {
    const recs = DATA.prompt_ids.map((pid) => newestFor(pid, faults)).filter(Boolean);
    if (!recs.length) return null;
    // Re-time the recordings back to back so playback pacing stays even.
    let clock = recs[0].events[0].t;
    const events = [{ side: "run", label: "SWEEP_START", t: clock, ids: recs.map((r) => r.prompt_id), faults,
      mode: "recorded", original_mode: DATA.source, model: DATA.model }];
    const rows = [];
    recs.forEach((rec, i) => {
      const shift = clock - rec.events[0].t;
      const evs = playback(rec, "recorded", { sweep: true, index: i + 1, total: recs.length })
        .filter((ev) => ev.label !== "ROW")
        .map((ev) => ({ ...ev, t: ev.t + shift }));
      const row = { ...rowFor(rec), t: evs[evs.length - 1].t };
      events.push(...evs, row);
      rows.push(row);
      clock = row.t + 0.5;
    });
    events.push(summary(rows, clock), { side: "run", label: "END", t: clock });
    return events;
  }

  const routes = {
    "/api/info": () => json({ mode: "recorded", source: DATA.source, model: DATA.model }),
    "/api/prompts": () => json(DATA.prompts),
    "/api/replays": () => json(DATA.recordings.map(({ events, ...meta }) => meta)),
    "/api/replay": (q) => {
      const rec = byName.get(q.get("name"));
      return rec ? sse(playback(rec, "replay")) : json({ error: "replay not found" }, 404);
    },
    "/api/run": (q) => {
      const pid = Number(q.get("prompt_id"));
      const faults = parseFaults(q.get("faults"));
      const rec = newestFor(pid, faults);
      if (!rec) {
        return json({ error: `No recorded run of prompt #${pid} with ${faults.length ? "this chaos" : "no chaos"}. `
          + "This hosted page plays recorded runs; run demo/web/server.py for live races." }, 404);
      }
      return sse([...playback(rec, "recorded"), { side: "run", label: "END", t: rec.events[rec.events.length - 1].t }]);
    },
    "/api/sweep": (q) => {
      const events = sweep(parseFaults(q.get("faults")));
      return events ? sse(events) : json({ error: "No recorded runs with this chaos setting." }, 404);
    },
  };

  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    const at = url.pathname.indexOf("/api/");
    const route = at >= 0 ? routes[url.pathname.slice(at)] : null;
    return route ? route(url.searchParams) : realFetch(input, init);
  };
})();
