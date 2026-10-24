"""Live demo: race the raw Gemma loop against gemma-harness in a browser.

    python demo/web/server.py           # needs GEMINI_API_KEY
    python demo/web/server.py --fake    # scripted fake model, no key needed

Then open http://127.0.0.1:8765. Every run that finishes without an ERROR is saved to
runs/replays/ so it can be replayed on stage if the API is down or out of quota.
"""
import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB = Path(__file__).resolve().parent
DEMO = WEB.parent
ROOT = DEMO.parent
for _p in (WEB, DEMO, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import httpx  # noqa: E402
import yaml  # noqa: E402
from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

from fake_model import FakeGenai  # noqa: E402
from gemma_harness import Client  # noqa: E402
from gemma_harness.chaos import FAULTS, ChaosClient, chaos_tools  # noqa: E402
from gemma_harness.trace import RECOVERY_LABELS, recoveries  # noqa: E402
from raw_agent import raw_run  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

STATIC = WEB / "static"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8"}
REPLAY_NAME = re.compile(r"^\d+-p\d+\.jsonl$")
REPLAY_GAP_CAP_S = 2.0
SWEEP_MAX_DELAY_S = 30.0
BACKOFF_LABELS = {"BACKOFF_429", "BACKOFF_5XX"}   # waiting out the API is not a fix
HTTP_TIMEOUT_MS = 45_000   # per API request; the SDK default is to wait forever
RUN_TIMEOUT_S = 150        # per prompt, both lanes; DemoServer.run_timeout_s overrides it
TIMEOUT_MESSAGE = "Timed out waiting for the model. Use REPLAY for a saved run."
API_ERRORS = (errors.APIError, httpx.TransportError)   # not the loop's fault: no verdict
GUARDED_PATHS = {"/api/run", "/api/sweep", "/api/replay"}
PROMPTS = {p["id"]: p for p in yaml.safe_load((DEMO / "prompts.yaml").read_text(encoding="utf-8"))}


# ---------- the race ----------
def api_error_message(e: Exception) -> str:
    """One line for a lane ERROR. The page adds the "use Replay" hint."""
    if isinstance(e, errors.APIError):
        message = str(getattr(e, "message", None) or e).strip().rstrip(".")
        return f"API error {getattr(e, 'code', None)}: {message}"
    return f"Network error ({type(e).__name__}): {e}"


def fix_labels(labels: list[str]) -> list[str]:
    """Recovery labels that count as fixes: quota waits are shown, not counted."""
    return [label for label in labels if label not in BACKOFF_LABELS]


def has_error(events: list[dict]) -> bool:
    return any(ev["label"] == "ERROR" for ev in events)


def raw_side(prompt, faults, client, model, emit):
    def lane(ev):
        emit({"side": "raw", **ev})

    def on_fire(fault):
        lane({"label": "CHAOS", "fault": fault})

    chaos = ChaosClient(client, faults, on_fire)
    try:
        text = raw_run(prompt, chaos_tools(ALL_TOOLS, faults, on_fire), model=model,
                       client=chaos, on_event=lane)
    except API_ERRORS as e:       # not the loop's fault; no verdict
        lane({"label": "ERROR", "message": api_error_message(e)})
        return
    except Exception as e:        # the baseline has no recovery: this is the failure we show
        lane({"label": "CRASH", "error": f"{type(e).__name__}: {e}"})
        text = ""
    lane({"label": "VERDICT", "verdict": "ANSWERED" if text.strip() else "SILENT_FAILURE",
          "fixes": 0, "text": text})


def harness_side(prompt, faults, client, model, emit):
    def lane(ev):
        emit({"side": "harness", **ev})

    def on_fire(fault):
        lane({"label": "CHAOS", "fault": fault})

    h = Client(model=model, raw=ChaosClient(client, faults, on_fire), on_event=lane)
    try:
        out = h.run(prompt, tools=chaos_tools(ALL_TOOLS, faults, on_fire))
    except API_ERRORS as e:
        lane({"label": "ERROR", "message": api_error_message(e)})
        return
    except Exception as e:        # a harness bug: show it, never hide it
        lane({"label": "CRASH", "error": f"{type(e).__name__}: {e}"})
        out = {"text": ""}
    labels = recoveries(h.trace)
    if {"SILENT_FAIL_PREVENTED", "STEP_CAP"} & set(labels):
        verdict = "FALLBACK"
    else:
        verdict = "ANSWERED" if out["text"].strip() else "SILENT_FAILURE"
    lane({"label": "VERDICT", "verdict": verdict, "fixes": len(fix_labels(labels)),
          "text": out["text"]})


def run_race(prompt, faults, client, model, emit):
    """Both sides at once, each with its own chaos wrappers. Returns when both finish."""
    sides = [threading.Thread(target=fn, args=(prompt, faults, client, model, emit), daemon=True)
             for fn in (raw_side, harness_side)]
    for t in sides:
        t.start()
    for t in sides:
        t.join()


def stamp(ev: dict) -> dict:
    return {**ev, "t": round(time.time(), 3)}


def run_prompt(srv: "DemoServer", pid: int, faults: list[str], **start_extra):
    """Run one prompt's race and yield its events as they happen, each stamped with `t`:
    START (plus `start_extra`), then either both lanes' events or a run-level ERROR if the
    client can't be created or the race outlives `srv.run_timeout_s`. Never yields END; the
    caller closes the stream."""
    q: queue.Queue = queue.Queue()
    done = object()

    def emit(ev):
        q.put(stamp(ev))

    def race():
        try:
            client = (FakeGenai(srv.fake_delay_s) if srv.fake else
                      genai.Client(http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS)))
        except Exception as e:
            emit({"side": "run", "label": "ERROR", "message":
                  f"Can't create the Gemini client ({e}). Set GEMINI_API_KEY, "
                  "start with --fake, or use Replay."})
        else:
            run_race(PROMPTS[pid]["prompt"], faults, client, srv.model, emit)
        finally:
            q.put(done)

    emit({"side": "run", "label": "START", "prompt_id": pid, "faults": faults,
          "mode": srv.mode, "model": srv.model, **start_extra})
    threading.Thread(target=race, daemon=True).start()
    deadline = time.monotonic() + srv.run_timeout_s
    while True:
        try:
            ev = q.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:   # the lane threads are abandoned; nobody reads what they emit
            yield stamp({"side": "run", "label": "ERROR", "message": TIMEOUT_MESSAGE})
            return
        if ev is done:
            return
        yield ev


def is_run_error(ev: dict) -> bool:
    return ev["side"] == "run" and ev["label"] == "ERROR"


# ---------- sweep ----------
def sweep_row(pid: int, events: list[dict]) -> dict:
    """One table row from a prompt's events (its START first)."""
    start_t = events[0]["t"]
    final = {ev["side"]: ev for ev in events
             if ev["side"] in ("raw", "harness") and ev["label"] in ("VERDICT", "ERROR")}

    def lane(side):
        ev = final.get(side)
        return {"verdict": ev["verdict"] if ev and ev["label"] == "VERDICT" else "ERROR",
                "s": round(ev["t"] - start_t, 1) if ev else None}

    labels = fix_labels([ev["label"] for ev in events
                         if ev["side"] == "harness" and ev["label"] in RECOVERY_LABELS])
    harness = lane("harness")
    harness_final = final.get("harness") or {}
    harness.update(fixes=harness_final.get("fixes", len(labels)), labels=labels)
    return {"side": "sweep", "label": "ROW", "prompt_id": pid,
            "control": bool(PROMPTS[pid].get("control")), "raw": lane("raw"), "harness": harness}


def sweep_summary(rows: list[dict]) -> dict:
    def count(side, verdict):
        return sum(r[side]["verdict"] == verdict for r in rows)

    return {"side": "sweep", "label": "SUMMARY", "total": len(rows),
            "raw_silent": count("raw", "SILENT_FAILURE"),
            "harness_silent": count("harness", "SILENT_FAILURE"),
            "harness_fallback": count("harness", "FALLBACK"),
            "harness_answered": count("harness", "ANSWERED"),
            "fixes": sum(r["harness"]["fixes"] for r in rows),
            "false_triggers": sum(r["control"] and bool(r["harness"]["labels"]) for r in rows)}


# ---------- request parsing (raise ValueError -> 400) ----------
def parse_faults(raw: str) -> list[str]:
    faults = [f for f in raw.split(",") if f]
    unknown = set(faults) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown faults {sorted(unknown)}")
    return sorted(set(faults), key=FAULTS.index)


def parse_ids(raw: str | None) -> list[int]:
    if raw is None:
        return list(PROMPTS)
    ids = []
    for part in (p.strip() for p in raw.split(",")):
        if not part:
            continue
        try:
            pid = int(part)
        except ValueError:
            raise ValueError(f"prompt ids must be numbers, got {part!r}") from None
        if pid not in PROMPTS:
            raise ValueError(f"unknown prompt {pid}")
        if pid not in ids:
            ids.append(pid)
    if not ids:
        raise ValueError("no prompt ids given")
    return ids


def parse_delay(raw: str | None) -> float:
    try:
        delay = float(raw) if raw is not None else 0.0
    except ValueError:
        raise ValueError("delay must be a number of seconds") from None
    if not 0 <= delay <= SWEEP_MAX_DELAY_S:   # also rejects nan
        raise ValueError(f"delay must be between 0 and {SWEEP_MAX_DELAY_S:g} seconds")
    return delay


# ---------- replays ----------
def save_replay(replay_dir: Path, prompt_id: int, events: list[dict]):
    replay_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(ev, default=str) + "\n" for ev in events)
    ms = int(events[0]["t"] * 1000)
    while True:   # two saves in the same millisecond must not overwrite each other
        try:
            with (replay_dir / f"{ms}-p{prompt_id}.jsonl").open("x", encoding="utf-8") as f:
                f.write(body)
            return
        except FileExistsError:
            ms += 1


def replay_outcome(events: list[dict]) -> dict:
    """Each lane's verdict ("ERROR" for a lane error, None if neither) and the harness's fixes."""
    out = {"raw": None, "harness": None, "fixes": None}
    for ev in events:
        side = ev.get("side")
        if side not in ("raw", "harness"):
            continue
        if ev.get("label") == "VERDICT":
            out[side] = ev.get("verdict")
            if side == "harness":
                out["fixes"] = ev.get("fixes")
        elif ev.get("label") == "ERROR" and out[side] is None:
            out[side] = "ERROR"
    return out


def list_replays(replay_dir: Path) -> list[dict]:
    if not replay_dir.is_dir():
        return []
    files = sorted((p for p in replay_dir.iterdir() if REPLAY_NAME.match(p.name)),
                   key=lambda p: int(p.name.split("-")[0]), reverse=True)
    out = []
    for p in files:
        try:
            events = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
            start = events[0]
            out.append({"name": p.name, "prompt_id": start["prompt_id"], "faults": start["faults"],
                        "mode": start["mode"], "saved_at": start["t"], **replay_outcome(events)})
        except (OSError, ValueError, IndexError, KeyError, TypeError, AttributeError):
            continue
    return out


# ---------- HTTP ----------
class Handler(BaseHTTPRequestHandler):
    server: "DemoServer"

    def do_GET(self):
        url = urlparse(self.path)
        qs = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path.startswith("/static/"):
            return self._static(url.path[len("/static/"):])
        route = {"/": self._index, "/api/prompts": self._prompts, "/api/replays": self._replays,
                 "/api/info": self._info, "/api/run": self._run, "/api/sweep": self._sweep,
                 "/api/replay": self._replay}.get(url.path)
        if route is None:
            return self._json({"error": "not found"}, 404)
        if url.path in GUARDED_PATHS and not self._from_this_page():
            return self._json({"error": "forbidden: open the demo at 127.0.0.1 or localhost"}, 403)
        route(qs)

    def _from_this_page(self) -> bool:
        """Runs spend quota: refuse other sites (cross-site requests, DNS rebinding)."""
        port = self.server.server_address[1]
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return False
        return (self.headers.get("Sec-Fetch-Site") or "").strip().lower() != "cross-site"

    def do_HEAD(self):   # health checks only; never starts a run
        self.send_response(200 if urlparse(self.path).path == "/" else 404)
        self.end_headers()

    # ----- plain responses -----
    def _json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES[path.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _index(self, qs):
        self._file(STATIC / "index.html")

    def _static(self, name):
        try:   # resolve, then insist it is a file directly in STATIC with exactly that name
            path = (STATIC / name).resolve()   # (Windows: "C:app.js" carries a drive)
            inside = path.parent == STATIC.resolve() and path.name == name
        except (OSError, ValueError):
            inside = False
        if (not inside or "/" in name or "\\" in name or name.startswith(".")
                or path.suffix not in CONTENT_TYPES or not path.is_file()):
            return self._json({"error": "not found"}, 404)
        self._file(path)

    def _prompts(self, qs):
        self._json(list(PROMPTS.values()))

    def _info(self, qs):
        self._json({"mode": self.server.mode, "model": self.server.model})

    def _replays(self, qs):
        self._json(list_replays(self.server.replay_dir))

    # ----- streams -----
    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _send(self, ev) -> bool:
        """Write one event; False once the browser has gone away."""
        try:
            self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
            self.wfile.flush()
            return True
        except OSError:
            return False

    def _run(self, qs):
        try:
            pid = int(qs.get("prompt_id", ""))
        except ValueError:
            return self._json({"error": "prompt_id must be a number"}, 400)
        if pid not in PROMPTS:
            return self._json({"error": f"unknown prompt {pid}"}, 400)
        try:
            faults = parse_faults(qs.get("faults", ""))
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        self._locked(self._stream_live, pid, faults)

    def _sweep(self, qs):
        try:
            faults = parse_faults(qs.get("faults", ""))
            ids = parse_ids(qs.get("ids"))
            delay = parse_delay(qs.get("delay"))
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        self._locked(self._stream_sweep, ids, faults, delay)

    def _locked(self, stream, *args):
        """One run or sweep at a time; the lock is always released, even if the browser left."""
        if not self.server.run_lock.acquire(blocking=False):
            return self._json({"error": "a run is already in progress"}, 409)
        try:
            try:
                self._sse_headers()
            except OSError:   # browser already gone; run anyway so the result is saved
                pass
            stream(*args)
        finally:
            self.server.run_lock.release()

    def _sender(self):
        """send(ev): write while the browser is there, then quietly drop, so the run keeps
        going to the end and still gets saved."""
        connected = True

        def send(ev):
            nonlocal connected
            connected = connected and self._send(ev)
        return send

    def _stream_live(self, pid, faults):
        send, events = self._sender(), []
        for ev in run_prompt(self.server, pid, faults):
            events.append(ev)
            send(ev)
        events.append(stamp({"side": "run", "label": "END"}))
        send(events[-1])
        if not has_error(events):   # a failed run is no use on stage
            save_replay(self.server.replay_dir, pid, events)

    def _stream_sweep(self, ids, faults, delay):
        s, send, rows = self.server, self._sender(), []
        send(stamp({"side": "run", "label": "SWEEP_START", "ids": ids, "faults": faults,
                    "mode": s.mode, "model": s.model}))
        for index, pid in enumerate(ids, 1):
            if index > 1 and delay:
                time.sleep(delay)
            events = []
            for ev in run_prompt(s, pid, faults, sweep=True, index=index, total=len(ids)):
                events.append(ev)
                send(ev)
            if any(is_run_error(ev) for ev in events):   # no client: the rest would fail too
                break
            row = stamp(sweep_row(pid, events))
            events.append(row)
            send(row)
            rows.append(row)
            if not has_error(events):
                save_replay(s.replay_dir, pid, events)
        else:
            send(stamp(sweep_summary(rows)))
        send(stamp({"side": "run", "label": "END"}))

    def _replay(self, qs):
        name = qs.get("name", "")
        path = self.server.replay_dir / name
        if not REPLAY_NAME.match(name) or not path.is_file():
            return self._json({"error": "replay not found"}, 404)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        self._sse_headers()
        prev = events[0]["t"]
        for ev in events:
            time.sleep(min(max(ev["t"] - prev, 0), REPLAY_GAP_CAP_S))
            prev = ev["t"]
            if ev["side"] == "run" and ev["label"] == "START":
                ev = {**ev, "mode": "replay", "original_mode": ev.get("mode"), "saved_at": ev["t"]}
            if not self._send(ev):
                return


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, fake, model, replay_dir, fake_delay_s):
        super().__init__(addr, Handler)
        self.fake, self.model = fake, model
        self.replay_dir, self.fake_delay_s = replay_dir, fake_delay_s
        self.run_timeout_s = RUN_TIMEOUT_S
        self.run_lock = threading.Lock()

    @property
    def mode(self) -> str:
        return "fake" if self.fake else "live"


def make_server(host="127.0.0.1", port=8765, fake=False, model="gemma-4-26b-a4b-it",
                replay_dir=None, fake_delay_s=0.6) -> DemoServer:
    replay_dir = Path(replay_dir) if replay_dir else ROOT / "runs" / "replays"
    return DemoServer((host, port), fake, model, replay_dir, fake_delay_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="gemma-4-26b-a4b-it")
    ap.add_argument("--fake", action="store_true", help="scripted fake model, no API key")
    args = ap.parse_args(argv)
    srv = make_server(port=args.port, fake=args.fake, model=args.model)
    if not args.fake and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("warning: GEMINI_API_KEY is not set; live runs will fail. Use --fake or Replay.")
    print(f"gemma-harness demo on http://127.0.0.1:{srv.server_address[1]}"
          f" ({'FAKE MODEL' if args.fake else args.model})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
