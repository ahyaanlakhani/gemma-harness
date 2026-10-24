"""Run the 15-prompt pack through the raw SDK loop and/or gemma-harness.

    python demo/stress.py --raw
    python demo/stress.py --harness
    python demo/stress.py --both --runs 3 --delay 4

Writes every harness event to runs/trace-<time>.jsonl. Needs GEMINI_API_KEY.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import httpx
import yaml
from google import genai
from google.genai import errors, types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from gemma_harness import Client  # noqa: E402
from gemma_harness.trace import pretty, recoveries, write_jsonl  # noqa: E402
from raw_agent import raw_run  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

NOT_FALSE_TRIGGERS = {"BACKOFF_429", "BACKOFF_5XX"}   # quota noise, not the harness misfiring
API_ERRORS = (errors.APIError, httpx.TransportError)   # the API failed, not the loop
API_ERROR = "API_ERROR"
HTTP_TIMEOUT_MS = 45_000   # per request; the SDK default is to wait forever


def load_prompts(only=None):
    pack = yaml.safe_load((HERE / "prompts.yaml").read_text(encoding="utf-8"))
    return [p for p in pack if not only or p["id"] in only]


def run_raw(p, model, client):
    stats, t0 = {}, time.time()
    try:
        text = raw_run(p["prompt"], ALL_TOOLS, model=model, stats=stats, client=client)
        outcome = "ok" if text.strip() else "SILENT (empty)"
    except API_ERRORS:
        outcome = API_ERROR
    except Exception as e:
        outcome = f"SILENT ({type(e).__name__})"
    return {"outcome": outcome, "silent": outcome.startswith("SILENT"), "s": time.time() - t0,
            "think_tok": stats.get("think_tok", 0), "total_tok": stats.get("total_tok", 0),
            "labels": []}


def run_harness(p, model, client, live, trace_path):
    def on_event(ev):
        if live:
            print("   " + pretty(ev))
    h = Client(model=model, on_event=on_event, raw=client)
    t0 = time.time()
    try:
        out = h.run(p["prompt"], tools=ALL_TOOLS)
        outcome = "ok" if out["text"].strip() else "SILENT (empty)"
    except API_ERRORS:
        outcome = API_ERROR
    except Exception as e:   # the harness should never get here; count it if it does
        outcome = f"SILENT ({type(e).__name__})"
    write_jsonl([{**ev, "prompt_id": p["id"]} for ev in h.trace], trace_path)
    calls = [ev for ev in h.trace if ev["label"] == "CALL"]
    labels = recoveries(h.trace)
    if outcome != API_ERROR and ("SILENT_FAIL_PREVENTED" in labels or "STEP_CAP" in labels):
        outcome = "fallback (labelled)"
    return {"outcome": outcome, "silent": outcome.startswith("SILENT"), "s": time.time() - t0,
            "think_tok": sum(ev.get("think_tok") or 0 for ev in calls),
            "total_tok": sum((ev.get("in_tok") or 0) + (ev.get("out_tok") or 0)
                             + (ev.get("think_tok") or 0) for ev in calls),
            "labels": labels}


def majority(flags) -> bool:
    """Strictly more than half: 1 of 2 runs is a tie, not a majority."""
    flags = list(flags)
    return sum(map(bool, flags)) * 2 > len(flags)


def stepped_in(r) -> bool:
    """The harness recovered something; waiting out a quota doesn't count."""
    return bool(set(r["labels"]) - NOT_FALSE_TRIGGERS)


def summarise(rows_by_prompt, prompts, mode):
    """Per prompt, a failure or recovery counts if it happened in a majority of the runs
    that reached the end. API errors are not the loop's fault: they get their own row."""
    done = {pid: [r for r in rows if r["outcome"] != API_ERROR]
            for pid, rows in rows_by_prompt.items()}
    silent = sum(majority(r["silent"] for r in rows) for rows in done.values())
    recovered = sum(majority(stepped_in(r) for r in rows) for rows in done.values())
    controls = {p["id"] for p in prompts if p.get("control")}
    false_trig = sum(majority(stepped_in(r) for r in rows)
                     for pid, rows in done.items() if pid in controls)
    all_rows = [r for rows in rows_by_prompt.values() for r in rows]
    api_errors = sum(r["outcome"] == API_ERROR for r in all_rows)
    runs = max(len(rows) for rows in rows_by_prompt.values())
    return {
        "Silent failures": f"{silent}/{len(rows_by_prompt)}",
        "API errors": f"{api_errors}/{len(all_rows)}",
        "Recovered turns": f"{recovered}/{len(rows_by_prompt)}" if mode == "harness" else "n/a",
        "Median latency per prompt (s)": f"{statistics.median(r['s'] for r in all_rows):.1f}",
        "Thinking tokens (per pass)": str(sum(r["think_tok"] for r in all_rows) // runs),
        "Total tokens (per pass)": str(sum(r["total_tok"] for r in all_rows) // runs),
        "False triggers on controls": str(false_trig) if mode == "harness" else "n/a",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--raw", action="store_true")
    g.add_argument("--harness", action="store_true")
    g.add_argument("--both", action="store_true")
    ap.add_argument("--model", default="gemma-4-26b-a4b-it")
    ap.add_argument("--runs", type=int, default=1, help="PRD asks for 3; start with 1 to save quota")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between prompts (free-tier RPM)")
    ap.add_argument("--only", type=int, nargs="*", help="prompt ids to run")
    ap.add_argument("--quiet", action="store_true", help="hide the live harness trace")
    args = ap.parse_args(argv)
    modes = ["raw"] if args.raw else ["harness"] if args.harness else ["raw", "harness"]
    prompts = load_prompts(args.only)
    if not prompts:
        ap.error(f"no prompts match --only {' '.join(map(str, args.only))}")

    client = genai.Client(http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS))
    trace_path = HERE.parent / "runs" / f"trace-{int(time.time())}.jsonl"
    results = {m: {p["id"]: [] for p in prompts} for m in modes}

    for p in prompts:
        print(f"\n#{p['id']:>2} {p['prompt']}")
        for mode in modes:
            for i in range(args.runs):
                r = (run_raw(p, args.model, client) if mode == "raw"
                     else run_harness(p, args.model, client, not args.quiet, trace_path))
                results[mode][p["id"]].append(r)
                tag = f" run {i + 1}" if args.runs > 1 else ""
                print(f"  {mode:<7}{tag}: {r['outcome']:<22} {r['s']:5.1f}s  {' '.join(r['labels'])}")
                if args.delay:
                    time.sleep(args.delay)

    print(f"\nModel {args.model}, {len(prompts)} prompts x {args.runs} run(s). Trace: {trace_path}")
    sums = {m: summarise(results[m], prompts, m) for m in modes}
    keys = list(next(iter(sums.values())))
    print(f"\n{'Metric':<32}" + "".join(f"{m:>16}" for m in modes))
    for k in keys:
        print(f"{k:<32}" + "".join(f"{sums[m][k]:>16}" for m in modes))


if __name__ == "__main__":
    main()
