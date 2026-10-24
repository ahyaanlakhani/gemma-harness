#!/usr/bin/env python
"""Deterministic comparison: the same injected faults, baseline vs harness.

    python demo/compare.py
    python demo/compare.py --trace empty_then_answer
    python demo/compare.py --jsonl-dir demo/traces

No API key, no network. Scripted replies simulate failures; they are injected,
not observed Gemma defects. See demo/scripted.py.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gemma_harness import Client, pretty_print, write_jsonl  # noqa: E402
from demo import scripted  # noqa: E402
from demo.baseline import baseline_run  # noqa: E402
from demo.tools import CACHEABLE_TOOLS, DEMO_TOOLS, reset_notes  # noqa: E402

MODEL = "gemma-4-26b-a4b-it"
MAX_STEPS = 8


def counted(tools, counter):
    """Wrap tools so we can report how many executions each engine caused."""

    def wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            counter[fn.__name__] = counter.get(fn.__name__, 0) + 1
            return fn(*args, **kwargs)

        return wrapper

    return [wrap(fn) for fn in tools]


def run_scenario(scenario):
    reset_notes()
    baseline_counter: dict[str, int] = {}
    baseline = baseline_run(
        scripted.ScriptedAdapter(scenario["turns"]),
        scenario["prompt"],
        counted(DEMO_TOOLS, baseline_counter),
        model=MODEL,
        max_steps=MAX_STEPS,
    )

    reset_notes()
    harness_counter: dict[str, int] = {}
    outcome = Client(
        model=MODEL,
        adapter=scripted.ScriptedAdapter(scenario["turns"]),
        max_steps=MAX_STEPS,
        cacheable_tools=CACHEABLE_TOOLS,
    ).run(scenario["prompt"], tools=counted(DEMO_TOOLS, harness_counter))

    return baseline, outcome, baseline_counter, harness_counter


def baseline_label(result: dict) -> str:
    reason = result.get("failure_reason") or ""
    if reason.startswith("unhandled_exception"):
        return "CRASH " + reason.split(": ")[-1]
    if reason == "blank_output":
        return "BLANK"
    if reason == "max_steps":
        return "NO ANSWER (step cap)"
    if result["text"]:
        return "completed"
    return "failed"


def check(scenario, text: str) -> bool:
    try:
        return bool(scenario["check"](text))
    except Exception:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", help="print the harness trace for one scenario id")
    parser.add_argument("--jsonl-dir", help="write one JSONL trace per harness run")
    args = parser.parse_args(argv)

    if args.trace and args.trace not in {s["id"] for s in scripted.SCENARIOS}:
        parser.error(f"unknown scenario {args.trace!r}")

    print(
        "DETERMINISTIC FAULT INJECTION - scripted model, local tools, no API key\n"
        "Failures are injected by demo/scripted.py. They are not observed Gemma defects.\n"
    )
    header = (
        f"{'scenario':<18} {'injected fault':<44} {'baseline':<24} "
        f"{'harness':<26} {'tools':<7} {'check':<5}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for scenario in scripted.SCENARIOS:
        baseline, outcome, bcount, hcount = run_scenario(scenario)
        ok = check(scenario, outcome["text"])
        b_ok = (
            check(scenario, baseline["text"])
            if baseline["status"] == "completed"
            else False
        )
        ok_mark = "ok" if ok else "MISS"
        harness_label = outcome["status"].upper()
        if outcome["status"] == "failed":
            harness_label = f"FAILED({outcome['failure_reason']})"
        baseline_text = baseline_label(baseline)
        if baseline["status"] == "completed" and not b_ok:
            baseline_text = "completed (WRONG)"
        tools = f"{sum(bcount.values())}/{sum(hcount.values())}"
        print(
            f"{scenario['id']:<18} {scenario['fault']:<44} {baseline_text:<24} "
            f"{harness_label:<26} {tools:<7} {ok_mark:<5}"
        )
        rows.append(
            {
                "id": scenario["id"],
                "kind": scenario["kind"],
                "baseline": baseline,
                "harness": outcome,
                "baseline_ok": b_ok,
                "harness_ok": ok,
                "baseline_tools": sum(bcount.values()),
                "harness_tools": sum(hcount.values()),
            }
        )
        if args.trace == scenario["id"]:
            print("\nharness trace:")
            pretty_print(outcome["trace"], colour=sys.stdout.isatty())
            print(f"\nharness answer: {outcome['text']}\n")
        if args.jsonl_dir:
            write_jsonl(
                outcome["trace"], os.path.join(args.jsonl_dir, f"{scenario['id']}.jsonl")
            )

    _summary(rows)
    expected = sum(1 for row in rows if row["harness_ok"])
    print(f"\nharness met the expected outcome in {expected}/{len(rows)} scenarios")
    return 0 if expected == len(rows) else 1


def _summary(rows: list[dict]) -> None:
    baseline = {"ok": 0, "wrong": 0, "blank": 0, "crash": 0, "other": 0}
    harness = {"ok": 0, "recovered_ok": 0, "failed_ok": 0, "wrong": 0, "failed_other": 0}
    for row in rows:
        b, h = row["baseline"], row["harness"]
        reason = b.get("failure_reason") or ""
        if reason.startswith("unhandled_exception"):
            baseline["crash"] += 1
        elif b["status"] != "completed" or not b["text"].strip():
            baseline["blank"] += 1
        elif row["baseline_ok"]:
            baseline["ok"] += 1
        else:
            baseline["wrong"] += 1

        if h["status"] == "failed":
            if row["kind"] == "explicit_failure":
                harness["failed_ok"] += 1
            else:
                harness["failed_other"] += 1
        elif row["harness_ok"] and h["status"] == "recovered":
            harness["recovered_ok"] += 1
        elif row["harness_ok"]:
            harness["ok"] += 1
        else:
            harness["wrong"] += 1

    tool_calls = (
        sum(r["baseline_tools"] for r in rows),
        sum(r["harness_tools"] for r in rows),
    )
    print("\nSUMMARY (injected faults, n=%d)" % len(rows))
    print("-" * 52)
    print(f"{'outcome':<34}{'baseline':>9}{'harness':>9}")
    print(f"{'expected outcome met':<34}{baseline['ok']:>9}{harness['ok'] + harness['recovered_ok'] + harness['failed_ok']:>9}")
    print(f"{'  of which after recovery':<34}{'-':>9}{harness['recovered_ok']:>9}")
    print(f"{'  of which explicit failure':<34}{'-':>9}{harness['failed_ok']:>9}")
    print(f"{'wrong answer':<34}{baseline['wrong']:>9}{harness['wrong']:>9}")
    print(f"{'blank output':<34}{baseline['blank']:>9}{harness['failed_other']:>9}")
    print(f"{'unhandled exception':<34}{baseline['crash']:>9}{0:>9}")
    print(f"{'tool calls attempted':<34}{tool_calls[0]:>9}{tool_calls[1]:>9}")
    print(
        "\n'tools' is calls attempted per engine. tool_exception is expected to end "
        "in an honest failure report; every other scenario requires a correct, "
        "tool-derived answer."
    )


if __name__ == "__main__":
    raise SystemExit(main())
