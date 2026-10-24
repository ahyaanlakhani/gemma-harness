#!/usr/bin/env python
"""Live Gemma run: baseline and harness on the same prompts and tools.

    export GEMINI_API_KEY=...
    python demo/live.py
    python demo/live.py --model gemma-4-31b-it

This is a demonstration, not a benchmark. It records what the model actually
did on this run; no live numbers are baked into the README. The deterministic
fault-injection demo is ``demo/compare.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gemma_harness import Client, GenaiAdapter  # noqa: E402
from demo.baseline import baseline_run  # noqa: E402
from demo.tools import CACHEABLE_TOOLS, DEMO_TOOLS, reset_notes  # noqa: E402

DEFAULT_MODEL = "gemma-4-26b-a4b-it"
MAX_STEPS = 8

PROMPTS = [
    "What's the weather in Karachi? Use get_weather.",
    "Add 2 and 3 using the add tool.",
    "Save a note that says 'ship it', then tell me its note id.",
    "Look up the population of Pakistan with fail_lookup.",
    "What is 2 + 2? No tools are needed.",
]


def preview(text: str, limit: int = 58) -> str:
    text = " ".join((text or "").split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("GEMMA_MODEL", DEFAULT_MODEL))
    args = parser.parse_args(argv)

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print(
            "GEMINI_API_KEY is not set.\n"
            "  live:    set the key, then rerun this script\n"
            "  offline: python demo/compare.py   # deterministic injected faults",
            file=sys.stderr,
        )
        return 2

    adapter = GenaiAdapter()
    print(
        f"LIVE RUN - model={args.model} - no injected faults\n"
        "Outcomes below are what this run actually produced. Not a benchmark.\n"
    )
    header = f"{'#':>2}  {'prompt':<58} {'baseline':<22} {'harness':<26} {'tok(b/h)':<12}"
    print(header)
    print("-" * len(header))

    for index, prompt in enumerate(PROMPTS, start=1):
        reset_notes()
        baseline = baseline_run(
            adapter, prompt, DEMO_TOOLS, model=args.model, max_steps=MAX_STEPS
        )
        reset_notes()
        outcome = Client(
            model=args.model,
            adapter=adapter,
            max_steps=MAX_STEPS,
            cacheable_tools=CACHEABLE_TOOLS,
        ).run(prompt, tools=DEMO_TOOLS)

        b_tok = "-" if not baseline.get("usage") else str(
            (baseline["usage"]["input_tokens"] or 0) + (baseline["usage"]["output_tokens"] or 0)
        )
        h_tok = (
            "-"
            if outcome["usage"]["input_tokens"] is None
            else str(
                (outcome["usage"]["input_tokens"] or 0)
                + (outcome["usage"]["output_tokens"] or 0)
            )
        )
        b_label = baseline["status"]
        if baseline["status"] == "completed" and baseline["failure_reason"] is None:
            b_label = f"completed ({baseline['elapsed_ms']}ms)"
        elif baseline["failure_reason"]:
            b_label = f"{baseline['status']}:{baseline['failure_reason']}"
        h_label = outcome["status"]
        if outcome["status"] == "failed":
            h_label = f"FAILED({outcome['failure_reason']})"
        print(
            f"{index:>2}  {preview(prompt):<58} {b_label:<22} {h_label:<26} "
            f"{b_tok}/{h_tok:<9}"
        )
        print(f"    baseline text: {preview(baseline['text'], 96) or '(empty)'}")
        print(f"    harness text : {preview(outcome['text'], 96)}")
        print(
            f"    harness usage: model_calls={outcome.usage['model_calls']} "
            f"elapsed_ms={outcome.usage['elapsed_ms']} "
            f"thinking_tokens={outcome.usage['thinking_tokens']}"
        )

    print(
        "\nReading this table: 'completed' means the model produced text, not that the\n"
        "answer is correct. 'failed' rows are explicit failures with a reason."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
