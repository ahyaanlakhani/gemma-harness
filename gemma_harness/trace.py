"""Write a run's trace as JSONL and pretty-print events for the terminal."""
import json
from pathlib import Path

# Labels that mean the harness stepped in; shown highlighted in the live trace.
RECOVERY_LABELS = {"PHANTOM_TOOL_LINTED", "RETRIED", "SILENT_FAIL_PREVENTED",
                   "TOOL_NAME_FIXED", "UNKNOWN_TOOL", "ARGS_COERCED", "ARGS_MISSING",
                   "TOOL_ERROR", "TOOL_TIMEOUT", "DEDUPED", "STEP_CAP",
                   "BACKOFF_429", "BACKOFF_5XX", "TRUNCATED"}


def write_jsonl(trace: list[dict], path, append=True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a" if append else "w", encoding="utf-8") as f:
        for ev in trace:
            f.write(json.dumps(ev, default=str) + "\n")


def pretty(ev: dict) -> str:
    extra = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("t", "step", "label"))
    mark = "*" if ev["label"] in RECOVERY_LABELS else " "
    return f"{mark} step {ev['step']:>2}  {ev['label']:<22} {extra}"


def recoveries(trace: list[dict]) -> list[str]:
    return [ev["label"] for ev in trace if ev["label"] in RECOVERY_LABELS]
