"""Trace formatting and JSONL export."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

__all__ = ["write_jsonl", "format_event", "pretty_print", "label_counts"]

RESET = "\033[0m"
DIM = "\033[2m"
_GREEN = {"TOOL_OK", "FINAL", "CACHED_TOOL_RESULT", "ARGS_COERCED"}
_RED = {"FAILED", "TOOL_ERROR", "INVALID_ARGS", "UNKNOWN_TOOL", "API_ERROR",
        "BUDGET_EXCEEDED", "REFUSAL"}
_YELLOW = {"EMPTY_RESPONSE", "UNUSABLE_RESPONSE", "REPAIR_REQUESTED", "API_RETRY"}


def write_jsonl(events: Iterable[dict], path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
    return path


def label_counts(events: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        label = str(event.get("label", "?"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _colour(label: str) -> str:
    if label in _RED:
        return "\033[31m"
    if label in _GREEN:
        return "\033[32m"
    if label in _YELLOW:
        return "\033[33m"
    return DIM


def format_event(event: dict[str, Any], colour: bool = False) -> str:
    label = str(event.get("label", "?"))
    if label == "MODEL_CALL":
        text = f"[step {event.get('step')}] MODEL_CALL  {event.get('ms')}ms"
    else:
        detail = ", ".join(
            f"{key}={value}"
            for key, value in event.items()
            if key not in ("t", "step", "label") and value is not None
        )
        text = f"[step {event.get('step')}] {label}"
        if detail:
            text += f"  {detail}"
    if not colour:
        return text
    return f"{_colour(label)}{text}{RESET}"


def pretty_print(events: Iterable[dict], stream=None, colour: bool = True) -> None:
    import sys

    stream = stream or sys.stdout
    for event in events:
        print(format_event(event, colour=colour), file=stream, flush=True)
