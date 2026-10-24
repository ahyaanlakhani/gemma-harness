"""The baseline: a competent minimal tool loop, no recovery.

Same model, same adapter, same tools, same prompts and budgets as the harness.
It does what the SDK docs show: call, dispatch, append, repeat. It has no
validation, no repair, no caching and no outcome model -- which is exactly the
gap the harness covers. It is not deliberately broken; it is just minimal.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from google.genai import types

from gemma_harness.adapters import SchemaClient

__all__ = ["baseline_run"]


def _calls(response: Any) -> list:
    try:
        return list(response.function_calls or [])
    except Exception:
        return []


def baseline_run(
    adapter: Any,
    prompt: str,
    tools: Iterable[Callable] = (),
    model: str = "gemma-4-26b-a4b-it",
    max_steps: int = 8,
) -> dict:
    """Return ``{status, text, failure_reason, steps, elapsed_ms, usage}``.

    Statuses: ``completed`` (non-empty text), ``failed`` with reason
    ``blank_output`` / ``unhandled_exception: <Type>`` / ``max_steps``.
    """
    started = time.monotonic()
    tool_map = {fn.__name__: fn for fn in tools}
    result = {
        "status": "failed",
        "text": "",
        "failure_reason": None,
        "steps": 0,
        "elapsed_ms": 0,
        "usage": None,
    }
    tokens = {"input": 0, "output": 0, "thinking": 0}
    seen_usage = False

    decls = [
        types.FunctionDeclaration.from_callable(client=SchemaClient(), callable=fn)
        for fn in tool_map.values()
    ]
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=decls)] if decls else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents: list[Any] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    try:
        for step in range(1, max_steps + 1):
            result["steps"] = step
            response = adapter.generate(model=model, contents=contents, config=config)
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                seen_usage = True
                tokens["input"] += getattr(usage, "prompt_token_count", 0) or 0
                tokens["output"] += getattr(usage, "candidates_token_count", 0) or 0
                tokens["thinking"] += getattr(usage, "thoughts_token_count", 0) or 0
            calls = _calls(response)
            if not calls:
                try:
                    text = (response.text or "").strip()
                except Exception:
                    text = ""
                result["text"] = text
                if text:
                    result["status"] = "completed"
                else:
                    result["failure_reason"] = "blank_output"
                break
            contents.append(response.candidates[0].content)
            parts = []
            for call in calls:
                try:
                    value = tool_map[call.name](**(call.args or {}))
                except Exception as exc:  # baseline exposes the raw failure
                    result["failure_reason"] = f"unhandled_exception: {type(exc).__name__}"
                    result["text"] = ""
                    result["steps"] = step
                    return _finish(result, started, tokens, seen_usage)
                parts.append(
                    types.Part.from_function_response(
                        name=call.name, response={"result": value}
                    )
                )
            contents.append(types.Content(role="user", parts=parts))
    except Exception as exc:  # provider/transport error
        result["failure_reason"] = f"unhandled_exception: {type(exc).__name__}"
        result["text"] = ""
    else:
        if result["status"] != "completed" and not result["failure_reason"]:
            result["failure_reason"] = "max_steps"
            result["text"] = ""

    return _finish(result, started, tokens, seen_usage)


def _finish(result: dict, started: float, tokens: dict, seen_usage: bool) -> dict:
    result["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    result["usage"] = (
        {
            "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
            "thinking_tokens": tokens["thinking"],
        }
        if seen_usage
        else None
    )
    return result
