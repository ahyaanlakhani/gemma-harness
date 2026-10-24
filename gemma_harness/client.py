"""gemma-harness: a bounded Gemma tool loop with explicit outcomes.

Contract: every ``run`` returns one of three terminal statuses.

* ``success``    -- the model answered on its own.
* ``recovered``  -- the harness repaired or bounced a bad turn, then the model
                    produced a final answer.
* ``failed``     -- no answer. ``failure_reason`` says why, and ``text`` is an
                    explicit apology, never a fabricated answer.

The loop never executes a tool call it could not validate, never retries a
safety refusal, never caches errors, and only deduplicates tools the caller has
explicitly marked as cacheable (read-only).
"""

from __future__ import annotations

import inspect
import json
import re
import time
import types as pytypes
import typing
from typing import Any, Callable, Iterable

from google.genai import errors, types

from .adapters import Adapter, GenaiAdapter, SchemaClient

__all__ = ["Client", "RunOutcome", "STATUSES", "validate_args"]

STATUS_SUCCESS = "success"
STATUS_RECOVERED = "recovered"
STATUS_FAILED = "failed"
STATUSES = (STATUS_SUCCESS, STATUS_RECOVERED, STATUS_FAILED)

# The provider reports these when the tool call itself was unusable.
UNUSABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",
    "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS",
    "FINISH_REASON_UNSPECIFIED",
    "OTHER",
}

# Never retried: a refusal is a decision, not a glitch.
REFUSAL_FINISH_REASONS = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "RECITATION",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "LANGUAGE",
}

RETRYABLE_CODES = {429, 500, 502, 503, 504}

_FAILURE_TEXT = (
    "I could not complete this task: {reason}. "
    "No answer was produced. Any tool calls that ran are listed in the trace."
)

_NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_INTEGER = re.compile(r"^[+-]?\d+$")


class _StopRun(Exception):
    """Internal: terminate the run with an explicit failure reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class RunOutcome(dict):
    """``{status, text, failure_reason, steps, trace, usage}`` with attribute access."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


def _candidate(response: Any) -> Any:
    candidates = getattr(response, "candidates", None) or []
    return candidates[0] if candidates else None


def _response_text(response: Any) -> str:
    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None
    return (text or "").strip()


def _function_calls(response: Any) -> list:
    try:
        return list(getattr(response, "function_calls", None) or [])
    except Exception:
        return []


def _finish_reason(response: Any) -> str:
    raw = getattr(_candidate(response), "finish_reason", None)
    if raw is None:
        return ""
    return str(raw).split(".")[-1].strip().upper()


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return {"repr": repr(value)[:2000]}


def _function_response_part(fc: Any, name: str, payload: dict) -> types.Part:
    """Echo the provider's tool-call id so the response is matched to the call."""
    call_id = getattr(fc, "id", None)
    response = types.FunctionResponse(name=name, response=payload)
    if call_id:
        response.id = call_id
    return types.Part(function_response=response)


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #
def _check_value(value: Any, expected: Any) -> tuple[bool, Any, str | None]:
    """Return (ok, coerced, note). Only unambiguous conversions are accepted."""
    if expected is None or expected is Any:
        return True, value, None
    origin = typing.get_origin(expected)
    if origin in (typing.Union, pytypes.UnionType):
        for option in typing.get_args(expected):
            ok, coerced, note = _check_value(value, option)
            if ok:
                return True, coerced, note
        return False, value, None
    if origin is not None:  # list[str], dict[str, int], ...
        container = origin
        if container is list:
            return isinstance(value, list), value, None
        if container is dict:
            return isinstance(value, dict), value, None
        if container is tuple:
            return isinstance(value, (list, tuple)), value, None
        return True, value, None
    if expected is bool:
        return isinstance(value, bool), value, None
    if expected is int:
        if isinstance(value, bool):
            return False, value, None
        if isinstance(value, int):
            return True, value, None
        if isinstance(value, float) and value.is_integer():
            return True, int(value), "float->int"
        if isinstance(value, str) and _INTEGER.match(value.strip()):
            return True, int(value.strip()), "string->int"
        return False, value, None
    if expected is float:
        if isinstance(value, bool):
            return False, value, None
        if isinstance(value, (int, float)):
            return True, float(value), None
        if isinstance(value, str) and _NUMERIC.match(value.strip()):
            return True, float(value.strip()), "string->float"
        return False, value, None
    if expected is str:
        return isinstance(value, str), value, None
    if isinstance(expected, type):
        return isinstance(value, expected), value, None
    return True, value, None


def validate_args(fn: Callable, args: Any) -> tuple[dict, list[str], dict[str, str]]:
    """Validate a tool call before execution.

    Returns ``(accepted_args, problems, coercions)``.  If ``problems`` is
    non-empty the caller must not execute the tool -- there is no partial call
    and no silent dropping of unknown keys.
    """
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    params = {
        name: param
        for name, param in signature.parameters.items()
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    accepts_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )

    if not isinstance(args, dict):
        return {}, [f"arguments must be a JSON object, got {type(args).__name__}"], {}

    problems: list[str] = []
    accepted: dict[str, Any] = {}
    coercions: dict[str, str] = {}
    supplied = set(args)

    for key, value in args.items():
        if key not in params:
            if accepts_kwargs:
                accepted[key] = value
            else:
                problems.append(
                    f"unknown argument {key!r}; this tool accepts {sorted(params)}"
                )
            continue
        ok, coerced, note = _check_value(value, hints.get(key))
        if not ok:
            expected = hints.get(key)
            problems.append(
                f"argument {key!r} must be "
                f"{getattr(expected, '__name__', expected) or 'JSON-compatible value'}, "
                f"got {value!r}"
            )
            continue
        accepted[key] = coerced
        if note:
            coercions[key] = note

    # Report a parameter as missing only when the model did not send it at all;
    # a rejected value already has its own, clearer problem line.
    missing = [
        name
        for name, param in params.items()
        if param.default is inspect.Parameter.empty and name not in supplied
    ]
    if missing:
        problems.append(f"missing required argument(s) {missing}")

    if problems:  # all-or-nothing: never hand back a partial call
        return {}, problems, {}
    return accepted, problems, coercions


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #
class Client:
    """Bounded Gemma tool loop with explicit terminal outcomes."""

    def __init__(
        self,
        model: str = "gemma-4-26b-a4b-it",
        adapter: Adapter | None = None,
        api_key: str | None = None,
        max_steps: int = 8,
        max_model_calls: int | None = None,
        time_budget_s: float = 60.0,
        max_repairs: int = 1,
        transport_retries: int = 2,
        cacheable_tools: Iterable[str] = (),
        system_instruction: str = "",
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.model = model
        self.adapter = adapter if adapter is not None else GenaiAdapter(api_key=api_key)
        self.max_steps = max(1, int(max_steps))
        self.max_model_calls = int(
            max_model_calls if max_model_calls is not None else self.max_steps * 2
        )
        self.time_budget_s = float(time_budget_s)
        self.max_repairs = max(0, int(max_repairs))
        self.transport_retries = max(0, int(transport_retries))
        self.cacheable_tools = tuple(cacheable_tools)
        self.system_instruction = system_instruction or ""
        self.on_event = on_event

        self.trace: list[dict] = []
        self._reset_budgets()

    # ---------- trace ----------
    def emit(self, step: int, label: str, **data: Any) -> None:
        event = {"t": round(time.time(), 3), "step": step, "label": label, **data}
        self.trace.append(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # a broken hook must not break the run
                pass

    def _reset_budgets(self) -> None:
        self._started = time.monotonic()
        self._model_calls = 0
        self._repairs = 0
        self._recovered = False
        self._usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        self._usage_seen = False

    # ---------- budget helpers ----------
    def _remaining_time(self) -> float | None:
        if self.time_budget_s <= 0:
            return None
        return self.time_budget_s - (time.monotonic() - self._started)

    def _check_budgets(self, step: int) -> None:
        remaining = self._remaining_time()
        if remaining is not None and remaining <= 0:
            self.emit(step, "BUDGET_EXCEEDED", budget="time", limit_s=self.time_budget_s)
            raise _StopRun("time_budget")
        if self._model_calls >= self.max_model_calls:
            self.emit(
                step, "BUDGET_EXCEEDED", budget="model_calls", limit=self.max_model_calls
            )
            raise _StopRun("model_call_budget")

    # ---------- model calls ----------
    def _generate(
        self, contents: list, decls: list, step: int, hint: str | None = None
    ) -> Any:
        system_text = self.system_instruction
        if hint:
            system_text = f"{system_text}\n{hint}".strip()
        config = types.GenerateContentConfig(
            system_instruction=system_text or None,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        for attempt in range(self.transport_retries + 1):
            self._check_budgets(step)
            self._model_calls += 1
            started = time.monotonic()
            try:
                response = self.adapter.generate(
                    model=self.model, contents=list(contents), config=config
                )
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                self.emit(step, "API_ERROR", code=code, attempt=attempt + 1)
                if code in RETRYABLE_CODES and attempt < self.transport_retries:
                    wait = min(4.0, 0.5 * 2**attempt)
                    self.emit(step, "API_RETRY", code=code, wait_s=wait, attempt=attempt + 1)
                    time.sleep(wait)
                    continue
                raise _StopRun("api_error") from exc
            except Exception as exc:  # adapter/transport failure
                self.emit(step, "API_ERROR", err=type(exc).__name__, attempt=attempt + 1)
                raise _StopRun("api_error") from exc

            self.emit(step, "MODEL_CALL", ms=int((time.monotonic() - started) * 1000))
            self._record_usage(getattr(response, "usage_metadata", None))
            return response
        raise _StopRun("api_error")  # pragma: no cover

    def _record_usage(self, metadata: Any) -> None:
        if metadata is None:
            return
        found = False
        for attr, key in (
            ("prompt_token_count", "input_tokens"),
            ("candidates_token_count", "output_tokens"),
            ("thoughts_token_count", "thinking_tokens"),
        ):
            value = getattr(metadata, attr, None)
            if value is not None:
                self._usage[key] += int(value)
                found = True
        self._usage_seen = self._usage_seen or found

    # ---------- terminal outcomes ----------
    def _usage_block(self) -> dict:
        usage = {
            "model_calls": self._model_calls,
            "input_tokens": self._usage["input_tokens"] if self._usage_seen else None,
            "output_tokens": self._usage["output_tokens"] if self._usage_seen else None,
            "thinking_tokens": (
                self._usage["thinking_tokens"] if self._usage_seen else None
            ),
            "elapsed_ms": int((time.monotonic() - self._started) * 1000),
        }
        return usage

    def _finish(self, step: int, text: str) -> RunOutcome:
        status = STATUS_RECOVERED if self._recovered else STATUS_SUCCESS
        self.emit(step, "FINAL", status=status, chars=len(text))
        return RunOutcome(
            status=status,
            text=text,
            failure_reason=None,
            steps=step,
            trace=self.trace,
            usage=self._usage_block(),
        )

    def _fail(self, step: int, reason: str, text: str | None = None) -> RunOutcome:
        message = text or _FAILURE_TEXT.format(reason=reason.replace("_", " "))
        self.emit(step, "FAILED", reason=reason, chars=len(message))
        return RunOutcome(
            status=STATUS_FAILED,
            text=message,
            failure_reason=reason,
            steps=step,
            trace=self.trace,
            usage=self._usage_block(),
        )

    # ---------- main loop ----------
    def run(self, prompt: str, tools: Iterable[Callable] = ()) -> RunOutcome:
        self.trace = []
        self._reset_budgets()

        tool_map: dict[str, Callable] = {}
        for fn in tools:
            name = getattr(fn, "__name__", None)
            if not name:
                raise TypeError(f"tools must be named callables, got {fn!r}")
            tool_map[name] = fn
        unknown_cacheable = set(self.cacheable_tools) - set(tool_map)
        if unknown_cacheable:
            raise ValueError(
                f"cacheable_tools {sorted(unknown_cacheable)} are not in this run's tools"
            )
        decls = [
            types.FunctionDeclaration.from_callable(
                client=SchemaClient(), callable=fn
            )
            for fn in tool_map.values()
        ]

        contents: list[Any] = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]
        cache: dict[str, Any] = {}
        step = 1
        try:
            for step in range(1, self.max_steps + 1):
                self._check_budgets(step)
                response = self._generate(contents, decls, step)

                if _finish_reason(response) in REFUSAL_FINISH_REASONS:
                    reason = _finish_reason(response).lower()
                    self.emit(step, "REFUSAL", finish_reason=reason)
                    return self._fail(
                        step,
                        "safety_refusal",
                        "The model declined this request "
                        f"({reason.replace('_', ' ')}), so no answer was produced.",
                    )

                response = self._repair_blank(contents, decls, step, response)
                calls = _function_calls(response)

                if not calls:
                    text = _response_text(response)
                    if text:
                        return self._finish(step, text)
                    return self._fail(step, "empty_response")

                candidate = _candidate(response)
                if candidate is not None and getattr(candidate, "content", None) is not None:
                    # Echo the model turn untouched: thought signatures included.
                    contents.append(candidate.content)

                parts = [
                    self._handle_call(fc, tool_map, cache, step) for fc in calls
                ]
                contents.append(types.Content(role="user", parts=parts))

            return self._fail(step, "max_steps")
        except _StopRun as stop:
            return self._fail(step, stop.reason)

    # ---------- blank / malformed repair (one bounded attempt) ----------
    def _repair_blank(self, contents: list, decls: list, step: int, response: Any) -> Any:
        reason = _finish_reason(response)
        blank = not _function_calls(response) and not _response_text(response)
        fixable = reason in UNUSABLE_FINISH_REASONS or blank
        if not fixable:
            return response

        self.emit(
            step,
            "EMPTY_RESPONSE" if blank else "UNUSABLE_RESPONSE",
            finish_reason=reason or "EMPTY",
        )
        if self._repairs >= self.max_repairs:
            return response

        self._repairs += 1
        self._recovered = True
        self.emit(
            step,
            "REPAIR_REQUESTED",
            reason=reason or "EMPTY",
            attempt=self._repairs,
        )
        names = [getattr(d, "name", "?") for d in decls]
        instruction = (
            "Your previous response was unusable "
            f"({reason or 'empty response'}). "
            f"Reply with either a valid call to one of {names} using correct argument "
            "names and JSON types, or a plain-text answer. Never return an empty reply."
        )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=instruction)])
        )
        repaired = self._generate(contents, decls, step)
        if _finish_reason(repaired) in REFUSAL_FINISH_REASONS:
            self.emit(step, "REFUSAL", finish_reason=_finish_reason(repaired).lower())
            raise _StopRun("safety_refusal")
        return repaired

    # ---------- one tool call ----------
    def _handle_call(
        self,
        fc: Any,
        tool_map: dict[str, Callable],
        cache: dict[str, Any],
        step: int,
    ) -> types.Part:
        name = (getattr(fc, "name", "") or "").strip()

        if name not in tool_map:
            self._recovered = True
            self.emit(step, "UNKNOWN_TOOL", got=name, valid=sorted(tool_map))
            return _function_response_part(
                fc,
                name or "unknown_tool",
                {
                    "error": (
                        f"Unknown tool {name!r}. Nothing was executed. "
                        f"Valid tools are {sorted(tool_map)}; re-send the call with one "
                        "of those names, or answer in plain text."
                    )
                },
            )

        arguments, problems, coercions = validate_args(tool_map[name], getattr(fc, "args", None) or {})
        if coercions:
            self.emit(step, "ARGS_COERCED", tool=name, coercions=coercions)
        if problems:
            self._recovered = True
            self.emit(step, "INVALID_ARGS", tool=name, problems=problems)
            return _function_response_part(
                fc,
                name,
                {
                    "error": (
                        f"Arguments rejected for {name}: {problems}. "
                        "Nothing was executed. Re-send a corrected call."
                    )
                },
            )

        if name in self.cacheable_tools:
            key = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
            if key in cache:
                self.emit(step, "CACHED_TOOL_RESULT", tool=name)
                return _function_response_part(
                    fc,
                    name,
                    {
                        "result": cache[key],
                        "note": "Repeated read-only call: this is the result already returned.",
                    },
                )

        try:
            result = tool_map[name](**arguments)
        except Exception as exc:  # noqa: BLE001 - reported to the model, not swallowed
            self._recovered = True
            self.emit(step, "TOOL_ERROR", tool=name, err=f"{type(exc).__name__}: {exc}"[:300])
            return _function_response_part(
                fc,
                name,
                {
                    "error": (
                        f"{type(exc).__name__}: {exc}. "
                        "The tool failed; do not invent its result."
                    )
                },
            )

        self.emit(step, "TOOL_OK", tool=name)
        serialised = _jsonable(result)
        if name in self.cacheable_tools:
            key = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
            cache[key] = serialised  # successes only
        return _function_response_part(fc, name, {"result": serialised})
