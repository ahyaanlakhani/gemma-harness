"""Unit tests: validation, bounded recovery, budgets, and terminal outcomes."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.genai import errors, types  # noqa: E402

from demo import scripted  # noqa: E402
from gemma_harness import Client, validate_args  # noqa: E402
from gemma_harness.trace import write_jsonl  # noqa: E402


class QueueAdapter:
    """Returns scripted replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self.replies:
            raise AssertionError("model was called more times than scripted")
        return self.replies.pop(0)


class FailingAdapter:
    def __init__(self, code: int):
        self.code = code
        self.calls = 0

    def generate(self, *, model, contents, config):
        self.calls += 1
        raise errors.APIError(
            self.code, {"error": {"code": self.code, "message": "nope", "status": "ERR"}}
        )


def labels(outcome) -> list[str]:
    return [event["label"] for event in outcome["trace"]]


@pytest.fixture()
def tools():
    counts = {"weather": 0, "add": 0, "note": 0, "boom": 0, "flag": 0, "nap": 0}

    def get_weather(city: str) -> dict:
        """Weather for a city."""
        counts["weather"] += 1
        return {"city": city, "temp_c": 31}

    def add(a: float, b: float) -> float:
        """Add two numbers."""
        counts["add"] += 1
        return a + b

    def save_note(text: str) -> dict:
        """Save a note."""
        counts["note"] += 1
        return {"note_id": f"note-{counts['note']}", "text": text}

    def boom(query: str) -> dict:
        """Always fails."""
        counts["boom"] += 1
        raise ConnectionError("upstream down")

    def set_flag(flag: bool) -> bool:
        """Set a boolean flag."""
        counts["flag"] += 1
        return flag

    def nap() -> str:
        """Sleep briefly."""
        counts["nap"] += 1
        time.sleep(0.05)
        return "done"

    fns = [get_weather, add, save_note, boom, set_flag, nap]
    return {fn.__name__: fn for fn in fns}, counts


# --------------------------------------------------------------------------- #
# happy path and recovery
# --------------------------------------------------------------------------- #
def test_plain_success_is_success_not_recovered(tools):
    table, counts = tools
    adapter = QueueAdapter(
        [
            scripted.calls(("get_weather", {"city": "Karachi"})),
            scripted.text("Karachi is 31C."),
        ]
    )
    out = Client(adapter=adapter, max_steps=4).run("weather?", tools=[table["get_weather"]])

    assert out["status"] == "success" and out["failure_reason"] is None
    assert out["text"] == "Karachi is 31C."
    assert out["steps"] == 2
    assert counts["weather"] == 1
    assert not {"INVALID_ARGS", "REPAIR_REQUESTED", "TOOL_ERROR"} & set(labels(out))


def test_empty_turn_is_repaired_then_answered(tools):
    table, _ = tools
    adapter = QueueAdapter(
        [scripted.empty("MALFORMED_FUNCTION_CALL"), scripted.text("It is 31C.")]
    )
    out = Client(adapter=adapter, max_repairs=1).run("weather?", tools=[table["get_weather"]])

    assert out["status"] == "recovered"
    assert out["text"] == "It is 31C."
    assert "EMPTY_RESPONSE" in labels(out) and "REPAIR_REQUESTED" in labels(out)
    # the repair is a real user turn carrying the failure reason
    repair_turn = adapter.calls[1]["contents"][-1]
    assert "unusable" in repair_turn.parts[0].text


def test_failed_repair_is_an_explicit_failure_not_an_answer(tools):
    table, _ = tools
    out = Client(
        adapter=QueueAdapter([scripted.empty("STOP"), scripted.empty("STOP")]),
        max_repairs=1,
    ).run("weather?", tools=[table["get_weather"]])

    assert out["status"] == "failed"
    assert out["failure_reason"] == "empty_response"
    assert "could not complete" in out["text"].lower()
    assert "31" not in out["text"]  # never fabricates tool data
    assert "FAILED" in labels(out)


def test_safety_refusal_is_not_retried(tools):
    table, _ = tools
    adapter = QueueAdapter([scripted.empty("SAFETY")])
    out = Client(adapter=adapter, max_repairs=2).run("something", tools=[table["get_weather"]])

    assert out["status"] == "failed" and out["failure_reason"] == "safety_refusal"
    assert len(adapter.calls) == 1
    assert "REPAIR_REQUESTED" not in labels(out)


# --------------------------------------------------------------------------- #
# tool name and argument validation: nothing invalid executes
# --------------------------------------------------------------------------- #
def test_unknown_tool_is_not_executed_and_model_is_told_valid_names(tools):
    table, counts = tools
    adapter = QueueAdapter(
        [
            scripted.calls(("get_wether", {"city": "Karachi"})),
            scripted.calls(("get_weather", {"city": "Karachi"})),
            scripted.text("It is 31C."),
        ]
    )
    out = Client(adapter=adapter).run("weather?", tools=[table["get_weather"]])

    assert out["status"] == "recovered" and out["text"] == "It is 31C."
    assert counts["weather"] == 1  # the bad call never executed
    error = adapter.calls[1]["contents"][-1].parts[0].function_response.response
    assert "Unknown tool" in error["error"] and "get_weather" in error["error"]
    assert "UNKNOWN_TOOL" in labels(out)
    assert "ARGS_COERCED" not in labels(out)  # no silent fuzzy repair


def test_unknown_argument_is_rejected_not_dropped(tools):
    table, counts = tools
    adapter = QueueAdapter(
        [
            scripted.calls(("get_weather", {"location": "Karachi"})),
            scripted.calls(("get_weather", {"city": "Karachi"})),
            scripted.text("31C."),
        ]
    )
    out = Client(adapter=adapter).run("weather?", tools=[table["get_weather"]])

    assert counts["weather"] == 1  # only the corrected call executed
    assert "INVALID_ARGS" in labels(out)
    rejected = adapter.calls[1]["contents"][-1].parts[0].function_response.response
    assert "unknown argument 'location'" in rejected["error"]


def test_missing_required_argument_blocks_execution(tools):
    table, counts = tools
    out = Client(
        adapter=QueueAdapter(
            [scripted.calls(("get_weather", {})), scripted.text("Which city?")]
        )
    ).run("weather?", tools=[table["get_weather"]])

    assert counts["weather"] == 0
    assert "INVALID_ARGS" in labels(out)
    assert out["text"] == "Which city?"


def test_string_is_not_silently_turned_into_a_bool(tools):
    table, counts = tools
    out = Client(
        adapter=QueueAdapter(
            [scripted.calls(("set_flag", {"flag": "true"})), scripted.text("ok")]
        )
    ).run("flag", tools=[table["set_flag"]])

    assert counts["flag"] == 0
    assert "INVALID_ARGS" in labels(out)


def test_unambiguous_numeric_strings_are_coerced_and_logged(tools):
    table, counts = tools
    adapter = QueueAdapter(
        [
            scripted.calls(("add", {"a": "7", "b": "8"})),
            scripted.text("15"),
        ]
    )
    out = Client(adapter=adapter).run("add", tools=[table["add"]])

    assert counts["add"] == 1 and out["status"] == "success"
    event = next(e for e in out["trace"] if e["label"] == "ARGS_COERCED")
    assert event["coercions"] == {"a": "string->float", "b": "string->float"}
    result = adapter.calls[1]["contents"][-1].parts[0].function_response.response
    assert result["result"] == 15.0


def test_validate_args_reports_every_problem_without_partial_output(tools):
    table, _ = tools
    accepted, problems, _ = validate_args(table["add"], {"a": "x", "b": 1, "c": 2})
    assert accepted == {}
    assert any("unknown argument 'c'" in p for p in problems)
    assert any("must be float" in p for p in problems)

    accepted, problems, _ = validate_args(table["get_weather"], {"city": "Karachi"})
    assert problems == [] and accepted == {"city": "Karachi"}

    accepted, problems, coercions = validate_args(table["add"], {"a": 7, "b": 8.0})
    assert problems == [] and accepted == {"a": 7.0, "b": 8.0}
    assert coercions == {}


# --------------------------------------------------------------------------- #
# tool execution failures and repetition
# --------------------------------------------------------------------------- #
def test_tool_exception_becomes_a_structured_error_and_can_be_recovered(tools):
    table, counts = tools
    adapter = QueueAdapter(
        [
            scripted.calls(("boom", {"query": "population"})),
            scripted.text("The lookup failed; I have no figure."),
        ]
    )
    out = Client(adapter=adapter).run("lookup", tools=[table["boom"]])

    assert out["status"] == "recovered" and counts["boom"] == 1
    payload = adapter.calls[1]["contents"][-1].parts[0].function_response.response
    assert "ConnectionError" in payload["error"]
    assert "TOOL_ERROR" in labels(out)


def test_tool_errors_are_not_cached(tools):
    table, counts = tools
    adapter = QueueAdapter(
        [
            scripted.calls(("boom", {"query": "x"})),
            scripted.calls(("boom", {"query": "x"})),
            scripted.text("It failed both times."),
        ]
    )
    out = Client(adapter=adapter, cacheable_tools=("boom",)).run(
        "lookup twice", tools=[table["boom"]]
    )

    assert counts["boom"] == 2  # errors never serve from cache
    assert "CACHED_TOOL_RESULT" not in labels(out)


def test_repeated_read_only_call_uses_cache(tools):
    table, counts = tools
    adapter = QueueAdapter(
        [
            scripted.calls(("get_weather", {"city": "Karachi"})),
            scripted.calls(("get_weather", {"city": "Karachi"})),
            scripted.text("31C."),
        ]
    )
    out = Client(adapter=adapter, cacheable_tools=("get_weather",)).run(
        "check twice", tools=[table["get_weather"]]
    )

    assert counts["weather"] == 1
    assert "CACHED_TOOL_RESULT" in labels(out)
    assert out["status"] == "success"  # a cache hit is not a recovered failure


def test_identical_writes_are_never_cached(tools):
    table, counts = tools
    adapter = QueueAdapter(
        [
            scripted.calls(("save_note", {"text": "x"})),
            scripted.calls(("save_note", {"text": "x"})),
            scripted.text("saved twice"),
        ]
    )
    out = Client(adapter=adapter, cacheable_tools=("get_weather",)).run(
        "save twice", tools=[table["save_note"], table["get_weather"]]
    )

    assert counts["note"] == 2
    assert "CACHED_TOOL_RESULT" not in labels(out)


def test_unknown_cacheable_tool_is_a_programming_error(tools):
    table, _ = tools
    with pytest.raises(ValueError):
        Client(adapter=QueueAdapter([]), cacheable_tools=("nope",)).run(
            "x", tools=[table["add"]]
        )


# --------------------------------------------------------------------------- #
# protocol fidelity
# --------------------------------------------------------------------------- #
def test_tool_call_ids_and_model_parts_are_preserved(tools):
    table, _ = tools
    signature = b"opaque-thought-signature"
    call = types.FunctionCall(name="get_weather", args={"city": "Karachi"}, id="call-123")
    model_turn = types.Part(function_call=call, thought_signature=signature)
    adapter = QueueAdapter(
        [scripted.MockResponse([model_turn]), scripted.text("31C.")]
    )
    Client(adapter=adapter).run("weather", tools=[table["get_weather"]])

    echo = adapter.calls[1]["contents"][1]
    assert echo.parts[0].thought_signature == signature
    response = adapter.calls[1]["contents"][-1].parts[0].function_response
    assert response.id == "call-123"


# --------------------------------------------------------------------------- #
# budgets and transport
# --------------------------------------------------------------------------- #
def test_step_budget_fails_explicitly(tools):
    table, _ = tools
    adapter = QueueAdapter([scripted.calls(("get_weather", {"city": "Karachi"}))] * 3)
    out = Client(adapter=adapter, max_steps=3).run("loop", tools=[table["get_weather"]])

    assert out["status"] == "failed" and out["failure_reason"] == "max_steps"
    assert out["steps"] == 3
    assert "FAILED" in labels(out)


def test_model_call_budget_stops_before_extra_calls(tools):
    table, _ = tools
    adapter = QueueAdapter([scripted.calls(("get_weather", {"city": "Karachi"}))])
    out = Client(adapter=adapter, max_steps=5, max_model_calls=1).run(
        "loop", tools=[table["get_weather"]]
    )

    assert out["failure_reason"] == "model_call_budget"
    assert len(adapter.calls) == 1


def test_time_budget_fails_explicitly(tools):
    table, _ = tools
    out = Client(
        adapter=QueueAdapter(
            [
                scripted.calls(("nap", {})),
                scripted.calls(("nap", {})),
            ]
        ),
        max_steps=5,
        time_budget_s=0.02,
    ).run("nap", tools=[table["nap"]])

    assert out["failure_reason"] == "time_budget"
    assert out["status"] == "failed"


def test_transport_retries_are_bounded_and_explicit(tools):
    adapter = FailingAdapter(429)
    out = Client(adapter=adapter, transport_retries=2).run("hi", tools=[])

    assert adapter.calls == 3
    assert out["status"] == "failed" and out["failure_reason"] == "api_error"
    assert labels(out).count("API_RETRY") == 2


def test_non_retryable_api_error_is_not_retried(tools):
    adapter = FailingAdapter(400)
    out = Client(adapter=adapter, transport_retries=2).run("hi", tools=[])

    assert adapter.calls == 1
    assert out["failure_reason"] == "api_error"


# --------------------------------------------------------------------------- #
# result/trace surface
# --------------------------------------------------------------------------- #
def test_result_shape_and_trace_jsonl(tmp_path):
    out = Client(
        adapter=QueueAdapter([scripted.text("hello")])
    ).run("hi", tools=[])
    assert set(out) == {"status", "text", "failure_reason", "steps", "trace", "usage"}
    assert out.usage["model_calls"] == 1
    assert out.usage["elapsed_ms"] >= 0
    assert out.usage["input_tokens"] is None  # scripted adapter reports none

    path = write_jsonl(out["trace"], str(tmp_path / "t.jsonl"))
    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert lines == out["trace"]
    assert all({"t", "step", "label"} <= set(line) for line in lines)


def test_on_event_hook_sees_every_event():
    seen = []
    out = Client(
        adapter=QueueAdapter([scripted.text("hi")]), on_event=seen.append
    ).run("hi", tools=[])
    assert [e["label"] for e in seen] == [e["label"] for e in out["trace"]]
    assert seen[-1]["label"] == "FINAL"
