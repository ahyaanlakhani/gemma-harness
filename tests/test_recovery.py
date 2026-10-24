"""Mocked-response tests: each recovery path fires and gets the right label."""
import time

import pytest
from google import genai
from google.genai import errors, types

from gemma_harness import Client
from gemma_harness import client as hc


# ---------- fakes ----------
def text_resp(text, reason="STOP"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason=reason)])


def call_resp(*calls):
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=parts), finish_reason="STOP")])


def blank_resp(reason="MALFORMED_FUNCTION_CALL"):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=None, finish_reason=reason)])


class FakeModels:
    def __init__(self, script):
        self.script, self.configs = list(script), []

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        nxt = self.script.pop(0) if self.script else text_resp("done")
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FakeRaw:
    """A real client for building declarations, with scripted model replies."""
    def __init__(self, script):
        self._real = genai.Client(api_key="test")
        self.models = FakeModels(script)

    def __getattr__(self, name):
        return getattr(self._real, name)


def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp_c": 31}


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def fetch_user(user_id: int) -> dict:
    """Fetch a user."""
    return {"id": user_id}


def labels(out):
    return [ev["label"] for ev in out["trace"]]


def harness(script, **kw):
    return Client(raw=FakeRaw(script), **kw)


# ---------- tests ----------
def test_malformed_reply_is_retried_then_answered():
    h = harness([blank_resp(), text_resp("It is 31C.")])
    out = h.run("weather?", tools=[get_weather])
    assert out["text"] == "It is 31C."
    assert "RETRIED" in labels(out) and labels(out)[-1] == "FINAL"


def test_blank_forever_becomes_labelled_fallback_not_empty():
    h = harness([blank_resp(), blank_resp("OTHER")])
    out = h.run("weather?", tools=[get_weather])
    assert out["text"].strip()
    assert "SILENT_FAIL_PREVENTED" in labels(out)


def test_wrong_name_wrong_arg_and_repeat_are_repaired():
    h = harness([
        call_resp(("get_wether", {"citty": "Karachi"})),
        call_resp(("get_weather", {"city": "Karachi"})),
        text_resp("31C in Karachi."),
    ])
    out = h.run("weather in Karachi", tools=[get_weather])
    ls = labels(out)
    assert "TOOL_NAME_FIXED" in ls and "ARGS_COERCED" in ls and "DEDUPED" in ls
    assert out["text"] == "31C in Karachi."


def test_unknown_tool_reports_available_tools():
    h = harness([call_resp(("send_email", {"to": "Ali"})), text_resp("I can't send email.")])
    out = h.run("email Ali", tools=[get_weather, add])
    assert "UNKNOWN_TOOL" in labels(out)


def test_missing_args_are_reported_without_calling_the_tool():
    h = harness([call_resp(("add", {"a": 1})), text_resp("need b")])
    out = h.run("add", tools=[add])
    assert "ARGS_MISSING" in labels(out) and "TOOL_OK" not in labels(out)


def test_step_cap_on_endless_distinct_calls():
    script = [call_resp(("get_weather", {"city": f"c{i}"})) for i in range(10)]
    out = harness(script, max_steps=3).run("loop", tools=[get_weather])
    assert labels(out)[-1] == "STEP_CAP" and out["text"].strip()


def test_backoff_on_429(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    h = harness([errors.APIError(429, {"error": {"message": "quota"}}), text_resp("hi")])
    out = h.run("hi", tools=[])
    assert "BACKOFF_429" in labels(out) and out["text"] == "hi"


def test_thinking_is_routed_minimal_until_last_step():
    h = harness([text_resp("hi")], max_steps=4)
    h.run("hi", tools=[])
    level = h.raw.models.configs[0].thinking_config.thinking_level
    assert str(level).split(".")[-1] == "MINIMAL"


def test_lint_flags_phantom_but_leaves_near_misses_for_name_fix():
    h = harness([])
    tm = {"get_weather": get_weather}
    assert hc.lint_prompt("Use web_search please", tm, h)
    assert hc.lint_prompt("Use get_weather_now please", tm, h) == ""
    assert hc.lint_prompt("call send_email", tm, h)


@pytest.mark.parametrize("fn,args,expected", [
    (get_weather, {"location": "Karachi"}, {"city": "Karachi"}),   # one stray key, one gap
    (add, {"a": "7", "b": "8"}, {"a": 7.0, "b": 8.0}),              # strings to numbers
    (fetch_user, {"user_id": 42.0}, {"user_id": 42}),               # integral float to int
    (add, {"a": "twelve", "b": 30}, {"a": "twelve", "b": 30}),      # can't coerce: leave it
])
def test_fix_args(fn, args, expected):
    assert hc.fix_args(harness([]), fn, args, 1) == expected


def test_tool_timeout_and_error_are_returned_to_the_model():
    h = harness([], tool_timeout_s=0.2)

    def slow(q: str) -> str:
        time.sleep(5)
        return "x"

    def broken(q: str) -> str:
        raise ConnectionError("db down")

    assert "timed out" in hc.call_tool(h, slow, {"q": "x"}, 1)["error"]
    assert "ConnectionError" in hc.call_tool(h, broken, {"q": "x"}, 1)["error"]
    assert [ev["label"] for ev in h.trace] == ["TOOL_TIMEOUT", "TOOL_ERROR"]
