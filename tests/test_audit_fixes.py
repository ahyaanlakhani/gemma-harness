"""Regression tests for the audit fixes. Each one failed before its fix."""
import datetime
import json
import time

import httpx
import pytest
from google.genai import errors, types

from gemma_harness import client as hc
from gemma_harness.trace import RECOVERY_LABELS
from test_recovery import add, blank_resp, call_resp, get_weather, harness, labels, text_resp

DEMO_TOOLS = dict.fromkeys(["get_weather", "add", "save_note", "slow_lookup", "flaky_db"])


def trace_labels(h):
    return [ev["label"] for ev in h.trace]


def record_history(h) -> list:
    """Capture the `contents` list each model call gets (one list, grown in place)."""
    seen = []
    generate = h.raw.models.generate_content

    def rec(model, contents, config):
        seen.append(contents)
        return generate(model=model, contents=contents, config=config)

    h.raw.models.generate_content = rec
    return seen


def call_with_id(call_id, name, args):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        role="model", parts=[types.Part(function_call=types.FunctionCall(
            id=call_id, name=name, args=args))]), finish_reason="STOP")])


@pytest.fixture
def sleeps(monkeypatch):
    got = []
    monkeypatch.setattr(time, "sleep", got.append)
    return got


# ---------- 1. dedupe must not make a one-off crash permanent ----------
def test_failed_call_runs_again_instead_of_being_deduped():
    state = {"n": 0}

    def flaky(q: str) -> str:
        """Fails the first time only."""
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionError("blip")
        return "ok"

    script = [call_resp(("flaky", {"q": "x"})), call_resp(("flaky", {"q": "x"})), text_resp("ok")]
    out = harness(script).run("go", tools=[flaky])
    ls = labels(out)
    assert ls.count("TOOL_ERROR") == 1 and "TOOL_OK" in ls and "DEDUPED" not in ls
    assert state["n"] == 2 and out["text"] == "ok"


def test_error_is_cached_after_its_second_identical_failure():
    state = {"n": 0}

    def broken(q: str) -> str:
        """Always fails."""
        state["n"] += 1
        raise ConnectionError("down")

    script = [call_resp(("broken", {"q": "x"})) for _ in range(3)] + [text_resp("gave up")]
    out = harness(script).run("go", tools=[broken])
    ls = labels(out)
    assert ls.count("TOOL_ERROR") == 2 and ls.count("DEDUPED") == 1 and state["n"] == 2


# ---------- 2. fuzzy name repair stays within the same verb ----------
@pytest.mark.parametrize("got,want", [
    ("get_weather_now", "get_weather"), ("get_wether", "get_weather"),
    ("get_weathe", "get_weather"), ("ad", "add"), ("flaky_d", "flaky_db"),
])
def test_tool_name_typos_are_still_repaired(got, want):
    h = harness([])
    assert hc.fix_tool_name(h, got, DEMO_TOOLS, 1) == want
    assert trace_labels(h) == ["TOOL_NAME_FIXED"]


@pytest.mark.parametrize("got", ["delete_note", "read_note", "get_note", "get_user"])
def test_tool_name_repair_never_crosses_verbs(got):
    h = harness([])
    assert hc.fix_tool_name(h, got, DEMO_TOOLS, 1) is None
    assert trace_labels(h) == ["UNKNOWN_TOOL"]


def test_lint_near_miss_uses_the_same_rule():
    h = harness([])
    assert hc.lint_prompt("Use get_weather_now please", DEMO_TOOLS, h) == ""
    assert hc.lint_prompt("Use delete_note please", DEMO_TOOLS, h)   # a phantom, not a typo
    assert h.trace[-1]["names"] == ["delete_note"]


# ---------- 3. tool results are made JSON-safe ----------
def test_non_json_tool_result_is_made_serializable():
    def stamp(q: str) -> dict:
        """Return a date and a set."""
        return {"at": datetime.date(2026, 1, 1), "tags": {"a"}}

    h = harness([call_resp(("stamp", {"q": "x"})), text_resp("done")])
    hist = record_history(h)
    out = h.run("when?", tools=[stamp])
    resp = hist[-1][-1].parts[0].function_response.response
    assert json.loads(json.dumps(resp)) == {"result": {"at": "2026-01-01", "tags": "{'a'}"}}
    assert out["text"] == "done"


# ---------- 4. transient API failures are backed off ----------
@pytest.mark.parametrize("code", [408, 502, 504])
def test_transient_status_is_backed_off(sleeps, code):
    out = harness([errors.APIError(code, {"error": {"message": "x"}}), text_resp("hi")]).run("hi")
    assert labels(out) == ["BACKOFF_5XX", "CALL", "FINAL"] and out["text"] == "hi"
    assert len(sleeps) == 1


def retry_info(delay):
    return errors.APIError(429, {"error": {"message": "quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}})


def test_429_honours_server_retry_delay(sleeps):
    out = harness([retry_info("3s"), text_resp("hi")]).run("hi")
    assert labels(out)[0] == "BACKOFF_429" and out["text"] == "hi"
    assert sleeps[0] >= 3


def test_server_retry_delay_is_capped(sleeps):
    harness([retry_info("600s"), text_resp("hi")]).run("hi")
    assert sleeps[0] <= 60


def test_transport_error_is_backed_off(sleeps):
    out = harness([httpx.ConnectError("refused"), text_resp("hi")]).run("hi")
    ev = out["trace"][0]
    assert ev["label"] == "BACKOFF_5XX" and ev["reason"] == "ConnectError"
    assert out["text"] == "hi"


def test_transport_error_is_raised_after_the_last_try(sleeps):
    with pytest.raises(httpx.ReadTimeout):
        harness([httpx.ReadTimeout("slow")] * hc.MAX_API_TRIES).run("hi")


def test_default_client_sets_an_http_timeout(monkeypatch):
    got = {}
    monkeypatch.setattr(hc.genai, "Client", lambda **kw: got.update(kw) or object())
    hc.Client(api_key="k")
    assert got["api_key"] == "k" and got["http_options"].timeout == 120_000


# ---------- 5. lint only fires on tool mentions, not prose ----------
@pytest.mark.parametrize("prompt", [
    "Write a read_only config and list_files, then run_tests",
    "Fetch the user_id from the post_body.",
])
def test_lint_ignores_snake_case_prose(prompt):
    h = harness([])
    assert hc.lint_prompt(prompt, DEMO_TOOLS, h) == "" and h.trace == []


@pytest.mark.parametrize("prompt,names", [
    ("Use web_search to find today's news in Karachi, then summarise it.", ["web_search"]),
    ("Call send_email to tell Ali the meeting moved to 5pm.", ["send_email"]),
    ("Then call the `create_ticket` tool.", ["create_ticket"]),
    ("First run search_docs(query).", ["search_docs"]),
    ("The send_sms tool is handy.", ["send_sms"]),
])
def test_lint_still_flags_named_tools(prompt, names):
    h = harness([])
    assert hc.lint_prompt(prompt, DEMO_TOOLS, h)
    assert h.trace[0]["names"] == names


# ---------- 7. a usable first reply is not thrown away ----------
def test_usable_first_reply_is_kept_when_the_retry_is_blank():
    h = harness([text_resp("It is 31C.", reason="OTHER"), blank_resp()])
    out = h.run("weather?", tools=[get_weather])
    assert out["text"] == "It is 31C."
    assert "RETRIED" in labels(out) and "SILENT_FAIL_PREVENTED" not in labels(out)


# ---------- 8. a reply cut off at MAX_TOKENS is labelled ----------
def test_max_tokens_reply_is_labelled_truncated():
    text = "It is 31C and"
    out = harness([text_resp(text, reason="MAX_TOKENS")]).run("weather?", tools=[get_weather])
    assert labels(out)[-2:] == ["TRUNCATED", "FINAL"] and out["text"] == text
    assert out["trace"][-2]["chars"] == len(text) and out["trace"][-2]["step"] == 1
    assert "TRUNCATED" in RECOVERY_LABELS


def test_normal_stop_is_not_truncated():
    out = harness([text_resp("hi")]).run("hi")
    assert "TRUNCATED" not in labels(out)


# ---------- 9. history: repaired name and call id ----------
def test_repaired_name_and_call_id_reach_history():
    h = harness([call_with_id("call-7", "get_wether", {"city": "Karachi"}), text_resp("31C")])
    hist = record_history(h)
    h.run("weather", tools=[get_weather])
    model_turn, tool_turn = hist[-1][1], hist[-1][2]
    assert model_turn.parts[0].function_call.name == "get_weather"
    fr = tool_turn.parts[0].function_response
    assert (fr.id, fr.name) == ("call-7", "get_weather")
    assert fr.response["result"]["city"] == "Karachi"


def test_unknown_tool_response_carries_call_id():
    h = harness([call_with_id("call-9", "send_email", {"to": "Ali"}), text_resp("can't")])
    hist = record_history(h)
    h.run("email Ali", tools=[get_weather])
    fr = hist[-1][2].parts[0].function_response
    assert (fr.id, fr.name) == ("call-9", "send_email") and "unknown tool" in fr.response["error"]


# ---------- 10. small argument bugs ----------
def tag(name: str, *rest, **extra) -> dict:
    """Tag something."""
    return {"name": name, **extra}


def test_var_params_are_never_missing_and_kwargs_pass_through():
    h = harness([])
    args = hc.fix_args(h, tag, {"name": "n", "color": "red"}, 1)
    assert args == {"name": "n", "color": "red"}
    assert hc.missing_args(tag, args) == [] and h.trace == []
    assert hc.missing_args(tag, {}) == ["name"]


def test_base_exception_in_tool_is_a_tool_error():
    def quits(q: str) -> str:
        raise SystemExit(3)

    h = harness([])
    res = hc.call_tool(h, quits, {"q": "x"}, 1)
    assert "SystemExit" in res["error"] and trace_labels(h) == ["TOOL_ERROR"]


@pytest.mark.parametrize("args", [{"a": 2, "b": 3, "bb": 99}, {"bb": 99, "a": 2, "b": 3}])
def test_fuzzy_key_rename_never_overwrites_a_set_arg(args):
    h = harness([])
    assert hc.fix_args(h, add, args, 1) == {"a": 2, "b": 3}
    assert h.trace[-1]["fixes"] == ["drop:bb"]
