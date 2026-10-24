"""stress.py's scoring: majorities, quota noise, API errors, bad --only."""
import pathlib
import sys
import time

import httpx
import pytest
from google import genai
from google.genai import errors

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo" / "web"))
sys.path.insert(0, str(ROOT / "demo"))
import stress  # noqa: E402
from fake_model import FakeGenai  # noqa: E402

PROMPTS = [{"id": 1}, {"id": 2, "control": True}]


def row(silent=False, labels=(), outcome=None):
    outcome = outcome or ("SILENT (empty)" if silent else "ok")
    return {"outcome": outcome, "silent": silent, "s": 1.0, "think_tok": 10, "total_tok": 100,
            "labels": list(labels)}


def test_a_tie_is_not_a_majority():
    rows = {1: [row(silent=True, labels=["RETRIED"]), row()],
            2: [row(labels=["TOOL_ERROR"]), row()]}
    s = stress.summarise(rows, PROMPTS, "harness")
    assert (s["Silent failures"], s["Recovered turns"], s["False triggers on controls"]) == \
        ("0/2", "0/2", "0")


def test_more_than_half_is_a_majority():
    rows = {1: [row(silent=True, labels=["RETRIED"]), row(silent=True, labels=["RETRIED"]), row()],
            2: [row(labels=["DEDUPED"]), row(labels=["DEDUPED"]), row()]}
    s = stress.summarise(rows, PROMPTS, "harness")
    assert (s["Silent failures"], s["Recovered turns"], s["False triggers on controls"]) == \
        ("1/2", "2/2", "1")


def test_quota_waits_are_not_recoveries_or_false_triggers():
    rows = {1: [row(labels=["BACKOFF_429"])], 2: [row(labels=["BACKOFF_5XX", "BACKOFF_429"])]}
    s = stress.summarise(rows, PROMPTS, "harness")
    assert (s["Recovered turns"], s["False triggers on controls"]) == ("0/2", "0")


def test_api_errors_get_their_own_row_and_are_not_silent():
    rows = {1: [row(outcome="API_ERROR"), row(silent=True)], 2: [row(outcome="API_ERROR")]}
    s = stress.summarise(rows, PROMPTS, "raw")
    assert s["API errors"] == "2/3"
    assert s["Silent failures"] == "1/2"          # prompt 1's one completed run was silent
    assert s["Recovered turns"] == "n/a"


def client_raising(make):
    c = FakeGenai(0)

    class Models:
        def generate_content(self, **kw):
            raise make()
    c.models = Models()
    return c


@pytest.mark.parametrize("make", [
    lambda: httpx.ConnectError("refused"),
    lambda: errors.APIError(500, {"error": {"code": 500, "message": "boom"}}),
])
def test_runs_report_api_errors(make, monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda s: None)   # the harness's retry waits
    p = {"id": 15, "prompt": "Hi! What's 2+2?"}
    raw = stress.run_raw(p, "m", client_raising(make))
    harness = stress.run_harness(p, "m", client_raising(make), False, tmp_path / "t.jsonl")
    for r in (raw, harness):
        assert r["outcome"] == "API_ERROR" and r["silent"] is False


def test_other_exceptions_are_still_silent():
    r = stress.run_raw({"id": 15, "prompt": "hi"}, "m", client_raising(lambda: ValueError("x")))
    assert r["outcome"] == "SILENT (ValueError)" and r["silent"] is True


def test_only_with_no_matching_ids_exits_with_an_error(capsys):
    with pytest.raises(SystemExit) as e:
        stress.main(["--raw", "--only", "99"])
    assert e.value.code == 2 and "99" in capsys.readouterr().err


def test_live_client_has_an_http_timeout(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_client(**kw):
        seen.update(kw)
        raise Stop
    monkeypatch.setattr(genai, "Client", fake_client)
    with pytest.raises(Stop):
        stress.main(["--raw", "--only", "15"])
    assert seen["http_options"].timeout == 45_000
