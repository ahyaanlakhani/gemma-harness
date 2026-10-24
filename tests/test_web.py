"""Demo server end to end, in fake mode: live race, errors, busy lock, replay."""
import http.client
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

import httpx
import pytest
from google.genai import errors, types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demo" / "web"))
import server  # noqa: E402
from fake_model import FakeGenai  # noqa: E402
from server import api_error_message, list_replays, make_server, save_replay  # noqa: E402


@pytest.fixture
def srv(tmp_path):
    s = make_server(port=0, fake=True, model="fake", replay_dir=tmp_path, fake_delay_s=0)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s
    s.shutdown()
    s.server_close()


def get(s, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{s.server_address[1]}{path}", timeout=30) as r:
        return r.status, r.read().decode()


def status_of(s, path):
    try:
        return get(s, path)[0]
    except urllib.error.HTTPError as e:
        return e.code


def events(body):
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def verdicts(evs):
    return {e["side"]: e["verdict"] for e in evs if e["label"] == "VERDICT"}


def test_index_and_prompts(srv):
    assert "GEMMA-HARNESS" in get(srv, "/")[1]
    assert get(srv, "/static/app.js")[0] == 200
    assert len(json.loads(get(srv, "/api/prompts")[1])) == 15


def test_live_run_streams_both_lanes_with_verdicts(srv):
    evs = events(get(srv, "/api/run?prompt_id=3&faults=bad_names,bad_args")[1])
    assert evs[0]["label"] == "START" and evs[0]["mode"] == "fake" and evs[-1]["label"] == "END"
    assert verdicts(evs) == {"raw": "SILENT_FAILURE", "harness": "ANSWERED"}
    harness = [e["label"] for e in evs if e["side"] == "harness"]
    assert "TOOL_NAME_FIXED" in harness and "ARGS_COERCED" in harness and "CHAOS" in harness
    assert any(e["label"] == "CRASH" for e in evs if e["side"] == "raw")


def test_blank_reply_is_silent_on_raw_and_retried_by_harness(srv):
    evs = events(get(srv, "/api/run?prompt_id=1&faults=blank")[1])
    assert verdicts(evs) == {"raw": "SILENT_FAILURE", "harness": "ANSWERED"}
    assert "RETRIED" in [e["label"] for e in evs if e["side"] == "harness"]


def test_no_faults_both_answer(srv):
    evs = events(get(srv, "/api/run?prompt_id=15")[1])
    assert verdicts(evs) == {"raw": "ANSWERED", "harness": "ANSWERED"}


def test_bad_requests(srv):
    assert status_of(srv, "/api/run?prompt_id=99") == 400
    assert status_of(srv, "/api/run?prompt_id=x") == 400
    assert status_of(srv, "/api/run?prompt_id=3&faults=rate_limit") == 400
    assert status_of(srv, "/api/replay?name=../server.py") == 404
    assert status_of(srv, "/static/../server.py") == 404
    assert status_of(srv, "/nope") == 404


def test_busy_returns_409(srv):
    srv.run_lock.acquire()
    try:
        assert status_of(srv, "/api/run?prompt_id=3") == 409
    finally:
        srv.run_lock.release()


def test_saved_run_replays(srv):
    live = events(get(srv, "/api/run?prompt_id=3&faults=blank")[1])
    saved = json.loads(get(srv, "/api/replays")[1])
    assert len(saved) == 1 and saved[0]["prompt_id"] == 3 and saved[0]["faults"] == ["blank"]
    rep = events(get(srv, f"/api/replay?name={saved[0]['name']}")[1])
    assert rep[0]["mode"] == "replay" and rep[0]["original_mode"] == "fake"
    assert [e["label"] for e in rep[1:]] == [e["label"] for e in live[1:]]


@pytest.fixture
def keyless(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = make_server(port=0, fake=False, model="m", replay_dir=tmp_path)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s
    s.shutdown()
    s.server_close()


def test_missing_key_reports_error(keyless):
    evs = events(get(keyless, "/api/run?prompt_id=3")[1])
    assert [e["label"] for e in evs] == ["START", "ERROR", "END"]
    assert evs[1]["side"] == "run"


# ---------- sweep: the whole pack in one stream ----------
def labelled(evs, label):
    return [e for e in evs if e["label"] == label]


def test_sweep_streams_each_prompt_then_a_summary(srv):
    evs = events(get(srv, "/api/sweep?ids=1,15&faults=bad_names")[1])
    first, last = evs[0], evs[-1]
    assert first["side"] == "run" and first["label"] == "SWEEP_START"
    assert first["ids"] == [1, 15] and first["faults"] == ["bad_names"]
    assert first["mode"] == "fake" and first["model"] == "fake"
    assert last["side"] == "run" and last["label"] == "END"
    assert len(labelled(evs, "END")) == 1                      # no per-prompt END
    assert all("t" in e for e in evs)

    starts = labelled(evs, "START")
    assert [(e["prompt_id"], e["sweep"], e["index"], e["total"]) for e in starts] == \
        [(1, True, 1, 2), (15, True, 2, 2)]
    assert starts[0]["faults"] == ["bad_names"] and starts[0]["mode"] == "fake"

    # each prompt's block is START .. both lane verdicts .. ROW, in order
    s1, s2 = evs.index(starts[0]), evs.index(starts[1])
    for block, pid in ((evs[s1:s2], 1), (evs[s2:-2], 15)):
        assert block[-1]["label"] == "ROW" and block[-1]["prompt_id"] == pid
        assert verdicts(block) == {"raw": "SILENT_FAILURE", "harness": "ANSWERED"}

    rows = labelled(evs, "ROW")
    assert [r["prompt_id"] for r in rows] == [1, 15]
    assert [r["control"] for r in rows] == [False, True]
    for r in rows:
        assert r["side"] == "sweep"
        assert r["raw"]["verdict"] == "SILENT_FAILURE"
        assert r["harness"]["verdict"] == "ANSWERED"
        assert "TOOL_NAME_FIXED" in r["harness"]["labels"]
        assert r["harness"]["fixes"] == len(r["harness"]["labels"])
        assert r["raw"]["s"] >= 0 and r["harness"]["s"] >= 0

    (summary,) = labelled(evs, "SUMMARY")
    assert evs[-2] is summary and summary["side"] == "sweep"
    assert {k: summary[k] for k in ("total", "raw_silent", "harness_silent", "harness_fallback",
                                    "harness_answered", "false_triggers")} == \
        {"total": 2, "raw_silent": 2, "harness_silent": 0, "harness_fallback": 0,
         "harness_answered": 2, "false_triggers": 1}           # 15 is a control and got fixed
    assert summary["fixes"] == sum(r["harness"]["fixes"] for r in rows)


def test_sweep_control_without_faults_has_no_false_trigger(srv):
    evs = events(get(srv, "/api/sweep?ids=15&delay=0.01")[1])
    (summary,) = labelled(evs, "SUMMARY")
    assert summary["total"] == 1 and summary["false_triggers"] == 0
    assert labelled(evs, "ROW")[0]["harness"]["verdict"] == "ANSWERED"


def test_sweep_bad_requests(srv):
    for q in ("ids=99", "ids=x", "ids=1,99", "delay=-1", "delay=abc", "delay=31",
              "faults=rate_limit"):
        assert status_of(srv, f"/api/sweep?{q}") == 400, q


def test_sweep_busy_returns_409(srv):
    srv.run_lock.acquire()
    try:
        assert status_of(srv, "/api/sweep?ids=1") == 409
    finally:
        srv.run_lock.release()
    assert status_of(srv, "/api/sweep?ids=1") == 200          # lock released after a sweep
    assert status_of(srv, "/api/run?prompt_id=1") == 200


def test_sweep_saves_one_replay_per_prompt(srv):
    get(srv, "/api/run?prompt_id=3")
    before = json.loads(get(srv, "/api/replays")[1])
    get(srv, "/api/sweep?ids=1,15&faults=bad_names")
    after = json.loads(get(srv, "/api/replays")[1])
    assert len(after) == len(before) + 2
    new = [r for r in after if r["name"] not in {b["name"] for b in before}]
    assert sorted(r["prompt_id"] for r in new) == [1, 15]
    assert all(r["faults"] == ["bad_names"] and r["mode"] == "fake" for r in new)
    rep = events(get(srv, f"/api/replay?name={new[0]['name']}")[1])
    assert rep[0]["label"] == "START" and rep[-1]["label"] == "ROW"


def test_replays_in_the_same_millisecond_do_not_overwrite(tmp_path):
    evs = [{"side": "run", "label": "START", "prompt_id": 1, "faults": [], "mode": "fake",
            "t": 1000.0}]
    save_replay(tmp_path, 1, evs)
    save_replay(tmp_path, 1, evs)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["1000000-p1.jsonl", "1000001-p1.jsonl"]


def test_sweep_stops_on_run_level_error(keyless):
    evs = events(get(keyless, "/api/sweep")[1])
    assert evs[0]["label"] == "SWEEP_START" and evs[0]["ids"] == list(range(1, 16))
    assert [e["label"] for e in evs] == ["SWEEP_START", "START", "ERROR", "END"]
    assert evs[2]["side"] == "run"


# ---------- stub model clients ----------
class StubModels:
    def __init__(self, fn):
        self.fn = fn

    def generate_content(self, **kw):
        return self.fn(**kw)


def stub_client(fn):
    """A fake-mode client whose every model call goes through fn(**kw)."""
    c = FakeGenai(0)
    c.models = StubModels(fn)
    return c


def use_stub(monkeypatch, fn):
    """Make the fake-mode server build stub_client(fn) instead of the scripted model."""
    monkeypatch.setattr(server, "FakeGenai", lambda delay_s=0: stub_client(fn))


def raising(make):
    def fn(**kw):
        raise make()
    return fn


def quota_error():
    return errors.APIError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                           "message": "Resource exhausted. Try again later."}})


def lane_errors(evs):
    return {e["side"]: e["message"] for e in evs if e["label"] == "ERROR"}


def no_backoff_waits(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)   # the harness's API/network retry waits


# ---------- 1. a stalled API call can't hang a run ----------
def test_live_client_has_an_http_timeout(keyless, monkeypatch):
    seen = {}

    def fake_client(**kw):
        seen.update(kw)
        raise RuntimeError("no key")
    monkeypatch.setattr(server.genai, "Client", fake_client)
    evs = events(get(keyless, "/api/run?prompt_id=3")[1])
    assert [e["label"] for e in evs] == ["START", "ERROR", "END"]
    assert seen["http_options"].timeout == 45_000


def test_stalled_model_times_out_and_frees_the_lock(srv, monkeypatch):
    use_stub(monkeypatch, lambda **kw: time.sleep(3))
    srv.run_timeout_s = 0.3
    t0 = time.time()
    evs = events(get(srv, "/api/run?prompt_id=15")[1])
    assert time.time() - t0 < 2.5
    assert [e["label"] for e in evs[-2:]] == ["ERROR", "END"]
    assert evs[-2]["side"] == "run" and evs[-2]["message"].startswith("Timed out waiting")
    assert not verdicts(evs)
    assert srv.run_lock.acquire(blocking=False)
    srv.run_lock.release()
    assert json.loads(get(srv, "/api/replays")[1]) == []        # a failed run isn't saved


def test_sweep_stops_on_timeout(srv, monkeypatch):
    use_stub(monkeypatch, lambda **kw: time.sleep(3))
    srv.run_timeout_s = 0.3
    labels = [e["label"] for e in events(get(srv, "/api/sweep?ids=1,15")[1])]
    assert labels[0] == "SWEEP_START" and labels[-2:] == ["ERROR", "END"]
    assert labels.count("START") == 1 and "ROW" not in labels and "SUMMARY" not in labels


# ---------- 2. network and API errors are lane errors, not silent failures ----------
def test_api_error_message():
    assert api_error_message(quota_error()) == "API error 429: Resource exhausted. Try again later"
    assert api_error_message(httpx.ReadTimeout("timed out")) == "Network error (ReadTimeout): timed out"


def test_network_error_is_a_lane_error(srv, monkeypatch):
    no_backoff_waits(monkeypatch)
    use_stub(monkeypatch, raising(lambda: httpx.ConnectError("connection refused")))
    evs = events(get(srv, "/api/run?prompt_id=15")[1])
    msg = "Network error (ConnectError): connection refused"
    assert lane_errors(evs) == {"raw": msg, "harness": msg}
    assert not verdicts(evs) and not labelled(evs, "CRASH")
    assert json.loads(get(srv, "/api/replays")[1]) == []


def test_quota_error_is_a_lane_error(srv, monkeypatch):
    no_backoff_waits(monkeypatch)
    use_stub(monkeypatch, raising(quota_error))
    evs = events(get(srv, "/api/run?prompt_id=15")[1])
    msg = "API error 429: Resource exhausted. Try again later"
    assert lane_errors(evs) == {"raw": msg, "harness": msg}
    assert not verdicts(evs)
    assert "BACKOFF_429" in [e["label"] for e in evs if e["side"] == "harness"]


def test_chaos_tool_crash_is_still_a_raw_crash(srv):
    evs = events(get(srv, "/api/run?prompt_id=15&faults=crash_tools")[1])
    crash = [e for e in evs if e["side"] == "raw" and e["label"] == "CRASH"]
    assert crash and crash[0]["error"].startswith("ConnectionError")
    assert verdicts(evs)["raw"] == "SILENT_FAILURE"


def test_sweep_keeps_going_past_lane_errors_but_saves_nothing(srv, monkeypatch):
    no_backoff_waits(monkeypatch)
    use_stub(monkeypatch, raising(lambda: httpx.ConnectError("down")))
    evs = events(get(srv, "/api/sweep?ids=1,15")[1])
    rows = labelled(evs, "ROW")
    assert [(r["raw"]["verdict"], r["harness"]["verdict"]) for r in rows] == [("ERROR", "ERROR")] * 2
    assert labelled(evs, "SUMMARY")[0]["total"] == 2
    assert json.loads(get(srv, "/api/replays")[1]) == []


# ---------- 3. waiting out a quota is not a fix ----------
def test_quota_waits_are_not_fixes(monkeypatch):
    no_backoff_waits(monkeypatch)
    real = FakeGenai(0)
    calls = []

    def first_call_429(**kw):
        calls.append(1)
        if len(calls) == 1:
            raise quota_error()
        return real.models.generate_content(**kw)

    evs = [server.stamp({"side": "run", "label": "START", "prompt_id": 15})]
    server.harness_side(server.PROMPTS[15]["prompt"], ["bad_names"], stub_client(first_call_429),
                        "m", lambda ev: evs.append(server.stamp(ev)))
    harness = [e["label"] for e in evs if e["side"] == "harness"]
    assert "BACKOFF_429" in harness                     # still shown in the lane
    (verdict,) = labelled(evs, "VERDICT")
    assert verdict["verdict"] == "ANSWERED" and verdict["fixes"] == 1
    row = server.sweep_row(15, evs)
    assert row["harness"]["labels"] == ["TOOL_NAME_FIXED"] and row["harness"]["fixes"] == 1
    assert server.sweep_summary([row])["fixes"] == 1


def test_backoff_alone_is_no_fix_and_no_false_trigger():
    evs = [{"side": "run", "label": "START", "prompt_id": 15, "t": 1.0},
           {"side": "harness", "label": "BACKOFF_429", "t": 2.0},
           {"side": "harness", "label": "BACKOFF_5XX", "t": 3.0},
           {"side": "harness", "label": "VERDICT", "verdict": "ANSWERED", "fixes": 0, "t": 4.0}]
    row = server.sweep_row(15, evs)
    assert row["harness"]["labels"] == [] and row["harness"]["fixes"] == 0
    summary = server.sweep_summary([row])
    assert summary["fixes"] == 0 and summary["false_triggers"] == 0


# ---------- 5. replays: no failed runs, and each one's outcome ----------
def test_missing_key_run_is_not_saved(keyless):
    get(keyless, "/api/run?prompt_id=3")
    assert json.loads(get(keyless, "/api/replays")[1]) == []


def test_replay_list_shows_outcomes(srv):
    get(srv, "/api/run?prompt_id=3&faults=bad_names")
    (r,) = json.loads(get(srv, "/api/replays")[1])
    assert (r["raw"], r["harness"], r["fixes"]) == ("SILENT_FAILURE", "ANSWERED", 1)


def test_list_replays_reads_verdicts_and_errors(tmp_path):
    start = {"side": "run", "label": "START", "prompt_id": 3, "faults": [], "mode": "live", "t": 1000.0}
    save_replay(tmp_path, 3, [start, {"side": "raw", "label": "ERROR", "message": "x", "t": 1001.0},
                              {"side": "harness", "label": "VERDICT", "verdict": "FALLBACK",
                               "fixes": 2, "text": "", "t": 1002.0}])
    save_replay(tmp_path, 4, [{**start, "prompt_id": 4, "t": 2000.0}])
    assert [(r["prompt_id"], r["raw"], r["harness"], r["fixes"]) for r in list_replays(tmp_path)] == \
        [(4, None, None, None), (3, "ERROR", "FALLBACK", 2)]


# ---------- 6. only this machine's page can start runs ----------
def raw_get(s, path, headers):
    conn = http.client.HTTPConnection("127.0.0.1", s.server_address[1], timeout=30)
    try:
        conn.request("GET", path, headers=headers)
        r = conn.getresponse()
        return r.status, r.read().decode()
    finally:
        conn.close()


def test_foreign_host_or_cross_site_is_refused(srv):
    port = srv.server_address[1]
    me = f"127.0.0.1:{port}"
    for path in ("/api/run?prompt_id=15", "/api/sweep?ids=15", "/api/replay?name=1-p1.jsonl"):
        for headers in ({"Host": f"evil.example:{port}"}, {"Host": "127.0.0.1:1"}, {"Host": "localhost"},
                        {"Host": me, "Sec-Fetch-Site": "cross-site"}):
            status, body = raw_get(srv, path, headers)
            assert status == 403 and "error" in json.loads(body), (path, headers)
    assert raw_get(srv, "/api/run?prompt_id=15", {"Host": f"localhost:{port}"})[0] == 200
    assert raw_get(srv, "/api/run?prompt_id=15", {"Host": me, "Sec-Fetch-Site": "same-origin"})[0] == 200
    assert raw_get(srv, "/api/prompts", {"Host": "evil.example"})[0] == 200


def test_static_rejects_drive_relative_names(srv):
    assert status_of(srv, f"/static/{server.STATIC.drive or 'C:'}app.js") == 404
    assert status_of(srv, "/static/app.js") == 200


# ---------- 7. mode before any run ----------
def test_info_reports_mode_and_model(srv, keyless):
    assert json.loads(get(srv, "/api/info")[1]) == {"mode": "fake", "model": "fake"}
    assert json.loads(get(keyless, "/api/info")[1]) == {"mode": "live", "model": "m"}


# ---------- 8. the fake model answers about the prompt's city ----------
def fake_turns(prompt):
    """(city the fake asks the weather for, its answer once it has the result)."""
    m = FakeGenai(0).models
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    first = m.generate_content(model="x", contents=contents)
    city = first.function_calls[0].args["city"]
    contents += [first.candidates[0].content, types.Content(role="user", parts=[
        types.Part.from_function_response(name="get_weather", response={"result": {"city": city}})])]
    return city, m.generate_content(model="x", contents=contents).text


def test_fake_model_uses_the_prompts_city(srv):
    assert fake_turns(server.PROMPTS[3]["prompt"]) == ("Lahore", "It's 31°C and dry in Lahore.")
    assert fake_turns(server.PROMPTS[14]["prompt"])[0] == "Islamabad"
    assert fake_turns(server.PROMPTS[15]["prompt"])[0] == "Karachi"
    evs = events(get(srv, "/api/run?prompt_id=3")[1])
    raw = [e for e in evs if e["side"] == "raw" and e["label"] == "VERDICT"][0]
    assert raw["text"] == "It's 31°C and dry in Lahore."
