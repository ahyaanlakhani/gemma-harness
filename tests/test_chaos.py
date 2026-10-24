"""Chaos mode: each fault fires once, identically on both sides, and the harness recovers."""
import inspect
import typing

import pytest

from gemma_harness import Client
from gemma_harness.chaos import FAULTS, ChaosClient, chaos_tools
from test_recovery import FakeRaw, add, call_resp, get_weather, text_resp


def first_call(r):
    fc = r.function_calls[0]
    return fc.name, dict(fc.args)


def gen(c):
    return c.models.generate_content(model="m", contents=[], config=None)


def test_unknown_fault_rejected():
    with pytest.raises(ValueError):
        ChaosClient(FakeRaw([]), {"rate_limit"})


def test_blank_fires_once():
    c = ChaosClient(FakeRaw([text_resp("hi"), text_resp("hi")]), {"blank"})
    r1, r2 = gen(c), gen(c)
    assert r1.function_calls is None
    assert str(r1.candidates[0].finish_reason).endswith("MALFORMED_FUNCTION_CALL")
    assert r2.candidates[0].content.parts[0].text == "hi"
    assert c.fired == ["blank"]


def test_bad_names_and_bad_args_once_each():
    script = [call_resp(("add", {"a": 2, "b": 3.5})), call_resp(("add", {"a": 2, "b": 3.5}))]
    c = ChaosClient(FakeRaw(script), {"bad_names", "bad_args"})
    r1, r2 = gen(c), gen(c)
    assert first_call(r1) == ("ad", {"aa": "2", "b": "3.5"})
    assert first_call(r2) == ("add", {"a": 2, "b": 3.5})


def test_same_faults_same_mutations_on_both_sides():
    faults = set(FAULTS) - {"blank"}
    a = ChaosClient(FakeRaw([call_resp(("get_weather", {"city": "Karachi"}))]), faults)
    b = ChaosClient(FakeRaw([call_resp(("get_weather", {"city": "Karachi"}))]), faults)
    assert first_call(gen(a)) == first_call(gen(b)) == ("get_weathe", {"citty": "Karachi"})


def test_on_fire_reports_each_fault():
    fired = []
    c = ChaosClient(FakeRaw([call_resp(("add", {"a": 1, "b": 2}))]), {"blank", "bad_names"},
                    on_fire=fired.append)
    gen(c), gen(c)
    assert fired == ["blank", "bad_names"]


def test_crash_tools_fires_once_and_keeps_signature():
    fired = []
    wrapped = chaos_tools([get_weather, add], {"crash_tools"}, on_fire=fired.append)
    assert [f.__name__ for f in wrapped] == ["get_weather", "add"]
    assert inspect.signature(wrapped[1]) == inspect.signature(add)
    assert typing.get_type_hints(wrapped[1]) == typing.get_type_hints(add)
    with pytest.raises(ConnectionError):
        wrapped[0](city="K")
    assert wrapped[0](city="K")["city"] == "K" and wrapped[1](a=1, b=2) == 3
    assert fired == ["crash_tools"]


def test_no_crash_fault_leaves_tools_alone():
    assert chaos_tools([get_weather], {"blank"}) == [get_weather]


def test_harness_recovers_from_every_fault():
    script = [call_resp(("get_weather", {"city": "Karachi"})), text_resp("31C")]
    raw = ChaosClient(FakeRaw(script), set(FAULTS))
    out = Client(raw=raw).run("weather", tools=chaos_tools([get_weather], set(FAULTS)))
    labels = [ev["label"] for ev in out["trace"]]
    for want in ("RETRIED", "TOOL_NAME_FIXED", "ARGS_COERCED", "TOOL_ERROR"):
        assert want in labels
    assert out["text"] == "31C"
