"""Chaos mode: inject the failures gemma-harness recovers from, once each per run.

Wrap a genai client and your tools, then run any agent loop on top:

    raw = ChaosClient(genai.Client(), {"bad_names", "blank"})
    tools = chaos_tools([get_weather, add], {"crash_tools"})

Make a fresh wrapper per run: each fault fires at its first opportunity, then
never again, so the run can still finish.
"""
import functools
from typing import Callable

from google.genai import types

FAULTS = ("bad_names", "bad_args", "blank", "crash_tools")


def _check(faults) -> set:
    unknown = set(faults) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown faults: {sorted(unknown)}; known: {list(FAULTS)}")
    return set(faults)


class ChaosClient:
    """Looks like a genai.Client; replies from `models.generate_content` get faults."""

    def __init__(self, raw, faults, on_fire: Callable[[str], None] | None = None):
        self._raw = raw
        self.faults = _check(faults)
        self.fired: list[str] = []
        self.on_fire = on_fire
        self.models = _ChaosModels(self)

    def __getattr__(self, name):
        # Everything else (used to build tool declarations) comes from the real client.
        return getattr(self._raw, name)

    def _fire(self, fault: str) -> bool:
        if fault not in self.faults or fault in self.fired:
            return False
        self.fired.append(fault)
        if self.on_fire:
            self.on_fire(fault)
        return True


class _ChaosModels:
    def __init__(self, chaos: ChaosClient):
        self._chaos = chaos

    def generate_content(self, **kw):
        c = self._chaos
        if c._fire("blank"):   # skip the real call: saves quota, same effect
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=None, finish_reason="MALFORMED_FUNCTION_CALL")])
        r = c._raw.models.generate_content(**kw)
        calls = r.function_calls or []
        if calls and c._fire("bad_names"):
            calls[0].name = calls[0].name[:-1]
        with_args = [fc for fc in calls if fc.args]
        if with_args and c._fire("bad_args"):
            fc = with_args[0]
            fc.args = {(_typo(k) if i == 0 else k): _stringify(v)
                       for i, (k, v) in enumerate(fc.args.items())}
        return r


def _typo(key: str) -> str:
    """Double the middle character: city -> citty, a -> aa."""
    i = len(key) // 2
    return key[:i] + key[i] + key[i:]


def _stringify(v):
    return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def chaos_tools(tools, faults, on_fire: Callable[[str], None] | None = None) -> list[Callable]:
    """With `crash_tools`, the first call to any of these tools raises ConnectionError."""
    tools = list(tools)
    if "crash_tools" not in _check(faults):
        return tools
    state = {"fired": False}

    def wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not state["fired"]:
                state["fired"] = True
                if on_fire:
                    on_fire("crash_tools")
                raise ConnectionError("chaos: tool crashed")
            return fn(*args, **kwargs)
        return wrapper

    return [wrap(f) for f in tools]
