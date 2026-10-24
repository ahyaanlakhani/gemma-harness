"""Deterministic fault injection.

Scripted replies are fed to *both* the baseline loop and the harness. They are
injected failures, not claims about how Gemma misbehaves. Every recovery
fixture ends with text derived from the actual tool result, so a passing check
means the task really completed.
"""

from __future__ import annotations

from typing import Any, Callable

from google.genai import types

from gemma_harness.adapters import SchemaClient  # noqa: F401  (documents intent)

__all__ = [
    "MockResponse",
    "ScriptedAdapter",
    "SCENARIOS",
    "scenario_by_id",
    "last_payload",
    "all_results",
]


class MockResponse:
    def __init__(self, parts=(), finish_reason: str = "STOP") -> None:
        self._parts = list(parts)
        self.candidates = [
            type(
                "Candidate",
                (),
                {
                    "content": types.Content(role="model", parts=list(self._parts)),
                    "finish_reason": finish_reason,
                },
            )()
        ]
        self._text = "".join(p.text for p in self._parts if getattr(p, "text", None))
        self.usage_metadata = None  # injected runs report no token usage

    @property
    def text(self) -> str:
        return self._text

    @property
    def function_calls(self) -> list[Any]:
        return [
            part.function_call
            for part in self._parts
            if getattr(part, "function_call", None) is not None
        ]


def text(body: str, finish: str = "STOP") -> MockResponse:
    return MockResponse([types.Part.from_text(text=body)], finish_reason=finish)


def calls(*pairs: tuple[str, dict], finish: str = "STOP") -> MockResponse:
    return MockResponse(
        [
            types.Part(function_call=types.FunctionCall(name=name, args=dict(args)))
            for name, args in pairs
        ],
        finish_reason=finish,
    )


def empty(finish: str = "STOP") -> MockResponse:
    return MockResponse([], finish_reason=finish)


def last_payload(contents: Any) -> dict:
    """The newest function response payload the model can see."""
    for content in reversed(contents or []):
        for part in getattr(content, "parts", None) or []:
            response = getattr(getattr(part, "function_response", None), "response", None)
            if isinstance(response, dict):
                return response
    return {}


def all_results(contents: Any) -> list[Any]:
    found = []
    for content in contents or []:
        for part in getattr(content, "parts", None) or []:
            response = getattr(getattr(part, "function_response", None), "response", None)
            if isinstance(response, dict) and "result" in response:
                found.append(response["result"])
    return found


def _turn_index(contents: Any) -> int:
    return max(
        0,
        sum(
            1
            for content in contents or []
            if (getattr(content, "role", None) or "") in ("user", "tool")
        )
        - 1,
    )


class ScriptedAdapter:
    """Replays one scripted reply per conversation turn. Records every call."""

    def __init__(self, turns: list[Any]) -> None:
        self.turns = turns
        self.calls: list[dict] = []

    def generate(self, *, model: str, contents: list, config: Any) -> MockResponse:
        turn = _turn_index(contents)
        spec = self.turns[min(turn, len(self.turns) - 1)]
        self.calls.append({"turn": turn, "contents": contents, "config": config, "model": model})
        response = spec(contents) if callable(spec) else spec
        return response

    @property
    def system_instructions(self) -> list[str]:
        return [
            call["config"].system_instruction or "" for call in self.calls
        ]


def _weather_text(contents: Any) -> MockResponse:
    payload = last_payload(contents)
    if "error" in payload:
        return text(f"I could not get the weather: {payload['error']}")
    result = payload.get("result") or {}
    return text(
        f"The weather in {result.get('city', 'the city')} is "
        f"{result.get('temp_c')}C and {result.get('conditions')}."
    )


def _sum_text(contents: Any) -> MockResponse:
    payload = last_payload(contents)
    if "error" in payload:
        return text(f"The arithmetic tool failed: {payload['error']}")
    return text(f"The result is {payload['result']}.")


def _lookup_failure_text(contents: Any) -> MockResponse:
    payload = last_payload(contents)
    if "error" not in payload:
        return text(f"The lookup returned {payload.get('result')}.")
    return text(
        "The population lookup failed (" + str(payload["error"]).split(":")[0] +
        "), so I have no figure. I will not invent one."
    )


def _notes_text(contents: Any) -> MockResponse:
    results = all_results(contents)
    ids = [str(r.get("note_id")) for r in results if isinstance(r, dict) and r.get("note_id")]
    return text(f"Saved {len(ids)} notes: " + " and ".join(ids) + ".")


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "empty_then_answer",
        "fault": "empty reply, then a valid tool call",
        "kind": "answer",
        "prompt": "What's the weather in Karachi? Use get_weather.",
        "turns": [
            empty("MALFORMED_FUNCTION_CALL"),
            calls(("get_weather", {"city": "Karachi"})),
            _weather_text,
        ],
        "check": lambda t: "31" in t,
    },
    {
        "id": "unknown_tool",
        "fault": "hallucinated tool name, then corrected",
        "kind": "answer",
        "prompt": "What's the weather in Karachi? Use get_wether.",
        "turns": [
            calls(("get_wether", {"city": "Karachi"})),
            calls(("get_weather", {"city": "Karachi"})),
            _weather_text,
        ],
        "check": lambda t: "31" in t,
    },
    {
        "id": "invalid_args",
        "fault": "unparsable number, then corrected",
        "kind": "answer",
        "prompt": "Add 'seven' and 30 using add.",
        "turns": [
            calls(("add", {"a": "seven", "b": 30})),
            calls(("add", {"a": 7, "b": 30})),
            _sum_text,
        ],
        "check": lambda t: "37" in t,
    },
    {
        "id": "numeric_strings",
        "fault": "numbers sent as strings (unambiguous coercion only)",
        "kind": "answer",
        "prompt": "Add \"7\" and \"8\" exactly as strings using add.",
        "turns": [
            calls(("add", {"a": "7", "b": "8"})),
            _sum_text,
        ],
        "check": lambda t: "15" in t,
    },
    {
        "id": "tool_exception",
        "fault": "tool raises ConnectionError",
        "kind": "honest_failure",
        "prompt": "Look up the population of Pakistan with fail_lookup.",
        "turns": [
            calls(("fail_lookup", {"query": "population of Pakistan"})),
            _lookup_failure_text,
        ],
        "check": lambda t: "failed" in t.lower() and "241" not in t,
    },
    {
        "id": "repeated_read",
        "fault": "same read-only call twice",
        "kind": "answer",
        "prompt": "Check the weather in Karachi twice.",
        "turns": [
            calls(("get_weather", {"city": "Karachi"})),
            calls(("get_weather", {"city": "Karachi"})),
            _weather_text,
        ],
        "check": lambda t: "31" in t,
    },
    {
        "id": "write_twice",
        "fault": "same write call twice (must NOT be deduplicated)",
        "kind": "answer",
        "prompt": "Save the note 'hello' twice.",
        "turns": [
            calls(("save_note", {"text": "hello"})),
            calls(("save_note", {"text": "hello"})),
            _notes_text,
        ],
        "check": lambda t: "note-1" in t and "note-2" in t,
    },
    {
        "id": "control_success",
        "fault": "none (control)",
        "kind": "answer",
        "prompt": "Add 2 and 3 using add.",
        "turns": [
            calls(("add", {"a": 2, "b": 3})),
            _sum_text,
        ],
        "check": lambda t: "5" in t,
    },
    {
        "id": "failed_recovery",
        "fault": "empty reply, repair also empty",
        "kind": "explicit_failure",
        "prompt": "What's the weather in Karachi? Use get_weather.",
        "turns": [empty("STOP"), empty("STOP")],
        "check": lambda t: "could not complete" in t.lower(),
    },
]


def scenario_by_id(scenario_id: str) -> dict:
    for scenario in SCENARIOS:
        if scenario["id"] == scenario_id:
            return scenario
    raise KeyError(scenario_id)
