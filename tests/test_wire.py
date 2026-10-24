"""End-to-end tests through the real google-genai HTTP path, against a local stub.

No API key and no network: a stub speaks the Generative Language JSON shape and
the harness talks to it through a real ``genai.Client``. Covers request
serialisation, tool-call id echo, and that transport retries do not multiply.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from gemma_harness import Client, GenaiAdapter  # noqa: E402


def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp_c": 31}


class StubServer:
    """Serves queued replies and records every request body."""

    def __init__(self) -> None:
        self.replies: list[tuple[int, dict]] = []
        self.bodies: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                outer.bodies.append(json.loads(self.rfile.read(length) or b"{}"))
                status, payload = outer.replies.pop(0) if outer.replies else (500, {})
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "StubServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def adapter(self, sdk_retries: int = 1) -> GenaiAdapter:
        client = genai.Client(
            api_key="test-key",
            http_options=types.HttpOptions(
                base_url=f"http://127.0.0.1:{self.httpd.server_address[1]}",
                retry_options=types.HttpRetryOptions(attempts=sdk_retries),
            ),
        )
        return GenaiAdapter(client=client)


def call_payload(name: str, args: dict, call_id: str | None = None) -> dict:
    call: dict = {"name": name, "args": args}
    if call_id:
        call["id"] = call_id
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"functionCall": call}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 6},
    }


def text_payload(text: str) -> dict:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 60, "candidatesTokenCount": 5},
    }


def error_payload(code: int, message: str, status: str) -> dict:
    return {"error": {"code": code, "message": message, "status": status}}


def test_round_trip_echoes_model_turn_and_tool_call_id():
    with StubServer() as server:
        server.replies = [
            (200, call_payload("get_weather", {"city": "Lahore"}, call_id="call-abc")),
            (200, text_payload("Lahore is 31C.")),
        ]
        out = Client(adapter=server.adapter()).run(
            "Weather in Lahore?", tools=[get_weather]
        )

        assert out["status"] == "success"
        assert out["text"] == "Lahore is 31C."
        assert out.usage["model_calls"] == 2
        assert out.usage["input_tokens"] == 100  # both responses reported usage

        first, second = server.bodies
        declared = first["tools"][0]["functionDeclarations"][0]
        assert declared["name"] == "get_weather"
        assert declared["parameters"]["required"] == ["city"]
        assert "thinkingConfig" not in json.dumps(first)

        echoed = second["contents"][1]
        assert echoed["role"] == "model"
        response = second["contents"][-1]["parts"][0]["functionResponse"]
        assert response["id"] == "call-abc"
        assert response["response"]["result"]["temp_c"] == 31


def test_unknown_tool_over_the_wire_is_bounced_then_repaired():
    with StubServer() as server:
        server.replies = [
            (200, call_payload("get_wether", {"city": "Lahore"})),
            (200, call_payload("get_weather", {"city": "Lahore"})),
            (200, text_payload("Lahore is 31C.")),
        ]
        out = Client(adapter=server.adapter()).run("Weather in Lahore?", tools=[get_weather])

        assert out["status"] == "recovered"
        error = server.bodies[1]["contents"][-1]["parts"][0]["functionResponse"]
        assert "Unknown tool" in error["response"]["error"]
        assert "get_weather" in error["response"]["error"]
        # the bad call never reached a tool: the first request is the only one
        # before the model is told the name was wrong
        assert len(server.bodies) == 3


def test_sdk_retry_is_disabled_so_retry_layers_do_not_multiply():
    with StubServer() as server:
        server.replies = [(503, error_payload(503, "overloaded", "UNAVAILABLE"))]
        out = Client(adapter=server.adapter(sdk_retries=1), transport_retries=0).run(
            "hi", tools=[get_weather]
        )

        assert len(server.bodies) == 1  # exactly one HTTP attempt
        assert out["status"] == "failed"
        assert out["failure_reason"] == "api_error"


def test_harness_transport_retry_recovers_after_one_503(monkeypatch):
    monkeypatch.setattr("gemma_harness.client.time.sleep", lambda _s: None)
    with StubServer() as server:
        server.replies = [(503, error_payload(503, "overloaded", "UNAVAILABLE")), (200, text_payload("ok"))]
        out = Client(adapter=server.adapter(), transport_retries=1).run("hi", tools=[])

        assert out["status"] == "success" and out["text"] == "ok"
        assert len(server.bodies) == 2
        assert "API_RETRY" in [event["label"] for event in out["trace"]]
