"""Model adapters: one tiny interface so the loop is testable without an API key."""

from __future__ import annotations

from typing import Any, Protocol

from google import genai
from google.genai import types

__all__ = ["Adapter", "GenaiAdapter", "SchemaClient"]


class Adapter(Protocol):
    """Anything that can answer one ``generate_content`` call."""

    def generate(
        self, *, model: str, contents: list, config: types.GenerateContentConfig
    ) -> Any: ...


class SchemaClient:
    """Stand-in used only to build function declarations.

    ``FunctionDeclaration.from_callable`` touches exactly one attribute of the
    client it is given (``vertexai``), so declarations do not require a network
    client -- which keeps scripted adapters and tests fully offline.
    """

    vertexai = False


class GenaiAdapter:
    """The real adapter, with the SDK's own retry loop turned off.

    google-genai retries transient HTTP failures internally (about 4 attempts).
    Leaving that on top of the harness budget makes retry accounting impossible,
    so a client created here is pinned to ``attempts=1`` and the harness owns
    retrying. If you inject your own ``genai.Client``, its retries remain.
    """

    def __init__(self, client: genai.Client | None = None, api_key: str | None = None) -> None:
        if client is None:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)
                ),
            )
        self.client = client

    def generate(
        self, *, model: str, contents: list, config: types.GenerateContentConfig
    ) -> Any:
        return self.client.models.generate_content(
            model=model, contents=contents, config=config
        )
