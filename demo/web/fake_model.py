"""A scripted stand-in for genai.Client, so the demo runs with no API key.

For any prompt it first calls get_weather for the city the prompt names (Karachi if
none), then answers in text. Chaos faults apply on top, so every toggle can be rehearsed.
"""
import re
import time

from google import genai
from google.genai import types

CITIES = ("Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan", "Sukkur", "Gilgit",
          "Hunza", "Skardu", "Murree", "Swat", "Chitral", "Gwadar")
DEFAULT_CITY = "Karachi"


def city_in(text: str) -> str:
    """The first known city named in the text, else Karachi."""
    found = [(m.start(), c) for c in CITIES if (m := re.search(rf"\b{c}\b", text, re.IGNORECASE))]
    return min(found)[1] if found else DEFAULT_CITY


def answer(city: str) -> str:
    return f"It's 31°C and dry in {city}."


class FakeGenai:
    def __init__(self, delay_s: float = 0.6):
        # A real client (never used for requests) to build tool declarations.
        self._real = genai.Client(api_key="fake-mode")
        self.models = _FakeModels(delay_s)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FakeModels:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    def generate_content(self, *, model, contents, config=None):
        time.sleep(self.delay_s)   # so the race is visible on screen
        first = contents[0].parts[0] if contents and contents[0].parts else None
        city = city_in((first.text if first else None) or "")
        answered = any(p.function_response for c in contents for p in (c.parts or []))
        if answered:
            part = types.Part(text=answer(city))
        else:
            part = types.Part(function_call=types.FunctionCall(
                name="get_weather", args={"city": city}))
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[part]), finish_reason="STOP")])
