"""Local demo tools. All data is simulated; nothing leaves this machine."""

from __future__ import annotations

from itertools import count

__all__ = ["DEMO_TOOLS", "CACHEABLE_TOOLS", "get_weather", "add", "save_note",
           "fail_lookup", "reset_notes"]

_note_ids = count(1)


def reset_notes() -> None:
    """Keep baseline and harness runs comparable (fresh id sequence per run)."""
    global _note_ids
    _note_ids = count(1)


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: City name.
    """
    return {"city": city, "temp_c": 31, "conditions": "clear", "simulated": True}


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def save_note(text: str) -> dict:
    """Save a note and return its id. This tool writes state.

    Args:
        text: Note body.
    """
    note_id = f"note-{next(_note_ids)}"
    return {"note_id": note_id, "text": text}


def fail_lookup(query: str) -> dict:
    """Look something up upstream. Deliberately broken.

    Args:
        query: What to look up.
    """
    raise ConnectionError("upstream lookup service is unavailable")


DEMO_TOOLS = [get_weather, add, save_note, fail_lookup]

# Only read-only tools may be cacheable. save_note writes state and must never be
# deduplicated: two identical calls can be two intended actions.
CACHEABLE_TOOLS = ("get_weather",)
