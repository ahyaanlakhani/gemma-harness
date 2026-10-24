"""gemma-harness — bounded recovery for Gemma tool loops, with explicit outcomes."""

from .adapters import Adapter, GenaiAdapter
from .client import (
    STATUSES,
    STATUS_FAILED,
    STATUS_RECOVERED,
    STATUS_SUCCESS,
    Client,
    RunOutcome,
    validate_args,
)
from .trace import format_event, label_counts, pretty_print, write_jsonl

__version__ = "0.2.0"

__all__ = [
    "Adapter",
    "Client",
    "GenaiAdapter",
    "RunOutcome",
    "STATUSES",
    "STATUS_FAILED",
    "STATUS_RECOVERED",
    "STATUS_SUCCESS",
    "format_event",
    "label_counts",
    "pretty_print",
    "validate_args",
    "write_jsonl",
    "__version__",
]
