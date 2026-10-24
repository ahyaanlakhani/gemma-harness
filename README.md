# gemma-harness

Gemma 4 agents that never fail silently. Swap one import, and your Gemma 4 agent stops returning blank replies, calling phantom tools and looping. Every failure becomes a labelled recovery in a trace.

## What it fixes

| Failure | Recovery | Trace label |
|---|---|---|
| Prompt mentions a tool you didn't declare | Tells the model which tools exist, before the call | `PHANTOM_TOOL_LINTED` |
| Blank or malformed reply (`MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, ...) | Retries once with a hint; else an explicit fallback, never a blank | `RETRIED`, `SILENT_FAIL_PREVENTED` |
| Misspelled tool name | Fuzzy-matches to a declared tool | `TOOL_NAME_FIXED` / `UNKNOWN_TOOL` |
| Wrong arg name or type | Renames the key, coerces the type | `ARGS_COERCED` |
| Missing required arg | Tells the model what's missing instead of crashing | `ARGS_MISSING` |
| Tool raises or hangs | Returns the error to the model so it can adapt | `TOOL_ERROR` / `TOOL_TIMEOUT` |
| Same call repeated | Returns the cached result with a nudge; hard step cap | `DEDUPED` / `STEP_CAP` |
| 429 / 5xx | Exponential backoff with jitter | `BACKOFF_429` / `BACKOFF_5XX` |
| Thinking latency | Minimal thinking on tool-routing steps | `CALL` (ms, think_tok) |

## Install

```
pip install -e .[dev]
```

Set `GEMINI_API_KEY` to a key from Google AI Studio.

## The one-line swap

```python
# before
from google import genai
# after
from gemma_harness import Client

h = Client(model="gemma-4-26b-a4b-it", on_event=lambda ev: print(ev["label"], ev))
out = h.run("Weather in Karachi and add 2+3", tools=[get_weather, add])
out["text"]; out["trace"]; out["steps"]
h.raw  # the underlying genai.Client, for anything else
```

## Config

| Option | Default | Meaning |
|---|---|---|
| `model` | `gemma-4-26b-a4b-it` | Or `gemma-4-31b-it` |
| `api_key` | env `GEMINI_API_KEY` | |
| `max_steps` | 8 | Model calls per run before `STEP_CAP` |
| `max_retries` | 1 | Retries on a blank or malformed reply |
| `tool_timeout_s` | 10 | Per tool call |
| `route_thinking` | True | Minimal thinking until the last allowed step |
| `system_instruction` | `""` | |
| `on_event` | None | Called with each trace event dict `{t, step, label, ...}` |

## Live demo

A web page that races the plain Gemma tool loop against gemma-harness on the same prompt. Each side streams into its own lane, and each ends with a verdict: silent failure, labelled fallback, or answered.

```
python demo/web/server.py          # live Gemma 4, needs GEMINI_API_KEY
python demo/web/server.py --fake   # scripted fake model, no key: for rehearsing
```

Open http://127.0.0.1:8765.

- **Chaos:** four toggles inject the same faults into both sides: misspelled tool names, wrong args, blank replies, and crashing tools. Each fires once per run, so the demo works even when Gemma 4 behaves.
- **Replay:** every run is saved to `runs/replays/`. If the API is down or out of quota on stage, pick a saved run and press REPLAY.

Chaos mode is also a library feature for testing your own agent:

```python
from gemma_harness.chaos import ChaosClient, chaos_tools
raw = ChaosClient(genai.Client(), {"bad_names", "blank"})
tools = chaos_tools([get_weather], {"crash_tools"})
```

## Stress test

```
python demo/stress.py --both              # raw SDK loop vs harness, 15 prompts, once each
python demo/stress.py --both --runs 3 --delay 4   # PRD's 3 runs; delay for free-tier limits
```

Every harness event is written to `runs/trace-<time>.jsonl`.

### Results

Not run yet against live Gemma 4. Fill this in from a real run, same model and same hour for both columns.

| Metric | Raw SDK loop | gemma-harness |
|---|---|---|
| Silent failures / 15 | | |
| Recovered turns / 15 | n/a | |
| Median latency per prompt (s) | | |
| Thinking tokens | | |
| Total tokens | | |
| False triggers on controls (#11, #15) | n/a | |

## Tests

```
pytest
```

The tests use scripted fake model replies, so they need no API key.

## Limitations

- Fuzzy fixes can pick the wrong tool or argument. The cutoff is 0.6, and every fix is logged.
- Minimal thinking may hurt multi-step accuracy. Turn it off with `route_thinking=False`.
- A timed-out tool keeps running in a background thread until it returns.
- Python only; no streaming.

## License

Apache-2.0
