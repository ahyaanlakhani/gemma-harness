# gemma-harness

**Your agent either completes the task or tells you exactly where it stopped.**

A small Python harness for Gemma tool loops. It bounds every run, refuses to
execute tool calls it cannot validate, repairs one unusable model turn, and ends
with one of three explicit outcomes — never a blank reply that looks like an
answer.

```python
from gemma_harness import Client
from demo.tools import get_weather, add

outcome = Client(
    model="gemma-4-26b-a4b-it",
    max_steps=8,
    cacheable_tools=("get_weather",),   # read-only tools only
).run("Weather in Karachi, then add 2+3", tools=[get_weather, add])

outcome["status"]        # "success" | "recovered" | "failed"
outcome["text"]          # a final answer, or an explicit failure explanation
outcome["failure_reason"]  # None, or "empty_response" | "max_steps" | ...
outcome["steps"]         # model turns used
outcome["trace"]         # list of {t, step, label, ...}
outcome.usage            # model_calls, tokens, elapsed_ms (missing values are None)
```

### Integration is a loop replacement, not a drop-in import

The PRD's "swap one import" framing is not accurate, and this repo does not
claim it. `gemma_harness.Client` **is** the agent loop: it calls the model,
dispatches tools, and owns the conversation. Code that calls `genai.Client`
directly keeps working untouched, but to get recovery you replace your loop with
`Client.run(...)`. Your existing tool functions are reused as-is.

## Outcomes, not vibes

| Status | Meaning | `text` |
|---|---|---|
| `success` | The model answered without any harness intervention. | the model's answer |
| `recovered` | The harness repaired a bad turn (empty response, unknown tool, rejected args, tool error), then the model answered. | the model's answer |
| `failed` | No answer. `failure_reason` is one of `empty_response`, `max_steps`, `model_call_budget`, `time_budget`, `api_error`, `safety_refusal`. | an explicit "I could not complete this task: …" message |

A fallback is never counted as success: `failed` text never contains tool data
the model did not receive.

## What the harness does

- **Bounded loop:** `max_steps`, `max_model_calls` (default `2 × max_steps`), and
  `time_budget_s` (default 60 s). Hitting any of them is an explicit failure.
- **One repair attempt** (`max_repairs=1`) for an empty response or a
  `MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` / `TOO_MANY_TOOL_CALLS`
  finish reason. The repair is a real user turn that names the reason and the
  valid tools.
- **Exact tool-name validation.** An unknown name is reported back to the model
  with the list of valid names. Nothing is executed and nothing is fuzzy-matched.
- **Strict argument validation before execution** against the function's
  signature and type hints. Unknown keys, missing required args, and wrong types
  reject the whole call; the model gets the reasons and may retry.
- **Only unambiguous conversions.** Numeric strings (`"7"` → `7.0`) and integral
  floats (`7.0` → `7`) are accepted and logged as `ARGS_COERCED`. Strings are
  never turned into booleans.
- **Tool exceptions** become `{"error": "TypeError: ..."}` function responses so
  the model can adapt; the error is never cached as a result.
- **Repeat detection only for declared read-only tools.** Pass
  `cacheable_tools=("get_weather",)`. Identical writes such as `save_note` run
  every time, because two identical calls can be two intended actions.
- **Protocol fidelity:** the model turn is echoed back untouched (thought
  signatures included) and tool-call `id`s are preserved on responses.
- **Safety refusals are not retried.** `SAFETY`, `PROHIBITED_CONTENT`,
  `BLOCKLIST`, `SPII`, `RECITATION` end the run as `failed/safety_refusal`.
- **Transport retries are bounded and owned by the harness.** The adapter creates
  `genai.Client` with `HttpRetryOptions(attempts=1)`, so SDK retries cannot
  multiply with harness retries. If you inject your own client, its retries stay.

Not included on purpose: phantom-tool regex linting, fuzzy tool repair, thinking
routing, structured output, streaming, multi-agent, MCP, and tool timeouts (a
thread timeout does not cancel the underlying operation, so pretending to cancel
would be dishonest — see Limitations).

## Demo A — deterministic fault injection (no API key)

The same scripted model replies and the same local tools are fed to a minimal
baseline loop and to the harness. Replies simulate failures; they are **injected,
not observed Gemma defects**.

```bash
python demo/compare.py                       # table + summary
python demo/compare.py --trace unknown_tool  # one readable trace
python demo/compare.py --jsonl-dir demo/traces
```

```
scenario           injected fault                               baseline                 harness                    tools   check
empty_then_answer  empty reply, then a valid tool call          BLANK                    RECOVERED                  0/1     ok
unknown_tool       hallucinated tool name, then corrected       CRASH KeyError           RECOVERED                  0/1     ok
invalid_args       unparsable number, then corrected            CRASH TypeError          RECOVERED                  1/1     ok
numeric_strings    numbers sent as strings                      completed (WRONG)        SUCCESS                    1/1     ok
tool_exception     tool raises ConnectionError                  CRASH ConnectionError    RECOVERED                  1/1     ok
repeated_read      same read-only call twice                    completed                SUCCESS                    2/1     ok
write_twice        same write call twice                        completed                SUCCESS                    2/2     ok
control_success    none (control)                               completed                SUCCESS                    1/1     ok
failed_recovery    empty reply, repair also empty               BLANK                    FAILED(empty_response)     0/0     ok

SUMMARY (injected faults, n=9)
expected outcome met        3 baseline / 9 harness
  of which after recovery   - / 4
  of which explicit failure - / 1
```

Every recovery fixture ends with text derived from the actual tool result, so a
`ok` check means the task completed — not that a canned string appeared.
`tool_exception` is expected to end in an honest failure report, not an answer.
`numeric_strings` shows the dangerous case: the baseline *completes* while
answering 78 instead of 15.

## Demo B — live Gemma (requires a key)

```bash
export GEMINI_API_KEY=...        # PowerShell: $env:GEMINI_API_KEY="..."
python demo/live.py
python demo/live.py --model gemma-4-31b-it
```

Live results are printed fresh and are deliberately **not** stored in this
README. The demo does not claim any prompt will make Gemma fail; it records what
happened on that run, and keeps live results separate from injected ones.

**Live verification status: incomplete.** This environment had no
`GEMINI_API_KEY`, so the harness has never been run against live Gemma here.
What *is* verified is the full SDK request/response path against a local stub of
the Generative Language API (`tests/test_wire.py`), including tool schemas,
function-call ids, and retry accounting. The model identifier is configurable
(`--model` / `GEMMA_MODEL`); if the configured model is unavailable the SDK error
surfaces as `failed/api_error` — no silent substitution.

## Install and test

Python 3.9+, `google-genai >= 2.33.0`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# .venv/bin/python -m pip install -e ".[dev]"         # macOS/Linux
pytest -q
```

## Trace

`outcome["trace"]` is a list of events; `write_jsonl(outcome["trace"], path)`
exports it. A repaired-run trace reads:

```
[step 1] MODEL_CALL  612ms
[step 1] EMPTY_RESPONSE  finish_reason=MALFORMED_FUNCTION_CALL
[step 1] REPAIR_REQUESTED  reason=MALFORMED_FUNCTION_CALL, attempt=1
[step 2] MODEL_CALL  388ms
[step 2] INVALID_ARGS  tool=add, problems=["argument 'a' must be float, got 'seven'"]
[step 3] MODEL_CALL  401ms
[step 3] TOOL_OK  tool=add
[step 4] MODEL_CALL  355ms
[step 4] FINAL  status=recovered, chars=19
```

(That is the shape of `demo/compare.py --trace invalid_args`, with an injected
first-turn failure prepended; run it yourself for the exact current output.)

Labels: `MODEL_CALL`, `EMPTY_RESPONSE`, `UNUSABLE_RESPONSE`, `REPAIR_REQUESTED`,
`UNKNOWN_TOOL`, `INVALID_ARGS`, `ARGS_COERCED`, `TOOL_OK`, `TOOL_ERROR`,
`CACHED_TOOL_RESULT`, `REFUSAL`, `API_ERROR`, `API_RETRY`, `BUDGET_EXCEEDED`,
`FINAL`, `FAILED`.

## Tests

`31 passed` locally, no API key required:

- `tests/test_recovery.py` (23): success vs recovered, failed repair, safety
  refusal not retried, unknown tool/args never executed, numeric coercion, tool
  exceptions, error-not-cached, read caching, write non-caching, id/thought
  preservation, step/call/time budgets, transport retry bounds, JSONL export.
- `tests/test_scenarios.py` (4): the shipped demo's scenarios, including
  baseline parity and the write-vs-read distinction.
- `tests/test_wire.py` (4): real `genai.Client` against a local stub — request
  serialisation, call-id echo, and that SDK retries are disabled so retry layers
  do not multiply.

## Limitations

- **No live Gemma run yet** in this environment (no key). Everything else is
  deterministic and verified.
- **No tool cancellation.** Tools run to completion; there is no timeout, because
  abandoning a thread does not stop the work it started. A hung tool hangs the
  run until it returns.
- **`recovered` describes the harness, not your goal.** It means "the model
  produced a final answer after a repaired fault". Whether that answer achieves
  the user's task is not automatable; the demo checks it per scenario.
- **No retry of a repair that is malformed twice.** `max_repairs` is a run-wide
  budget (default 1); later unusable turns end in `failed/empty_response`.
- **Validation is signature-based.** Tools with `**kwargs`, exotic annotations,
  or side effects that contradict their types are only as safe as their hints.
- **No token-based budget.** Budgets are steps, model calls, and wall clock. Token
  usage is reported when the provider returns it.
- **Single provider/transport.** Gemini API via `google-genai`; no Vertex adapter
  is exercised here.

## Next improvement

One repair per run is deliberately conservative. The highest-value next step is
**retry budget per failure class** (e.g. two repairs for empty turns, one for
argument rejection, zero for refusals), with each class's success rate measured
on a fixed prompt pack — that turns the harness from "always bounded" into
"bounded and calibrated".
