"""gemma-harness: a Gemma 4 tool-calling loop that never fails silently.

Every failure becomes a labelled event in the trace. See the PRD's failure
taxonomy for the full list of labels.
"""
import difflib
import hashlib
import inspect
import json
import random
import re
import threading
import time
import typing
from typing import Any, Callable

import httpx
from google import genai
from google.genai import errors, types

FAIL_REASONS = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL",
                "TOO_MANY_TOOL_CALLS", "OTHER", "FINISH_REASON_UNSPECIFIED"}
# Verbs that make a snake_case word look like a tool name in a prompt.
TOOL_VERBS = {"get", "search", "send", "fetch", "lookup", "create", "delete", "web",
              "book", "list", "update", "read", "write", "post", "run"}
# A snake_case word counts as a tool mention only in a tool-ish context:
# "use/call/calling/using/with/invoke [the] [`]name", or "name(" / "name tool".
TOOL_MENTION = re.compile(
    r"\b(?i:use|call|calling|using|with|invoke)\s+(?:(?i:the)\s+)?`?([a-z]+_[a-z_]+)\b"
    r"|\b([a-z]+_[a-z_]+)(?=\(|\s+tool)")
NAME_CUTOFF = 0.6
BACKOFF_CODES = (408, 429, 500, 502, 503, 504)
MAX_API_TRIES = 5
MAX_BACKOFF_S = 60.0
HTTP_TIMEOUT_MS = 120_000


class Client:
    """Drop-in-ish wrapper: Client(...).run(prompt, tools=[fn, ...]) -> dict"""

    def __init__(self, model="gemma-4-26b-a4b-it", api_key=None, max_steps=8,
                 max_retries=1, tool_timeout_s=10.0, route_thinking=True,
                 system_instruction="", on_event: Callable[[dict], None] | None = None,
                 raw=None):
        # `raw` lets tests inject a fake client; normally reads GEMINI_API_KEY.
        # The timeout turns a hung connection into an error we can back off from.
        self.raw = raw if raw is not None else genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS))
        self.model, self.max_steps, self.max_retries = model, max_steps, max_retries
        self.tool_timeout_s, self.route_thinking = tool_timeout_s, route_thinking
        self.system_instruction = system_instruction
        self.trace: list[dict] = []
        self.on_event = on_event

    # ---------- trace ----------
    def emit(self, step: int, label: str, **data):
        ev = {"t": round(time.time(), 3), "step": step, "label": label, **data}
        self.trace.append(ev)
        if self.on_event:
            self.on_event(ev)

    # ---------- one model call with 429 / 5xx / network backoff ----------
    def _generate(self, contents, decls, thinking: str, step: int, extra_sys: str = ""):
        cfg = types.GenerateContentConfig(
            system_instruction=(self.system_instruction + extra_sys) or None,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=thinking),
        )
        for attempt in range(MAX_API_TRIES):
            last = attempt == MAX_API_TRIES - 1
            try:
                t0 = time.time()
                r = self.raw.models.generate_content(model=self.model, contents=contents, config=cfg)
                um = r.usage_metadata
                self.emit(step, "CALL", ms=int((time.time() - t0) * 1000), thinking=thinking,
                          in_tok=getattr(um, "prompt_token_count", None),
                          out_tok=getattr(um, "candidates_token_count", None),
                          think_tok=getattr(um, "thoughts_token_count", None))
                return r
            except errors.APIError as e:
                if e.code not in BACKOFF_CODES or last:
                    raise
                wait = _backoff_s(attempt, _retry_delay_s(e))
                self.emit(step, "BACKOFF_429" if e.code == 429 else "BACKOFF_5XX",
                          code=e.code, wait_s=round(wait, 1))
            except httpx.TransportError as e:   # connect / read timeout / dropped connection
                if last:
                    raise
                wait = _backoff_s(attempt)
                self.emit(step, "BACKOFF_5XX", reason=type(e).__name__, wait_s=round(wait, 1))
            time.sleep(wait)

    def _thinking_for(self, step: int) -> str:
        if not self.route_thinking:
            return "high"
        return "minimal" if step < self.max_steps else "high"

    # ---------- main loop ----------
    def run(self, prompt: str, tools: list[Callable] = ()) -> dict:
        self.trace = []
        tool_map = {f.__name__: f for f in tools}
        decls = [types.FunctionDeclaration.from_callable(client=self.raw, callable=f) for f in tools]
        extra = lint_prompt(self.system_instruction + "\n" + prompt, tool_map, self)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        seen: dict[str, Any] = {}      # call key -> result, reused by DEDUPED
        failed: set[str] = set()       # call keys that failed once; cached if they fail again
        for step in range(1, self.max_steps + 1):
            r = self._generate(contents, decls, self._thinking_for(step), step, extra)
            r, _ = recover_blank(self, r, contents, decls, step, extra)
            calls = r.function_calls or []
            if not calls:
                text = response_text(r)
                if not text:
                    self.emit(step, "SILENT_FAIL_PREVENTED")
                    text = fallback_text(tool_map)
                elif finish_reason(r) == "MAX_TOKENS":
                    self.emit(step, "TRUNCATED", chars=len(text))
                self.emit(step, "FINAL", chars=len(text))
                return {"text": text, "trace": self.trace, "steps": step}
            contents.append(r.candidates[0].content)
            parts = [self._handle_call(fc, tool_map, seen, step, failed) for fc in calls]
            contents.append(types.Content(role="user", parts=parts))
        self.emit(self.max_steps, "STEP_CAP")
        return {"text": fallback_text(tool_map), "trace": self.trace, "steps": self.max_steps}

    def _handle_call(self, fc, tool_map: dict, seen: dict, step: int, failed: set) -> types.Part:
        """Repair one function call, run it, and return the function-response part.

        Repairs are written back onto `fc`, which is the same object as the model turn
        already in the history, so the call and its response always agree.
        """
        name = fix_tool_name(self, fc.name, tool_map, step)
        if name is None:
            fc.name = fc.name or "unknown"
            return _response_part(fc, {"error": f"unknown tool. available: {list(tool_map)}"})
        fc.name = name
        fn = tool_map[name]
        args = fix_args(self, fn, dict(fc.args or {}), step)
        missing = missing_args(fn, args)
        if missing:
            self.emit(step, "ARGS_MISSING", tool=name, missing=missing)
            return _response_part(fc, {
                "error": f"missing required arguments: {missing}. Call {name} again with them."})
        key = hashlib.md5(f"{name}{json.dumps(args, sort_keys=True, default=str)}".encode()).hexdigest()
        if key in seen:
            self.emit(step, "DEDUPED", tool=name)
            return _response_part(fc, {
                "result": seen[key],
                "note": "ALREADY CALLED with same args. Use this result; call a different tool or answer."})
        result, ok = _run_tool(self, fn, args, step)
        if ok or key in failed:   # a one-off failure may be transient: let the model retry it once
            seen[key] = result
        else:
            failed.add(key)
        return _response_part(fc, {"result": result})


# ---------- helpers ----------
def response_text(r) -> str:
    """Text parts only. Avoids the SDK's warning when a reply also holds tool calls."""
    cand = r.candidates[0] if r.candidates else None
    parts = (cand.content.parts if cand and cand.content else None) or []
    return "".join(p.text for p in parts if p.text and not p.thought).strip()


def finish_reason(r) -> str:
    cand = r.candidates[0] if r.candidates else None
    return str(getattr(cand, "finish_reason", "") or "").split(".")[-1]


def fallback_text(tool_map: dict) -> str:
    return ("Sorry, I couldn't finish this with my tools "
            f"(available: {', '.join(tool_map) or 'none'}). Here is what I know without them.")


def _response_part(fc, response: dict) -> types.Part:
    """A function response that names and ids the call it answers."""
    return types.Part(function_response=types.FunctionResponse(
        id=fc.id, name=fc.name, response=response))


def _backoff_s(attempt: int, server_delay: float | None = None) -> float:
    """Exponential backoff with jitter; a server-supplied delay wins, capped at 60 s."""
    wait = min(30, 2 ** attempt) + random.random()
    if server_delay is not None:
        wait = max(wait, server_delay)
    return min(wait, MAX_BACKOFF_S)


def _retry_delay_s(e: errors.APIError) -> float | None:
    """The google.rpc.RetryInfo delay ("17s") in an API error's details, if any."""
    body = e.details if isinstance(e.details, dict) else {}
    err = body.get("error", body)
    for d in (err.get("details") if isinstance(err, dict) else None) or []:
        if isinstance(d, dict) and str(d.get("@type", "")).endswith("google.rpc.RetryInfo"):
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    return None


def _closest_tool(name: str, tools) -> str | None:
    """The declared tool `name` is most plausibly a typo of, or None.

    Both snake_case: the verb (text before the first "_") must be identical and the
    rest similar, so delete_note never becomes save_note. Otherwise compare whole names.
    """
    best, best_score = None, 0.0
    for t in tools:
        if "_" in name and "_" in t:
            (verb, rest), (t_verb, t_rest) = name.split("_", 1), t.split("_", 1)
            if verb != t_verb:
                continue
            score = difflib.SequenceMatcher(None, rest, t_rest).ratio()
        else:
            score = difflib.SequenceMatcher(None, name, t).ratio()
        if score >= NAME_CUTOFF and score > best_score:
            best, best_score = t, score
    return best


# ---------- recovery functions ----------
def lint_prompt(text: str, tool_map: dict, h: Client) -> str:
    """Phantom tool: the prompt asks for a tool-like name that is not declared.

    Only names in a tool context count ("use web_search", "call `send_email`",
    "search_docs(...)", "the send_sms tool"), so snake_case prose is left alone.
    Near-misses of a declared tool (get_weather_now vs get_weather) are left for
    fix_tool_name, so the trace shows TOOL_NAME_FIXED instead of a lint.
    """
    mentioned = {a or b for a, b in TOOL_MENTION.findall(text)}
    phantom = sorted(
        m for m in mentioned
        if m not in tool_map and m.split("_")[0] in TOOL_VERBS
        and not _closest_tool(m, tool_map))
    if phantom:
        h.emit(0, "PHANTOM_TOOL_LINTED", names=phantom)
        return (f"\nTools NOT available this turn: {phantom}. Only call: {list(tool_map)}."
                " If none fits, answer from your own knowledge and say so.")
    return ""


def _usable(r) -> bool:
    return bool(r.function_calls or response_text(r))


def recover_blank(h: Client, r, contents, decls, step, extra):
    """Retry a blank or failed reply. If no retry does better, keep the original when
    it had anything usable (e.g. text that came back with finish_reason OTHER)."""
    original = r
    fr = finish_reason(r)
    if fr in FAIL_REASONS or not _usable(r):
        for i in range(h.max_retries):
            h.emit(step, "RETRIED", reason=fr or "EMPTY", attempt=i + 1)
            hint = (f"\nYour previous reply failed ({fr or 'empty'}). Call exactly one of "
                    f"{[d.name for d in decls]} with valid JSON args, or answer in plain text.")
            r = h._generate(contents, decls, "minimal", step, extra + hint)
            fr = finish_reason(r)
            if _usable(r):
                return r, True
        if _usable(original):
            return original, False
    return r, False


def fix_tool_name(h: Client, name: str, tool_map: dict, step: int):
    if name in tool_map:
        return name
    m = _closest_tool(name or "", tool_map)
    h.emit(step, "TOOL_NAME_FIXED" if m else "UNKNOWN_TOOL", got=name, used=m)
    return m


def _coerce(t, v):
    """Coerce v to type t when the model sent the wrong JSON type. Raises ValueError."""
    if t is bool and isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    if t in (int, float) and isinstance(v, str):
        return t(v.strip())
    if t is int and isinstance(v, float) and v.is_integer():
        return int(v)
    return v


_VAR_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _named_params(sig: inspect.Signature) -> dict:
    """Parameters a keyword argument can fill by name (not *args / **kwargs)."""
    return {p: s for p, s in sig.parameters.items() if s.kind not in _VAR_KINDS}


def fix_args(h: Client, fn: Callable, args: dict, step: int) -> dict:
    sig, hints = inspect.signature(fn), typing.get_type_hints(fn)
    named = _named_params(sig)
    takes_kwargs = any(s.kind is inspect.Parameter.VAR_KEYWORD for s in sig.parameters.values())
    unknown = [k for k in args if k not in named]
    required_free = [p for p, s in named.items() if s.default is s.empty and p not in args]
    out, fixed = {}, []
    for k, v in args.items():
        if k not in named:
            m = difflib.get_close_matches(k, list(named), n=1, cutoff=0.6)
            if m and m[0] not in args and m[0] not in out:   # never overwrite a set argument
                target = m[0]
            elif takes_kwargs:
                out[k] = v                  # **kwargs takes any key as is
                continue
            elif not m and len(unknown) == 1 and len(required_free) == 1:
                target = required_free[0]   # one stray key, one gap: they must match
            else:
                fixed.append(f"drop:{k}")
                continue
            fixed.append(f"{k}->{target}")
            k = target
        try:
            nv = _coerce(hints.get(k), v)
            if nv is not v:
                fixed.append(f"coerce:{k}")
            v = nv
        except ValueError:
            pass   # leave it; the tool reports the error and the model can adapt
        out[k] = v
    if fixed:
        h.emit(step, "ARGS_COERCED", fixes=fixed)
    return out


def missing_args(fn: Callable, args: dict) -> list[str]:
    named = _named_params(inspect.signature(fn))
    return [p for p, s in named.items() if s.default is s.empty and p not in args]


def _jsonable(result):
    """What the next request can serialize: dates, sets, Decimals, ... become str."""
    try:
        return json.loads(json.dumps(result, default=str))
    except (TypeError, ValueError):   # e.g. non-string dict keys or a reference cycle
        return str(result)


def _run_tool(h: Client, fn: Callable, args: dict, step: int) -> tuple[Any, bool]:
    """Run a tool with a timeout; returns (result, succeeded).

    A daemon thread, so a hung tool can't block exit.
    """
    box: dict[str, Any] = {}

    def target():
        try:
            box["ok"] = fn(**args)
        except BaseException as e:   # incl. SystemExit: reported to the model, not raised
            box["err"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(h.tool_timeout_s)
    if th.is_alive():
        h.emit(step, "TOOL_TIMEOUT", tool=fn.__name__)
        return {"error": f"{fn.__name__} timed out after {h.tool_timeout_s}s"}, False
    if "err" in box:
        e = box["err"]
        h.emit(step, "TOOL_ERROR", tool=fn.__name__, err=repr(e)[:200])
        return {"error": f"{type(e).__name__}: {e}"}, False
    h.emit(step, "TOOL_OK", tool=fn.__name__)
    return _jsonable(box["ok"]), True


def call_tool(h: Client, fn: Callable, args: dict, step: int):
    """Run a tool with a timeout. Errors and timeouts come back as {"error": ...}."""
    return _run_tool(h, fn, args, step)[0]
