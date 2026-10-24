"""Baseline: a plain SDK tool loop with NO recovery (the "before" column)."""
from google import genai
from google.genai import types


def raw_run(prompt, tools, model="gemma-4-26b-a4b-it", max_steps=8, stats=None, client=None,
            on_event=None):
    """Returns the final text, "" on blank reply or step cap. Bad names/args raise.

    `stats`, if given, collects token counts so stress.py can compare costs.
    `on_event`, if given, gets a dict per step so the web demo can show the raw lane.
    """
    emit = on_event or (lambda ev: None)
    c = client or genai.Client()
    tm = {f.__name__: f for f in tools}
    decls = [types.FunctionDeclaration.from_callable(client=c, callable=f) for f in tools]
    cfg = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    for step in range(1, max_steps + 1):
        emit({"label": "CALL", "step": step})
        r = c.models.generate_content(model=model, contents=contents, config=cfg)
        if stats is not None and r.usage_metadata:
            um = r.usage_metadata
            stats["think_tok"] = stats.get("think_tok", 0) + (um.thoughts_token_count or 0)
            stats["total_tok"] = stats.get("total_tok", 0) + (um.total_token_count or 0)
        if not r.function_calls:
            text = r.text or ""            # may be "" -> silent failure
            emit({"label": "FINAL", "step": step, "text": text})
            return text
        contents.append(r.candidates[0].content)
        parts = []
        for fc in r.function_calls:
            emit({"label": "TOOL_CALL", "step": step, "name": fc.name, "args": dict(fc.args or {})})
            result = tm[fc.name](**fc.args)   # KeyError / TypeError on bad name/args -> crash
            emit({"label": "TOOL_RESULT", "step": step, "name": fc.name, "result": result})
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))
    emit({"label": "STEP_CAP", "step": max_steps})
    return ""
