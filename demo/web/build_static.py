"""Build a single-file, server-free copy of the demo page that plays recorded runs.

    python demo/web/build_static.py                         # records fresh fake-model runs
    python demo/web/build_static.py --replays runs/replays  # use your own (e.g. live) recordings

Writes dist/gemma-harness-demo.html. The page answers its own /api/* calls from the
recordings (see static_shim.js), so it can be hosted anywhere static: an Artifact,
GitHub Pages, or a file opened from disk. Every run on it is labelled as recorded.
"""
import argparse
import json
import re
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))

from server import PROMPTS, REPLAY_NAME, make_server  # noqa: E402

STATIC = WEB / "static"
DIST = WEB.parent.parent / "dist"
# Chaos settings recorded for the hosted page: none, each fault alone, and all four.
COMBOS = [[], ["bad_names"], ["bad_args"], ["blank"], ["crash_tools"],
          ["bad_names", "bad_args", "blank", "crash_tools"]]


def record(replay_dir: Path):
    """Run every prompt under every chaos combo against the fake model."""
    srv = make_server(port=0, fake=True, model="fake", replay_dir=replay_dir, fake_delay_s=0.05)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for faults in COMBOS:
            url = f"http://127.0.0.1:{srv.server_address[1]}/api/sweep?faults={','.join(faults)}"
            with urllib.request.urlopen(url, timeout=600) as r:
                r.read()
            print(f"recorded {len(PROMPTS)} prompts with chaos: {faults or 'none'}")
    finally:
        srv.shutdown()
        srv.server_close()


def load(replay_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(replay_dir.iterdir(), key=lambda p: int(p.name.split("-")[0]) if REPLAY_NAME.match(p.name) else 0,
                    reverse=True):
        if not REPLAY_NAME.match(p.name):
            continue
        events = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not events or events[0].get("label") != "START":
            continue
        if any(ev["label"] == "ERROR" for ev in events):   # API trouble isn't worth showing
            continue
        verdict = {ev["side"]: ev for ev in events if ev["label"] == "VERDICT"}
        start = events[0]
        recs.append({"name": p.name, "prompt_id": start["prompt_id"], "faults": start["faults"],
                     "mode": start["mode"], "saved_at": start["t"],
                     "raw": verdict.get("raw", {}).get("verdict"),
                     "harness": verdict.get("harness", {}).get("verdict"),
                     "fixes": verdict.get("harness", {}).get("fixes"), "events": events})
    return recs


def body_of(html: str) -> str:
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    return re.sub(r'\s*<script src="/static/app.js"></script>', "", body)


def script(js: str) -> str:
    return "<script>\n" + js.replace("</script", "<\\/script") + "\n</script>"


def build(recs: list[dict], out: Path):
    modes = {r["mode"] for r in recs}
    data = {"source": modes.pop() if len(modes) == 1 else "mixed",
            "model": "fake" if all(r["mode"] == "fake" for r in recs) else "gemma-4-26b-a4b-it",
            "prompts": list(PROMPTS.values()), "prompt_ids": list(PROMPTS), "recordings": recs}
    page = "\n".join([
        "<title>gemma-harness Live Race</title>",
        "<style>\n" + (STATIC / "style.css").read_text(encoding="utf-8") + "\n</style>",
        body_of((STATIC / "index.html").read_text(encoding="utf-8")),
        "<script>window.__DEMO_DATA__ = " + json.dumps(data, default=str).replace("</", "<\\/") + ";</script>",
        script((WEB / "static_shim.js").read_text(encoding="utf-8")),
        script((STATIC / "app.js").read_text(encoding="utf-8")),
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page) // 1024} KB, {len(recs)} recorded runs, source: {data['source']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", type=Path, help="use these recordings instead of recording fake runs")
    ap.add_argument("--out", type=Path, default=DIST / "gemma-harness-demo.html")
    args = ap.parse_args(argv)
    if args.replays:
        recs = load(args.replays)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            record(Path(tmp))
            recs = load(Path(tmp))
    if not recs:
        sys.exit("no usable recordings found")
    build(recs, args.out)


if __name__ == "__main__":
    main()
