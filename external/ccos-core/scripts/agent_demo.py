#!/usr/bin/env python3
"""Agent-loop demo — CCOS as an agent's external memory.

Scenario: an agent's test fails in ``api.rs``. The bug's real cause lives two
files away (``api`` → ``repo`` → ``db``) and is lexically dissimilar (``db``
talks about *connection pool / timeout*, the failing test about *handle*). A flat
top-k / lexical retriever fetches ``api.rs`` and misses ``db.rs``.

This shows CCOS, used as **external memory**, recall the failing file *and* the
causally-related files a fix must touch — within a token budget, excluding
unrelated code (``util.rs``, ``log.rs``).

Runs fully offline; the value is observable without a model. If ``OLLAMA_ENDPOINT``
is set, the recalled window is additionally sent to the model to propose a fix.

Usage:
    python scripts/agent_demo.py                 # offline
    OLLAMA_ENDPOINT=http://localhost:11434 OLLAMA_MODEL=qwen2.5:7b-instruct \\
        python scripts/agent_demo.py             # + a real model
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# A tiny project: a causal chain db -> repo -> api, plus unrelated files.
WORKSPACE = {
    "src/db.rs": "// low-level connection pool\n"
    "pub fn timeout_ms() -> i64 { 30 } // BUG: far too short under load\n",
    "src/repo.rs": "use crate::db;\npub fn fetch() -> i64 { db::timeout_ms() * 2 }\n",
    "src/api.rs": "use crate::repo;\npub fn handle() -> i64 { repo::fetch() + 1 }\n",
    "src/util.rs": "pub fn format_date() -> String { String::new() }\n",
    "src/log.rs": "pub fn info(msg: &str) { let _ = msg; }\n",
}
FAILING_FILE = "file:src/api.rs"


def find_ccos() -> str:
    repo = Path(__file__).resolve().parents[1]
    rel = repo / "target" / "release" / "ccos"
    return str(rel) if rel.exists() else "ccos"


def mem(ccos: str, path: str, reqs: list[dict]) -> list[dict]:
    """Drive `ccos memory` with JSON-Lines requests; return parsed responses."""
    inp = "\n".join(json.dumps(r) for r in reqs)
    out = subprocess.run(
        [ccos, "memory", "--path", path],
        input=inp,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0 and not out.stdout.strip():
        sys.exit(f"ccos memory failed: {out.stderr.strip()}")
    return [json.loads(l) for l in out.stdout.splitlines() if l.strip()]


def files_only(window: dict) -> list[tuple[float, str]]:
    return [
        (round(i["score"], 3), i["uri"])
        for i in window["items"]
        if i["uri"].startswith("file:")
    ]


def ask_ollama(prompt: str) -> str | None:
    endpoint = os.environ.get("OLLAMA_ENDPOINT")
    if not endpoint:
        return None
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        f"{endpoint}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]
    except Exception as e:  # noqa: BLE001 - demo: any failure → skip the LLM step
        print(f"  (LLM step skipped: {e})")
        return None


def rule(title: str) -> None:
    print(f"\n\033[1m── {title} ─────────────────────────────────────\033[0m")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="CCOS external-memory agent-loop demo.")
    ap.add_argument("--ccos-bin", default=find_ccos())
    ap.add_argument("--budget", type=int, default=4000)
    args = ap.parse_args(argv)

    path = tempfile.mktemp(suffix=".ccos")
    ccos = args.ccos_bin

    rule("1. The agent ingests its workspace into CCOS memory")
    ingest = [{"op": "ingest", "uri": u, "source": s} for u, s in WORKSPACE.items()]
    for u, rep in zip(WORKSPACE, mem(ccos, path, ingest)):
        print(f"   ingested {u:<16} (+{rep['nodes_added']} nodes, +{rep['edges_added']} edges)")

    rule("2. A test fails in api.rs — but the cause is 2 files away")
    print("   The bug is db::timeout_ms()==30; it surfaces as a failing test on")
    print("   api::handle(). 'handle' shares no words with 'timeout'/'pool', so a")
    print("   lexical/top-k retriever would fetch api.rs and miss db.rs.")

    rule("3. CCOS recall(Around api.rs) — the causal region")
    win = mem(ccos, path, [
        {"op": "recall", "strategy": "around", "anchor": FAILING_FILE, "budget": args.budget}
    ])[0]
    got = files_only(win)
    for score, uri in got:
        print(f"   {score:>5}  {uri}")
    names = {u for _, u in got}
    ok = "file:src/db.rs" in names and "file:src/util.rs" not in names
    print(f"\n   → cause db.rs recalled: {'file:src/db.rs' in names}; "
          f"unrelated util.rs excluded: {'file:src/util.rs' not in names}  "
          f"[{'PASS' if ok else 'CHECK'}]")

    rule("4. The agent signals the failure; pressure flows along the chain")
    aff = mem(ccos, path, [{"op": "failure", "node": FAILING_FILE, "depth": 3}])[0]
    ws = mem(ccos, path, [
        {"op": "recall", "strategy": "working_set", "budget": args.budget}
    ])[0]
    print(f"   {aff['affected']} nodes affected. Working set by causal score (files):")
    for score, uri in files_only(ws):
        bar = "█" * int(score * 24)
        print(f"   {score:>5}  {uri:<20} {bar}")
    print("\n   → the causal chain (api/repo/db) outranks the noise (util/log).")

    rule("5. The context window the agent would send to an LLM")
    prompt = (
        "You are fixing a failing test on api::handle(). Here is the causally-"
        "relevant context CCOS recalled:\n\n"
        + "\n".join(f"// === {i['uri']} ===\n{i['content']}" for i in win["items"]
                    if i["uri"].startswith("file:"))
        + "\n\nName the file and function holding the root cause, and the one-line fix."
    )
    print(f"   ({len(prompt)} chars, ~{len(prompt)//4} tokens — bounded by budget)")
    answer = ask_ollama(prompt)
    if answer:
        rule("6. LLM proposal from the recalled window")
        print("   " + answer.strip().replace("\n", "\n   "))
    else:
        rule("6. (no LLM configured — set OLLAMA_ENDPOINT to run the model step)")
        print("   The window above is what the agent feeds the model. Offline, the")
        print("   point stands: CCOS put the root cause (db.rs) in front of the agent.")

    os.path.exists(path) and os.remove(path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
