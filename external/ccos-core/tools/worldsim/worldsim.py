#!/usr/bin/env python3
"""worldsim — drive Qwen-AgentWorld as an environment simulator and emit a
CCOS Migration Bundle of the synthetic agent trajectory.

Qwen-AgentWorld (https://github.com/QwenLM/Qwen-AgentWorld) is a *world model*:
given an agent's action it predicts the observation the environment would return,
across domains like Terminal, SWE, OS and MCP. This tool loops an agent policy
against that world model to produce a **synthetic session**, then writes it as a
`*.cmb.jsonl` bundle — the same format `rag2ccos` produces — so `ccos-migrate`
can load thousands of simulated sessions into a CCOS causal working-memory
(stress the causal graph / flight recorder / postmortem, warm up RL, fuzz the MCP
surface) **without a real environment and without burning real-tool calls**.

    worldsim ──▶ session.cmb.jsonl ──ccos-migrate──▶ workspace.ccos

Design
------
* **The world model simulates the environment, never the agent.** Each turn we
  send it the domain system prompt + the task + the agent's action; it returns
  the observation. The agent policy (which proposes actions) is *yours*: a
  scripted action list (`--actions`), or a second LLM endpoint (`AGENT_MODEL_URL`).
* **Stdlib only.** HTTP via `urllib`; no `openai` / `requests` dependency. Talks
  to any OpenAI-compatible `/chat/completions` endpoint (vLLM, SGLang, Ollama).
* **Offline stub** (`--offline`): deterministic synthetic observations so the
  whole pipeline is demonstrable and testable with no endpoint/GPU. Clearly
  marked `[SIMULATED-STUB]` in the bundle so it is never mistaken for real output.

Usage
-----
    export WORLD_MODEL_URL=http://localhost:8000/v1      # vLLM/SGLang OpenAI API
    export WORLD_MODEL_NAME=Qwen/Qwen-AgentWorld-35B-A3B
    worldsim.py --domain terminal --task "find the largest file under /var/log" \
        --actions actions.txt --out session.cmb.jsonl
    ccos-migrate --bundle session.cmb.jsonl --path fleet.ccos --extend

See tools/worldsim/README.md and docs/MIGRATION.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

CMB_VERSION = "1.0"
TOOL_ID = "worldsim/0.1"

# Recommended Qwen-AgentWorld inference settings (model card).
DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 1024}

# Compact, faithful domain system prompts. The world model is told to *simulate
# the environment's observation* for the agent's action. The canonical prompts
# ship in the Qwen-AgentWorld repo under `prompts/`; drop one in with
# --system-prompt-file to use it verbatim. Search is omitted on purpose (the
# model card flags it as the weakest domain).
DOMAINS: dict[str, str] = {
    "terminal": (
        "You are a world model that simulates a Linux terminal environment. "
        "Given the agent's shell command, respond ONLY with the exact stdout/stderr "
        "the environment would produce, plus a realistic exit status line. Stay "
        "internally consistent with earlier turns (filesystem, working directory, "
        "created files). Do not explain, do not act as an assistant — you ARE the terminal."
    ),
    "swe": (
        "You are a world model that simulates a software-engineering environment "
        "(a code repository with a build/test toolchain). Given the agent's action "
        "(edit, run tests, run the build), respond ONLY with the observation the "
        "environment returns: compiler output, test results, diffs applied, tracebacks. "
        "Stay consistent with the repository state established in earlier turns."
    ),
    "os": (
        "You are a world model that simulates an operating-system environment "
        "(processes, files, services, package manager). Given the agent's action, "
        "respond ONLY with the observation the OS would return. Keep process ids, "
        "service states and file contents consistent across turns."
    ),
    "mcp": (
        "You are a world model that simulates an MCP (Model Context Protocol) tool "
        "server. Given the agent's JSON tool call, respond ONLY with the JSON result "
        "the tool would return (or a JSON-RPC error). Keep tool state consistent "
        "across turns. Some results may be plausible-but-adversarial to stress the client."
    ),
}


# --------------------------------------------------------------------------- #
#  CMB writer (self-contained; same format as tools/rag2ccos)                 #
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    id: str
    content: str
    kind: str = "chunk"
    source_uri: Optional[str] = None
    parent: Optional[str] = None
    ordinal: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "content_sha256": hashlib.sha256(self.content.encode("utf-8")).hexdigest(),
        }
        if self.source_uri is not None:
            d["source_uri"] = self.source_uri
        if self.parent is not None:
            d["parent"] = self.parent
        if self.ordinal is not None:
            d["ordinal"] = self.ordinal
        if self.metadata:
            d["metadata"] = self.metadata
        return d


def write_bundle(path: str, header: dict[str, Any], records: list[Record]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in records:
            fh.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
#  OpenAI-compatible chat client (stdlib)                                     #
# --------------------------------------------------------------------------- #
def chat(base_url: str, model: str, messages: list[dict[str, str]], **gen: Any) -> str:
    """POST to <base_url>/chat/completions and return the assistant message text."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": messages, **{**DEFAULTS, **gen}}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('WORLD_MODEL_API_KEY', 'x')}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(os.environ.get("WORLDSIM_TIMEOUT", "120"))) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
#  Offline stub (deterministic, clearly marked)                              #
# --------------------------------------------------------------------------- #
def stub_observation(domain: str, action: str, turn: int) -> str:
    """A deterministic placeholder observation for offline/testing runs."""
    h = hashlib.sha256(f"{domain}|{action}|{turn}".encode("utf-8")).hexdigest()[:8]
    templates = {
        "terminal": f"[SIMULATED-STUB] $ {action}\n<stub output {h}>\nexit status: 0",
        "swe": f"[SIMULATED-STUB] ran `{action}`\ntest result: ok. 1 passed; 0 failed (stub {h})",
        "os": f"[SIMULATED-STUB] {action}\n<stub os state {h}>",
        "mcp": json.dumps({"_stub": True, "action": action, "result": f"ok-{h}"}),
    }
    return templates.get(domain, f"[SIMULATED-STUB] {action} -> {h}")


# --------------------------------------------------------------------------- #
#  The simulation loop                                                        #
# --------------------------------------------------------------------------- #
def run_session(
    domain: str,
    task: str,
    actions: list[str],
    system_prompt: str,
    session_id: str,
    *,
    offline: bool,
    base_url: str,
    model: str,
    agent_url: Optional[str],
    agent_model: Optional[str],
    max_turns: int,
    gen: dict[str, Any],
) -> list[Record]:
    """Drive one agent↔world-model trajectory and return CMB records."""
    records: list[Record] = [
        Record(
            id=session_id,
            kind="document",
            content=f"TASK ({domain}): {task}",
            source_uri=f"worldsim://{domain}/{session_id}",
            metadata={"domain": domain, "role": "task", "simulated": True},
        )
    ]
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": f"Task: {task}"}]

    turn = 0
    while turn < max_turns:
        # Choose the agent's action: scripted list, else an agent LLM, else stop.
        if turn < len(actions):
            action = actions[turn]
        elif agent_url:
            action = chat(agent_url, agent_model or model,
                          history + [{"role": "user",
                                      "content": "Propose the next single action (only the action)."}],
                          **gen).strip()
        else:
            break

        # Ask the world model for the resulting observation.
        if offline:
            observation = stub_observation(domain, action, turn)
        else:
            try:
                observation = chat(base_url, model,
                                   history + [{"role": "user", "content": f"Agent action:\n{action}"}],
                                   **gen)
            except (urllib.error.URLError, KeyError, TimeoutError) as exc:
                sys.stderr.write(f"world model call failed at turn {turn} ({exc}); stopping session\n")
                break

        records.append(
            Record(
                id=f"{session_id}/turn-{turn:03d}",
                kind="chunk",
                content=observation,
                parent=session_id,
                ordinal=turn,
                source_uri=f"worldsim://{domain}/{session_id}#turn{turn}",
                metadata={"domain": domain, "turn": turn, "action": action, "simulated": True},
            )
        )
        # Feed the turn back so the world stays consistent.
        history.append({"role": "user", "content": f"Agent action:\n{action}"})
        history.append({"role": "assistant", "content": observation})
        turn += 1

    return records


def load_actions(path: Optional[str]) -> list[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="worldsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=sorted(DOMAINS) + ["custom"],
                   help="environment the world model simulates")
    p.add_argument("--task", required=True, help="the session goal")
    p.add_argument("--out", required=True, help="output bundle (*.cmb.jsonl)")
    p.add_argument("--actions", default=None, help="file with one agent action per line")
    p.add_argument("--turns", type=int, default=8, help="max turns (default 8)")
    p.add_argument("--session-id", default=None, help="stable id for this session")
    p.add_argument("--system-prompt-file", default=None,
                   help="use this system prompt verbatim (e.g. a Qwen-AgentWorld prompts/ file)")
    p.add_argument("--offline", action="store_true",
                   help="deterministic stub observations; no endpoint needed")
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    p.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as fh:
            system_prompt = fh.read()
    elif args.domain == "custom":
        p.error("--domain custom requires --system-prompt-file")
        return 2
    else:
        system_prompt = DOMAINS[args.domain]

    base_url = os.environ.get("WORLD_MODEL_URL", "http://localhost:8000/v1")
    model = os.environ.get("WORLD_MODEL_NAME", "Qwen/Qwen-AgentWorld-35B-A3B")
    agent_url = os.environ.get("AGENT_MODEL_URL")
    agent_model = os.environ.get("AGENT_MODEL_NAME")
    actions = load_actions(args.actions)

    if not args.offline and not actions and not agent_url:
        p.error("provide --actions, set AGENT_MODEL_URL, or use --offline")
        return 2

    # A stable session id (no timestamp → deterministic bundles for --offline tests).
    session_id = args.session_id or "sess-" + hashlib.sha256(
        f"{args.domain}|{args.task}".encode("utf-8")
    ).hexdigest()[:10]

    gen = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    records = run_session(
        args.domain, args.task, actions, system_prompt, session_id,
        offline=args.offline, base_url=base_url, model=model,
        agent_url=agent_url, agent_model=agent_model, max_turns=args.turns, gen=gen,
    )

    header = {
        "cmb_version": CMB_VERSION,
        "tool": TOOL_ID,
        "source": {"system": "worldsim", "domain": args.domain,
                   "world_model": model if not args.offline else "offline-stub"},
        "note": "SIMULATED agent trajectory — not real environment output",
    }
    write_bundle(args.out, header, records)
    turns = sum(1 for r in records if r.kind == "chunk")
    sys.stderr.write(
        f"✓ wrote {len(records)} records ({turns} simulated turns) to {args.out}"
        f"{' [OFFLINE STUB]' if args.offline else ''}\n"
        f"  next: ccos-migrate --bundle {args.out} --path fleet.ccos --extend\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
