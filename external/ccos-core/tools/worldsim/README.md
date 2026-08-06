# worldsim — Qwen-AgentWorld → CCOS

Turn **[Qwen-AgentWorld](https://github.com/QwenLM/Qwen-AgentWorld)** — a *language
world model* that simulates agentic environments — into a **synthetic-corpus
generator** for CCOS. `worldsim` loops an agent policy against the world model to
produce a session, writes it as a **CCOS Migration Bundle** (`*.cmb.jsonl`, the
same format [`rag2ccos`](../rag2ccos/README.md) emits), and `ccos-migrate` loads
it into a causal working-memory:

```
  agent policy ⇄ Qwen-AgentWorld  ──worldsim──▶  session.cmb.jsonl  ──ccos-migrate──▶  workspace.ccos
```

## Why

Qwen-AgentWorld predicts the **observation** an environment (Terminal, SWE, OS,
MCP, …) returns for an agent's action. So you can generate **thousands of
realistic agent sessions with no real environment and no real-tool calls**, then
pour them into CCOS to:

- **stress the causal graph / flight recorder / `ccos postmortem`** on volumes of
  sessions you could never collect by hand;
- **warm up RL** against a decoupled, controllable simulator;
- **fuzz the MCP surface** with plausible-but-adversarial tool results.

`worldsim` is the environment side; **CCOS is the memory that records, scopes and
replays it.**

## The key point: the world model simulates the *environment*, not the agent

Each turn `worldsim` sends the world model the domain system prompt + the task +
the **agent's action**, and the model returns the **observation**. The agent
policy (which proposes actions) is yours:

- a **scripted** action list (`--actions actions.txt`, one action per line), or
- a second **LLM** as the agent (`AGENT_MODEL_URL` / `AGENT_MODEL_NAME`).

## Install & configure

Stdlib only — nothing to install. Point it at any OpenAI-compatible endpoint
(vLLM, SGLang, Ollama) serving Qwen-AgentWorld:

```bash
export WORLD_MODEL_URL=http://localhost:8000/v1
export WORLD_MODEL_NAME=Qwen/Qwen-AgentWorld-35B-A3B
# optional: WORLD_MODEL_API_KEY, WORLDSIM_TIMEOUT, AGENT_MODEL_URL, AGENT_MODEL_NAME
```

No endpoint yet? Use `--offline` for deterministic stub observations (clearly
marked `[SIMULATED-STUB]`) so you can exercise the whole pipeline first.

## Usage

```bash
python worldsim.py --domain terminal \
    --task "find the largest file under /var/log" \
    --actions actions.txt --out session.cmb.jsonl --turns 8
ccos-migrate --bundle session.cmb.jsonl --path fleet.ccos --extend
```

| Option | Meaning |
|---|---|
| `--domain terminal\|swe\|os\|mcp\|custom` | which environment the world model simulates (Search is omitted — the model card flags it weakest) |
| `--task` | the session goal (the task node) |
| `--actions F` | file with one agent action per line (the scripted policy) |
| `--turns N` | max turns (default 8) |
| `--system-prompt-file F` | use a prompt verbatim — e.g. a canonical `prompts/<domain>` file from the Qwen-AgentWorld repo (required for `--domain custom`) |
| `--offline` | deterministic stub observations, no endpoint |
| `--session-id`, `--temperature`, `--top-p`, `--max-tokens` | overrides (defaults 0.6 / 0.95 / 1024, per the model card) |

## What it writes

A CMB bundle per session: a `document` node for the **task**, then one `chunk`
per **turn** (the observation), with `parent` = the session and `ordinal` = the
turn index — so CCOS rebuilds **containment** (session → turns) and **sequence**
(turn *i* → turn *i+1*) edges. Each turn's `metadata` carries the `action`,
`domain`, `turn`, and `simulated: true`. The header notes the run is SIMULATED,
and `--offline` is tagged `offline-stub` so it is never mistaken for real output.

## Accumulate a fleet

Because it emits standard CMB, run many sessions and fold them into one memory
with **`ccos-migrate --extend`** (incremental import — keeps the existing graph
and hash-chained logs):

```bash
for task in "$@"; do
  python worldsim.py --domain terminal --task "$task" --actions pol.txt --out /tmp/s.cmb.jsonl
  ccos-migrate --bundle /tmp/s.cmb.jsonl --path fleet.ccos --extend
done
ccos mcp fleet.ccos     # explore the synthetic fleet as a live CCOS memory
```

## Honesty

Everything `worldsim` produces is **simulated** and marked as such (`simulated:
true`, header `note`, `[SIMULATED-STUB]` for offline). Treat it as synthetic data
for stressing, warming up and fuzzing — not as measured real-environment output.
Per the Qwen-AgentWorld model card, prefer the Terminal / SWE / OS / MCP domains
and keep a ≥ 128K context on the served model for coherent long sessions.
