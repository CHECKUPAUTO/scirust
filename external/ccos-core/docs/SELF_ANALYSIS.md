# Self-analysis — dogfooding CCOS as an agent's causal memory

CCOS is a **cognitive MMU**: it manages an agent's working context the way a CPU's
MMU manages memory. This page wires CCOS into a coding agent (Claude Code) so the
agent's own runs feed a causal memory you can then **debug post-mortem** — find the
exact moment an agent drifted, when the real cause was evicted from its context.

The whole loop reuses what already ships: `ccos mcp` (the persistent MCP server) and
`ccos postmortem` (the time-travel debugger). Build once:

```bash
cargo build --release         # -> ./target/release/ccos
```

## The "hardware intercept" — why feeding must be transparent

In a CPU, the program doesn't politely call the MMU to announce a cache miss — the
**hardware** traps the faulting instruction. For CCOS to be an invisible MMU rather
than a library the agent must remember to call, the feeding has to be automatic:
every file the agent reads and every test it runs should leave a trace in the causal
graph *without spending any of the agent's thinking budget*.

There are two ways to plug in. Pick **one writer per workspace** (the live server and
the hook should not write the same `workspace.ccos` at once; if they collide the
consistency guard self-heals to the snapshot, but you lose timeline fidelity).

### Mode A — tool-driven (`.mcp.json`)

The repo ships a project-scoped [`.mcp.json`](../.mcp.json) that registers `ccos mcp`
as an MCP server — and `ccos setup` writes the same entry for you on any other
project (idempotent, consent-gated; see [`SETUP.md`](SETUP.md)). Open the repo in
Claude Code, approve the server, and the agent
gains CCOS as native tools (`ingest`, `recall`, `signal_failure`, `page_fault`,
`timeline`, `recall_what_if`, …) plus the `ccos://session/context` resource. The
agent can *query* its memory live — but it has to call the tools itself, so this is
not the transparent intercept.

### Mode B — automatic feed (the hook) — recommended for dogfooding

[`scripts/ccos_self_feed.py`](../scripts/ccos_self_feed.py) is a **PostToolUse hook**:
after every tool the agent runs, it intercepts the side effect and feeds it to CCOS.

```
[Agent host]
   ├─► runs a tool (e.g. `cargo test`, or reads src/db.rs)
   └─► PostToolUse trigger  ──►  ccos_self_feed.py  (the glue)
            ├── source file read/written ─►  ccos `ingest`
            └── cargo test/build errored ─►  ccos `page_fault`
```

Wire it into your Claude Code settings (`.claude/settings.json` in the project, or
your user settings):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Read|Edit|Write|Bash",
        "hooks": [
          { "type": "command", "command": "python3 scripts/ccos_self_feed.py" }
        ]
      }
    ]
  }
}
```

Each event opens a short-lived `ccos mcp workspace.ccos` session that applies one
operation and checkpoints, so `workspace.ccos` + `workspace.ccos.oplog` accumulate a
complete, replayable record of what the agent touched and where it failed — with zero
prompts to the model. Env knobs: `CCOS_BIN`, `CCOS_WORKSPACE` (and the op-log bounds
`CCOS_OPLOG_MAX` / `CCOS_OPLOG_KEEP`).

## The post-mortem protocol

When a run goes wrong — the agent chases a symptom, edits the wrong file, loops —
reconstruct what its memory looked like and find where attention diverged:

```bash
ccos postmortem workspace.ccos      # reads the accumulated .oplog
```

A repeatable protocol for analysing a drift:

1. **`timeline`** — read the cognitive history; locate roughly where it went wrong.
2. **`missing <cause-node> [budget]`** — name the file/symbol that *should* have
   stayed in context (the real root cause). The eviction watchpoint reports the exact
   step it dropped out of the budgeted window, the op that triggered it, and the
   token gap — e.g. `·●●●●●○○●●`: in context, then squeezed out when a failure made a
   neighbour hot, then pulled back by a page-fault.
3. **`energy A B`** — across that step, see the node-level score + failure-pressure
   migration (the causal "heat" moving through the AST), which the file-level `diff`
   can miss when the file set is stable.
4. **`goto K` + `recall [budget]`** — look at the exact window the agent had at the
   eviction step, and `recall_what_if`-style compare it against a wider budget to
   confirm the cause *would* have been in reach.

That is the scientific loop: a deterministic, replayable account of how the agent's
working memory evolved, so a drift is reproducible and explainable rather than a
vibe.

## Collecting field data

In production a workspace is just two portable JSON files: `workspace.ccos` (the
causal-memory snapshot + hash-chained log) and `workspace.ccos.oplog` (the cognitive
timeline). Because the timeline replays bit-for-bit, the field record is
**reproducible off-site** — copy the files to a workstation and `ccos postmortem`
them to get exactly what happened on the device, time-travel included.

**Archive / extract a session** without the REPL:

```bash
ccos postmortem workspace.ccos --json > archive/$(date +%F_%H%M).json
```

`--json` dumps an analytics-ready record — `stats`, `integrity` (the hash-chain
verdict), `timeline`, `compaction_floor`, and the current `working_set` — and exits.
Run it on a cron/systemd-timer to archive history *before* the next compaction folds
older steps into the baseline; the raw `.oplog` stays the structured source of truth.

**Collect from a fleet** (local-first — `rsync` + `ccos`, no central server):

```bash
scripts/fleet_collect.sh --out ./fleet \
  user@node1:~/agent/workspace.ccos  user@node2:~/agent/workspace.ccos
# → ./fleet/<host>/{workspace.ccos, workspace.ccos.oplog, session.json}
```

It rsyncs each node's workspace to a hub and writes a `session.json` per node;
integrity is checked offline, so a truncated/tampered transfer surfaces as
`integrity.valid = false` in the record.

Two honest operational notes:

- **No built-in telemetry.** CCOS is local-first and zero-network by design; fleet
  aggregation is the thin `rsync`-and-extract layer above, not a daemon phoning home.
- **Compaction bounds per-step history.** The memory state and replay-to-now are
  always complete, but rewind below the compaction floor is folded into the baseline.
  For full unbounded history, raise `CCOS_OPLOG_MAX` / `CCOS_OPLOG_KEEP`, or archive
  with the `--json` export above on a schedule.

## Honest caveats

- **Feeding is best-effort.** The hook only sees tools its matcher catches and only
  page-faults on recognised error markers; it is a heuristic intercept, not a
  ground-truth tracer.
- **One writer per workspace.** Run Mode A *or* Mode B against a given
  `workspace.ccos`, not both at once.
- **Bounded history.** The op-log compacts (`CCOS_OPLOG_MAX`/`CCOS_OPLOG_KEEP`); a
  drift older than the retained tail is reported against the compaction floor, not
  reconstructed step-by-step. Memory state and replay-to-now are always exact.
