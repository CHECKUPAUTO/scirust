# CCOS Core — the repository's causal memory

CCOS Core (Causal Context Operating System) is installed in this repository as
the **agent's causal memory**: it maps the side effects of a coding session
(files read/edited, `cargo test`/`build` failures, panics) into a causal
graph, pages that graph under a token budget, and logs every transition into
a **deterministic, bit-replayable, hash-chained** log.

- Vendored source: [`external/ccos-core/`](../external/ccos-core/) —
  a copy of the `Memorithm/CCOS-Core` repository, upstream commit `d3f499325874d848dce07e007610d55437e2c4c0`.
- Upstream references: `external/ccos-core/README.md`,
  `external/ccos-core/docs/SELF_ANALYSIS.md` (agent dogfooding),
  `external/ccos-core/docs/USAGE.md`, `external/ccos-core/docs/MEMORY_INTERFACE.md`.

## Installation

```bash
sh scripts/ccos/install.sh
# = cargo build --release --features llm,license (in external/ccos-core)
#   + install the `ccos` binary (/usr/local/bin, otherwise ~/.cargo/bin or ~/.local/bin)
#   + `ccos doctor`
```

The toolchain is pinned by `external/ccos-core/rust-toolchain.toml` (1.89.0).
The default build is **community tier** (no Pro crypto, fail-closed) —
this is the expected behavior as long as no vendor key is configured.

`scripts/ccos/ensure_built.sh` is an idempotent guard designed for a
`SessionStart` hook: no-op if the binary exists, otherwise launches the build
in the background (log in `.ccos/build.log`) without ever blocking.

## Runtime state

| Path | Role | Git |
|---|---|---|
| `.ccos/workspace.ccos` | memory snapshot (graph + hash-chained log) | ignored |
| `.ccos/workspace.ccos.oplog` | cognitive timeline (time-travel) | ignored |
| `external/ccos-core/target/` | build artifacts | ignored |

Typical bootstrap of a fresh workspace (ingesting the repository core):

```bash
python3 - <<'EOF' | ccos memory --path .ccos/workspace.ccos > /dev/null
import json, os
for root in ("src", "scirust-core/src"):
    for dirpath, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f.endswith(".rs"):
                p = os.path.join(dirpath, f)
                src = open(p, encoding="utf-8", errors="replace").read()
                print(json.dumps({"op": "ingest", "uri": p, "source": src}))
EOF
```

## Two feed modes — one writer per workspace

> Upstream rule (`docs/SELF_ANALYSIS.md`): the persistent MCP server and the
> self-feeding hook must **never** write the same `workspace.ccos` at the
> same time. Choose one mode.

### Mode A — MCP server (active by default here)

[`.mcp.json`](../.mcp.json) declares the server (consent-gated: the host asks
for approval when the session opens):

```json
{ "mcpServers": { "ccos": { "command": "ccos", "args": ["mcp", ".ccos/workspace.ccos"] } } }
```

The agent gets the native tools `ingest`, `recall`, `signal_failure`,
`page_fault`, `stats`, `verify`, `timeline`, `recall_what_if`, plus the
`ccos://session/context` resource (the self-bounded working window, ready to
inject).

### Mode B — transparent self-feeding (PostToolUse hook)

`scripts/ccos/self_feed_hook.sh` intercepts every agent side effect
(source file read/edit → `ingest`; cargo failure → `page_fault`)
and feeds the memory **at zero cognitive cost to the agent**. It never blocks
(always exit 0, no-op if the binary is missing).

To enable it, add it yourself in `.claude/settings.json` (an agent cannot
wire its own hooks — this action is reserved for the human):

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [ { "type": "command",
          "command": "sh \"$CLAUDE_PROJECT_DIR\"/scripts/ccos/ensure_built.sh" } ] }
    ],
    "PostToolUse": [
      { "matcher": "Read|Edit|Write|NotebookEdit|Bash",
        "hooks": [ { "type": "command", "async": true,
          "command": "sh \"$CLAUDE_PROJECT_DIR\"/scripts/ccos/self_feed_hook.sh" } ] }
    ]
  }
}
```

**If you enable Mode B, disable Mode A** (remove/deny the `ccos` server from
`.mcp.json`) to respect the "one writer" rule.
Both scripts are pipe-tested: a synthetic `Read` event does produce an
`Ingest(...)` visible in `ccos postmortem`.

## Reading the memory, debugging drift

```bash
# causal recall around a file (self-bounded window)
printf '%s\n' '{"op":"recall","strategy":"around","anchor":"file:scirust-core/src/lib.rs","budget":2048}' \
  | ccos memory --path .ccos/workspace.ccos

# integrity + stats
printf '%s\n' '{"op":"verify"}' '{"op":"stats"}' | ccos memory --path .ccos/workspace.ccos

# post-mortem time-travel (timeline, diff, energy, missing <node>)
ccos postmortem .ccos/workspace.ccos

# analytical session archive
ccos postmortem .ccos/workspace.ccos --json > archive_$(date +%F).json
```

Drift protocol (upstream, §SELF_ANALYSIS): `timeline` → `missing <cause>` →
`energy A B` → `goto K` + `recall` — to date precisely the eviction of the
real cause outside the budgeted window.
