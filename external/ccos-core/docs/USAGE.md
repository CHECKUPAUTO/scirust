# CCOS Usage Guide

A practical, example-driven reference for the `ccos` CLI. For the design
rationale see [`PAPER.md`](PAPER.md); for the code map see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

- [Quick start](#quick-start)
- [Concepts in 60 seconds](#concepts-in-60-seconds)
- [Command reference](#command-reference)
  - [`analyze`](#analyze) · [`top`](#top) · [`blame`](#blame) ·
    [`export`](#export) · [`diff`](#diff) · [`failure`](#failure) ·
    [`verify` / `replay`](#verify--replay) · [`chaos`](#chaos) ·
    [`demo`](#demo)
  - [v0.3 runtime](#v03-runtime-scan--agents--benchmark--runtime)
- [End-to-end walkthrough](#end-to-end-walkthrough)
- [Node id scheme](#node-id-scheme)
- [Exit codes](#exit-codes)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Quick start

```bash
# Build (a recent stable toolchain is all you need)
cargo build --release            # binary at target/release/ccos

# Analyze CCOS's own source and save a snapshot
cargo run -- analyze src --cycles --out run.json

# Inspect it
cargo run -- top src --limit 15                       # hottest nodes
cargo run -- blame run.json file:src/memory.rs        # causes + blast radius
cargo run -- export run.json --out graph.graphml      # GraphML for Gephi/yEd
```

Throughout this guide, `cargo run --` is interchangeable with the built binary
`./target/release/ccos`.

## Concepts in 60 seconds

CCOS parses Rust source into a **causal memory graph**. Every file, module,
`use`, symbol and external dependency becomes a **node**; containment and
dependency become directed **edges** (`source → target`). Each node gets a
**causal score** blending importance, failure-relevance, recency and access
frequency. A bounded **context window** is paged in/out of that graph like
RAM ↔ VRAM, and every state transition is recorded in an append-only,
hash-chained **event log** that replays deterministically.

A **snapshot** (`analyze --out run.json`) bundles the graph + both logs into one
JSON file that `verify`, `replay`, `diff`, `failure`, `blame` and `export`
consume.

---

## Command reference

### `analyze`

```
ccos analyze <path> [--json] [--cycles] [--dot FILE] [--out FILE]
                    [--max-nodes N] [--budget N]
```

Ingest every `.rs` file under `<path>` and print a structural report.

| Flag | Meaning | Default |
| ---- | ------- | ------- |
| `--json` | Emit JSON instead of the human report | off |
| `--cycles` | Detect and list dependency cycles | off |
| `--dot FILE` | Also write the graph as Graphviz DOT | — |
| `--out FILE` | Save a full kernel snapshot (graph + logs) | — |
| `--max-nodes N` | Paging cap (max nodes held in memory) | 5000 |
| `--budget N` | Context-window token budget | 2048 |

```bash
cargo run -- analyze src --cycles            # report + dependency cycles
cargo run -- analyze src --json | jq .nodes  # machine-readable
cargo run -- analyze src --dot ccos.dot      # render: dot -Tsvg ccos.dot -o ccos.svg
cargo run -- analyze src --out run.json       # snapshot for the other commands
```

Exit code is `1` if any dangling edge survives (an invariant violation) or the
path has no `.rs` files.

### `top`

```
ccos top <path> [--limit N] [--json] [--max-nodes N]
```

Like Unix `top`, but for context: the highest causal-score nodes — the working
set the kernel would page in first.

```bash
cargo run -- top src --limit 15
cargo run -- top src --json
```

```
  562 nodes / 653 edges — top 5:

    SCORE   TYPE      NODE
   0.5483  Symbol    dep:crate
   0.5398  Symbol    dep:ccos
   0.5379  Symbol    dep:std
   ...
```

### `blame`

```
ccos blame <snapshot.json> <node-id> [--depth N] [--json]
```

Trace a node's causal neighbourhood in both directions:

- **Causes** (upstream, `target → source`): what the node rests on.
- **Blast radius** (downstream, `source → target`): what breaks if it fails —
  the same direction `failure` propagates.

`--depth N` bounds the walk (default 3). Get node ids from
`analyze <path> --json` or the [node id scheme](#node-id-scheme).

```bash
cargo run -- analyze src --out run.json
cargo run -- blame run.json file:src/memory.rs --depth 4
cargo run -- blame run.json sym:src/guard.rs:GuardLayer --json
```

```
  ── Causes (upstream — what it rests on): 0 ──

  ── Blast radius (downstream — what breaks with it): 56 ──
    d1  0.4097  sym:src/memory.rs:MemoryGraph
    d1  0.3750  mod:src/memory.rs:tests
    ...
```

(A `file:` node is a graph root, so it has no upstream causes.)

### `export`

```
ccos export <snapshot.json> [--out FILE] [--format graphml]
```

Export the snapshot's causal graph as **GraphML** — an XML interchange format
read by Gephi, yEd, Cytoscape and networkx. Output is deterministic
(id-sorted), so it diffs cleanly. Default output is `ccos.graphml`.

```bash
cargo run -- export run.json --out graph.graphml
python3 -c "import networkx as nx; print(nx.read_graphml('graph.graphml'))"
```

Node data carries `label`, `type` and `score`; edge data carries `weight` and
`type`. (GraphML is currently the only format; `--format` is accepted for
forward compatibility.)

### `diff`

```
ccos diff <old.json> <new.json>
```

Structural difference between two snapshots: nodes/edges added & removed, plus
the biggest causal-score movers among common nodes.

```bash
cargo run -- analyze src   --out a.json
cargo run -- analyze tests --out b.json
cargo run -- diff a.json b.json
```

### `failure`

```
ccos failure <snapshot.json> <node-id> [--depth N]
```

Inject a fault (severity 0.95) at a node and propagate it across causal edges,
reporting the affected neighbourhood ranked by resulting failure-relevance.
Whereas `blame` is read-only graph traversal, `failure` runs the kernel's
weighted propagation model.

```bash
cargo run -- failure run.json file:src/memory.rs --depth 2
```

### `verify` / `replay`

```
ccos verify <snapshot.json>      # hash chain valid? dangling edges? → exit 0/1
ccos replay <snapshot.json>      # deterministic event-log replay + stats
```

`verify` re-checks integrity (hash chain + `edges ⊆ nodes × nodes`). `replay`
re-runs the event log deterministically, then rebuilds the graph purely from the
log and confirms it matches the snapshot (event-sourcing round-trip).

```bash
cargo run -- analyze src --out run.json
cargo run -- verify run.json && cargo run -- replay run.json
```

### `chaos`

```
ccos chaos [--iters N]
```

Drive adversarial payloads (JSON corruption, hallucination, prompt injection,
timeouts) through the guard and assert it **never** emits invalid JSON. Exit `1`
if the safety invariant is ever violated.

```bash
cargo run -- chaos --iters 5000
```

### `demo`

```
ccos demo            # also the default when no command is given
```

A scripted single cycle over a small synthetic workspace touching every
subsystem. The LLM call targets an [Ollama](https://ollama.com)-style endpoint
and falls back to a deterministic stub when none is reachable:

```bash
OLLAMA_ENDPOINT=http://localhost:11434 OLLAMA_MODEL=codellama cargo run -- demo
```

### v0.3 runtime (`scan` / `agents` / `benchmark` / `runtime`)

```
ccos scan <path>                          # async FS scan → causal graph delta
ccos agents <path>                        # Coder/Reviewer/Security over the code
ccos benchmark [--cycles N] [--cap N] [--out FILE]   # → benchmark_report.json
ccos runtime <path> [--state DIR] [--budget N]       # scan→schedule→agents→persist
```

```bash
cargo run -- scan src
cargo run -- agents src
cargo run -- benchmark --cycles 100000 --out benchmark_report.json
cargo run -- runtime src --state data --budget 2048
```

See [`../CCOS_v0.3_REPORT.md`](../CCOS_v0.3_REPORT.md) for the full v0.3 report.

### `memory` — external-memory façade (stdio JSON)

Use CCOS as an agent's external working memory from any language: one JSON
request per line on stdin, one JSON response per line on stdout. The workspace is
loaded from `--path` (default `workspace.ccos`) and checkpointed back on mutation.

```bash
printf '%s\n' \
  '{"op":"ingest","uri":"src/db.rs","source":"pub fn query() {}"}' \
  '{"op":"failure","node":"file:src/db.rs","depth":3}' \
  '{"op":"recall","strategy":"around","anchor":"file:src/db.rs","budget":2048}' \
  '{"op":"verify"}' \
  | ccos memory --path workspace.ccos
```

Ops: `ingest`, `failure`, `recall` (`strategy` ∈ `around`/`task`/`working_set`),
`impact`, `causes`, `verify`, `stats`. Full contract and a Python example:
[`MEMORY_INTERFACE.md`](MEMORY_INTERFACE.md).

### `mcp [workspace.ccos]` — serve memory as MCP tools + resources (stdio JSON-RPC)

Expose the same memory to any **MCP-compatible agent** (Claude, a local agent on
the Jetson, …) over stdio JSON-RPC 2.0 — no HTTP server, no extra dependency. Pass a
**workspace path** (or set `CCOS_MCP_WORKSPACE`) to reload it on start and checkpoint
after every memory-changing call, so the memory survives restarts; omit it for an
ephemeral in-process session. The on-disk form is shared with `ccos memory`.

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"src/db.rs","source":"pub fn query() {}"}}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall","arguments":{"strategy":"around","anchor":"file:src/db.rs","budget":2048}}}' \
  | ccos mcp workspace.ccos
```

Tools: `ingest`, `recall`, `signal_failure`, `page_fault` (feed cargo-test/panic
output back in), `stats`, `verify`, plus the time-travel pair `timeline` /
`recall_what_if` (rewind to a past step and re-run a recall — the timeline persists
in a `<workspace>.oplog` sidecar, so this spans restarts). Resources:
`ccos://session/context` (the self-bounding working set, ready to inject into a
system prompt) and `ccos://session/timeline`. Point an MCP client's stdio transport
at `ccos mcp` (`{"command":"ccos","args":["mcp","workspace.ccos"]}`). Full handshake,
tool schemas and a client-config snippet:
[`MEMORY_INTERFACE.md`](MEMORY_INTERFACE.md#serving-over-mcp-ccos-mcp).

---

### `postmortem [workspace.ccos]` — interactive time-travel debugger

The "GDB for the agent's memory": walk a recorded cognitive timeline by hand and
watch the working set drift as failures propagate. With a workspace path it loads the
persisted op-log (`<workspace>.oplog` from `ccos mcp`, even after a crashed run); with
none it walks a built-in session that drifts.

```bash
printf '%s\n' 'timeline' 'goto 4' 'recall' 'goto 9' 'recall' 'diff 4 9' 'energy 4 9' 'quit' \
  | cargo run -- postmortem
```

REPL commands: `timeline`/`tl` (journal, `▸` marks the cursor), `goto N` / `next` /
`prev` (move the time-travel cursor), `recall [budget]` / `around <anchor> [budget]` /
`task <text…>` (the window as of the cursor), `stats`, and two drift views:

- `diff A B` — which **files** entered/left the working set between two steps.
- `energy A B` — node-level **Δscore + failure-pressure**: the migration of causal
  "heat" through the AST as failures propagate (e.g. the deep cause `db.rs` surging
  `fail 0.00→0.95` after a page-fault, even when the file set itself is unchanged).
- `missing <node> [budget]` — an **eviction watchpoint**: scan the timeline for the
  first step a node is in the graph but has dropped out of the budgeted window, with
  the triggering op and how many tokens short it fell. A status strip shows it at a
  glance — `·●●●●●○○●●` reads as "the cause was in context until a failure on a
  neighbour squeezed it out (`○○`), then a page-fault pulled it back". It composes
  with compaction: an eviction in folded history is reported against the floor rather
  than guessed.

Every command reconstructs state deterministically from the op-log via
[`recall_what_if`](MEMORY_INTERFACE.md) / replay — exact and side-effect free.

---

## End-to-end walkthrough

Analyze a real project, inspect it, and prove the session is reproducible:

```bash
# 1. Ingest a codebase into a snapshot.
cargo run -- analyze /path/to/project/src --cycles --out run.json

# 2. What's hot, and what's structurally central?
cargo run -- top /path/to/project/src --limit 20

# 3. Pick a node and study its causal neighbourhood.
cargo run -- analyze /path/to/project/src --json | jq -r '.top_nodes[].id' | head
cargo run -- blame run.json <node-id-from-above> --depth 4

# 4. Model a fault and watch it ripple.
cargo run -- failure run.json <node-id> --depth 3

# 5. Prove reproducibility: integrity + deterministic replay.
cargo run -- verify run.json
cargo run -- replay run.json

# 6. Track drift over time and export for visualization.
cargo run -- analyze /path/to/project/src --out run2.json   # after some edits
cargo run -- diff run.json run2.json
cargo run -- export run2.json --out graph.graphml
```

## Node id scheme

Node ids are namespaced strings (see `ARCHITECTURE.md`):

| Prefix | Example | Meaning |
| ------ | ------- | ------- |
| `file:<path>` | `file:src/memory.rs` | a source file |
| `mod:<path>:<name>` | `mod:src/lib.rs:parser` | a module declaration |
| `use:<path>:<full-path>` | `use:src/main.rs:std::path::Path` | an import |
| `sym:<path>:<name>` | `sym:src/guard.rs:GuardLayer` | a function/struct/enum/trait/… |
| `dep:<root>` | `dep:serde` | an external dependency root |

`blame` and `failure` take any of these. List the live ids for a tree with
`ccos analyze <path> --json` (the `top_nodes` array) or `ccos top <path> --json`.

## Exit codes

| Code | Meaning |
| ---- | ------- |
| `0` | success |
| `1` | runtime error (bad path, load failure, **invariant violation** such as a dangling edge, or failed `verify`/`chaos`) |
| `2` | usage error (missing required argument) |

These make CCOS scriptable in CI — e.g. `ccos verify run.json && deploy`.

## Troubleshooting & FAQ

**`no .rs files found under '<path>'`** — `analyze`/`top`/`scan` only read Rust
sources, skipping `target/`, `.git/` and hidden directories. Point them at a
directory that actually contains `.rs` files.

**`node '<id>' not found` from `blame`/`failure`** — the id isn't in the graph.
Copy an exact id from `ccos analyze <path> --json` (`top_nodes[].id`) or
`ccos top <path> --json`, and remember ids are path-relative to how you invoked
`analyze` (e.g. `file:src/memory.rs`, not an absolute path).

**The LLM call "fails" but `demo` still finishes** — that's by design. With no
reachable Ollama endpoint the `llm` client returns a deterministic fallback and
the guard substitutes a safe, valid-JSON response, so offline runs are fully
reproducible. Set `OLLAMA_ENDPOINT` / `OLLAMA_MODEL` to use a real model.

**`verify` reports failure** — either the hash chain doesn't validate (the
snapshot was edited/corrupted) or a dangling edge survived. Re-`analyze` to
regenerate the snapshot; if it persists on a fresh snapshot, it's a bug worth an
issue.

**A function inside a comment shows up (or a multi-line item is missed)** — the
parser is line-based (no `syn`). It strips `//` and inline `/* … */` comments,
but **multi-line** block comments and multi-line declarations aren't tracked.
This is the top roadmap item (see [`../ROADMAP.md`](../ROADMAP.md)).

**Is anything non-deterministic?** — No, by construction: node scoring, eviction
and replay are totally ordered, and `top`/`blame`/`export` sort their output, so
identical inputs yield byte-identical results. (The `adversarial` fuzzer and
live LLM calls are the only randomized/external paths, and neither feeds the
replayable log.)

**Where are the heavy stress tests?** — In [`../scripts/`](../scripts/)
(multi-day chaos, 100k-cycle stress, replay-consistency, memory-pressure). The
in-tree suite (`cargo test`) already covers the invariants; add `-- --ignored`
for the 1,000,000-cycle long-stability run.
