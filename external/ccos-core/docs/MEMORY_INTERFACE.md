# CCOS external memory interface

A single, documented façade for using CCOS as an agent's **external working
memory**: write code and failure signals in, recall a bounded, causally-coherent
context window out, and keep an auditable, hash-chained state on disk.

It is the in-process Rust surface (`ccos::external_memory`). Two transports ship on
top of it and call exactly this API — a stdio JSON-Lines CLI (`ccos memory`) and a
**Model Context Protocol server** (`ccos mcp`) — but the façade is the contract.

- [Why](#why)
- [Quick start](#quick-start)
- [The contract](#the-contract)
- [Recall strategies](#recall-strategies)
- [Node identity](#node-identity)
- [Persistence & integrity](#persistence--integrity)
- [A typical agent loop](#a-typical-agent-loop)
- [Driving it from any language (`ccos memory`)](#driving-it-from-any-language-ccos-memory)
- [Serving over MCP (`ccos mcp`)](#serving-over-mcp-ccos-mcp)
- [Guarantees](#guarantees)
- [Limitations](#limitations)

## Why

An LLM agent's context window is a scarce, bounded resource. CCOS manages it like
an OS manages memory: source is parsed into a **causal graph**, nodes are scored
(importance · failure-pressure · recency · use), and a bounded window is paged in.
The external-memory façade exposes that machinery as a handful of verbs an agent
actually needs — `ingest`, `signal_failure`, `recall`, `verify`, `checkpoint` —
instead of the kernel's a-dozen internal types.

## Quick start

```rust
use ccos::external_memory::{CcosMemory, ExternalMemory, Recall};

// Open (or create) a persistent workspace memory.
let mut mem = CcosMemory::open("workspace.ccos")?;

// Write the workspace in.
mem.ingest_source("src/db.rs", &std::fs::read_to_string("src/db.rs")?);
mem.ingest_source("src/api.rs", &std::fs::read_to_string("src/api.rs")?);

// The agent's task failed at a test that exercises db.rs — tell the memory.
let affected = mem.signal_failure("file:src/db.rs", 3)?;   // propagate 3 hops

// Recall a bounded, causally-coherent window for the model (≤ 2048 tokens).
let window = mem.recall(&Recall::around("file:src/db.rs"), 2048);
for item in &window.items {
    println!("{:.3}  {:<28}  ({} chars)", item.score, item.uri, item.content.len());
}

// Persist; the hash chain stays verifiable.
mem.checkpoint()?;
assert!(mem.verify().valid);
# Ok::<(), ccos::external_memory::MemoryError>(())
```

Add the dependency (path or git) and use the crate as `ccos`.

## The contract

`ExternalMemory` is the stable trait; `CcosMemory` is the implementation.

| Operation | Signature | Meaning |
| --------- | --------- | ------- |
| `ingest_source` | `(&mut self, uri, source) -> IngestReport` | parse a file, fold the delta into the graph, extend the hash chain. Re-ingesting identical text is a no-op delta. |
| `signal_failure` | `(&mut self, node, depth) -> Result<usize, _>` | mark `node` failing (severity `0.95`) and propagate downstream up to `depth` hops; returns affected count. |
| `recall` | `(&self, &Recall, budget_tokens) -> RecallWindow` | select a bounded window under a strategy (below). |
| `verify` | `(&self) -> Integrity` | check both hash chains are intact. |
| `stats` | `(&self) -> MemoryStats` | counts (nodes / edges / events / files / clock). |
| `checkpoint` | `(&self) -> Result<(), _>` | persist the whole state to the bound path. |

Inherent helpers on `CcosMemory` (not in the trait):

| Method | Meaning |
| ------ | ------- |
| `open(path)` / `new()` | load-or-create / empty in-memory |
| `checkpoint_to(path)` | persist to an explicit path and bind it |
| `impact(node, depth)` | downstream **blast radius** (`Vec<Reached>`) |
| `causes(node, depth)` | upstream **causes** (`Vec<Reached>`) |
| `tick()` | advance the logical clock (recency decay) |
| `graph()` | read-only access to the raw `MemoryGraph` (escape hatch) |

### Returned types

- `IngestReport { uri, nodes_added, nodes_removed, edges_added }`
- `RecallWindow { strategy, items: Vec<RecallItem>, tokens }`
- `RecallItem { uri, score, kind, content }` — `content` is the file's ingested
  source when known, else the node's own content.
- `Integrity { valid, events, errors }`
- `MemoryStats { nodes, edges, events, files, clock }`
- `MemoryError` — `NodeNotFound(id)`, `Io`, `Serde`, `NoPath` (implements
  `std::error::Error`).

All result types derive `Serialize`, so a server/CLI layer can return them as JSON
verbatim.

## Recall strategies

```rust
Recall::working_set()        // hottest nodes globally by causal score
Recall::around("file:src/db.rs")  // the causal region anchored on a node
Recall::task("fix db timeout")    // lexical entry point → its region
```

- **`WorkingSet`** — the globally hottest nodes. Use when there is no specific
  anchor (a fresh session, a "what matters now?" query).
- **`Around(uri)`** — the causal **region** containing the anchor (the active
  file, the failing test). This is the workspace-anchored recall and the one to
  prefer for a focused task: selection is driven by *structure*, not by matching
  the query text, so it is robust when the task description is vague or
  misleading. If the anchor belongs to no region, its k-hop causal neighbourhood
  (causes + impact) is used instead.
- **`Task(text)`** — when all you have is free text: a simple lexical entry point
  (token overlap on node labels/content) is picked, then expanded to its region.
  Weaker than `Around` (it trusts the query); prefer `Around` whenever you have a
  real anchor.

Every strategy is bounded by `budget_tokens` (≈ 4 chars/token) and is
deterministic: ties break on the node id.

## Node identity

Node ids are namespaced strings:

| Prefix | Example | Meaning |
| ------ | ------- | ------- |
| `file:` | `file:src/db.rs` | a source file |
| `mod:` | `mod:src/db.rs:tests` | a module |
| `sym:` | `sym:src/db.rs:query` | a symbol (fn/const/struct/…) |
| `use:` | `use:src/db.rs:std::io` | an import |
| `dep:` | `dep:serde` | an external dependency |

`ingest_source("src/db.rs", …)` creates `file:src/db.rs` (and the module/symbol
nodes under it). `signal_failure` and `recall` accept either a full node id or a
**bare path** — `file:` is assumed when no known prefix is present, so
`signal_failure("src/db.rs", 3)` and `signal_failure("file:src/db.rs", 3)` are
equivalent.

## Persistence & integrity

`open(path)` loads an existing checkpoint or starts empty with `path` bound;
`checkpoint()` writes the whole state (graph + both logs + retained sources) as
one JSON file. Every `ingest_source` appends to a canonical **SHA-256 hash chain**
over the event's replayable content, so:

```rust
let report = mem.verify();          // Integrity { valid, events, errors }
assert!(report.valid);              // tampering with the checkpoint is detectable
```

A checkpoint **round-trips**: reloading reproduces the graph and the chain
(covered by `checkpoint_roundtrips_through_a_file`).

## A typical agent loop

```text
on session start:        for each workspace file → ingest_source(uri, src)
on a tool/test failure:  signal_failure(failing_file, depth=3)
before each model call:  window = recall(Around(active_file), budget)
                         feed window.items (uri + content) to the model
on edit:                 ingest_source(uri, new_src)   // O(Δ) re-index
periodically:            checkpoint(); assert verify().valid
```

`recall(Around(active_file), budget)` is the key step: it returns the failing
file *and the causally-related files the fix is likely to touch*, within the token
budget — the thing a flat top-k retriever misses.

A runnable version of this loop is [`scripts/agent_demo.py`](../scripts/agent_demo.py):
it ingests a small workspace whose bug's cause is two files away and lexically
dissimilar, recalls the causal region, signals the failure, and (if
`OLLAMA_ENDPOINT` is set) asks a local model to propose a fix. It runs offline —
the value is observable without a model.

## Driving it from any language (`ccos memory`)

The same façade is exposed as a **stdio JSON-Lines** command, so any language can
use CCOS as memory via a subprocess — no server to run:

```bash
printf '%s\n' \
  '{"op":"ingest","uri":"src/db.rs","source":"pub fn query() {}"}' \
  '{"op":"failure","node":"file:src/db.rs","depth":3}' \
  '{"op":"recall","strategy":"around","anchor":"file:src/db.rs","budget":2048}' \
  '{"op":"verify"}' \
  | ccos memory --path workspace.ccos
```

One request object per line in, one JSON response per line out. The workspace is
loaded from `--path` (default `workspace.ccos`) and checkpointed back if any
request mutated it. Operations: `ingest` (`uri`, `source`), `failure` (`node`,
`depth`), `recall` (`strategy` ∈ `around` / `task` / `working_set`, plus
`anchor` / `text`, `budget`), `impact` / `causes` (`node`, `depth`), `verify`,
`stats`. Responses are the `Serialize` types above verbatim; errors are
`{"error":"…"}`.

From Python:

```python
import subprocess, json

def mem(reqs, path="workspace.ccos"):
    inp = "\n".join(json.dumps(r) for r in reqs)
    out = subprocess.run(["ccos", "memory", "--path", path],
                         input=inp, capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines()]

mem([{"op": "ingest", "uri": "src/db.rs", "source": open("src/db.rs").read()}])
win = mem([{"op": "recall", "strategy": "around",
            "anchor": "file:src/db.rs", "budget": 2048}])[0]
for it in win["items"]:
    print(it["score"], it["uri"])
```

## Serving over MCP (`ccos mcp`)

`ccos mcp` exposes the same façade as a [Model Context Protocol](https://modelcontextprotocol.io)
server over **stdio JSON-RPC 2.0**, so any MCP-compatible agent (Claude, a local
agent on the Jetson, …) can use CCOS as native working memory — no HTTP server, no
new dependency (it is `serde_json` only). The session is event-sourced, so the
whole interaction stays replayable.

It speaks the standard handshake (`initialize` → `notifications/initialized` →
`tools/list` / `tools/call` / `resources/list` / `resources/read`, plus `ping`) and
advertises **sixteen core tools**:

| Tool | Arguments | Maps to |
| ---- | --------- | ------- |
| `ingest` | `uri`, `source` | `ingest_source` |
| `recall` | `strategy` ∈ `around`/`task`/`semantic`/`hybrid`/`working_set`/`causal-flash` (+ the Pro `octa-semantic` in `octasoma` builds), `anchor`/`text`, `budget` — plus the OpenClaw aliases `query`/`limit`/`minScore`/`sessionKey` | `recall` |
| `signal_failure` | `node`, `depth` | `signal_failure` |
| `page_fault` | `output` (cargo-test/panic text), `budget` | parse trace → signal faulting files → recall |
| `stats` | — | `stats` |
| `verify` | — | `verify` |
| `timeline` | — | the event-sourced cognitive journal (`AgentSession::timeline`) |
| `recall_what_if` | `step`, `strategy`/`anchor`/`text`, `budget` | **time-travel** — rewind to `step` and re-run a recall (`AgentSession::recall_what_if`) |
| `ccos_retrieve` | `ccr_ref` | the original (uncompressed) content of a compressed recall item |
| `causal_intervene` | `node`, `magnitude`/`damping`/`depth` | do(X): the nodes a change transitively forces |
| `causal_blame` | `node`, `damping`/`depth` | candidate root causes — what a node depends on |
| `drift_cause` | `node` | change-point attribution: which recorded op moved a node's score |
| `retrodict_belief` | `claim`, `stride`/`half_life`/`q`/`r` | the RTS-smoothed belief trajectory |
| `causal_flash` | `horizon`/`decay`/`include_callers`/… | a bounded causal-cone context window over the active frontier |
| `get` | `path`, `from`/`lines` | read an ingested source file by path (the OpenClaw `memory_get` surface) |
| `sync` | `force` | boot/refresh ack — checkpoint the session, report the timeline step |

`timeline` and `recall_what_if` expose the capability a RAG stack structurally lacks:
the ordered record of every memory operation, and a deterministic *replay* of the
agent's memory to a past step under different parameters (a larger budget, a
different anchor) — debugging an agent's context by rewinding it. The analysis tools
are read-only, so they never trigger a checkpoint.

**Premium namespaces (CCOS_EXTENDED).** In fused builds the same server also
multiplexes three feature-gated namespaces, each Pro-gated at runtime by the
offline license (visible `isError` refusal on the community tier — never a silent
downgrade; the `*.explain` tools are free prose):

| Namespace | Cargo feature | Tools |
| --------- | ------------- | ----- |
| `slha.*` | `slhav2-full` | `explain`, `audit` (kernel self-audit), `compress` (codec-selectable: `int4`/`grouped`/`nf4`/`mixed`/`tq3`), `score`, `benchmark` |
| `octa.*` | `octacore` | `explain`, `cascade_recall` (causal-narrow → exact-cosine rerank over the live session, read-only) |
| `rsi.*` | `rsi` | `explain`, `status` (compiled capabilities + license gates) — **no execution tools by design**: DGM self-modification is unreachable over MCP |

An `octasoma` build additionally advertises the `octa_feedback` tool (the
relevance-feedback channel calibrating the `octa-semantic` conformal anchor gate).

Each `tools/call` returns the corresponding `Serialize` type (above) as JSON inside
the MCP `content[0].text` field. Tool-level failures come back as
`{"content":[…],"isError":true}`; protocol errors use standard JSON-RPC codes
(`-32601` unknown method, `-32602` bad params, `-32700` parse error).

### Resources — auto-injectable context

Two read-only **resources** let a host pull memory in without an explicit tool call:

| Resource URI | Contents |
| ------------ | -------- |
| `ccos://session/context` | the current causally-scored, token-bounded **working set**, linearised as text for direct injection into a system prompt — reflects accumulated failure pressure and recency, and self-bounds at the causal region (no `K` to tune). Budget via `CCOS_MCP_CONTEXT_BUDGET` (default 2048). |
| `ccos://session/timeline` | the cognitive journal as text (audit / replay). |

A host can read `ccos://session/context` before each turn and drop it straight into
the model's context — the "self-bounding ~hundreds-of-tokens" window the
measurements point to, kept current as the agent ingests code and hits failures.

### Persistence

`ccos mcp` takes an **optional workspace path** — `ccos mcp [workspace.ccos]`, or the
`CCOS_MCP_WORKSPACE` env var. With one bound, the session **reloads** that checkpoint
on start and **re-checkpoints after every memory-changing call** (`ingest`,
`signal_failure`, `page_fault`) and once more at EOF, so the causal memory survives
restarts. The on-disk form is the *same snapshot* `ccos memory` reads and writes, so
the two transports can share one `workspace.ccos`. With no path, the session stays
purely in-process (nothing is written). Checkpoint failures are reported on stderr;
stdout is reserved for JSON-RPC.

The **cognitive timeline persists too**, in a sidecar next to the snapshot
(`<workspace>.oplog`): it stores the op-log plus the baseline it replays on, so
`timeline` and `recall_what_if` (time-travel) span the **whole recorded history
across restarts** — you can rewind to a step that happened in a previous process. The
op-log is trusted only if it reproduces the loaded snapshot exactly; if the snapshot
was changed out-of-band (e.g. by a `ccos memory` run that doesn't touch the sidecar),
the snapshot wins and the timeline resets from it — the memory is never corrupted by
a stale log.

To keep a long-running daemon bounded, the op-log **compacts**: once it grows past
`CCOS_OPLOG_MAX` operations (default 512) the oldest are folded into the baseline,
keeping the most recent `CCOS_OPLOG_KEEP` (default 128) individually replayable
(`CCOS_OPLOG_MAX=0` disables it). Compaction is index-stable — logical step numbers
never shift, so `recall_what_if(step=…)` keeps referring to the same moment — and
lossless for the *memory* (the live state is untouched); it only trades away the
ability to rewind *below the floor* (steps older than the retained tail collapse to
the baseline). So time-travel depth is bounded, memory and replay-to-now are not.

Point an MCP client's **stdio transport** at the binary. For example, a client
config entry:

```json
{
  "mcpServers": {
    "ccos-memory": {
      "command": "ccos",
      "args": ["mcp", "workspace.ccos"]
    }
  }
}
```

Drop the `"workspace.ccos"` argument for an ephemeral in-memory session.

Or drive it directly over a pipe (one JSON message per line in, one per line out):

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"src/db.rs","source":"pub fn query() {}"}}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall","arguments":{"strategy":"around","anchor":"file:src/db.rs","budget":2048}}}' \
  | ccos mcp workspace.ccos
```

## Guarantees

- **Deterministic recall** — a total order on `(score, uri)`; same memory, same
  budget ⇒ same window.
- **Tamper-evidence** — `verify()` covers the whole history via the hash chain.
- **Round-trip** — checkpoint then reload reproduces the graph and the chain.
- **`edges ⊆ nodes × nodes`** — the kernel never holds dangling edges.

## Limitations

- Retained source is held per ingested file (so `recall` can return content); for
  very large workspaces this is memory the kernel graph itself does not keep.
- `Around`/`Task` recompute region clustering per call (correctness over speed);
  fine for interactive use, not for tight inner loops on huge graphs.
- `Task` uses a deliberately simple lexical entry point (no embeddings) — it is a
  convenience, not a semantic retriever; prefer `Around` with a real anchor.
- Two transports ship on top of this façade: a **stdio JSON-Lines CLI**
  (`ccos memory`) and an **MCP server** (`ccos mcp`, an event-sourced session with
  tools + resources). Both checkpoint to the same `workspace.ccos` snapshot format; a
  network/HTTP server is not included, but would layer on the same trait.
- An MCP workspace persists in two files: `workspace.ccos` (the memory snapshot,
  shared with `ccos memory`) and `workspace.ccos.oplog` (the timeline sidecar, so
  time-travel spans restarts). The op-log compacts to stay bounded in operation count
  (`CCOS_OPLOG_MAX`/`CCOS_OPLOG_KEEP`); its baseline is a workspace-sized snapshot, so
  the sidecar is ~the snapshot's size, not the history's. The whole sidecar is still
  rewritten on each checkpoint (not an incremental append) — fine at interactive
  cadence, not for a very high-frequency write loop.
- Compaction discards the *details* of folded ops to bound storage, so you cannot
  rewind below the floor (older than the retained tail). The memory and replay-to-now
  are unaffected; only deep historical time-travel is traded away.
