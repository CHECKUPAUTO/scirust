# CCOS Architecture & Developer Guide

A practical map of the codebase for contributors. For the conceptual write-up see
[`PAPER.md`](PAPER.md); for the roadmap see [`../ROADMAP.md`](../ROADMAP.md).

## Module map

| Module                      | Responsibility | Key types |
| --------------------------- | -------------- | --------- |
| `parser`                    | Modules / `use` / symbols from Rust source — real `syn` AST by default (line-heuristic fallback via `--no-default-features`) | `ASTParser`, `ParseResult`, `Symbol`, `SymbolKind` |
| `memory`                    | The causal graph: scoring, paging, failure propagation, analytics | `MemoryGraph`, `GraphNode`, `GraphEdge`, `NodeId`, `NodeType`, `EdgeType` |
| `incremental`               | `O(Δ)` graph maintenance on file edits | `IncrementalGraphEngine`, `DeltaMutation`, `MutationOp`, `FileState` |
| `event_log`                 | Append-only event log with a canonical tamper-evident hash chain, deterministic replay & graph reconstruction | `EventLog`, `TraceEvent`, `EventType`, `EventPayload`, `LogIntegrity`, `EventReplayer`, `GraphReconstructor` |
| `distributed_event_log`     | Tamper-evident hash-chained log | `DistributedEventLog`, `HashChainLink`, `IntegrityReport` |
| `llm`                       | Async Ollama-style client + retries + fallback | `LlmClient`, `LlmConfig`, `ValidatedResponse` |
| `guard`                     | Validate/sanitize model output | `GuardLayer`, `GuardConfig`, `GuardResult` |
| `consensus`                 | Majority / confidence-weighted voting | `ConsensusEngine`, `LlmVote`, `ConsensusResult` |
| `adversarial`               | Fault injection for hardening | `AdversarialEngine`, `AdversarialMode` |
| `persist`                   | Save/load a kernel snapshot (one file) | `KernelSnapshot` |
| `query`                     | Read-only causal queries: impact/cause walks, hot set, GraphML export | `Reached`, `Direction`, `impact_set`, `source_set`, `hot_set`, `to_graphml` |
| **`external_memory`**       | Documented façade: agent-facing external working memory (ingest / recall / signal-failure / checkpoint) over the kernel. See [`MEMORY_INTERFACE.md`](MEMORY_INTERFACE.md) | `ExternalMemory`, `CcosMemory`, `Recall`, `RecallWindow`, `MemoryError` |
| **`agent_session`**         | Event-sourced cognitive timeline: record ops, `replay_to(step)` (deterministic), `recall_what_if(…)` (time-travel debugging), tamper-evident op chain (`verify_timeline`, `timeline_head`, `audit_workspace`), multi-agent store (`export_bundle`/`import_bundle`/`merged_view` — chain-verified, convergent; see `docs/SYNC.md`) | `AgentSession`, `TimelineAudit`, `SyncBundle`, `SyncError` |
| **`trace`**                 | Parse `cargo test`/panic/backtrace into the crash's source locations (dynamic layer) | `parse_cargo_test_output`, `ExecutionTrace`, `TraceHit` |
| **`region_engine`** (v0.3)  | Clusters the graph into spatial regions; activation → context window; deterministic replay | `ContextRegionEngine`, `ContextWindow`, `RegionQuery` |
| **`context_region`** (v0.3) | Spatial-memory data model (3-D embedding, temperature, density) | `ContextRegion`, `ContextPoint` |
| **`context_policy`** (v0.3) | Dynamic context-admission score (replaces the static threshold) | `ContextPolicy` |
| **`region_metrics`** (v0.3) | Flat-vs-region locality measurement (precision/recall/tokens) | `LocalityReport`, `locality_report`, `causal_neighborhood` |
| **`experiment`** (v0.3)     | LLM-free hypothesis simulation: regional memory vs RAG/GraphRAG baselines on synthetic causal tasks | `ExperimentConfig`, `ExperimentReport`, `run_experiment` |
| **`eval`** (v0.3)           | Real-LLM evaluation harness (auto-gradable causal-chain tasks; OpenAI/Ollama) | `EvalConfig`, `EvalReport`, `run_eval` |
| `util`                      | Shared helpers (`sha256_hex`) | `sha256_hex` |
| **`scheduler`** (v0.3)      | HOT/WARM/COLD context paging by token budget | `ContextScheduler`, `MemoryZone` |
| **`workspace`** (v0.3)      | Async real-FS scanner; add/modify/remove delta | `WorkspaceScanner`, `WorkspaceDelta` |
| **`agents`** (v0.3)         | Multi-agent execution behind a trait | `Agent`, `AgentExecutor`, `CoderAgent`, `ReviewerAgent`, `SecurityAgent` |
| **`persistence`** (v0.3)    | Durable runtime state (directory) + verify | `PersistentRuntime`, `RuntimeState` |
| **`benchmark`** (v0.3)      | Cycle benchmark → JSON report | `BenchmarkHarness`, `BenchmarkReport` |
| `main` (bin)                | Thin entry → `commands_demo` / `commands_runtime` + inline commands (+ the premium `slha`/`octa`/`rsi` subcommands in fused builds) | — |

> The table above is the **core** (always-compiled) map; the hardening modules
> `sanitizer`, `injection_classifier`, `compressor`, `license` and `egress` are
> also always compiled. The premium fusion surface is below.

## CCOS_EXTENDED — fused premium modules & crates

Everything in this section is **off by default** (opt-in cargo feature), runtime
Pro-gated by the offline license, and architecturally one-directional: the
vendored crates never depend on `ccos` (the circular dependencies were inverted —
the adapters live on the CCOS side). See [`FUSION_PLAN.md`](FUSION_PLAN.md) for
the full architecture and [`AUDIT_FUSION_2026-07.md`](AUDIT_FUSION_2026-07.md)
for the audit.

| Module / crate | Feature | Responsibility | Key types |
| -------------- | ------- | -------------- | --------- |
| `slha_full` | `slhav2-full` | Pro gate to the REAL SLHAv2 kernel (backend + safety guard) | `FullSlhaAccess` |
| `octa_index` | `octasoma` | OctaSoma-backed region-sharded semantic index + conformal feedback | `SemanticMemoryAccess`, `SemanticFeedback` |
| `octacore_bridge` | `octacore` | CCOS-side half of the dep inversion: `ExternalMemory` as a cascade scope | `CcosScope`, `CausalCascadeAccess` |
| `rsi_bridge` | `rsi` / `rsi-dgm` | RSI audit adapter + Linux OS-sandboxed DGM (allowlist, air-gapped evaluator, hash-chain audit) | `CcosAudit`, `RsiAccess`, `DgmAccess`, `GuardedDgm` |
| `mcp_ext` | any premium | The `slha.*` / `octa.*` / `rsi.*` MCP namespaces multiplexed into `mcp` | `tool_specs`, `call_tool` |
| `crates/ccos-memory-runtime` | `slhav2`(+`-full`) | Runtime-neutral `MemoryProvider` middleware (distilled tile backend; scirust backend behind `slhav2-full`) | `MemoryProvider`, `ScirustBackend` |
| `crates/ccos-scirust` (+`-mcp`,`-c`,`-python`) | `slhav2-full` | The vendored SLHAv2 kernel: 128-byte tiles, INT4/grouped/NF4/mixed/TQ3 codecs, SIMD, `ElasticKvCache`, `LatentSafetyGuard`, EventLog persistence | `SciRustSlhaTile`, `ElasticKvCache` |
| `crates/ccos-octasoma` | `octasoma` | The vendored OctaSoma semantic-memory engine (`#![forbid(unsafe_code)]`) | `Embedder`, `HashEmbedder` |
| `crates/ccos-octacore` | `octacore` | The vendored cascade: causal-narrow → exact-cosine rerank | `Cascade`, `CausalScope` |
| `crates/ccos-rsi` | `rsi` | The vendored CERVO/RSI engine (meta-search, DGM, wasm sandbox, audit) — no `ccos` edge | `RSIAgent`, `DgmEngine`, `AuditLog` |

## Core data structures

```
MemoryGraph
├── nodes: HashMap<NodeId, GraphNode>     // the working set
├── edges: Vec<GraphEdge>                 // invariant: edges ⊆ nodes × nodes
├── paging_threshold: f64
├── max_in_memory_nodes: usize            // paging cap
└── clock: u64                            // advanced by tick(); drives recency decay

GraphNode { id, label, content, node_type,
            base_importance, failure_relevance, recency,
            access_count, created_at, last_accessed }
```

Node ids are namespaced strings: `file:<path>`, `mod:<path>:<name>`,
`use:<path>:<path>`, `sym:<path>:<name>`, `dep:<root>`. The `incremental` engine
relies on these prefixes to evict exactly one file's subgraph.

## Invariants (and where they live)

| Invariant | Enforced by |
| --------- | ----------- |
| `edges ⊆ nodes × nodes` (no dangling edges) | `MemoryGraph::add_edge` (rejects absent endpoints), `prune_dangling_edges`, `enforce_paging` |
| Node count ≤ `max_in_memory_nodes` | `enforce_paging` (called from `upsert_node`) |
| Deterministic eviction/ordering | total order *(score, NodeId)* in `enforce_paging`, `get_node_scores`, `select_context_window` |
| Guard output is always valid JSON | `GuardLayer::validate_and_sanitize` → `fallback_response` |
| Hash chain is tamper-evident | `EventLog::verify_integrity` (primary log, canonical chain) + `DistributedEventLog::compute_link_hash` / `verify_integrity` + `AgentSession::verify_timeline` (op-log chain; `open` refuses a tampered sidecar) |

Regression coverage: `tests/graph_invariants.rs` (dangling-free, bounded,
deterministic), `tests/long_term_stability.rs` (10k cycles),
`tests/snapshot_differential.rs` (reproducible hashes),
`tests/llm_adversarial_test.rs` + `tests/ccos_adversarial_suite.rs` (guard).

## Control flow

### `ccos analyze <path>`

```
collect_rs_files → for each file:
    IncrementalGraphEngine::process_delta(file, None, source, &mut graph)
        → ASTParser::parse_source → update_memory_graph (upsert nodes, add edges)
        → enforce_paging (deterministic eviction)
    EventLog::append(Parsing …); DistributedEventLog::append(hash)
prune_dangling_edges → report (scores, types, cycles, orphans) → [--dot|--out]
```

### `ccos demo`

A scripted single cycle touching every subsystem: parse → LLM+guard → consensus
→ incremental delta → failure propagation → context selection → replay → paging →
hash-chain integrity.

### `ccos top` / `blame` / `export`

Read-only queries (module `query`) over a graph built fresh from a path (`top`)
or loaded from a snapshot (`blame`, `export`):

- `top`    → `query::hot_set` — top-N nodes by causal score (the working set).
- `blame`  → `query::source_set` (upstream **causes**, walking `target → source`)
  plus `query::impact_set` (downstream **blast radius**, walking `source →
  target`); both are a deterministic BFS bounded by `--depth`.
- `export` → `query::to_graphml` — deterministic, id-sorted GraphML.

## How to extend

- **New node/edge type** — add a variant to `NodeType` / `EdgeType` (both are
  `serde`-tagged enums); update `MemoryGraph::to_dot` color/label maps.
- **New event** — add a variant to `EventPayload`; handle it in
  `EventReplayer::handle_event` and (if it should be hashed in tests) in
  `tests/snapshot_differential.rs::event_log_hash`.
- **New CLI command** — add a match arm in `main()`, an `OptsX::parse`, and a
  `run_x(...) -> i32`; document it in `print_help`, `README.md` and this file.
- **New invariant** — add a `MemoryGraph` method that checks/repairs it and a
  test in `tests/graph_invariants.rs`.

## Build, test, lint

```bash
cargo build --all-targets
cargo test                    # full suite — default build = real `syn` AST; add --no-default-features (heuristic) / --features llm (async)
cargo clippy --all-targets    # warning-clean (CI denies warnings)
cargo doc --open              # rendered module docs
```

Heavier chaos/stress harnesses live in [`../scripts/`](../scripts/).
