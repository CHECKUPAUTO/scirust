# Memory + Discovery Roadmap

Branch: **`feat/memory-discovery`** (off current `master`).

This roadmap extends SciRust's **algorithm-discovery** crates
(`scirust-algogen`, `scirust-synthesis`, `scirust-nas`, `scirust-symreg`,
`scirust-tn/discovered*`) with features that improve **memory management for
LLMs, AI agents, and causal memory organizations**. It was produced by a
multi-agent workflow (5 explorers → 4 design lenses → 3 adversarial judges that
verified every `file:symbol` anchor by `grep`/`read`) and then implemented.

## Key finding

SciRust already ships a **complete but unwired** LLM KV-compression subsystem:

- `PagedKvCache` — `scirust-core/src/nn/paged_attention.rs:21` (block paging)
- `ElasticKvCache` — `scirust-core/src/nn/elastic_kv_cache.rs:186` (two-level
  INT4 + bounded-budget eviction)
- `kvquant_kv` — `scirust-core/src/quantization.rs:1556` (per-channel K /
  per-token V); NF4 / SqueezeLLM codebooks
- `TTLinear` — `scirust-core/src/nn/tt_linear.rs:31` (drop-in `Module`,
  10–100× weight compression)

…but `MultiHeadAttention.kv_cache` is still a plain
`RefCell<Option<(Tensor, Tensor)>>` (`nn/transformer/attention.rs:28`,
`infer_step` at `:254`) — none of the bounded/compressed caches serve live
decode. **Wiring them is the single highest-leverage LLM-memory feature.**

The discovery crates have never been pointed at memory problems, and the memory
primitives (`scirust-arena`, `scirust-events-*`, `scirust-retrieval`,
`scirust-som`) are siloed and grow unbounded.

## P0 — applied on this branch

Additive, backward-compatible, all building on existing crates (no new substrate
invented — a judge requirement).

| # | Feature | Host | Status |
|---|---------|------|--------|
| 1 | **`AttentionBackend` trait** + adapters (plain / paged / elastic) and a numeric `decode_step` that wires the bounded KV caches into a live decode path of `MultiHeadAttention`. | `scirust-core/src/nn/kv_backend.rs` | ✅ applied |
| 2 | **`EpisodicEventLog`** — append-only, time-indexed, capacity-bounded episodic store over `scirust-events-core::Event`. | `scirust-events-core/src/episodic.rs` | ✅ applied |
| 3 | **`BoundedSemanticMemory`** — capacity-capped, decay-aware `DenseIndex` wrapper with importance/recency eviction; plus a backward-compatible `replay_cap` on `ImprovementLoop`. | `scirust-retrieval/src/forgetting.rs` (+ `feedback.rs`) | ✅ applied |
| 4 | **`CausalDag`** — directed, acyclic graph substrate (`scirust-graph` is undirected today): cycle-checking `add_directed_edge`, topological order, ancestors/descendants, `to_undirected` interop. | `scirust-graph/src/dag.rs` | ✅ applied |
| 5 | **Causal-aware retrieval** — `causal_rerank` re-ranks similarity hits by causal-graph proximity + intervention overlap. | `scirust-retrieval/src/causal_rerank.rs` | ✅ applied |
| 6 | **RAG context windowing** — `RagContext` retrieves top-k chunks and builds a bounded augmented prompt; optional `generate_with` runs `MiniLLM`. | `scirust-retrieval/src/rag.rs` | ✅ applied |
| 7 | **TT-attention** — apply `TTLinear` to `w_q/w_k/w_v/w_o`. | `scirust-core` | ⏳ deferred → P1 |

#7 is deferred because the four projection fields are typed `Linear`, not a
trait object, so doing it without struct surgery (which would touch
`state_dict` / `load_state_dict` / `sync` / `forward_3d` / `infer_step`) risks
breaking `master`. It is reclassified P1 and tracked below.

## P1 — next on this branch

- **TT-attention** (#7): generalize the four projections behind an
  `AttentionProjection` trait/enum so `TTLinear` can replace `Linear` for the
  q/k/v/o projections while keeping `state_dict`/`sync` working.
- **`KvCodec` trait** — compose `kvquant_kv` / NF4 into `ElasticKvCache`'s
  `KvTile` as a pluggable codec (`elastic_kv_cache.rs:138` `compress_grouped`).
- **`WorkingMemorySlab`** — bounded working-memory tier over
  `scirust-arena::Slab<T,N>` with TTL/LRU eviction (`scirust-arena/src/slab.rs`).
- **`ConsolidationPipeline`** — scheduled episodic→semantic transfer wired
  through `ImprovementLoop` (`scirust-events-runtime`).
- **`AgentMemory`** — unified trait + façade across episodic/working/semantic
  tiers with a pluggable `EvictionPolicy`. **New crate `scirust-agent-memory`**
  (depends on `scirust-events-core`, `scirust-arena`, `scirust-retrieval`).
- **`CausalMemoryStore`** — memory-valued structural causal model extending
  `scirust-neuro-symbolic::CausalEngine` (`probabilistic/causal.rs:8`, today a
  linear SCM with a single `do()`; nodes are numeric, not memory items).
- **Do-calculus & counterfactual engine** — backdoor/frontdoor adjustment,
  identifiability, abduction–action–prediction counterfactuals (new module next
  to `causal.rs`).
- **`RetrievalPolicySelect`** — port `scirust-algogen`'s
  `select_*` / `fitness_sort` / `evolve_sort` selection pattern to auto-select a
  retrieval/encoding strategy per workload.

## P2 — discovery-driven (gated behind a KV-cache trace-replay harness)

The bridge from "algorithm discovery" to "memory": use the discovery crates to
**search** memory-management policies instead of hardcoding them. All depend on
a shared **`MemoryTraceReplay`** harness (replay a recorded KV-cache trace under
a candidate policy) — build it first, in `scirust-core`.

- **`MemoryNas`** — retarget `scirust-nas` from layer *shapes* to memory
  *topologies* via an open `MemoryLayerSpec`
  (`FullKv`/`Paged`/`ElasticInt4`/`KvQuant`/`Summarize`/`RAG`), 3-objective Pareto
  fitness reusing `scirust-symreg::pareto_insert`.
- **`EvictionPolicySymReg`** — `scirust-symreg::discover` learns a closed-form,
  auditable eviction-scoring formula from KV traces; drops into
  `ElasticKvCache::append` in place of FIFO.
- **`MemorySExpr`** — extend `scirust-synthesis::SExpr` with
  `Store`/`Recall`/`Evict`/`LookupSimilar` ops + a memory-aware cost model
  (bytes/bandwidth/working-set); synthesize memory-management programs.
- **`MemoryForgeRegistry`** — generalize the `inject_elite` harness in
  `scirust-tn/discovered*.rs` into a `DiscoveryProblem` trait; register KV
  compression / GEMM as the first problems.

## Branch recommendation

`origin/feat/nlp-compression-primitives` is stale/divergent (~21k deletions,
old base) — **not** used as a base. The discovery + memory crates all exist and
build on current `master`, so this work branches fresh as
`feat/memory-discovery`, lands P0 → P1 → (trace-replay harness) → P2.

## New crate

`scirust-agent-memory` — the `AgentMemory` façade unifying episodic / working /
semantic tiers with a pluggable `EvictionPolicy`. Everything else extends an
existing crate.

## Verification

Each applied P0 feature carries its own `#[cfg(test)]` tests. The branch is kept
green with `cargo fmt --all --check`, `cargo clippy --workspace --all-targets
-- -D warnings`, and the affected crates' test suites.