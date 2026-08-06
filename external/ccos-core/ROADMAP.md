# CCOS — Improvement Plan

Prioritized roadmap from the audit. Effort: S/M/L.

> **CCOS_EXTENDED (premium fusion)** is tracked separately: architecture &
> phasing in [`docs/FUSION_PLAN.md`](docs/FUSION_PLAN.md) (P0–P6 ✅), audit in
> [`docs/AUDIT_FUSION_2026-07.md`](docs/AUDIT_FUSION_2026-07.md).

## 🚀 Direction — unbounded working memory (frugality × available RAM)

"Infinite" working memory is a **direction**, not a literal claim. CCOS keeps the
resident window tiny (*frugality*) and uses the machine's RAM as the backing
store, so the effective working memory is **as large as RAM allows relative to a
small window** — and that ratio is how far we stretch toward the asymptote. The
cognitive MMU, made real: page, don't drop.

- ✅ **Slice 1 — non-destructive eviction → a COLD tier (the swap).** Eviction now
  *demotes* a node (with its incident edges) into a COLD tier instead of dropping
  it; the resident set stays bounded by `max_in_memory_nodes`, the backing store
  grows into RAM, and `MemoryGraph::page_in` brings anything back. A failure on a
  demoted node resurrects it (a page fault) rather than erroring. Observable via
  `MemoryStats.cold`. Deterministic (sorted demotion, BTreeMap COLD store).
- ✅ **Slice 2 — page-fault from COLD on the read paths.** `page_fault` resurrects
  cold faulting files (via the cold-aware `signal_failure`); a `recall` *around* a
  demoted node pages it **and its cold neighbours** (`MemoryGraph::cold_neighbours`)
  back via `CcosMemory::ensure_resident`, wired into `AgentSession::recall*` and
  **reproduced on replay** so `replay == live`. `set_max_resident` exposes the
  frugal-window knob.
- ✅ **Slice 3 — spill COLD content to disk → RAM-bounded content, disk-unbounded.**
  An opt-in `attach_cold_spill(dir, inline_budget)` flushes the coldest COLD
  *content* blobs to a content-addressed on-disk store (SHA-256 keys, the CCR
  addressing scheme) once resident COLD content passes a byte budget, leaving a
  hash **stub** in RAM; `page_in` faults them back, **hash-verified** (a tampered
  or missing blob is a cold-miss, never a silent empty restore). Identical content
  is **deduplicated**; spill is lossless and deterministic (coldest-first). Off by
  default ⇒ byte-identical serialization, so the replay/snapshot invariants are
  untouched. `MemoryStats.cold_spilled{,_bytes}` surface it. **Honest scope:** only
  the unbounded *content* moves to disk — per-cold-node *metadata* (small) still
  grows in RAM (a true O(1) on-disk index is future work); blobs are stored verbatim
  (dedup, no codec yet); a snapshot taken with spill on needs its `dir` re-attached
  to restore (sidecar). (M)
- ✅ **Slice 4 — compact the coldest tail → a frugal backing store.** The deepest
  tier. An opt-in `set_cold_content_budget(Some(bytes))` keeps total COLD *content*
  (inline + spilled) toward `bytes` by **lossily compacting** the coldest entries:
  routed by kind, code is skeletonised / prose summarised / JSON crushed
  (`CausalAst` / `CausalSumm` / `CausalCrusher`), and the full original is
  discarded. Deterministic (coldest-first), **observable** (`is_compacted`,
  `MemoryStats.cold_compacted`), and explicitly the place where "infinite working
  memory as a *direction*" bottoms out: at the floor frugality wins, and CCOS
  compacts to a summary — **never silently drops**. Off by default ⇒ COLD stays
  lossless, serialization byte-identical. **Honest scope:** this bounds the cold
  *content* footprint, **not** the entry *count* (the `BTreeMap` still holds a stub
  per node — its resident *size* is bounded by slices 5/5b below; bounding the
  *count* itself is still future work); compaction is lossy and, like spill, an
  operational mode layered on the deterministic default, not part of replay. (M)
- ✅ **Slice 5 — deep-spill the per-entry metadata (lossless, measured-first).** A
  measurement (`docs/MEASUREMENT_cold_ram.md`, `examples/cold_ram.rs`) first showed
  the COLD tier's dominant *resident* cost is per-entry **metadata** — ~2.8× the
  spilled content, ~60% of it edges — and that lossy edge-contraction is the *wrong*
  lever (it inflates that edge cost on hubs). So `set_cold_resident_budget(Some(b))`
  drives resident COLD metadata toward `b` by deep-spilling the coldest entries to
  the same content-addressed store, keeping only the neighbour **ids** resident for
  `cold_neighbours`/region paging. Lossless (faults back, hash-verified, on
  `page_in`), deterministic, off by default (byte-identical default snapshot/replay).
  Shrinks edges to ids — never adds bridge edges — so hubs get *cheaper*, not the
  O(degree²) blow-up contraction would cause. (M)
- ✅ **Slice 5b — compact husk → drop the per-entry struct floor.** Slice 5 kept a
  full `ColdNode` husk and stalled at ~11% on the fixture against the
  `size_of::<ColdNode>()` floor. A deep-spilled entry is now archived *whole* to one
  blob and represented in RAM by a compact `DeepHusk` (body-blob stub + neighbour
  ids) in its own `cold_deep` map, so *every* entry shrinks and the budget is
  actually reached: resident COLD metadata **halves (−50%)** on the same fixture
  (~108K/120K entries archived). Deep husks are terminal (excluded from further
  spill/compaction); still lossless, deterministic, and off by default
  (`cold_deep` serde-elided when empty ⇒ byte-identical default path). This bounds
  the per-entry resident *size*; bounding the entry *count* is slice 5c below. (M)
- ✅ **Slice 5c — bound the entry *count* → an `O(1)`-resident COLD tier ("Lever 2").**
  The deep tier no longer keeps a `BTreeMap` node per husk in RAM. Husks live in a
  hand-rolled, dependency-free LSM-lite (`src/cold_index.rs`: sorted segments + sparse
  index, memtable + flush, tombstones + compaction, bounded LRU cache, each model-check
  property-tested before wiring). The resident `cold_deep` map is gone; `cold_neighbours`
  is `O(degree)` via a keyed on-disk **reverse-adjacency** index, and `flush_cold_tier`
  durabilises at checkpoint (crash-recovery tested). Measured (`examples/cold_count.rs`):
  **≈2 B/husk resident** (vs 146 B), 1 GiB at **~537 M husks**. `replay == live` is
  untouched (the log is the source of truth; the cold tier is a rebuildable cache). A
  cross-cutting *NodeId interning* would shave resident id strings further but is a poor
  trade on the already-bounded engine (public-API + hot-path ripple) — designed,
  deferred. See `docs/DESIGN_cold_entry_count.md`. (L)

## 🎯 Direction — better retrieval
- ✅ **Slice A — hybrid entry fusion.** A new `Recall::Hybrid` resolves a
  free-text task's entry node by **reciprocal-rank fusion** of three independent
  rankings — lexical token overlap, semantic INT4-TF-IDF cosine, and the causal
  **active-failure focus** — before the usual causal-region expansion. RRF needs
  no cross-signal calibration (it ranks, not scores), so a node strong on any axis
  surfaces while consensus wins; `K = 60`. The causal vote is **sparse** (only
  nodes under failure pressure), so it abstains on a quiet graph (no id bias) and
  speaks for the active problem region once a failure is signalled — the
  CCOS-native signal. Deterministic; wired through `recall()`, the MCP `recall`
  tool (`strategy:"hybrid"`), and the runtime CLI. (M)
- ✅ **Slice C — self-improving retrieval from the replayable log.** The CCOS-native
  gem. A retrieval **reward** is read straight off the hash-chained timeline: for
  each recorded recall, was the node the agent engaged *next* (a failure / page-fault)
  in the window that recall would have produced? `AgentSession::tune_recall_weights`
  then learns the `ScoringWeights` that maximise that hit rate by **deterministic
  coordinate ascent, evaluated by replay** (same log ⇒ same weights).
  `adopt_tuned_recall_weights` applies them *and records an `Op::Retune`* so the
  learned policy is **auditable and reproduced on replay** — `replay == live` still
  holds. Better retrieval that also reinforces the moat: nobody else has a
  deterministic, replayable causal log to train on. **Honest scope:** the reward is
  a proxy (the next failing node = the context recall should have surfaced); the
  optimiser is greedy (a local optimum) over the four scoring weights; evaluation is
  one replay per candidate, so it is an offline/maintenance call. (L)
- ✅ **Slice B — opt-in learned embedder behind a feature flag.** A `learned-embed`
  feature distils the deterministic INT4 TF-IDF into a learned **latent-semantic
  (LSA / truncated-SVD) projection** — the top singular vectors of the corpus's own
  document–term matrix, found by a fixed cyclic-Jacobi sweep (`src/lsa.rs`). It
  captures synonymy/transitivity raw TF-IDF can't (a query term that only
  *co-occurs* with a doc's terms still matches), yet is **zero-new-dependency and
  fully deterministic**, so the replay invariant holds. INT4 TF-IDF stays the
  default (the measured baseline); the projection is wired into `build_embeddings`
  only under the feature, so the default build is byte-identical. **Honest scope:**
  LSA is a *linear* distillation (not a neural model); it helps most when there are
  enough documents to truncate (`rank < docs`); the Gram-matrix eigensolve adds cost
  to the per-recall embed build, so it is an opt-in. **Measured (see
  `docs/MEASUREMENT_recall.md`):** on a synthetic recall benchmark LSA does *not*
  beat — and can hurt — the default TF-IDF in CCOS's entry-selection use (it dropped
  hybrid's overall hit-rate 58%→38%). It works in the micro-test but its dense-ranking
  strength doesn't transfer to picking a single entry node; so it **stays opt-in and
  off**, awaiting a future experiment that wires it into a *ranking* stage. The
  capability is built and honest; the recommendation is "not yet." (L)
- ✅ **Slice D — LSA wired where it earns its keep: re-ranking, not entry selection.**
  The follow-up B's measurement asked for. `set_lsa_rerank(Some(rank))` (opt-in) re-orders
  the recalled *region* by rank-`rank` LSA similarity — the recall@k≥5 regime, not
  recall@1. Measured (`examples/lsa_rerank.rs`): target mean rank 11.8 → 2.1. Honest
  limiter, also measured: synonym *entry* selection (TF-IDF scores a synonym ≈0) gates it,
  so re-ranking re-orders what entry selection found and never repairs an empty region.
  Deterministic; `replay == live` untouched (recall is read-only). (M)
- ✅ **Slice E — natural-language queries match code identifiers (subword tokenization).**
  The TF-IDF tokenizer splits `snake_case`/`camelCase`, so a query like "connection pool
  acquire" — which shared *zero* tokens with `connection_pool_acquire` before — now
  matches. Measured (`examples/identifier_recall.rs`): 6/6 NL queries recall their target
  at rank ≤2; on the LSA corpus the topic target's mean rank improves 11.8 → 2.0 (the
  semantic signal, not just a re-ranker, now does the work). Deterministic. (S)

## ✅ Done — audit pass 4 (hardening the new arcs)

Four adversarial auditors (determinism, `replay == live`, default-path byte-identity,
resource bounds) **confirmed the crown invariants hold on the default path** for all
seven new slices, and surfaced three real issues, now fixed:

- **Spill-blob GC** — the on-disk spill store had no delete path, so re-ingest /
  remove / compaction of a spilled node leaked its blob. Added a **dedup-safe**
  reclaim (`release_blob_if_orphan`: delete only when the hash's last referent is
  gone). The "future GC pass" the slice-3 comment promised now exists.
- **Compaction floor** — un-shrinkable cold entries were re-tried (and re-read from
  disk) every ingest; now parked via `ColdNode.at_floor` and skipped.
- **LSA determinism** — `build_embeddings` pins corpus order by id, so the
  `learned-embed` Gram-matrix f64 sum no longer depends on `HashMap` order.

## ✅ Done — perf pass (measure-then-fix)

A latency benchmark (`examples/recall_latency.rs`, `docs/MEASUREMENT_latency.md`)
showed recall was **super-linear** in corpus size because every query recall rebuilt
derived structures: `around`/`task` re-ran the whole region clustering, and
`semantic`/`hybrid` re-fit the embedding store (plus the LSA eigensolve). Fixed by
memoising both behind a **graph version counter** on `CcosMemory` — reused only at
the same version, so **never stale**, byte-identical to a rebuild, so
**determinism / `replay == live` hold** (regression-tested). At 2000 nodes:
`around` **75 ms → 13 µs** (~5700×), `semantic` ~42×, `hybrid` ~21×.

Still deferred (confined to **opt-in / scale** paths, not the default hot path):
per-ingest `O(cold)` budget re-scans and the `cold_neighbours` scan — both only
when a spill store / compaction budget is attached or the COLD tier is populated.

## ✅ Done — audit pass 1 (correctness)

- **Fixed critical edge leak.** `enforce_paging()` ran re-entrantly inside
  `upsert_node()`, so `add_edge()` attached edges to just-evicted nodes; dangling
  edges grew `O(cycles)` (9,000+ edges for 200 nodes), breaking the `O(Δ)` promise
  and the stability budget. `add_edge` now rejects dangling endpoints,
  `enforce_paging` prunes defensively, and `tests/graph_invariants.rs` locks the
  invariant. (10k-cycle slowdown: 11× → 1.08×.)
- **Deterministic eviction/ordering** — tie-break on `NodeId` across paging,
  context selection and score listing → reproducible replays/snapshots.
- **Hardened the guard** — `is_valid_json` now requires the *whole* payload to
  parse (was accepting any valid prefix → trailing injection slipped through).
  Replaced tautological adversarial assertions with real safety checks.
- **Real CLI** (`demo`, `analyze`, `help`, `version`) replacing the hard-coded
  demo that ended in `process::exit(0)`.
- **Docs**: `README.md`, crate-level rustdoc, this roadmap. Zero clippy warnings.

## ✅ Done — audit pass 2 (capability)

- **Enforced `GuardConfig::max_nesting_depth`** (was a defined-but-unused config
  field) + tests. *(was P0.2)*
- **Persistence** — `persist::KernelSnapshot` (graph + event log + hash chain)
  with `save`/`load`; new `ccos analyze --out`, `ccos verify`, `ccos replay`
  commands. *(was P2.7)*
- **Multi-model consensus wired in** — `LlmClient::query_models` +
  `ConsensusEngine` in the demo. *(was P1.4)*
- **Adversarial chaos mode** — `ccos chaos [--iters N]` drives fault injection
  through the guard and asserts it never emits invalid JSON. *(was P1.5)*
- **Hash-chained log integrated** — the demo and `analyze` now build a
  `DistributedEventLog`; `verify` checks its integrity. *(was P1.6, partial)*
- **Recency clock wired** — the demo calls `MemoryGraph::tick()` between cycles
  so recency actually decays. *(was P2.8)*
- **Richer `analyze`** — `--json` export, `--cycles` dependency-cycle detection,
  node-type histogram. *(was P2.9)*
- **Fallible `LlmClient::try_new`** alongside the panicking `new`. *(was P3.12)*
- **CI** — `.github/workflows/ci.yml` runs `build`, `clippy -D warnings`,
  `test`, and a CLI smoke test (`analyze → verify → replay → chaos`).
- **Full graph reconstruction from the event log** — `EventLog::record_graph`
  emits `NodeUpserted`/`EdgeAdded` events; `GraphReconstructor` rebuilds an
  identical graph from the log alone (`replay` reports `matches snapshot: true`).
  Closes the event-sourcing loop. *(was P0.3 / P0.2)*
- **Graphviz export** (`analyze --dot`) and **orphan-node** reporting.

## ✅ Done — v0.3 (Autonomous Context Runtime)

- **Context scheduler** (`scheduler.rs`) — HOT/WARM/COLD paging by token budget
  and priority; `allocate`/`evict`/`optimize`, no node lost.
- **Real workspace scanner** (`workspace.rs`) — async `tokio::fs` scan with
  add/modify/remove delta detection feeding only Δ to the engine.
- **Multi-agent layer** (`agents.rs`) — Coder/Reviewer/Security agents, guarded
  + logged + deterministic.
- **Persistent runtime** (`persistence.rs`) — directory-based save/load/restore
  with verify-on-restore.
- **Benchmark framework** (`benchmark.rs`) — cycle benchmark → JSON report
  (100k stress in CI; 1M opt-in).
- **CLI** — `scan`, `agents`, `benchmark`, `runtime` (capstone).
- **Quality** — `main.rs` split into `commands_demo`/`commands_runtime`
  (1206 → 679 lines); `util::sha256_hex` DRY consolidation; dead code removed;
  config flags (`--max-nodes`, `--budget`); property tests; criterion benches.
  *(was P2.4, P2.5, P3.8)*

---

## ✅ Done — v0.3 Context Region Engine (spatial memory)

- **Context regions** (`context_region`, `region_engine`) — the 1-D scored graph
  is lifted into a spatial map: nodes are embedded in a 3-D context space
  (structural / causal / temporal) and clustered into regions (connected
  components of the cross-file causal-link graph) with a temperature and causal
  density. Regions are hydrated as `ContextWindow`s instead of loading files.
- **Dynamic admission policy** (`context_policy`) — the static `0.6` threshold
  becomes a function of token pressure, task complexity, region temperature and
  density.
- **Event sourcing + deterministic replay** — `RegionCreated/Activated/Merged/
  Evicted/ContextWindowGenerated` events; `replay_from` reconstructs regions
  bit-for-bit from a rebuilt graph (proof + 10k-cycle no-drift test).
- **Locality metrics** (`region_metrics`) + `scripts/region_benchmark.sh`: region
  selection covers 97% of a task's causal neighbourhood vs 35% flat, ≈48% fewer
  tokens; regions 95.5% internally connected.
- **Docs**: `docs/context_regions.md` + an arXiv research paper in `docs/paper/`
  (formal model, determinism theorem, falsifiable comparison protocol vs
  RAG/GraphRAG/MemGPT/LangGraph). `ccos regions` CLI.

---

## Audit pass 3 — full-codebase review (2026-06-24)

A four-axis read of the whole tree (correctness/determinism, architecture/redundancy,
honesty code↔docs↔paper, tests/API). **Fixed in this pass:**

- **`truncate()` UTF-8 panic** (`main.rs`) — byte-sliced a multi-byte char on any
  non-ASCII id/message; now cuts on a char boundary (+ regression test).
- **Non-reproducible distributed-log chain** — `compute_link_hash` hashed the
  wall-clock timestamp; now excludes it (mirrors `EventLog`), so the chain is
  replay-reproducible (+ test). *On-disk chain hashes change — a format bump.*
- **`parser.rs` slice OOB** under `--features syn-parser` (a span line past EOF) —
  clamped into the slice (+ test).
- **`bench_compress` panicked** (hard-coded `/root/CCOS`) — derives the corpus from
  `CARGO_MANIFEST_DIR`; README §5 table re-measured on the real 38-file corpus.
- **Test/file counts reconciled** — docs disagreed (156/202/212/285/288); the real
  default-feature `cargo test` count is **364**, fixed across README, PITCH,
  ARCHITECTURE, PAPER.md and all six `.tex`; "33 Rust files" → 38.
- **Honesty** — `mcp.rs` docstring now lists all 9 tools; `COMPETITIVE.md` no longer
  implies *live* semantic recall; `lib.rs` documents which recent modules are in-tree
  but **not yet on the live path**.

**Remaining (decisions / larger work, not bugs):**

### A — Wire in the unwired recent modules — ✅ done
- ✅ `embeddings` → a `Recall::Semantic` strategy (INT4 TF-IDF cosine entry), exposed
  via the MCP `recall` tool + `ccos memory`. *(Follow-up: cache the per-call store —
  see C.)*
- ✅ `eviction_policy` → blended into `MemoryGraph::enforce_paging` (untrained ⇒
  identical to the greedy; `train_eviction_policy` fits it offline). *(Follow-up:
  derive training transitions from the live op-log instead of an external feed.)*
- ✅ `injection_classifier` → `IngestReport.injection_score` / `injection_flagged`
  via a shared detector, on the live ingest/MCP path.

### B — Collapse duplicated abstractions
- ✅ One snapshot type: `persistence::RuntimeState` is now a type alias for
  `persist::KernelSnapshot` (the duplicate payload struct is gone); the
  integrity check is shared via `KernelSnapshot::verify_integrity` (used by the
  runtime restore and a new `KernelSnapshot::load_verified`).
- One event chain: collapse `distributed_event_log` onto `event_log`'s chain, or
  document why two exist. (M)
- One context selector: designate `CcosMemory::recall` canonical; demote
  `select_context_window` / `hot_set` / `hot_context` / `activate_region`. (M)
- One snapshot-error type (unify on `MemoryError`). (S)

### C — Encapsulation & API
- ✅ `MemoryGraph.{nodes,edges}` are now `pub(crate)` + read accessors (`node`,
  `node_mut`, `node_ids`, `node_entries`, `node_values`, `contains_node`, `edges`)
  — external callers can no longer push a dangling edge or orphan a node and break
  the `edges ⊆ nodes²` invariant. *(Still `pub` and could get the same treatment:
  `EventLog.events`, the `DistributedEventLog` fields, the `LinearModel` fields.)*
- ✅ `lib.rs` re-exports the core entry types at the crate root (`ccos::CcosMemory`,
  `Recall`, `MemoryGraph`, `AgentSession`, `KernelSnapshot`, …) with a crate-level
  doc-test; `#[non_exhaustive]` on the three **error** enums (`MemoryError`,
  `PersistenceError`, `ModelError`). The event/`Recall`/`NodeType`/`EdgeType` enums
  are **deliberately left exhaustive** — CCOS is its own only consumer, so the
  compiler's exhaustiveness check (catch a new variant nobody handled) is worth
  more here than cross-crate add-without-break.
- Cache the recall-time region clustering instead of rebuilding it per `around`/`task`
  call. (M)

### D — Test coverage
- ✅ The CLI now has coverage: a black-box `tests/cli.rs` (version/help, unknown
  command → exit 2, `analyze → verify → replay` round-trip, `sanitize --strict`
  on a Trojan-Source bidi override) driven via `CARGO_BIN_EXE_ccos`, plus unit
  tests for the option parsers (`analyze`/`top`/`chaos`/`blame`/`focus`, covering
  every distinct parse pattern — value flags, positionals, two-positionals, and
  the `--workspace` optional-arg branch). The remaining parsers reuse these
  patterns.
- ✅ Compressor reversibility-under-eviction — the test found and fixed a real
  (latent) bug: CCR eviction could drop a ref the current window had just handed
  back. See the CHANGELOG "Fixed" entry.
- Still TODO: `persist` disk save→load hash-stability, MCP parse-error envelopes,
  and an equal-score eviction-order tie-break test. (M)

### E — Hygiene
- Extract `main.rs` (2.3 KLoC) into per-domain command modules. (S)
- Consolidate the 3 license files; document the three-log taxonomy in `lib.rs`. (S)
- Port the security subsection + originality framing into the zh/ko/ar papers (en/fr/es
  have them) — *deferred by the maintainer; test counts already corrected.* (S)

---

## Remaining

### P0 — Correctness

- ✅ **`syn`-based AST parser — now the *default*.** The parser builds a real Rust
  AST (nested-module bodies, multi-line signatures, grouped `use`, impl methods) by
  default, with the zero-extra-dependency line heuristic as the fallback (selected by
  `--no-default-features`, or used automatically when a file does not parse as valid
  Rust). Measured 36.5% more faithful than the heuristic on real code (import recall
  66.9% → 100%; see `docs/MEASUREMENT_ast.md`), and `syn`/`proc-macro2` are already in
  the tree via serde, so it is free. CI lints and tests both paths. See
  `src/parser.rs::syn_ast`.

### P1 — Depth

- ✅ **Canonical hash-chained log** — *done.* The primary `EventLog` is now
  tamper-evident: every `append` links the event into a SHA-256 chain over its
  replayable content (sequence + type + payload, excluding the non-deterministic
  `id`/`timestamp` so the chain stays reproducible). `EventLog::verify_integrity`
  detects any payload tamper, reorder, insertion or deletion, and `ccos verify` /
  `ccos replay` check it on every run. See `src/event_log.rs`.

  ✅ **…and the session op-log (paper §9 item 2, closing slice)** — *done.* The third
  log — the `AgentSession` op-log, the timeline that actually carries `replay == live` —
  is now chained too: every recorded `Op` links into a SHA-256 chain (`chain`/`anchor`),
  the replay **baseline** is pinned by a commitment (`baseline_hash`), compaction moves
  the *anchor* but never the head (one hash commits to the whole history since genesis),
  and `AgentSession::open` **refuses** a workspace whose sidecar fails the check
  (`MemoryError::TimelineTampered`, sidecar left intact for forensics) while chain-valid
  staleness still self-heals. Pre-chain sidecars load and are backfilled. `ccos verify
  <workspace>` audits a sidecar without opening it. Every run is auditable by default —
  the paper's item 2 is fully landed. See `src/agent_session.rs`.

  ✅ **Distributed multi-agent store (paper §9 item 5)** — *done; the paper's future-work
  list is fully cleared.* File-based, chain-verified timeline exchange (`ccos sync
  export|import|status|materialize`, `SyncBundle` = plain JSON over any transport incl.
  sneakernet). Imports verify every link and refuse tampering, gaps, self-imports and
  **equivocation** (one agent id, two histories — caught by the op-log chain). The shared
  brain is `merged_view()`: a pure function over all known per-agent logs ⇒ agents holding
  the same logs materialize **bit-identical** views (`CcosMemory::state_fingerprint`) — a
  state-based CRDT, no consensus, no network, no new dependency. Measured in
  `examples/sync_crux.rs`; contract in `docs/SYNC.md`.

  ✅ **Signed bundles — cryptographic agent identity (`signed-sync`, item 4 of the paper's
  refreshed §9)** — *done.* Per-workspace ed25519 identity (`ccos sync keygen` →
  `<workspace>.ccos.key`), exports sign the canonical bundle payload automatically, imports
  verify + **TOFU-pin** the key per agent id: spoofing (`KeyMismatch`), stripped-signature
  downgrades (`UnsignedFromPinned`), bad/half signatures (`BadSignature`, every build) and
  signed-bundles-on-crypto-free-builds (`SignedUnsupported`) all refuse; a refused bundle
  changes no state, pins included. Zero new crates (reuses the optional `ed25519-dalek` +
  in-tree `getrandom`); the default build stays crypto-free and unsigned federations work
  unchanged. See `docs/SYNC.md` §"Signed bundles".
3. **Semantic edges.** (L) — *both halves landed; deep polish underway.* Call-graph (fn→fn `Calls`:
   bare, qualified, `self`/`Self` methods — #74/#75/#76/#77; plus **renamed-import alias calls**
   `use a::b as c` and **cross-impl-block self-calls** via per-type unioned method sets) **and
   data-flow** (`fn → static/const` `DataFlow` edges — Slice 1 bare `SCREAMING_SNAKE`, **Slice 2
   qualified `m::CONST`** through the shared `resolve_qualified` machinery; `src/memory.rs`,
   resolve-uniquely-or-skip throughout — never a wrong edge). The data-flow blind spot lexical
   retrieval misses (the cross-vocabulary co-reader link) is measured in
   `docs/MEASUREMENT_data_flow_crux.md`. `x.bar()` receiver-type inference **landed** (#23 + #126
   typed idioms, then Slice 4: field receivers `self.field.bar()` + declared-return chains
   `f().bar()` / `x.m().bar()` — the paper's "richer receiver inference" item, **+107 Calls edges
   (+10.6 %)** on the CCOS corpus; trait-object dispatch skipped *by design*, see
   `docs/MEASUREMENT_resolution_coverage.md`). Remaining: data-flow write/read direction.

   - ✅ **Spectral primitive (#13, first slice).** `src/spectral.rs::eigenvector_centrality` — a
     deterministic, dependency-free power-iteration eigenvector centrality over the (symmetrized,
     `A + I`-shifted) adjacency, complementary to the damped `MemoryGraph::eigencentrality`. A clean
     `A x = λ x` brick for the deferred spectral-regions / temporal-tensor / `scirust` work.
   - ✅ **Q-Page dual-evidence belief layer (slice 1) — *contested knowledge*.** A claim node carries
     two typed evidence surfaces, `EdgeType::Supports` (`S_A`) and `EdgeType::Contradicts` (`S_¬A`),
     and `MemoryGraph::qbelief` derives a **signed** `belief ∈ [−1, 1]` = `(s−c)/(s+c+ε)` and a
     **geometric** `conflict`, each edge weighted by its source **authority** — pure, no stored
     state, `replay == live`. `conflict` flags a *contested* claim, the state similarity-only retrieval
     cannot represent (`docs/MEASUREMENT_contradiction_crux.md`: a refutation's cosine sits *inside*
     the support band, so no threshold separates support from refutation; the typed edge does).
     Contradictions are explicit, replayable assertions (`CcosMemory::assert_contradiction` /
     `Op::Assert`); the **LLM extractor** (`src/extractor.rs`, behind `llm`) now distills them from raw
     text into the same `{claim, source, stance, authority}` shape (provider-agnostic; output recorded
     as `Op::Assert` so `replay == live`). Per-source **authority** weights the belief, and
     `QBelief::is_validated(min_belief, max_conflict)` gates action — measured by the **Conflict-of-
     Origins** bench (`docs/MEASUREMENT_conflict_of_origins.md`: a credible source outweighs a dubious
     one; `conflict` flags the residual dispute). Rule-based detection and a trust/authority policy are
     follow-ups.
   - ✅ **Q-Page decay — knowledge half-life (slice 2).** `MemoryGraph::qbelief_decayed(claim, half_life)`
     fades each evidence edge by `0.5^(age / half_life)` (age from its `created_at` vs the current
     `clock`) — lazy, pure, deterministic (`replay == live`), with no history mutated. A fresh
     assertion outweighs an ageing one, so recent evidence *resolves* a stale, never-reaffirmed
     dissent that plain `qbelief` deadlocks forever (`docs/MEASUREMENT_decay_crux.md`: conflict
     `0.67 → 0.06` as the objection ages, vs a frozen `0.67`). Per-class half-life and decay on the
     retrieval path are follow-ups.
   - ✅ **Q-Page belief propagation — single hop (slice 3).** `MemoryGraph::propagate_beliefs(threshold,
     damping)`: for each `Causes` edge `A → B` whose source is *resolved* (`|belief| ≥ threshold`), add
     a derived, attenuated `Supports`/`Contradicts` on `B` (`edge.weight · damping · |belief|`), so a
     claim with no direct evidence inherits a weaker, signed belief from its causes — belief revision
     across the causal graph (`docs/MEASUREMENT_propagation_crux.md`: an effect inherits `±0.31` from a
     `±0.75` cause; the wavefront stops at one hop, no cascade). Deterministic, idempotent. Multi-hop +
     scheduler + an `Op::Propagate` for replay are the next slice.
   - ✅ **Q-Page belief propagation — multi-hop to convergence (slice 4).**
     `MemoryGraph::propagate_beliefs_converge(threshold, damping, max_rounds)` repeats the single-hop
     sweep until a **fixpoint** (a round derives no new edge) or `max_rounds`, so a chain `A → B → C`
     revises in one call instead of a sweep per hop. Bounded and terminating by construction:
     `add_edge` dedups on `(source, target, type)` so each directed `(cause, effect, polarity)` triple
     derives at most one edge (the well-defined fixpoint), and the attenuation `damping · |belief| < 1`
     makes the wavefront **stop on its own** once inherited belief falls below `threshold` — the
     `max_rounds` cap only backstops a `Causes` cycle. Deterministic (reuses the single-hop
     collect→sort→dedup→add discipline), `derived-not-stored` (adds no snapshot state), demonstrated in
     `examples/propagation_crux.rs` (multi-hop lifts `C` where one hop leaves it at 0). A priority
     scheduler (topological order, fewer rounds) and a replayable `Op::Propagate` — which needs
     session-level causal-edge authoring first — remain the next slice. See
     `docs/PROPOSAL_scirust_enhancements.md` §2.1.

### P2 — Ergonomics

4. ✅ **Configurable scoring/paging/guard** — *done.* Scoring (`ScoringWeights::from_env`,
   `CCOS_W_*`), paging (`MemoryGraph::new_from_env` / `paging_threshold_from_env`:
   `CCOS_PAGING_THRESHOLD`, `CCOS_MAX_RESIDENT`), and the output guard (`GuardConfig::from_env`:
   `CCOS_GUARD_MAX_OUTPUT/REQUIRE_JSON/RELIABILITY/SANITIZE/MAX_DEPTH`) are all env-overridable,
   default-identical. (S)
5. ✅ **Benchmarks** — *done.* `benches/delta_bench.rs` (criterion) times `process_delta` across
   backgrounds 0→8000, plus a **deterministic CI guard** (`process_delta_footprint_is_background_independent`)
   so the `O(Δ)` claim is asserted, not just measured.
6. **`analyze` extras** — *GraphML export* ✅ (`query::to_graphml`, `ccos export`) and *dead-symbol
   detection* ✅ (`MemoryGraph::dead_symbols`, surfaced by `ccos analyze`). Remaining: per-file
   failure simulation. (S)

### P3 — Hygiene

8. ✅ **Property tests** — *done.* `proptest` covers graph invariants under random edit sequences
   (`tests/property_invariants.rs`), edit determinism, parser ingestion (`tests/integration_ccos.rs`),
   and snapshot round-trips (`tests/snapshot_roundtrip_property.rs`).
9. ✅ **Result-returning CLI commands** — *done.* A `CliError { code, message }` / `CliResult`
   abstraction threads `Result` through all 17 command handlers; `main()` maps the final result to
   the exact exit code (status-only failures keep their printed report, error messages print once).
   Exit-code behaviour byte-identical (`tests/cli.rs` asserts it). See `src/main.rs`.

### P4 — Premium / enterprise licensing (zero-knowledge, offline)

The Pro tier is a **locally-verified signed token** — no network, no telemetry; a customer runs
air-gapped. The pluggable `LicenseVerifier` trait + `Licensing` runtime gate (`require`/`allows`)
gate only advanced operator tooling; the **core** (ingestion, causal graph, Q-Page) is never gated
or degraded. See `src/license.rs` and `docs/DEPLOYMENT.md` §4.

- ✅ **Slice 29a — the gate + offline ed25519 verifier.** Tiers, the `Feature` set, the
  fail-closed all-zero placeholder key, `load_license_blob` (the one I/O entry), and
  `Ed25519Verifier` behind the `license` cargo feature. The default build pulls in no cryptography.
- ✅ **Slice 29b — the three Pro behaviors.** `CustomAuthorityWeights`, `TensionVisualization`,
  and `AuditReports`, built and gated through `require()`; `set_custom_authorities` is the gating
  precedent (`replay == live` — the final weight is what is logged).
- ✅ **Slice 29c — post-quantum SLH-DSA verifier (`license-pq`).** A second, independent offline
  verifier (NIST FIPS 205, SLH-DSA-SHAKE-128s: 32-byte pk, 7,856-byte sig) alongside ed25519,
  behind the orthogonal `license-pq` feature. A `slhdsa.` scheme tag dispatches `Licensing::detect`
  and is bound into the signed message (no cross-scheme replay); both features compose in one build.
  Uses `lattice-slh-dsa` (RustCrypto's `slh-dsa` pins a pre-release `signature` that conflicts with
  `ed25519-dalek`); **unaudited** — see `docs/DEPLOYMENT.md` §4b.
- ✅ **SLHAv2-embeddings gating.** The repo's grouped/adaptive INT4 quantization (the "SLHAv2
  two-level INT4") is now `Feature::SlhAv2Embeddings` — Pro keeps the grouped scheme, community
  falls back to uniform INT4. Decided at session open from the host tier; runtime-only, so
  `replay == live` holds.

---

### Suggested order

~~`P0.1 (syn)`~~ ✅ → ~~`P1.2 (canonical log)`~~ ✅ → ~~`P1.3 (call graph, Slices 1–3a)`~~ ✅ →
~~`P2.5 (benches)`~~ ✅ → ~~`P2.6 (analyze extras)`~~ ✅ → ~~`P3.8 (property tests)`~~ ✅ →
~~`P2.4 (config)`~~ ✅ → ~~`P3.9 (Result CLI)`~~ ✅ → ~~`P1.3 data-flow edges (Slices 1–2)`~~ ✅ →
~~`call-graph polish (renamed imports, cross-impl self-calls)`~~ ✅ → **`#13 spectral`**
(eigenvector-centrality first slice ✅; spectral regions + temporal tensor deferred to a design pass).
Call-graph receiver inference is **complete** (typed idioms #23/#126 + Slice 4 field/chain
receivers; dynamic trait-object dispatch skipped by design). **All P2/P3 quick wins are done.**

### Design pass — SciRust-distilled causality / security / time-travel

The deferred "temporal tensor + scirust work" design pass is written up in
[`docs/PROPOSAL_scirust_enhancements.md`](docs/PROPOSAL_scirust_enhancements.md):
a prioritized set of enhancements that distill mature SciRust algorithms (SCM
`do()`, RTS Kalman smoother, CUSUM/Mann-Kendall changepoints, conformal
calibration, IBP certificates, MinHash-LSH, DTW) into CCOS's three moat axes —
each with an exact integration point, a determinism (`replay == live`) analysis,
and the structural RAG-differentiation it buys. Tier-1 quick wins: causal-of-drift
attribution (`cause`/`blame`), convergent multi-hop belief propagation
(`Op::Propagate`), RTS belief retrodiction, and a certified injection-verdict
robustness radius. All respect the "distill, don't link" + off-by-default
discipline; none touch the default-build byte-identity.
