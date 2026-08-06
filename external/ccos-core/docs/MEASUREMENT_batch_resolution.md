# Batch resolution — deferring the whole-graph passes turns O(N²) ingestion into O(N)

> Reproduce: `cargo run --release --example ingest_profile`

The ingestion profiler (`docs/` companion to `examples/ingest_profile.rs`) localised the cost of
turning source into the causal graph. Parsing (the `syn` AST) is cheap; the cost is the three
**whole-graph resolve passes** — `link_module_imports` (imports → `DependsOn`/`Contains`),
`resolve_symbol_calls` (`Calls`), `resolve_data_flow` (`DataFlow`). The first fix, **B1**, made
`add_edge` dedup O(1) instead of an O(E) linear scan, dropping ingestion from ~cubic to ~quadratic.
The remaining quadratic is structural: **per-file ingestion re-runs all three whole-graph passes
after every file**, so a batch of N files pays N resolutions of an O(N) graph.

This note records **B2-batch**: the passes are *order-independent pure functions of the final node +
pending-ref set*, so running them **once at the batch boundary** instead of per file collapses the
cost to a single O(N) pass — and what it revealed about resolution **semantics**.

## The measurement

`examples/ingest_profile.rs`, synthetic corpus (20 fns/file, cross-file `use` + call + shared const),
no paging. "Per-file" re-runs the three passes after each file (the historical `ingest_source`
pattern); "B2-batch" ingests every file first, then resolves once:

```
# Scaling — whole-graph resolve re-run per file (the real ingest pattern)
  files   resolve total(ms)   ratio when files ×2
    150             790.3     —
    300            3160.7    4.00      ← ×4 per doubling  ⇒ O(N²)
    600           15596.0    4.93

# B2-batch — defer, then resolve ONCE after all files
  files   batch resolve(ms)   ratio when files ×2
    150              13.4     —
    300              32.5    2.43      ← ×~2.5 per doubling ⇒ O(N)
    600              89.5    2.75
```

At 600 files the batch resolves in **89.5 ms vs 15,596 ms — ~174× faster**, and the scaling is linear
(~×2.5 per doubling) instead of quadratic (~×4.9). The win grows with N, exactly as the
O(N²)→O(N) shape predicts.

`CcosMemory::ingest_deferred` + `CcosMemory::resolve` expose this directly: `ingest_deferred` records a
file and marks resolution pending (no passes); `resolve` runs the three passes **once** and clears the
flag (idempotent and near-free when clean). The eager `CcosMemory::ingest_source` is unchanged — it is
`ingest_deferred` + `resolve`, so a single-file ingest still leaves a fully-resolved graph (every
`&self` reader — `recall`, serialise — sees resolved edges, the contract the whole test suite relies
on). A `debug_assert` in `recall`/`to_json`/`checkpoint` turns any future "deferred ingest, then read
without resolve" into a loud failure.

## B2-full — order-independent resolution (landed): eager ≡ batch

Deferring resolution *used to* change the answer, not just the speed. Resolution is
**resolve-uniquely-or-skip**, and the original passes were **add-only** (they never removed an edge),
so a name globally-unique when a caller is ingested, then made ambiguous by a later file, left an
order-dependent stale edge:

```
src/a.rs       pub fn target() -> i32 { 1 }      // defines target
src/caller.rs  pub fn run() -> i32 { target() }  // calls target — unique *now*
src/b.rs       pub fn target() -> i32 { 2 }      // a SECOND target — now ambiguous
```

Old eager (per-file, add-only) kept `run → a::target` (resolved while `target` was unique, never
removed); only the batch saw the final ambiguity and dropped it. The two graphs differed.

**The fix** (`MemoryGraph::resolve_all`): *prune the resolution-owned edges, then rebuild from the
final state* on every resolve. A name that became ambiguous is simply not re-linked, so eager
(per-file) and batch (deferred) — and a replay that re-ingests the same files — all converge on the
**same** graph. `eager_and_batch_agree_under_late_ambiguity` verifies eager now drops the edge too.

**Selective prune — the `serde(skip)` constraint.** `pending_calls`/`pending_data_refs` are runtime-
only, so they are **empty after a checkpoint load**: a `Calls`/`DataFlow` edge of a loaded file can
*not* be rebuilt. So the prune has two tiers:

| Edge type           | Created by                                   | Pruned by `resolve_all`?                         |
|---------------------|----------------------------------------------|--------------------------------------------------|
| `Calls`             | `resolve_symbol_calls` only                  | only if the source file has pending refs (rebuildable) |
| `DataFlow`          | `resolve_data_flow` only                     | only if the source file has pending refs (rebuildable) |
| `DependsOn`         | parser (`file:→use:`, `use:→dep:`) + imports | always, but only `file:→file:` (rebuilt from durable nodes) |
| `Contains`          | parser (`file:→mod:/sym:`) + module hierarchy| always, but only `file:→file:` (rebuilt from durable nodes) |
| `Supports`/`Contradicts` | assertion path                          | never                                            |

Import/hierarchy edges rebuild from the **durable** `file:`/`use:` node set, so they are always pruned
and re-derived (loss-free even after load). `Calls`/`DataFlow` are pruned **only** for files whose
pending refs are present (this session, or a replay re-ingest); a loaded file with no pending refs
keeps its edges. `checkpoint_load_then_ingest_keeps_loaded_call_edges` proves a loaded call edge
survives a later ingest+resolve.

**`replay == live` stays exact.** Replay loads the same baseline and replays the same ops, so it has
the same pending-ref-presence pattern as live — selective prune behaves identically, and the converged
graphs match. This is why the replayable path may now batch safely (the follow-up below). The full
suite — including the `replay == live` and snapshot round-trip property tests — is green.

## Batched replay/agent path (landed): O(N) time-travel

The semantic blocker being gone, the O(N) batch now extends to the replayable path. `replay_to` (the
engine behind time-travel debugging, `recall_what_if`, and the counterfactual `retrieval_reward`)
used to call the eager `ingest_source` for **every** `Ingest` op — N resolutions of an O(N) graph,
**O(N²)** per reconstruction. It now **defers** every ingest and runs the resolve passes **once**:
before each op that reads cross-file edges (a recall page-in, a failure / page-fault propagation) and
once at the end. `AgentSession::ingest_batch` does the same on the live ingest path. Ingestion never
demotes to COLD (only an explicit resident-cap change does, and none is logged), so deferring the
resolve cannot reorder paging — and B2-full makes the resolved edge set order-independent — so the
reconstructed graph is byte-identical.

`examples/replay_batch_crux.rs`, synthetic cross-referencing corpus sized under the resident cap so
the timing isolates resolution from COLD-tier paging:

```
# Replay/time-travel reconstruction — eager (pre-#33) vs batched (#33)
  files   eager replay(ms)   batched replay(ms)   speedup   batched ratio ×2
    150              226.1                 18.9     12.0x       —
    300              894.9                 38.2     23.4x      2.02   ← ×2 ⇒ O(N)
    600             3689.2                 77.7     47.5x      2.03
```

Eager reconstruction is **quadratic** (~×4 per doubling — each of N ingests re-runs the O(N) resolve
passes); batched is **linear** (~×2 — one resolve for the whole log), **47.5× faster at 600 ops** and
widening with N. `tests/replay_equivalence_property.rs` proves the byte-for-byte `replay == live`
equivalence still holds for *any* interleaving of ingests, failures, recalls and page-faults — the
order-independent core (B2-full) is what makes deferring safe.

**Bottom line:** measure first. The bottleneck was algorithmic (per-file whole-graph re-resolution),
not cache layout — DOD/SoA would have shaved a constant factor off the wrong thing. Deferring the
passes to the batch boundary is ~174× at 600 files (B2-batch); making resolution prune-and-rebuild
(B2-full) then removed the eager-vs-batch divergence entirely, so the speedup is now also *correct*
and order-independent everywhere — measured, tested, and documented rather than papered over.
