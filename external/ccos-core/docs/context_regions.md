# CCOS Context Region Engine (v0.3)

> From a 1-D scored list of files to a **spatial map of causal regions** that an
> agent hydrates as working sets.

- [Why a spatial memory](#why-a-spatial-memory)
- [Mental model](#mental-model)
- [Architecture & causal flow](#architecture--causal-flow)
- [Formal definitions](#formal-definitions)
- [Dynamic admission policy](#dynamic-admission-policy)
- [Determinism & replay](#determinism--replay)
- [CLI](#cli)
- [Measured locality](#measured-locality)

## Why a spatial memory

CCOS v0.2 represents context as a `MemoryGraph`: files/symbols are nodes with a
scalar causal score, and the kernel pages in the top-scoring nodes. That is a
**1-D** view — a ranked list.

A real LLM working context is not a list of files; it is a *region* of knowledge:
files that are causally close, their dependencies, their associated errors, a
temporality, an activity level, an importance, a memory **temperature** and a
cognitive **pressure**. The Context Region Engine makes that explicit. An agent
no longer "loads files" — it **hydrates a region**.

## Mental model

Every node is embedded in an abstract 3-D context space:

```
X = structural proximity   (same file / module cluster together)
Y = causal proximity       (failure-relevance, dependency involvement)
Z = temporality            (recency of access)
```

The memory becomes a map of regions:

```
            REGION ACTIVE (hot)                 REGION COLD
        auth.rs                                README.md
            │                                  docs/
   sorter.rs ── logger.rs
```

A hot region (recent edits, propagated failures) is paged in first; a cold one is
evicted.

## Architecture & causal flow

The engine is a new layer between the causal graph and the LLM — it does **not**
replace the graph, the event log, the guard or the incremental builder:

```
Raw code → AST parser → MemoryGraph → ContextRegionEngine → LLM context window
                              │              │
                       (causal scores)  (regions, temperatures, policy)
```

Files (CCOS source):

- `src/context_region.rs` — `ContextPoint`, `ContextRegion` (the data model).
- `src/region_engine.rs` — `ContextRegionEngine`: clustering, activation,
  cooldown/eviction, deterministic `replay_from`.
- `src/context_policy.rs` — `ContextPolicy`: the dynamic admission score.
- `src/region_metrics.rs` — flat-vs-region locality measurement.
- `event_log.rs` — `RegionCreated / RegionActivated / RegionMerged /
  RegionEvicted / ContextWindowGenerated` events.

## Formal definitions

**Causal distance.** Let `G = (V, E)` be the causal graph with edge weights
`w(e) ∈ (0, 1]`. Treat `G` as undirected and assign each edge a cost
`c(e) = −ln w(e) ≥ 0` (a stronger causal link ⇒ a shorter distance). The causal
distance `d(u, v)` is the minimum total cost over paths from `u` to `v`
(`∞` if disconnected). The **k-hop causal neighbourhood** of a node `t` is
`N_k(t) = { v ∈ V : hops(t, v) ≤ k }`.

**Region membership.** Collapse external dependency hubs (`dep:*`) — every other
node is first keyed by its owning file. Define a relation `a ∼ b` on file keys:
two files are *directly linked* iff there is an edge between a node of one and a
node of the other that is **not** a shared external dependency. A **region** is a
connected component of the reflexive-transitive closure `∼*`:

> Two nodes belong to the same region iff their files are in the same connected
> component of the cross-file causal-link graph (external hubs excluded).

This yields one region per file by default, and a **merged** multi-file region
whenever files are genuinely linked (a dependency edge, a propagated failure).
The partition is a **pure function of the graph**: connected components are
computed by a sorted BFS, so the result is independent of `HashMap` order.

**Region scalars.** For a region `R` with member nodes `M`:

```
total_score(R) = Σ_{n∈M} score(n)
temperature(R) = clamp( mean_{n∈M} heat(n), 0, 1 )
   heat(n)     = 0.5·score(n) + 0.3·failure(n) + 0.2·recency(n)
density(R)    = |{ e∈E : both endpoints ∈ M }| / |M|     (internal edges / member)
```

On `src/`, regions reach an **average density of 0.955** — they are almost
entirely internally connected, i.e. genuine causal clusters.

## Dynamic admission policy

The static `paging_threshold = 0.6` becomes dynamic. With `u` = fraction of the
token budget used and `κ` = task complexity:

```
threshold(u, κ)      = clamp( 0.6 + 0.3·u − 0.2·κ , 0.05, 0.95 )
admission_score(R)   = 0.55·temperature(R) + 0.30·squash(density(R)) + 0.15·κ
                       where squash(d) = d / (1 + d)
admit(R)  ⇔  admission_score(R) ≥ threshold(u, κ)
```

A hot, cohesive region can clear the bar even when the static 0.6 would reject
it; a full window raises the bar so only the hottest regions get in.

## Determinism & replay

Clustering is a pure function of the graph; activation advances a **logical
clock**, never wall-clock time. Therefore a session replays bit-for-bit:

1. rebuild the graph from the event log (`GraphReconstructor`);
2. re-cluster (`ContextRegionEngine::replay_from`) — identical base regions;
3. re-apply the recorded `RegionActivated` / `RegionEvicted` events.

`tests/context_region_tests.rs` asserts `engine == replay_from(rebuilt_graph,
log)` and that 10 000 activate/cool cycles cause **no drift** in region count or
temperatures.

## CLI

```bash
# Cluster a tree into regions (sorted by temperature).
cargo run -- regions src

# Hydrate the context window for a node's region.
cargo run -- regions src --activate file:src/memory.rs

# Flat (v0.2) vs region (v0.3) locality for a target node.
cargo run -- regions src --metrics sym:src/memory.rs:MemoryGraph --radius 2 --json
```

## Measured locality

`scripts/region_benchmark.sh` compares, on a tree, the **flat** strategy (page in
the globally highest-scoring nodes — what v0.2 does) against the **region**
strategy, using each target's 2-hop causal neighbourhood as ground truth. On
CCOS's own `src/` (705 nodes, 822 edges, 26 regions):

| Strategy | causal precision | causal recall |
| -------- | ---------------- | ------------- |
| flat (v0.2)   | 0.021 | 0.347 |
| **region (v0.3)** | **0.057** | **0.972** |

Region selection covers **97%** of a task's causal neighbourhood vs **35%** for
flat, using **≈48% fewer tokens** to reach equal coverage. Absolute precision is
low because the v0.2 parser emits only containment/import edges (shallow
neighbourhoods); richer **semantic edges** (call graph / data flow, roadmap
P1.3) would tighten `N_k` and amplify the precision gain.

## Testing the hypothesis

> *Does an agent with regional causal memory solve long, multi-file tasks better
> than a classical RAG agent?*

The full answer needs LLM rollouts, but its **necessary condition** is testable
now, deterministically, without an LLM: *an agent cannot solve a task whose
required causal context is absent from its window.* `ccos experiment` runs that
test — modular synthetic repos, cross-file causal tasks of growing **diameter**,
six strategies at equal budget, oracle success = "required causal set ⊆ window".
Crucially it runs **two scenarios**: a **clean** query (points at the target) and
a **noisy** query (a decoy out-scores the target lexically — the realistic case
where the task description doesn't name the distant cause). RAG/GraphRAG locate
code from the *query*; CCOS anchors on the *workspace signal* (the active file).

```bash
cargo run -- experiment --tasks 800
```

| strategy | clean (d=1·2·3·4) | **noisy (d=1·2·3·4)** |
| -------- | ----------------- | --------------------- |
| `rag-dense` (lexical RAG) | 0 · 0 · 0 · 0 | 0 · 0 · 0 · 0 |
| `rag-hybrid` | 1 · 0 · 0 · 0 | 1 · 0 · 0 · 0 |
| `graphrag-1hop` | 1 · 0 · 0 · 0 | 0 · 0 · 0 · 0 |
| `graphrag-bfs` (strong graph baseline) | 1 · 1 · 1 · 1 | **0 · 0 · 0 · 0** |
| `ccos-from-query` (ablation: CCOS trusts the query) | 1 · 1 · 1 · 1 | **0 · 0 · 0 · 0** |
| **`ccos-region`** (anchored on workspace) | **1 · 1 · 1 · 1** | **1 · 1 · 1 · 1** |

**Honest reading:**

1. Lexical RAG *fails entirely* on cross-file causal tasks (0%) — the hypothesis's
   premise validated.
2. **Clean query:** `graphrag-bfs`, `ccos-from-query` and `ccos-region` all reach
   100% — the lever is causal **structure**, *not CCOS specifically*.
3. **Noisy query:** everything that locates code lexically collapses to **0%** —
   including the strong `graphrag-bfs` *and* the `ccos-from-query` ablation. Only
   **`ccos-region`**, anchored on the workspace signal rather than the query,
   survives. The ablation isolates the differentiator: it is the **anchor source**
   (a structural workspace signal vs. a lexical query), not the region machinery.
4. **Assumptions:** this credits CCOS with a reliable anchor (active file / failing
   test) — which an OS-style memory has but a retrieval pipeline doesn't — and
   needs **modular** repos (a giant region over budget collapses it).

This is a **simulation under a stated oracle** (necessary condition), not an LLM
evaluation (sufficient condition). See [`paper/`](paper/) for the formal model,
the determinism proof, and the proposed real-LLM comparison against RAG / GraphRAG
/ MemGPT / LangGraph.
