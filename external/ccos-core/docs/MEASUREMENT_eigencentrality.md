# Eigenvector vs in-degree centrality — where does the global signal actually pay off?

> Reproduce:
> `cargo run --release --example eigencentrality_retention`
> (the signals' disagreement on the full graph is also pinned by the unit test
> `memory::tests::eigenvector_centrality_captures_recursive_importance_indegree_misses`).

The centrality score term (off by default, `w_centrality`) had one mode: `ln(1 + in_degree)`
— a **local** count. This adds a second, [`CentralityMode::Eigenvector`]: **eigenvector
centrality** by deterministic damped power iteration (the Katz/PageRank form — pure `Ax = λx`
is degenerate on a code graph, which is largely a DAG). It is *global* and recursive: a node
is important if *important* nodes depend on it. This is the first brick of the "scirust"
direction — using linear algebra over the AST topology to compute structural importance.

Two questions, measured separately.

## 1. Do the signals actually differ? (Yes — on the full graph.)

Unit test, on a hierarchy `15 leaves → 3 mids → 1 top`:

| node | raw in-degree | eigenvector |
|------|:-------------:|:-----------:|
| a mid | 5 | 0.36 |
| **top** | **3** | **1.00** |

In-degree ranks a **mid above top** (5 > 3); eigenvector ranks **top above the mids**, because
the mids' importance flows into the core they depend on. This is the case in-degree gets wrong,
and it is exactly what a "structural pillar" is.

## 2. Does that help *eviction* retention? (No — paging masks it.)

`examples/eigencentrality_retention.rs` runs a deterministic hierarchical re-engagement
workload and counts page-faults on `top` under each mode, across resident budgets:

| budget | in-degree top-faults | eigenvector top-faults | Δ |
|-------:|:--------------------:|:----------------------:|:--:|
| 3 | 9 | 9 | 0 |
| 4 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 |

**Identical retention** (eigenvector is in fact marginally worse on *total* faults). The reason
is structural: **eviction scores the resident window, and paging keeps only a thin slice of the
hierarchy resident.** While the agent works one region, `top`'s *resident* in-degree is ~1 (one
mid resident), so the local and global signals see almost the same slice — the recursive
advantage that needs the *whole* `leaves → mids → top` fan-in is simply not present in RAM at
eviction time.

## Verdict — a negative result that aims the next brick

Eigenvector centrality is correct, deterministic, and genuinely more discriminating than
in-degree — **on the full graph**. But wiring it into per-tick **eviction** buys nothing,
because eviction acts on a thin resident slice where the difference collapses. So:

- **Keep it off the eviction path.** `CentralityMode::Eigenvector` ships (off by default, tested,
  serde-elided so snapshots are byte-identical) as the reusable primitive
  [`MemoryGraph::eigencentrality`], not as a recommended eviction knob.
- **The payoff is the full persistent topology**, where the whole `leaves → mids → top` structure
  is visible: **region formation** (spectral clustering of the Laplacian), **offline pillar
  ranking**, and the temporal **3D tensor** (centrality trajectory across ticks). That is where
  the next scirust brick should compute centrality — over resident **+** cold edges (the COLD tier
  already keeps a reverse-adjacency index), not the resident window.

This is the same measure-first discipline as the recall/RAG result: build the capability, measure
where it helps, and let the measurement — not the intuition — decide where to invest next.

## Follow-up: the full graph, on real code (`examples/pillar_ranking.rs`)

The verdict above said the eigenvector payoff is the **full** graph (where the whole hierarchy is
visible), not the resident slice. So the next test runs it there: parse CCOS's *own* source into one
fully-resident causal graph (~1.8k nodes), and rank the **file** nodes by in-degree vs eigenvector
centrality — scored against an objective ground truth, each file's **transitive-dependent count**
(`query::source_set`, i.e. how much of the system depends on it).

| signal | Spearman vs. transitive dependents |
|--------|:----------------------------------:|
| in-degree (local) | **0.936** |
| eigenvector centrality (global) | 0.892 |

Both predict the real pillars well, and the two rankings are nearly identical (top files:
`memory.rs`, `util.rs`, `event_log.rs`, `context_region.rs`). But the **local in-degree predicts
structural importance *slightly better* than the global eigenvector** — the spectral sophistication
does not pay off even on the full graph. The reason is structural: a real code dependency graph is
**shallow** (importer→imported chains are short), so importance is overwhelmingly *direct*, and the
eigenvector's recursive weighting adds variance rather than signal.

**Two independent measurements now agree:** the rank-2 spectrum does not beat a simple count on
CCOS — once at eviction (tie), once at full-graph pillar ranking (in-degree wins, 0.936 vs 0.892).
That is the honest baseline any higher-rank / "tensor" elaboration has to beat to earn its
complexity. It does not close the door on the direction, but it sets the bar: a new dimension must
**measurably** out-predict in-degree before it ships.

[`CentralityMode::Eigenvector`]: ../src/memory.rs
[`MemoryGraph::eigencentrality`]: ../src/memory.rs
