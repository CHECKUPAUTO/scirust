# Node lifecycle state — separating *health* from *topology*

> Reproduce: `cargo run --release --example node_lifecycle`

A node has two unrelated properties: where it sits in the dependency graph (**topology**) and
what state it's in (**lifecycle** — verified, under-edit, or dead). A single 2-D graph conflates
them, and the conflation pollutes two signals. `NodeState` (`Stable` / `Working` / `Orphan`) is the
fix — and the right shape for it is a **per-node enum field**, not a tensor dimension: a node's
state is *single-valued* (it is one state at a time), so it is a label, not an axis to contract
over.

## What it fixes (measured on a controlled graph)

A `pillar` module is depended on by 6 **real** nodes and 6 **dead** nodes (dead code that still
references it).

**1. Centrality pollution.** Dead dependents inflate the pillar's structural weight:

```
pillar in-degree: 12 (dead code counted)  →  6 (orphans excluded)
```

Labeling the dead nodes `Orphan` drops them from the centrality signal (in-degree *and*
eigenvector), so the pillar's score reflects its *real* load-bearing role.

**2. Eviction pollution.** The dead code was edited and then abandoned, so it is **fresh** (high
recency) while the real working set has aged. A recency-driven policy keeps the dead code:

| budget 7, 6 aged-real + 6 freshly-edited-dead | real nodes retained | dead nodes resident |
|---|:---:|:---:|
| all `Stable` (baseline) | **1 / 6** | 6 |
| dead labeled `Orphan` | **6 / 6** | 0 |

Without the label, fresh dead code squats in memory and evicts 5 of 6 real nodes. With it, the
`Orphan` is driven to the bottom of the eviction order **regardless of recency**, freeing every
slot for real work. (`Working` is the dual, not shown here: pinned resident as the current focus
*even as its recency decays*.)

## Why a field, not a tensor

This is the concrete answer to "should lifecycle be an extra tensor dimension?" — **no.** A tensor
axis is for a *multi-valued relation you contract over*; a node's lifecycle is *single-valued*, so
it is an attribute. Keeping it as a field (next to `recency`, `failure_relevance`, `access_count`)
and combining it with the topology *at compute time* (mask `Orphan` out of centrality; bias the
eviction score) gives the separation cleanly, in O(1) per node, type-checked, with **zero** new
data structure. Off by default (`Stable`): the centrality, the score, and the serialized snapshot
are all byte-identical until a state is set. Deterministic; `set_node_state` invalidates the
centrality caches so the signal stays consistent.
