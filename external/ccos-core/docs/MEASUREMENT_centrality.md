# The centrality scoring term vs COLD-tier retention — does β·centrality earn its keep?

> Reproduce: `cargo run --release --example centrality_retention`.

`compute_node_score` carries an **off-by-default** structural term,
`w_centrality · ln(1 + resident_in_degree)`, so a *hub* — a shared module many resident
nodes depend on — can be retained even when it is not the most recently touched thing.
This is the one refinement the "prune the causal graph by causal weight" discussion
pointed at that CCOS had **shipped but not measured**. Measure-first: build a realistic
re-engagement workload under paging pressure and see whether enabling centrality actually
reduces page-faults on the hubs, and at what cost.

## The workload (deterministic — no RNG)

`R = 4` regions, each a hub + `W = 10` leaves that `DependsOn` it (global hub in-degree
10). The agent works one region at a time across `P = 2` passes; within a region it
sweeps a **sliding window** (width 3, advancing by 1 so windows overlap → leaves are
revisited) and **re-consults the hub every `cadence` leaf-accesses**. An access to a cold
node is a **page-fault** (`page_in`); to a resident node, a **hit** (`touch`, which
refreshes recency/access exactly as a fault does, so recency is a fair baseline). The
agent re-asserts the `leaf → hub` edge while both are resident (paging archives incident
edges on demote), keeping the *resident* causal graph faithful — which is the exact
resident in-degree the term scores on. We instrument that in-degree to confirm the
mechanism is engaged, not assumed.

## Results (budget 5 unless noted)

**1 — magnitude vs on/off.** Any positive weight gives the *same* result; `0.01 ≡ 1.0`:

| w_centrality | hub faults | leaf faults | TOTAL | avg resident in-degree |
|-------------:|-----------:|------------:|------:|-----------------------:|
| 0.00         | 8          | 64          | 72    | 2.77                   |
| 0.01         | 7          | 64          | 71    | 2.92                   |
| 0.30         | 7          | 64          | 71    | 2.92                   |
| 1.00         | 7          | 64          | 71    | 2.92                   |

Eviction is a discrete argmin: a positive weight reorders **one** bottom-of-heap decision
and that is the whole effect — larger weights change nothing. Paging bounds the hub's
*resident* in-degree to ~3 (of a global 10), so the bonus is a self-limited nudge, never a
hammer.

**2 — centrality targets hub faults; the colder the hub, the bigger the win:**

| cadence (hub every N) | hub faults w0 → w0.3 | hub miss-rate w0 → w0.3 | total w0 → w0.3 |
|----------------------:|:--------------------:|:-----------------------:|:---------------:|
| 2                     | 8 → 7                | 10% → 9%                | 72 → 71 (−1)    |
| 4                     | 8 → 7                | 20% → 18%               | 72 → 71 (−1)    |
| 6                     | 8 → 7                | 33% → 29%               | 72 → 71 (−1)    |
| 8                     | 16 → 7               | 100% → 44%              | 80 → 71 (−9)    |

When the hub is consulted rarely (cadence 8) the baseline evicts-and-re-faults it on
*every* consult (100% miss); centrality holds it in via its resident dependents and cuts
that to 44% — a −9 total-fault swing.

**3 — memory pressure: centrality never costs leaf faults:**

| budget | avg in-deg | hub faults w0→w0.3 | leaf faults w0→w0.3 | total w0→w0.3 |
|-------:|-----------:|:------------------:|:-------------------:|:-------------:|
| 4      | 2.30       | 15 → 7             | 64 → 64             | 79 → 71 (−8)  |
| 5      | 2.92       | 8 → 7              | 64 → 64             | 72 → 71 (−1)  |
| 8      | 3.65       | 8 → 7              | 64 → 64             | 72 → 71 (−1)  |
| 11     | 3.67       | 8 → 7              | 64 → 64             | 72 → 71 (−1)  |

Leaf faults are **identical** across the whole range: the slot centrality spends on the
hub comes from a leaf that was the next eviction anyway. The tighter the budget the bigger
the hub-fault win (budget 4: 15→7). It is fully inert only once the whole graph fits
resident.

## Verdict

Enabling centrality is a **small, consistent, low-risk** retention win on re-engagement
workloads: it cuts hub page-faults (most when the hub is cold-ish or memory is tight),
costs **no** leaf faults, is **binary** in `w` (so there is nothing to fine-tune — it is
effectively a boolean), and **self-limits** via resident in-degree (it cannot hoard hubs
you have stopped using — leave a region and its leaves demote, dropping the bonus to ~0).

It therefore stays **OFF by default**: the gain is real but workload-dependent and modest,
and off keeps snapshots/replay byte-identical (`w_centrality == 0` skips the in-degree map
entirely and is `serde`-elided). The log-tuner (`AgentSession::tune_recall_weights`) can
switch it on where a hub-heavy access pattern makes it worthwhile — and this measurement is
what tells it the switch is safe (no leaf-fault regression) rather than a guess.

This is the structural-retention companion to the recall measurements: the same "prune by
causal weight" intuition, measured on the *eviction* side instead of the *retrieval* side.
