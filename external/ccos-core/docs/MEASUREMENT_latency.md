# Recall latency — measured, then fixed (the perf pass)

> Reproduce: `cargo run --release --example recall_latency`
> (add `--features learned-embed` for the LSA numbers).

Step 3 was **measure first, optimise what the data points at**. The micro-benchmark
times each recall strategy at growing corpus sizes.

## What the measurement found

Latency was **super-linear** in corpus size for the query strategies, and the cause
was *per-recall reconstruction of derived structures*:

- `around` / `task` rebuilt the **entire region clustering** (`initialize_regions`
  over the whole graph) on every call — the dominant cost, ~`O(V^1.6)`.
- `semantic` / `hybrid` additionally **re-fit the TF-IDF store** (and, under
  `learned-embed`, re-ran the LSA eigensolve) over all nodes on every call.

Per recall, µs/call, **before** (default INT4 TF-IDF):

| nodes | working_set | around | task | semantic | hybrid |
|------:|------------:|-------:|-----:|---------:|-------:|
| 200   |        279  |  2383  | 172  |   4760   |  4991  |
| 800   |       1208  | 24624  | 730  |  34041   | 35118  |
| 2000  |       3350  | 74747  | 1302 |  93213   | 96517  |

So an `around` recall (the **primary** workspace recall) cost **75 ms** at 2000 nodes,
and a hybrid recall **97 ms** — and worse super-linearly.

## The fix

`CcosMemory` now memoises the two per-recall derived structures behind a **graph
version counter** bumped on every resident-graph mutation (ingest / failure / tick /
page / re-cap / re-weight):

- `region_cache` — the `ContextRegionEngine` clustering, reused across recalls until
  the graph changes.
- `embed_cache` — the fitted embedding store, likewise.

Invalidation **over-approximates** (the version bumps even for a change that doesn't
affect a given cache), so a cache is *never* stale — and because the result is
byte-identical to a fresh rebuild, **determinism and `replay == live` are preserved**
(a regression test asserts a post-warm ingest is visible to the next recall; the full
replay/determinism suite still passes).

Per recall, µs/call, **after** (default):

| nodes | working_set | around | task | semantic | hybrid |
|------:|------------:|-------:|-----:|---------:|-------:|
| 200   |        289  |     4  | 185  |    246   |   510  |
| 800   |       1359  |     8  | 747  |    988   |  2459  |
| 2000  |       3665  |    13  | 1534 |   2236   |  4498  |

Speed-ups at 2000 nodes: **around ~5700×** (75 ms → 13 µs), **semantic ~42×**,
**hybrid ~21×**.

**Honest reading:** the benchmark issues repeated recalls between mutations, so most
calls are cache *hits* — the realistic steady state (ingest a batch, then recall many
times). The **first** recall after any mutation still pays the rebuild; the win is on
the repeated recalls, which is the common pattern. `working_set`/`task` are unchanged
(they never built the cached structures).

## Still deferred (honest)

Two audit-pass-4 items remain, both confined to **opt-in / scale** paths, so they do
not affect the default hot path the fix above addresses:

- **Per-ingest `O(cold)` budget re-scan** — only runs when a spill store or compaction
  budget is *attached* (the enforcers early-out otherwise). Worth incremental counters
  if/when spill is used at scale.
- **`cold_neighbours` `O(cold·edges)` per around-recall** — only with a populated COLD
  tier; the region cache already removed the larger around cost.
