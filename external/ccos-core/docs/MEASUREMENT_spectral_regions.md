# Spectral (Fiedler) clustering for regions — tensor brick 2

> Reproduce: `cargo run --release --example spectral_regions`

The eigencentrality measurement (`docs/MEASUREMENT_eigencentrality.md`) pointed the "tensor"
direction at the **full persistent topology** — and named **region formation by spectral
clustering of the Laplacian** as the next thing to test. This is that test.

## Method

Build CCOS's **file-dependency graph** (the file→file edges from import resolution, made
symmetric — 41 files, 134 edges). Partition it by **recursive Fiedler bisection**: the
2nd-smallest eigenvector of the Laplacian `L = D − A` (classic spectral graph theory), computed
by deterministic damped power iteration on `(σI − L)` with the constant eigenvector deflated,
recursing to ≤ 8 clusters. Score the partition by Newman **modularity** `Q` and compare to
honest baselines.

## Result

| clustering | modularity Q |
|------------|:------------:|
| **spectral (Fiedler, 8 clusters)** | **0.085** |
| name-prefix grouping (naive structural) | −0.054 |
| random partition (avg of 50, seeded) | −0.047 |
| one cluster | 0.000 |

Two honest readings, both true:

1. **The spectral cut is real signal, not noise** — it clearly beats the naive baselines
   (0.085 vs ≈ −0.05).
2. **But the structure it finds is weak.** `Q = 0.085` is far below the ~0.3 that marks strong
   community structure, and the clusters show why: **23 of 41 files collapse into one
   densely-coupled core** (memory, event_log, external_memory, parser, query, region_engine,
   …). CCOS's kernel is small and tightly interconnected — there are no clean modular "regions"
   for a spectral method to carve out and page on.

## Verdict

The method is sound and deterministic; the *graph* lacks the modularity to exploit. A larger,
more loosely-coupled codebase could have real communities — but on CCOS itself, spectral regions
do not earn their complexity.

This is the **third consistent measurement** of the tensor/spectral direction, and they line up:

| brick | question | result |
|-------|----------|--------|
| 1a — eigencentrality (eviction) | retain pillars better than in-degree? | tie (paging masks it) |
| 1b — pillar ranking (full graph) | predict pillars better than in-degree? | no — in-degree wins, 0.936 vs 0.892 |
| 2 — spectral regions | find modular regions to page on? | no — Q 0.085, graph too interconnected |

Each tensor elaboration so far **beats a trivial baseline but not the simple count / not by
enough to earn its cost on CCOS**. That is the honest bar: a higher-rank or richer-dimension
method has to clear *this*, measurably, before it ships. Measure-first keeps the door open
without paying for sophistication that, so far, the data does not reward.
