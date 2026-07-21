# SRCC Robust Structural Intelligence Program

Tracking document for the nine-phase program that extends SciRust's SRCC
capabilities from deterministic robust source clustering into a broader robust
scientific-computing platform. This file is maintained incrementally: each phase
appends its design summary, merge commit, benchmark fingerprint, and limitations.

It contains no speculative marketing claims. Every strong statement is tied to a
mathematical assumption, a machine-testable property, and a deterministic
benchmark.

## Motivation

SRCC (resonant-consensus closure and projection) already performs deterministic
robust source clustering with observed-source medoids and leave-one-out stability
certificates (PRs #719, #720). To grow it into a platform that is safe to use on
real industrial data, the surrounding numerical foundation must be made:

- **robust** — location and scale summaries that resist a *minority* of grossly
  aberrant observations;
- **scale-aware** — geometry that does not silently depend on the raw magnitude or
  physical units of coordinates;
- **honest** — every strong guarantee (majority robustness, global optimality,
  affine invariance, industrial superiority) is either proven and certified, or
  not claimed.

The program is delivered as nine sequential, independently reviewable pull
requests (referred to as phases 721–729). Later phases build only on merged
earlier phases.

## Program-wide invariants

These hold in every phase and are re-checked in each PR's self-review:

- **Pure Rust.** No Python/C/C++ FFI, no BLAS/LAPACK FFI, no network access in
  library code or tests.
- **Deterministic.** Fixed accumulation order, explicit seeds, canonical sorting
  by `f64::total_cmp`, documented tie-breaking, no thread-scheduling-dependent
  reductions. Bit-identical benchmark output verified by running twice and
  comparing with `cmp` + SHA-256.
- **Numerically honest.** NaN, infinity, singularity, rank deficiency,
  non-convergence, ambiguous consensus, zero robust scale and exhausted budgets
  are surfaced as typed errors or explicit certificate fields — never hidden.
- **Backward-compatible by default.** Existing public APIs stay source-compatible;
  new behaviour is opt-in. No required public fields added to existing structs.
- **Typed errors, no `unsafe`.** Error enums use manual `Display` + `Error` impls
  (the established SciRust convention); crates keep `#![forbid(unsafe_code)]`.
- **MSRV 1.89**, nightly-2026-07-02 pinned toolchain. No `--all-features` workspace
  builds (mutually exclusive BLAS backends).

## Mathematical assumptions (as introduced per phase)

- **Robust descriptive statistics (721).** Breakdown is bounded: the median, MAD
  and IQR resist up to (in the limit) 50 % contamination; a symmetric
  `α`-trimmed/winsorized mean resists up to a fraction `α` per tail;
  median-of-means resists corruption of strictly fewer than half of its blocks.
  None of these tolerate an arbitrary numerical majority of adversarial samples.
- Later phases add: scale/affine invariance groups (722–723), robust-regression
  loss/breakdown assumptions (724), explicit identifiability assumptions for
  majority contamination (725), certified-optimality proofs vs. gaps (726),
  benchmark preregistration and no-leakage protocol (727–728), and shadow /
  promotion-gate safety (729).

## Phase dependency graph

```
721 robust descriptive stats (scirust-stats)
  └─> 722 robust multivariate geometry (scirust-multivariate + stats/units/solvers)
        └─> 723 scale-aware SRCC source geometry (scirust-srcc)
        └─> 724 robust regression (scirust-learning + solvers/stats/multivariate)
              └─> 725 trust & contamination models (scirust-srcc + estimation/spc/pdm)
              └─> 726 certified medoid clustering (scirust-solvers + srcc)
                    └─> 727 industrial benchmark harness (bench-schema + method crates)
                          └─> 728 real industrial evaluation
                                └─> 729 shadow deployment & promotion gates (scirust-mlops)
```

Integration contracts: geometry (722) reuses the 721 primitives rather than
re-implementing MAD/weighted-median/trimming; SRCC (723) consumes fitted metric
models rather than owning a second robust scaler; regression (724) uses
`scirust-solvers` QR/SVD; benchmark crates depend on method crates, never the
reverse; MLOps (729) consumes benchmark manifests/certificates and never drives
deployment from within a method crate.

## Phase status

| Phase | Title | Branch | PR | Merge commit | Status |
|------:|-------|--------|----|--------------|--------|
| 721 | Robust descriptive statistics | `claude/scirust-srcc-robust-stats-6ue9xc` | [#725](https://github.com/Memorithm/scirust/pull/725) | _(pending)_ | In review |
| 722 | Robust multivariate geometry | `feat/multivariate-robust-geometry` | — | — | Not started |
| 723 | Scale-aware SRCC source geometry | `feat/srcc-scale-aware-source-geometry` | — | — | Not started |
| 724 | Deterministic robust regression | `feat/learning-robust-regression` | — | — | Not started |
| 725 | Trust & contamination models | `feat/srcc-trust-contamination-models` | — | — | Not started |
| 726 | Certified medoid clustering | `feat/solvers-certified-medoid-clustering` | — | — | Not started |
| 727 | Industrial benchmark harness | `feat/srcc-industrial-benchmark-harness` | — | — | Not started |
| 728 | Real industrial evaluation | `feat/srcc-industrial-real-data-evaluation` | — | — | Not started |
| 729 | Shadow deployment & promotion gates | `feat/mlops-srcc-shadow-deployment` | — | — | Not started |

> Notes for phase 721:
> - The program's suggested branch was `feat/stats-robust-descriptive`, but this
>   session's fixed development branch is `claude/scirust-srcc-robust-stats-6ue9xc`.
> - GitHub assigned this phase the real number **PR #725** (numbers 721–724 were
>   consumed by other concurrent branches). The phase→number mapping is *not* 1:1:
>   "Phase 721" is tracked by GitHub PR #725. This is recorded honestly rather than
>   forcing a matching number.

---

## Phase 721 — Robust descriptive statistics

**Crate:** `scirust-stats` · **Module:** `scirust-stats/src/robust.rs`

### Design

A single new module `robust` provides the shared robust-statistics foundation for
all later phases. It deliberately **reuses** `describe::median`, `describe::quantile`
and `describe::mean` (no second median/quantile convention) and the crate's seeded
`rng::SplitMix64` (no hidden RNG).

Public surface (re-exported at the crate root and from `prelude`):

- `median_absolute_deviation(values, MadConsistency) -> Result<f64, RobustStatsError>`
  — raw `median(|xᵢ − median(x)|)`, or scaled by the documented normal-consistency
  factor `1 / Φ⁻¹(3/4) ≈ 1.4826` (applied only on request, never silently; a test
  cross-checks the literal against the crate's own audited normal quantile).
- `interquartile_range(values) -> Result<f64, RobustStatsError>` — `Q3 − Q1` using
  the existing `type-7` quantile rule.
- `weighted_median(values, weights) -> Result<f64, RobustStatsError>` — ascending
  order with index tie-break; documented lower/upper averaging when cumulative
  weight hits exactly half the total.
- `trimmed_mean` / `winsorized_mean(values, trim_fraction)` — symmetric `⌊n·α⌋`
  trimming (floor convention, `0 ≤ α < 0.5`), sharing one validated trim helper.
- `median_of_means(values, MedianOfMeansConfig)` — `Contiguous` or deterministic
  `SeededPermutation` (SplitMix64 Fisher–Yates) partition into non-empty blocks.

Errors are a single typed enum `RobustStatsError` (`EmptyInput`, `NonFiniteValue`,
`NonFiniteWeight`, `NegativeWeight`, `ZeroTotalWeight`, `LengthMismatch`,
`InvalidTrimFraction`, `InvalidBlockCount`, `TooManyBlocks`) with manual `Display`
+ `std::error::Error` impls. No non-finite result is ever returned silently.

### Determinism contract

Canonical `f64::total_cmp` sorting with original-index tie-breaks; weighted-median
total and running sum accumulate in the *same* order so the exact-half comparison
is meaningful; the seeded permutation is a fixed SplitMix64 Fisher–Yates. The
`robust_descriptive` example is byte-for-byte reproducible across runs.

### Tests

30 inline unit tests (hand-computed MAD/IQR/weighted-median/trimmed/winsorized/MoM,
invalid input, non-finite input with reported index, deterministic ties,
zero-dispersion, translation invariance/equivariance, positive-scale equivariance)
plus 7 `proptest` properties (gated `#[cfg(all(test, not(miri)))]` like the
existing `dist` proptests): MAD non-negativity and translation invariance, IQR
scale equivariance, trimmed-mean translation equivariance and range containment,
weighted-median permutation invariance, MoM seeded reproducibility.

### Benchmark fingerprint

Example `scirust-stats/examples/robust_descriptive.rs` — a deterministic
contamination sweep (0 %, 5 %, 10 %, 20 %, 30 %, 40 %, 45 % front-loaded gross
outliers) comparing mean, median, trimmed, winsorized, median-of-means, MAD, IQR
and weighted median, with absolute error against the known location.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
9699b5151f9f0daecc14dbf89c499408dd2d8ca19f5b86f5c4e84c8a8d5ad131
```

The fingerprint tells an honest breakdown story: the mean diverges from the first
outlier (0 % breakdown); median and weighted median stay within ~1.4 of truth even
at 45 %; trimmed/winsorized(0.25) hold until ~25 % then break; median-of-means with
11 blocks breaks once more than half its blocks are corrupted (visible near 10 %),
exactly as the theory predicts. No estimator is presented as majority-robust.

### Compatibility

Purely additive. No existing `scirust-stats` item changed. `cargo check --workspace
--all-targets --locked` passes. The only `scirust_stats::prelude::*` glob in the
workspace is the crate's own doctest, so the new prelude entries cannot collide
with any dependent.

### Known limitations / deferred

- Bounded breakdown only (see assumptions above); majority-corruption handling is
  deferred to phase 725 under explicit identifiability assumptions.
- The seeded permutation uses `next_u64() % (i+1)`; the negligible modulo bias is
  deterministic and irrelevant to the estimate, but it is not a uniform shuffle.
- Cross-platform bit-identity of the fingerprint relies on `scirust-special`'s
  audited normal quantile; only same-toolchain run-to-run identity is verified here.
- Multivariate/covariance robustness, robust regression and SRCC integration are
  out of scope for this phase.
