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
| 721 | Robust descriptive statistics | `claude/scirust-srcc-robust-stats-6ue9xc` | [#725](https://github.com/Memorithm/scirust/pull/725) | `f2b87380eb02dda91e39c7c45f1ab73d6f9a5a36` | **Merged** |
| 722 | Robust multivariate geometry | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | _(open)_ | _(pending)_ | In review |
| 723 | Scale-aware SRCC source geometry | `feat/srcc-scale-aware-source-geometry` | — | — | Not started |
| 724 | Deterministic robust regression | `feat/learning-robust-regression` | — | — | Not started |
| 725 | Trust & contamination models | `feat/srcc-trust-contamination-models` | — | — | Not started |
| 726 | Certified medoid clustering | `feat/solvers-certified-medoid-clustering` | — | — | Not started |
| 727 | Industrial benchmark harness | `feat/srcc-industrial-benchmark-harness` | — | — | Not started |
| 728 | Real industrial evaluation | `feat/srcc-industrial-real-data-evaluation` | — | — | Not started |
| 729 | Shadow deployment & promotion gates | `feat/mlops-srcc-shadow-deployment` | — | — | Not started |

> Branch and numbering notes:
> - The program's suggested per-phase branches (`feat/stats-robust-descriptive`,
>   `feat/multivariate-robust-geometry`, …) are replaced by this session's fixed
>   development branch `claude/scirust-srcc-robust-stats-6ue9xc`, which is
>   restarted from the merged `master` for each successive phase (one PR per
>   phase is preserved).
> - GitHub assigned phase 721 the real number **PR #725** (numbers 721–724 were
>   consumed by other concurrent branches). The phase→number mapping is *not*
>   1:1; real numbers are recorded here honestly rather than forced to match.

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

---

## Phase 722 — Robust multivariate scaling and geometry

**Crate:** `scirust-multivariate` · **Module:** `scirust-multivariate/src/robust_geometry.rs`

### Design

Fitted geometry models that remove accidental dependence on raw coordinate
scales and units. New public surface (re-exported at the crate root):

- `RobustScaler` (`fit` / `transform` / `inverse_transform`) with
  `RobustScalerConfig { center, scale_method, zero_scale_policy, minimum_scale }`,
  `RobustScaleMethod { StandardDeviation, MedianAbsoluteDeviation,
  InterquartileRange }` and `ZeroScalePolicy { Error, UnitScale, DropDimension }`.
  Locations are the mean (std-dev method) or median (MAD/IQR); the location is
  always fitted and stored, so dropped dimensions are restorable. Degenerate
  dimensions (scale ≤ `minimum_scale`) follow the explicit policy — never a
  silent substitution. Config and scaler derive serde.
- `FittedDistanceMetric { RawEuclidean, RelativeNorm { epsilon },
  RobustDiagonal { scaler }, RegularizedMahalanobis { location, inverse_scatter,
  ridge } }` with `distance`, `fit_robust_diagonal`,
  `fit_regularized_mahalanobis`, `fitted_dimension_count`, and
  `validate_feature_descriptors`. The relative norm is
  `‖x − y‖ / max(‖x‖, ‖y‖, ε)`. The Mahalanobis variant is named
  `RegularizedMahalanobis` — **not** `RobustAffineInvariant` — because its
  location/scatter are the classical mean/covariance plus an explicit ridge; a
  singular regularized scatter is a typed `SingularScatter` error via a strict
  Cholesky, never a hidden fallback (the crate's historical regularizing
  `cholesky` is untouched; its `invert_lower_triangular` is reused).
- `FeatureDescriptor { name, dimension: scirust_units::Dimension }` — the units
  boundary. Raw metrics (`RawEuclidean`, `RelativeNorm`) require one common
  physical dimension across features; fitted metrics render coordinates
  dimensionless and accept mixed dimensions (count must match). No serde on this
  type (`Dimension` does not serialize).
- Typed `RobustGeometryError` (manual `Display`/`Error` with `source()` for the
  embedded `RobustStatsError`); ragged matrices, non-finite entries, invalid
  ε/ridge/minimum-scale, dimension mismatches and empty descriptors are all
  explicit errors. No silent `NaN`.

Reuse: MAD/IQR/median/mean come from `scirust-stats` (phase 721) — no duplicate
robust statistics. Dependencies added to `scirust-multivariate`: `scirust-stats`,
`scirust-units` (both cycle-free; multivariate has no reverse dependencies), and
dev-only `serde_json` for a serialization round-trip test. `#![forbid(unsafe_code)]`
added to the crate (it contained no unsafe).

### Invariance groups (documented, tested, never overstated)

| Metric | Invariant to | Not invariant to |
|---|---|---|
| `RawEuclidean` | rigid motions | any rescaling |
| `RelativeNorm` | common positive rescaling (ε-inactive regime) | per-coordinate rescaling, translation |
| `RobustDiagonal` (refit) | positive per-coordinate rescaling + translation | rotations, general affine maps |
| `RegularizedMahalanobis` (refit) | affine maps in exact arithmetic only; ridge + floating point break exact equivariance | — |

### Determinism contract

Pure fixed-order loops, no RNG, no thread-dependent reductions; benchmark
neighbour ranking uses `f64::total_cmp` with index tie-breaks. Fitting the same
matrix twice yields bit-identical models (tested).

### Tests

22 inline unit tests: transform/inverse round trips (all methods × center
on/off), hand-computed MAD scaling, deterministic fitting (bit-identical),
row-order invariance for order-statistic scalers, all three zero-scale policies,
all-constant matrix → `NoActiveDimensions`, ragged/non-finite/mismatch
rejections, per-coordinate rescaling and translation invariance after refit,
relative-norm common-scale invariance and zero-vector behaviour, Mahalanobis vs
hand-computed isotropic case, singular scatter as typed error resolved by an
explicit ridge, invalid ε/ridge rejection, units-descriptor validation, serde
round trip, and a no-silent-NaN sweep.

### Benchmark fingerprint

Example `scirust-multivariate/examples/robust_geometry.rs` — fixed two-cluster
dataset; global scales `{1, 1e3, 1e6, 1e9}` and an anisotropic column scaling
`diag(1, 1e3, 1e6, 1e9)`; metrics refit per transform; reports kNN-set
preservation, two-cluster nearest-medoid recovery, and max pairwise distortion.
Timings go to stderr only — never into the hashed artifact.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
bad8cf079c7d0aa4a2320fa40d96ff843f222507ed77142ee2bd3703ccf87dee
```

Honest highlights: under anisotropic scaling, raw-Euclidean kNN preservation
collapses to 2.5 % while robust-diagonal stays at 100 % with ~1e-15 distortion;
the relative norm also collapses under anisotropy (documented — it is only
common-scale invariant) and shows weak cluster recovery (65 %) even at baseline
because it is not translation invariant; the Mahalanobis ridge visibly breaks
exact scale equivariance (~8e-10 distortion). Negative results retained.

### Compatibility

Purely additive to the public API; the historical `cholesky`/`invert_matrix`
Mahalanobis helpers are untouched and all 42 pre-existing tests pass unchanged.
`Cargo.lock` gains only the three new dependency edges (no new external
packages). `cargo check --workspace --all-targets --locked` passes.

### Known limitations / deferred

- The robust scatter problem (MCD-style robust covariance) is **not** solved
  here; `RegularizedMahalanobis` is a classical baseline by design.
- Scale invariance of `RobustDiagonal` holds when the scaler is refit on the
  rescaled data — a frozen scaler applied to differently-scaled inputs is not
  invariant (and is validated only for dimension count, not provenance).
- SRCC integration is deferred to phase 723.
