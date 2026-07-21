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
| 722 | Robust multivariate geometry | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#729](https://github.com/Memorithm/scirust/pull/729) | `c4e49bfbeb936182b4474a67afd83595ade6727e` | **Merged** |
| 723 | Scale-aware SRCC source geometry | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | _(open)_ | _(pending)_ | In progress |
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

---

## Phase 723 — Scale-aware SRCC source geometry

**Crate:** `scirust-srcc` · **Module:** `scirust-srcc/src/robust_source_geometry.rs`

### Design

Opt-in scale-aware geometry for SRCC source clustering, leave-one-out
stability, and stable search. New public surface:

- `SrccSourceGeometrySpec { RawEuclidean, RobustDiagonal { scaler_config } }` —
  `RawEuclidean` delegates to the frozen historical `source_distance` body, so
  the historical pipeline is reproduced **bit for bit at every radius**;
  `RobustDiagonal` fits a `scirust-multivariate::RobustScaler` (phase 722) on
  the sources pooled across the fitted views and measures distances in
  fitted-scale units.
- `SrccScaleAwareSourceClusteringConfig { geometry, clustering }` — geometry and
  radius travel together because the radius' meaning depends on the geometry.
- `SrccScaleAwareSourceClusteredSearchConfig { source, base_config,
  distortion_weight }`.
- Four opt-in functions mirroring the PR #719/#720 grammar:
  `fit_scale_aware_source_clustered_robust_srcc_projector_from_views`,
  `evaluate_scale_aware_source_clustered_robust_leave_one_out_stability`,
  `search_scale_aware_source_clustered_robust_srcc_structures_from_views`,
  `search_stable_scale_aware_source_clustered_robust_srcc_structures_from_views`.
- Three appended `SrccRobustFitError` variants (Eq-safe payloads, additive
  Display): `InvalidSourceGeometry`, `DegenerateSourceScale { dimension }`,
  `NoActiveSourceDimensions`. The existing `From` chain surfaces them through
  LOO/search/stable-search with no new error types.

**Injection mechanism.** A `pub(crate) enum SourceMetric { Raw, Diagonal {
inverse_scales } }` is threaded through the four private clustering helpers in
`robust_source.rs` (signatures only; bodies untouched). `Raw` calls the frozen
`source_distance`; `Diagonal` calls `scaled_source_distance`, a line-for-line
mirror whose only change is multiplying each coordinate difference by its
fitted inverse scale — a zero inverse scale reproduces the historical zero-skip
branch, implementing dropped dimensions with no extra logic. Distance
evaluation deliberately stays srcc-local: the multivariate crate's Euclidean is
**not** bit-compatible with srcc's hypot-style scaled accumulation. Fitting
reuses `RobustScaler` — no second robust scaler exists.

### No-leakage contract

The geometry model is fitted **inside** every fit call from the sources of
exactly the views being fitted; the leave-one-out evaluator therefore refits
the scaler after every removal. Sources are globally sorted into the crate's
canonical vector order before fitting so the fitted metric is invariant to both
sample order and view order.

### Determinism and invariance (never overstated)

Clustering decisions under `RobustDiagonal` are invariant (within FP tolerance)
to positive per-coordinate rescaling of the sources when the scaler is refit.
Nothing makes the learned transports or projector rescaling-invariant —
transport learning still sees raw coordinates — and no affine invariance is
claimed. Global-scale, coordinate-wise and affine invariance are explicitly
distinguished; only the first two (for clustering decisions, after refit) are
claimed and tested.

### Tests

14 new tests (99 total in the crate, all green): four full-struct equality
suites proving `RawEuclidean` ≡ historical pipeline (fit, LOO, search, stable
search; zero and positive radii), transitivity to the exact-source pipeline at
radius 0, an anisotropic two-state fixture where raw geometry fails with a
*typed* consensus ambiguity at every radius while robust-diagonal geometry
recovers the states, per-coordinate rescaling invariance after refit,
sample/view-order invariance, bit-identical determinism, `Error`-policy
degenerate-coordinate reporting, invalid-config and error-precedence checks,
stable-search certification, and a **breakdown regression**: an unbalanced view
pair (4+3 / 3+3) whose single removal balances the pooled states 6–6, inflating
the signal coordinate's MAD to the separation itself — the LOO evaluation fails
with a typed ambiguity, never a silent certificate.

### Benchmark fingerprint

Example `scirust-srcc/examples/scale_aware_source_geometry.rs` — three
runtime-verified sections: (1) RawEuclidean compatibility across the PR #720
jitter grid plus exact-source equivalence at radius 0; (2) the anisotropic
two-state fixture under common source rescaling 1/1e3/1e6/1e9 — raw geometry
yields the typed ambiguity, robust-diagonal yields an identical structural
outcome (rejected dimension 2, LOO ratio 1.0) at the same scaled radius at
every scale; (3) the majority-breakdown honest negative described above. The
historical examples are untouched and byte-identical before/after the change
(verified by `cmp`).

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
scale_aware_source_geometry:  fc0dcbca9f9763d25c6ba5e6485a95d26daad9da9648044948e4d595580fb84c
source_clustered_stable_search (unchanged): b2a1a0415693addc40ab4224c8d498325ac106c52dfc11ebcc691d731f8766cb
robust_source_clustering (unchanged):       4943a4e48312224268abf555d4f304c3ab33095d1c15813423be2c407fa2dc27
```

### Compatibility

No existing public item changed; no field added to any existing struct; the
two historical entry points pass `SourceMetric::Raw` into the refactored
helpers and execute the identical frozen distance body on identical inputs in
identical order. All 85 pre-existing tests pass unchanged. `scirust-srcc` gains
one cycle-free dependency edge (`scirust-multivariate`); `Cargo.lock` changes
by exactly that edge.

### Known limitations / deferred

- The pooled-source MAD inherits the ~50 % breakdown of robust scales: balanced
  two-state populations inflate the signal coordinate's scale (demonstrated and
  typed, not hidden). Explicit trust/identifiability assumptions are phase 725.
- `RobustDiagonal` requires genuinely spread coordinates: a majority of exactly
  repeated values gives a zero MAD (policy-controlled). The historical
  exact-repetition fixtures are therefore served by `RawEuclidean`, not by MAD
  geometry.
- Only diagonal scaling is offered; no rotation/affine-invariant source metric
  is claimed or provided.
