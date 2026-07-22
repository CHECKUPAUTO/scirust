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
| 723 | Scale-aware SRCC source geometry | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#737](https://github.com/Memorithm/scirust/pull/737) | `96b73ea9c45b4c97456ca11cbb0a1e3164d3cfae` | **Merged** |
| 724 | Deterministic robust regression | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#739](https://github.com/Memorithm/scirust/pull/739) | `c1c10946a48cb57daaff3fb4875a69064a0bb148` | **Merged** |
| 725 | Trust & contamination models | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#742](https://github.com/Memorithm/scirust/pull/742) | `b1eca0ff7df85168539a841a7223d7a0af059e7b` | **Merged** |
| 726 | Certified medoid clustering | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#745](https://github.com/Memorithm/scirust/pull/745) | `091cefad9c3ec00c37aab4631d2890146482b613` | **Merged** |
| 727 | Industrial benchmark harness | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#747](https://github.com/Memorithm/scirust/pull/747) | `2e44bada7f20c50afa427e3c94a816c2348819d4` | **Merged** |
| 728 | Real industrial evaluation | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#750](https://github.com/Memorithm/scirust/pull/750) | `ea45a4d073c1a1aa1e2e9c87335146bf2c158ab2` | **Merged** |
| 728D | Diagnostic — why the 728 nulls (framing, not power) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#751](https://github.com/Memorithm/scirust/pull/751) | `8c83bf6ca74e1f4fb157982594ee98df5ac87659` | **Merged** |
| 728R1 | Re-framing lever 1 — piecewise RUL + reduced decimation | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#753](https://github.com/Memorithm/scirust/pull/753) | `c152e850d7a38f8e03c6d05d0796cb78b63ebfe5` | **Merged** |
| 728R2 | Re-framing lever 2 — high-leverage contamination (decisive test) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#754](https://github.com/Memorithm/scirust/pull/754) | `2467b18b80657774f843fc116335e9ada9126ffd` | **Merged** |
| 728R3 | Re-framing lever 3 — SECOM supervised reformulation | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#755](https://github.com/Memorithm/scirust/pull/755) | `c60c5769` | **Merged** |
| F1a | Follow-up 1 — isotonic (monotone) regression method (`scirust-learning`) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#757](https://github.com/Memorithm/scirust/pull/757) | `2f43f713` | **Merged** |
| F1b | Follow-up 1 — monotone RUL evaluation (realize the lever-1 ceiling) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#758](https://github.com/Memorithm/scirust/pull/758) | `6342217c` | **Merged** |
| F2 | Follow-up 2 — stabilized supervised SECOM (feature selection + within-class LDA) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#759](https://github.com/Memorithm/scirust/pull/759) | `eaf6c2f3` | **Merged** |
| F3 | Follow-up 3 — native heavy-tailed noise (the regime robustness wins) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#760](https://github.com/Memorithm/scirust/pull/760) | `1c75ab3d` | **Merged** |
| 729 | Shadow deployment & promotion gates (follow-up 4) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#761](https://github.com/Memorithm/scirust/pull/761) | `15a7319c` | **Merged** |
| A1a | Axis 1 — RBF kernel ridge regression method (`scirust-learning`) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#763](https://github.com/Memorithm/scirust/pull/763) | `c8f5beb3` | **Merged** |
| A1b | Axis 1 — multivariate nonlinear RUL evaluation (kernel ridge vs isotonic) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | [#764](https://github.com/Memorithm/scirust/pull/764) | `b119f1e2` | **Merged** |
| A2 | Axis 2 — SECOM past the linear ceiling (drift correction vs degree-2 nonlinearity) | `claude/scirust-srcc-robust-stats-6ue9xc` (restarted) | _pending_ | _pending_ | Draft |

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
to positive per-coordinate rescaling of the sources when the scaler is refit
**and the scaler configuration is itself scale-covariant** (`minimum_scale = 0`
with policy `Error` or `DropDimension`). `UnitScale` keeps degenerate
coordinates in raw units and a positive `minimum_scale` is an absolute raw-unit
threshold; both deliberately re-introduce a scale dependence and void the
invariance (documented on the API). Nothing makes the learned transports or
projector rescaling-invariant — transport learning still sees raw coordinates —
and no affine invariance is claimed. The raw pipeline's failure on the
anisotropic fixture is precisely characterised as **three regimes** (singleton
fragmentation below the signal separation, typed consensus ambiguity in the
bridging band, silent majority-vote state merging above the noise spread) — it
is *not* claimed that every raw radius produces a typed error.

### Adversarial review (pre-merge)

A 25-agent adversarial review (5 dimensions × refutation-based verification,
with empirical probe programs) confirmed and led to fixing before merge:

- **Non-finite fitted scale bypassed every zero-scale policy** (overflowing MAD
  × 1.4826 or mean/variance on huge finite values produced `scale = ∞` marked
  *active*; `1/∞ = 0` then silently deactivated the coordinate — demonstrated
  end-to-end as a silent wrong `Ok`). Fixed by a finiteness gate in
  `RobustScaler::fit` (new typed `RobustGeometryError::NonFiniteScale`) plus a
  reciprocal-finiteness check at metric fitting (new typed
  `SrccRobustFitError::NonFiniteSourceScale`, also covering subnormal scales
  whose reciprocal overflows).
- **Dropped coordinates were not exactly inert**: `(±∞ diff) × 0.0 = NaN`
  aborted the fit on a coordinate the policy had removed. Fixed by a structural
  skip of zero-inverse-scale coordinates before any arithmetic.
- **Overclaimed raw-failure and invariance statements** (see above) reworded
  and pinned by a three-regime test; the rescaling-invariance test now asserts
  exact canonical-cluster equality (rescaled canonicalization ==
  coordinate-wise-rescaled original canonicalization, bit-exact), not a
  rejected-dimension proxy; a StandardDeviation view/sample-order test now
  guards the canonical pooled-source sort (deleting the sort fails it);
  `UnitScale`, `NoActiveSourceDimensions` and discovery-before-geometry error
  precedence gained dedicated tests.
- The RawEuclidean equality suite proves *delegation*; absolute historical
  behaviour is pinned independently by the pre-existing exact-value tests and
  the byte-identical historical example outputs (both verified unchanged).

### Tests

18 new tests (103 total in the crate, all green; plus one new
overflow-gate test in `scirust-multivariate`, 65 total there): four full-struct
equality
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

---

## Phase 724 — Deterministic robust regression

**Crate:** `scirust-learning` · **Module:** `scirust-learning/src/robust_regression.rs`

### Design

General, SRCC-independent robust linear/affine regression with typed errors and
convergence metadata. New public surface (re-exported at the crate root):
`RegressionDataset` (features `n×p`, targets `n×k`, optional weights),
`LinearRegressionModel` (+ `predict`), `RobustLoss { Squared, AbsoluteApprox,
Huber, TukeyBisquare }`, `RobustRegressionMethod { OrdinaryLeastSquares,
IterativelyReweightedLeastSquares, TrimmedLeastSquares { retained_fraction },
MedianOfMeans { block_count, seed } }`, `RobustRegressionConfig` (+ `Default`),
`RobustRegressionReport`, `RobustRegressionError`, `fit_robust_regression`.

- **Solves** run through `scirust-solvers` Householder QR
  (`qr_decompose` + `solve_qr_least_squares`) — never normal equations; ridge is
  row augmentation of the QR design (never penalizing the intercept); rank
  deficiency without ridge is the typed `LeastSquaresSolveFailed`.
- **OLS** supports multiple outputs; the robust methods are single-output with a
  typed `UnsupportedMultiOutput` otherwise.
- **IRLS** initializes at the deterministic OLS solution; the residual scale is
  the normal-consistent MAD from `scirust-stats`; the `σ = 0` case (majority of
  residuals vanish) uses the documented `σ→0` weight limit with a warning.
  `AbsoluteApprox` is honestly named a smoothed approximation, never "exact
  LAD"; Tukey bisquare is documented non-convex/initialization-dependent.
- **Trimmed LS** is the C-step iteration with `f64::total_cmp` ranking and
  original-index tie-breaks; convergence is retained-set fixed point **or
  objective stagnation** (near-exact fits leave FP-tied residuals whose ranked
  subset churns among equivalent sets without changing the objective), plus
  explicit cycle detection with a warning. Rejected indices are reported.
- **Median-of-means** uses a seeded `SplitMix64` Fisher–Yates permutation and
  as-even-as-possible contiguous blocks; per-block OLS coefficients are
  aggregated by coordinate-wise median — a **documented heuristic** (not affine
  equivariant, no optimality certificate, requires a majority of clean blocks;
  each outlier can poison an entire block, so the guarantee needs
  `outliers ≤ ⌊(blocks−1)/2⌋`).
- Optional feature standardization refits `scirust-multivariate::RobustScaler`
  (policy `Error`: a degenerate feature scale is typed, never a silent unit
  fallback). Historical `linear_regression`/`polynomial_fit` baselines
  untouched.

### Determinism contract

The only pseudo-randomness is the explicit MoM seed. Same input + config ⇒
bit-identical reports (tested per method). Row order is documented as *not* a
free invariance: QR rotations differ at the last bit under permutation (tested
to tolerance), and the MoM partition composes the seed with the caller's row
order.

### Tests

20 new tests (71 total in the crate, all green): exact affine/linear/
multi-output recovery, cross-check against the historical 1-D
`linear_regression`, Huber vs OLS under minority outliers, trimmed exact
recovery + rejected-index reporting, MoM seeded reproducibility with the
`outliers ≤ ⌊(blocks−1)/2⌋` guarantee **and** an honest majority-block
breakdown test, rank-deficiency typed failure + ridge resolution, zero-weight
samples, non-convergence reporting, standardized rescaling invariance of
predictions, row-permutation tolerance agreement, per-method bit determinism,
and the full typed-error battery (inputs, weights, config, multi-output,
degenerate feature scale, underdetermined designs).

### Benchmark fingerprint

`scirust-learning/examples/robust_regression_contamination.rs` — fixed noisy
affine truth, contamination sweep 0/5/10/20/30/40 % (front-loaded `+100` target
shifts), methods OLS / Huber(1.345) / trimmed(0.7) / MoM(5 blocks); metrics:
parameter error, clean RMSE, clean median absolute error, worst clean error,
effective samples, iterations, convergence.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
8bb566fb73f2d255d6653e9c9b814fcf835f64d62f01eea71ad2070255ad3722
```

Honest highlights (negative results retained): OLS breaks from 5 %; MoM breaks
at 10 % (six outliers touch three of five blocks); Huber hits its iteration cap
at 20 % (`converged=false` in the output) and breaks at 30 %; trimmed h = 0.7
holds through 20 % and fails exactly at its 30 % boundary; at 30–40 % the
robust methods can be *worse* than OLS. No majority-corruption robustness is
claimed anywhere.

### Compatibility

Purely additive: no existing `scirust-learning` item changed; `Cargo.lock`
gains exactly the three cycle-free workspace edges (`scirust-solvers`,
`scirust-stats`, `scirust-multivariate`). `cargo check --workspace
--all-targets --locked` passes; clippy `-D warnings` and `fmt` clean.

### Known limitations / deferred

- Robust methods are single-output; multi-output robust fitting is deferred.
- No exact LAD (linear programming) and no MM-estimation; `AbsoluteApprox` is a
  smoothed surrogate.
- Breakdown honesty: Huber's breakdown shrinks with leverage (no leverage-based
  weighting here); trimmed tolerates at most `1 − h`; MoM needs a clean-block
  majority. SRCC integration and identifiability assumptions are later phases.

---

## Phase 725 — Trust fusion and identifiable majority contamination

**Crate:** `scirust-srcc` · **Module:** `scirust-srcc/src/trust.rs`

### Design

Explicit, machine-checkable trust models that keep SRCC useful when aberrant
observations form a global numerical majority — only under declared
identifiability assumptions, never as arbitrary-majority robustness. Principle:
if two incompatible explanations are equally consistent with the supplied
assumptions, the result is the typed
`SrccTrustError::UnidentifiableContamination { competing_hypotheses }`.

New public surface: `SrccTrustProviderId` (typed, no strings),
`SrccTrustEvidenceKind` (8 kinds with per-kind score semantics: anchors are
exactly `{0,1}`, temporal predictions are non-negative errors, everything else
`[0,1]`), `SrccTrustEvidence`, `SrccObservationTrust` (caller addressing,
finite non-negative priors), `SrccTrustModel`, `SrccTrustPolicy { Unweighted,
TrustedAnchors, IndependentViews, TemporalPersistence,
GroupContaminationBound, CompositeAll }` (conjunction only — a disjunction
would certify nothing and is deliberately not exposed), the
`SrccTrustEvidenceProvider` trait + `collect_trust_evidence` (the integration
point for estimation/SPC/PdM monitors; no heavy dependency is hard-wired),
`fit_trusted_robust_srcc_projector_from_views`,
`evaluate_trusted_robust_leave_one_out_stability`, `SrccTrustedFitResult` with
`SrccTrustCertificate` (assumptions + per-group total weight, winning support,
runner-up support, anchor count) and typed `SrccTrustError` /
`SrccTrustedFitError`.

### Consensus mathematics

- The per-group target consensus becomes the **weighted medoid**
  `argmin_c Σᵢ wᵢ·d²(c, targetᵢ)` over positive-weight candidates, scanned in
  the historical fixed order; under `Unweighted` every weight is `1.0` and
  `1.0 · x == x` in IEEE arithmetic, so the historical unweighted result is
  reproduced **bit for bit** through the same code path (full-struct equality
  test against `fit_robust_srcc_projector_from_views`).
- **Adversarial margin**: with `support(c)` the summed weight of observations
  bit-identical to `c`, a policy bounding corrupted weight by `β·W` accepts
  only when `support(winner) − support(runner-up) > 2·β·W` — otherwise an
  adversary inside the declared bound could have flipped the outcome, and the
  typed unidentifiability is returned. Bounds are validated `β ∈ [0, 0.5)`.
- Weights are **not** probabilities and are never multiplied across evidence
  kinds; each policy states exactly how evidence gates weights (anchors gate
  non-anchors to zero; persistence gates observations lacking
  `minimum_consistent_steps` predictions within `maximum_prediction_error`;
  `CompositeAll` takes the per-observation minimum and the strictest margins).
- Tied distinct targets are counted (deduplicated) as competing hypotheses;
  under an anchor policy a weighted tie means the anchors themselves disagree
  and is remapped to the explicit `ConflictingAnchors`.

### Leave-one-out contract

`evaluate_trusted_robust_leave_one_out_stability` re-derives the trust weights
for every reduced view set (records addressing the removed sample are dropped,
later records shift down) and refits the complete trusted pipeline — trust
decisions are recomputed per variant, never frozen (verified: removing one of
exactly-minimum anchors yields the typed `InsufficientAnchorSupport`).

### Tests

14 new tests (117 total in the crate): the bit-exact Unweighted compatibility
test plus the program's ten adversarial scenarios — (1) count-majority /
weight-minority accepted under the per-view bound with certificate; (2)
view-concentrated 80 % weight rejected as unidentifiable; (3) unweighted
majority corruption silently electing the contaminant (exposed) while the
group bound refuses; (4) trusted anchors recovering under count-majority
corruption; (5) conflicting anchors failing explicitly; (6) temporal
persistence defeating a burst attack; (7) a 50/50 persistent split reported as
`UnidentifiableContamination { 2 }`, not blindly rejected; (8) conflicting
policies under `CompositeAll` failing loudly (`AllObservationsUntrusted`); (9)
deterministic weighted ties; (10) trusted leave-one-out recomputation — plus
the invalid-model battery (bounds, duplicates, scores, priors, nested/empty
composites, insufficient views) and provider-collection determinism.

### Benchmark fingerprint

`scirust-srcc/examples/trusted_consensus_contamination.rs` — a 225-row
contamination matrix (corrupt count {0..4, 6} against 5 clean per view ×
weight concentration × anchor corruption × burst/persistent attack × five
policies) with typed outcomes and certificate fields per cell, driven through
a deterministic `SrccTrustEvidenceProvider` adapter.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
7d0a6d72c15ac5bc316761bd9c258439704b5bb15c06021ab9daf02256353a9b
```

Honest failure surface, printed not hidden: at the 55 % corrupt majority the
unweighted consensus silently elects the contaminant (`accepted:bad`); the
bounded policies refuse (`unidentifiable:2`); trusted anchors recover
(`accepted:clean`); temporal persistence gates the burst but **accepts the
contaminant under a persistent majority attack** — persistence alone is not an
identifiability assumption against persistent adversaries. The 50/50
persistent split is unidentifiable under every policy, by construction.

### Compatibility

Purely additive: no existing public item changed; two private helpers of
`robust.rs` promoted `pub(crate)` (bodies untouched); all three historical
srcc example outputs remain byte-identical (`b2a1a041…`, `4943a4e4…`,
`fc0dcbca…`). No new dependency edge. `cargo check --workspace --all-targets
--locked` passes.

### Known limitations / deferred

- Trust integrates with the exact-source robust fitter; composition with the
  scale-aware clustered pipeline (723) is deferred to the benchmark-harness
  phase where both opt-ins meet.
- Anchor incorruptibility, per-scope corruption bounds and persistence are
  *assumptions*, not detections; the certificates state them, the policies
  enforce their logical consequences, and nothing here verifies them against
  ground truth (that is what phases 727–728 measure).
- The margin analysis covers vote flipping between bit-identical target
  candidates; continuous-target displacement attacks inside a candidate are
  bounded only by the medoid geometry, not by the margin certificate.

---

## Phase 726 — Certified globally-optimal medoid clustering

**Crates:** `scirust-solvers` (`src/combinatorial/mod.rs`, new module) and
`scirust-srcc` (`src/certified_source.rs`, opt-in integration)

### Design

Replaces "greedy complete-link is good enough" with an opt-in solver that
either **proves** a globally optimal source partition or returns the best
known partition with a valid lower bound and an explicit optimality gap —
never an unproven optimality claim.

Problem: given `n` points with pairwise distances and a diameter budget `D`,
partition so every intra-cluster pair satisfies `d(i, j) ≤ D`, optimizing the
lexicographic objective (1) minimum cluster count, (2) minimum total
**observed-medoid** cost `Σ_C min_{m∈C} Σ_{p∈C} d(p, m)` (medoids are observed
points, smallest index on ties), (3) lexicographically smallest canonical
assignment (labels by first occurrence). Connecting pairs with `d > D` yields
the *incompatibility graph*; a valid cluster is an independent set, so the
count objective is graph coloring — NP-hard in general, hence certificates
instead of unconditional speed claims.

New public surface (solvers): `DistanceMatrix` (validated: exact symmetry with
zero tolerance, exactly-zero diagonal, finite non-negative entries),
`CertifiedClusteringMode { Exact, Hybrid { maximum_nodes, maximum_iterations }
}`, `CertifiedMedoidClusteringConfig`, `ClusteringCertificate`,
`CertifiedMedoidClusteringResult`, `certified_medoid_clustering`, typed
`MedoidClusteringError` (8 variants). New public surface (srcc):
`SrccSourceClusteringSolver { GreedyCompleteLink, CertifiedMedoid { mode } }`,
`SrccCertifiedFitResult { fit, view_certificates }`,
`fit_certified_source_clustered_robust_srcc_projector_from_views`, plus
re-exported `CertifiedClusteringMode` / `ClusteringCertificate`.

### Algorithm and certificates

- Deterministic DSATUR-style branch and bound: saturation-first vertex
  selection (ties: descending incompatibility degree, then ascending index),
  colors tried in ascending label order plus at most one fresh label, pruned
  against the incumbent's lexicographic objective. Greedy first-fit warm start
  in both modes; `Hybrid` adds a bounded deterministic local-improvement pass
  and a node budget.
- Count lower bound: deterministic greedy clique on the incompatibility graph
  (`χ ≥ ω` — valid, not claimed tight; the benchmark's `weak_lower_bound`
  family shows it strictly undershooting).
- `proven_optimal = true` **only** when the search space is exhausted. On
  budget exhaustion: incumbent + clique lower bound + `lower_bound_medoid_cost
  = None` (distances are not required to satisfy the triangle inequality, so
  the trivial `0` is the only generally valid cost bound and no nontrivial
  bound is claimed) + documented positive gap (integer count gap when
  positive, else the conservative `cost/(1+cost)` fraction). Budget exhaustion
  is a certificate state, not an error.
- Exactness is verified against an exhaustive oracle enumerating **all set
  partitions** via restricted growth strings: sizes 2–7 × 12 seeds × 3
  diameters (216 instances), bit-exact cost equality (`to_bits`).

### SRCC integration semantics (honest)

`GreedyCompleteLink` delegates to the scale-aware pipeline unchanged
(runtime-verified full-struct equality; empty certificate vector — greedy
certifies nothing). `CertifiedMedoid` solves each view's pairwise
source-distance matrix under the fitted geometry (canonical sample order,
distances computed once and mirrored), rewrites members to observed source
medoids, and delegates target aggregation to the frozen robust fitter. The
semantic difference is stated, not hidden: a bridge sample the greedy pass
rejects as `AmbiguousSourceClusterAssignment` is *assigned* here — optimally,
deterministically, with a certificate (demonstrated on the recorded PR #719
bridge fixture). Both behaviours are legitimate; the caller chooses by
choosing the solver. Defaults are untouched.

### Tests

12 new solver tests (201 total in scirust-solvers): validation battery,
single-point/one-cluster/all-singleton edges, greedy-suboptimal bridge
repaired by search, cost refinement at equal counts, canonical tie
resolution on the perfect square, the 216-instance oracle equivalence, hybrid
budget exhaustion (feasibility preserved, gap positive, no cost bound), hybrid
with ample budget proving optimality, determinism, zero-diameter duplicate
grouping. 6 new srcc tests (123 total): bit-exact greedy delegation, certified
= greedy on a proven two-cluster fixture, bridge resolution the greedy pass
rejects, determinism, per-view gapped certificates under an exhausted budget,
radius validation ordering.

### Benchmark fingerprint

`scirust-solvers/examples/certified_medoid_clustering.rs` — seven families
(separable / chain / bridge / seeded metric-free adversarial / cost ties with
canonical winners / odd anti-cycle with provably weak clique bound /
increasing n) × three arms (`exact`, `hybrid_n1_i1`, `hybrid_n64_i4`), CSV
with count, cost, both lower bounds, proven flag, gap, explored/pruned nodes
and canonical assignments.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
c7da6bb4e3990d000ac3d269bc2cb75349dc71b378b8ddecf5f50f4471ee5f3d
```

Honest readings printed by the benchmark itself: `hybrid_n1_i1` often *finds*
the optimum but never *proves* it (nonzero gap on every non-trivial row — the
certificate separates finding from proving); on `increasing_n` n=8 it is
strictly suboptimal in count (4 vs 3, integer gap 1); on `weak_lower_bound`
(odd 13-cycle incompatibility, χ=3 > ω=2) the exact proof costs 2049 nodes,
the 64-node hybrid exhausts with gap 1, and the clique bound provably
undershoots.

### Compatibility

Purely additive. New dependency edge `scirust-srcc → scirust-solvers`
(cycle-free: solvers depends only on autodiff/symbolic). No existing public
item changed; `fit_source_metric` promoted `pub(crate)` (body untouched); one
appended `SrccRobustFitError::CertifiedClusteringFailed` variant (additive —
the enum is not `#[non_exhaustive]`, so downstream exhaustive matches gain one
arm, as this repo's own example did). All seven historical example outputs
re-verified byte-identical (`9699b515…`, `bad8cf07…`, `b2a1a041…`,
`4943a4e4…`, `fc0dcbca…`, `7d0a6d72…`, `8bb566fb…`). `cargo clippy --workspace
--all-targets --locked -- -D warnings` and `cargo fmt --all --check` pass.

### Known limitations / deferred

- Worst-case exponential: `Exact` is intended for the small per-view source
  sets SRCC actually produces; large instances belong to `Hybrid`, whose
  certificate says exactly what was and was not proven.
- The medoid-cost lower bound on budget exhaustion is deliberately absent
  rather than approximate; a Lagrangian or LP bound is future work and will
  come with its own validity proof or not at all.
- The certified path is per-view; cross-view consistency of the partitions is
  enforced downstream by the frozen robust fitter, not by the solver.
- Scale-aware + certified + trusted composition in one call is deferred to
  the benchmark-harness phase (727), where the opt-ins meet under one harness.

---

## Phase 727 — Industrial benchmark protocol and harness

**Crate:** `scirust-srcc-bench` (new, workspace layer above every method
crate) · **Preregistration:**
`docs/research/SRCC_INDUSTRIAL_BENCHMARK_PREREGISTRATION.md`

### Design

Protocol, not verdict: the harness provides the machinery to test industrial
claims honestly and contains **no superiority claims** — producing evidence
is phase 728's job, under the committed preregistration. New-crate decision:
the workspace precedent is one benchmark crate per program
(`scirust-tdi-bench` over `scirust-tdi`), all emitting
`scirust-bench-schema::BenchRecord`; `scirust-srcc-bench` follows it, sits
strictly above the method layers (bench-schema/estimation/stats/solvers/
unsupervised → multivariate/spc → srcc/learning), and no method crate
depends back on it.

Seven modules:

- `dataset`: `TabularDataset { features, targets, groups, time_index }` with
  typed shape/finiteness validation and a canonical versioned little-endian
  IEEE-754-bit SHA-256 (`content_sha256`) — bit-level identity, stricter
  than `==` (`-0.0` ≠ `+0.0` in the hash, verified);
- `manifest`: `DatasetManifest` (name, version, source, license, sha256,
  shape, target description, per-feature descriptors with units) validated
  *against the data*: checksum or shape disagreement is a typed error. No
  network anywhere; absent large data ⇒ skip, never download, in tests;
- `splits`: `RandomHoldout` / `GroupedHoldout` / `Temporal` /
  `LeaveOneGroupOut` with structural leakage prevention (groups never
  straddle, training never postdates evaluation, ties in `time_index`
  break by row index canonically), seeded Fisher–Yates, and a
  `SplitManifest` (seed, strategy, grouping key, dataset checksum) on every
  assignment;
- `contamination`: the program's nine kinds (`AdditiveNoise`,
  `CoordinateScaleShift`, `TargetFlip`, `SourceDuplication`,
  `CoherentAlternativeCluster`, `SensorBias`, `SensorDropout`,
  `BurstAttack` — temporally contiguous window, `ViewConcentratedAttack` —
  within one group) with exact `ContaminationManifest`s (kind verbatim,
  seed, affected rows, appended rows, input/output checksums).
  `fraction = 0` is a recorded no-op; overflow to non-finite is a typed
  error, not an emitted value. `TargetFlip` is documented as exactly
  involutive on `{0,1}` labels only (`1 − (1 − y) ≠ y` in IEEE — stated,
  not hidden);
- `adapter`: the common interface `BaselineAdapter` with **declared
  capabilities**: `TaskKind` (Regression / AnomalyDetection / StreamAlarm),
  `FittingProtocol` (Inductive / Transductive — LOF and DBSCAN are declared
  transductive rather than discovered), and `AdapterOutput` (Predictions /
  AnomalyScores / AnomalyLabels / AlarmSteps) which decides which metrics
  exist at all. Eleven adapters: OLS, Huber IRLS, trimmed LS,
  median-of-means (scirust-learning); Isolation Forest, LOF, DBSCAN-noise
  (scirust-unsupervised); regularized Mahalanobis (scirust-multivariate
  fitted metric), Hotelling T², CUSUM, EWMA (scirust-spc). Underlying
  failures surface as typed `AdapterError`s carrying the method's own error
  text;
- `metrics`: RMSE/MAE/median/worst absolute error; rank-based AUROC with
  average ranks on ties (score producers only; degenerate label sets are
  typed errors); confusion-derived rates as typed absences when undefined
  (`None`, never `0.0`); typed `DetectionOutcome::{Detected{delay},
  Missed}` with pre-onset false alarms counted separately; Rand and
  adjusted Rand indices (zero-denominator ARI is a typed error);
- `paired` + `records`: seeded percentile paired bootstrap (documented
  floor-quantile rule, paired Cohen's d as `None` on zero variance) →
  `ConfidenceInterval`; `RecordKey` emission helpers encode the
  capability-honesty rule once (label-only detectors never get an AUROC
  row; missed detections have no delay row; undefined rates are absent);
  `RunMetadata` (git commit, dataset SHA, configuration SHA, toolchain,
  feature flags) serialized **next to** result files, never inside hashed
  scientific content.

### Preregistration

Committed in this phase, before any real data: five falsifiable hypotheses
(H1 scale invariance, H2 robust regression under contamination, H3 trust
identifiability, H4 certified-vs-greedy clustering — with "greedy matches
everywhere" explicitly an acceptable publishable outcome, H5 end-task
transfer), primary/secondary metrics, dataset requirements incl. the fixed
SRCC transport-view construction (16 channels, preregistered horizon,
per-trajectory median centering), split strategy, baselines, hyperparameter
policy (never on test), exclusion rules, failure reporting, and a four-part
superiority criterion (paired CI excluding zero, ≥ 5 % relative, no
safety-metric degradation beyond budget, replication) below which the only
permitted conclusions are "no significant difference" / "underperformed" /
"inconclusive".

### Tests

49 tests: dataset hashing identity (every observable field changes the hash;
`-0.0` vs `+0.0`); manifest validation against tampered content and shapes;
split determinism, coverage, group-straddle impossibility, temporal
no-future-training with canonical tie-break, typed error battery;
contamination determinism + exact manifests, burst temporal contiguity on
scrambled time indices, view confinement, aligned duplication, binary-flip
involution, overflow detection; hand-computed metric values (AUROC tie
cases, ARI including its IEEE rounding), typed degeneracies; bootstrap
determinism, seed sensitivity, degenerate-variance handling; adapter
recovery of a noiseless affine law, gross-outlier ranking by all four score
producers, DBSCAN noise flags, stream alarms after a level shift with no
false alarms on the in-control prefix, typed shape/degeneracy errors;
record emission (AUROC present for scores, absent for labels; absent ≠
zero; missed detections carry no delay row).

### Benchmark fingerprint

`scirust-srcc-bench/examples/industrial_protocol_demo.rs` — the full
protocol on a deterministic 6-machine synthetic plant: manifest; grouped
split; coherent-cluster training contamination sweep over four estimators;
leave-one-machine-out paired OLS-vs-Huber bootstrap; five anomaly detectors
against an evaluation-set coherent cluster; CUSUM/EWMA on a constructed
level shift; all rows re-emitted as `BenchRecord` JSONL.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
167c13def9ef160ac7aec91955485f696d4c5d7ea74628584b47d8c66746bfc9
```

Honest readings printed by the demo itself: median-of-means is poor on this
fixture even at zero contamination (its block-majority heuristic is a
documented weakness, shown); the six-machine paired interval
`[−1.24e−2, 7.44e−3]` straddles zero — the demo's own conclusion is "no
significant difference", exercising the anti-overclaim path; the
transductive density methods **fail structurally** on the coherent
alternative cluster (LOF AUROC 0.39, DBSCAN balanced accuracy 0.50 — a
dense fake cluster is not "noise") while train-distribution methods
(Isolation Forest, Mahalanobis, T²) separate it perfectly — precisely the
adversarial scenario the trust phases exist for.

### Compatibility

Purely additive: one new workspace member; no existing crate modified; all
eight historical example outputs re-verified byte-identical (`9699b515…`,
`bad8cf07…`, `b2a1a041…`, `4943a4e4…`, `fc0dcbca…`, `7d0a6d72…`,
`8bb566fb…`, `c7da6bb4…`). Edition 2024 (the `scirust-bench-schema`
precedent); `sha2`/`serde` are established workspace dependencies. `cargo
clippy --workspace --all-targets --locked -- -D warnings` and `cargo fmt
--all --check` pass.

### Known limitations / deferred

- SRCC-family adapters are not in the 727 demo: SRCC consumes transport
  views, not tabular rows; the preregistration fixes the view construction
  and phase 728 wires those adapters against real trajectories.
- Runtime/memory metrics are declared side channels; the harness does not
  measure them yet (no timings may enter hashed stdout).
- The paired bootstrap is percentile-only (no BCa correction); the exact
  floor-quantile convention is documented and deterministic.
- `RandomHoldout` deliberately offers no leakage protection and says so;
  grouped/temporal strategies are the industrial defaults.

---

## Phase 728 — Real industrial evaluation under the preregistration

**Crate:** `scirust-srcc-bench` (new `loaders`, `missing`, `srcc_views`
modules + `industrial-eval` binary) · **Report:**
`docs/research/SRCC_INDUSTRIAL_EVALUATION_REPORT.md` · **Data:**
`scirust-srcc-bench/DATASETS.md`, `scripts/fetch_industrial_datasets.sh`

### Design

Runs the phase-727 preregistration on real data. Two industrial families,
three workloads: **C-MAPSS FD001 and FD003** (NASA turbofan run-to-failure,
public domain — the PdM replication pair) and **SECOM** (UCI semiconductor
process/yield, CC BY 4.0 — real anomaly detection). An earlier draft used
in-repo OBD2 automotive telemetry as a third workload; it was removed as
out-of-domain (consumer driving data, not industrial machinery), and the
SRCC replication requirement is met within the turbofan domain by the
FD001/FD003 pair.

New crate surface: `loaders` (typed C-MAPSS / SECOM / OBD2 text parsers —
the OBD2 parser stays in the library, unused by the binary, since it is
tested and harmless), `missing` (train-fitted drop/impute policy that also
guards least-squares rank deficiency by dropping constant training columns),
`srcc_views` (the preregistered transport-view construction: one view per
engine, `(time, row)` order, ≤16 zero-padded channels, per-trajectory median
centering, every failure typed). The `industrial-eval` binary embeds the
frozen `configs/phase728.json` (its SHA-256 in the run metadata), verifies
every input file against a pinned checksum before use, and never touches the
network.

Data discipline: nothing is fetched by `cargo test` or library code; the
only download path is `scripts/fetch_industrial_datasets.sh`, which verifies
archive and extracted-file checksums. The full datasets are git-ignored
(`/data/`); small license-clean head fixtures are committed under
`tests/data/` and the integration tests **skip loudly** when the full data
is absent. Trajectory decimation (stride 20, fixed a priori) keeps certified
branch-and-bound and 100-engine leave-one-out refitting tractable — a
documented subsampling, never tuned on outcomes.

### Results (every outcome, positive and negative)

- **H1 scale invariance — supported and replicated (the strongest result):**
  robust-diagonal clustering preserves 48/48 assignments under the ±unit
  rescalings on both FD001 and FD003; raw Euclidean preserves 15/48 and
  13/48. Decision invariance under unit change, demonstrated on real turbofan
  sensors in both fault-mode regimes.
- **H4 certified vs. greedy — supported and modest:** certified is never
  worse on cluster count (2/48 strictly fewer, 46/48 equal), strictly cheaper
  in cost on 20/48 at equal count, and 48/48 `proven_optimal`. Greedy is
  usually already count-optimal here, so certification most often adds proof
  rather than a better partition — the acceptable-outcome branch the
  preregistration named.
- **H2 robust regression — inconclusive, does not replicate:** Huber beats
  OLS with a CI excluding zero on FD003 (`+1.52 [+0.37,+2.75]`) but not FD001
  (`+0.59 [−0.17,+1.36]`); replication fails, so no superiority is claimed.
  Median-of-means breaks down catastrophically (RMSE ~10⁵) exactly as its
  block-majority assumption predicts.
- **SECOM anomaly — negative, exposed by the frozen test:** every
  score-producing detector lands at or below chance on the frozen test
  (Mahalanobis 0.469, LOF 0.441, isolation forest 0.424 AUROC) despite
  Mahalanobis reaching 0.648 on validation — a validation-to-test collapse
  aggregate reporting would have hidden. Hotelling T² returns a typed
  degenerate-fit failure.
- **Trust — measured null:** the one-view target-shift attack produces 0.0
  projector displacement under both Unweighted and GroupContaminationBound,
  because continuous industrial sources make every exact-source group a
  singleton; the trust margin has no group structure to act on. Not
  applicable here, measured rather than assumed; the trust models remain
  validated on the phase-725 synthetic battery.
- **Stability:** leave-one-out mean Frobenius 0.018 (max 0.354), 38 stable
  dimensions — stable on real data.
- **Streams:** CUSUM/EWMA detect the injected sensor burst (delay 1 and 0)
  but with 9 and 16 pre-onset false alarms — the false-alarm budget is not
  met at the preregistered settings, reported not tuned away.

No result licenses "SRCC is superior"; the one clean win (H1) is a specific
testable property, stated as exactly that.

### Fingerprint

```
SHA-256 (results/industrial_728.jsonl, deterministic scientific content):
d97f23f4c6069ff500579daee51babbcb2cf83a396251a51a66e1b6ff7ebd7a4
```

1197 `BenchRecord` rows + JSON manifests committed under `results/`;
`run_metadata.json` (git commit, toolchain, config hash) is environment
identity, excluded from the determinism fingerprint. The JSONL and every
scientific manifest are byte-identical across runs.

### Tests

7 new library tests (loaders: C-MAPSS RUL derivation and malformed-row
rejection, SECOM missing-value and label parsing, OBD2 header handling;
missing policy: drop/impute with train-only medians, typed degeneracies) and
3 integration tests over the committed head fixtures (C-MAPSS RUL-0 at last
cycle, SECOM binary labels, plus a full-data check that skips loudly when
absent). `srcc_views` adds 5 tests (canonical order, per-trajectory
centering, horizon, typed errors, determinism). 71 crate tests total.

### Compatibility

Purely additive: no existing crate touched; `parse_obd2` remains exported
but unused by the binary. `cargo clippy --workspace --all-targets --locked
-- -D warnings` and `cargo fmt --all --check` pass; all eight historical
example SHAs re-verified byte-identical.

### Known limitations / deferred

- The C-MAPSS RUL regression is a deliberately simple pooled linear baseline
  (RUL is piecewise); the point is the contamination *comparison*, not
  absolute RUL accuracy.
- Decimation trades statistical power for tractability (100-engine CIs, ~10
  cycles per engine fit).
- Runtime/memory unmeasured (declared side channels).
- Trust and shadow-deployment integration of these results is phase 729.
