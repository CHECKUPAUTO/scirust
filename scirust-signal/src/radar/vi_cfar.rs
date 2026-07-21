//! Variability-Index CFAR (VI-CFAR): a composite detector that adapts its
//! noise estimator to cell-averaging (CA), greatest-of (GO), smallest-of (SO),
//! or a robust pooled trimmed mean, based on two per-CUT statistics computed
//! from the reference cells.
//!
//! # Provenance
//!
//! Classical VI-CFAR is due to M. E. Smith and P. K. Varshney, "Intelligent
//! CFAR processor based on data variability," *IEEE Transactions on Aerospace
//! and Electronic Systems*, vol. 36, no. 3, pp. 837-847, 2000 (an earlier
//! version appeared as "VI-CFAR: a novel CFAR algorithm based on data
//! variability," IEEE 1997 National Radar Conference). This module's classical
//! path implements the qualitative structure that is consistently described
//! across the literature surveyed for this implementation — a per-half
//! variability classification plus a mean-ratio edge test selecting among
//! CA/GO/SO — **not** a transcription of Smith & Varshney's full original
//! decision table, which was not accessible in this environment (the
//! dissertation reproducing it, M. E. Smith, "Application of the variability
//! index (VI) statistic to radar CFAR processing," Syracuse University, is
//! available only through ProQuest). Where the accessible literature was
//! ambiguous or inaccessible, this module makes an explicit, documented
//! choice rather than guessing at an unverified table — see "Switching
//! structure" below for exactly which parts are which.
//!
//! # The Variability Index
//!
//! For a reference half-window with mean `μ` and variance `σ²`,
//!
//! ```text
//! VI = 1 + σ²/μ²
//! ```
//!
//! **This module uses the *sample* variance** `s² = M2/(n-1)` in `VI`, not the
//! population variance `M2/n` that a from-scratch reading of the general
//! CFAR literature might default to. This is a deliberate, evidence-based
//! choice: an openly accessible reproduction of Smith & Varshney's VI
//! (H. Xu et al., "FPGA Implementation of Efficient CFAR Algorithm for Radar
//! Systems," *Sensors* 23(2):954, 2023, eq. 6) writes it explicitly as
//! `VI = 1 + [1/(n-1)]·Σ(xᵢ-x̄)²/x̄²` — the `n-1` divisor is the unbiased
//! sample variance, not `n`. [`variability_index`] is convention-agnostic
//! (it just computes `1 + variance/mean²`); this module calls it with
//! [`crate::sliding_stats::SlidingMoments::sample_variance`]/the
//! sample-variance form of its own internal half-window statistics
//! specifically because of that citation. Callers who want the
//! population-variance form for their own purposes can call
//! [`variability_index`] with
//! [`crate::sliding_stats::SlidingMoments::population_variance`] instead —
//! both are available from the underlying
//! [`crate::sliding_stats`] primitive, and this module does not hide that
//! choice.
//!
//! Equivalent raw-moment form (verified by substitution, not just quoted):
//! with sample variance `s² = [Σxᵢ² - n·x̄²]/(n-1)`,
//!
//! ```text
//! VI = 1 + [Σxᵢ² - n·x̄²] / [(n-1)·x̄²] = 1 + [ (1/(n-1))Σxᵢ² - (n/(n-1))·x̄² ] / x̄²
//! ```
//!
//! which is the sample-variance analogue of the population-variance identity
//! `VI = (1/N)Σxᵢ² / μ²` quoted in some treatments (that simpler form holds
//! exactly only for the *population*-variance convention, `σ²_pop = M2/N`:
//! `1 + M2/(N·μ²) = 1 + [Σxᵢ² - Nμ²]/(Nμ²) = (Σxᵢ²/N)/μ²`). Both forms are
//! algebraically equivalent to their own variance convention; this module
//! does not use the population raw-moment shortcut because it uses sample
//! variance (see above).
//!
//! For a homogeneous, unit-mean-exponential half-window (`Var = Mean²`, the
//! Rayleigh-power clutter model [`super::clutter`] documents), `VI → 2` as
//! `n → ∞`; finite-`n` fluctuations around 2 are exactly what the classifier
//! below tests against `k_vi`.
//!
//! # The Mean Ratio
//!
//! ```text
//! MR = max(μ_lag, μ_lead) / min(μ_lag, μ_lead)      (so MR ≥ 1)
//! ```
//!
//! [`mean_ratio`]'s zero-mean policy (both means are assumed non-negative,
//! the domain [`InputValidationPolicy::RejectNegative`] enforces):
//!
//! * **both means exactly zero** → `MR := 1.0` (two all-zero windows are
//!   trivially "equal," not an error);
//! * **exactly one mean zero** → `MR := +∞` (maximal, well-defined asymmetry
//!   — `f64::INFINITY` compares correctly against any finite `k_mr` with no
//!   special-casing needed downstream);
//! * **very small positive means** → the plain ratio; no epsilon guard, since
//!   a genuine large proportional difference between two tiny means is a
//!   real signal, not noise to be hidden;
//! * **negative means** → only reachable under
//!   [`InputValidationPolicy::AllowNegative`]; `MR`'s `≥ 1` guarantee and its
//!   homogeneity interpretation are **not** established for that case by this
//!   module (the formula is still evaluated, but its meaning here is
//!   unverified outside the non-negative domain);
//! * **non-finite means** → cannot occur: every pushed/sliced sample is
//!   validated finite before it can contribute to a mean (see
//!   [`crate::sliding_stats`]).
//!
//! # Reference-window layout
//!
//! ```text
//! [ lagging reference (reference_cells) ][ lagging guard (guard_cells) ][ CUT ][ leading guard (guard_cells) ][ leading reference (reference_cells) ]
//! ```
//!
//! For a finite slice, the CUT is the sample at the index named in
//! [`CfarDecision::cut_index`]; a cell within `reference_cells + guard_cells`
//! of either edge of the slice has no full window on that side and is never
//! evaluated ([`EdgePolicy::Exclude`], currently the only policy — see its
//! docs). For [`CfarStreamDetector`], a decision for the sample pushed at
//! stream position `t` is emitted only once the leading window has filled,
//! i.e. after `guard_cells + reference_cells` further pushes — see
//! [`CfarStreamDetector::push`] for the exact latency contract.
//!
//! # Switching structure
//!
//! Under [`DetectorPolicy::ClassicalViCfar`], each half is classified
//! `non-homogeneous` if its `VI > k_vi`, else `homogeneous` (this
//! classify-by-threshold structure, and the mean-ratio edge test, are
//! consistently described across every source surveyed for this
//! implementation, including the original abstract and independent
//! reproductions of it):
//!
//! | lagging | leading | MR | mode |
//! |---|---|---|---|
//! | homogeneous | homogeneous | `≤ k_mr` | [`CfarMode::Ca`] |
//! | homogeneous | homogeneous | `> k_mr` | [`CfarMode::Go`] (clutter edge, no interferer) |
//! | homogeneous | non-homogeneous | — | [`CfarMode::So`] |
//! | non-homogeneous | homogeneous | — | [`CfarMode::So`] |
//! | non-homogeneous | non-homogeneous | — | [`CfarMode::RobustTrimmed`] or [`CfarMode::RobustCensored`], whichever [`CfarConfig::robust_estimator`] configures |
//!
//! **Rows 1, 2 and 5 are this module's own explicit design choices, clearly
//! isolated as such — not a transcription of a verified source:**
//!
//! * Row 5 (**double contamination**) is exactly the case the accessible
//!   literature says degrades classical VI-CFAR's detection probability
//!   without giving (in anything this implementation could reach) the
//!   classical fallback rule — and it is exactly the case this module's
//!   robust extension (see below) targets. Rather than guess a classical rule
//!   here, SciRust always applies the configured
//!   [`RobustNoiseEstimator`] pooled over both halves.
//! * Row 3/4 (**exactly one half non-homogeneous**) is implemented as
//!   smallest-of — the mean of the *smaller* half — rather than literally
//!   "the mean of whichever half tested homogeneous" (the closer reading of
//!   "VI-CFAR dynamically chooses the leading, lagging, or combined
//!   reference cells" that recurs across sources). In the motivating case (an
//!   interferer raises both the mean *and* the VI of one half) the two
//!   coincide; this module uses the SO rule specifically because it is
//!   independently well-established and independently calibrated here (see
//!   "Threshold calibration"), not as a claim that it is bit-for-bit the
//!   historical rule.
//!
//! [`DetectorPolicy::Ca`], [`DetectorPolicy::Go`], [`DetectorPolicy::So`] and
//! [`DetectorPolicy::AlwaysRobust`] force the corresponding estimator for
//! every CUT, bypassing the switch entirely — the "independently selectable
//! baseline" Step 7 of this module's design brief calls for, and the
//! controlled comparisons the test suite below uses.
//!
//! `k_vi`/`k_mr` are **required configuration, with no built-in default**.
//! Published values found during this implementation's literature survey
//! disagree by design point: `k_vi = 4.76, k_mr = 1.806` for a stated
//! classification-error design of `α = 4×10⁻⁴, β = 0.1`, versus
//! `k_vi = 6.72, k_mr = 2.064` cited elsewhere for an unstated design point —
//! confirming these are context/`N`-dependent design parameters, not universal
//! constants, so this module refuses to silently pick one as "the" default.
//! [`calibrate_k_mr`] (exact, via the `F(2n,2n)` distribution `MR` follows
//! under the switch's null case) and [`calibrate_k_vi`] (Monte Carlo, seeded
//! and deterministic — `VI` has no equivalently simple exact distribution)
//! give a starting point for the `α` half of that design point from first
//! principles instead of copying a published pair verbatim; they do not
//! calibrate `β` (the missed-non-homogeneity rate), which is inherently
//! specific to an assumed interferer/target model this module does not
//! invent — see those functions' own docs.
//!
//! # Threshold calibration
//!
//! All four modes hold a *design* `P_fa` under the i.i.d.
//! unit-mean-exponential reference-cell model (the Rayleigh-power clutter
//! case [`super::clutter`] documents) — but **not with the same analytical
//! rigor**, and this module does not blur that distinction:
//!
//! * **CA** — the exact closed form is re-derived (not just quoted) in
//!   [`super::cfar::ca_cfar_alpha`]'s module and reused directly: the noise
//!   estimate `(μ_lag+μ_lead)/2` over `N = 2·reference_cells` cells is
//!   `Gamma(N,1)/N`-distributed, whose Laplace transform gives the exact
//!   identity `(1+α/N)^{-N} = P_fa`.
//! * **RobustTrimmed** — exact under the same model, derived in this module
//!   from the Rényi representation of exponential order statistics (the same
//!   family of exact result [`super::cfar::os_cfar_alpha`] already uses for a
//!   single order statistic, generalized here to a trimmed *mean*): solved by
//!   bisection, not a closed form, but exact up to bisection/floating-point
//!   precision, not statistical sampling error.
//! * **GO / SO** — also exact under the same model, via a closed-form-up-to-
//!   quadrature derived in `pfa_so_exact`/`pfa_go_exact`: for a
//!   continuous nonnegative `X` with `P(X>0)=1`, integrating `E[e^{-tX}]` by
//!   parts gives `E[e^{-tX}] = 1 - t∫₀^∞ e^{-tm}P(X>m)dm` (verified here by
//!   direct integration and by checking both boundary limits, not assumed).
//!   With `M=min(S1,S2)`, `P(M>m)=Q_n(m)²` (`Q_n` = the reference half-
//!   window's `Gamma(n,1)` survival function), giving
//!   `P_fa_SO(α) = 1 - t·∫e^{-tm}Q_n(m)²dm`; with `M'=max(S1,S2)`,
//!   `P(M'>m)=2Q_n(m)-Q_n(m)²`, giving `P_fa_GO(α) = 2(1+t)^{-n} - P_fa_SO(α)`
//!   (the `(1+t)^{-n}` term itself closed-form via the geometric sum
//!   `Σe^{-(t+1)m}m^k/k!`). The integral is evaluated by
//!   [`scirust_solvers::quadrature::simpson_adaptive`] against
//!   [`scirust_stats::Gamma`]'s tested `sf` (survival function) — both
//!   pre-existing, independently-tested SciRust primitives, not new special-
//!   function code — over a truncated range whose remainder is bounded by
//!   `0≤Q_n≤1` and *verified negligible at runtime* (not just assumed) before
//!   the result is trusted; see `truncation_bound`. This closed form is
//!   cross-checked in this module's own tests against an independently
//!   implemented Monte-Carlo estimate of the same `P_fa`, and again,
//!   empirically, by the `P_fa` validation tests in
//!   `tests/vi_cfar_monte_carlo.rs` using a third, independent RNG
//!   (`scirust-stats`'s `SplitMix64`).
//!
//! **CFAR claim scope**: CA, RobustTrimmed *and* GO/SO now have an *exact*
//! `P_fa` under the stated model (GO/SO's "exact" means deterministic,
//! bounded-error numerical integration of a closed-form integral, not
//! Monte-Carlo sampling — the quadrature/truncation error is provably small,
//! not merely small with high probability). What remains **not** shown is
//! that the `ClassicalViCfar` switch *as a whole* (which mixes all four modes
//! based on data) holds an exact overall `P_fa` — only that each branch,
//! taken in isolation, is calibrated as described above; the composite
//! switching detector's false-alarm rate has only been checked empirically
//! (`tests/vi_cfar_monte_carlo.rs`, including a multi-scenario
//! `chi_square_gof` check — see that file), not derived analytically. Do not
//! read "CFAR" applied to the *composite* detector as a proven invariance
//! claim.
//!
//! **Model scope**: every calibration above — CA, GO, SO, RobustTrimmed,
//! RobustCensored, and the composite switch built from them — targets one
//! specific clutter model: i.i.d. unit-mean-exponential power (equivalently,
//! Rayleigh-amplitude clutter, [`super::clutter`]'s baseline case). This is
//! the classical calm-sea/thermal-noise model, not a claim about every
//! clutter environment. Real sea clutter at low grazing angles and high
//! range resolution is well documented to be spikier — heavier-tailed —
//! than this (Weibull-amplitude with shape `< 2`, or log-normal, both
//! already modeled in [`super::clutter`]). `tests/vi_cfar_non_rayleigh_clutter.rs`
//! *measures* (does not merely assert) what happens then: every mode's
//! observed `P_fa` rises to roughly 3-5x the design target under
//! moderately-to-severely spiky synthetic clutter parameterized from ranges
//! commonly cited for real X-band sea clutter (that file has no recorded
//! sensor data to validate against — see its own docs for why). The ranking
//! across modes there is not what "robust beats classical" would predict:
//! GO holds up best, and `RobustTrimmed`/`RobustCensored` — calibrated via
//! the same exponential-order-statistics argument as CA/GO/SO — degrade
//! *worst*, since discarding/censoring cells raises estimator variance
//! without correcting a wrong-distribution-family mismatch trimming was
//! never designed to fix. Do not read any mode's calibration here as
//! validated for non-Rayleigh clutter; a deployment against real spiky
//! clutter needs its own calibration against the actual clutter model in
//! force, not this module's exponential-power constants.
//!
//! # Robust double-contamination strategy
//!
//! Both [`RobustNoiseEstimator`] variants pool both reference half-windows
//! (`2·reference_cells` cells) and sort them with an allocation-free
//! `sort_unstable_by(f64::total_cmp)` (deterministic ordering; `total_cmp`
//! gives NaN — never actually reachable here, since input is validated finite
//! — a defined total order rather than the panics/ambiguity of `partial_cmp`):
//!
//! * [`RobustNoiseEstimator::TrimmedMean`] discards `trim_low` smallest and
//!   `trim_high` largest, averaging the rest ([`trimmed_mean`]).
//! * [`RobustNoiseEstimator::CensoredMean`] instead *replaces* (Winsorizes)
//!   those cells with the nearest retained value and averages **all**
//!   `2·reference_cells` cells ([`censored_mean`]). This started as a
//!   deliberately deferred variant — the design brief explicitly sanctions
//!   "start with a correctness-first trimmed mean," and a second robust
//!   estimator without equally rigorous calibration would violate the "never
//!   silently substitute an unverified factor" principle this module holds
//!   to — but the Rényi-spacings argument `trimmed_mean_alpha` uses turns
//!   out to survive replacement exactly as it does discarding (each censored
//!   term stays *linear* in the underlying exponential spacings; the
//!   multiplicities `trim_low`/`trim_high` only rescale coefficients, they
//!   never attach a max/min or other non-additive dependency — see
//!   `censored_mean_alpha`'s derivation), so it was verified and
//!   implemented with the same exactness as `TrimmedMean`, not left as a
//!   stub.
//!
//! # Complexity
//!
//! [`evaluate_slice`] is `O(len · reference_cells)`: each CUT's half-window
//! statistics are recomputed directly from the slice (matching this crate's
//! existing [`super::cfar`]/[`super::cfar_variants`] convention), not
//! incrementally. [`CfarStreamDetector::push`] is `O(1)` amortized per sample
//! for the CA/GO/SO path (backed by [`crate::sliding_stats::SlidingMoments`],
//! whose `push` is O(1)) and `O(reference_cells · log(reference_cells))`-ish... in practice
//! `O(reference_cells)` for the `sort_unstable` call, only when
//! [`CfarMode::RobustTrimmed`] is actually selected for that sample. Storage
//! is `O(reference_cells)` (two `SlidingMoments` windows plus one scratch
//! buffer), never reallocated after construction.

use thiserror::Error;

use scirust_stats::Distribution;

use crate::sliding_stats::{SampleDomain, SlidingMomentsDyn, SlidingMomentsError};

// ============================================================================
// Pure math: Variability Index and Mean Ratio
// ============================================================================

/// `VI = 1 + variance/mean²`. Convention-agnostic: pass
/// [`crate::sliding_stats::SlidingMoments::sample_variance`] or
/// [`crate::sliding_stats::SlidingMoments::population_variance`] depending
/// which convention you need (see the module docs for which one this
/// module's own detector uses,
/// and why).
///
/// `mean == 0.0` is an explicit, defined case, not left to fall out of
/// `0.0/0.0`: in the non-negative power domain this detector is designed for,
/// a zero mean forces every cell in the window to be exactly zero (hence
/// `variance == 0.0` too), the most homogeneous a window can possibly be — so
/// `variability_index(0.0, _)` is defined as `1.0`, the same value a
/// zero-variance *nonzero*-mean window gets, rather than `NaN`. This matters
/// beyond aesthetics: an *undefined* `NaN` compares `false` against any
/// threshold, which would have made an all-zero window's classification an
/// accident of floating-point comparison semantics rather than a documented
/// decision; defining it as `1.0` makes "an all-zero window is homogeneous"
/// an explicit contract instead.
pub fn variability_index(mean: f64, variance: f64) -> f64 {
    if mean == 0.0
    {
        1.0
    }
    else
    {
        1.0 + variance / (mean * mean)
    }
}

/// `MR = max(mean_a, mean_b) / min(mean_a, mean_b)`, with the zero-mean
/// policy documented in the module docs (`MR := 1.0` if both are zero;
/// `+∞` if exactly one is zero). Assumes non-negative means for the `≥ 1`
/// guarantee to hold; see the module docs for the negative-mean case.
pub fn mean_ratio(mean_a: f64, mean_b: f64) -> f64 {
    let hi = mean_a.max(mean_b);
    let lo = mean_a.min(mean_b);
    if hi == 0.0 { 1.0 } else { hi / lo }
}

// ============================================================================
// Configuration
// ============================================================================

/// Behavior for cells near the beginning/end of a finite slice (or, for
/// streaming, before enough samples have arrived) that do not have a full
/// reference window on both sides.
///
/// Currently the only policy: such cells are never evaluated (finite slice:
/// absent from the returned `Vec`; streaming: `push` returns `Ok(None)`).
/// Written as an enum (rather than being an unconditional, undocumented
/// behavior) so a future edge-extension policy has somewhere to go without
/// an API break — no such policy is implemented today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EdgePolicy {
    /// Cells without a full reference window on both sides are skipped.
    #[default]
    Exclude,
}

/// Whether pushed/sliced power samples must be non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputValidationPolicy {
    /// Reject negative samples (the power/energy domain this detector's
    /// formulas are designed for). Finiteness is *always* required regardless
    /// of this policy.
    RejectNegative,
    /// Accept negative samples. [`mean_ratio`]'s `≥ 1` guarantee and this
    /// module's homogeneity interpretation of `MR` are not established in
    /// this regime — see the module docs.
    AllowNegative,
}

/// `k_vi`/`k_mr` for [`DetectorPolicy::ClassicalViCfar`]. Required,
/// non-defaulted configuration — see the module docs for why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwitchingThresholds {
    /// A half-window is classified non-homogeneous when its `VI > k_vi`.
    pub k_vi: f64,
    /// Both halves (when both are homogeneous) are classified as having
    /// "the same" clutter level when `MR <= k_mr`.
    pub k_mr: f64,
}

/// A robust estimator of the pooled reference-window noise level, used when
/// both half-windows are contaminated (see the module docs, "Robust
/// double-contamination strategy").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustNoiseEstimator {
    /// Sort the pooled `2 * reference_cells` cells, discard `trim_low`
    /// smallest and `trim_high` largest, average the rest.
    TrimmedMean {
        /// Cells discarded from the low end.
        trim_low: usize,
        /// Cells discarded from the high end.
        trim_high: usize,
    },
    /// Sort the pooled `2 * reference_cells` cells, *replace* (Winsorize)
    /// the `trim_low` smallest with the value of the smallest retained cell
    /// and the `trim_high` largest with the value of the largest retained
    /// cell, then average *all* cells (none discarded). See
    /// [`censored_mean`] and `censored_mean_alpha` for the derivation —
    /// unlike `TrimmedMean`'s discard, this was judged too likely to break
    /// the exact-Pfa derivation to attempt without independent verification;
    /// it was verified (the Rényi-spacings argument stays linear in the
    /// underlying exponential spacings under replacement, exactly as it does
    /// under discarding) before being implemented.
    CensoredMean {
        /// Cells replaced (Winsorized) at the low end.
        trim_low: usize,
        /// Cells replaced (Winsorized) at the high end.
        trim_high: usize,
    },
}

impl RobustNoiseEstimator {
    /// `(trim_low, trim_high)` regardless of variant — both share the same
    /// shape and the same validation rule (`trim_low + trim_high < n_ref`).
    fn trim_counts(self) -> (usize, usize) {
        match self
        {
            Self::TrimmedMean {
                trim_low,
                trim_high,
            }
            | Self::CensoredMean {
                trim_low,
                trim_high,
            } => (trim_low, trim_high),
        }
    }
}

/// Which noise estimator to apply, and how to choose it per CUT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectorPolicy {
    /// Always cell-average both halves (optimal only in truly homogeneous
    /// clutter; the classical CA-CFAR baseline).
    Ca,
    /// Always take the larger of the two half-window means.
    Go,
    /// Always take the smaller of the two half-window means.
    So,
    /// The classical VI/MR-driven switch among CA/GO/SO, falling back to
    /// [`CfarConfig::robust_estimator`] on double contamination (SciRust's
    /// extension). See the module docs, "Switching structure".
    ClassicalViCfar(SwitchingThresholds),
    /// Always apply `robust_estimator` over the pooled window, bypassing
    /// VI/MR classification entirely. Useful as a controlled baseline/
    /// comparison and for benchmarking.
    AlwaysRobust,
}

/// Complete configuration for one VI-CFAR evaluation (finite-slice or
/// streaming). See the module docs for the mathematical contract behind each
/// field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfarConfig {
    /// Reference cells *per side* (lagging and leading are the same size —
    /// see the module docs' reference-window layout). Must be `>= 2` (a
    /// sample variance, hence `VI`, needs at least two cells) and, well
    /// beyond any real radar reference window, `<= 100_000` — a practical
    /// cap keeping calibration cost bounded (see `MAX_PRACTICAL_REFERENCE_CELLS`
    /// and [`CfarError::ReferenceWindowTooLarge`]).
    pub reference_cells: usize,
    /// Guard cells per side (may be `0`).
    pub guard_cells: usize,
    /// Design false-alarm probability, `0 < pfa < 1`.
    pub pfa: f64,
    /// Behavior for cells without a full reference window.
    pub edge_policy: EdgePolicy,
    /// Whether negative power samples are rejected.
    pub input_validation: InputValidationPolicy,
    /// Which estimator(s) to use and how to choose among them.
    pub detector: DetectorPolicy,
    /// The estimator applied when [`DetectorPolicy::AlwaysRobust`] is active,
    /// or when [`DetectorPolicy::ClassicalViCfar`] detects double
    /// contamination.
    pub robust_estimator: RobustNoiseEstimator,
}

/// Errors from [`CfarConfig`] validation and per-sample input checks.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CfarError {
    /// `reference_cells < 2`.
    #[error(
        "reference_cells must be at least 2 (got {0}); a per-half sample \
         variance (hence the Variability Index) needs at least two cells"
    )]
    TooFewReferenceCells(usize),
    /// `reference_cells`/`guard_cells` are too large to be usable: either
    /// computing the reference-window size (`reference_cells + guard_cells`,
    /// or `2 * reference_cells`) would overflow `usize` arithmetic, or
    /// `reference_cells` exceeds [`MAX_PRACTICAL_REFERENCE_CELLS`] — caught
    /// here, deterministically, rather than left to panic on overflow, or
    /// (for the practical bound) to hang inside calibration, wherever that
    /// happens to occur downstream.
    #[error(
        "reference_cells={reference_cells}, guard_cells={guard_cells} is not a usable \
         reference window (overflows reference-window-size arithmetic, or exceeds the \
         practical limit of {MAX_PRACTICAL_REFERENCE_CELLS} reference cells)"
    )]
    ReferenceWindowTooLarge {
        reference_cells: usize,
        guard_cells: usize,
    },
    /// `pfa` outside `(0, 1)`, or non-finite.
    #[error("pfa must satisfy 0 < pfa < 1 (got {0})")]
    InvalidPfa(f64),
    /// `k_vi`/`k_mr` non-finite, non-positive, or `k_mr < 1.0`.
    #[error(
        "switching thresholds must be finite with k_vi > 0 and k_mr >= 1.0 \
         (got k_vi={k_vi}, k_mr={k_mr})"
    )]
    InvalidSwitchingThresholds { k_vi: f64, k_mr: f64 },
    /// `trim_low + trim_high` leaves no retained cell in the pooled window.
    #[error(
        "trim counts leave no retained cell: pooled reference window has \
         {n_ref} cells, trim_low={trim_low}, trim_high={trim_high}"
    )]
    InvalidTrimCounts {
        n_ref: usize,
        trim_low: usize,
        trim_high: usize,
    },
    /// A sample was not finite.
    #[error("sample #{index} is not finite: {value}")]
    NonFiniteSample { index: usize, value: f64 },
    /// A sample was negative under [`InputValidationPolicy::RejectNegative`].
    #[error(
        "sample #{index} is negative ({value}) but the configured input-validation policy \
         rejects negative power samples"
    )]
    NegativeSample { index: usize, value: f64 },
    /// A [`SlidingMomentsDyn`] instance backing a streaming detector reported
    /// a numerical-integrity failure.
    #[error("sliding-moments numerical integrity failure: {0}")]
    SlidingMoments(#[from] SlidingMomentsError),
    /// The exact GO/SO/CensoredMean quadrature-based calibration failed:
    /// either the quadrature routine itself did not converge, or the
    /// runtime-verified truncation bound (see `truncation_bound`) turned
    /// out not to be negligible for the given `reference_cells`. The error
    /// message names which.
    #[error("exact threshold calibration failed: {0}")]
    ExactCalibrationFailed(String),
}

/// Practical upper bound on `reference_cells`, enforced by
/// [`CfarConfig::validate`] — not an arithmetic limit (`usize` overflow is
/// caught separately, well past this point) but a *cost* one:
/// `CalibratedThresholds::compute`'s `TrimmedMean`/`CensoredMean` calibration
/// path (`trimmed_mean_alpha`/`censored_mean_alpha`) evaluates a product over
/// all `n_ref = 2 * reference_cells` pooled cells, and does so up to roughly
/// [`BISECTION_MAX_ITERS`] times inside [`bisect_decreasing`] — `O(n_ref)`
/// per call. A `reference_cells` value nowhere near overflowing (so not
/// caught by the overflow check above) but merely huge — found by widening
/// this module's own proptest strategy to draw large `usize`s, which then
/// hung indefinitely instead of finishing — would make `CfarDetector::new`/
/// `evaluate_slice`/`CfarStreamDetector::new` block for an unbounded time
/// instead of returning a fast, structured error. No real radar reference
/// window is remotely close to this bound (tens to low hundreds of cells is
/// typical); it exists purely to turn that unbounded hang into an immediate
/// `Err`.
const MAX_PRACTICAL_REFERENCE_CELLS: usize = 100_000;

impl CfarConfig {
    /// Validate every field. Called internally by [`evaluate_slice`] and
    /// [`CfarStreamDetector::new`]; exposed so callers can validate a config
    /// once and reuse it with confidence.
    pub fn validate(&self) -> Result<(), CfarError> {
        if self.reference_cells < 2
        {
            return Err(CfarError::TooFewReferenceCells(self.reference_cells));
        }
        // Every downstream computation needs `reference_cells + guard_cells`
        // (the half-window span, e.g. in `CfarDetector::evaluate`) and
        // `2 * reference_cells` (the pooled robust-estimator window size) to
        // fit in a `usize` without wrapping — checked here, once, rather
        // than left to panic wherever that arithmetic happens to occur.
        // `reference_cells` alone is also capped well below where any of
        // that could overflow (see `MAX_PRACTICAL_REFERENCE_CELLS`), since
        // calibration cost — not just overflow — must stay bounded.
        let window_size_overflows = self.reference_cells.checked_add(self.guard_cells).is_none()
            || self.reference_cells.checked_mul(2).is_none();
        let window_size_impractical = self.reference_cells > MAX_PRACTICAL_REFERENCE_CELLS;
        if window_size_overflows || window_size_impractical
        {
            return Err(CfarError::ReferenceWindowTooLarge {
                reference_cells: self.reference_cells,
                guard_cells: self.guard_cells,
            });
        }
        if !(self.pfa.is_finite() && self.pfa > 0.0 && self.pfa < 1.0)
        {
            return Err(CfarError::InvalidPfa(self.pfa));
        }
        if let DetectorPolicy::ClassicalViCfar(t) = self.detector
        {
            if !(t.k_vi.is_finite() && t.k_vi > 0.0 && t.k_mr.is_finite() && t.k_mr >= 1.0)
            {
                return Err(CfarError::InvalidSwitchingThresholds {
                    k_vi: t.k_vi,
                    k_mr: t.k_mr,
                });
            }
        }
        let (trim_low, trim_high) = self.robust_estimator.trim_counts();
        let n_ref = 2 * self.reference_cells; // safe: checked above
        let trim_counts_invalid = match trim_low.checked_add(trim_high)
        {
            Some(sum) => sum >= n_ref,
            None => true,
        };
        if trim_counts_invalid
        {
            return Err(CfarError::InvalidTrimCounts {
                n_ref,
                trim_low,
                trim_high,
            });
        }
        Ok(())
    }

    fn validate_sample(&self, index: usize, value: f64) -> Result<(), CfarError> {
        if !value.is_finite()
        {
            return Err(CfarError::NonFiniteSample { index, value });
        }
        if self.input_validation == InputValidationPolicy::RejectNegative && value < 0.0
        {
            return Err(CfarError::NegativeSample { index, value });
        }
        Ok(())
    }
}

// ============================================================================
// Diagnostic output
// ============================================================================

/// Which noise estimator produced a [`CfarDecision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CfarMode {
    /// Cell-averaging: `(μ_lag + μ_lead) / 2`.
    Ca,
    /// Greatest-of: `max(μ_lag, μ_lead)`.
    Go,
    /// Smallest-of: `min(μ_lag, μ_lead)`.
    So,
    /// Pooled trimmed mean over both halves.
    RobustTrimmed,
    /// Pooled censored (Winsorized) mean over both halves.
    RobustCensored,
}

/// Full diagnostic record for one cell-under-test, suitable for research and
/// testing. See the module docs for every field's mathematical definition.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CfarDecision {
    /// Index of the CUT (in the original finite slice, or the streaming
    /// sample counter).
    pub cut_index: usize,
    /// The CUT's own power value.
    pub cut_power: f64,
    /// Lagging half-window mean.
    pub lagging_mean: f64,
    /// Leading half-window mean.
    pub leading_mean: f64,
    /// Lagging half-window Variability Index (sample-variance convention;
    /// see the module docs).
    pub lagging_vi: f64,
    /// Leading half-window Variability Index (sample-variance convention).
    pub leading_vi: f64,
    /// `max/min` of the two half-window means.
    pub mean_ratio: f64,
    /// The noise level estimate actually used (meaning depends on `mode`).
    pub noise_estimate: f64,
    /// `alpha(mode) * noise_estimate` — the decision threshold.
    pub threshold: f64,
    /// `cut_power > threshold`.
    pub detected: bool,
    /// Which estimator was used for this CUT.
    pub mode: CfarMode,
}

// ============================================================================
// Threshold calibration
// ============================================================================

/// Exact (given the i.i.d. unit-mean-exponential reference-cell model)
/// threshold factor for a trimmed mean over `n_ref` pooled reference cells,
/// keeping the middle `n_ref - trim_low - trim_high` order statistics.
///
/// Derivation (Rényi representation of exponential order statistics): let
/// `X₍₁₎ ≤ ... ≤ X₍ₙ₎` be the order statistics of `n_ref` i.i.d. `Exp(1)`
/// variables. Rényi's representation writes `X₍ₘ₎ = Σ_{i=1}^{m} Eᵢ/(n_ref-i+1)`
/// for i.i.d. `Eᵢ ~ Exp(1)` ("spacings"). Summing the kept order statistics
/// `m = trim_low+1 .. n_ref-trim_high` and collecting each `Eᵢ`'s
/// coefficient gives a weighted sum `Σ wᵢ·Eᵢ` with
///
/// ```text
/// wᵢ = kept / (n_ref - i + 1)                         for i <= trim_low
/// wᵢ = (n_ref - trim_high - i + 1) / (n_ref - i + 1)   for trim_low < i <= n_ref - trim_high
/// wᵢ = 0                                                for i > n_ref - trim_high
/// ```
///
/// (`kept = n_ref - trim_low - trim_high`). Since `CUT ~ Exp(1)` independent
/// of the reference cells, and `E[e^{-sE}] = 1/(1+s)` for `E ~ Exp(1)`,
///
/// ```text
/// P_fa(alpha) = E[e^{-alpha * (1/kept) * Σ wᵢ Eᵢ}] = ∏ᵢ 1 / (1 + (alpha/kept)·wᵢ)
/// ```
///
/// a finite product, strictly decreasing in `alpha`, solved here by
/// bisection. Sanity checks performed during development (not asserted at
/// runtime, but exercised by this module's tests): `trim_low=trim_high=0`
/// collapses to `ca_cfar_alpha`'s exact identity `(1+α/N)^{-N}=P_fa`; keeping
/// only the smallest or only the largest order statistic collapses to
/// [`super::cfar::os_cfar_alpha`]'s `k=1`/`k=N` cases exactly.
fn trimmed_mean_alpha(n_ref: usize, trim_low: usize, trim_high: usize, pfa: f64) -> f64 {
    let kept = (n_ref - trim_low - trim_high) as f64;
    let last = n_ref - trim_high;
    let weight = |i: usize| -> f64 {
        let n_i = (n_ref - i + 1) as f64;
        if i <= trim_low
        {
            kept / n_i
        }
        else
        {
            (last - i + 1) as f64 / n_i
        }
    };
    let pfa_of = |alpha: f64| -> f64 {
        (1..=last)
            .map(|i| 1.0 / (1.0 + (alpha / kept) * weight(i)))
            .product()
    };
    bisect_decreasing(pfa_of, pfa)
}

/// Exact (given the i.i.d. unit-mean-exponential reference-cell model)
/// threshold factor for a *censored* (Winsorized) mean over `n_ref` pooled
/// reference cells: the `trim_low` smallest are replaced by
/// `X₍trim_low+1₎` and the `trim_high` largest by `X₍n_ref-trim_high₎`, and
/// *all* `n_ref` values (not just the kept middle) are averaged.
///
/// Derivation: the Winsorized sum is
/// `W = trim_low·X₍trim_low+1₎ + Σ_{m=trim_low+1}^{n_ref-trim_high} X₍m₎ + trim_high·X₍n_ref-trim_high₎`.
/// Substituting the same Rényi representation `X₍ₘ₎ = Σ_{i=1}^{m} Eᵢ/(n_ref-i+1)`
/// used in [`trimmed_mean_alpha`] and collecting each `Eᵢ`'s coefficient
/// (each term of `W` is individually linear in the `Eᵢ`, and a constant
/// multiple — `trim_low·` / `trim_high·` — of an already-linear order
/// statistic is still linear; replacing rather than discarding does not
/// introduce a max/min or any other non-additive dependency) gives, per
/// spacing index `i` (verified directly, not just by pattern-matching
/// against the trimmed case):
///
/// ```text
/// dᵢ = 1 / (n_ref - i + 1)    for i <= trim_low + 1
/// dᵢ = 1 / n_ref              for trim_low + 1 < i <= n_ref - trim_high
/// dᵢ = 0                       for i > n_ref - trim_high
/// ```
///
/// so `W/n_ref = Σ dᵢ·Eᵢ` and, exactly as in `trimmed_mean_alpha`,
///
/// ```text
/// P_fa(alpha) = E[e^{-alpha · Σ dᵢ Eᵢ}] = ∏ᵢ 1 / (1 + alpha·dᵢ)
/// ```
///
/// Sanity check exercised by this module's tests: `trim_low=trim_high=0`
/// gives `dᵢ=1/n_ref` for every `i`, collapsing to `(1+α/n_ref)^{-n_ref}=P_fa`
/// — `ca_cfar_alpha`'s exact identity, as it must (no censoring at all is
/// plain cell-averaging).
fn censored_mean_alpha(n_ref: usize, trim_low: usize, trim_high: usize, pfa: f64) -> f64 {
    let n = n_ref as f64;
    let last = n_ref - trim_high;
    let weight = |i: usize| -> f64 {
        if i <= trim_low + 1
        {
            1.0 / (n_ref - i + 1) as f64
        }
        else
        {
            1.0 / n
        }
    };
    let pfa_of = |alpha: f64| -> f64 {
        (1..=last)
            .map(|i| 1.0 / (1.0 + alpha * weight(i)))
            .product()
    };
    bisect_decreasing(pfa_of, pfa)
}

/// Relative convergence tolerance for [`bisect_decreasing`]/
/// [`bisect_decreasing_fallible`]: once the bracket `[lo, hi]` is this
/// narrow relative to `hi`, `mid` can no longer move (it rounds back to
/// `lo` or `hi`, since there is no representable `f64` strictly between two
/// values this close), so further iterations are a no-op that still costs a
/// full `pfa_of` call. `1e-12` is deliberately a little looser than full
/// `f64` relative precision (`~2.2e-16`) — comfortably beyond what any
/// practical detection design needs from a threshold factor, and already
/// matched to [`pfa_so_exact`]'s own truncation-bound acceptance criterion
/// (`< 1e-12`), so bisection does not chase precision the calibration's own
/// other error source has already given up past.
const BISECTION_REL_TOL: f64 = 1.0e-12;
/// Hard cap on bisection iterations — reached only if the convergence check
/// above somehow never triggers (it always has in this module's testing);
/// kept as a safety bound, not the normal exit path.
const BISECTION_MAX_ITERS: usize = 100;

/// Shared bisection for a strictly-decreasing-in-`alpha`, `P_fa(0) = 1`
/// false-alarm function: brackets by doubling, then bisects to
/// [`BISECTION_REL_TOL`] (or [`BISECTION_MAX_ITERS`], whichever comes
/// first).
fn bisect_decreasing(pfa_of: impl Fn(f64) -> f64, pfa: f64) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while pfa_of(hi) > pfa && hi < 1.0e12
    {
        hi *= 2.0;
    }
    for _ in 0..BISECTION_MAX_ITERS
    {
        if hi - lo <= BISECTION_REL_TOL * hi.max(1.0)
        {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if pfa_of(mid) > pfa
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Same as [`bisect_decreasing`], but for a `P_fa` function that can itself
/// fail (the quadrature-based GO/SO evaluators below).
fn bisect_decreasing_fallible(
    mut pfa_of: impl FnMut(f64) -> Result<f64, CfarError>,
    pfa: f64,
) -> Result<f64, CfarError> {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while pfa_of(hi)? > pfa && hi < 1.0e12
    {
        hi *= 2.0;
    }
    for _ in 0..BISECTION_MAX_ITERS
    {
        if hi - lo <= BISECTION_REL_TOL * hi.max(1.0)
        {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if pfa_of(mid)? > pfa
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// `Q_n(m) = P(Gamma(n,1) > m)` — the survival function of a sum of `n`
/// i.i.d. `Exp(1)` reference cells — via `scirust-stats`'s `Gamma`
/// distribution (`Gamma::new(n, 1.0).sf(m)` is exactly
/// `regularized_gamma_q(n, m)`, tested there against tabulated reference
/// points, an mpmath-checked boundary regression, and an independent
/// `ChiSquared`-reparameterization cross-check).
fn gamma_survival(n: f64, m: f64) -> f64 {
    scirust_stats::Gamma::new(n, 1.0).sf(m)
}

/// How many standard deviations of `Gamma(n,1)` (`std = sqrt(n)`) beyond the
/// mean the quadrature truncation bound (below) uses. `Q_n` decays much
/// faster than any power beyond a handful of standard deviations for any
/// `n ≥ 1`; 50 is an enormous, deliberately non-tight margin, and
/// [`pfa_so_exact`] additionally *verifies at runtime* (not just assumes)
/// that `Q_n` at the resulting bound is negligible before trusting the
/// truncated integral, rather than relying on this margin being "obviously
/// enough" without checking.
const TRUNCATION_MARGIN_STD_DEVS: f64 = 50.0;

fn truncation_bound(n: f64) -> f64 {
    n + TRUNCATION_MARGIN_STD_DEVS * n.sqrt() + 50.0
}

/// Absolute tolerance target for [`pfa_so_exact`]'s quadrature — the
/// dominant cost in exact GO/SO calibration, since [`bisect_decreasing_fallible`]
/// re-evaluates it roughly [`BISECTION_MAX_ITERS`] times. `1e-9` resolves a
/// probability to nine significant digits, far beyond what any practical
/// `P_fa` design specifies (radar `P_fa` specs rarely go below `1e-6`, and
/// never need the *calibration constant* known to better than that);
/// looser than the truncation bound's own `< 1e-12` acceptance criterion, so
/// quadrature error — not truncation error — is now the larger of the two,
/// but both stay negligible relative to any real use of `alpha`. Achieving
/// the previously-used `1e-13` over the wide interval `[0, M_trunc]` this
/// quadrature spans costs many more adaptive subdivisions — each another
/// [`gamma_survival`] (special-function) evaluation — for digits of `alpha`
/// no downstream use reads.
const QUADRATURE_TOLERANCE: f64 = 1.0e-9;

/// Exact `P_fa` for smallest-of over two independent `n`-cell halves
/// (`M = min(S1, S2)`, `S1, S2 ~ Gamma(n,1)` i.i.d.), under the i.i.d.
/// unit-mean-exponential reference-cell model, at threshold factor `alpha`.
///
/// Derivation: for a continuous nonnegative random variable `X` with
/// `P(X>0)=1` and survival function `S`, integrating `E[e^{-tX}]` by parts
/// gives the identity (verified here by direct integration, not assumed)
///
/// ```text
/// E[e^{-tX}] = 1 - t · ∫₀^∞ e^{-tm} S(m) dm
/// ```
///
/// `CUT ~ Exp(1)` independent of the reference cells, so
/// `P_fa(alpha) = P(CUT > (alpha/n)·M) = E[e^{-t·M}]` with `t = alpha/n`.
/// `M = min(S1,S2)` has survival function `P(M>m) = P(S1>m)P(S2>m) = Q_n(m)²`
/// (independence), giving
///
/// ```text
/// P_fa_SO(alpha) = 1 - t · ∫₀^∞ e^{-t·m} Q_n(m)² dm
/// ```
///
/// Sanity checks (also exercised by this module's tests): as `alpha → 0`
/// (`t → 0`), the integral tends to a finite positive constant while `t → 0`,
/// so `P_fa_SO → 1` (matches `P(CUT>0)=1`, the correct limit at a zero
/// threshold); as `alpha → ∞`, `t·∫ → 1` (Watson's-lemma-type leading
/// behavior at `m=0` where `Q_n(0)=1`), so `P_fa_SO → 0`.
///
/// The integral is evaluated by [`scirust_solvers::quadrature::simpson_adaptive`]
/// over `[0, M_trunc]`; the tail `∫_{M_trunc}^∞ e^{-tm}Q_n(m)² dm` is bounded
/// (since `0 ≤ Q_n ≤ 1`) by `∫_{M_trunc}^∞ e^{-tm}dm = e^{-t·M_trunc}/t`, and
/// `Q_n(M_trunc)` is checked at runtime to be negligible (see
/// [`truncation_bound`]) rather than assumed — an out-of-range/pathological
/// `n` would surface as [`CfarError::ExactCalibrationFailed`], not a
/// silently wrong threshold.
fn pfa_so_exact(n: f64, alpha: f64) -> Result<f64, CfarError> {
    if alpha <= 0.0
    {
        return Ok(1.0);
    }
    let t = alpha / n;
    let m_trunc = truncation_bound(n);
    let tail_q = gamma_survival(n, m_trunc);
    if !(tail_q.is_finite() && tail_q < 1.0e-12)
    {
        return Err(CfarError::ExactCalibrationFailed(format!(
            "truncation bound insufficient: Q_{n}({m_trunc}) = {tail_q}, expected < 1e-12"
        )));
    }
    let integral = scirust_solvers::quadrature::simpson_adaptive(
        |m| {
            let q = gamma_survival(n, m);
            (-t * m).exp() * q * q
        },
        0.0,
        m_trunc,
        QUADRATURE_TOLERANCE,
        50,
    )
    .map_err(|e| CfarError::ExactCalibrationFailed(e.to_string()))?;
    Ok(1.0 - t * integral)
}

/// Exact `P_fa` for greatest-of (`M' = max(S1,S2)`), derived from
/// [`pfa_so_exact`]: `P(M'>m) = 1-(1-Q_n(m))^2 = 2Q_n(m)-Q_n(m)^2`, so by the
/// same identity `P_fa_GO(alpha) = 1 - 2t·I₁(t) + t·I₂(t)` where
/// `I₂(t) = ∫e^{-tm}Q_n(m)^2 dm` (the integral inside [`pfa_so_exact`]) and
/// `I₁(t) = ∫₀^∞ e^{-tm}Q_n(m) dm`. For integer `n`,
/// `Q_n(m) = e^{-m}Σ_{k=0}^{n-1} m^k/k!`, and integrating term-by-term gives
/// the closed geometric sum `I₁(t) = [1-(1+t)^{-n}]/n·... `, more precisely
/// `t·I₁(t) = 1-(1+t)^{-n}` (each term `∫e^{-(t+1)m}m^k dm = k!/(t+1)^{k+1}`,
/// summing the resulting geometric series in `1/(t+1)`). Substituting
/// `t·I₂(t) = 1 - P_fa_SO(alpha)` (from `pfa_so_exact`'s own identity) gives
///
/// ```text
/// P_fa_GO(alpha) = 2·(1+t)^{-n} - P_fa_SO(alpha)
/// ```
///
/// Sanity checks: at `t=0`, `2·1 - 1 = 1` (matches `P(CUT>0)=1`); as
/// `t→∞`, `2·(1+t)^{-n}→0` and `P_fa_SO→0`, so `P_fa_GO→0`. Both checked by
/// this module's tests, alongside numerical agreement with the independent
/// Monte-Carlo calibration this exact form replaces as the production path
/// (kept, test-only, as a cross-check — see `calibrate_go_so_alpha_monte_carlo`).
fn pfa_go_exact(n: f64, alpha: f64) -> Result<f64, CfarError> {
    if alpha <= 0.0
    {
        return Ok(1.0);
    }
    let t = alpha / n;
    let so = pfa_so_exact(n, alpha)?;
    Ok(2.0 * (1.0 + t).powf(-n) - so)
}

/// Deterministically calibrates the threshold factor `alpha` for the
/// greatest-of (`take_max = true`) or smallest-of (`take_max = false`)
/// combining rule over two `reference_cells`-sized halves, under the i.i.d.
/// unit-mean-exponential reference-cell model — **exact**, via
/// [`pfa_so_exact`]/[`pfa_go_exact`] (deterministic numerical integration,
/// bounded non-statistical error), not Monte Carlo. See the module docs,
/// "Threshold calibration".
fn exact_go_so_alpha(reference_cells: usize, pfa: f64, take_max: bool) -> Result<f64, CfarError> {
    let n = reference_cells as f64;
    bisect_decreasing_fallible(
        |alpha| {
            if take_max
            {
                pfa_go_exact(n, alpha)
            }
            else
            {
                pfa_so_exact(n, alpha)
            }
        },
        pfa,
    )
}

/// The four threshold factors this detector can need, computed once (not
/// per-CUT) from a validated [`CfarConfig`] — `None` for whichever of the
/// four [`DetectorPolicy`] never selects that isn't needed at all: forcing
/// e.g. [`DetectorPolicy::Ca`] means [`decide`] can never read `go`/`so`/
/// `robust`, so [`CalibratedThresholds::compute`] does not pay GO/SO's
/// quadrature-bisection cost (the dominant cost in calibration — see the
/// module docs, "Threshold calibration") to produce a value nothing uses.
/// Only [`DetectorPolicy::ClassicalViCfar`] can route to any of the four at
/// runtime, so it alone needs all four calibrated up front.
#[derive(Debug, Clone, Copy)]
struct CalibratedThresholds {
    ca: Option<f64>,
    go: Option<f64>,
    so: Option<f64>,
    robust: Option<f64>,
}

impl CalibratedThresholds {
    fn compute(config: &CfarConfig) -> Result<Self, CfarError> {
        let n_ref = 2 * config.reference_cells;
        let (need_ca, need_go, need_so, need_robust) = match config.detector
        {
            DetectorPolicy::Ca => (true, false, false, false),
            DetectorPolicy::Go => (false, true, false, false),
            DetectorPolicy::So => (false, false, true, false),
            DetectorPolicy::AlwaysRobust => (false, false, false, true),
            DetectorPolicy::ClassicalViCfar(_) => (true, true, true, true),
        };

        let ca = need_ca.then(|| super::cfar::ca_cfar_alpha(n_ref, config.pfa));
        let go = need_go
            .then(|| exact_go_so_alpha(config.reference_cells, config.pfa, true))
            .transpose()?;
        let so = need_so
            .then(|| exact_go_so_alpha(config.reference_cells, config.pfa, false))
            .transpose()?;
        let robust = need_robust.then(|| {
            let (trim_low, trim_high) = config.robust_estimator.trim_counts();
            match config.robust_estimator
            {
                RobustNoiseEstimator::TrimmedMean { .. } =>
                {
                    trimmed_mean_alpha(n_ref, trim_low, trim_high, config.pfa)
                },
                RobustNoiseEstimator::CensoredMean { .. } =>
                {
                    censored_mean_alpha(n_ref, trim_low, trim_high, config.pfa)
                },
            }
        });

        Ok(Self { ca, go, so, robust })
    }
}

// ============================================================================
// Design-point calibration helpers (k_vi, k_mr)
// ============================================================================
//
// The module docs, "Switching structure," document that this module refuses
// to pick a default `k_vi`/`k_mr` because the published design points
// disagree and are context-dependent (each pins a *pair* of classification
// error rates, `alpha` = P(a homogeneous half-window is misclassified as
// non-homogeneous) and `beta` = P(a non-homogeneous half-window is missed),
// under a *specific assumed interferer strength* neither this module nor
// its literature survey can generalize). What these two functions calibrate
// is the *one* piece of that design point which does not depend on any
// interferer-strength assumption: `alpha` alone, under the switch's own
// null case of a genuinely homogeneous half-window. That is a strictly
// narrower, but honestly deliverable, service — a starting point for
// `alpha`, not a substitute for choosing `beta` against a real target/
// interferer model, which remains the caller's own domain-specific choice.

/// Exact `k_mr` for a target false-non-homogeneity rate `alpha`: `P(MR >
/// k_mr) = alpha` under the switch's null case (both reference half-windows
/// homogeneous, i.i.d. unit-mean-exponential power matching the CUT).
///
/// Derivation: `MR = mean_a / mean_b = A / B` where `A = reference_cells *
/// mean_a ~ Gamma(reference_cells, 1)` (a sum of `reference_cells` i.i.d.
/// `Exp(1)` samples), independently for `B`. `2 * Gamma(k, 1) ~ chi²(2k)`
/// (a standard identity — `Gamma(k, 1)` in shape-scale form *is*
/// `chi²(2k)/2`), and the ratio of two independent `chi²(k1)/k1` and
/// `chi²(k2)/k2` terms is `F(k1, k2)` by definition, so
///
/// ```text
/// MR = A / B = (2A / 2n) / (2B / 2n) ~ F(2n, 2n),  n = reference_cells
/// ```
///
/// `k_mr` solving `P(MR > k_mr) = alpha` is then exactly `F(2n,2n)`'s
/// `(1 - alpha)` quantile — [`scirust_stats::FisherF`]'s already-tested
/// quantile function, not a bisection or Monte Carlo of this module's own
/// devising.
///
/// This calibrates only `alpha`; see this section's own docs above for why
/// `beta` (the missed-non-homogeneity rate) is not calibrated here.
pub fn calibrate_k_mr(reference_cells: usize, alpha: f64) -> Result<f64, CfarError> {
    if reference_cells < 2
    {
        return Err(CfarError::TooFewReferenceCells(reference_cells));
    }
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0)
    {
        return Err(CfarError::ExactCalibrationFailed(format!(
            "alpha must satisfy 0 < alpha < 1, got {alpha}"
        )));
    }
    let n = reference_cells as f64;
    let f_dist = scirust_stats::FisherF::new(2.0 * n, 2.0 * n);
    Ok(f_dist.quantile(1.0 - alpha))
}

/// A small, self-contained, deterministic LCG (no OS/clock entropy) for
/// [`calibrate_k_vi`]'s Monte Carlo trials — the same generator family used
/// throughout this crate's own tests and benchmarks, kept private to this
/// function rather than reused from `#[cfg(test)]`-only code.
struct DesignPointRng(u64);

impl DesignPointRng {
    fn unit_exponential(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let uniform01 = ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        -uniform01.ln()
    }
}

/// Practical caps on [`calibrate_k_vi`]'s total Monte Carlo work. Its
/// runtime has two independent `O(_)` terms — generating `reference_cells`
/// unit-exponential draws for each of `trials` samples, then rescanning all
/// `trials` samples up to roughly [`BISECTION_MAX_ITERS`] times inside
/// [`bisect_decreasing`] — fit directly from release-build timings
/// (`reference_cells, trials, seconds`: `(8, 300_000, 0.034)`,
/// `(64, 1_000_000, 0.677)`, `(1_000, 1_000_000, 9.88)`) to
/// `time ≈ 1.0e-8·(reference_cells·trials) + 3.4e-8·trials`. Neither
/// `reference_cells` (already under [`MAX_PRACTICAL_REFERENCE_CELLS`]) nor
/// `trials` alone need look unreasonable for their product, or `trials` on
/// its own, to still make a call run for an unbounded time, so both terms
/// are capped independently, each to roughly half a second worst case;
/// existing calibration test/doc usage stays an order of magnitude or more
/// under either.
const MAX_PRACTICAL_MONTE_CARLO_SAMPLES: usize = 100_000_000;
const MAX_PRACTICAL_MONTE_CARLO_TRIALS: usize = 10_000_000;

/// Monte-Carlo-calibrates `k_vi` for a target false-non-homogeneity rate
/// `alpha`: `P(VI > k_vi) = alpha` under the switch's own null case (a
/// homogeneous reference half-window, i.i.d. unit-mean-exponential power).
///
/// Unlike [`calibrate_k_mr`], `VI`'s distribution under this model has no
/// simple closed form: `VI = 1 + s²/mean²` mixes the sample mean and a
/// quadratic form in the deviations from it, and for *exponential* — not
/// Gaussian — samples that quadratic form does not decouple into an exact
/// distribution independent of the mean the way Cochran's theorem gives for
/// Gaussian samples (which is what let [`calibrate_k_mr`] be exact above:
/// `MR` needed no such decoupling, only the ratio of two whole half-window
/// sums). This is therefore a deterministic, seeded Monte Carlo estimate,
/// not an exact calibration, and is documented as such — precision improves
/// with `trials` at the usual `O(1/√trials)` Monte-Carlo rate; `seed` makes
/// a given call exactly reproducible.
///
/// This calibrates only `alpha`; see this section's own docs above for why
/// `beta` (the missed-non-homogeneity rate) is not calibrated here.
pub fn calibrate_k_vi(
    reference_cells: usize,
    alpha: f64,
    trials: usize,
    seed: u64,
) -> Result<f64, CfarError> {
    if reference_cells < 2
    {
        return Err(CfarError::TooFewReferenceCells(reference_cells));
    }
    if reference_cells > MAX_PRACTICAL_REFERENCE_CELLS
    {
        return Err(CfarError::ExactCalibrationFailed(format!(
            "reference_cells={reference_cells} exceeds the practical limit of \
             {MAX_PRACTICAL_REFERENCE_CELLS} (see MAX_PRACTICAL_REFERENCE_CELLS)"
        )));
    }
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0)
    {
        return Err(CfarError::ExactCalibrationFailed(format!(
            "alpha must satisfy 0 < alpha < 1, got {alpha}"
        )));
    }
    if trials == 0
    {
        return Err(CfarError::ExactCalibrationFailed(
            "trials must be at least 1".to_string(),
        ));
    }
    // See `MAX_PRACTICAL_MONTE_CARLO_SAMPLES`/`MAX_PRACTICAL_MONTE_CARLO_TRIALS`:
    // two independent cost terms, so both are checked -- `trials` alone
    // bounds the rescan cost (dominant for small `reference_cells`), the
    // product bounds the generation cost (dominant for large
    // `reference_cells`). Rejected here rather than left to block for an
    // open-ended time.
    if trials > MAX_PRACTICAL_MONTE_CARLO_TRIALS
    {
        return Err(CfarError::ExactCalibrationFailed(format!(
            "trials={trials} exceeds the practical limit of \
             {MAX_PRACTICAL_MONTE_CARLO_TRIALS}"
        )));
    }
    let work = reference_cells.checked_mul(trials);
    let work_impractical = match work
    {
        Some(w) => w > MAX_PRACTICAL_MONTE_CARLO_SAMPLES,
        None => true,
    };
    if work_impractical
    {
        return Err(CfarError::ExactCalibrationFailed(format!(
            "reference_cells={reference_cells} * trials={trials} exceeds the practical \
             Monte Carlo sample budget of {MAX_PRACTICAL_MONTE_CARLO_SAMPLES}"
        )));
    }
    let mut rng = DesignPointRng(seed);
    let n = reference_cells as f64;
    let vi_samples: Vec<f64> = (0..trials)
        .map(|_| {
            let cells: Vec<f64> = (0..reference_cells)
                .map(|_| rng.unit_exponential())
                .collect();
            let mean = cells.iter().sum::<f64>() / n;
            let sample_variance =
                cells.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
            variability_index(mean, sample_variance)
        })
        .collect();
    let pfa_of =
        |k: f64| -> f64 { vi_samples.iter().filter(|&&vi| vi > k).count() as f64 / trials as f64 };
    Ok(bisect_decreasing(pfa_of, alpha))
}

// ============================================================================
// Robust trimmed-mean primitive
// ============================================================================

/// Diagnostic result of one [`trimmed_mean`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimmedMeanResult {
    /// Mean of the retained cells.
    pub estimate: f64,
    /// Number of cells retained (`scratch.len() - rejected_low - rejected_high`).
    pub accepted: usize,
    /// Number of smallest cells discarded.
    pub rejected_low: usize,
    /// Number of largest cells discarded.
    pub rejected_high: usize,
}

/// Sorts `scratch` in place (allocation-free `sort_unstable_by`;
/// deterministic under `f64::total_cmp` — equal values are interchangeable
/// for a mean, so instability is immaterial here) and averages the
/// `scratch.len() - trim_low - trim_high` middle values.
///
/// # Errors
/// [`CfarError::InvalidTrimCounts`] if `trim_low + trim_high >= scratch.len()`
/// (no cell would remain).
pub fn trimmed_mean(
    scratch: &mut [f64],
    trim_low: usize,
    trim_high: usize,
) -> Result<TrimmedMeanResult, CfarError> {
    let n_ref = scratch.len();
    let trim_counts_invalid = match trim_low.checked_add(trim_high)
    {
        Some(sum) => sum >= n_ref,
        None => true,
    };
    if trim_counts_invalid
    {
        return Err(CfarError::InvalidTrimCounts {
            n_ref,
            trim_low,
            trim_high,
        });
    }
    scratch.sort_unstable_by(f64::total_cmp);
    let kept = &scratch[trim_low..n_ref - trim_high];
    let estimate = kept.iter().sum::<f64>() / kept.len() as f64;
    Ok(TrimmedMeanResult {
        estimate,
        accepted: kept.len(),
        rejected_low: trim_low,
        rejected_high: trim_high,
    })
}

// ============================================================================
// Robust censored-mean (Winsorizing) primitive
// ============================================================================

/// Diagnostic result of one [`censored_mean`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensoredMeanResult {
    /// Mean of all cells after Winsorizing (all `scratch.len()` cells
    /// contribute, unlike [`TrimmedMeanResult`]).
    pub estimate: f64,
    /// Number of cells left unmodified in the middle.
    pub retained: usize,
    /// Number of smallest cells replaced (Winsorized) at the low end.
    pub censored_low: usize,
    /// Number of largest cells replaced (Winsorized) at the high end.
    pub censored_high: usize,
}

/// Sorts `scratch` in place (allocation-free `sort_unstable_by`, same
/// rationale as [`trimmed_mean`]), replaces the `trim_low` smallest values
/// with the smallest *retained* value and the `trim_high` largest with the
/// largest retained value, and averages **all** `scratch.len()` values (none
/// discarded — see [`RobustNoiseEstimator::CensoredMean`]).
///
/// # Errors
/// [`CfarError::InvalidTrimCounts`] if `trim_low + trim_high >= scratch.len()`
/// (no retained cell to Winsorize toward).
pub fn censored_mean(
    scratch: &mut [f64],
    trim_low: usize,
    trim_high: usize,
) -> Result<CensoredMeanResult, CfarError> {
    let n_ref = scratch.len();
    let trim_counts_invalid = match trim_low.checked_add(trim_high)
    {
        Some(sum) => sum >= n_ref,
        None => true,
    };
    if trim_counts_invalid
    {
        return Err(CfarError::InvalidTrimCounts {
            n_ref,
            trim_low,
            trim_high,
        });
    }
    scratch.sort_unstable_by(f64::total_cmp);
    let low_value = scratch[trim_low];
    let high_value = scratch[n_ref - trim_high - 1];
    let retained_sum: f64 = scratch[trim_low..n_ref - trim_high].iter().sum();
    let winsorized_sum = trim_low as f64 * low_value + retained_sum + trim_high as f64 * high_value;
    Ok(CensoredMeanResult {
        estimate: winsorized_sum / n_ref as f64,
        retained: n_ref - trim_low - trim_high,
        censored_low: trim_low,
        censored_high: trim_high,
    })
}

// ============================================================================
// Shared decision core (used by both the finite-slice and streaming APIs)
// ============================================================================

/// A reference half-window's mean and sample variance — the only summary a
/// CUT decision needs, regardless of whether it was computed by direct
/// summation over a slice or read off a
/// [`crate::sliding_stats::SlidingMoments`].
#[derive(Debug, Clone, Copy)]
struct HalfWindowStats {
    mean: f64,
    sample_variance: f64,
}

/// Direct (two-pass, `O(len)`) computation for the finite-slice API. `cells`
/// must have length `>= 2` (guaranteed by [`CfarConfig::validate`]'s
/// `reference_cells >= 2`).
fn half_window_stats(cells: &[f64]) -> HalfWindowStats {
    let n = cells.len() as f64;
    let mean = cells.iter().sum::<f64>() / n;
    let m2: f64 = cells.iter().map(|&x| (x - mean) * (x - mean)).sum();
    HalfWindowStats {
        mean,
        sample_variance: m2 / (n - 1.0),
    }
}

/// The shared core: given both half-windows' statistics and (for the robust
/// path only) the pooled raw cells in `scratch`, classify and produce a full
/// [`CfarDecision`]. Neither the finite-slice nor the streaming API
/// duplicates this logic — both call it.
#[allow(clippy::too_many_arguments)]
fn decide(
    cut_index: usize,
    cut_power: f64,
    lagging: HalfWindowStats,
    leading: HalfWindowStats,
    scratch: &mut [f64],
    config: &CfarConfig,
    alphas: &CalibratedThresholds,
) -> Result<CfarDecision, CfarError> {
    let lagging_vi = variability_index(lagging.mean, lagging.sample_variance);
    let leading_vi = variability_index(leading.mean, leading.sample_variance);
    let mr = mean_ratio(lagging.mean, leading.mean);

    // Which CfarMode the configured RobustNoiseEstimator corresponds to —
    // used both by AlwaysRobust and by ClassicalViCfar's double-contamination
    // branch, so the reported mode always matches the estimator actually run.
    let robust_mode = match config.robust_estimator
    {
        RobustNoiseEstimator::TrimmedMean { .. } => CfarMode::RobustTrimmed,
        RobustNoiseEstimator::CensoredMean { .. } => CfarMode::RobustCensored,
    };

    let mode = match config.detector
    {
        DetectorPolicy::Ca => CfarMode::Ca,
        DetectorPolicy::Go => CfarMode::Go,
        DetectorPolicy::So => CfarMode::So,
        DetectorPolicy::AlwaysRobust => robust_mode,
        DetectorPolicy::ClassicalViCfar(t) =>
        {
            let lag_nonhomog = lagging_vi > t.k_vi;
            let lead_nonhomog = leading_vi > t.k_vi;
            match (lag_nonhomog, lead_nonhomog)
            {
                (false, false) if mr <= t.k_mr => CfarMode::Ca,
                (false, false) => CfarMode::Go,
                (true, true) => robust_mode,
                _ => CfarMode::So,
            }
        },
    };

    let (noise_estimate, threshold) = match mode
    {
        CfarMode::Ca =>
        {
            let noise = 0.5 * (lagging.mean + leading.mean);
            let alpha = alphas.ca.expect(
                "CfarMode::Ca is only ever selected when config.detector needs alphas.ca \
                 (Ca, or ClassicalViCfar which computes all four) — see CalibratedThresholds::compute",
            );
            (noise, alpha * noise)
        },
        CfarMode::Go =>
        {
            let noise = lagging.mean.max(leading.mean);
            let alpha = alphas.go.expect(
                "CfarMode::Go is only ever selected when config.detector needs alphas.go \
                 (Go, or ClassicalViCfar which computes all four) — see CalibratedThresholds::compute",
            );
            (noise, alpha * noise)
        },
        CfarMode::So =>
        {
            let noise = lagging.mean.min(leading.mean);
            let alpha = alphas.so.expect(
                "CfarMode::So is only ever selected when config.detector needs alphas.so \
                 (So, or ClassicalViCfar which computes all four) — see CalibratedThresholds::compute",
            );
            (noise, alpha * noise)
        },
        CfarMode::RobustTrimmed =>
        {
            let (trim_low, trim_high) = config.robust_estimator.trim_counts();
            let result = trimmed_mean(scratch, trim_low, trim_high)?;
            let alpha = alphas.robust.expect(
                "a robust CfarMode is only ever selected when config.detector needs \
                 alphas.robust (AlwaysRobust, or ClassicalViCfar which computes all four) — \
                 see CalibratedThresholds::compute",
            );
            (result.estimate, alpha * result.estimate)
        },
        CfarMode::RobustCensored =>
        {
            let (trim_low, trim_high) = config.robust_estimator.trim_counts();
            let result = censored_mean(scratch, trim_low, trim_high)?;
            let alpha = alphas.robust.expect(
                "a robust CfarMode is only ever selected when config.detector needs \
                 alphas.robust (AlwaysRobust, or ClassicalViCfar which computes all four) — \
                 see CalibratedThresholds::compute",
            );
            (result.estimate, alpha * result.estimate)
        },
    };

    Ok(CfarDecision {
        cut_index,
        cut_power,
        lagging_mean: lagging.mean,
        leading_mean: leading.mean,
        lagging_vi,
        leading_vi,
        mean_ratio: mr,
        noise_estimate,
        threshold,
        detected: cut_power > threshold,
        mode,
    })
}

// ============================================================================
// Finite-slice API
// ============================================================================

/// Evaluate VI-CFAR over an entire finite power slice.
///
/// Validates `config` and every sample in `power` up front (see
/// [`CfarError`]); a cell without a full reference window on both sides is
/// absent from the result ([`EdgePolicy::Exclude`]). `O(power.len() *
/// reference_cells)`: one scratch buffer sized `2 * reference_cells` is
/// allocated once and reused across every CUT (no per-CUT allocation).
///
/// ```
/// use scirust_signal::radar::vi_cfar::{CfarConfig, DetectorPolicy, EdgePolicy, InputValidationPolicy, RobustNoiseEstimator, evaluate_slice};
///
/// let mut power = vec![1.0_f64; 60];
/// power[30] = 50.0; // an isolated target on a flat homogeneous floor
/// let config = CfarConfig {
///     reference_cells: 8,
///     guard_cells: 2,
///     pfa: 0.01,
///     edge_policy: EdgePolicy::Exclude,
///     input_validation: InputValidationPolicy::RejectNegative,
///     detector: DetectorPolicy::Ca,
///     robust_estimator: RobustNoiseEstimator::TrimmedMean { trim_low: 1, trim_high: 1 },
/// };
/// let decisions = evaluate_slice(&power, &config).unwrap();
/// assert!(decisions.iter().find(|d| d.cut_index == 30).unwrap().detected);
/// ```
pub fn evaluate_slice(power: &[f64], config: &CfarConfig) -> Result<Vec<CfarDecision>, CfarError> {
    CfarDetector::new(*config)?.evaluate(power)
}

/// A [`CfarConfig`] with its threshold factors calibrated once and its
/// scratch buffer allocated once, reused across many [`evaluate`](Self::evaluate)
/// calls.
///
/// [`evaluate_slice`] is a convenience wrapper around
/// `CfarDetector::new(config)?.evaluate(power)` — simplest for one-shot use,
/// but it recalibrates (including GO/SO's and CensoredMean's quadrature/
/// bisection-based exact calibration) on *every* call. A caller processing
/// many slices (or dwells, in a
/// real radar loop) under the same configuration should construct one
/// `CfarDetector` and reuse it — calibration then happens exactly once, and
/// no allocation happens inside [`evaluate`](Self::evaluate) beyond the
/// output `Vec` itself. This is also what makes the per-CUT cost
/// independently benchmarkable (see `benches/vi_cfar_bench.rs`) rather than
/// dominated by calibration on every measured iteration.
#[derive(Debug, Clone)]
pub struct CfarDetector {
    config: CfarConfig,
    alphas: CalibratedThresholds,
    scratch: Vec<f64>,
}

impl CfarDetector {
    /// Validate `config` and calibrate its threshold factors once.
    pub fn new(config: CfarConfig) -> Result<Self, CfarError> {
        config.validate()?;
        let alphas = CalibratedThresholds::compute(&config)?;
        let scratch = vec![0.0_f64; 2 * config.reference_cells];
        Ok(Self {
            config,
            alphas,
            scratch,
        })
    }

    /// The configuration this detector was constructed with.
    pub fn config(&self) -> &CfarConfig {
        &self.config
    }

    /// Evaluate over one finite power slice, reusing this detector's cached
    /// calibration and scratch buffer (no allocation beyond the returned
    /// `Vec`). See [`evaluate_slice`] for the per-cell contract (input
    /// validation, edge handling).
    pub fn evaluate(&mut self, power: &[f64]) -> Result<Vec<CfarDecision>, CfarError> {
        for (index, &value) in power.iter().enumerate()
        {
            self.config.validate_sample(index, value)?;
        }

        let train = self.config.reference_cells;
        let guard = self.config.guard_cells;
        let half = train + guard;
        let n = power.len();
        let mut decisions = Vec::with_capacity(n.saturating_sub(2 * half));

        // EdgePolicy::Exclude (the only policy today): the loop bound below
        // simply never visits a cell without a full window on both sides.
        if n > 2 * half
        {
            for cut in half..n - half
            {
                let lagging_cells = &power[cut - half..cut - guard];
                let leading_cells = &power[cut + guard + 1..cut + half + 1];
                let lagging = half_window_stats(lagging_cells);
                let leading = half_window_stats(leading_cells);
                self.scratch[..train].copy_from_slice(lagging_cells);
                self.scratch[train..].copy_from_slice(leading_cells);
                decisions.push(decide(
                    cut,
                    power[cut],
                    lagging,
                    leading,
                    &mut self.scratch,
                    &self.config,
                    &self.alphas,
                )?);
            }
        }
        Ok(decisions)
    }
}

// ============================================================================
// 2-D range-Doppler API
// ============================================================================

/// Applies the 1-D VI-CFAR switch independently along the range axis, once
/// per Doppler bin, over a range-Doppler power map (`power[range][doppler]`
/// — the same `power[range][doppler]` layout [`super::detect::ca_cfar_2d`]
/// uses).
///
/// This is the classical *per-Doppler-bin range-CFAR* pattern — not a
/// genuinely 2-D (range × Doppler) reference window the way
/// [`super::detect::ca_cfar_2d`]'s square training region is. Extending
/// this module's switch (the VI/MR non-homogeneity classification, the
/// double-contamination robust fallback) to a truly 2-D reference geometry
/// would need its own independently-derived and independently-verified
/// theory — out of scope here, and not attempted by guessing. Reusing the
/// already-calibrated, already-tested 1-D [`CfarDetector`] column-by-column
/// instead keeps every `P_fa`/switching guarantee documented above intact:
/// each Doppler bin's range profile is exactly the 1-D problem
/// [`evaluate_slice`] already solves, run `cols` times against one
/// detector calibrated once (not once per column — see [`CfarDetector`]'s
/// own docs on why that matters).
///
/// `config` is validated and calibrated exactly once via
/// [`CfarDetector::new`], which is where an invalid config surfaces as a
/// [`CfarError`]. Shape problems in `power` itself (empty, zero-width, or
/// ragged rows) are not configuration errors and are handled the same way
/// [`super::detect::ca_cfar_2d`] handles them: silently, by returning an
/// all-`false` mask of the input's own shape (or an empty map, if `power`
/// itself is empty) rather than an error, since a malformed *measurement*
/// is a different kind of problem than a malformed *configuration*.
pub fn vi_cfar_2d(power: &[Vec<f64>], config: &CfarConfig) -> Result<Vec<Vec<bool>>, CfarError> {
    // Config validation/calibration happens first, unconditionally --
    // before any check of `power`'s shape -- so an invalid config always
    // surfaces as `Err`, regardless of whether `power` is empty, ragged, or
    // zero-width (matching `evaluate_slice`, which validates `config`
    // before ever looking at `power`'s content).
    let mut detector = CfarDetector::new(*config)?;

    let rows = power.len();
    if rows == 0
    {
        return Ok(Vec::new());
    }
    let cols = power[0].len();
    let mut det = vec![vec![false; cols]; rows];
    if cols == 0 || power.iter().any(|row| row.len() != cols)
    {
        return Ok(det);
    }

    let mut column = vec![0.0_f64; rows];
    for d in 0..cols
    {
        for (r, row) in power.iter().enumerate()
        {
            column[r] = row[d];
        }
        for decision in detector.evaluate(&column)?
        {
            det[decision.cut_index][d] = decision.detected;
        }
    }
    Ok(det)
}

// ============================================================================
// Streaming API
// ============================================================================

/// Streaming VI-CFAR over a reference-window size set at construction time
/// from [`CfarConfig::reference_cells`] — no compile-time parameter needed.
///
/// Backed by two [`SlidingMomentsDyn`] (lagging/leading half-windows, `O(1)`
/// per push after construction) plus a small FIFO delay line (`to_lagging`)
/// that lets a value already consumed by `leading` be released into
/// `lagging` exactly `2 * guard_cells + reference_cells + 1` pushes later —
/// see "Derivation" below, which still calls the reference-window size
/// `TRAIN` as a mathematical symbol (the derivation doesn't care whether it
/// is a `const` or a runtime value). [`SlidingMomentsDyn`] runs the exact
/// same recurrences as the compile-time-sized `SlidingMoments<N>` (see
/// `crate::sliding_stats`'s module docs) over a `Box<[f64]>` allocated once
/// here at construction — [`push`](Self::push) itself never allocates.
///
/// # Latency
///
/// A decision for the sample pushed at (0-based) stream position `t` is
/// returned by [`push`](Self::push) only once the leading window has filled —
/// i.e. the call that pushes the sample at position
/// `t + guard_cells + TRAIN` returns `Ok(Some(decision))` with
/// `decision.cut_index == t`. Every earlier call returns `Ok(None)`. This is
/// not implementation-specific latency to be optimized away: a streaming
/// detector cannot know a CUT's leading (future-relative) reference cells
/// before they arrive.
///
/// # Derivation
///
/// `leading` is fed *every* sample once `lagging`'s initial fill completes,
/// unconditionally — at the moment the sample at absolute index
/// `k = t + guard + TRAIN` arrives, "the trailing `TRAIN` samples" is exactly
/// `{k-TRAIN+1, .., k} = {t+guard+1, .., t+guard+TRAIN}`, which *is*
/// `CUT = t`'s leading reference window by definition — so `leading`'s state
/// is automatically correct for whichever CUT is currently decidable, with no
/// extra bookkeeping.
///
/// `lagging` needs `{t-guard-TRAIN, .., t-guard-1}` at that same moment. The
/// value at absolute index `t-guard-1` was fed to `leading` when it arrived,
/// `(t+guard+TRAIN) - (t-guard-1) = 2*guard+TRAIN+1` pushes ago. So: queue
/// every value fed to `leading` in `to_lagging` (a FIFO capped at
/// `2*guard+TRAIN+1` entries); once adding a new one pushes the queue over
/// that cap, pop the oldest and feed *that* into `lagging`. Once the queue has
/// reached its cap for the first time (and forever after, since each further
/// push immediately overflows it by exactly one and pops back down), the
/// value at position `guard` within it (0-indexed from the front) is exactly
/// the pending CUT's own power — this position is likewise a fixed
/// consequence of the cap size and does not change over time.
///
/// (Verified in this module's tests by direct agreement with
/// [`evaluate_slice`] on interior points of a shared homogeneous signal, for
/// both `guard_cells == 0` and `guard_cells > 0`, rather than by this
/// derivation alone.)
///
/// ```
/// use scirust_signal::radar::vi_cfar::{CfarConfig, CfarStreamDetector, DetectorPolicy, EdgePolicy, InputValidationPolicy, RobustNoiseEstimator};
///
/// let config = CfarConfig {
///     reference_cells: 8,
///     guard_cells: 2,
///     pfa: 0.01,
///     edge_policy: EdgePolicy::Exclude,
///     input_validation: InputValidationPolicy::RejectNegative,
///     detector: DetectorPolicy::Ca,
///     robust_estimator: RobustNoiseEstimator::TrimmedMean { trim_low: 1, trim_high: 1 },
/// };
/// let mut detector = CfarStreamDetector::new(config).unwrap();
///
/// let mut samples = vec![1.0_f64; 60];
/// samples[30] = 50.0; // an isolated target
/// let mut saw_the_target = false;
/// for &x in &samples {
///     // `push` returns `None` during warm-up/latency; `Some(decision)` once
///     // a CUT's full window (both guards and both reference halves) has
///     // arrived — decision.cut_index tells you which sample it is for.
///     if let Some(decision) = detector.push(x).unwrap() {
///         if decision.cut_index == 30 {
///             assert!(decision.detected);
///             saw_the_target = true;
///         }
///     }
/// }
/// assert!(saw_the_target);
/// ```
#[derive(Debug)]
pub struct CfarStreamDetector {
    config: CfarConfig,
    alphas: CalibratedThresholds,
    lagging: SlidingMomentsDyn,
    leading: SlidingMomentsDyn,
    /// FIFO delay line; see the type-level "Derivation" docs. Capacity
    /// reserved once at construction (`2*guard+reference_cells+2`, one more
    /// than its steady-state cap to hold the momentary overflow before each
    /// pop) and never reallocated after that.
    to_lagging: std::collections::VecDeque<f64>,
    /// Absolute index of the next pushed sample.
    next_index: usize,
    scratch: Vec<f64>,
}

impl CfarStreamDetector {
    /// Construct a streaming detector sized from `config.reference_cells`.
    pub fn new(config: CfarConfig) -> Result<Self, CfarError> {
        config.validate()?;
        let reference_cells = config.reference_cells;
        let domain = match config.input_validation
        {
            InputValidationPolicy::RejectNegative => SampleDomain::NonNegative,
            InputValidationPolicy::AllowNegative => SampleDomain::Real,
        };
        let new_moments = || -> Result<SlidingMomentsDyn, SlidingMomentsError> {
            match domain
            {
                SampleDomain::NonNegative => SlidingMomentsDyn::new_non_negative(reference_cells),
                SampleDomain::Real => SlidingMomentsDyn::new(reference_cells),
            }
        };
        let lagging = new_moments()?;
        let leading = new_moments()?;
        let alphas = CalibratedThresholds::compute(&config)?;
        let cap = 2 * config.guard_cells + reference_cells + 1;
        let to_lagging = std::collections::VecDeque::with_capacity(cap + 1);
        let scratch = vec![0.0_f64; 2 * reference_cells];
        Ok(Self {
            config,
            alphas,
            lagging,
            leading,
            to_lagging,
            next_index: 0,
            scratch,
        })
    }

    /// Push one sample. Returns `Ok(None)` while warming up (see the
    /// type-level docs for the exact latency), `Ok(Some(decision))` once a
    /// CUT's full window (both guards and both reference halves) is
    /// available.
    pub fn push(&mut self, value: f64) -> Result<Option<CfarDecision>, CfarError> {
        let idx = self.next_index;
        self.config.validate_sample(idx, value)?;
        self.next_index += 1;
        let reference_cells = self.config.reference_cells;

        if self.lagging.len() < reference_cells
        {
            self.lagging.push(value)?;
            return Ok(None);
        }

        let cap = 2 * self.config.guard_cells + reference_cells + 1;
        self.to_lagging.push_back(value);
        if self.to_lagging.len() > cap
        {
            let old = self
                .to_lagging
                .pop_front()
                .expect("just checked len > cap >= 1");
            self.lagging.push(old)?;
        }
        self.leading.push(value)?;

        if self.to_lagging.len() < cap
        {
            return Ok(None);
        }

        let cut_index = idx - self.config.guard_cells - reference_cells;
        let cut_power = self.to_lagging[self.config.guard_cells];
        let lagging = HalfWindowStats {
            mean: self.lagging.mean().expect("lagging is full at this point"),
            sample_variance: self
                .lagging
                .sample_variance()
                .expect("lagging is full (reference_cells >= 2, enforced by CfarConfig::validate)"),
        };
        let leading = HalfWindowStats {
            mean: self.leading.mean().expect("leading is full at this point"),
            sample_variance: self
                .leading
                .sample_variance()
                .expect("leading is full (reference_cells >= 2, enforced by CfarConfig::validate)"),
        };
        // Only [`CfarMode::RobustTrimmed`] reads `scratch`, but populating it
        // is cheap (two O(reference_cells) copies) relative to the rest of
        // this function, and keeps `decide` identical for both APIs.
        self.scratch[..reference_cells].copy_from_slice(self.lagging.as_slice());
        self.scratch[reference_cells..].copy_from_slice(self.leading.as_slice());
        let decision = decide(
            cut_index,
            cut_power,
            lagging,
            leading,
            &mut self.scratch,
            &self.config,
            &self.alphas,
        )?;
        Ok(Some(decision))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CfarConfig {
        CfarConfig {
            reference_cells: 8,
            guard_cells: 2,
            pfa: 0.01,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector: DetectorPolicy::Ca,
            robust_estimator: RobustNoiseEstimator::TrimmedMean {
                trim_low: 1,
                trim_high: 1,
            },
        }
    }

    // ---- variability_index / mean_ratio -----------------------------------

    #[test]
    fn variability_index_of_homogeneous_exponential_tends_to_two() {
        // For unit-mean Exp(1), Var = Mean^2 = 1, so VI = 1 + 1 = 2 exactly
        // when variance and mean^2 are exact.
        assert!((variability_index(1.0, 1.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn variability_index_zero_variance_is_one() {
        assert_eq!(variability_index(5.0, 0.0), 1.0);
    }

    #[test]
    fn mean_ratio_both_zero_is_one() {
        assert_eq!(mean_ratio(0.0, 0.0), 1.0);
    }

    #[test]
    fn mean_ratio_one_zero_is_infinite() {
        assert_eq!(mean_ratio(0.0, 5.0), f64::INFINITY);
        assert_eq!(mean_ratio(5.0, 0.0), f64::INFINITY);
    }

    #[test]
    fn mean_ratio_is_symmetric_and_at_least_one() {
        assert_eq!(mean_ratio(3.0, 9.0), 3.0);
        assert_eq!(mean_ratio(9.0, 3.0), 3.0);
        assert_eq!(mean_ratio(7.0, 7.0), 1.0);
    }

    #[test]
    fn mean_ratio_handles_very_small_positive_means_without_an_epsilon_guard() {
        let mr = mean_ratio(1e-300, 3e-300);
        assert!((mr - 3.0).abs() < 1e-9);
    }

    // ---- CfarConfig::validate ----------------------------------------------

    #[test]
    fn config_rejects_too_few_reference_cells() {
        let mut c = default_config();
        c.reference_cells = 1;
        assert_eq!(c.validate(), Err(CfarError::TooFewReferenceCells(1)));
    }

    #[test]
    fn config_rejects_invalid_pfa() {
        for bad in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY]
        {
            let mut c = default_config();
            c.pfa = bad;
            assert!(c.validate().is_err(), "pfa={bad} should be rejected");
        }
    }

    #[test]
    fn config_rejects_invalid_switching_thresholds() {
        let mut c = default_config();
        c.detector = DetectorPolicy::ClassicalViCfar(SwitchingThresholds {
            k_vi: 0.0,
            k_mr: 2.0,
        });
        assert!(c.validate().is_err());
        c.detector = DetectorPolicy::ClassicalViCfar(SwitchingThresholds {
            k_vi: 5.0,
            k_mr: 0.5,
        });
        assert!(c.validate().is_err(), "k_mr < 1.0 must be rejected");
        c.detector = DetectorPolicy::ClassicalViCfar(SwitchingThresholds {
            k_vi: 5.0,
            k_mr: 2.0,
        });
        assert!(c.validate().is_ok());
    }

    #[test]
    fn config_rejects_trim_counts_leaving_no_retained_cell() {
        let mut c = default_config();
        c.reference_cells = 4; // pooled n_ref = 8
        c.robust_estimator = RobustNoiseEstimator::TrimmedMean {
            trim_low: 4,
            trim_high: 4,
        };
        assert_eq!(
            c.validate(),
            Err(CfarError::InvalidTrimCounts {
                n_ref: 8,
                trim_low: 4,
                trim_high: 4
            })
        );
    }

    #[test]
    fn config_rejects_reference_cells_that_would_overflow_2x() {
        // Adversarial-audit regression: `2 * reference_cells` used to be
        // computed with a plain `*`, panicking on overflow instead of
        // returning a structured error.
        let mut c = default_config();
        c.reference_cells = usize::MAX / 2 + 10;
        assert_eq!(
            c.validate(),
            Err(CfarError::ReferenceWindowTooLarge {
                reference_cells: usize::MAX / 2 + 10,
                guard_cells: c.guard_cells,
            })
        );
    }

    #[test]
    fn config_rejects_guard_cells_that_would_overflow_the_half_window() {
        // Adversarial-audit regression: `guard_cells` had *no* validation at
        // all, so `reference_cells + guard_cells` (computed in
        // `CfarDetector::evaluate`, well after validation) could overflow
        // and panic on an otherwise entirely ordinary `reference_cells`.
        let mut c = default_config();
        c.guard_cells = usize::MAX;
        assert_eq!(
            c.validate(),
            Err(CfarError::ReferenceWindowTooLarge {
                reference_cells: c.reference_cells,
                guard_cells: usize::MAX,
            })
        );
        // And the detector constructors that call `validate()` first must
        // also reject it, rather than ever reaching the arithmetic that
        // used to panic.
        assert!(matches!(
            CfarDetector::new(c),
            Err(CfarError::ReferenceWindowTooLarge { .. })
        ));
    }

    #[test]
    fn config_rejects_reference_cells_beyond_the_practical_calibration_bound() {
        // New-finding regression (surfaced by widening this module's own
        // proptest strategies to draw large-but-valid `usize`s, which then
        // hung indefinitely instead of finishing): `reference_cells` not
        // overflowing `2 * reference_cells` is not enough on its own --
        // `CalibratedThresholds::compute`'s `TrimmedMean`/`CensoredMean`
        // path (`trimmed_mean_alpha`/`censored_mean_alpha`) is an O(n_ref)
        // loop run up to ~140 times inside `bisect_decreasing`, so a merely
        // astronomically large (nowhere near overflowing) `reference_cells`
        // used to make calibration -- and so `CfarDetector::new` -- block
        // for an unbounded time instead of returning a structured error.
        let mut c = default_config();
        c.reference_cells = MAX_PRACTICAL_REFERENCE_CELLS + 1;
        c.detector = DetectorPolicy::AlwaysRobust;
        assert_eq!(
            c.validate(),
            Err(CfarError::ReferenceWindowTooLarge {
                reference_cells: MAX_PRACTICAL_REFERENCE_CELLS + 1,
                guard_cells: c.guard_cells,
            })
        );
        // `CfarDetector::new` must reject it before ever reaching
        // `CalibratedThresholds::compute` -- this assertion itself is the
        // hang-vs-fast-`Err` check: the test process would never reach here
        // before this fix.
        assert!(matches!(
            CfarDetector::new(c),
            Err(CfarError::ReferenceWindowTooLarge { .. })
        ));
    }

    #[test]
    fn config_accepts_reference_cells_right_at_the_practical_calibration_bound() {
        // Boundary check for `MAX_PRACTICAL_REFERENCE_CELLS` itself: sitting
        // exactly at the limit, with the O(n_ref) `TrimmedMean` calibration
        // path exercised, must still validate and calibrate successfully
        // (and promptly) -- not be off-by-one rejected.
        let mut c = default_config();
        c.reference_cells = MAX_PRACTICAL_REFERENCE_CELLS;
        c.detector = DetectorPolicy::AlwaysRobust;
        assert!(c.validate().is_ok());
        assert!(CfarDetector::new(c).is_ok());
    }

    // ---- trimmed_mean primitive --------------------------------------------

    #[test]
    fn trimmed_mean_basic_correctness() {
        let mut data = [5.0, 1.0, 100.0, 2.0, 3.0, -50.0];
        // sorted: -50, 1, 2, 3, 5, 100. trim_low=1, trim_high=1 -> keep {1,2,3,5}.
        let r = trimmed_mean(&mut data, 1, 1).unwrap();
        assert_eq!(r.accepted, 4);
        assert_eq!(r.rejected_low, 1);
        assert_eq!(r.rejected_high, 1);
        assert!((r.estimate - 2.75).abs() < 1e-12);
    }

    #[test]
    fn trimmed_mean_handles_ties_deterministically() {
        let mut data = [2.0, 2.0, 2.0, 2.0];
        let r = trimmed_mean(&mut data, 1, 1).unwrap();
        assert!((r.estimate - 2.0).abs() < 1e-12);
        assert_eq!(r.accepted, 2);
    }

    #[test]
    fn trimmed_mean_rejects_trim_leaving_no_sample() {
        let mut data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            trimmed_mean(&mut data, 2, 2),
            Err(CfarError::InvalidTrimCounts {
                n_ref: 4,
                trim_low: 2,
                trim_high: 2
            })
        );
    }

    #[test]
    fn trimmed_mean_rejects_overflowing_trim_counts_instead_of_panicking() {
        // trim_low + trim_high must not be computed with a plain `+` -- an
        // adversarial-audit regression: `usize::MAX + usize::MAX` overflows
        // and panics in a debug build rather than returning a structured
        // error, exactly the "never panic, only Ok or CfarError" contract
        // this module documents.
        let mut data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            trimmed_mean(&mut data, usize::MAX, usize::MAX),
            Err(CfarError::InvalidTrimCounts {
                n_ref: 4,
                trim_low: usize::MAX,
                trim_high: usize::MAX,
            })
        );
    }

    #[test]
    fn trimmed_mean_no_trim_is_the_plain_mean() {
        let mut data = [1.0, 2.0, 3.0, 4.0];
        let r = trimmed_mean(&mut data, 0, 0).unwrap();
        assert!((r.estimate - 2.5).abs() < 1e-12);
    }

    // ---- censored_mean (Winsorizing) primitive -----------------------------

    #[test]
    fn censored_mean_basic_correctness() {
        let mut data = [5.0, 1.0, 100.0, 2.0, 3.0, -50.0];
        // sorted: -50, 1, 2, 3, 5, 100. trim_low=1, trim_high=1: the -50 is
        // replaced by 1 (smallest retained) and the 100 by 5 (largest
        // retained), then ALL SIX values are averaged:
        // (1 + 1 + 2 + 3 + 5 + 5) / 6 = 17/6.
        let r = censored_mean(&mut data, 1, 1).unwrap();
        assert_eq!(r.retained, 4);
        assert_eq!(r.censored_low, 1);
        assert_eq!(r.censored_high, 1);
        assert!((r.estimate - 17.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn censored_mean_no_trim_is_the_plain_mean() {
        let mut data = [1.0, 2.0, 3.0, 4.0];
        let r = censored_mean(&mut data, 0, 0).unwrap();
        assert!((r.estimate - 2.5).abs() < 1e-12);
    }

    #[test]
    fn censored_mean_rejects_trim_leaving_no_retained_cell() {
        let mut data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            censored_mean(&mut data, 2, 2),
            Err(CfarError::InvalidTrimCounts {
                n_ref: 4,
                trim_low: 2,
                trim_high: 2
            })
        );
    }

    #[test]
    fn censored_mean_rejects_overflowing_trim_counts_instead_of_panicking() {
        let mut data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            censored_mean(&mut data, usize::MAX, usize::MAX),
            Err(CfarError::InvalidTrimCounts {
                n_ref: 4,
                trim_low: usize::MAX,
                trim_high: usize::MAX,
            })
        );
    }

    #[test]
    fn censored_mean_uses_all_cells_unlike_trimmed_mean() {
        // Same data, same trim counts: trimmed_mean averages 4 of 6 cells;
        // censored_mean averages all 6 (two of them replaced), so the two
        // must differ here (the discarded extremes are not equal to their
        // replacement neighbors).
        let mut trimmed_data = [5.0, 1.0, 100.0, 2.0, 3.0, -50.0];
        let mut censored_data = trimmed_data;
        let trimmed = trimmed_mean(&mut trimmed_data, 1, 1).unwrap();
        let censored = censored_mean(&mut censored_data, 1, 1).unwrap();
        assert_ne!(trimmed.estimate.to_bits(), censored.estimate.to_bits());
    }

    // ---- Threshold calibration ---------------------------------------------

    /// A small, self-contained, deterministic uniform/exponential source used
    /// *only* by this test module to cross-check the production quadrature-
    /// based [`exact_go_so_alpha`] against an *independently implemented*
    /// Monte-Carlo estimate of the same `P_fa` — two different methods
    /// (deterministic integration vs. simulation) agreeing is stronger
    /// evidence than either alone. Deliberately independent of the RNG used
    /// in `tests/vi_cfar_monte_carlo.rs` (`scirust-stats`'s `SplitMix64`) —
    /// three independent implementations (exact quadrature, this Monte
    /// Carlo, and the outer empirical validation) checking one claim.
    struct CalibrationRng(u64);

    impl CalibrationRng {
        fn uniform01(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
        }

        fn unit_exponential(&mut self) -> f64 {
            -self.uniform01().ln()
        }
    }

    /// Monte-Carlo estimate of the GO/SO threshold factor, independent of
    /// [`exact_go_so_alpha`]'s quadrature — test-only cross-check, not a
    /// production path (see the module docs, "Threshold calibration").
    fn calibrate_go_so_alpha_monte_carlo(reference_cells: usize, pfa: f64, take_max: bool) -> f64 {
        const TRIALS: usize = 300_000;
        const SEED: u64 = 0x5643_4641_525F_4341;
        let mut rng = CalibrationRng(SEED);
        let trials: Vec<(f64, f64, f64)> = (0..TRIALS)
            .map(|_| {
                let cut = rng.unit_exponential();
                let s1: f64 = (0..reference_cells).map(|_| rng.unit_exponential()).sum();
                let s2: f64 = (0..reference_cells).map(|_| rng.unit_exponential()).sum();
                (cut, s1, s2)
            })
            .collect();
        let n = reference_cells as f64;
        let pfa_of = |alpha: f64| -> f64 {
            let alarms = trials
                .iter()
                .filter(|&&(cut, s1, s2)| {
                    let combined = if take_max { s1.max(s2) } else { s1.min(s2) };
                    cut > (alpha / n) * combined
                })
                .count();
            alarms as f64 / TRIALS as f64
        };
        bisect_decreasing(pfa_of, pfa)
    }

    #[test]
    fn pfa_so_exact_has_the_correct_boundary_limits() {
        let n = 16.0;
        // alpha -> 0 (zero threshold): P_fa -> 1 exactly (P(CUT > 0) = 1).
        assert_eq!(pfa_so_exact(n, 0.0).unwrap(), 1.0);
        // A very large alpha drives P_fa toward 0.
        assert!(pfa_so_exact(n, 1.0e6).unwrap() < 1e-9);
    }

    #[test]
    fn pfa_go_exact_has_the_correct_boundary_limits() {
        let n = 16.0;
        assert_eq!(pfa_go_exact(n, 0.0).unwrap(), 1.0);
        assert!(pfa_go_exact(n, 1.0e6).unwrap() < 1e-9);
    }

    #[test]
    fn pfa_go_is_never_below_pfa_so_at_the_same_alpha() {
        // noise_GO = max(S1,S2) >= min(S1,S2) = noise_SO pointwise, so at a
        // fixed alpha GO's threshold is never below SO's, hence GO's P_fa is
        // never above SO's — checked here as an inequality on the exact
        // formulas themselves (not just on simulated outcomes).
        let n = 12.0;
        for alpha in [0.5, 1.0, 3.0, 8.0, 20.0]
        {
            let go = pfa_go_exact(n, alpha).unwrap();
            let so = pfa_so_exact(n, alpha).unwrap();
            assert!(
                go <= so,
                "alpha={alpha}: pfa_go={go} should be <= pfa_so={so}"
            );
        }
    }

    #[test]
    fn exact_go_so_alpha_agrees_with_independent_monte_carlo_calibration() {
        // The production (quadrature) and test-only (Monte Carlo) methods
        // are independently derived and independently implemented; close
        // agreement is strong evidence neither has a bug the other shares.
        for &(n, pfa) in &[(8usize, 0.05), (16, 0.02), (32, 0.01)]
        {
            for take_max in [true, false]
            {
                let exact = exact_go_so_alpha(n, pfa, take_max).unwrap();
                let monte_carlo = calibrate_go_so_alpha_monte_carlo(n, pfa, take_max);
                let rel_err = (exact - monte_carlo).abs() / exact.max(1e-9);
                assert!(
                    rel_err < 0.03,
                    "n={n}, pfa={pfa}, take_max={take_max}: exact={exact}, monte_carlo={monte_carlo}, rel_err={rel_err}"
                );
            }
        }
    }

    // ---- Design-point calibration helpers (k_vi, k_mr) ----------------------

    #[test]
    fn calibrate_k_mr_holds_its_design_alpha_empirically() {
        // Independent empirical check of the F(2n,2n) derivation: draw two
        // fresh, independent i.i.d.-Exp(1) half-windows per trial (a
        // *different* RNG than calibrate_k_vi's own DesignPointRng, using
        // the module's other private LCG), and confirm the observed
        // exceedance rate of the calibrated k_mr matches the target alpha.
        for &(n, alpha) in &[(8usize, 0.05), (16, 0.02), (32, 0.1)]
        {
            let k_mr = calibrate_k_mr(n, alpha).unwrap();
            let trials = 200_000;
            let mut rng = CalibrationRng(0x4b5f_4d52_5f43_4845 ^ (n as u64));
            let exceed = (0..trials)
                .filter(|_| {
                    let a: f64 = (0..n).map(|_| rng.unit_exponential()).sum();
                    let b: f64 = (0..n).map(|_| rng.unit_exponential()).sum();
                    (a / b) > k_mr
                })
                .count() as f64
                / trials as f64;
            let se = (alpha * (1.0 - alpha) / trials as f64).sqrt();
            assert!(
                (exceed - alpha).abs() < 4.0 * se,
                "n={n}, alpha={alpha}: k_mr={k_mr}, observed exceedance={exceed}, target={alpha}, 4*SE={}",
                4.0 * se
            );
        }
    }

    #[test]
    fn calibrate_k_mr_matches_its_own_boundary_expectations() {
        // At n=1 boundary excluded by TooFewReferenceCells; check the
        // monotonicity any quantile function must have: a smaller alpha
        // (rarer false-non-homogeneity) demands a larger k_mr.
        let loose = calibrate_k_mr(16, 0.2).unwrap();
        let tight = calibrate_k_mr(16, 0.01).unwrap();
        assert!(
            tight > loose,
            "smaller alpha should require a larger k_mr: tight={tight} vs loose={loose}"
        );
        // k_mr must be > 1: MR=1 (equal means) can never itself count as
        // "too imbalanced", for any alpha < 1.
        assert!(loose > 1.0 && tight > 1.0);
    }

    #[test]
    fn calibrate_k_mr_rejects_invalid_input() {
        assert!(matches!(
            calibrate_k_mr(1, 0.05),
            Err(CfarError::TooFewReferenceCells(1))
        ));
        assert!(matches!(
            calibrate_k_mr(16, 0.0),
            Err(CfarError::ExactCalibrationFailed(_))
        ));
        assert!(matches!(
            calibrate_k_mr(16, 1.0),
            Err(CfarError::ExactCalibrationFailed(_))
        ));
        assert!(matches!(
            calibrate_k_mr(16, f64::NAN),
            Err(CfarError::ExactCalibrationFailed(_))
        ));
    }

    #[test]
    fn calibrate_k_vi_holds_its_design_alpha_empirically() {
        // Independent empirical check, using yet another RNG/seed than the
        // one calibrate_k_vi uses internally, matching this module's
        // established convention of cross-checking every calibration
        // against a materially independent generator.
        for &(n, alpha) in &[(8usize, 0.05), (16, 0.02)]
        {
            let k_vi = calibrate_k_vi(n, alpha, 300_000, 0x4b5f_5649_5f43_414c).unwrap();
            let trials = 200_000;
            let mut rng = CalibrationRng(0x5645_5249_4659_4b56 ^ (n as u64));
            let exceed = (0..trials)
                .filter(|_| {
                    let cells: Vec<f64> = (0..n).map(|_| rng.unit_exponential()).collect();
                    let mean = cells.iter().sum::<f64>() / n as f64;
                    let sample_variance =
                        cells.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>()
                            / (n as f64 - 1.0);
                    variability_index(mean, sample_variance) > k_vi
                })
                .count() as f64
                / trials as f64;
            let se = (alpha * (1.0 - alpha) / trials as f64).sqrt();
            assert!(
                (exceed - alpha).abs() < 4.0 * se,
                "n={n}, alpha={alpha}: k_vi={k_vi}, observed exceedance={exceed}, target={alpha}, 4*SE={}",
                4.0 * se
            );
        }
    }

    #[test]
    fn calibrate_k_vi_rejects_invalid_input() {
        assert!(matches!(
            calibrate_k_vi(1, 0.05, 1000, 0),
            Err(CfarError::TooFewReferenceCells(1))
        ));
        assert!(matches!(
            calibrate_k_vi(16, 0.0, 1000, 0),
            Err(CfarError::ExactCalibrationFailed(_))
        ));
        assert!(matches!(
            calibrate_k_vi(16, 0.05, 0, 0),
            Err(CfarError::ExactCalibrationFailed(_))
        ));
    }

    #[test]
    fn calibrate_k_vi_rejects_reference_cells_beyond_the_practical_bound() {
        // Symmetry with `CfarConfig::validate`'s own practical bound:
        // `calibrate_k_vi`'s `reference_cells` is the same per-side
        // reference-window size, so it is bounded the same way.
        assert!(matches!(
            calibrate_k_vi(MAX_PRACTICAL_REFERENCE_CELLS + 1, 0.05, 1000, 0),
            Err(CfarError::ExactCalibrationFailed(_))
        ));
    }

    #[test]
    fn calibrate_k_vi_rejects_trials_beyond_the_practical_bound() {
        assert!(matches!(
            calibrate_k_vi(16, 0.05, MAX_PRACTICAL_MONTE_CARLO_TRIALS + 1, 0),
            Err(CfarError::ExactCalibrationFailed(_))
        ));
    }

    #[test]
    fn calibrate_k_vi_rejects_a_reasonable_looking_product_beyond_the_practical_bound() {
        // New-finding regression: `reference_cells` and `trials` can each
        // look entirely reasonable in isolation (both far under their own
        // caps) while their product -- the actual O(reference_cells *
        // trials) sample-generation cost -- is not. Direct timing before
        // this fix showed this class of combination taking multiple
        // seconds and scaling linearly with the product, unbounded.
        let reference_cells = 2_000;
        let trials = MAX_PRACTICAL_MONTE_CARLO_SAMPLES / reference_cells + 1;
        assert!(reference_cells <= MAX_PRACTICAL_REFERENCE_CELLS);
        assert!(trials <= MAX_PRACTICAL_MONTE_CARLO_TRIALS);
        assert!(matches!(
            calibrate_k_vi(reference_cells, 0.05, trials, 0),
            Err(CfarError::ExactCalibrationFailed(_))
        ));
    }

    #[test]
    fn calibrate_k_vi_accepts_a_product_right_at_the_practical_bound() {
        // Boundary check: sitting exactly at `MAX_PRACTICAL_MONTE_CARLO_SAMPLES`
        // must still succeed (and complete promptly) -- not be off-by-one
        // rejected.
        let reference_cells = MAX_PRACTICAL_REFERENCE_CELLS;
        let trials = MAX_PRACTICAL_MONTE_CARLO_SAMPLES / reference_cells;
        assert!(trials <= MAX_PRACTICAL_MONTE_CARLO_TRIALS);
        assert!(calibrate_k_vi(reference_cells, 0.05, trials, 0).is_ok());
    }

    #[test]
    fn calibrate_k_vi_is_deterministic() {
        let a = calibrate_k_vi(16, 0.05, 5_000, 0x1234).unwrap();
        let b = calibrate_k_vi(16, 0.05, 5_000, 0x1234).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
        let c = calibrate_k_vi(16, 0.05, 5_000, 0x5678).unwrap();
        assert_ne!(
            a.to_bits(),
            c.to_bits(),
            "a different seed should (almost surely) differ"
        );
    }

    #[test]
    fn trimmed_mean_alpha_with_no_trim_matches_ca_cfar_alpha() {
        let (n_ref, pfa) = (32, 0.01);
        let trimmed = trimmed_mean_alpha(n_ref, 0, 0, pfa);
        let ca = super::super::cfar::ca_cfar_alpha(n_ref, pfa);
        assert!(
            (trimmed - ca).abs() < 1e-6,
            "trimmed(0,0)={trimmed}, ca={ca}"
        );
    }

    #[test]
    fn trimmed_mean_alpha_keeping_only_the_smallest_matches_os_cfar_k1() {
        let (n_ref, pfa) = (32, 0.01);
        let trimmed = trimmed_mean_alpha(n_ref, 0, n_ref - 1, pfa); // keep only X(1)
        let os = super::super::cfar::os_cfar_alpha(n_ref, 1, pfa);
        assert!(
            (trimmed - os).abs() < 1e-6,
            "trimmed={trimmed}, os(k=1)={os}"
        );
    }

    #[test]
    fn trimmed_mean_alpha_keeping_only_the_largest_matches_os_cfar_kn() {
        let (n_ref, pfa) = (32, 0.01);
        let trimmed = trimmed_mean_alpha(n_ref, n_ref - 1, 0, pfa); // keep only X(N)
        let os = super::super::cfar::os_cfar_alpha(n_ref, n_ref, pfa);
        assert!(
            (trimmed - os).abs() < 1e-6,
            "trimmed={trimmed}, os(k=N)={os}"
        );
    }

    #[test]
    fn censored_mean_alpha_with_no_trim_matches_ca_cfar_alpha() {
        // No Winsorizing at all reduces to plain cell-averaging.
        let (n_ref, pfa) = (32, 0.01);
        let censored = censored_mean_alpha(n_ref, 0, 0, pfa);
        let ca = super::super::cfar::ca_cfar_alpha(n_ref, pfa);
        assert!(
            (censored - ca).abs() < 1e-6,
            "censored(0,0)={censored}, ca={ca}"
        );
    }

    #[test]
    fn censored_mean_alpha_differs_from_trimmed_mean_alpha_when_censoring() {
        // Same n_ref/trim counts/pfa, but Winsorizing (average all n_ref,
        // some replaced) is a genuinely different estimator from discarding
        // (average only the kept n_ref-trim_low-trim_high), so their exact
        // alphas should differ.
        let (n_ref, pfa) = (16, 0.02);
        let trimmed = trimmed_mean_alpha(n_ref, 2, 2, pfa);
        let censored = censored_mean_alpha(n_ref, 2, 2, pfa);
        assert!(
            (trimmed - censored).abs() > 1e-3,
            "trimmed={trimmed}, censored={censored} should differ"
        );
    }

    #[test]
    fn censored_mean_alpha_is_deterministic_and_monotonic_in_pfa() {
        let n_ref = 16;
        let low = censored_mean_alpha(n_ref, 1, 1, 0.001);
        let mid = censored_mean_alpha(n_ref, 1, 1, 0.01);
        let high = censored_mean_alpha(n_ref, 1, 1, 0.1);
        // A smaller design Pfa needs a larger threshold multiplier.
        assert!(low > mid && mid > high, "low={low}, mid={mid}, high={high}");
        assert_eq!(
            censored_mean_alpha(n_ref, 1, 1, 0.01).to_bits(),
            mid.to_bits()
        );
    }

    #[test]
    fn go_so_calibration_brackets_ca_in_the_expected_order() {
        // Classical qualitative fact: for the same target Pfa, GO's
        // systematically-larger noise estimate needs a *smaller* multiplier
        // than CA, and SO's systematically-smaller estimate needs a *larger*
        // one: alpha_go < alpha_ca < alpha_so.
        let (n_per_side, pfa) = (16, 0.05);
        let alpha_ca = super::super::cfar::ca_cfar_alpha(2 * n_per_side, pfa);
        let alpha_go = exact_go_so_alpha(n_per_side, pfa, true).unwrap();
        let alpha_so = exact_go_so_alpha(n_per_side, pfa, false).unwrap();
        assert!(
            alpha_go < alpha_ca && alpha_ca < alpha_so,
            "alpha_go={alpha_go}, alpha_ca={alpha_ca}, alpha_so={alpha_so}"
        );
    }

    #[test]
    fn calibration_is_deterministic_across_repeated_calls() {
        let a = exact_go_so_alpha(12, 0.02, true).unwrap();
        let b = exact_go_so_alpha(12, 0.02, true).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // ---- CfarDetector (reusable, pre-calibrated) ----------------------------

    #[test]
    fn cfar_detector_matches_evaluate_slice() {
        let mut power = vec![1.0_f64; 60];
        power[30] = 200.0;
        let config = default_config();
        let via_free_fn = evaluate_slice(&power, &config).unwrap();
        let via_detector = CfarDetector::new(config).unwrap().evaluate(&power).unwrap();
        assert_eq!(via_free_fn, via_detector);
    }

    #[test]
    fn cfar_detector_is_reusable_across_many_evaluate_calls() {
        let config = default_config();
        let mut detector = CfarDetector::new(config).unwrap();
        let mut power_a = vec![1.0_f64; 60];
        power_a[30] = 200.0;
        let power_b = vec![1.0_f64; 60];

        let a1 = detector.evaluate(&power_a).unwrap();
        let b = detector.evaluate(&power_b).unwrap();
        let a2 = detector.evaluate(&power_a).unwrap();

        assert!(a1.iter().find(|d| d.cut_index == 30).unwrap().detected);
        assert!(b.iter().all(|d| !d.detected));
        assert_eq!(a1, a2, "reusing the detector must not change its behavior");
    }

    // ---- evaluate_slice: basic scenarios ------------------------------------

    #[test]
    fn homogeneous_flat_floor_no_detection() {
        let power = vec![1.0_f64; 60];
        let config = default_config();
        let decisions = evaluate_slice(&power, &config).unwrap();
        assert!(!decisions.is_empty());
        assert!(decisions.iter().all(|d| !d.detected));
        assert!(decisions.iter().all(|d| d.mode == CfarMode::Ca));
    }

    #[test]
    fn isolated_target_on_flat_floor_is_detected() {
        let mut power = vec![1.0_f64; 60];
        power[30] = 200.0;
        let config = default_config();
        let decisions = evaluate_slice(&power, &config).unwrap();
        let at_target = decisions.iter().find(|d| d.cut_index == 30).unwrap();
        assert!(at_target.detected);
        assert_eq!(at_target.mode, CfarMode::Ca);
    }

    #[test]
    fn insufficient_length_yields_no_decisions_not_an_error() {
        let power = vec![1.0_f64; 5];
        let config = default_config();
        let decisions = evaluate_slice(&power, &config).unwrap();
        assert!(decisions.is_empty());
    }

    #[test]
    fn all_zero_reference_cells_and_zero_cut() {
        let power = vec![0.0_f64; 60];
        let config = default_config();
        let decisions = evaluate_slice(&power, &config).unwrap();
        assert!(decisions.iter().all(|d| !d.detected));
        assert!(decisions.iter().all(|d| d.mean_ratio == 1.0));
        assert!(decisions.iter().all(|d| d.noise_estimate == 0.0));
        // VI must be the documented 1.0, not NaN: an all-zero window's
        // classification must be an explicit contract, not an accident of
        // `NaN`-comparison-is-always-false.
        assert!(
            decisions
                .iter()
                .all(|d| d.lagging_vi == 1.0 && d.leading_vi == 1.0)
        );
    }

    #[test]
    fn variability_index_all_zero_window_is_defined_not_nan() {
        assert_eq!(variability_index(0.0, 0.0), 1.0);
    }

    #[test]
    fn very_large_cut_on_flat_floor_is_detected() {
        let mut power = vec![1.0_f64; 60];
        power[30] = 1.0e12;
        let decisions = evaluate_slice(&power, &default_config()).unwrap();
        assert!(
            decisions
                .iter()
                .find(|d| d.cut_index == 30)
                .unwrap()
                .detected
        );
    }

    #[test]
    fn nan_sample_is_rejected_with_its_index() {
        let mut power = vec![1.0_f64; 60];
        power[12] = f64::NAN;
        let err = evaluate_slice(&power, &default_config()).unwrap_err();
        assert!(matches!(err, CfarError::NonFiniteSample { index: 12, .. }));
    }

    #[test]
    fn negative_sample_is_rejected_under_reject_negative_policy() {
        let mut power = vec![1.0_f64; 60];
        power[9] = -1.0;
        let err = evaluate_slice(&power, &default_config()).unwrap_err();
        assert!(matches!(err, CfarError::NegativeSample { index: 9, .. }));
    }

    #[test]
    fn negative_sample_accepted_under_allow_negative_policy() {
        let mut power = vec![1.0_f64; 60];
        power[9] = -1.0;
        let mut config = default_config();
        config.input_validation = InputValidationPolicy::AllowNegative;
        assert!(evaluate_slice(&power, &config).is_ok());
    }

    #[test]
    fn invalid_config_is_rejected_before_scanning_samples() {
        let power = vec![1.0_f64; 60];
        let mut config = default_config();
        config.pfa = 2.0;
        assert_eq!(
            evaluate_slice(&power, &config),
            Err(CfarError::InvalidPfa(2.0))
        );
    }

    // ---- Classical switching structure -------------------------------------

    fn vi_cfar_config(
        reference_cells: usize,
        guard_cells: usize,
        k_vi: f64,
        k_mr: f64,
    ) -> CfarConfig {
        CfarConfig {
            reference_cells,
            guard_cells,
            pfa: 0.01,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector: DetectorPolicy::ClassicalViCfar(SwitchingThresholds { k_vi, k_mr }),
            robust_estimator: RobustNoiseEstimator::TrimmedMean {
                trim_low: 2,
                trim_high: 2,
            },
        }
    }

    #[test]
    fn homogeneous_clutter_selects_ca() {
        let power = vec![1.0_f64; 80];
        let config = vi_cfar_config(16, 2, 6.0, 2.0);
        let decisions = evaluate_slice(&power, &config).unwrap();
        assert!(decisions.iter().all(|d| d.mode == CfarMode::Ca));
        assert!(decisions.iter().all(|d| !d.detected));
    }

    #[test]
    fn clutter_edge_without_interferers_selects_go() {
        // Step up in the noise floor at index 40; both halves stay internally
        // homogeneous (VI near 2) until the CUT straddles the step, but the
        // means differ sharply there, so MR should trip the edge branch.
        let mut power = vec![1.0_f64; 80];
        for x in power.iter_mut().skip(40)
        {
            *x = 20.0;
        }
        let config = vi_cfar_config(16, 2, 6.0, 1.3);
        let decisions = evaluate_slice(&power, &config).unwrap();
        let at_edge = decisions.iter().find(|d| d.cut_index == 40).unwrap();
        assert_eq!(at_edge.mode, CfarMode::Go, "{at_edge:?}");
    }

    #[test]
    fn clutter_edge_cut_before_on_and_after_the_transition() {
        // Same step as above; check the CUT well before the step (both halves
        // low, MR~1 -> CA), straddling it on each side within
        // `guard+reference_cells` of index 40 (-> GO), and well after (both
        // halves high, MR~1 -> CA again) — plus a threshold-continuity
        // diagnostic: the straddling GO threshold should land between the
        // low-floor and high-floor CA thresholds, not jump outside that band.
        let mut power = vec![1.0_f64; 100];
        for x in power.iter_mut().skip(40)
        {
            *x = 20.0;
        }
        let config = vi_cfar_config(16, 2, 6.0, 1.3);
        let decisions = evaluate_slice(&power, &config).unwrap();
        let at = |i: usize| decisions.iter().find(|d| d.cut_index == i).unwrap();

        assert_eq!(
            at(20).mode,
            CfarMode::Ca,
            "well before the edge: {:?}",
            at(20)
        );
        assert_eq!(
            at(70).mode,
            CfarMode::Ca,
            "well after the edge: {:?}",
            at(70)
        );
        // Lagging ref for cut `c` is `[c-guard-reference, c-guard-1]` = `[c-18,
        // c-3]`; it starts touching the step (index 40) once `c-3 >= 40`, i.e.
        // `c >= 43`, and is fully past it once `c-18 >= 40`, i.e. `c >= 58` —
        // so `[43, 57]` is where lagging is a *mixed* low/high window. Pick
        // cuts spanning that range, including its near-symmetric midpoint
        // (50: 8 low + 8 high cells), where the mean-ratio imbalance is
        // strongest: right at either end of the mixed range one side is
        // nearly homogeneous again (e.g. cut 57 has only one low cell left in
        // 16, MR too close to 1.3 to trip — a real, correct effect, not a
        // test bug) even though the leading window is already fully cleared
        // from cut 39 on (`cut+guard+1 >= 40` i.e. `cut >= 39`).
        for cut in [39usize, 45, 50]
        {
            assert_eq!(
                at(cut).mode,
                CfarMode::Go,
                "straddling the edge at {cut}: {:?}",
                at(cut)
            );
        }

        let low_threshold = at(20).threshold;
        let high_threshold = at(70).threshold;
        for cut in [39usize, 45, 50]
        {
            let t = at(cut).threshold;
            assert!(
                t >= low_threshold.min(high_threshold)
                    && t <= low_threshold.max(high_threshold) * 1.01,
                "threshold continuity: cut={cut} threshold={t}, low={low_threshold}, high={high_threshold}"
            );
        }
    }

    #[test]
    fn target_spacing_near_guard_cells_is_still_correctly_excluded() {
        // An interferer placed exactly at the guard/reference boundary must
        // land in the reference window (biasing the mean) when just inside
        // it, and must NOT when it is a guard cell (guard cells are excluded
        // from every mean/variance computation regardless of their value).
        let (reference_cells, guard_cells) = (16usize, 2usize);
        let cut = 40usize;
        let mut power = vec![1.0_f64; 90];
        power[cut] = 8.0;
        // Last lagging-guard cell (must NOT affect the lagging mean at all).
        power[cut - 1] = 1000.0;
        let config = vi_cfar_config(reference_cells, guard_cells, 6.0, 2.0);
        let decisions = evaluate_slice(&power, &config).unwrap();
        let d = decisions.iter().find(|dd| dd.cut_index == cut).unwrap();
        assert!(
            (d.lagging_mean - 1.0).abs() < 1e-9,
            "a guard cell must not bias the lagging mean: {d:?}"
        );

        // Now move the same interferer one cell further out: the *last*
        // lagging reference cell — it must now bias the lagging mean.
        let mut power2 = vec![1.0_f64; 90];
        power2[cut] = 8.0;
        power2[cut - guard_cells - 1] = 1000.0;
        let decisions2 = evaluate_slice(&power2, &config).unwrap();
        let d2 = decisions2.iter().find(|dd| dd.cut_index == cut).unwrap();
        assert!(
            d2.lagging_mean > 10.0,
            "the last reference cell must bias the lagging mean: {d2:?}"
        );
    }

    #[test]
    fn single_sided_interferer_selects_so() {
        // A strong interferer sits only in the lagging half-window relative
        // to a weak target CUT; the lagging half's VI spikes, the leading
        // half stays homogeneous.
        let mut power = vec![1.0_f64; 80];
        power[30] = 8.0; // weak target CUT
        power[16] = 300.0; // strong interferer in the lagging reference window
        let config = vi_cfar_config(16, 2, 6.0, 2.0);
        let decisions = evaluate_slice(&power, &config).unwrap();
        let at_cut = decisions.iter().find(|d| d.cut_index == 30).unwrap();
        assert_eq!(at_cut.mode, CfarMode::So, "{at_cut:?}");
        assert!(at_cut.detected, "SO should still see the weak target");
    }

    #[test]
    fn double_contamination_selects_robust_trimmed_not_a_classical_mode() {
        let mut power = vec![1.0_f64; 80];
        power[30] = 8.0; // weak target CUT
        power[16] = 300.0; // interferer in the lagging half
        power[44] = 300.0; // interferer in the leading half
        let config = vi_cfar_config(16, 2, 6.0, 2.0);
        let decisions = evaluate_slice(&power, &config).unwrap();
        let at_cut = decisions.iter().find(|d| d.cut_index == 30).unwrap();
        assert_eq!(at_cut.mode, CfarMode::RobustTrimmed, "{at_cut:?}");
    }

    #[test]
    fn double_contamination_with_censored_mean_selects_robust_censored() {
        let mut power = vec![1.0_f64; 80];
        power[30] = 8.0; // weak target CUT
        power[16] = 300.0; // interferer in the lagging half
        power[44] = 300.0; // interferer in the leading half
        let mut config = vi_cfar_config(16, 2, 6.0, 2.0);
        config.robust_estimator = RobustNoiseEstimator::CensoredMean {
            trim_low: 2,
            trim_high: 2,
        };
        let decisions = evaluate_slice(&power, &config).unwrap();
        let at_cut = decisions.iter().find(|d| d.cut_index == 30).unwrap();
        assert_eq!(at_cut.mode, CfarMode::RobustCensored, "{at_cut:?}");
        assert!(
            at_cut.detected,
            "censored mean should still see the weak target"
        );
    }

    #[test]
    fn robust_trimmed_reduces_masking_versus_classical_so_under_double_contamination() {
        // A weak target with an interferer in *each* half. Classical SO
        // (forced) takes min(lag_mean, lead_mean) — both halves are inflated
        // by their own interferer, so SO's noise estimate is still pulled up
        // by whichever interferer is smaller, more than a trimmed mean
        // (which discards the single largest cell from each half) is. This
        // demonstrates, on this one controlled construction, that the robust
        // path's noise estimate is lower than classical SO's; it is not a
        // claim that this holds for every possible contamination pattern.
        let mut power = vec![1.0_f64; 80];
        power[30] = 8.0;
        power[16] = 300.0;
        power[44] = 250.0;
        let mut so_config = vi_cfar_config(16, 2, 6.0, 2.0);
        so_config.detector = DetectorPolicy::So;
        let so_noise = evaluate_slice(&power, &so_config)
            .unwrap()
            .into_iter()
            .find(|d| d.cut_index == 30)
            .unwrap()
            .noise_estimate;

        let mut robust_config = vi_cfar_config(16, 2, 6.0, 2.0);
        robust_config.detector = DetectorPolicy::AlwaysRobust;
        let robust_noise = evaluate_slice(&power, &robust_config)
            .unwrap()
            .into_iter()
            .find(|d| d.cut_index == 30)
            .unwrap()
            .noise_estimate;

        assert!(
            robust_noise < so_noise,
            "robust_noise={robust_noise} should be below classical so_noise={so_noise}"
        );
    }

    #[test]
    fn several_interferers_in_one_half_are_censored_by_trimming() {
        let mut power = vec![1.0_f64; 80];
        power[30] = 8.0;
        for i in [16usize, 17, 18]
        {
            power[i] = 300.0;
        }
        let mut config = vi_cfar_config(16, 2, 6.0, 2.0);
        config.robust_estimator = RobustNoiseEstimator::TrimmedMean {
            trim_low: 0,
            trim_high: 4,
        };
        config.detector = DetectorPolicy::AlwaysRobust;
        let decisions = evaluate_slice(&power, &config).unwrap();
        let at_cut = decisions.iter().find(|d| d.cut_index == 30).unwrap();
        assert!(
            at_cut.detected,
            "trimming 3 interferers should still detect the weak target"
        );
    }

    // ---- 2-D range-Doppler API -----------------------------------------------

    #[test]
    #[allow(clippy::needless_range_loop)] // 2-D indexing by (range, doppler) reads clearer explicit
    fn vi_cfar_2d_matches_per_column_evaluate_slice() {
        // Cross-check against the already-verified 1-D path: build a
        // range-Doppler map with a few embedded targets, run `vi_cfar_2d`,
        // and independently run `evaluate_slice` on each column extracted
        // by hand -- every `detected` flag must agree exactly.
        let (rows, cols) = (80, 5);
        let mut power = vec![vec![1.0_f64; cols]; rows];
        power[30][1] = 40.0; // isolated target in column 1
        power[50][3] = 60.0; // isolated target in column 3
        let config = vi_cfar_config(16, 2, 6.0, 2.0);

        let det = vi_cfar_2d(&power, &config).unwrap();
        assert_eq!(det.len(), rows);
        for d in 0..cols
        {
            let column: Vec<f64> = (0..rows).map(|r| power[r][d]).collect();
            let decisions = evaluate_slice(&column, &config).unwrap();
            for decision in &decisions
            {
                assert_eq!(
                    det[decision.cut_index][d], decision.detected,
                    "column {d}, range {}: 2-D vs 1-D disagree",
                    decision.cut_index
                );
            }
            // Every range outside the evaluated (edge-excluded) interior
            // must stay `false` -- `vi_cfar_2d` never flags a cell
            // `evaluate_slice` itself would not have evaluated.
            let evaluated: std::collections::HashSet<usize> =
                decisions.iter().map(|dd| dd.cut_index).collect();
            for r in 0..rows
            {
                if !evaluated.contains(&r)
                {
                    assert!(
                        !det[r][d],
                        "column {d}, range {r}: flagged without a full window"
                    );
                }
            }
        }
        assert!(det[30][1], "target at (30, 1) should be detected");
        assert!(det[50][3], "target at (50, 3) should be detected");
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // 2-D indexing by (range, doppler) reads clearer explicit
    fn vi_cfar_2d_detects_only_the_embedded_target() {
        let (rows, cols) = (60, 3);
        let mut power = vec![vec![1.0_f64; cols]; rows];
        power[25][1] = 50.0;
        let config = vi_cfar_config(16, 2, 6.0, 2.0);
        let det = vi_cfar_2d(&power, &config).unwrap();
        for r in 0..rows
        {
            for c in 0..cols
            {
                let expect_detected = (r, c) == (25, 1);
                assert_eq!(
                    det[r][c], expect_detected,
                    "unexpected detection state at ({r}, {c})"
                );
            }
        }
    }

    #[test]
    fn vi_cfar_2d_empty_map_returns_empty() {
        let config = vi_cfar_config(16, 2, 6.0, 2.0);
        let det = vi_cfar_2d(&[], &config).unwrap();
        assert!(det.is_empty());
    }

    #[test]
    fn vi_cfar_2d_ragged_rows_return_an_all_false_mask() {
        // Matches `ca_cfar_2d`'s convention: a shape problem in the
        // *measurement* is not a configuration error, so it is reported by
        // an all-`false` mask (sized from the first row), not `Err`.
        let power = vec![vec![1.0; 5], vec![1.0; 4], vec![1.0; 5]];
        let config = vi_cfar_config(16, 2, 6.0, 2.0);
        let det = vi_cfar_2d(&power, &config).unwrap();
        assert_eq!(det.len(), 3);
        assert!(
            det.iter()
                .all(|row| row.len() == 5 && row.iter().all(|&d| !d))
        );
    }

    #[test]
    fn vi_cfar_2d_propagates_a_calibration_error() {
        let mut config = vi_cfar_config(16, 2, 6.0, 2.0);
        config.reference_cells = 1; // invalid: TooFewReferenceCells
        let power = vec![vec![1.0; 4]; 4];
        let err = vi_cfar_2d(&power, &config).unwrap_err();
        assert!(matches!(err, CfarError::TooFewReferenceCells(1)));
    }

    #[test]
    fn vi_cfar_2d_propagates_a_calibration_error_even_with_a_degenerate_shape() {
        // Adversarial-audit regression: an invalid config used to be
        // silently swallowed (returning `Ok`) whenever `power` was also
        // empty, ragged, or zero-width, because the shape checks ran
        // *before* `CfarDetector::new` -- contradicting this function's own
        // documented contract that config errors always surface via `Err`.
        let mut invalid = vi_cfar_config(16, 2, 6.0, 2.0);
        invalid.reference_cells = 1; // invalid: TooFewReferenceCells

        let err = vi_cfar_2d(&[], &invalid).unwrap_err();
        assert!(
            matches!(err, CfarError::TooFewReferenceCells(1)),
            "empty power: {err:?}"
        );

        let ragged = vec![vec![1.0; 5], vec![1.0; 4]];
        let err = vi_cfar_2d(&ragged, &invalid).unwrap_err();
        assert!(
            matches!(err, CfarError::TooFewReferenceCells(1)),
            "ragged power: {err:?}"
        );

        let zero_width: Vec<Vec<f64>> = vec![Vec::new(); 3];
        let err = vi_cfar_2d(&zero_width, &invalid).unwrap_err();
        assert!(
            matches!(err, CfarError::TooFewReferenceCells(1)),
            "zero-width power: {err:?}"
        );
    }

    // ---- Streaming API ------------------------------------------------------

    #[test]
    fn streaming_matches_finite_slice_on_interior_points() {
        let reference_cells = 8;
        let guard = 2;
        let mut power = vec![1.0_f64; 100];
        power[50] = 40.0;
        let config = CfarConfig {
            reference_cells,
            guard_cells: guard,
            pfa: 0.01,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector: DetectorPolicy::Ca,
            robust_estimator: RobustNoiseEstimator::TrimmedMean {
                trim_low: 1,
                trim_high: 1,
            },
        };
        let finite = evaluate_slice(&power, &config).unwrap();

        let mut streamed = Vec::new();
        let mut detector = CfarStreamDetector::new(config).unwrap();
        for &x in &power
        {
            if let Some(d) = detector.push(x).unwrap()
            {
                streamed.push(d);
            }
        }

        assert_eq!(finite.len(), streamed.len());
        for (f, s) in finite.iter().zip(streamed.iter())
        {
            assert_eq!(f.cut_index, s.cut_index);
            assert_eq!(f.detected, s.detected);
            assert!((f.threshold - s.threshold).abs() < 1e-9, "{f:?} vs {s:?}");
            assert!((f.noise_estimate - s.noise_estimate).abs() < 1e-9);
        }
    }

    #[test]
    fn streaming_zero_guard_matches_finite_slice() {
        let reference_cells = 6;
        let mut power = vec![1.0_f64; 60];
        power[30] = 25.0;
        let config = CfarConfig {
            reference_cells,
            guard_cells: 0,
            pfa: 0.02,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector: DetectorPolicy::Ca,
            robust_estimator: RobustNoiseEstimator::TrimmedMean {
                trim_low: 1,
                trim_high: 1,
            },
        };
        let finite = evaluate_slice(&power, &config).unwrap();
        let mut streamed = Vec::new();
        let mut detector = CfarStreamDetector::new(config).unwrap();
        for &x in &power
        {
            if let Some(d) = detector.push(x).unwrap()
            {
                streamed.push(d);
            }
        }
        assert_eq!(finite.len(), streamed.len());
        for (f, s) in finite.iter().zip(streamed.iter())
        {
            assert_eq!(f.cut_index, s.cut_index);
            assert_eq!(f.detected, s.detected);
        }
    }

    #[test]
    fn streaming_latency_contract() {
        let reference_cells = 4;
        let guard = 1;
        let config = CfarConfig {
            reference_cells,
            guard_cells: guard,
            pfa: 0.05,
            edge_policy: EdgePolicy::Exclude,
            input_validation: InputValidationPolicy::RejectNegative,
            detector: DetectorPolicy::Ca,
            robust_estimator: RobustNoiseEstimator::TrimmedMean {
                trim_low: 0,
                trim_high: 0,
            },
        };
        let mut detector = CfarStreamDetector::new(config).unwrap();
        let expected_first_output_push = 2 * reference_cells + 2 * guard; // 0-based index of the push that first returns Some
        let mut first_output_push = None;
        for i in 0..40
        {
            if let Some(d) = detector.push(1.0 + (i as f64) * 0.0).unwrap()
            {
                first_output_push = Some(i);
                assert_eq!(d.cut_index, i - guard - reference_cells);
                break;
            }
        }
        assert_eq!(first_output_push, Some(expected_first_output_push));
    }

    #[test]
    fn streaming_reference_cells_is_runtime_configurable() {
        // The fix for this crate's earlier documented limitation ("TRAIN
        // must be a compile-time constant"): `CfarStreamDetector` is no
        // longer generic at all, so a caller can build detectors sized from
        // values only known at runtime (here, deliberately routed through a
        // `Vec` rather than a `const`) without triggering a distinct
        // monomorphization per size.
        let runtime_sizes: Vec<usize> = vec![3, 5, 9, 20];
        for reference_cells in runtime_sizes
        {
            let mut config = default_config();
            config.reference_cells = reference_cells;
            let mut detector = CfarStreamDetector::new(config).unwrap();
            let mut saw_output = false;
            for i in 0..(4 * reference_cells + 10)
            {
                if detector.push(1.0 + (i as f64) * 0.01).unwrap().is_some()
                {
                    saw_output = true;
                }
            }
            assert!(
                saw_output,
                "reference_cells={reference_cells} should have produced output"
            );
        }
    }

    #[test]
    fn streaming_propagates_non_finite_rejection() {
        let mut config = default_config();
        config.reference_cells = 4;
        let mut detector = CfarStreamDetector::new(config).unwrap();
        let err = detector.push(f64::NAN).unwrap_err();
        assert!(matches!(err, CfarError::NonFiniteSample { index: 0, .. }));
    }
}
