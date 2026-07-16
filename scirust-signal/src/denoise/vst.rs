//! **Variance-stabilizing transforms (VST) with bias-corrected inverses** — the
//! matched pre/post step that lets the Gaussian denoisers of [`super`] handle
//! *signal-dependent* noise.
//!
//! Most denoisers in this toolkit assume additive, homoscedastic (level-independent)
//! Gaussian noise — the assumption behind the universal wavelet threshold, the Wiener
//! gain, and every least-squares smoother. Real sensors often break it: photon
//! counting is Poisson (`σ² = level`), speckle and gain noise are multiplicative
//! (`σ ∝ level`), and many transducers sit in between. The classical cure is a
//! **variance-stabilizing transform**: a pointwise monotone map `φ` chosen so that
//! `φ(x)` carries (approximately) unit-variance noise regardless of the local level.
//! The pipeline is
//!
//! ```text
//! x → matched φ (VST) → any Gaussian denoiser → bias-corrected φ⁻¹ → x̂
//! ```
//!
//! The **inverse must be bias-corrected**: since `φ` is nonlinear, naively applying the
//! algebraic inverse to the *smoothed* transform commits the retransformation error
//! `E[φ⁻¹(y)] ≠ φ⁻¹(E[y])` (a Jensen-gap bias, proportional to curvature × variance).
//! Two corrections are implemented, each matched to its transform:
//!
//! * **Exact unbiased inverse for the Anscombe transform** — for Poisson data the
//!   mapping `λ ↦ E[2√(X + 3/8)]`, `X ~ Poisson(λ)`, can be inverted exactly;
//!   [`anscombe_inverse_exact_unbiased`] uses the closed-form approximation of
//!   M. Mäkitalo & A. Foi, *"Optimal inversion of the Anscombe transformation in
//!   low-count Poisson image denoising"*, IEEE Trans. Image Processing 20(1):99-109,
//!   2011 (and their closed-form companion note, IEEE TIP 2011).
//! * **Duan smearing for the generic transforms** — N. Duan, *"Smearing estimate: a
//!   nonparametric retransformation method"*, JASA 78(383):605-610, 1983: average the
//!   algebraic inverse over the empirical distribution of the transformed-domain
//!   residuals. It assumes homoscedastic residuals in the transformed domain — exactly
//!   what a matched VST produces.
//!
//! Transforms provided ([`VstKind`]): the Anscombe root (F. J. Anscombe, *"The
//! transformation of Poisson, binomial and negative-binomial data"*, Biometrika
//! 35(3/4):246-254, 1948), the **Generalized Anscombe Transformation** for the
//! mixed Poisson-Gaussian CCD model (F. Murtagh, J.-L. Starck & A. Bijaoui 1995;
//! exact unbiased inverse: M. Mäkitalo & A. Foi, *"Optimal inversion of the
//! generalized Anscombe transformation for Poisson-Gaussian noise"*, IEEE TIP
//! 22(1):91-103, 2013), the Box-Cox power family (G. E. P. Box & D. R. Cox, *"An
//! analysis of transformations"*, JRSS B 26(2):211-252, 1964), and the signed
//! logarithm / signed square root for data that may cross zero.
//!
//! ## Known limitation: fast carriers
//!
//! A VST is pointwise and nonlinear, so it does not commute with the spectrum: a
//! *fast* sinusoidal component riding on the intensity is converted into a harmonic
//! stack in the transformed domain, and the inner denoiser's linear shrinkage then
//! clips those harmonics — a distortion the corrected inverse cannot undo. Measured
//! with the [`super::stft_wiener_auto`] inner denoiser on a 40-cycle/4096-sample
//! Poisson-Gaussian carrier: the sandwich *loses* ≈ 1 dB against the identity
//! pipeline for every calibration probed, while the same calibrations gain +1.4 to
//! +3.0 dB on slow intensity profiles (3 cycles; the acceptance-gate fixtures).
//! The VST pipeline is for slowly varying intensities — photon-flux imaging,
//! sensor drift, envelope-scale structure — not for narrowband carriers near the
//! upper spectrum (pinned by `gat_fast_carrier_regime_is_a_measured_limitation`).
//!
//! ## Conservative selection
//!
//! A *mismatched* VST degrades the result (e.g. a log transform on additive Gaussian
//! noise), so [`detect_noise_model`] is deliberately conservative: it returns a
//! non-identity kind only when the level-vs-scale regression over many windows shows
//! a clear power law, and defaults to [`VstKind::Identity`] on any doubt — short
//! records, non-positive levels, weak correlation, or an exponent outside the
//! Poisson-like / multiplicative bands. [`vst_denoise_auto`] applies the selected
//! transform around [`super::stft_wiener_auto`], and for the identity verdict
//! returns the input unchanged: this entry point answers "does a VST help here?" —
//! plain denoising is the job of [`super::denoise_auto`].
//!
//! ## Domains and clamps (documented, never silent)
//!
//! * **Anscombe** is defined for `x ≥ −3/8`; inputs below are **clamped to −3/8**
//!   (the standard treatment for slightly negative pre-processed counts).
//! * **Box-Cox** is defined for `x > 0`; inputs at or below zero are **clamped to
//!   `1e-12`**, and the inverse clamps `λ·y + 1` at the same floor before taking the
//!   power. A non-finite `λ` degrades to the identity map. Beware `BoxCox(0.0)` (pure
//!   logarithm) on data that can touch zero: the clamp turns a near-zero sample into
//!   a `ln(1e-12) ≈ −27.6` spike that dominates any transformed-domain denoiser —
//!   for such data prefer [`VstKind::SignedLog`], which is bounded through zero.
//! * The signed transforms are defined on all reals; no clamp.
//! * Non-finite samples propagate through the signed transforms; for the clamped
//!   transforms a NaN is absorbed by the clamp (`f64::max` ignores NaN).

use super::{mad, median, stft_wiener_auto};
use core::f64::consts::SQRT_2;

/// Domain offset of the Anscombe transform: `2·√(x + 3/8)`.
const ANSCOMBE_OFFSET: f64 = 0.375;
/// Positivity clamp for the Box-Cox domain (`x > 0`) — documented in the module docs.
const BOXCOX_MIN: f64 = 1.0e-12;
/// Largest residual subsample used by the Duan smearing inverse: above this size the
/// residual set is compressed to this many empirical quantiles so the inverse costs
/// `O(K·n)` instead of `O(n²)`.
const SMEAR_MAX_RESIDUALS: usize = 64;

/// Window length (samples) of the noise-model detector.
const DETECT_WINDOW: usize = 32;
/// Minimum number of complete windows the detector requires (`n ≥ 512`).
const DETECT_MIN_WINDOWS: usize = 16;
/// Cap on the number of windows entering the O(W²) Theil-Sen fit; longer records are
/// deterministically strided down to this many windows.
const DETECT_MAX_WINDOWS: usize = 512;
/// Fraction of windows that must survive the positivity/finiteness filter.
const DETECT_MIN_KEPT: f64 = 0.75;
/// Minimum absolute Pearson correlation of (log level, log scale) to trust the fit.
const DETECT_MIN_CORR: f64 = 0.6;
/// Minimum level dynamic range (`max level / min level`) the detector requires.
/// Calibrated by measurement, not convention: the `vst_protocol` example (P4b)
/// locates the +0.5 dB materiality crossover at ≈ ×3 — at ×2 dynamic range and
/// 30 % multiplicative noise the VST sandwich is a −0.77 dB material *loss*, so
/// firing there would violate the "never degrade" rule the selector exists for.
const DETECT_MIN_RANGE: f64 = 3.0;

/// The variance-stabilizing transform families supported by this module.
///
/// Each kind fixes a pointwise map `φ`, its algebraic inverse, and the
/// bias-corrected inverse used by [`vst_denoise`]. Domains are clamped as
/// documented (module docs), never silently redefined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VstKind {
    /// No transform: `φ(x) = x`. The safe default of [`detect_noise_model`].
    Identity,
    /// Anscombe root `2·√(x + 3/8)` (Anscombe 1948) — stabilizes Poisson noise to
    /// unit variance. Domain `x ≥ −3/8`; smaller inputs are clamped to −3/8.
    Anscombe,
    /// Signed logarithm `sign(x)·ln(1 + |x|)` — stabilizes multiplicative noise
    /// (`σ ∝ level`) while remaining defined (and bounded) through zero.
    SignedLog,
    /// Signed square root `sign(x)·√|x|` — stabilizes `σ ∝ √level` noise of
    /// arbitrary gain (manual-use kind; never returned by the selector).
    SignedSqrt,
    /// Box-Cox power transform `(x^λ − 1)/λ` (Box & Cox 1964), with the `λ = 0`
    /// limit `ln(x)`. Domain `x > 0`; inputs at or below zero are clamped to a tiny
    /// positive floor (manual-use kind; never returned by the selector — see the
    /// module docs for the near-zero hazard of `λ = 0`).
    BoxCox(f64),
    /// Generalized Anscombe Transformation for **mixed Poisson-Gaussian** noise
    /// `x = gain·p + n`, `p ~ Poisson(λ)`, `n ~ N(0, σ²)` — the CCD/CMOS sensor
    /// model (Murtagh, Starck & Bijaoui 1995):
    ///
    /// ```text
    /// φ(x) = (2/gain)·√(gain·x + (3/8)·gain² + σ²)
    /// ```
    ///
    /// which stabilizes the mixture to unit variance; `gain = 1, sigma = 0` reduces
    /// exactly to [`VstKind::Anscombe`]. The bias-corrected inverse is the
    /// closed-form **exact unbiased GAT inverse** (Mäkitalo & Foi, IEEE TIP
    /// 22(1):91-103, 2013), which reuses the Anscombe closed form on the normalized
    /// count scale and subtracts the read-noise term `(σ/gain)²`.
    ///
    /// `gain` (detector conversion gain) and `sigma` (Gaussian read-noise std) are
    /// **calibration inputs** — estimating them from a single record is its own
    /// research problem (Foi et al. 2008) and out of scope, so this kind is
    /// manual-use only and never returned by [`detect_noise_model`]. Degenerate
    /// parameters (`gain ≤ 0`, non-finite, `sigma < 0`) degrade every function of
    /// this module to a pass-through copy. The forward argument
    /// `gain·x + (3/8)·gain² + σ²` is clamped at 0 (same convention as Anscombe's
    /// domain clamp).
    Gat {
        /// Detector conversion gain (`> 0`).
        gain: f64,
        /// Gaussian read-noise standard deviation (`≥ 0`).
        sigma: f64,
    },
}

/// `true` when the GAT parameters are usable; degenerate parameters make every GAT
/// path degrade to a pass-through copy (documented on [`VstKind::Gat`]).
fn gat_params_ok(gain: f64, sigma: f64) -> bool {
    gain.is_finite() && gain > 0.0 && sigma.is_finite() && sigma >= 0.0
}

/// The scalar forward transform `φ` of `kind` — the single source of truth shared
/// by the batch [`vst_forward`] and the causal [`super::streaming::StreamingVst`],
/// so the two agree bit-for-bit on the pointwise map.
///
/// Out-of-domain inputs are clamped as documented on [`VstKind`] — Anscombe at
/// `−3/8`, Box-Cox at `1e-12` — never silently redefined elsewhere. A Box-Cox or
/// GAT with degenerate parameters degrades to the identity map (`φ(x) = x`).
pub(crate) fn forward_scalar(kind: VstKind, x: f64) -> f64 {
    match kind
    {
        VstKind::Identity => x,
        VstKind::Anscombe => 2.0 * (x.max(-ANSCOMBE_OFFSET) + ANSCOMBE_OFFSET).sqrt(),
        VstKind::SignedLog => x.signum() * x.abs().ln_1p(),
        VstKind::SignedSqrt => x.signum() * x.abs().sqrt(),
        VstKind::BoxCox(lambda) =>
        {
            if !lambda.is_finite()
            {
                return x;
            }
            let xc = x.max(BOXCOX_MIN);
            if lambda == 0.0
            {
                xc.ln()
            }
            else
            {
                (xc.powf(lambda) - 1.0) / lambda
            }
        },
        VstKind::Gat { gain, sigma } =>
        {
            if !gat_params_ok(gain, sigma)
            {
                return x;
            }
            let offset = ANSCOMBE_OFFSET * gain * gain + sigma * sigma;
            2.0 / gain * (gain * x + offset).max(0.0).sqrt()
        },
    }
}

/// Apply the forward transform `φ` of `kind` pointwise.
///
/// Out-of-domain inputs are clamped as documented on [`VstKind`] — Anscombe at
/// `−3/8`, Box-Cox at `1e-12` — never silently redefined elsewhere. A Box-Cox with
/// non-finite `λ` degrades to a copy of the input.
pub fn vst_forward(kind: VstKind, signal: &[f64]) -> Vec<f64> {
    signal.iter().map(|&x| forward_scalar(kind, x)).collect()
}

/// The scalar algebraic inverse `φ⁻¹` of `kind` — shared by [`vst_inverse_naive`],
/// the smearing estimator of [`vst_inverse_corrected`], and the causal
/// [`super::streaming::StreamingVst`].
pub(crate) fn inverse_naive_scalar(kind: VstKind, y: f64) -> f64 {
    match kind
    {
        VstKind::Identity => y,
        VstKind::Anscombe =>
        {
            let h = 0.5 * y;
            h * h - ANSCOMBE_OFFSET
        },
        VstKind::SignedLog => y.signum() * y.abs().exp_m1(),
        VstKind::SignedSqrt => y.signum() * (y * y),
        VstKind::BoxCox(lambda) =>
        {
            if !lambda.is_finite()
            {
                return y;
            }
            if lambda == 0.0
            {
                y.exp()
            }
            else
            {
                (lambda * y + 1.0).max(BOXCOX_MIN).powf(1.0 / lambda)
            }
        },
        VstKind::Gat { gain, sigma } =>
        {
            if !gat_params_ok(gain, sigma)
            {
                return y;
            }
            let h = 0.5 * gain * y;
            (h * h - ANSCOMBE_OFFSET * gain * gain - sigma * sigma) / gain
        },
    }
}

/// Apply the **algebraic** (naive) inverse `φ⁻¹` of `kind` pointwise.
///
/// This inverse is *biased* when applied to a smoothed transform — the Jensen-gap
/// retransformation error `E[φ⁻¹(y)] ≠ φ⁻¹(E[y])` (Duan 1983; Mäkitalo & Foi 2011).
/// It is kept public for comparison and testing; production pipelines should use
/// [`vst_inverse_corrected`].
pub fn vst_inverse_naive(kind: VstKind, transformed: &[f64]) -> Vec<f64> {
    transformed
        .iter()
        .map(|&y| inverse_naive_scalar(kind, y))
        .collect()
}

/// Exact unbiased inverse of the Anscombe transformation — the closed-form
/// approximation of Mäkitalo & Foi (IEEE TIP 20(1):99-109, 2011):
///
/// ```text
/// x̂(y) = y²/4 + √(3/2)/(4y) − 11/(8y²) + 5√(3/2)/(8y³) − 1/8      for y ≥ 2√(3/8)
/// ```
///
/// and `x̂(y) = 0` below `2√(3/8)` — the exact unbiased inverse maps the transform of
/// zero counts to 0, and the closed form is exactly 0 at the threshold, so the map is
/// continuous and monotone non-decreasing. The result is clamped at 0 to keep the
/// non-negativity of the Poisson mean under floating-point rounding.
pub fn anscombe_inverse_exact_unbiased(y: f64) -> f64 {
    let sqrt_3_2 = 1.5f64.sqrt();
    let threshold = 2.0 * ANSCOMBE_OFFSET.sqrt(); // 2·√(3/8) = √(3/2)
    if y.is_nan() || y < threshold
    {
        return 0.0;
    }
    let y2 = y * y;
    let y3 = y2 * y;
    (0.25 * y2 + 0.25 * sqrt_3_2 / y - 1.375 / y2 + 0.625 * sqrt_3_2 / y3 - 0.125).max(0.0)
}

/// The scalar **bias-corrected inverse for the pointwise kinds** — Identity,
/// Anscombe (exact unbiased inverse), and GAT (exact unbiased GAT inverse) — which
/// need no residual set. Shared by the batch [`vst_inverse_corrected`] and the
/// causal [`super::streaming::StreamingVst`] so the two agree bit-for-bit. The
/// smearing kinds (signed log / signed sqrt / Box-Cox) are *not* pointwise — they
/// need the residual distribution — and fall back here to the naive inverse only as
/// a degenerate guard; callers handle them with a smearing average instead.
pub(crate) fn inverse_corrected_pointwise_scalar(kind: VstKind, y: f64) -> f64 {
    match kind
    {
        VstKind::Identity => y,
        VstKind::Anscombe => anscombe_inverse_exact_unbiased(y),
        VstKind::Gat { gain, sigma } =>
        {
            if !gat_params_ok(gain, sigma)
            {
                return y;
            }
            let read_noise = (sigma / gain) * (sigma / gain);
            gain * (anscombe_inverse_exact_unbiased(y) - read_noise).max(0.0)
        },
        VstKind::SignedLog | VstKind::SignedSqrt | VstKind::BoxCox(_) =>
        {
            inverse_naive_scalar(kind, y)
        },
    }
}

/// Compress the residual set to at most [`SMEAR_MAX_RESIDUALS`] values: non-finite
/// residuals are dropped; a larger set is sorted (with `total_cmp`, so a stray NaN
/// cannot panic the sort) and represented by evenly spaced midpoint order statistics
/// (empirical quantiles), keeping the smearing average deterministic and `O(K·n)`.
fn smearing_sample(residuals: &[f64]) -> Vec<f64> {
    let mut finite: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();
    if finite.len() <= SMEAR_MAX_RESIDUALS
    {
        return finite;
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let m = finite.len() as f64;
    (0..SMEAR_MAX_RESIDUALS)
        .map(|k| finite[((k as f64 + 0.5) * m / SMEAR_MAX_RESIDUALS as f64) as usize])
        .collect()
}

/// Apply the **bias-corrected** inverse of `kind` to a smoothed transform.
///
/// * [`VstKind::Identity`] — pass-through (a linear map has no retransformation bias;
///   `residuals` are ignored).
/// * [`VstKind::Anscombe`] — the closed-form **exact unbiased inverse**
///   ([`anscombe_inverse_exact_unbiased`]) applied pointwise. `residuals` are
///   deliberately **ignored**: the exact inverse is derived from the Poisson model
///   itself, which strictly dominates a nonparametric correction when the model
///   matches — smearing would re-introduce estimation noise (and low-count Anscombe
///   residuals are only approximately homoscedastic, weakening its premise).
/// * [`VstKind::Gat`] — the closed-form **exact unbiased GAT inverse** (Mäkitalo &
///   Foi 2013): on the normalized count scale `z = x/gain` the GAT is the Anscombe
///   transform of `z` with an extra `(σ/gain)²` inside the root, so
///   `ẑ = A⁻¹(y) − (σ/gain)²` with `A⁻¹` the Anscombe closed form, clamped at 0,
///   then `x̂ = gain·ẑ`. `residuals` are ignored for the same reason as Anscombe;
///   degenerate parameters degrade to a pass-through copy.
/// * [`VstKind::SignedLog`] / [`VstKind::SignedSqrt`] / [`VstKind::BoxCox`] — the
///   **Duan (1983) smearing estimate**: `x̂ᵢ = mean_j φ⁻¹(fᵢ + rⱼ)` over the
///   transformed-domain residual set, the standard nonparametric fix for the
///   retransformation bias `E[φ⁻¹(f + ε)] ≠ φ⁻¹(f)`. It assumes homoscedastic
///   residuals in the transformed domain — exactly what a matched VST produces.
///   Non-finite residuals are dropped; if none remain the naive inverse is the
///   fallback. Residual sets larger than 64 are compressed to 64 empirical quantiles
///   (deterministically), bounding the cost at `O(64·n)`.
pub fn vst_inverse_corrected(kind: VstKind, filtered: &[f64], residuals: &[f64]) -> Vec<f64> {
    match kind
    {
        VstKind::Identity | VstKind::Anscombe | VstKind::Gat { .. } => filtered
            .iter()
            .map(|&y| inverse_corrected_pointwise_scalar(kind, y))
            .collect(),
        VstKind::SignedLog | VstKind::SignedSqrt | VstKind::BoxCox(_) =>
        {
            let sample = smearing_sample(residuals);
            if sample.is_empty()
            {
                return vst_inverse_naive(kind, filtered);
            }
            let inv_k = 1.0 / sample.len() as f64;
            filtered
                .iter()
                .map(|&f| {
                    let sum: f64 = sample
                        .iter()
                        .map(|&r| inverse_naive_scalar(kind, f + r))
                        .sum();
                    sum * inv_k
                })
                .collect()
        },
    }
}

/// Median of pairwise slopes (Theil-Sen estimator) of `ys` against `xs`. Pairs with
/// (near-)equal abscissae or a non-finite slope are skipped; `None` when no valid
/// pair remains.
fn theil_sen_slope(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let mut slopes = Vec::new();
    for i in 0..xs.len()
    {
        for j in (i + 1)..xs.len()
        {
            let dx = xs[j] - xs[i];
            if dx.abs() > 1.0e-12
            {
                let s = (ys[j] - ys[i]) / dx;
                if s.is_finite()
                {
                    slopes.push(s);
                }
            }
        }
    }
    if slopes.is_empty()
    {
        None
    }
    else
    {
        Some(median(&slopes))
    }
}

/// Pearson correlation of two equally long samples; `None` when either variance is
/// degenerate (constant series) or fewer than two points remain.
fn pearson_r(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() < 2 || xs.len() != ys.len()
    {
        return None;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys)
    {
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
        sxy += (x - mx) * (y - my);
    }
    if sxx <= 0.0 || syy <= 0.0
    {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// **Conservative** noise-model selector: identify a signal-dependent noise law from
/// the level-vs-scale relationship across windows, defaulting to
/// [`VstKind::Identity`] on any doubt.
///
/// A wrong match *degrades* the result (a log transform on additive noise loses
/// accuracy — see the module docs), so every gate below fails toward Identity:
///
/// 1. Split the signal into non-overlapping 32-sample windows; require at least 16
///    windows (`n ≥ 512`). Very long records are strided down to at most 512 windows
///    (deterministically) to bound the O(W²) fit.
/// 2. Per window: level `m = median`, scale `s = MAD(first differences)/(0.6745·√2)`
///    — first differences cancel the local trend, and the MAD is immune to a minority
///    of outliers; the √2 undoes the variance doubling of differencing.
/// 3. Keep windows with finite `m > 0` and `s > 0`; require ≥ 75 % kept **and** a
///    level dynamic range `max(m)/min(m) ≥ 3` — without level variation no law is
///    identifiable, and the threshold is *measured*, not conventional: the
///    `vst_protocol` sweep (P4b) puts the +0.5 dB materiality crossover at ≈ ×3,
///    with ×2 being a material −0.77 dB loss at 30 % multiplicative noise.
/// 4. Fit `log s = α + β·log m` with the robust Theil-Sen slope (median of pairwise
///    slopes, sorted with `total_cmp`); require Pearson `|r| ≥ 0.6` on the log-log
///    points.
/// 5. Map the exponent: `β ∈ [0.3, 0.7]` → [`VstKind::Anscombe`] (Poisson-like,
///    `σ ≈ √level`) — *additionally* requiring the fitted gain `exp(α)` to lie in
///    `[0.5, 2]`, because the exact unbiased inverse assumes the *unit-gain* Poisson
///    law `σ = √level`; a √level shape with a very different gain (e.g. scaled
///    Gaussian read noise) would be over-corrected. `β ∈ [0.75, 1.25]` →
///    [`VstKind::SignedLog`] (multiplicative; the smearing inverse is nonparametric,
///    so any gain is acceptable). Anything else → Identity.
///
/// [`VstKind::SignedSqrt`], [`VstKind::BoxCox`] and [`VstKind::Gat`] are manual-use
/// kinds and are never returned by this selector (the GAT additionally needs the
/// detector's calibration parameters, which no single record identifies reliably).
pub fn detect_noise_model(signal: &[f64]) -> VstKind {
    let n_windows = signal.len() / DETECT_WINDOW;
    if n_windows < DETECT_MIN_WINDOWS
    {
        return VstKind::Identity;
    }
    let stride = n_windows.div_ceil(DETECT_MAX_WINDOWS);
    let mut used = 0usize;
    let mut levels = Vec::new(); // log m per kept window
    let mut scales = Vec::new(); // log s per kept window
    for (w, win) in signal.as_chunks::<DETECT_WINDOW>().0.iter().enumerate()
    {
        if w % stride != 0
        {
            continue;
        }
        used += 1;
        let m = median(win);
        let diffs: Vec<f64> = win.windows(2).map(|p| p[1] - p[0]).collect();
        let s = mad(&diffs) / (0.6745 * SQRT_2);
        if m.is_finite() && s.is_finite() && m > 0.0 && s > 0.0
        {
            levels.push(m.ln());
            scales.push(s.ln());
        }
    }
    if (levels.len() as f64) < DETECT_MIN_KEPT * used as f64
    {
        return VstKind::Identity;
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &l in &levels
    {
        lo = lo.min(l);
        hi = hi.max(l);
    }
    if hi - lo < DETECT_MIN_RANGE.ln()
    {
        return VstKind::Identity;
    }
    let Some(beta) = theil_sen_slope(&levels, &scales)
    else
    {
        return VstKind::Identity;
    };
    let Some(r) = pearson_r(&levels, &scales)
    else
    {
        return VstKind::Identity;
    };
    if r.abs() < DETECT_MIN_CORR
    {
        return VstKind::Identity;
    }
    if (0.3..=0.7).contains(&beta)
    {
        // Poisson-like exponent: additionally require a near-unit gain, since the
        // exact unbiased inverse is calibrated to σ = √level exactly.
        let intercepts: Vec<f64> = levels
            .iter()
            .zip(&scales)
            .map(|(&lm, &ls)| ls - beta * lm)
            .collect();
        let gain = median(&intercepts).exp();
        if (0.5..=2.0).contains(&gain)
        {
            return VstKind::Anscombe;
        }
        return VstKind::Identity;
    }
    if (0.75..=1.25).contains(&beta)
    {
        return VstKind::SignedLog;
    }
    VstKind::Identity
}

/// Run the full VST pipeline: `φ` forward, the supplied Gaussian `denoiser` on the
/// transformed signal, then the **bias-corrected** inverse
/// ([`vst_inverse_corrected`]: exact unbiased for Anscombe, Duan smearing for the
/// other transforms, pass-through for Identity), with the transformed-domain
/// residuals `t − denoiser(t)` feeding the smearing correction.
///
/// Graceful degradation: an empty signal returns empty; a signal without a single
/// finite sample is returned unchanged (there is nothing to denoise); a denoiser
/// that returns a different length than its input returns the input unchanged.
pub fn vst_denoise(
    signal: &[f64],
    kind: VstKind,
    denoiser: impl Fn(&[f64]) -> Vec<f64>,
) -> Vec<f64> {
    if signal.is_empty()
    {
        return Vec::new();
    }
    if !signal.iter().any(|v| v.is_finite())
    {
        return signal.to_vec();
    }
    let transformed = vst_forward(kind, signal);
    let filtered = denoiser(&transformed);
    if filtered.len() != signal.len()
    {
        return signal.to_vec();
    }
    let residuals: Vec<f64> = transformed
        .iter()
        .zip(&filtered)
        .map(|(t, f)| t - f)
        .collect();
    vst_inverse_corrected(kind, &filtered, &residuals)
}

/// Result of [`vst_denoise_auto`].
#[derive(Debug, Clone)]
pub struct VstAutoResult {
    /// The transform the selector chose.
    pub kind: VstKind,
    /// Human-readable pipeline description, e.g.
    /// `"anscombe ∘ stft_wiener_auto ∘ exact_unbiased_inverse"`.
    pub method: String,
    /// The output signal (same length as the input).
    pub output: Vec<f64>,
}

/// Detect the noise model and, when it is signal-dependent, run the matched VST
/// around [`super::stft_wiener_auto`] with the bias-corrected inverse.
///
/// ## Why `stft_wiener_auto` as the inner Gaussian denoiser
///
/// The short-time Wiener filter tracks a noise *floor* under the assumption that
/// noise power is independent of the signal — precisely the assumption
/// signal-dependent noise violates (where the signal is loud, so is its noise, and
/// the tracker cannot separate the two) and precisely the one the VST restores.
/// Measured on this crate's Poisson and multiplicative strong-regime fixtures it is
/// both the strongest identity-domain baseline of the toolkit *and* the largest
/// beneficiary of stabilization (+2.5 to +5 dB — the acceptance tests below pin
/// this down). The classic literature beneficiary, VisuShrink wavelet
/// thresholding, is *not* used here: its raw-domain MAD calibration lands on a
/// mid-range σ, which on level-correlated signals acts as an accidental
/// level-adaptive threshold and measured *better* un-stabilized than stabilized —
/// an honest negative result, consistent with the conservative philosophy of this
/// module. Callers who want a different inner denoiser use [`vst_denoise`] directly.
///
/// For an [`VstKind::Identity`] verdict the input is returned **unchanged**: this
/// entry point answers "does a VST help here?", and plain denoising is the job of
/// the existing [`super::denoise_auto`] — wiring the VST as a conditional pre/post
/// step of that selector is done there, not here.
pub fn vst_denoise_auto(signal: &[f64]) -> VstAutoResult {
    let kind = detect_noise_model(signal);
    if kind == VstKind::Identity
    {
        return VstAutoResult {
            kind,
            method: "identity (no signal-dependent noise detected; no VST applied)".into(),
            output: signal.to_vec(),
        };
    }
    let (label, inverse) = match kind
    {
        VstKind::Anscombe => ("anscombe".to_string(), "exact_unbiased_inverse"),
        VstKind::SignedLog => ("signed_log".to_string(), "smearing_inverse"),
        VstKind::SignedSqrt => ("signed_sqrt".to_string(), "smearing_inverse"),
        VstKind::BoxCox(lambda) => (format!("box_cox({lambda})"), "smearing_inverse"),
        VstKind::Gat { gain, sigma } => (
            format!("gat(gain={gain}, sigma={sigma})"),
            "exact_unbiased_inverse",
        ),
        VstKind::Identity => unreachable!("handled above"),
    };
    let output = vst_denoise(signal, kind, stft_wiener_auto);
    VstAutoResult {
        kind,
        method: format!("{label} ∘ stft_wiener_auto ∘ {inverse}"),
        output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denoise::testutil::{Lcg, snr_db};
    use crate::denoise::{ThresholdMode, moving_average, wavelet_denoise};
    use core::f64::consts::PI;

    /// Poisson sampler — Knuth's product-of-uniforms algorithm on the shared
    /// deterministic LCG; adequate for `λ ≲ 30` (all fixtures stay below 13).
    fn poisson(rng: &mut Lcg, lambda: f64) -> f64 {
        if lambda <= 0.0
        {
            return 0.0;
        }
        let l = (-lambda).exp();
        let mut k: u64 = 0;
        let mut p = 1.0;
        loop
        {
            k += 1;
            p *= rng.uniform();
            if p <= l
            {
                break;
            }
        }
        (k - 1) as f64
    }

    /// Low-count Poisson intensity: `λᵢ = 6 + 5·sin(2π·3·i/n) + i/n`, so `λ ∈ [1, ~12]`.
    fn poisson_intensity(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 6.0 + 5.0 * (2.0 * PI * 3.0 * i as f64 / n as f64).sin() + i as f64 / n as f64)
            .collect()
    }

    /// `(clean intensity, Poisson counts)` — the strong-regime fixture of the
    /// acceptance gate (research report §12, Phase 1).
    fn poisson_fixture(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let clean = poisson_intensity(n);
        let mut rng = Lcg::new(seed);
        let noisy = clean.iter().map(|&l| poisson(&mut rng, l)).collect();
        (clean, noisy)
    }

    /// `(clean, noisy)` multiplicative *soft-regime* fixture: levels in `[2, 4.5]`
    /// (×2.25 — beneath the selector's measured ×3 materiality gate, where the
    /// VST is a small loss; the selector must stay Identity here), `x = s·(1 + 0.3·g)`.
    fn multiplicative_fixture(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let clean: Vec<f64> = (0..n)
            .map(|i| 3.0 + (2.0 * PI * 3.0 * i as f64 / n as f64).sin() + 0.5 * i as f64 / n as f64)
            .collect();
        let mut rng = Lcg::new(seed);
        let noisy = clean
            .iter()
            .map(|&s| s * (1.0 + 0.3 * rng.gauss()))
            .collect();
        (clean, noisy)
    }

    /// `(clean, noisy)` multiplicative *gate* fixture: same 30 % multiplicative
    /// noise, but levels in `[4, 44]` (×10 dynamic range — a genuinely strong
    /// regime, where the raw-domain σ spans more than a decade).
    fn strong_multiplicative_fixture(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let clean: Vec<f64> = (0..n)
            .map(|i| {
                22.0 + 18.0 * (2.0 * PI * 3.0 * i as f64 / n as f64).sin()
                    + 4.0 * i as f64 / n as f64
            })
            .collect();
        let mut rng = Lcg::new(seed);
        let noisy = clean
            .iter()
            .map(|&s| s * (1.0 + 0.3 * rng.gauss()))
            .collect();
        (clean, noisy)
    }

    fn assert_close(a: f64, b: f64) {
        let tol = 1.0e-12 * a.abs().max(1.0);
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn naive_inverse_round_trips_every_kind() {
        // Anscombe: in-domain means x ≥ −3/8.
        let xa = [-0.375, -0.2, 0.0, 0.5, 1.0, 4.0, 12.5, 100.0];
        for (&x, &y) in xa.iter().zip(&vst_inverse_naive(
            VstKind::Anscombe,
            &vst_forward(VstKind::Anscombe, &xa),
        ))
        {
            assert_close(x, y);
        }
        // The signed transforms and the identity are defined on all reals.
        let xs = [-50.0, -3.2, -1.0, -0.1, 0.0, 0.7, 2.0, 10.0];
        for kind in [VstKind::Identity, VstKind::SignedLog, VstKind::SignedSqrt]
        {
            for (&x, &y) in xs
                .iter()
                .zip(&vst_inverse_naive(kind, &vst_forward(kind, &xs)))
            {
                assert_close(x, y);
            }
        }
        // Box-Cox: x > 0, λ ∈ {0, 1/2}.
        let xp = [1.0e-6, 0.1, 1.0, 2.5, 10.0, 1.0e4];
        for lambda in [0.0, 0.5]
        {
            let kind = VstKind::BoxCox(lambda);
            for (&x, &y) in xp
                .iter()
                .zip(&vst_inverse_naive(kind, &vst_forward(kind, &xp)))
            {
                assert_close(x, y);
            }
        }
    }

    #[test]
    fn exact_unbiased_inverse_is_monotone_and_nonnegative() {
        let mut prev = anscombe_inverse_exact_unbiased(0.0);
        assert!(prev >= 0.0);
        let mut k = 1u32;
        while f64::from(k) * 1.0e-3 <= 20.0
        {
            let y = f64::from(k) * 1.0e-3;
            let v = anscombe_inverse_exact_unbiased(y);
            assert!(v >= 0.0, "negative at y = {y}");
            assert!(
                v + 1.0e-12 >= prev,
                "not monotone at y = {y}: {prev} -> {v}"
            );
            prev = v;
            k += 1;
        }
    }

    #[test]
    fn poisson_bias_oracle_flat_intensity() {
        // Flat λ = 4: a strong smoother on the Anscombe domain approximates
        // E[2√(X + 3/8)]; the exact unbiased inverse must recover 4, the naive
        // algebraic inverse must land visibly below it (Jensen gap ≈ 0.25 here).
        let n = 8192;
        let mut rng = Lcg::new(42);
        let noisy: Vec<f64> = (0..n).map(|_| poisson(&mut rng, 4.0)).collect();
        let t = vst_forward(VstKind::Anscombe, &noisy);
        let smooth = moving_average(&t, 65);
        let residuals: Vec<f64> = t.iter().zip(&smooth).map(|(a, b)| a - b).collect();
        let naive = vst_inverse_naive(VstKind::Anscombe, &smooth);
        let unbiased = vst_inverse_corrected(VstKind::Anscombe, &smooth, &residuals);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let bias_naive = (mean(&naive) - 4.0).abs();
        let bias_unbiased = (mean(&unbiased) - 4.0).abs();
        assert!(
            bias_unbiased <= 0.05,
            "exact unbiased inverse off by {bias_unbiased:.4} (mean {:.4})",
            mean(&unbiased)
        );
        assert!(
            bias_naive > bias_unbiased + 0.05,
            "naive bias {bias_naive:.4} not visibly worse than corrected {bias_unbiased:.4}"
        );
    }

    #[test]
    fn poisson_acceptance_gate_strong_regime() {
        // Report §12 Phase 1 acceptance criterion: on low-count Poisson (strong
        // signal dependence, λ ∈ [1, ~12]) the corrected VST pipeline must beat the
        // identity pipeline by ≥ 1 dB, and the corrected inverse must not trail the
        // naive one. Both arms run stft_wiener_auto — the toolkit's strongest
        // homoscedasticity-assuming baseline (see the `vst_denoise_auto` docs for
        // why VisuShrink is not the comparison denoiser: its raw-domain MAD
        // calibration is accidentally level-adaptive and never benefits from
        // stabilization on this fixture — measured mean −1.0 dB over five seeds).
        let n = 4096;
        let (clean, noisy) = poisson_fixture(n, 7);
        let identity_out = stft_wiener_auto(&noisy);
        let vst_out = vst_denoise(&noisy, VstKind::Anscombe, stft_wiener_auto);
        // Naive-inverse variant built inline for comparison.
        let t = vst_forward(VstKind::Anscombe, &noisy);
        let d = stft_wiener_auto(&t);
        let naive_out = vst_inverse_naive(VstKind::Anscombe, &d);
        let s_identity = snr_db(&clean, &identity_out);
        let s_vst = snr_db(&clean, &vst_out);
        let s_naive = snr_db(&clean, &naive_out);
        assert!(
            s_vst >= s_identity + 1.0,
            "acceptance gate failed: VST {s_vst:.2} dB vs identity {s_identity:.2} dB"
        );
        assert!(
            s_vst >= s_naive,
            "corrected inverse {s_vst:.2} dB trails naive inverse {s_naive:.2} dB"
        );
    }

    #[test]
    fn multiplicative_gate_strong_regime() {
        // 30 % multiplicative noise over a ×10 level range: signed-log VST with the
        // smearing inverse must beat the identity pipeline by ≥ 1 dB. (On the ×2.25
        // selector fixture the gain is ≈ 0 — the report's "soft regime, gain
        // declared null" — so the gate runs where the regime is actually strong.)
        let n = 4096;
        let (clean, noisy) = strong_multiplicative_fixture(n, 9);
        let identity_out = stft_wiener_auto(&noisy);
        let vst_out = vst_denoise(&noisy, VstKind::SignedLog, stft_wiener_auto);
        let s_identity = snr_db(&clean, &identity_out);
        let s_vst = snr_db(&clean, &vst_out);
        assert!(
            s_vst >= s_identity + 1.0,
            "multiplicative gate failed: VST {s_vst:.2} dB vs identity {s_identity:.2} dB"
        );
    }

    #[test]
    fn soft_regime_vst_does_not_lose() {
        // Mild dependence (σ = 0.1·√level on levels 2-4): the report (§10) declares
        // any gain < 0.5 dB null in this regime — the honesty requirement is that
        // the matched VST (signed square root: exactly stabilizes σ ∝ √level of any
        // gain, with the nonparametric smearing inverse) does not LOSE more than
        // 0.5 dB against the identity pipeline. (Anscombe is deliberately not used:
        // its exact unbiased inverse assumes the unit-gain Poisson law and would
        // over-correct 0.1·√s Gaussian noise by ≈ +0.25.)
        let n = 4096;
        let clean: Vec<f64> = (0..n)
            .map(|i| 3.0 + (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let mut rng = Lcg::new(11);
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&s| s + 0.1 * s.sqrt() * rng.gauss())
            .collect();
        let identity_out = stft_wiener_auto(&noisy);
        let vst_out = vst_denoise(&noisy, VstKind::SignedSqrt, stft_wiener_auto);
        let s_identity = snr_db(&clean, &identity_out);
        let s_vst = snr_db(&clean, &vst_out);
        assert!(
            s_vst >= s_identity - 0.5,
            "soft regime: VST {s_vst:.2} dB lost more than 0.5 dB vs identity {s_identity:.2} dB"
        );
    }

    #[test]
    fn selector_is_conservative() {
        // (a) Additive Gaussian on a zero-mean sine: many window medians are ≤ 0
        // and the scale does not track the level → Identity.
        let n = 4096;
        let mut rng = Lcg::new(5);
        let additive: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin() + 0.2 * rng.gauss())
            .collect();
        assert_eq!(detect_noise_model(&additive), VstKind::Identity);
        let auto = vst_denoise_auto(&additive);
        assert_eq!(auto.kind, VstKind::Identity);
        assert_eq!(
            auto.output, additive,
            "identity verdict must return the input unchanged"
        );
        assert!(auto.method.contains("identity"), "method: {}", auto.method);

        // (b) The Poisson acceptance fixture → Anscombe.
        let (_, noisy_poisson) = poisson_fixture(4096, 7);
        assert_eq!(detect_noise_model(&noisy_poisson), VstKind::Anscombe);

        // (c) The strong multiplicative fixture (×10 levels) → SignedLog; the SOFT
        // one (×2.25) must now stay Identity: the vst_protocol P4b sweep measured
        // the +0.5 dB materiality crossover at ≈ ×3 dynamic range (×2 is a
        // −0.77 dB material loss), so the DETECT_MIN_RANGE gate deliberately
        // excludes it — firing there would violate "never degrade".
        let (_, noisy_mult) = multiplicative_fixture(4096, 9);
        assert_eq!(detect_noise_model(&noisy_mult), VstKind::Identity);
        let (_, noisy_strong) = strong_multiplicative_fixture(4096, 9);
        assert_eq!(detect_noise_model(&noisy_strong), VstKind::SignedLog);

        // (d) Too short (< 512 samples) → Identity, even on perfect Poisson data.
        assert_eq!(detect_noise_model(&noisy_poisson[..256]), VstKind::Identity);

        // (e) Constant signal: zero scale in every window → Identity.
        assert_eq!(detect_noise_model(&[5.0; 1024]), VstKind::Identity);
    }

    #[test]
    fn graceful_on_degenerate_inputs() {
        let kinds = [
            VstKind::Identity,
            VstKind::Anscombe,
            VstKind::SignedLog,
            VstKind::SignedSqrt,
            VstKind::BoxCox(0.5),
        ];
        // Empty input.
        let empty: [f64; 0] = [];
        for kind in kinds
        {
            assert!(vst_forward(kind, &empty).is_empty());
            assert!(vst_inverse_naive(kind, &empty).is_empty());
            assert!(vst_denoise(&empty, kind, |x| x.to_vec()).is_empty());
        }
        assert_eq!(detect_noise_model(&empty), VstKind::Identity);
        assert!(vst_denoise_auto(&empty).output.is_empty());
        // Very short inputs.
        for len in 1..=3usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64).collect();
            for kind in kinds
            {
                let out = vst_denoise(&x, kind, |v| wavelet_denoise(v, 0, ThresholdMode::Soft));
                assert_eq!(out.len(), len);
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "{kind:?} len {len}: {out:?}"
                );
            }
        }
        // All-NaN input: echoed back (nothing to denoise), never a panic.
        let nans = [f64::NAN; 64];
        for kind in kinds
        {
            let out = vst_denoise(&nans, kind, |v| wavelet_denoise(v, 0, ThresholdMode::Soft));
            assert_eq!(out.len(), nans.len());
            assert!(
                out.iter().all(|v| v.is_nan()),
                "{kind:?} must echo the NaN input"
            );
        }
        assert_eq!(detect_noise_model(&nans), VstKind::Identity);
        let auto = vst_denoise_auto(&nans);
        assert_eq!(auto.kind, VstKind::Identity);
        assert_eq!(auto.output.len(), nans.len());
        // A length-changing denoiser degrades to the input.
        let x: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let out = vst_denoise(&x, VstKind::Anscombe, |_| vec![0.0; 3]);
        assert_eq!(out, x);
    }

    #[test]
    fn auto_is_deterministic() {
        let (_, noisy) = poisson_fixture(4096, 7);
        let a = vst_denoise_auto(&noisy);
        let b = vst_denoise_auto(&noisy);
        assert_eq!(
            a.kind,
            VstKind::Anscombe,
            "fixture must exercise the non-identity path"
        );
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.method, b.method);
        assert_eq!(
            a.output, b.output,
            "vst_denoise_auto must be bit-for-bit deterministic"
        );
    }

    /// `(clean mean gain·λ, noisy)` mixed Poisson-Gaussian fixture: the CCD model
    /// `x = gain·p + n`, `p ~ Poisson(λᵢ)`, `n ~ N(0, σ²)`, on the same slow
    /// low-count intensity profile as the validated Anscombe gate
    /// ([`poisson_intensity`], 3 cycles + ramp). A *fast* carrier is deliberately
    /// not used here — see [`gat_fast_carrier_regime_is_a_measured_limitation`].
    fn gat_fixture(n: usize, gain: f64, sigma: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let lambda = poisson_intensity(n);
        let mut rng = Lcg::new(seed);
        let noisy: Vec<f64> = lambda
            .iter()
            .map(|&l| gain * poisson(&mut rng, l) + sigma * rng.gauss())
            .collect();
        let clean = lambda.iter().map(|&l| gain * l).collect();
        (clean, noisy)
    }

    #[test]
    fn gat_reduces_to_anscombe_at_unit_gain_zero_sigma() {
        let x = [-0.4, 0.0, 0.5, 1.0, 4.0, 13.0, 250.0];
        let gat = VstKind::Gat {
            gain: 1.0,
            sigma: 0.0,
        };
        assert_eq!(vst_forward(gat, &x), vst_forward(VstKind::Anscombe, &x));
        let y = vst_forward(gat, &x);
        assert_eq!(
            vst_inverse_corrected(gat, &y, &[]),
            vst_inverse_corrected(VstKind::Anscombe, &y, &[])
        );
    }

    #[test]
    fn gat_naive_inverse_round_trips() {
        let gat = VstKind::Gat {
            gain: 1.4,
            sigma: 1.2,
        };
        // In-domain: gain·x + 3/8·gain² + σ² ≥ 0 ⇔ x ≥ −(3/8·gain² + σ²)/gain.
        let x = [-1.5, -0.5, 0.0, 0.7, 3.0, 12.0, 400.0];
        for (&a, &b) in x.iter().zip(&vst_inverse_naive(gat, &vst_forward(gat, &x)))
        {
            assert_close(a, b);
        }
    }

    #[test]
    fn gat_stabilizes_mixed_poisson_gaussian_noise() {
        // Per-level σ of the transformed mixture must be ≈ 1 across a ×6 level
        // range — the defining property of the GAT.
        let (gain, sigma) = (1.4, 1.2);
        let gat = VstKind::Gat { gain, sigma };
        let mut rng = Lcg::new(31);
        for lambda in [2.0, 5.0, 12.0]
        {
            let x: Vec<f64> = (0..20000)
                .map(|_| gain * poisson(&mut rng, lambda) + sigma * rng.gauss())
                .collect();
            let t = vst_forward(gat, &x);
            let mean = t.iter().sum::<f64>() / t.len() as f64;
            let var = t.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / t.len() as f64;
            let sd = var.sqrt();
            assert!(
                (0.85..=1.15).contains(&sd),
                "transformed σ at λ = {lambda} is {sd:.3}, not ≈ 1"
            );
        }
    }

    #[test]
    fn gat_bias_oracle_flat_intensity() {
        // Flat λ = 4 through the CCD model: the exact unbiased GAT inverse must
        // recover the clean mean gain·λ; the naive inverse must be visibly worse.
        let (gain, sigma) = (1.5, 1.0);
        let gat = VstKind::Gat { gain, sigma };
        let n = 8192;
        let mut rng = Lcg::new(42);
        let noisy: Vec<f64> = (0..n)
            .map(|_| gain * poisson(&mut rng, 4.0) + sigma * rng.gauss())
            .collect();
        let t = vst_forward(gat, &noisy);
        let smooth = moving_average(&t, 65);
        let corrected = vst_inverse_corrected(gat, &smooth, &[]);
        let naive = vst_inverse_naive(gat, &smooth);
        let target = gain * 4.0;
        let mean_of = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let bias_corrected = (mean_of(&corrected) - target).abs();
        let bias_naive = (mean_of(&naive) - target).abs();
        assert!(
            bias_corrected < 0.1,
            "corrected GAT inverse bias {bias_corrected:.3} exceeds 0.1"
        );
        assert!(
            bias_naive > bias_corrected + 0.05,
            "naive bias {bias_naive:.3} should visibly exceed corrected {bias_corrected:.3}"
        );
    }

    #[test]
    fn gat_acceptance_gate_mixed_noise() {
        // Same acceptance philosophy as the Anscombe gate (report §12 Phase 1),
        // on the mixed Poisson-Gaussian model: the GAT sandwich around the same
        // inner denoiser must beat the identity pipeline by ≥ 1 dB. (1.3, 1.5) is
        // the harshest mix probed — the read noise nearly drowns the low-count
        // Poisson floor; measured +1.4/+1.6/+1.4 dB over seeds 7/71/151, and
        // +2.2 to +3.0 dB for the more Poisson-dominated calibrations.
        let (gain, sigma) = (1.3, 1.5);
        let gat = VstKind::Gat { gain, sigma };
        let (clean, noisy) = gat_fixture(4096, gain, sigma, 7);
        let identity = stft_wiener_auto(&noisy);
        let stabilized = vst_denoise(&noisy, gat, stft_wiener_auto);
        let s_id = snr_db(&clean, &identity);
        let s_gat = snr_db(&clean, &stabilized);
        assert!(
            s_gat >= s_id + 1.0,
            "GAT sandwich {s_gat:.2} dB must beat identity {s_id:.2} dB by ≥ 1 dB"
        );
    }

    #[test]
    fn gat_fast_carrier_regime_is_a_measured_limitation() {
        // An honest negative pin (see the module docs, "Known limitation: fast
        // carriers"). On a FAST sinusoidal intensity (40 cycles / 4096 samples)
        // the pointwise root converts the carrier into harmonics; the inner
        // linear shrinkage then clips those harmonics, and the sandwich measured
        // ≈ −1 dB against the identity pipeline across every (gain, σ) probed —
        // even quasi-pure Poisson. The pin below asserts the loss stays bounded
        // (< 2 dB); if this test ever fails in the *positive* direction, the
        // limitation note in the module docs should be revisited.
        let (gain, sigma) = (1.3, 1.5);
        let gat = VstKind::Gat { gain, sigma };
        let n = 4096;
        let lambda: Vec<f64> = (0..n)
            .map(|i| 6.0 + 5.0 * (2.0 * PI * 40.0 * i as f64 / n as f64).sin())
            .collect();
        let mut rng = Lcg::new(7);
        let noisy: Vec<f64> = lambda
            .iter()
            .map(|&l| gain * poisson(&mut rng, l) + sigma * rng.gauss())
            .collect();
        let clean: Vec<f64> = lambda.iter().map(|&l| gain * l).collect();
        let s_id = snr_db(&clean, &stft_wiener_auto(&noisy));
        let s_gat = snr_db(&clean, &vst_denoise(&noisy, gat, stft_wiener_auto));
        assert!(
            s_gat >= s_id - 2.0,
            "fast-carrier loss should stay bounded: identity {s_id:.2} dB, GAT {s_gat:.2} dB"
        );
        assert!(
            s_gat <= s_id + 1.0,
            "fast-carrier regime measured as a ≈ −1 dB limitation; it now WINS \
             ({s_gat:.2} vs {s_id:.2} dB) — revisit the module-doc limitation note"
        );
    }

    #[test]
    fn gat_degenerate_parameters_degrade_to_copy() {
        let x = [0.5, 1.0, -2.0, 7.0];
        for (gain, sigma) in [
            (0.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, -0.5),
            (1.0, f64::NAN),
        ]
        {
            let gat = VstKind::Gat { gain, sigma };
            assert_eq!(vst_forward(gat, &x), x);
            assert_eq!(vst_inverse_naive(gat, &x), x);
            assert_eq!(vst_inverse_corrected(gat, &x, &[0.1, -0.1]), x);
        }
    }
}
