//! # Saturating companders — for *reconstruction-free* uses only
//!
//! Bounded, strictly monotone pointwise maps (tanh / arctangent / softsign) for
//! **display compression** and **robust feature extraction**: taming outliers and
//! huge dynamic ranges while preserving small-signal behaviour exactly (unit slope
//! at the origin).
//!
//! ## Why there is deliberately no inverse here
//!
//! The TSHF research program (`TSHF_RESEARCH_2026-07-16.md`, §12 recommendation 3)
//! measured what happens when saturating maps are used as the φ of an inverted
//! denoising pipeline `φ⁻¹ ∘ filter ∘ φ`: the inverse's Lipschitz factor
//! `max |dφ⁻¹/dy|` reaches **×101.4 for tanh** (×22.1 sigmoid, ×16.0 softsign,
//! ×10.0 atan) exactly where the signal is strong, and the Jensen retransformation
//! bias reaches **−6.5 % of the level** (tanh, report E2/E4). These maps are fine
//! *companders* (μ-law relatives, OFDM PAPR literature) but disqualified as
//! invertible denoising transforms — so this module exposes the forward maps only,
//! and the variance-stabilizing pipeline lives in [`super::vst`] with properly
//! bias-corrected inverses instead.
//!
//! Every map here is scaled so that `soft_clip(x) ≈ x` for `|x| ≪ limit` and
//! `|soft_clip(x)| < limit` for all finite `x` (mathematically — f64 rounding can
//! land exactly on the bound for extreme arguments):
//!
//! | kind | formula | approach to the bound |
//! |------|---------|----------------------|
//! | [`SoftClipKind::Tanh`] | `limit·tanh(x/limit)` | exponential (fastest) |
//! | [`SoftClipKind::Atan`] | `(2·limit/π)·atan(π·x/(2·limit))` | ∝ 1/x |
//! | [`SoftClipKind::Softsign`] | `x/(1 + |x|/limit)` | ∝ 1/x (cheapest, no transcendentals) |

use super::{mad, median};

/// Which saturating map [`soft_clip`] applies. All three are odd, strictly
/// increasing, have unit slope at the origin and asymptote `±limit`; they differ
/// only in how fast they saturate (see the module table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftClipKind {
    /// `limit·tanh(x/limit)` — smooth, saturates exponentially fast.
    Tanh,
    /// `(2·limit/π)·atan(π·x/(2·limit))` — gentlest roll-off, heavy tails survive
    /// proportionally more.
    Atan,
    /// `x/(1 + |x|/limit)` — algebraic (no transcendental calls), roll-off like atan.
    Softsign,
}

fn clip_one(kind: SoftClipKind, limit: f64, x: f64) -> f64 {
    if x.is_infinite()
    {
        // All three maps have the same asymptote; evaluating softsign at ±∞ would
        // produce ∞/∞ = NaN, so the limit value is taken explicitly.
        return limit.copysign(x);
    }
    match kind
    {
        SoftClipKind::Tanh => limit * (x / limit).tanh(),
        SoftClipKind::Atan =>
        {
            let c = 2.0 * limit / core::f64::consts::PI;
            c * (x / c).atan()
        },
        SoftClipKind::Softsign => x / (1.0 + (x / limit).abs()),
    }
}

/// Soft-limit a signal to `(−limit, +limit)` around zero: outliers are tamed,
/// values well inside the bound pass through essentially unchanged (unit slope at
/// the origin). Strictly monotone, so order statistics of the input are preserved.
///
/// This is a *display/feature* transform, *not* a denoiser: it removes no noise,
/// it reshapes amplitude. There is deliberately no inverse (see the module doc).
///
/// Degrades gracefully: a non-finite or non-positive `limit` returns the input
/// unchanged. Non-finite samples propagate (`NaN → NaN`, `±∞ → ±limit`), so a
/// poisoned sample stays visibly poisoned instead of being silently invented.
pub fn soft_clip(signal: &[f64], kind: SoftClipKind, limit: f64) -> Vec<f64> {
    if !(limit.is_finite() && limit > 0.0)
    {
        return signal.to_vec();
    }
    signal.iter().map(|&x| clip_one(kind, limit, x)).collect()
}

/// Robust display compression: soft-limit around the *median*, with the bound set
/// to `n_sigmas` robust standard deviations (`1.4826·MAD`). The bulk of a
/// well-behaved signal (within ±1σ of its median) is left essentially untouched
/// while spikes land just outside the `n_sigmas` band instead of dominating the
/// plot or the feature scale.
///
/// Degrades gracefully: with fewer than two samples, a non-finite/non-positive
/// `n_sigmas`, or a degenerate scale (`MAD = 0`, e.g. a constant signal), the
/// input is returned unchanged — a constant has no outliers to tame.
pub fn soft_clip_robust(signal: &[f64], kind: SoftClipKind, n_sigmas: f64) -> Vec<f64> {
    if signal.len() < 2 || !(n_sigmas.is_finite() && n_sigmas > 0.0)
    {
        return signal.to_vec();
    }
    let center = median(signal);
    let sigma = 1.4826 * mad(signal);
    let limit = n_sigmas * sigma;
    if !(limit.is_finite() && limit > 0.0)
    {
        return signal.to_vec();
    }
    signal
        .iter()
        .map(|&x| center + clip_one(kind, limit, x - center))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::testutil::Lcg;
    use super::*;

    const KINDS: [SoftClipKind; 3] = [
        SoftClipKind::Tanh,
        SoftClipKind::Atan,
        SoftClipKind::Softsign,
    ];

    #[test]
    fn strictly_monotone_and_bounded() {
        for kind in KINDS
        {
            let xs: Vec<f64> = (-400..=400).map(|i| i as f64 * 0.05).collect();
            let ys = soft_clip(&xs, kind, 2.0);
            for w in ys.windows(2)
            {
                assert!(w[1] > w[0], "{kind:?} not strictly increasing");
            }
            for &y in &ys
            {
                assert!(y.abs() < 2.0, "{kind:?} escaped the bound: {y}");
            }
        }
    }

    #[test]
    fn unit_slope_at_origin_and_saturation() {
        for kind in KINDS
        {
            let limit = 3.0;
            // Small signals pass through essentially unchanged (unit slope).
            let small = 0.01 * limit;
            let y = soft_clip(&[small], kind, limit)[0];
            assert!(
                (y - small).abs() < 1.0e-4 * limit,
                "{kind:?} distorts small signals: {small} → {y}"
            );
            // Huge signals approach the bound without escaping it. (`<=` rather
            // than `<`: mathematically tanh never reaches 1, but tanh(100) rounds
            // to exactly 1.0 in f64.)
            let y = soft_clip(&[100.0 * limit], kind, limit)[0];
            assert!(y > 0.95 * limit && y <= limit, "{kind:?} saturation: {y}");
            // Odd symmetry.
            let yn = soft_clip(&[-100.0 * limit], kind, limit)[0];
            assert!((yn + y).abs() < 1.0e-12);
        }
    }

    #[test]
    fn robust_variant_tames_outliers_and_keeps_the_bulk() {
        let mut rng = Lcg::new(41);
        let n = 1024;
        let mut x: Vec<f64> = (0..n).map(|_| 5.0 + 0.2 * rng.gauss()).collect();
        for i in (13..n).step_by(97)
        {
            x[i] += 40.0; // huge spikes far above the bulk
        }
        for kind in KINDS
        {
            let y = soft_clip_robust(&x, kind, 4.0);
            let center = 5.0;
            let sigma = 0.2;
            for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate()
            {
                if xi > 20.0
                {
                    // Outliers land near (never beyond) the ±4σ band.
                    assert!(
                        (yi - center).abs() < 4.0 * sigma * 1.6,
                        "{kind:?} left outlier at {yi} (i = {i})"
                    );
                }
                else if (xi - center).abs() <= sigma
                {
                    // The ±1σ bulk moves by a small fraction of the noise scale.
                    assert!(
                        (yi - xi).abs() < 0.25 * sigma,
                        "{kind:?} disturbed bulk sample {xi} → {yi} (i = {i})"
                    );
                }
            }
        }
    }

    #[test]
    fn order_statistics_survive() {
        // Strict monotonicity ⇒ the sample ranks are unchanged; the median of the
        // compressed signal is the compressed median.
        let mut rng = Lcg::new(97);
        let x: Vec<f64> = (0..257).map(|_| 3.0 * rng.gauss()).collect();
        for kind in KINDS
        {
            let y = soft_clip(&x, kind, 1.5);
            let m = super::super::median(&x);
            let expected = soft_clip(&[m], kind, 1.5)[0];
            assert!((super::super::median(&y) - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn degrades_gracefully() {
        let x = [1.0, f64::NAN, -3.0];
        for kind in KINDS
        {
            // Bad limit → identity copy.
            assert_eq!(soft_clip(&x[..1], kind, 0.0), &x[..1]);
            assert_eq!(soft_clip(&x[..1], kind, f64::NAN), &x[..1]);
            assert_eq!(soft_clip(&x[..1], kind, -1.0), &x[..1]);
            // NaN propagates visibly; finite neighbours are unaffected.
            let y = soft_clip(&x, kind, 2.0);
            assert!(y[0].is_finite() && y[1].is_nan() && y[2].is_finite());
            // ±∞ saturates to the bound instead of poisoning the output.
            let y = soft_clip(&[f64::INFINITY, f64::NEG_INFINITY], kind, 2.0);
            assert!((y[0] - 2.0).abs() < 1.0e-12 && (y[1] + 2.0).abs() < 1.0e-12);
            // Constant signal: MAD = 0 → identity (nothing to tame).
            let c = vec![7.0; 32];
            assert_eq!(soft_clip_robust(&c, kind, 4.0), c);
            // Empty and tiny inputs.
            assert!(soft_clip(&[], kind, 1.0).is_empty());
            assert_eq!(soft_clip_robust(&[1.0], kind, 4.0), vec![1.0]);
        }
    }
}
