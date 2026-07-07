//! Process-capability indices, the inertial index `Cpi`, and conformity
//! (PPM / sigma level).
//!
//! The classical indices judge a `[LSL, USL]` interval:
//!
//! ```text
//! Cp   = (USL − LSL) / (6σ)                              (potential, spread only)
//! Cpk  = min(USL − μ, μ − LSL) / (3σ)                    (actual, centred spread)
//! Cpm  = (USL − LSL) / (6·√(σ² + (μ − T)²))              (Taguchi, penalises off-target)
//! Cpmk = min(USL − μ, μ − LSL) / (3·√(σ² + (μ − T)²))
//! ```
//!
//! `Cp`/`Cpk` computed from the *within-subgroup* σ are the short-term
//! capability; the same formulas on the *overall* σ are the long-term
//! performance `Pp`/`Ppk`. The functions here are σ-agnostic — pass the σ you
//! mean — and [`pp`]/[`ppk`] are provided as intention-revealing aliases.
//!
//! Inertial tolerancing replaces the interval with an inertia budget `I_max`
//! and reports the **inertial capability index**
//!
//! ```text
//! Cpi = I_max / I ,     I = √(σ² + (μ − T)²).
//! ```
//!
//! `Cpi ≥ 1` ⇔ the batch is inside the inertia cone. With the `Cp = 1` budget
//! `I_max = (USL − LSL)/6`, `Cpi` coincides exactly with `Cpm` — the inertial
//! index *is* the Taguchi index, read against an explicit inertia limit.

use crate::inertia::Inertia;
use crate::special::{chi2_quantile, inv_normal_cdf, normal_cdf, normal_sf};
use serde::{Deserialize, Serialize};

/// `Cp = (USL − LSL) / (6σ)` — potential capability (spread only, ignores
/// centering). `+∞` for a degenerate `σ = 0`.
pub fn cp(sigma: f64, lsl: f64, usl: f64) -> f64 {
    if sigma <= 0.0
    {
        return f64::INFINITY;
    }
    (usl - lsl) / (6.0 * sigma)
}

/// `Cpk = min(USL − μ, μ − LSL) / (3σ)` — actual capability, penalising
/// off-centering within the interval. `+∞` for `σ = 0`.
pub fn cpk(mean: f64, sigma: f64, lsl: f64, usl: f64) -> f64 {
    if sigma <= 0.0
    {
        return f64::INFINITY;
    }
    ((usl - mean).min(mean - lsl)) / (3.0 * sigma)
}

/// `Cpm = (USL − LSL) / (6·I)` with `I = √(σ² + (μ − T)²)` — the Taguchi
/// capability index, which penalises departure from the target `T`.
pub fn cpm(mean: f64, sigma: f64, target: f64, lsl: f64, usl: f64) -> f64 {
    let i = Inertia::from_moments(mean, sigma, target).value();
    if i <= 0.0
    {
        return f64::INFINITY;
    }
    (usl - lsl) / (6.0 * i)
}

/// `Cpmk = min(USL − μ, μ − LSL) / (3·I)` with `I = √(σ² + (μ − T)²)` — the
/// centered Taguchi index (combines `Cpk`'s one-sided margin with `Cpm`'s
/// target penalty).
pub fn cpmk(mean: f64, sigma: f64, target: f64, lsl: f64, usl: f64) -> f64 {
    let i = Inertia::from_moments(mean, sigma, target).value();
    if i <= 0.0
    {
        return f64::INFINITY;
    }
    ((usl - mean).min(mean - lsl)) / (3.0 * i)
}

/// `Pp` — long-term performance analogue of [`cp`] (identical formula, meant
/// to be called with the overall σ).
pub fn pp(sigma: f64, lsl: f64, usl: f64) -> f64 {
    cp(sigma, lsl, usl)
}

/// `Ppk` — long-term performance analogue of [`cpk`] (identical formula,
/// meant to be called with the overall σ).
pub fn ppk(mean: f64, sigma: f64, lsl: f64, usl: f64) -> f64 {
    cpk(mean, sigma, lsl, usl)
}

/// The **inertial capability index** `Cpi = I_max / I`.
///
/// `Cpi ≥ 1` ⇔ the characteristic is inside the inertia cone of radius
/// `I_max`. `+∞` for a null inertia; `0` for a non-positive budget.
pub fn cpi(inertia: &Inertia, i_max: f64) -> f64 {
    if i_max <= 0.0
    {
        return 0.0;
    }
    let i = inertia.value();
    if i <= 0.0
    {
        return f64::INFINITY;
    }
    i_max / i
}

/// Non-conformity of a normal characteristic against `[LSL, USL]`, in parts
/// per million:
///
/// ```text
/// PPM = 10⁶ · [ Φ((LSL − μ)/σ) + (1 − Φ((USL − μ)/σ)) ].
/// ```
///
/// Each tail is evaluated directly (via `erfc`) so deep-tail (5–6σ) rates stay
/// relatively accurate. A degenerate `σ = 0` yields `0` inside the interval,
/// `10⁶` outside.
pub fn nonconformity_ppm(mean: f64, sigma: f64, lsl: f64, usl: f64) -> f64 {
    if sigma <= 0.0
    {
        return if mean >= lsl && mean <= usl { 0.0 } else { 1e6 };
    }
    let below = normal_cdf((lsl - mean) / sigma);
    let above = normal_sf((usl - mean) / sigma);
    (below + above) * 1e6
}

/// Short-hand "sigma level" (process z-bench): the one-sided normal quantile
/// whose upper-tail probability equals the *total* out-of-spec fraction.
/// A defect-free process returns `+∞`.
pub fn sigma_level(mean: f64, sigma: f64, lsl: f64, usl: f64) -> f64 {
    let frac = nonconformity_ppm(mean, sigma, lsl, usl) / 1e6;
    if frac <= 0.0
    {
        return f64::INFINITY;
    }
    // z such that P(Z > z) = frac  ⇒  z = Φ⁻¹(1 − frac).
    crate::special::inv_normal_cdf(1.0 - frac)
}

/// A one-call capability/inertia summary of a sample against a bilateral
/// specification, computed on the sample's **overall** dispersion (so the
/// `Cp`/`Cpk` fields are, strictly, the long-term `Pp`/`Ppk`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySummary {
    /// Sample mean `μ̂`.
    pub mean: f64,
    /// Sample (population) dispersion `σ̂`.
    pub sigma: f64,
    /// Estimated inertia `Î = √(δ̂² + σ̂²)`.
    pub inertia: f64,
    /// `Cp` (`= Pp` here) — potential capability.
    pub cp: f64,
    /// `Cpk` (`= Ppk` here) — actual capability.
    pub cpk: f64,
    /// `Cpm` — Taguchi index against the target.
    pub cpm: f64,
    /// `Cpi = I_max / Î` — inertial capability index.
    pub cpi: f64,
    /// Predicted non-conformity in parts per million.
    pub ppm: f64,
}

impl CapabilitySummary {
    /// Compute every index for `data` against `[lsl, usl]` with target
    /// `target` and inertia budget `i_max`.
    pub fn from_sample(data: &[f64], lsl: f64, usl: f64, target: f64, i_max: f64) -> Self {
        let inertia = Inertia::from_sample(data, target);
        let mean = target + inertia.off_centering;
        let sigma = inertia.sigma;
        Self {
            mean,
            sigma,
            inertia: inertia.value(),
            cp: cp(sigma, lsl, usl),
            cpk: cpk(mean, sigma, lsl, usl),
            cpm: cpm(mean, sigma, target, lsl, usl),
            cpi: cpi(&inertia, i_max),
            ppm: nonconformity_ppm(mean, sigma, lsl, usl),
        }
    }
}

/// Two-sided confidence interval for `Cp` from an estimate `cp_hat` on a sample
/// of size `n`, at confidence `conf`. `Cp ∝ 1/σ` and `(n−1)ŝ²/σ² ~ χ²_{n−1}`, so
///
/// ```text
/// [ Ĉp·√(χ²_{α/2,ν}/ν) ,  Ĉp·√(χ²_{1−α/2,ν}/ν) ] ,   ν = n−1, α = 1−conf,
/// ```
///
/// an **exact** interval under normality. Returns `None` for `n < 2` or `conf`
/// outside `(0, 1)`.
pub fn cp_confidence_interval(cp_hat: f64, n: usize, conf: f64) -> Option<(f64, f64)> {
    if n < 2 || conf <= 0.0 || conf >= 1.0
    {
        return None;
    }
    let nu = (n - 1) as f64;
    let alpha = 1.0 - conf;
    let lo = cp_hat * (chi2_quantile(nu, alpha / 2.0) / nu).sqrt();
    let hi = cp_hat * (chi2_quantile(nu, 1.0 - alpha / 2.0) / nu).sqrt();
    Some((lo, hi))
}

/// Two-sided confidence interval for `Cpk` from an estimate `cpk_hat` on a
/// sample of size `n`, at confidence `conf`, by the Bissell large-sample normal
/// approximation
///
/// ```text
/// Ĉpk ± z_{1−α/2}·√( 1/(9n) + Ĉpk²/(2(n−1)) ) .
/// ```
///
/// Approximate (there is no exact closed form for `Cpk`), and accurate for
/// `n ≳ 30`. Returns `None` for `n < 2` or `conf` outside `(0, 1)`.
pub fn cpk_confidence_interval(cpk_hat: f64, n: usize, conf: f64) -> Option<(f64, f64)> {
    if n < 2 || conf <= 0.0 || conf >= 1.0
    {
        return None;
    }
    let z = inv_normal_cdf(1.0 - 0.5 * (1.0 - conf));
    let se = (1.0 / (9.0 * n as f64) + cpk_hat * cpk_hat / (2.0 * (n - 1) as f64)).sqrt();
    Some((cpk_hat - z * se, cpk_hat + z * se))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn classical_indices_on_centered_process() {
        // μ = T = 10, σ = 1, spec ±3 ⇒ Cp = Cpk = Cpm = 1.
        assert_relative_eq!(cp(1.0, 7.0, 13.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(cpk(10.0, 1.0, 7.0, 13.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(cpm(10.0, 1.0, 10.0, 7.0, 13.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn off_centering_splits_cp_from_cpk_and_cpm() {
        // μ = 11, T = 10, σ = 1, spec [7,13]: Cp = 1, Cpk = 2/3.
        assert_relative_eq!(cp(1.0, 7.0, 13.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(cpk(11.0, 1.0, 7.0, 13.0), 2.0 / 3.0, epsilon = 1e-12);
        // Cpm = 6 / (6·√2) = 1/√2.
        assert_relative_eq!(
            cpm(11.0, 1.0, 10.0, 7.0, 13.0),
            1.0 / 2.0_f64.sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn cpi_equals_cpm_when_budget_is_cp1() {
        // With I_max = (USL−LSL)/6, Cpi ≡ Cpm.
        let (mean, sigma, target, lsl, usl) = (11.0, 1.0, 10.0, 7.0, 13.0);
        let inertia = Inertia::from_moments(mean, sigma, target);
        let i_max = (usl - lsl) / 6.0;
        assert_relative_eq!(
            cpi(&inertia, i_max),
            cpm(mean, sigma, target, lsl, usl),
            epsilon = 1e-12
        );
    }

    #[test]
    fn cpi_flags_cone_boundary() {
        // I = 0.1, budget 0.1 ⇒ Cpi = 1 exactly (on the boundary).
        let inertia = Inertia::new(0.06, 0.08);
        assert_relative_eq!(cpi(&inertia, 0.1), 1.0, epsilon = 1e-12);
        assert!(cpi(&inertia, 0.12) > 1.0);
        assert!(cpi(&inertia, 0.08) < 1.0);
    }

    #[test]
    fn ppm_of_a_centered_3sigma_process() {
        // Centered, spec at ±3σ ⇒ 2·Φ(−3) ≈ 2699.8 ppm.
        let ppm = nonconformity_ppm(0.0, 1.0, -3.0, 3.0);
        assert_relative_eq!(ppm, 2699.796, epsilon = 0.05);
    }

    #[test]
    fn ppm_tail_is_accurate_at_six_sigma() {
        // Centered, spec at ±6σ ⇒ 2·Φ(−6) ≈ 0.001973 ppm.
        let ppm = nonconformity_ppm(0.0, 1.0, -6.0, 6.0);
        assert_relative_eq!(ppm, 0.001_973, epsilon = 1e-6);
    }

    #[test]
    fn sigma_level_recovers_three_sigma() {
        let z = sigma_level(0.0, 1.0, -3.0, 3.0);
        // total tail 2·Φ(−3); z = Φ⁻¹(1 − tail) ≈ 2.782.
        assert_relative_eq!(z, 2.782, epsilon = 1e-2);
    }

    #[test]
    fn summary_bundles_everything() {
        let data = [9.9, 10.1, 10.0, 10.2, 9.8, 10.05];
        let s = CapabilitySummary::from_sample(&data, 9.4, 10.6, 10.0, 0.2);
        assert!(s.cp.is_finite() && s.cpk.is_finite());
        assert!(s.cpi > 1.0); // tight batch, generous 0.2 budget
        assert!(s.ppm >= 0.0);
        assert_relative_eq!(s.inertia, s.inertia, epsilon = 1e-12);
    }

    #[test]
    fn cp_confidence_interval_brackets_estimate_and_tightens() {
        let (lo, hi) = cp_confidence_interval(1.33, 30, 0.95).unwrap();
        assert!(lo < 1.33 && 1.33 < hi);
        // Larger n ⇒ narrower interval.
        let (lo2, hi2) = cp_confidence_interval(1.33, 200, 0.95).unwrap();
        assert!((hi2 - lo2) < (hi - lo));
        assert!(cp_confidence_interval(1.0, 1, 0.95).is_none());
    }

    #[test]
    fn cpk_confidence_interval_is_symmetric_and_shrinks() {
        let (lo, hi) = cpk_confidence_interval(1.33, 50, 0.95).unwrap();
        assert!(lo < 1.33 && 1.33 < hi);
        // Symmetric about the estimate (normal approximation).
        assert_relative_eq!(1.33 - lo, hi - 1.33, epsilon = 1e-12);
        let (lo2, hi2) = cpk_confidence_interval(1.33, 500, 0.95).unwrap();
        assert!((hi2 - lo2) < (hi - lo));
    }
}
