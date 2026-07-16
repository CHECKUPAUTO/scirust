// scirust-core/src/nn/intervals.rs
//
// Parametric prediction intervals — the distributional complement to the
// distribution-free `ConformalRegressor` in `nn::conformal`.
//
// Conformal intervals need no distributional assumption but pay for it with a
// constant, non-adaptive half-width and (at small calibration sizes) looseness.
// When the residuals `r = y − ŷ` of an unbiased predictor are plausibly iid
// Gaussian, a *parametric* interval is tighter and exact:
//
//   half-width(c) = qₚ · σ̂ ,   p = (1 + c) / 2 ,   σ̂ = √(Σ rᵢ² / n)
//
// where `qₚ` is the `p`-quantile of the standard normal (large `n`) or of
// Student-t with `ν = n` degrees of freedom (small `n`). Since `σ̂` is itself
// estimated from `n` residuals, the Student-t quantile is the *correct* one; it
// is wider than the normal for small `n` and converges to it as `n → ∞`. (With
// a known-zero mean and `σ̂` the residual RMS, `r_new / σ̂ ∼ tₙ`, so no extra
// `√(1+1/n)` term is needed.)
//
// The Student-t CDF is built on `scirust_special::regularized_incomplete_beta`
// (core has no Student-t otherwise); the normal quantile reuses the existing
// `nn::smoothing::inv_normal_cdf`.

use crate::nn::smoothing::inv_normal_cdf;

/// Standard-normal CDF, `Φ(x) = ½(1 + erf(x/√2))`.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + scirust_special::erf(x / std::f64::consts::SQRT_2))
}

/// Student-t CDF with `nu` degrees of freedom, via the regularized incomplete
/// beta: for `x = ν/(ν + t²)`, `Iₓ(ν/2, ½)` is the two-tailed mass beyond `|t|`.
pub fn student_t_cdf(t: f64, nu: f64) -> f64 {
    if nu <= 0.0
    {
        return f64::NAN;
    }
    if t == 0.0
    {
        return 0.5;
    }
    let x = nu / (nu + t * t);
    let ib = scirust_special::regularized_incomplete_beta(0.5 * nu, 0.5, x);
    if t > 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
}

/// Student-t inverse CDF (quantile) with `nu` dof, by bisection on the
/// monotone CDF. `p` is clamped to `(0, 1)`.
pub fn student_t_ppf(p: f64, nu: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if (p - 0.5).abs() < 1e-15
    {
        return 0.0;
    }
    // Symmetric, so bracket by magnitude and mirror the sign.
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let target = if p > 0.5 { p } else { 1.0 - p };
    while student_t_cdf(hi, nu) < target && hi < 1e9
    {
        hi *= 2.0;
    }
    for _ in 0..200
    {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, nu) < target
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    let mag = 0.5 * (lo + hi);
    if p > 0.5 { mag } else { -mag }
}

/// Gaussian prediction interval: assumes iid `N(0, σ²)` residuals with `σ`
/// estimated as their RMS. The large-sample interval `ŷ ± Φ⁻¹((1+c)/2)·σ̂`.
#[derive(Debug, Clone, Copy)]
pub struct GaussianInterval {
    sigma: f32,
}

impl GaussianInterval {
    /// Estimate `σ̂` from calibration residuals `r = y − ŷ`.
    pub fn from_residuals(residuals: &[f32]) -> Self {
        Self {
            sigma: rms(residuals),
        }
    }

    /// Half-width of the interval at coverage `confidence` (e.g. 0.95).
    pub fn half_width(&self, confidence: f32) -> f32 {
        let p = 0.5 * (1.0 + confidence as f64);
        (inv_normal_cdf(p) * self.sigma as f64) as f32
    }

    /// Prediction interval `[ŷ − h, ŷ + h]` around a point estimate.
    pub fn interval(&self, y_hat: f32, confidence: f32) -> (f32, f32) {
        let h = self.half_width(confidence);
        (y_hat - h, y_hat + h)
    }

    /// The estimated residual scale `σ̂`.
    pub fn sigma(&self) -> f32 {
        self.sigma
    }
}

/// Student-t prediction interval: same Gaussian-residual assumption but with the
/// small-sample-correct quantile `t_{ν=n}`, which accounts for estimating `σ`
/// from `n` residuals. Wider than [`GaussianInterval`] for small `n`, and
/// converges to it as `n → ∞`.
#[derive(Debug, Clone, Copy)]
pub struct StudentTInterval {
    sigma: f32,
    dof: f64,
}

impl StudentTInterval {
    /// Estimate `σ̂` and set `ν = n` from `n` calibration residuals.
    pub fn from_residuals(residuals: &[f32]) -> Self {
        Self {
            sigma: rms(residuals),
            dof: residuals.len().max(1) as f64,
        }
    }

    /// Half-width `t_{ν,(1+c)/2} · σ̂` at coverage `confidence`.
    pub fn half_width(&self, confidence: f32) -> f32 {
        let p = 0.5 * (1.0 + confidence as f64);
        (student_t_ppf(p, self.dof) * self.sigma as f64) as f32
    }

    /// Prediction interval `[ŷ − h, ŷ + h]` around a point estimate.
    pub fn interval(&self, y_hat: f32, confidence: f32) -> (f32, f32) {
        let h = self.half_width(confidence);
        (y_hat - h, y_hat + h)
    }

    pub fn sigma(&self) -> f32 {
        self.sigma
    }
    pub fn dof(&self) -> f64 {
        self.dof
    }
}

/// Root-mean-square (residual scale under a known-zero mean).
fn rms(residuals: &[f32]) -> f32 {
    if residuals.is_empty()
    {
        return 0.0;
    }
    let ss: f64 = residuals.iter().map(|&r| (r as f64) * (r as f64)).sum();
    (ss / residuals.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_cdf_landmarks() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-12);
        assert!((normal_cdf(1.959964) - 0.975).abs() < 1e-5);
        assert!((normal_cdf(-1.0) - 0.1586553).abs() < 1e-5);
    }

    #[test]
    fn student_t_cdf_landmarks() {
        // Symmetry and known quantiles.
        assert!((student_t_cdf(0.0, 10.0) - 0.5).abs() < 1e-12);
        assert!((student_t_cdf(2.228139, 10.0) - 0.975).abs() < 1e-4);
        // ν=1 is Cauchy: CDF(1) = 0.75.
        assert!((student_t_cdf(1.0, 1.0) - 0.75).abs() < 1e-4);
    }

    #[test]
    fn student_t_ppf_matches_tables() {
        // Textbook two-sided 95% t-values.
        assert!((student_t_ppf(0.975, 1.0) - 12.7062).abs() < 1e-2);
        assert!((student_t_ppf(0.975, 10.0) - 2.2281).abs() < 1e-3);
        assert!((student_t_ppf(0.975, 30.0) - 2.0423).abs() < 1e-3);
        // Large ν → normal quantile.
        assert!((student_t_ppf(0.975, 100000.0) - 1.95996).abs() < 1e-3);
        // Symmetry.
        assert!((student_t_ppf(0.25, 7.0) + student_t_ppf(0.75, 7.0)).abs() < 1e-6);
    }

    #[test]
    fn student_t_wider_than_gaussian_for_small_n_and_converges() {
        // Same residuals; Student-t interval must be wider for small n and
        // essentially equal to Gaussian for large n.
        let small: Vec<f32> = (0..5).map(|i| (i as f32 - 2.0) * 0.5).collect();
        let g = GaussianInterval::from_residuals(&small).half_width(0.95);
        let t = StudentTInterval::from_residuals(&small).half_width(0.95);
        assert!(t > g, "student-t ({t}) should exceed gaussian ({g}) at n=5");

        let big: Vec<f32> = (0..5000).map(|i| ((i as f32) * 0.001).sin()).collect();
        let gb = GaussianInterval::from_residuals(&big).half_width(0.95);
        let tb = StudentTInterval::from_residuals(&big).half_width(0.95);
        assert!(
            (tb - gb).abs() / gb < 0.01,
            "should converge: t={tb} g={gb}"
        );
    }

    #[test]
    fn interval_is_centered_and_symmetric() {
        let iv = GaussianInterval::from_residuals(&[1.0, -1.0, 0.5, -0.5]);
        let (lo, hi) = iv.interval(3.0, 0.9);
        assert!((0.5 * (lo + hi) - 3.0).abs() < 1e-5, "centered on ŷ");
        assert!(hi > 3.0 && lo < 3.0);
    }
}
