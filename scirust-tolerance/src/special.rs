//! Special functions needed for capability / conformity computations.
//!
//! Kept local (like `scirust-spc`'s `ln_gamma`) so the crate stays a
//! leaf with a single `serde` dependency rather than pulling in a shared
//! special-functions crate.
//!
//! - [`erf`] / [`erfc`] — the error function and its complement. `erf` uses
//!   the all-positive Kummer series (no cancellation, valid for every `x`);
//!   `erfc` switches to a Lentz continued fraction in the tail so the deep
//!   tail keeps full *relative* accuracy (a `1 - erf(x)` subtraction would
//!   not — it is catastrophic cancellation past `x ≈ 4`).
//! - [`normal_cdf`] / [`normal_sf`] — standard-normal CDF `Φ` and survival
//!   `1 - Φ`, both routed through [`erfc`] so each tail is computed directly.
//! - [`chi2_cdf`] / [`chi2_quantile`] — χ² CDF (regularised incomplete gamma)
//!   and its Newton-refined inverse, used by the piloting chart's upper limit.
//! - [`ncchi2_cdf`] — non-central χ² CDF (Poisson mixture of central χ²), the
//!   sampling law behind the acceptance-sampling operating-characteristic curve.

use core::f64::consts::PI;

/// Error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
///
/// Uses the confluent-hypergeometric (Kummer-transformed) series
///
/// ```text
/// erf(x) = (2/√π) · e^{-x²} · Σ_{n≥0}  2ⁿ x^{2n+1} / (1·3·5···(2n+1))
/// ```
///
/// whose terms are all positive, so there is no subtractive cancellation.
/// Accurate to full `f64` precision; the term count grows like `x²`, which is
/// irrelevant for the ranges capability work touches. For `|x| ≥ 6` the result
/// saturates at `±1` — exact in `f64` there, since `erfc(6) ≈ 2.15e-17` is
/// below the ULP of `1.0` — which also avoids the `e^{x²}` intermediate
/// overflow the series would otherwise hit near `|x| ≈ 27`.
pub fn erf(x: f64) -> f64 {
    // Delegated to the audited `scirust-special` implementation (regularized
    // incomplete gamma), which is at least as accurate as the former local
    // Kummer series and shared with every other consumer. The module's own
    // tests below now cross-validate the shared version.
    scirust_special::erf(x)
}

/// Complementary error function `erfc(x) = 1 - erf(x)`, accurate in the tail.
///
/// For `|x| < 4` this is `1 - erf(x)` (both operands `O(1)`). For larger
/// `x` it evaluates the continued fraction
///
/// ```text
/// erfc(x) = e^{-x²}/√π · 1/(x + ½/(x + 1/(x + 3⁄2/(x + 2/(x + …)))))
/// ```
///
/// by the modified-Lentz algorithm, which keeps full relative accuracy
/// where `erf(x) → 1` and `1 - erf(x)` would lose all significant digits.
pub fn erfc(x: f64) -> f64 {
    // Delegated to `scirust-special`, whose `erfc` is likewise tail-accurate
    // (upper incomplete gamma), replacing the former local Lentz continued
    // fraction. See the note on `erf`.
    scirust_special::erfc(x)
}

/// Standard-normal cumulative distribution `Φ(z) = P(Z ≤ z)`.
///
/// Computed as `½·erfc(-z/√2)` so the lower tail keeps relative accuracy.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / core::f64::consts::SQRT_2)
}

/// Standard-normal survival function `1 - Φ(z) = P(Z > z)`.
///
/// Computed as `½·erfc(z/√2)` so the upper tail keeps relative accuracy.
pub fn normal_sf(z: f64) -> f64 {
    0.5 * erfc(z / core::f64::consts::SQRT_2)
}

/// Inverse standard-normal CDF `Φ⁻¹(p)` (Acklam's rational approximation,
/// refined by one Halley step). Valid on the open interval `(0, 1)`.
pub fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0
    {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0
    {
        return f64::INFINITY;
    }
    // Coefficients (Peter Acklam).
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.024_25;
    let p_high = 1.0 - P_LOW;
    let mut x;
    if p < P_LOW
    {
        let q = (-2.0 * p.ln()).sqrt();
        x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    else if p <= p_high
    {
        let q = p - 0.5;
        let r = q * q;
        x = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }
    else
    {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        x = -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    // One Halley refinement.
    let e = normal_cdf(x) - p;
    let u = e * (2.0 * PI).sqrt() * (x * x / 2.0).exp();
    x -= u / (1.0 + x * u / 2.0);
    x
}

/// Natural log of the gamma function, for `x > 0` — delegated to
/// `scirust-special` (was a byte-identical local Lanczos copy).
fn ln_gamma(x: f64) -> f64 {
    scirust_special::ln_gamma(x)
}

/// Regularised lower incomplete gamma function `P(a, x) = γ(a, x) / Γ(a)`,
/// delegated to `scirust-special::regularized_gamma_p` (was a local
/// Numerical-Recipes series/continued-fraction copy).
fn gammp(a: f64, x: f64) -> f64 {
    scirust_special::regularized_gamma_p(a, x)
}

/// Cumulative distribution of the chi-square law with `dof` degrees of freedom,
/// `P(X ≤ x) = P(dof/2, x/2)`.
///
/// For very large `dof` the incomplete-gamma series would need `O(√dof)` terms,
/// so above `dof = 20000` it switches to the Wilson–Hilferty normal
/// approximation `Φ( ((x/k)^{1/3} − (1 − 2/(9k))) / √(2/(9k)) )`, accurate to a
/// few parts in ten-thousand there and `O(1)` in cost.
pub fn chi2_cdf(dof: f64, x: f64) -> f64 {
    if x <= 0.0
    {
        return 0.0;
    }
    if dof > 20_000.0
    {
        let t = (x / dof).cbrt();
        let m = 1.0 - 2.0 / (9.0 * dof);
        let s = (2.0 / (9.0 * dof)).sqrt();
        return normal_cdf((t - m) / s).clamp(0.0, 1.0);
    }
    gammp(dof / 2.0, x / 2.0)
}

/// Cumulative distribution of the **non-central** chi-square law with `dof`
/// degrees of freedom and non-centrality `lambda ≥ 0`, evaluated as the
/// Poisson-weighted mixture of central chi-square CDFs
///
/// ```text
/// F(x; k, λ) = Σ_{j≥0} e^{−λ/2} (λ/2)ʲ / j! · F_central(x; k + 2j).
/// ```
///
/// Reduces to [`chi2_cdf`] at `lambda = 0`. This is the sampling law of the
/// scaled inertia estimator `n·Î²/σ²` (non-centrality `λ = n·δ²/σ²`), so it
/// underpins the acceptance-sampling operating-characteristic curves.
///
/// The Poisson weights are concentrated within roughly `±40√(λ/2)` of the mode
/// `λ/2`; the sum runs over exactly that window. When the window would be
/// impractically wide (very large `λ`, i.e. a nearly-degenerate `σ`), it falls
/// back to **Patnaik's** central-χ² approximation
/// `χ'²(k, λ) ≈ ρ·χ²(ν)` with `ν = (k+λ)²/(k+2λ)`, `ρ = (k+2λ)/(k+λ)`, which is
/// increasingly accurate exactly where the exact sum becomes unwieldy.
pub fn ncchi2_cdf(dof: f64, lambda: f64, x: f64) -> f64 {
    if x <= 0.0
    {
        return 0.0;
    }
    if lambda <= 0.0
    {
        return chi2_cdf(dof, x);
    }
    let half = lambda / 2.0;
    // Significant Poisson mass lies within ~±40·√(λ/2) of the mode λ/2.
    let spread = 40.0 * half.sqrt() + 40.0;
    let j_lo = (half - spread).floor().max(0.0);
    let j_hi = (half + spread).ceil();
    // Fall back to Patnaik's approximation when the exact window is too wide.
    if j_hi - j_lo > 60_000.0
    {
        let nu = (dof + lambda) * (dof + lambda) / (dof + 2.0 * lambda);
        let rho = (dof + 2.0 * lambda) / (dof + lambda);
        return chi2_cdf(nu, x / rho).clamp(0.0, 1.0);
    }
    // Exact Poisson-weighted sum over the significant window (log-space weights).
    let ln_half = half.ln();
    let mut sum = 0.0;
    let mut j = j_lo as u64;
    let hi = j_hi as u64;
    while j <= hi
    {
        let jf = j as f64;
        let w = (-half + jf * ln_half - ln_gamma(jf + 1.0)).exp();
        if w > 0.0
        {
            sum += w * chi2_cdf(dof + 2.0 * jf, x);
        }
        j += 1;
    }
    sum.clamp(0.0, 1.0)
}

/// Quantile (inverse CDF) of the chi-square distribution with `dof` degrees of
/// freedom at probability `p`.
///
/// Seeded with the Wilson–Hilferty cube-root-normal approximation
/// `χ²_{dof;p} ≈ dof·(1 − 2/(9·dof) + z_p·√(2/(9·dof)))³` and refined by
/// Newton steps on the exact CDF ([`chi2_cdf`]), so it is accurate to full
/// `f64` precision at every `dof ≥ 1` (Wilson–Hilferty alone is ~1 % near
/// `dof = 2`).
pub fn chi2_quantile(dof: f64, p: f64) -> f64 {
    if p <= 0.0
    {
        return 0.0;
    }
    if p >= 1.0
    {
        return f64::INFINITY;
    }
    let z = inv_normal_cdf(p);
    let a = 2.0 / (9.0 * dof);
    let t = 1.0 - a + z * a.sqrt();
    let mut x = (dof * t * t * t).max(1e-6);
    let ln_norm = (dof / 2.0) * 2.0_f64.ln() + ln_gamma(dof / 2.0);
    for _ in 0..60
    {
        let err = chi2_cdf(dof, x) - p;
        // pdf(x) = x^{k/2−1} e^{−x/2} / (2^{k/2} Γ(k/2)).
        let ln_pdf = (dof / 2.0 - 1.0) * x.ln() - x / 2.0 - ln_norm;
        let pdf = ln_pdf.exp();
        if pdf <= 0.0
        {
            break;
        }
        let step = err / pdf;
        let mut nx = x - step;
        if nx <= 0.0
        {
            nx = x / 2.0; // keep the iterate positive
        }
        x = nx;
        if step.abs() <= x.abs() * 1e-14
        {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
    }

    fn close_rel(a: f64, b: f64, rel: f64) {
        assert!((a - b).abs() <= rel * b.abs(), "{a} vs {b} (rel {rel})");
    }

    #[test]
    fn erf_matches_reference_values() {
        close(erf(0.0), 0.0, 1e-15);
        close(erf(0.5), 0.520_499_877_813_046_5, 1e-12);
        close(erf(1.0), 0.842_700_792_949_714_9, 1e-12);
        close(erf(2.0), 0.995_322_265_018_952_7, 1e-12);
        close(erf(-1.0), -0.842_700_792_949_714_9, 1e-12);
        close(erf(3.0), 0.999_977_909_503_001_4, 1e-12);
    }

    #[test]
    fn erf_saturates_for_large_x_without_nan() {
        // Regression: the all-positive series overflows e^{x²} near |x|≈27; the
        // saturation branch must return exactly ±1 and never NaN/inf or >1.
        for &x in &[6.0, 15.0, 27.0, 28.0, 30.0, 100.0, 1e9]
        {
            assert_eq!(erf(x), 1.0, "erf({x}) should saturate to 1.0");
            assert_eq!(erf(-x), -1.0, "erf({}) should saturate to -1.0", -x);
        }
        // Continuity: just below the cutoff the series already yields 1.0 in f64.
        close(erf(5.9), 1.0, 1e-15);
    }

    #[test]
    fn erfc_keeps_tail_accuracy() {
        // Reference values (relative accuracy in the deep tail is the point).
        close_rel(erfc(3.0), 2.209_049_699_858_544e-5, 1e-10);
        close_rel(erfc(5.0), 1.537_459_794_428_035e-12, 1e-9);
        close_rel(erfc(6.0), 2.151_973_671_249_891e-17, 1e-8);
        close(erfc(0.0), 1.0, 1e-15);
        close_rel(erfc(-3.0), 2.0 - 2.209_049_699_858_544e-5, 1e-12);
    }

    #[test]
    fn normal_cdf_and_sf_are_consistent() {
        close(normal_cdf(0.0), 0.5, 1e-12);
        close(normal_cdf(1.0), 0.841_344_746_068_542_9, 1e-10);
        close(normal_cdf(1.96), 0.975_002_104_851_780_1, 1e-9);
        // sf is the mirror; both directly computed, product-tail should agree.
        close(normal_sf(3.0), 1.0 - normal_cdf(3.0), 1e-12);
        close_rel(normal_sf(6.0), 9.865_876_450_376_9e-10, 1e-6);
    }

    #[test]
    fn inv_normal_cdf_inverts_normal_cdf() {
        for &z in &[-3.0, -1.5, -0.3, 0.0, 0.7, 1.96, 2.5]
        {
            let p = normal_cdf(z);
            close(inv_normal_cdf(p), z, 1e-6);
        }
        close(inv_normal_cdf(0.975), 1.959_963_984_540_054, 1e-6);
    }

    #[test]
    fn chi2_quantile_matches_tables() {
        // χ²_{n; 0.95}: n=1 → 3.841, n=2 → 5.991, n=5 → 11.070, n=10 → 18.307.
        // Newton-refined ⇒ exact to table precision even at low dof.
        close(chi2_quantile(1.0, 0.95), 3.8415, 1e-3);
        close(chi2_quantile(2.0, 0.95), 5.9915, 1e-3);
        close(chi2_quantile(5.0, 0.95), 11.0705, 1e-3);
        close(chi2_quantile(10.0, 0.95), 18.3070, 1e-3);
        close(chi2_quantile(8.0, 0.99), 20.0902, 1e-3);
    }

    #[test]
    fn chi2_cdf_inverts_quantile() {
        for &(dof, p) in &[
            (1.0, 0.5),
            (3.0, 0.9),
            (5.0, 0.0027),
            (8.0, 0.99),
            (20.0, 0.5),
        ]
        {
            let x = chi2_quantile(dof, p);
            close(chi2_cdf(dof, x), p, 1e-9);
        }
        // Known point: χ²₂ CDF is 1 − e^{−x/2}; at x=2 that's 1−e⁻¹≈0.6321.
        close(chi2_cdf(2.0, 2.0), 1.0 - (-1.0f64).exp(), 1e-10);
    }

    #[test]
    fn ncchi2_cdf_reduces_to_central_and_matches_monte_carlo() {
        // λ = 0 must reproduce the central CDF exactly.
        for &(dof, x) in &[(1.0, 1.0), (4.0, 9.488), (8.0, 20.0)]
        {
            close(ncchi2_cdf(dof, 0.0, x), chi2_cdf(dof, x), 1e-12);
        }
        // Independent Monte-Carlo anchors (500k samples, seed fixed).
        close(ncchi2_cdf(4.0, 0.0, 9.488), 0.9497, 8e-3);
        close(ncchi2_cdf(2.0, 2.0, 3.0), 0.4883, 8e-3);
        close(ncchi2_cdf(4.0, 2.0, 8.0), 0.7458, 8e-3);
        close(ncchi2_cdf(5.0, 3.0, 11.0), 0.7736, 8e-3);
        // Shifting mass right: at fixed x, larger λ ⇒ smaller CDF.
        assert!(ncchi2_cdf(4.0, 5.0, 8.0) < ncchi2_cdf(4.0, 1.0, 8.0));
    }

    #[test]
    fn ncchi2_cdf_terminates_and_is_correct_for_huge_lambda() {
        // Regression: a near-degenerate σ pushes λ ~ 7e14. The naive
        // walk-outward Poisson sum would run ~1e8 iterations (a de-facto hang);
        // the bounded-window + Patnaik path must return quickly and correctly.
        // At x = mean = k+λ the CDF is ≈ 0.5; below/above it → 1 / 0.
        let lambda = 7.2e14;
        let k = 5.0;
        close(ncchi2_cdf(k, lambda, k + lambda), 0.5, 0.02);
        assert!(ncchi2_cdf(k, lambda, 0.5 * (k + lambda)) < 1e-6);
        assert!(ncchi2_cdf(k, lambda, 2.0 * (k + lambda)) > 1.0 - 1e-6);
        // Continuity across the exact-sum → Patnaik threshold (λ ≈ 1.1e6).
        close(
            ncchi2_cdf(6.0, 1.10e6, 1.12e6),
            ncchi2_cdf(6.0, 1.14e6, 1.16e6),
            2e-2,
        );
    }

    #[test]
    fn chi2_cdf_large_dof_uses_accurate_normal_branch() {
        // Wilson–Hilferty branch (dof > 20000) must agree with the exact CDF
        // just below the switch, and stay in [0,1] far above it.
        close(
            chi2_cdf(19_999.0, 20_000.0),
            chi2_cdf(20_001.0, 20_002.0),
            2e-3,
        );
        let p = chi2_cdf(1e9, 1e9); // at the mean ⇒ ≈ 0.5
        close(p, 0.5, 1e-3);
    }
}
