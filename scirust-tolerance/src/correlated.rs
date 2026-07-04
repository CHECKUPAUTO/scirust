//! Correlated and non-linear tolerance chains.
//!
//! [`crate::chain`] combines component inertias under two assumptions: the
//! relation `Y = Σ αᵢ Xᵢ` is **linear**, and the components are **independent**.
//! Real assemblies break both — a shared fixture or a common tool correlates two
//! characteristics, and a linkage or contact makes `Y` a non-linear function of
//! its parts. This module lifts each assumption.
//!
//! **Correlation.** With a correlation matrix `R` (`Rᵢⱼ = ρᵢⱼ`, `Rᵢᵢ = 1`) the
//! statistical combination becomes a quadratic form,
//!
//! ```text
//! I_Y² = Σᵢ Σⱼ αᵢ αⱼ ρᵢⱼ Iᵢ Iⱼ = (α ∘ I)ᵀ R (α ∘ I) ,
//! ```
//!
//! which reduces to [`crate::chain::assembly_inertia_statistical`]'s
//! `√(Σ αᵢ² Iᵢ²)` when `R = 𝕀`. More generally, for known component
//! covariances `Σ`, the resultant variance of any linearised response with
//! sensitivities `g` is [`correlated_variance`] `gᵀ Σ g`.
//!
//! **Non-linearity.** For a transfer closure `f`, the first-order sensitivities
//! are the gradient at the nominal point ([`gradient`], central differences),
//! and the resultant variance is `gᵀ Σ g` as above. The mean picks up a
//! **second-order** correction from the curvature ([`second_order_mean`]),
//!
//! ```text
//! E[Y] ≈ f(μ) + ½ Σᵢ Hᵢᵢ σᵢ² ,
//! ```
//!
//! with `Hᵢᵢ` the diagonal of the Hessian ([`hessian_diagonal`]). Both are
//! validated against a full [`crate::montecarlo`] simulation in the example
//! `fuzz_crosscheck`.

/// Gradient of a transfer function `f` at the nominal point `x0`, by central
/// finite differences with step `h`: `gᵢ = (f(x0 + h eᵢ) − f(x0 − h eᵢ)) / 2h`.
/// For a linear `f = Σ αᵢ xᵢ` this returns exactly the coefficients `αᵢ`.
pub fn gradient<F>(f: F, x0: &[f64], h: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut x = x0.to_vec();
    let mut g = vec![0.0; x0.len()];
    for i in 0..x0.len()
    {
        let xi = x0[i];
        x[i] = xi + h;
        let fp = f(&x);
        x[i] = xi - h;
        let fm = f(&x);
        x[i] = xi;
        g[i] = (fp - fm) / (2.0 * h);
    }
    g
}

/// Diagonal of the Hessian of `f` at `x0`, by the central second difference
/// `Hᵢᵢ = (f(x0 + h eᵢ) − 2 f(x0) + f(x0 − h eᵢ)) / h²`. The curvature terms
/// that drive the [`second_order_mean`] correction.
pub fn hessian_diagonal<F>(f: F, x0: &[f64], h: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut x = x0.to_vec();
    let f0 = f(x0);
    let mut hd = vec![0.0; x0.len()];
    for i in 0..x0.len()
    {
        let xi = x0[i];
        x[i] = xi + h;
        let fp = f(&x);
        x[i] = xi - h;
        let fm = f(&x);
        x[i] = xi;
        hd[i] = (fp - 2.0 * f0 + fm) / (h * h);
    }
    hd
}

/// Second-order estimate of the response mean for **independent** components:
/// `E[Y] ≈ f(x0) + ½ Σᵢ Hᵢᵢ σᵢ²`, where `variances[i] = σᵢ²`. Captures the bias
/// a curved transfer function introduces (a linear model would report `f(x0)`).
pub fn second_order_mean<F>(f: F, x0: &[f64], h: f64, variances: &[f64]) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let f0 = f(x0);
    let hd = hessian_diagonal(&f, x0, h);
    f0 + 0.5
        * hd.iter()
            .zip(variances)
            .map(|(hii, v)| hii * v)
            .sum::<f64>()
}

/// Resultant variance of a linearised response with sensitivities `sens` and
/// component covariance matrix `cov` (row-major, `n × n`): the quadratic form
/// `gᵀ Σ g = Σᵢ Σⱼ gᵢ gⱼ Σᵢⱼ`. For a diagonal `cov` this is the familiar
/// `Σ gᵢ² σᵢ²`. Returns 0 if the shapes are inconsistent.
pub fn correlated_variance(sens: &[f64], cov: &[f64]) -> f64 {
    let n = sens.len();
    if cov.len() != n * n
    {
        return 0.0;
    }
    let mut v = 0.0;
    for i in 0..n
    {
        for j in 0..n
        {
            v += sens[i] * sens[j] * cov[i * n + j];
        }
    }
    v
}

/// Correlated statistical assembly inertia
/// `I_Y = √(Σᵢ Σⱼ αᵢ αⱼ ρᵢⱼ Iᵢ Iⱼ)` from influence coefficients `coeffs`,
/// component inertia budgets `inertias`, and a correlation matrix `corr`
/// (row-major, `n × n`, unit diagonal).
///
/// Reduces to [`crate::chain::assembly_inertia_statistical`] for the identity
/// correlation, and to `|Σ αᵢ Iᵢ|` for full positive correlation (`ρ ≡ 1`). A
/// negative quadratic form (an inconsistent, non-PSD `corr`) is clamped to 0
/// before the square root. Returns 0 on shape mismatch.
pub fn correlated_inertia(coeffs: &[f64], inertias: &[f64], corr: &[f64]) -> f64 {
    let n = coeffs.len();
    if inertias.len() != n || corr.len() != n * n
    {
        return 0.0;
    }
    let scaled: Vec<f64> = coeffs.iter().zip(inertias).map(|(a, i)| a * i).collect();
    let mut q = 0.0;
    for i in 0..n
    {
        for j in 0..n
        {
            q += scaled[i] * scaled[j] * corr[i * n + j];
        }
    }
    q.max(0.0).sqrt()
}

/// Build an `n × n` correlation matrix (row-major) with unit diagonal and a
/// single common off-diagonal correlation `rho` — a convenient stand-in for
/// "every pair shares the same fixture/tool". `rho` is clamped to `[−1, 1]`.
pub fn uniform_correlation(n: usize, rho: f64) -> Vec<f64> {
    let rho = rho.clamp(-1.0, 1.0);
    let mut m = vec![rho; n * n];
    for i in 0..n
    {
        m[i * n + i] = 1.0;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{Contributor, assembly_inertia_statistical};
    use approx::assert_relative_eq;

    #[test]
    fn correlated_inertia_reduces_to_statistical_for_identity() {
        let coeffs = [1.0, -1.0, 0.5];
        let inertias = [0.10, 0.08, 0.20];
        let corr = uniform_correlation(3, 0.0);
        let got = correlated_inertia(&coeffs, &inertias, &corr);
        let cs: Vec<Contributor> = coeffs
            .iter()
            .zip(&inertias)
            .map(|(a, i)| Contributor::new("x", *a, *i))
            .collect();
        assert_relative_eq!(got, assembly_inertia_statistical(&cs), epsilon = 1e-12);
    }

    #[test]
    fn full_positive_correlation_is_linear_sum() {
        // ρ ≡ 1, all coeffs +1 ⇒ I_Y = Σ Iᵢ.
        let coeffs = [1.0, 1.0, 1.0];
        let inertias = [0.10, 0.08, 0.20];
        let corr = uniform_correlation(3, 1.0);
        assert_relative_eq!(
            correlated_inertia(&coeffs, &inertias, &corr),
            0.38,
            epsilon = 1e-12
        );
    }

    #[test]
    fn gradient_of_linear_is_the_coefficients() {
        let coeffs = [1.0, -2.0, 0.5, 3.0];
        let f = |x: &[f64]| coeffs.iter().zip(x).map(|(a, v)| a * v).sum::<f64>();
        let g = gradient(f, &[0.0, 1.0, -1.0, 2.0], 1e-4);
        for (gi, ci) in g.iter().zip(&coeffs)
        {
            assert_relative_eq!(gi, ci, epsilon = 1e-9);
        }
    }

    #[test]
    fn second_order_mean_is_exact_for_a_quadratic() {
        // f(x) = x², E[f] = μ² + σ²; the 2nd-order estimate is exact.
        let f = |x: &[f64]| x[0] * x[0];
        let m = second_order_mean(f, &[3.0], 1e-3, &[0.25]);
        assert_relative_eq!(m, 9.0 + 0.25, epsilon = 1e-6);
    }

    #[test]
    fn correlated_variance_reduces_to_sum_for_diagonal_cov() {
        let sens = [1.0, -1.0, 0.5];
        // diagonal covariance with σ² = 0.01, 0.04, 0.09
        let cov = [0.01, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.09];
        let want = 1.0 * 0.01 + 1.0 * 0.04 + 0.25 * 0.09;
        assert_relative_eq!(correlated_variance(&sens, &cov), want, epsilon = 1e-12);
    }

    #[test]
    fn shape_mismatch_returns_zero() {
        assert_eq!(
            correlated_inertia(&[1.0, 1.0], &[0.1], &[1.0, 0.0, 0.0, 1.0]),
            0.0
        );
        assert_eq!(correlated_variance(&[1.0, 1.0], &[1.0]), 0.0);
    }
}
