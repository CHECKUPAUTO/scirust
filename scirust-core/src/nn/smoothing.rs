//! **Certified robustness via randomized smoothing** (Cohen, Rosenfeld & Kolter,
//! ICML 2019, arXiv:1902.02918).
//!
//! Randomized smoothing turns *any* base classifier `f` into a **smoothed**
//! classifier `g(x) = argmax_c P(f(x + ε) = c)`, `ε ~ N(0, σ²I)`, that comes with
//! a **provable L2 robustness radius**: if the top class `A` has probability
//! `pₐ ≥ ½`, then `g` is constant on the ball `‖δ‖₂ ≤ σ·Φ⁻¹(pₐ)`. Because `pₐ` is
//! estimated by Monte-Carlo, the certificate is made rigorous with a **lower
//! confidence bound** on `pₐ` — the one-sided **Clopper–Pearson** bound — so the
//! radius holds with probability `≥ 1 − α` over the sampling. This is exactly
//! scirust's "certifiable AI" thesis: a robustness guarantee you can *test*.
//!
//! Everything is deterministic given the seeded [`PcgEngine`]: the Gaussian
//! samples, the vote counts, and the special-function evaluations (`Φ⁻¹` by
//! Acklam's rational approximation; the Clopper–Pearson bound by the regularized
//! incomplete beta) are all pure `f64`/`f32` in a fixed order.

use crate::nn::PcgEngine;

// ----- Special functions (f64 internally for accuracy) -------------------------

/// Natural log of the Gamma function (Lanczos approximation, ~1e-10 relative).
fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000000000190015;
    for &c in COF.iter()
    {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Continued-fraction core of the incomplete beta (Numerical Recipes `betacf`).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 3.0e-13;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN
    {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT
    {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN
        {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN
        {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN
        {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN
        {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS
        {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `Iₓ(a, b)` — the CDF of the Beta(a,b) law.
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0
    {
        return 0.0;
    }
    if x >= 1.0
    {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0)
    {
        bt * betacf(a, b, x) / a
    }
    else
    {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Inverse of [`betai`] in `x`: the `p`-quantile of Beta(a,b), by bisection
/// (monotone CDF, so 60 steps give ~`2⁻⁶⁰` precision).
fn beta_inv(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0
    {
        return 0.0;
    }
    if p >= 1.0
    {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..60
    {
        let mid = 0.5 * (lo + hi);
        if betai(a, b, mid) < p
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

/// One-sided **Clopper–Pearson** lower confidence bound on a binomial proportion:
/// given `k` successes in `n` trials, returns `p` such that `P(success) ≥ p` with
/// confidence `1 − alpha`. Exact (binomial), via the Beta quantile
/// `p = Betaᵢₙᵥ(alpha; k, n−k+1)`; `k = 0 ⇒ 0`.
pub fn clopper_pearson_lower(k: usize, n: usize, alpha: f64) -> f64 {
    assert!(k <= n, "clopper_pearson_lower: k must be ≤ n");
    assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0,1)");
    if k == 0
    {
        return 0.0;
    }
    beta_inv(alpha, k as f64, (n - k + 1) as f64)
}

/// Inverse standard-normal CDF (probit `Φ⁻¹`), Acklam's rational approximation
/// (~1.15e-9 absolute error over `(0,1)`).
pub fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if p <= 0.0
    {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0
    {
        return f64::INFINITY;
    }
    if p < P_LOW
    {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
    else if p <= 1.0 - P_LOW
    {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
    else
    {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ----- Smoothed classifier -----------------------------------------------------

/// Outcome of [`SmoothedClassifier::certify`].
#[derive(Clone, Copy, Debug)]
pub struct Certificate {
    /// The predicted (top-vote) class.
    pub class: usize,
    /// Certified L2 robustness radius (`0` when abstaining).
    pub radius: f32,
    /// The Clopper–Pearson lower bound on the top class's probability.
    pub p_a_lower: f32,
    /// `true` when the bound could not clear `½` (no certificate).
    pub abstained: bool,
}

/// A smoothed classifier wrapping a base classifier under Gaussian noise `σ`.
pub struct SmoothedClassifier {
    sigma: f32,
}

impl SmoothedClassifier {
    /// New smoothed classifier at noise level `sigma > 0`.
    pub fn new(sigma: f32) -> Self {
        assert!(sigma > 0.0 && sigma.is_finite(), "sigma must be positive");
        Self { sigma }
    }

    /// Monte-Carlo class vote counts from `n` Gaussian samples around `x`.
    fn vote_counts<F: Fn(&[f32]) -> usize>(
        &self,
        f: &F,
        x: &[f32],
        n: usize,
        num_classes: usize,
        rng: &mut PcgEngine,
    ) -> Vec<usize> {
        let mut counts = vec![0usize; num_classes];
        let mut buf = vec![0.0f32; x.len()];
        for _ in 0..n
        {
            for (b, &xi) in buf.iter_mut().zip(x)
            {
                *b = xi + rng.normal(0.0, self.sigma);
            }
            counts[f(&buf)] += 1;
        }
        counts
    }

    /// Index of the largest count (ties → lower index, deterministic).
    fn argmax(counts: &[usize]) -> usize {
        let mut best = 0usize;
        for (i, &c) in counts.iter().enumerate().skip(1)
        {
            if c > counts[best]
            {
                best = i;
            }
        }
        best
    }

    /// The smoothed prediction `g(x)` from `n` samples.
    pub fn predict<F: Fn(&[f32]) -> usize>(
        &self,
        f: &F,
        x: &[f32],
        n: usize,
        num_classes: usize,
        rng: &mut PcgEngine,
    ) -> usize {
        Self::argmax(&self.vote_counts(f, x, n, num_classes, rng))
    }

    /// **Certify** at `x`: estimate the top class from `n` samples, lower-bound its
    /// probability with Clopper–Pearson at confidence `1 − alpha`, and return the
    /// certified L2 radius `σ·Φ⁻¹(pₐ)` (or abstain if `pₐ ≤ ½`). This is the
    /// single-sample variant (selection and estimation from one batch); it is exact
    /// for the binary case. The radius holds w.p. `≥ 1 − alpha` over the sampling.
    pub fn certify<F: Fn(&[f32]) -> usize>(
        &self,
        f: &F,
        x: &[f32],
        n: usize,
        num_classes: usize,
        alpha: f64,
        rng: &mut PcgEngine,
    ) -> Certificate {
        let counts = self.vote_counts(f, x, n, num_classes, rng);
        let class = Self::argmax(&counts);
        let p_a_lower = clopper_pearson_lower(counts[class], n, alpha);
        if p_a_lower <= 0.5
        {
            Certificate {
                class,
                radius: 0.0,
                p_a_lower: p_a_lower as f32,
                abstained: true,
            }
        }
        else
        {
            let radius = self.sigma * inv_normal_cdf(p_a_lower) as f32;
            Certificate {
                class,
                radius,
                p_a_lower: p_a_lower as f32,
                abstained: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Φ⁻¹` at landmark points: median 0, the 97.5 % point ≈ 1.959964, the
    /// 84.1345 % point ≈ 1, and odd symmetry `Φ⁻¹(p) = −Φ⁻¹(1−p)`.
    #[test]
    fn inv_normal_cdf_landmarks() {
        assert!(
            inv_normal_cdf(0.5).abs() < 1e-9,
            "Φ⁻¹(0.5) = {}",
            inv_normal_cdf(0.5)
        );
        assert!((inv_normal_cdf(0.975) - 1.959964).abs() < 1e-4);
        assert!((inv_normal_cdf(0.8413447) - 1.0).abs() < 1e-4);
        assert!((inv_normal_cdf(0.1) + inv_normal_cdf(0.9)).abs() < 1e-6);
    }

    /// The regularized incomplete beta is a CDF: `I₀ = 0`, `I₁ = 1`, monotone, and
    /// `I_{0.5}(2,2) = 0.5` (Beta(2,2) is symmetric about ½).
    #[test]
    fn betai_is_a_cdf() {
        assert_eq!(betai(2.0, 3.0, 0.0), 0.0);
        assert_eq!(betai(2.0, 3.0, 1.0), 1.0);
        assert!((betai(2.0, 2.0, 0.5) - 0.5).abs() < 1e-9);
        assert!(betai(2.0, 5.0, 0.3) < betai(2.0, 5.0, 0.6)); // monotone
    }

    /// Clopper–Pearson lower bound is below the point estimate, inverts `betai`
    /// (`I_{p_L}(k, n−k+1) = alpha`), and matches a known value: for 50/100 at 95 %
    /// the one-sided lower bound is ≈ 0.416.
    #[test]
    fn clopper_pearson_bound_is_sound() {
        let p_l = clopper_pearson_lower(50, 100, 0.05);
        assert!(p_l < 0.5 && p_l > 0.0, "p_l = {p_l}");
        assert!((p_l - 0.416).abs() < 0.01, "CP(50,100,0.05) = {p_l}");
        // Inversion identity: I_{p_l}(k, n-k+1) ≈ alpha.
        assert!((betai(50.0, 51.0, p_l) - 0.05).abs() < 1e-6);
        // k = 0 ⇒ 0; more successes ⇒ larger lower bound.
        assert_eq!(clopper_pearson_lower(0, 100, 0.05), 0.0);
        assert!(clopper_pearson_lower(90, 100, 0.05) > clopper_pearson_lower(60, 100, 0.05));
    }

    /// **The randomized-smoothing certificate, tested against an exact closed
    /// form.** For a half-space classifier `f(x) = 1[x₀ > 0]` at a point whose
    /// signed distance to the boundary is `d`, the population radius is *exactly*
    /// `d` (since `pₐ = Φ(d/σ)` ⇒ `σ·Φ⁻¹(pₐ) = d`). The Monte-Carlo + Clopper–
    /// Pearson estimate recovers `d` (a touch below, being a conservative lower
    /// bound), independent of `σ`. Deterministic given the seed.
    #[test]
    fn certified_radius_matches_halfspace_distance() {
        let halfspace = |z: &[f32]| -> usize { usize::from(z[0] > 0.0) };
        let n = 20000;
        let alpha = 0.001;

        let certify_at = |d: f32, sigma: f32| -> Certificate {
            let mut rng = PcgEngine::new(7);
            let smc = SmoothedClassifier::new(sigma);
            smc.certify(&halfspace, &[d, 0.0, 0.0], n, 2, alpha, &mut rng)
        };

        // Radius ≈ d (conservative ⇒ within [0.8 d, 1.02 d]); class is 1.
        for &d in &[0.5f32, 1.0, 1.5]
        {
            let c = certify_at(d, 1.0);
            assert_eq!(c.class, 1);
            assert!(!c.abstained);
            assert!(
                c.radius > 0.8 * d && c.radius < 1.02 * d,
                "d={d}: radius {} not ≈ d",
                c.radius
            );
        }
        // σ-invariance: the certified radius tracks d, not σ.
        let r_lo = certify_at(1.0, 0.5).radius;
        let r_hi = certify_at(1.0, 2.0).radius;
        assert!(r_lo > 0.8 && r_lo < 1.02, "σ=0.5: radius {r_lo}");
        assert!(r_hi > 0.8 && r_hi < 1.02, "σ=2.0: radius {r_hi}");

        // Determinism: same seed ⇒ identical certificate.
        assert_eq!(certify_at(1.0, 1.0).radius, certify_at(1.0, 1.0).radius);
    }

    /// Deep inside the safe region the smoothed classifier is confident and the
    /// certificate is sound (the prediction is the true class); right on the
    /// boundary it abstains (`pₐ ≈ ½` cannot be certified).
    #[test]
    fn certify_soundness_and_abstention() {
        let halfspace = |z: &[f32]| -> usize { usize::from(z[0] > 0.0) };
        let smc = SmoothedClassifier::new(1.0);

        // Far from the boundary: certified, class 1, positive radius.
        let mut rng = PcgEngine::new(3);
        let far = smc.certify(&halfspace, &[2.5, 0.0], 5000, 2, 0.01, &mut rng);
        assert_eq!(far.class, 1);
        assert!(!far.abstained && far.radius > 0.0);

        // On the boundary (d = 0): pₐ ≈ 0.5, cannot clear ½ ⇒ abstain.
        let mut rng = PcgEngine::new(3);
        let edge = smc.certify(&halfspace, &[0.0, 0.0], 5000, 2, 0.01, &mut rng);
        assert!(edge.abstained && edge.radius == 0.0);
    }
}
