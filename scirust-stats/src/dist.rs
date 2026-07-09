//! Probability distributions with a unified [`Distribution`] trait.
//!
//! Continuous CDFs are expressed through the audited `scirust-special`
//! primitives (erf for the normal, the regularized incomplete gamma for χ²/gamma,
//! the regularized incomplete beta for Student-t / F / beta), so every tail
//! probability in the platform traces back to one validated numeric base.

use scirust_special::{
    erfc, erfinv, ln_beta, ln_gamma, regularized_gamma_p, regularized_gamma_q,
    regularized_incomplete_beta,
};

use crate::rng::SplitMix64;

use std::f64::consts::PI;

/// A univariate probability distribution.
///
/// The `sample` default draws by inverse-CDF transform of a uniform variate, so
/// any distribution with a `quantile` is sampleable deterministically from a
/// seeded [`SplitMix64`]. `sf` (survival / upper tail) defaults to `1 - cdf` but
/// is overridden where a direct form is more accurate in the far tail.
pub trait Distribution {
    /// Probability density (continuous) at `x`.
    fn pdf(&self, x: f64) -> f64;
    /// Cumulative distribution `P(X ≤ x)`.
    fn cdf(&self, x: f64) -> f64;
    /// Inverse CDF (quantile / percent-point) for `p ∈ (0, 1)`.
    fn quantile(&self, p: f64) -> f64;
    /// Distribution mean (may be `NaN`/`∞` where undefined).
    fn mean(&self) -> f64;
    /// Distribution variance (may be `NaN`/`∞` where undefined).
    fn variance(&self) -> f64;

    /// Standard deviation, `sqrt(variance)`.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// Survival function `P(X > x) = 1 − cdf(x)`. Override for tail accuracy.
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    /// One deterministic draw via inverse-CDF from a seeded uniform source.
    fn sample(&self, rng: &mut SplitMix64) -> f64 {
        // Clamp away from the open-interval endpoints so ±∞ never appears.
        let u = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
        self.quantile(u)
    }
}

/// Robust inverse of a monotone-increasing CDF by bracket-and-bisect. Fully
/// deterministic (fixed iteration budget); precise to ~1e-12 on the bracket.
fn invert_cdf(cdf: impl Fn(f64) -> f64, p: f64, mut lo: f64, mut hi: f64) -> f64 {
    // Expand the bracket outward until it straddles `p`.
    let mut guard = 0;
    while cdf(lo) > p && guard < 200
    {
        let span = (hi - lo).max(1.0);
        lo -= span;
        guard += 1;
    }
    guard = 0;
    while cdf(hi) < p && guard < 200
    {
        let span = (hi - lo).max(1.0);
        hi += span;
        guard += 1;
    }
    for _ in 0..128
    {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
        if (hi - lo).abs() <= 1e-13 * (1.0 + mid.abs())
        {
            break;
        }
    }
    0.5 * (lo + hi)
}

// ============================================================ //
//  Normal                                                      //
// ============================================================ //

/// Normal (Gaussian) distribution `N(μ, σ²)`.
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    mean: f64,
    sd: f64,
}

impl Normal {
    /// `N(μ, σ)` with standard deviation `sd > 0`.
    pub fn new(mean: f64, sd: f64) -> Self {
        assert!(sd > 0.0, "Normal: standard deviation must be > 0");
        Self { mean, sd }
    }
    /// The standard normal `N(0, 1)`.
    pub fn standard() -> Self {
        Self { mean: 0.0, sd: 1.0 }
    }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        (-0.5 * z * z).exp() / (self.sd * (2.0 * PI).sqrt())
    }
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        0.5 * erfc(-z / std::f64::consts::SQRT_2)
    }
    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        0.5 * erfc(z / std::f64::consts::SQRT_2)
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        self.mean + self.sd * std::f64::consts::SQRT_2 * erfinv(2.0 * p - 1.0)
    }
    fn mean(&self) -> f64 {
        self.mean
    }
    fn variance(&self) -> f64 {
        self.sd * self.sd
    }
}

// ============================================================ //
//  Exponential & Uniform (closed-form)                         //
// ============================================================ //

/// Exponential distribution with rate `λ > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    rate: f64,
}
impl Exponential {
    /// Rate parameter `λ > 0` (mean `1/λ`).
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0, "Exponential: rate must be > 0");
        Self { rate }
    }
}
impl Distribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0
        {
            0.0
        }
        else
        {
            self.rate * (-self.rate * x).exp()
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0
        {
            0.0
        }
        else
        {
            1.0 - (-self.rate * x).exp()
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x < 0.0 { 1.0 } else { (-self.rate * x).exp() }
    }
    fn quantile(&self, p: f64) -> f64 {
        -(1.0 - p).ln() / self.rate
    }
    fn mean(&self) -> f64 {
        1.0 / self.rate
    }
    fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }
}

/// Continuous uniform distribution on `[a, b]`.
#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    a: f64,
    b: f64,
}
impl Uniform {
    /// Uniform on `[a, b]` with `a < b`.
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a < b, "Uniform: require a < b");
        Self { a, b }
    }
}
impl Distribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b
        {
            0.0
        }
        else
        {
            1.0 / (self.b - self.a)
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a
        {
            0.0
        }
        else if x >= self.b
        {
            1.0
        }
        else
        {
            (x - self.a) / (self.b - self.a)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        self.a + p.clamp(0.0, 1.0) * (self.b - self.a)
    }
    fn mean(&self) -> f64 {
        0.5 * (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let d = self.b - self.a;
        d * d / 12.0
    }
}

// ============================================================ //
//  Gamma & Chi-squared                                         //
// ============================================================ //

/// Gamma distribution with shape `k > 0` and scale `θ > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    shape: f64,
    scale: f64,
}
impl Gamma {
    /// Shape `k > 0`, scale `θ > 0` (mean `kθ`).
    pub fn new(shape: f64, scale: f64) -> Self {
        assert!(
            shape > 0.0 && scale > 0.0,
            "Gamma: shape, scale must be > 0"
        );
        Self { shape, scale }
    }
}
impl Distribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            return 0.0;
        }
        let k = self.shape;
        let t = self.scale;
        ((k - 1.0) * x.ln() - x / t - k * t.ln() - ln_gamma(k)).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            0.0
        }
        else
        {
            regularized_gamma_p(self.shape, x / self.scale)
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            1.0
        }
        else
        {
            regularized_gamma_q(self.shape, x / self.scale)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 0.0;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        let hi = self.mean() + 12.0 * self.std_dev();
        invert_cdf(|x| self.cdf(x), p, 0.0, hi.max(1.0))
    }
    fn mean(&self) -> f64 {
        self.shape * self.scale
    }
    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }
}

/// Chi-squared distribution with `k > 0` degrees of freedom.
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared {
    k: f64,
}
impl ChiSquared {
    /// `k` degrees of freedom (`k > 0`).
    pub fn new(k: f64) -> Self {
        assert!(k > 0.0, "ChiSquared: degrees of freedom must be > 0");
        Self { k }
    }
}
impl Distribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        // χ²(k) is Gamma(k/2, 2).
        Gamma::new(self.k / 2.0, 2.0).pdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            0.0
        }
        else
        {
            regularized_gamma_p(self.k / 2.0, x / 2.0)
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            1.0
        }
        else
        {
            regularized_gamma_q(self.k / 2.0, x / 2.0)
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 0.0;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        let hi = self.mean() + 12.0 * self.std_dev();
        invert_cdf(|x| self.cdf(x), p, 0.0, hi.max(1.0))
    }
    fn mean(&self) -> f64 {
        self.k
    }
    fn variance(&self) -> f64 {
        2.0 * self.k
    }
}

// ============================================================ //
//  Student-t & Fisher-F & Beta                                 //
// ============================================================ //

/// Student's t distribution with `ν > 0` degrees of freedom.
#[derive(Debug, Clone, Copy)]
pub struct StudentT {
    nu: f64,
}
impl StudentT {
    /// `ν` degrees of freedom (`ν > 0`).
    pub fn new(nu: f64) -> Self {
        assert!(nu > 0.0, "StudentT: degrees of freedom must be > 0");
        Self { nu }
    }
}
impl Distribution for StudentT {
    fn pdf(&self, t: f64) -> f64 {
        let nu = self.nu;
        let ln_norm = ln_gamma((nu + 1.0) / 2.0) - ln_gamma(nu / 2.0) - 0.5 * (nu * PI).ln();
        (ln_norm - (nu + 1.0) / 2.0 * (1.0 + t * t / nu).ln()).exp()
    }
    fn cdf(&self, t: f64) -> f64 {
        // I_{ν/(ν+t²)}(ν/2, 1/2) gives the two-tailed mass; split by sign.
        let nu = self.nu;
        let x = nu / (nu + t * t);
        let ib = 0.5 * regularized_incomplete_beta(nu / 2.0, 0.5, x);
        if t >= 0.0 { 1.0 - ib } else { ib }
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        invert_cdf(|t| self.cdf(t), p, -100.0, 100.0)
    }
    fn mean(&self) -> f64 {
        if self.nu > 1.0 { 0.0 } else { f64::NAN }
    }
    fn variance(&self) -> f64 {
        if self.nu > 2.0
        {
            self.nu / (self.nu - 2.0)
        }
        else
        {
            f64::INFINITY
        }
    }
}

/// Fisher–Snedecor F distribution with `(d1, d2)` degrees of freedom.
#[derive(Debug, Clone, Copy)]
pub struct FisherF {
    d1: f64,
    d2: f64,
}
impl FisherF {
    /// Numerator `d1 > 0` and denominator `d2 > 0` degrees of freedom.
    pub fn new(d1: f64, d2: f64) -> Self {
        assert!(d1 > 0.0 && d2 > 0.0, "FisherF: both dof must be > 0");
        Self { d1, d2 }
    }
}
impl Distribution for FisherF {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            return 0.0;
        }
        let (d1, d2) = (self.d1, self.d2);
        // ln pdf = (d1/2)ln(d1/d2) + (d1/2−1)ln x − ((d1+d2)/2)ln(1+d1 x/d2) − lnB(d1/2,d2/2)
        let ln = (d1 / 2.0) * (d1 / d2).ln() + (d1 / 2.0 - 1.0) * x.ln()
            - (d1 + d2) / 2.0 * (1.0 + d1 * x / d2).ln()
            - ln_beta(d1 / 2.0, d2 / 2.0);
        ln.exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            return 0.0;
        }
        let (d1, d2) = (self.d1, self.d2);
        let y = d1 * x / (d1 * x + d2);
        regularized_incomplete_beta(d1 / 2.0, d2 / 2.0, y)
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0
        {
            return 1.0;
        }
        let (d1, d2) = (self.d1, self.d2);
        // Complement via the symmetry of the incomplete beta.
        let y = d2 / (d1 * x + d2);
        regularized_incomplete_beta(d2 / 2.0, d1 / 2.0, y)
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 0.0;
        }
        if p >= 1.0
        {
            return f64::INFINITY;
        }
        invert_cdf(|x| self.cdf(x), p, 0.0, 100.0)
    }
    fn mean(&self) -> f64 {
        if self.d2 > 2.0
        {
            self.d2 / (self.d2 - 2.0)
        }
        else
        {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        let (d1, d2) = (self.d1, self.d2);
        if d2 > 4.0
        {
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0).powi(2) * (d2 - 4.0))
        }
        else
        {
            f64::NAN
        }
    }
}

/// Beta distribution on `[0, 1]` with shapes `a, b > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Beta {
    a: f64,
    b: f64,
}
impl Beta {
    /// Shapes `a > 0`, `b > 0`.
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a > 0.0 && b > 0.0, "Beta: shapes must be > 0");
        Self { a, b }
    }
}
impl Distribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0
        {
            return 0.0;
        }
        ((self.a - 1.0) * x.ln() + (self.b - 1.0) * (1.0 - x).ln() - ln_beta(self.a, self.b)).exp()
    }
    fn cdf(&self, x: f64) -> f64 {
        regularized_incomplete_beta(self.a, self.b, x)
    }
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0
        {
            return 0.0;
        }
        if p >= 1.0
        {
            return 1.0;
        }
        invert_cdf(|x| self.cdf(x), p, 0.0, 1.0)
    }
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let s = self.a + self.b;
        self.a * self.b / (s * s * (s + 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn normal_matches_reference() {
        let n = Normal::standard();
        assert!(close(n.cdf(0.0), 0.5, 1e-12));
        assert!(close(n.cdf(1.96), 0.975_002_104_851_780, 1e-10));
        assert!(close(n.quantile(0.975), 1.959_963_984_540_054, 1e-9));
        // pdf peak = 1/√(2π).
        assert!(close(n.pdf(0.0), 1.0 / (2.0 * PI).sqrt(), 1e-13));
        // round-trip cdf∘quantile.
        for &p in &[0.01, 0.3, 0.5, 0.84, 0.999]
        {
            assert!(close(n.cdf(n.quantile(p)), p, 1e-9), "p = {p}");
        }
        // shifted/scaled
        let m = Normal::new(10.0, 2.0);
        assert!(close(m.cdf(10.0), 0.5, 1e-12));
        assert!(close(m.std_dev(), 2.0, 1e-15));
    }

    #[test]
    fn student_t_matches_reference() {
        let t = StudentT::new(10.0);
        assert!(close(t.cdf(0.0), 0.5, 1e-12));
        // t_{0.975, 10} = 2.228138852...
        assert!(close(t.quantile(0.975), 2.228_138_851_986_273, 1e-6));
        assert!(close(t.cdf(2.228_138_851_986_273), 0.975, 1e-9));
        // symmetry
        assert!(close(t.cdf(-1.5), 1.0 - t.cdf(1.5), 1e-12));
        // large ν → normal
        let big = StudentT::new(1e6);
        assert!(close(big.cdf(1.96), Normal::standard().cdf(1.96), 1e-4));
    }

    #[test]
    fn chi_squared_matches_reference() {
        let c = ChiSquared::new(5.0);
        assert!(close(c.mean(), 5.0, 1e-15));
        assert!(close(c.variance(), 10.0, 1e-15));
        // χ²_{0.95, 5} = 11.0704976935...
        assert!(close(c.quantile(0.95), 11.070_497_693_516_35, 1e-7));
        assert!(close(c.cdf(11.070_497_693_516_35), 0.95, 1e-9));
        // sf + cdf = 1
        assert!(close(c.cdf(7.0) + c.sf(7.0), 1.0, 1e-14));
    }

    #[test]
    fn fisher_f_matches_reference() {
        let f = FisherF::new(5.0, 10.0);
        // F_{0.95}(5,10) = 3.325834529...
        assert!(close(f.quantile(0.95), 3.325_834_529_923_105, 1e-5));
        assert!(close(f.cdf(3.325_834_529_923_105), 0.95, 1e-8));
        assert!(close(f.cdf(2.0) + f.sf(2.0), 1.0, 1e-12));
        assert!(close(f.mean(), 10.0 / 8.0, 1e-13));
    }

    #[test]
    fn gamma_beta_exponential_uniform() {
        // Gamma(k=1) is Exponential(1/scale).
        let g = Gamma::new(1.0, 2.0);
        let e = Exponential::new(0.5);
        assert!(close(g.cdf(3.0), e.cdf(3.0), 1e-12));
        assert!(close(e.quantile(0.5), 2.0 * 2.0_f64.ln(), 1e-12));
        // Beta(2,2) is symmetric about 0.5.
        let b = Beta::new(2.0, 2.0);
        assert!(close(b.cdf(0.5), 0.5, 1e-12));
        assert!(close(b.mean(), 0.5, 1e-15));
        // Uniform
        let u = Uniform::new(2.0, 6.0);
        assert!(close(u.cdf(4.0), 0.5, 1e-15));
        assert!(close(u.mean(), 4.0, 1e-15));
        assert!(close(u.variance(), 16.0 / 12.0, 1e-14));
    }

    #[test]
    fn sampling_is_deterministic_and_plausible() {
        let n = Normal::new(5.0, 2.0);
        let mut r1 = SplitMix64::new(123);
        let mut r2 = SplitMix64::new(123);
        let s: Vec<f64> = (0..50_000).map(|_| n.sample(&mut r1)).collect();
        // reproducible
        for _ in 0..50_000
        {
            // consume r2 in lockstep — bit identical
        }
        let mut r2b = SplitMix64::new(123);
        for &x in s.iter().take(1000)
        {
            let y = n.sample(&mut r2b);
            assert_eq!(x.to_bits(), y.to_bits());
        }
        let _ = r2.next_f64();
        // sample mean/var near the true values
        let m: f64 = s.iter().sum::<f64>() / s.len() as f64;
        let v: f64 = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (s.len() - 1) as f64;
        assert!((m - 5.0).abs() < 0.05, "mean {m}");
        assert!((v - 4.0).abs() < 0.15, "var {v}");
    }
}
