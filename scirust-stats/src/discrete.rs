//! Discrete probability distributions with a unified [`DiscreteDistribution`]
//! trait.
//!
//! Tail probabilities reuse the audited `scirust-special` primitives wherever
//! a closed identity exists — the binomial CDF through the regularized
//! incomplete beta, the Poisson CDF through the regularized incomplete gamma —
//! so no pmf summation loop is needed for the two workhorse laws. The
//! hypergeometric CDF sums its (finite, at most `draws + 1` term) support
//! directly from exact log-space pmfs.
//!
//! Conventions match SciPy: `cdf(k) = P(X ≤ k)`, `sf(k) = P(X > k)`, and
//! `quantile(p)` is the smallest `k` with `cdf(k) ≥ p`. [`Geometric`] counts
//! the number of trials up to and including the first success (support
//! `k ≥ 1`, SciPy's `geom`), not the number of failures (R's `dgeom`);
//! [`NegativeBinomial`] counts the failures before the `r`-th success
//! (SciPy's `nbinom`). [`Skellam`] lives on all of ℤ, so it exposes its own
//! `i64` methods instead of the non-negative-integer trait.

use crate::comb::{ln_binomial, ln_factorial};
use crate::rng::SplitMix64;
use scirust_special::{
    ln_beta, ln_gamma, regularized_gamma_p, regularized_gamma_q, regularized_incomplete_beta,
    riemann_zeta, riemann_zeta_tail,
};

/// A univariate distribution on the non-negative integers.
///
/// `pmf` defaults to `exp(ln_pmf)`; `quantile` defaults to a deterministic
/// bracket-and-bisect on the CDF (smallest `k` with `cdf(k) ≥ p`, the SciPy
/// `ppf` convention); `sample` draws by inverse-CDF transform from a seeded
/// [`SplitMix64`], so every draw is reproducible bit-for-bit.
pub trait DiscreteDistribution {
    /// Natural log of the probability mass at `k` (`−∞` outside the support).
    fn ln_pmf(&self, k: u64) -> f64;
    /// Cumulative distribution `P(X ≤ k)`.
    fn cdf(&self, k: u64) -> f64;
    /// Distribution mean.
    fn mean(&self) -> f64;
    /// Distribution variance.
    fn variance(&self) -> f64;

    /// Probability mass `P(X = k)`.
    fn pmf(&self, k: u64) -> f64 {
        self.ln_pmf(k).exp()
    }
    /// Survival function `P(X > k) = 1 − cdf(k)`. Override for tail accuracy.
    fn sf(&self, k: u64) -> f64 {
        1.0 - self.cdf(k)
    }
    /// Standard deviation, `sqrt(variance)`.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// Smallest `k` such that `cdf(k) ≥ p` (percent-point function).
    ///
    /// `p` is clamped to `[0, 1]`; `p ≤ cdf(0)` gives `0`. Deterministic:
    /// exponential bracket expansion, then binary search.
    fn quantile(&self, p: f64) -> u64 {
        let pt = p.clamp(0.0, 1.0);
        if self.cdf(0) >= pt
        {
            return 0;
        }
        // Expand an upper bracket from a moment-based guess. On finite
        // supports the CDF reaches exactly 1.0, so the loop terminates.
        let guess = self.mean() + 10.0 * self.std_dev();
        let mut hi: u64 = if guess.is_finite() && guess >= 1.0
        {
            guess.ceil() as u64
        }
        else
        {
            1
        };
        let mut guard = 0;
        while self.cdf(hi) < pt && guard < 200
        {
            hi = hi.saturating_mul(2);
            guard += 1;
        }
        // Smallest k in (0, hi] with cdf(k) >= pt.
        let mut lo: u64 = 0;
        while lo < hi
        {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) < pt
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        lo
    }
    /// One deterministic draw via inverse-CDF from a seeded uniform source.
    fn sample(&self, rng: &mut SplitMix64) -> u64 {
        let u = rng.next_f64().clamp(1e-15, 1.0 - 1e-15);
        self.quantile(u)
    }
}

// ============================================================ //
//  Binomial                                                    //
// ============================================================ //

/// Binomial distribution: number of successes in `n` independent trials with
/// success probability `p`.
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    /// `n` trials with success probability `p ∈ [0, 1]`.
    pub fn new(n: u64, p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Binomial: p must be within [0, 1]"
        );
        Self { n, p }
    }
}

impl DiscreteDistribution for Binomial {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k > self.n
        {
            return f64::NEG_INFINITY;
        }
        // Guard the 0·ln(0) corners so p = 0 and p = 1 stay exact.
        let t_succ = if k == 0 { 0.0 } else { k as f64 * self.p.ln() };
        let t_fail = if k == self.n
        {
            0.0
        }
        else
        {
            (self.n - k) as f64 * (-self.p).ln_1p()
        };
        ln_binomial(self.n, k) + t_succ + t_fail
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 1.0;
        }
        // P(X ≤ k) = I_{1−p}(n − k, k + 1).
        regularized_incomplete_beta((self.n - k) as f64, k as f64 + 1.0, 1.0 - self.p)
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 0.0;
        }
        // P(X > k) = I_p(k + 1, n − k) — direct form, no 1 − cdf cancellation.
        regularized_incomplete_beta(k as f64 + 1.0, (self.n - k) as f64, self.p)
    }
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

// ============================================================ //
//  Poisson                                                     //
// ============================================================ //

/// Poisson distribution: count of events at mean rate `λ > 0`.
#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Mean rate `λ > 0`.
    pub fn new(lambda: f64) -> Self {
        assert!(
            lambda > 0.0 && lambda.is_finite(),
            "Poisson: λ must be finite and > 0"
        );
        Self { lambda }
    }
}

impl DiscreteDistribution for Poisson {
    fn ln_pmf(&self, k: u64) -> f64 {
        k as f64 * self.lambda.ln() - self.lambda - ln_factorial(k)
    }
    fn cdf(&self, k: u64) -> f64 {
        // P(X ≤ k) = Q(k + 1, λ).
        regularized_gamma_q(k as f64 + 1.0, self.lambda)
    }
    fn sf(&self, k: u64) -> f64 {
        // P(X > k) = P(k + 1, λ) — direct lower tail, accurate far out.
        regularized_gamma_p(k as f64 + 1.0, self.lambda)
    }
    fn mean(&self) -> f64 {
        self.lambda
    }
    fn variance(&self) -> f64 {
        self.lambda
    }
}

// ============================================================ //
//  Hypergeometric                                              //
// ============================================================ //

/// Hypergeometric distribution: number of marked items in a sample of
/// `draws` taken *without replacement* from a population of size `population`
/// containing `successes` marked items.
///
/// This is the law that governs lottery matches (see [`crate::lottery`]),
/// acceptance sampling, and capture–recapture estimates.
#[derive(Debug, Clone, Copy)]
pub struct Hypergeometric {
    population: u64,
    successes: u64,
    draws: u64,
}

impl Hypergeometric {
    /// Population `N ≥ 1` containing `K ≤ N` marked items, sampled `n ≤ N`
    /// times without replacement.
    pub fn new(population: u64, successes: u64, draws: u64) -> Self {
        assert!(population >= 1, "Hypergeometric: population must be ≥ 1");
        assert!(
            successes <= population && draws <= population,
            "Hypergeometric: require successes ≤ population and draws ≤ population"
        );
        Self {
            population,
            successes,
            draws,
        }
    }
    /// Smallest attainable count, `max(0, draws + successes − population)`.
    pub fn support_min(&self) -> u64 {
        (self.draws + self.successes).saturating_sub(self.population)
    }
    /// Largest attainable count, `min(draws, successes)`.
    pub fn support_max(&self) -> u64 {
        self.draws.min(self.successes)
    }
}

impl DiscreteDistribution for Hypergeometric {
    fn ln_pmf(&self, k: u64) -> f64 {
        // Outside the support one of the two ln C(·,·) terms is −∞.
        if k > self.draws
        {
            return f64::NEG_INFINITY;
        }
        ln_binomial(self.successes, k)
            + ln_binomial(self.population - self.successes, self.draws - k)
            - ln_binomial(self.population, self.draws)
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.support_max()
        {
            return 1.0;
        }
        let mut acc = 0.0;
        for i in self.support_min()..=k
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }
    fn sf(&self, k: u64) -> f64 {
        // Sum the (often much shorter) upper tail directly.
        let hi = self.support_max();
        if k >= hi
        {
            return 0.0;
        }
        let mut acc = 0.0;
        for i in (k + 1).max(self.support_min())..=hi
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }
    fn mean(&self) -> f64 {
        self.draws as f64 * self.successes as f64 / self.population as f64
    }
    fn variance(&self) -> f64 {
        let (nn, kk, n) = (
            self.population as f64,
            self.successes as f64,
            self.draws as f64,
        );
        if self.population == 1
        {
            return 0.0;
        }
        n * (kk / nn) * (1.0 - kk / nn) * (nn - n) / (nn - 1.0)
    }
}

// ============================================================ //
//  Geometric                                                   //
// ============================================================ //

/// Geometric distribution: number of Bernoulli(`p`) trials up to and
/// including the first success. Support `k ≥ 1` (SciPy's `geom` convention).
#[derive(Debug, Clone, Copy)]
pub struct Geometric {
    p: f64,
}

impl Geometric {
    /// Per-trial success probability `p ∈ (0, 1]`.
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p <= 1.0, "Geometric: p must be within (0, 1]");
        Self { p }
    }
    /// `ln(1 − p)`, computed as `ln_1p(−p)` for accuracy at small `p`.
    fn ln_q(&self) -> f64 {
        (-self.p).ln_1p()
    }
}

impl DiscreteDistribution for Geometric {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0
        {
            return f64::NEG_INFINITY;
        }
        // Guard k = 1 so p = 1 avoids 0·ln(0).
        let tail = if k > 1
        {
            (k - 1) as f64 * self.ln_q()
        }
        else
        {
            0.0
        };
        self.p.ln() + tail
    }
    fn cdf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        // 1 − (1 − p)^k without cancellation.
        -(k as f64 * self.ln_q()).exp_m1()
    }
    fn sf(&self, k: u64) -> f64 {
        (k as f64 * self.ln_q()).exp()
    }
    fn mean(&self) -> f64 {
        1.0 / self.p
    }
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
}

// ============================================================ //
//  Negative binomial                                           //
// ============================================================ //

/// Negative binomial distribution: number of **failures** before the `r`-th
/// success of independent Bernoulli(`p`) trials (SciPy's `nbinom` convention;
/// R's `dnbinom` counts the same way). `r` may be real-valued (the
/// Pólya / overdispersed-Poisson parametrization used in count regression).
#[derive(Debug, Clone, Copy)]
pub struct NegativeBinomial {
    r: f64,
    p: f64,
}

impl NegativeBinomial {
    /// `r > 0` successes (possibly non-integer), per-trial success
    /// probability `p ∈ (0, 1]`.
    pub fn new(r: f64, p: f64) -> Self {
        assert!(
            r > 0.0 && r.is_finite(),
            "NegativeBinomial: r must be finite and > 0"
        );
        assert!(
            p > 0.0 && p <= 1.0,
            "NegativeBinomial: p must be within (0, 1]"
        );
        Self { r, p }
    }
}

impl DiscreteDistribution for NegativeBinomial {
    fn ln_pmf(&self, k: u64) -> f64 {
        // ln C(k + r − 1, k) generalized to real r via ln Γ.
        let kf = k as f64;
        let ln_coeff = ln_gamma(kf + self.r) - ln_gamma(self.r) - ln_factorial(k);
        // Guard k = 0 so p = 1 avoids 0·ln(0).
        let t_fail = if k == 0 { 0.0 } else { kf * (-self.p).ln_1p() };
        ln_coeff + self.r * self.p.ln() + t_fail
    }
    fn cdf(&self, k: u64) -> f64 {
        // P(X ≤ k) = I_p(r, k + 1).
        regularized_incomplete_beta(self.r, k as f64 + 1.0, self.p)
    }
    fn sf(&self, k: u64) -> f64 {
        // P(X > k) = I_{1−p}(k + 1, r) — direct upper tail.
        regularized_incomplete_beta(k as f64 + 1.0, self.r, 1.0 - self.p)
    }
    fn mean(&self) -> f64 {
        self.r * (1.0 - self.p) / self.p
    }
    fn variance(&self) -> f64 {
        self.r * (1.0 - self.p) / (self.p * self.p)
    }
}

// ============================================================ //
//  Beta-binomial                                               //
// ============================================================ //

/// Beta-binomial distribution: a Binomial(`n`, `p`) whose `p` is itself
/// Beta(`a`, `b`)-distributed — the standard model for overdispersed
/// proportions (defect rates varying batch to batch, per-site response
/// rates…). `a = b = 1` reduces to the discrete uniform on `0..=n`.
#[derive(Debug, Clone, Copy)]
pub struct BetaBinomial {
    n: u64,
    a: f64,
    b: f64,
}

impl BetaBinomial {
    /// `n` trials, Beta shape parameters `a > 0`, `b > 0`.
    pub fn new(n: u64, a: f64, b: f64) -> Self {
        assert!(
            a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite(),
            "BetaBinomial: shapes must be finite and > 0"
        );
        Self { n, a, b }
    }
}

impl DiscreteDistribution for BetaBinomial {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k > self.n
        {
            return f64::NEG_INFINITY;
        }
        let kf = k as f64;
        ln_binomial(self.n, k) + ln_beta(kf + self.a, (self.n - k) as f64 + self.b)
            - ln_beta(self.a, self.b)
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 1.0;
        }
        let mut acc = 0.0;
        for i in 0..=k
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 0.0;
        }
        // Direct upper-tail sum over the finite support.
        let mut acc = 0.0;
        for i in (k + 1)..=self.n
        {
            acc += self.pmf(i);
        }
        acc.min(1.0)
    }
    fn mean(&self) -> f64 {
        self.n as f64 * self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let (n, a, b) = (self.n as f64, self.a, self.b);
        let s = a + b;
        n * a * b * (s + n) / (s * s * (s + 1.0))
    }
}

// ============================================================ //
//  Zipfian (finite)                                            //
// ============================================================ //

/// Finite Zipfian distribution on ranks `1..=n`: `pmf(k) ∝ k^(−s)`
/// (SciPy's `zipfian`). The rank-frequency law of natural language, city
/// sizes, and access patterns; `s = 0` is the discrete uniform on `1..=n`.
///
/// The infinite-support zeta distribution (SciPy's `zipf`) needs the Riemann
/// ζ function and is deliberately not approximated here.
#[derive(Debug, Clone, Copy)]
pub struct Zipfian {
    s: f64,
    n: u64,
    /// Generalized harmonic normalizer `H(n, s) = Σ_{j=1..n} j^(−s)`,
    /// pre-summed smallest-terms-first in a fixed order (deterministic).
    h: f64,
}

impl Zipfian {
    /// Exponent `s ≥ 0` over ranks `1..=n`, `n ≥ 1`.
    pub fn new(s: f64, n: u64) -> Self {
        assert!(
            s >= 0.0 && s.is_finite(),
            "Zipfian: s must be finite and ≥ 0"
        );
        assert!(n >= 1, "Zipfian: n must be ≥ 1");
        Self {
            s,
            n,
            h: Self::harmonic(s, n),
        }
    }
    /// `H(n, s) = Σ_{j=1..n} j^(−s)` summed descending (small terms first).
    fn harmonic(s: f64, n: u64) -> f64 {
        let mut acc = 0.0;
        for j in (1..=n).rev()
        {
            acc += (j as f64).powf(-s);
        }
        acc
    }
    /// `Σ_{j=1..n} j^(power)` with the same deterministic order.
    fn power_sum(&self, power: f64) -> f64 {
        let mut acc = 0.0;
        for j in (1..=self.n).rev()
        {
            acc += (j as f64).powf(power);
        }
        acc
    }
}

impl DiscreteDistribution for Zipfian {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0 || k > self.n
        {
            return f64::NEG_INFINITY;
        }
        -self.s * (k as f64).ln() - self.h.ln()
    }
    fn pmf(&self, k: u64) -> f64 {
        if k == 0 || k > self.n
        {
            return 0.0;
        }
        (k as f64).powf(-self.s) / self.h
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 1.0;
        }
        let mut acc = 0.0;
        for j in (1..=k.min(self.n)).rev()
        {
            acc += (j as f64).powf(-self.s);
        }
        (acc / self.h).min(1.0)
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.n
        {
            return 0.0;
        }
        let mut acc = 0.0;
        for j in ((k + 1)..=self.n).rev()
        {
            acc += (j as f64).powf(-self.s);
        }
        (acc / self.h).min(1.0)
    }
    fn mean(&self) -> f64 {
        // Σ k·k^(−s) / H = H(n, s−1) / H(n, s).
        self.power_sum(1.0 - self.s) / self.h
    }
    fn variance(&self) -> f64 {
        let m = self.mean();
        self.power_sum(2.0 - self.s) / self.h - m * m
    }
}

// ============================================================ //
//  Skellam (support ℤ — outside the u64 trait)                 //
// ============================================================ //

/// Skellam distribution: the difference `X₁ − X₂` of two independent Poisson
/// counts with rates `μ₁` and `μ₂` (score differences, detector count
/// differences, queue drift…).
///
/// Its support is **all of ℤ**, so it deliberately does not implement
/// [`DiscreteDistribution`] (which lives on the non-negative integers);
/// the same method names are provided over `i64`. The pmf is evaluated by
/// the defining convolution `Σ_j pois₁(k + j)·pois₂(j)` with a fixed
/// deterministic truncation rule (stop once terms fall below 1e-18 of the
/// running peak past the summand's mode) rather than via Bessel `I_k`, so it
/// stays on the audited `scirust-special` base; accuracy vs SciPy is ~1e-12.
#[derive(Debug, Clone, Copy)]
pub struct Skellam {
    mu1: f64,
    mu2: f64,
}

impl Skellam {
    /// Rates `μ₁ > 0`, `μ₂ > 0` of the two Poisson components.
    pub fn new(mu1: f64, mu2: f64) -> Self {
        assert!(
            mu1 > 0.0 && mu1.is_finite() && mu2 > 0.0 && mu2.is_finite(),
            "Skellam: both rates must be finite and > 0"
        );
        Self { mu1, mu2 }
    }

    /// Convolution engine: `Σ_{j ≥ j0} w(j)` where `w` climbs to a single
    /// peak then decays super-exponentially. Deterministic truncation.
    fn convolve(&self, j0: u64, term: impl Fn(u64) -> f64) -> f64 {
        let mut acc = 0.0;
        let mut peak = 0.0_f64;
        let mut j = j0;
        loop
        {
            let t = term(j);
            acc += t;
            peak = peak.max(t);
            // Past the peak and negligible: stop. The +8 floor makes the
            // rule fixed for tiny rates too.
            if (t < peak * 1e-18 && j > j0 + 8) || j > j0 + 100_000
            {
                break;
            }
            j += 1;
        }
        acc
    }

    /// Probability mass `P(X₁ − X₂ = k)`, `k ∈ ℤ`.
    pub fn pmf(&self, k: i64) -> f64 {
        let p1 = Poisson::new(self.mu1);
        let p2 = Poisson::new(self.mu2);
        // X₁ = k + j, X₂ = j, j ≥ max(0, −k).
        let j0 = (-k).max(0) as u64;
        self.convolve(j0, |j| {
            (p1.ln_pmf((k + j as i64) as u64) + p2.ln_pmf(j)).exp()
        })
    }

    /// Cumulative distribution `P(X₁ − X₂ ≤ k)`.
    pub fn cdf(&self, k: i64) -> f64 {
        let p1 = Poisson::new(self.mu1);
        let p2 = Poisson::new(self.mu2);
        // Condition on X₂ = j: P(X₁ ≤ k + j); zero until k + j ≥ 0.
        let j0 = (-k).max(0) as u64;
        self.convolve(j0, |j| p2.pmf(j) * p1.cdf((k + j as i64) as u64))
            .min(1.0)
    }

    /// Survival function `P(X₁ − X₂ > k)`, summed directly (no `1 − cdf`).
    pub fn sf(&self, k: i64) -> f64 {
        let p1 = Poisson::new(self.mu1);
        let p2 = Poisson::new(self.mu2);
        // Condition on X₂ = j: P(X₁ > k + j), which is 1 until k + j ≥ 0.
        let mut acc = 0.0;
        // Terms with k + j < 0 contribute pois₂(j) whole.
        if k < 0
        {
            for j in 0..((-k) as u64)
            {
                acc += p2.pmf(j);
            }
        }
        let j0 = (-k).max(0) as u64;
        acc + self.convolve(j0, |j| p2.pmf(j) * p1.sf((k + j as i64) as u64))
    }

    /// Mean `μ₁ − μ₂`.
    pub fn mean(&self) -> f64 {
        self.mu1 - self.mu2
    }
    /// Variance `μ₁ + μ₂`.
    pub fn variance(&self) -> f64 {
        self.mu1 + self.mu2
    }
    /// Standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// One deterministic draw as the difference of two inverse-CDF Poisson
    /// draws consuming the rng in a fixed order (X₁ first, then X₂).
    pub fn sample(&self, rng: &mut SplitMix64) -> i64 {
        let x1 = Poisson::new(self.mu1).sample(rng) as i64;
        let x2 = Poisson::new(self.mu2).sample(rng) as i64;
        x1 - x2
    }
}

// ============================================================ //
//  Zeta (infinite Zipf)                                        //
// ============================================================ //

/// Zeta distribution on `k ≥ 1`: `pmf(k) = k^(−s) / ζ(s)`, `s > 1` (SciPy's
/// `zipf`). The infinite-support limit of [`Zipfian`], now expressible since
/// `scirust-special` provides `riemann_zeta`.
///
/// The far tail is handled without `ζ(s) − partial-sum` cancellation via the
/// Euler–Maclaurin tail (`riemann_zeta_tail`), so `sf`/`cdf` are O(1) for
/// `k ≥ 19` — which keeps the default bracket-and-bisect `quantile` usable
/// even in the heavy-tail regime `s ≤ 2` where the mean is infinite.
#[derive(Debug, Clone, Copy)]
pub struct Zeta {
    s: f64,
    zeta_s: f64,
}

impl Zeta {
    /// Exponent `s > 1` (the pmf is not normalizable at `s ≤ 1`).
    pub fn new(s: f64) -> Self {
        assert!(s > 1.0 && s.is_finite(), "Zeta: s must be finite and > 1");
        Self {
            s,
            zeta_s: riemann_zeta(s),
        }
    }
}

impl DiscreteDistribution for Zeta {
    fn ln_pmf(&self, k: u64) -> f64 {
        if k == 0
        {
            return f64::NEG_INFINITY;
        }
        -self.s * (k as f64).ln() - self.zeta_s.ln()
    }
    fn pmf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        (k as f64).powf(-self.s) / self.zeta_s
    }
    fn cdf(&self, k: u64) -> f64 {
        if k == 0
        {
            return 0.0;
        }
        if k < 20
        {
            // Short head: direct sum, no cancellation.
            let mut acc = 0.0;
            for j in (1..=k).rev()
            {
                acc += (j as f64).powf(-self.s);
            }
            return (acc / self.zeta_s).min(1.0);
        }
        // cdf ≈ 1 here; the tiny complement carries the accuracy.
        1.0 - self.sf(k)
    }
    fn sf(&self, k: u64) -> f64 {
        let t = if k < 19
        {
            // Tail = the few explicit terms up to 19 plus the E–M remainder.
            let mut acc = riemann_zeta_tail(self.s, 20.0);
            for j in ((k + 1)..20).rev()
            {
                acc += (j as f64).powf(-self.s);
            }
            acc
        }
        else
        {
            riemann_zeta_tail(self.s, k as f64 + 1.0)
        };
        (t / self.zeta_s).min(1.0)
    }
    fn mean(&self) -> f64 {
        if self.s > 2.0
        {
            riemann_zeta(self.s - 1.0) / self.zeta_s
        }
        else
        {
            f64::INFINITY
        }
    }
    fn variance(&self) -> f64 {
        if self.s > 3.0
        {
            let m = self.mean();
            riemann_zeta(self.s - 2.0) / self.zeta_s - m * m
        }
        else
        {
            f64::INFINITY
        }
    }
}

// ============================================================ //
//  Poisson-binomial                                            //
// ============================================================ //

/// Poisson-binomial distribution: number of successes among `n` independent
/// Bernoulli trials with **heterogeneous** probabilities `p₁ … pₙ` — the
/// exact law of "how many of these n distinct risky events occur" (system
/// reliability, portfolio defaults, per-lot defect counts).
///
/// The full mass vector is computed once at construction by the standard
/// O(n²) convolution recurrence — exact, deterministic, no FFT round-off —
/// so `pmf`/`cdf`/`sf` are table lookups afterwards.
#[derive(Debug, Clone)]
pub struct PoissonBinomial {
    mass: Vec<f64>,
    mean: f64,
    var: f64,
}

impl PoissonBinomial {
    /// Success probabilities, each in `[0, 1]`; at least one trial.
    pub fn new(probs: &[f64]) -> Self {
        assert!(
            !probs.is_empty(),
            "PoissonBinomial: need at least one trial"
        );
        assert!(
            probs.iter().all(|&p| (0.0..=1.0).contains(&p)),
            "PoissonBinomial: every probability must be within [0, 1]"
        );
        let mut mass = vec![0.0; probs.len() + 1];
        mass[0] = 1.0;
        for (i, &p) in probs.iter().enumerate()
        {
            for k in (1..=i + 1).rev()
            {
                mass[k] = mass[k] * (1.0 - p) + mass[k - 1] * p;
            }
            mass[0] *= 1.0 - p;
        }
        let mean = probs.iter().sum();
        let var = probs.iter().map(|&p| p * (1.0 - p)).sum();
        Self { mass, mean, var }
    }
    /// Number of trials `n`.
    pub fn trials(&self) -> u64 {
        (self.mass.len() - 1) as u64
    }
}

impl DiscreteDistribution for PoissonBinomial {
    fn ln_pmf(&self, k: u64) -> f64 {
        self.pmf(k).ln()
    }
    fn pmf(&self, k: u64) -> f64 {
        usize::try_from(k)
            .ok()
            .and_then(|i| self.mass.get(i))
            .copied()
            .unwrap_or(0.0)
    }
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.trials()
        {
            return 1.0;
        }
        let mut acc = 0.0;
        for i in 0..=k as usize
        {
            acc += self.mass[i];
        }
        acc.min(1.0)
    }
    fn sf(&self, k: u64) -> f64 {
        if k >= self.trials()
        {
            return 0.0;
        }
        let mut acc = 0.0;
        for i in (k as usize + 1)..self.mass.len()
        {
            acc += self.mass[i];
        }
        acc.min(1.0)
    }
    fn mean(&self) -> f64 {
        self.mean
    }
    fn variance(&self) -> f64 {
        self.var
    }
}

// ============================================================ //
//  Multinomial (vector-valued — outside the univariate trait)  //
// ============================================================ //

/// Multinomial distribution: `n` independent trials, each landing in one of
/// `m ≥ 2` categories with probabilities `p₁ … pₘ`; the outcome is the vector
/// of category counts. Vector-valued, so it exposes its own slice-based API
/// instead of the univariate [`DiscreteDistribution`] trait.
#[derive(Debug, Clone)]
pub struct Multinomial {
    n: u64,
    probs: Vec<f64>,
}

impl Multinomial {
    /// `n` trials over `probs.len() ≥ 2` categories; probabilities must be
    /// non-negative and sum to 1 within 1e-9 (they are renormalized exactly).
    pub fn new(n: u64, probs: &[f64]) -> Self {
        assert!(
            probs.len() >= 2,
            "Multinomial: need at least two categories"
        );
        assert!(
            probs.iter().all(|&p| p >= 0.0 && p.is_finite()),
            "Multinomial: probabilities must be finite and ≥ 0"
        );
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() <= 1e-9,
            "Multinomial: probabilities must sum to 1"
        );
        Self {
            n,
            probs: probs.iter().map(|&p| p / total).collect(),
        }
    }

    /// Natural log of `P(counts)`; `−∞` unless `Σ counts = n` (and every
    /// zero-probability category has a zero count). Panics if `counts` has
    /// the wrong length.
    pub fn ln_pmf(&self, counts: &[u64]) -> f64 {
        assert_eq!(
            counts.len(),
            self.probs.len(),
            "Multinomial: counts length must match the number of categories"
        );
        if counts.iter().sum::<u64>() != self.n
        {
            return f64::NEG_INFINITY;
        }
        let mut acc = ln_factorial(self.n);
        for (&k, &p) in counts.iter().zip(&self.probs)
        {
            acc -= ln_factorial(k);
            if k > 0
            {
                if p == 0.0
                {
                    return f64::NEG_INFINITY;
                }
                acc += k as f64 * p.ln();
            }
        }
        acc
    }
    /// Probability mass `P(counts)`.
    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.ln_pmf(counts).exp()
    }
    /// Mean vector `n·pᵢ`.
    pub fn mean(&self) -> Vec<f64> {
        self.probs.iter().map(|&p| self.n as f64 * p).collect()
    }
    /// Covariance matrix: `n·pᵢ(1−pᵢ)` on the diagonal, `−n·pᵢpⱼ` off it.
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        let n = self.n as f64;
        self.probs
            .iter()
            .enumerate()
            .map(|(i, &pi)| {
                self.probs
                    .iter()
                    .enumerate()
                    .map(|(j, &pj)| {
                        if i == j
                        {
                            n * pi * (1.0 - pi)
                        }
                        else
                        {
                            -n * pi * pj
                        }
                    })
                    .collect()
            })
            .collect()
    }
    /// One deterministic draw: sequential conditional binomials, one uniform
    /// consumed per category except the last (fixed order ⇒ reproducible).
    pub fn sample(&self, rng: &mut SplitMix64) -> Vec<u64> {
        let m = self.probs.len();
        let mut out = Vec::with_capacity(m);
        let mut remaining = self.n;
        let mut rest = 1.0_f64;
        for (i, &p) in self.probs.iter().enumerate()
        {
            if i + 1 == m
            {
                out.push(remaining);
                break;
            }
            let cond = if rest > 0.0
            {
                (p / rest).clamp(0.0, 1.0)
            }
            else
            {
                1.0
            };
            let k = Binomial::new(remaining, cond).sample(rng);
            out.push(k);
            remaining -= k;
            rest -= p;
        }
        out
    }
}

// ============================================================ //
//  Multivariate hypergeometric (vector-valued)                 //
// ============================================================ //

/// Multivariate hypergeometric distribution: draw `draws` items without
/// replacement from an urn holding `colors[i]` items of each of `m ≥ 2`
/// colors; the outcome is the vector of per-color counts (stratified lot
/// sampling, multi-tier lottery pools, capture panels).
#[derive(Debug, Clone)]
pub struct MultivariateHypergeometric {
    colors: Vec<u64>,
    total: u64,
    draws: u64,
}

impl MultivariateHypergeometric {
    /// Urn composition (`≥ 2` colors) and number of draws `≤ Σ colors`.
    pub fn new(colors: &[u64], draws: u64) -> Self {
        assert!(
            colors.len() >= 2,
            "MultivariateHypergeometric: need at least two colors"
        );
        let total: u64 = colors.iter().sum();
        assert!(
            draws <= total,
            "MultivariateHypergeometric: draws must not exceed the urn size"
        );
        Self {
            colors: colors.to_vec(),
            total,
            draws,
        }
    }

    /// Natural log of `P(counts)`; `−∞` unless `Σ counts = draws` with every
    /// `counts[i] ≤ colors[i]`. Panics if `counts` has the wrong length.
    pub fn ln_pmf(&self, counts: &[u64]) -> f64 {
        assert_eq!(
            counts.len(),
            self.colors.len(),
            "MultivariateHypergeometric: counts length must match the number of colors"
        );
        if counts.iter().sum::<u64>() != self.draws
        {
            return f64::NEG_INFINITY;
        }
        let mut acc = -ln_binomial(self.total, self.draws);
        for (&k, &c) in counts.iter().zip(&self.colors)
        {
            acc += ln_binomial(c, k); // −∞ when k > c
        }
        acc
    }
    /// Probability mass `P(counts)`.
    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.ln_pmf(counts).exp()
    }
    /// Mean vector `draws·colorsᵢ/total`.
    pub fn mean(&self) -> Vec<f64> {
        self.colors
            .iter()
            .map(|&c| self.draws as f64 * c as f64 / self.total as f64)
            .collect()
    }
    /// One deterministic draw: sequential conditional univariate
    /// hypergeometrics over the remaining urn (fixed order ⇒ reproducible).
    pub fn sample(&self, rng: &mut SplitMix64) -> Vec<u64> {
        let m = self.colors.len();
        let mut out = Vec::with_capacity(m);
        let mut pop = self.total;
        let mut remaining = self.draws;
        for (i, &c) in self.colors.iter().enumerate()
        {
            if i + 1 == m
            {
                out.push(remaining);
                break;
            }
            let k = Hypergeometric::new(pop.max(1), c, remaining).sample(rng);
            out.push(k);
            pop -= c;
            remaining -= k;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    // Oracle values: SciPy 1.17.1 (binom, poisson, hypergeom, geom) and exact
    // fractions via Python `math.comb` — see the values quoted inline.

    #[test]
    fn binomial_matches_scipy() {
        let b = Binomial::new(20, 0.3);
        assert!(close(b.pmf(0), 0.000_797_922_662_976_117_1, 1e-12));
        assert!(close(b.pmf(3), 0.071_603_672_205_262_1, 1e-12));
        assert!(close(b.pmf(6), 0.191_638_982_753_442_54, 1e-12));
        assert!(close(b.pmf(20), 3.486_784_400_999_997_5e-11, 1e-11));
        assert_eq!(b.pmf(21), 0.0);
        assert!(close(b.cdf(6), 0.608_009_812_200_924_4, 1e-12));
        assert!(close(b.sf(6), 0.391_990_187_799_075_6, 1e-12));
        assert!(close(b.cdf(10), 0.982_855_183_568_741_6, 1e-12));
        assert!(close(b.sf(10), 0.017_144_816_431_258_418, 1e-12));
        assert_eq!(b.quantile(0.5), 6);
        assert_eq!(b.quantile(0.95), 9);
        assert!(close(b.mean(), 6.0, 1e-15));
        assert!(close(b.variance(), 4.2, 1e-15));
        // Small-p regime (ln_1p path).
        let b2 = Binomial::new(1000, 0.001);
        assert!(close(b2.pmf(0), 0.367_695_424_770_963_9, 1e-12));
        assert!(close(b2.cdf(2), 0.919_790_657_159_799, 1e-12));
    }

    #[test]
    fn binomial_degenerate_edges() {
        let zero = Binomial::new(10, 0.0);
        assert!(close(zero.pmf(0), 1.0, 1e-15));
        assert_eq!(zero.pmf(1), 0.0);
        assert!(close(zero.cdf(0), 1.0, 1e-15));
        let one = Binomial::new(10, 1.0);
        assert!(close(one.pmf(10), 1.0, 1e-15));
        assert_eq!(one.pmf(9), 0.0);
        assert_eq!(one.quantile(0.99), 10);
        let point = Binomial::new(0, 0.5);
        assert!(close(point.pmf(0), 1.0, 1e-15));
        assert_eq!(point.quantile(0.7), 0);
    }

    #[test]
    fn poisson_matches_scipy() {
        let p = Poisson::new(4.2);
        assert!(close(p.pmf(0), 0.014_995_576_820_477_703, 1e-12));
        assert!(close(p.pmf(3), 0.185_165_382_579_258_7, 1e-12));
        assert!(close(p.pmf(7), 0.068_592_664_322_660_6, 1e-12));
        assert!(close(p.cdf(3), 0.395_403_369_602_356_17, 1e-12));
        assert!(close(p.sf(3), 0.604_596_630_397_643_8, 1e-12));
        assert!(close(p.cdf(7), 0.936_056_660_272_578_9, 1e-12));
        assert!(close(p.sf(7), 0.063_943_339_727_421_1, 1e-12));
        assert_eq!(p.quantile(0.95), 8);
        assert!(close(p.mean(), 4.2, 1e-15));
        assert!(close(p.variance(), 4.2, 1e-15));
        // cdf + sf = 1 across the range.
        for k in 0..30
        {
            assert!(close(p.cdf(k) + p.sf(k), 1.0, 1e-13), "k = {k}");
        }
    }

    #[test]
    fn hypergeometric_matches_exact_6_of_49() {
        // Classic 6/49: population 49, 6 winning numbers, player draws 6.
        // Exact fractions: pmf(k) = C(6,k)·C(43,6−k)/C(49,6).
        let h = Hypergeometric::new(49, 6, 6);
        assert!(close(h.pmf(0), 0.435_964_975_511_691_5, 1e-12));
        assert!(close(h.pmf(1), 0.413_019_450_484_760_4, 1e-12));
        assert!(close(h.pmf(2), 0.132_378_029_001_525_76, 1e-12));
        assert!(close(h.pmf(3), 0.017_650_403_866_870_102, 1e-12));
        assert!(close(h.pmf(4), 0.000_968_619_724_401_408, 1e-12));
        assert!(close(h.pmf(5), 1.844_989_951_240_777_2e-5, 1e-12));
        assert!(close(h.pmf(6), 1.0 / 13_983_816.0, 1e-12));
        assert!(close(h.cdf(2), 0.981_362_454_997_977_7, 1e-12));
        assert!(close(h.sf(2), 0.018_637_545_002_022_343, 1e-12));
        assert!(close(h.mean(), 0.734_693_877_551_020_4, 1e-13));
        assert!(close(h.variance(), 0.577_571_845_064_556_4, 1e-13));
        // Total mass is 1.
        let total: f64 = (0..=6).map(|k| h.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-13));
    }

    #[test]
    fn hypergeometric_larger_case_and_support() {
        let h = Hypergeometric::new(500, 50, 60);
        assert!(close(h.pmf(5), 0.173_200_819_493_689_73, 1e-11));
        assert!(close(h.cdf(5), 0.427_334_645_995_490_37, 1e-11));
        assert_eq!(h.quantile(0.5), 6);
        // Truncated support: draw 8 from 10 with 9 marked ⇒ at least 7 marked.
        let t = Hypergeometric::new(10, 9, 8);
        assert_eq!(t.support_min(), 7);
        assert_eq!(t.support_max(), 8);
        assert_eq!(t.pmf(6), 0.0);
        assert!(close(t.pmf(7) + t.pmf(8), 1.0, 1e-13));
    }

    #[test]
    fn geometric_matches_scipy() {
        let g = Geometric::new(0.25);
        assert_eq!(g.pmf(0), 0.0);
        assert!(close(g.pmf(1), 0.25, 1e-15));
        assert!(close(g.pmf(3), 0.140_625, 1e-15));
        assert!(close(g.pmf(8), 0.033_370_971_679_687_5, 1e-14));
        assert!(close(g.cdf(3), 0.578_125, 1e-15));
        assert!(close(g.sf(3), 0.421_875, 1e-14));
        assert_eq!(g.quantile(0.99), 17);
        assert!(close(g.mean(), 4.0, 1e-15));
        assert!(close(g.variance(), 12.0, 1e-14));
        // p = 1: certain success on the first trial.
        let sure = Geometric::new(1.0);
        assert!(close(sure.pmf(1), 1.0, 1e-15));
        assert_eq!(sure.pmf(2), 0.0);
        assert_eq!(sure.quantile(0.999), 1);
    }

    #[test]
    // Miri deliberately randomizes the last ULPs of float intrinsics, so the
    // lockstep bit-identity below cannot hold under the interpreter (same
    // rationale as the continuous sampling test in `dist`).
    #[cfg_attr(miri, ignore)]
    fn sampling_is_deterministic_and_plausible() {
        let b = Binomial::new(40, 0.35);
        let mut r1 = SplitMix64::new(42);
        let mut r2 = SplitMix64::new(42);
        let s1: Vec<u64> = (0..20_000).map(|_| b.sample(&mut r1)).collect();
        let s2: Vec<u64> = (0..20_000).map(|_| b.sample(&mut r2)).collect();
        assert_eq!(s1, s2);
        let m = s1.iter().sum::<u64>() as f64 / s1.len() as f64;
        assert!((m - 14.0).abs() < 0.1, "mean {m}");
        // Poisson sample moments near λ.
        let p = Poisson::new(6.5);
        let mut r = SplitMix64::new(7);
        let sp: Vec<u64> = (0..20_000).map(|_| p.sample(&mut r)).collect();
        let mp = sp.iter().sum::<u64>() as f64 / sp.len() as f64;
        assert!((mp - 6.5).abs() < 0.1, "mean {mp}");
    }

    #[test]
    fn negative_binomial_matches_scipy() {
        // SciPy nbinom(5, 0.4): failures before the 5th success.
        let nb = NegativeBinomial::new(5.0, 0.4);
        assert!(close(nb.pmf(0), 0.010_239_999_999_999_996, 1e-12));
        assert!(close(nb.pmf(4), 0.092_897_280_000_000_03, 1e-12));
        assert!(close(nb.pmf(10), 0.061_979_281_588_224_036, 1e-12));
        assert!(close(nb.cdf(7), 0.561_821_777_92, 1e-12));
        assert!(close(nb.sf(7), 0.438_178_222_08, 1e-12));
        assert_eq!(nb.quantile(0.5), 7);
        assert_eq!(nb.quantile(0.95), 16);
        assert!(close(nb.mean(), 7.5, 1e-14));
        assert!(close(nb.variance(), 18.75, 1e-13));
        // Real-valued r (Pólya).
        let nb2 = NegativeBinomial::new(2.5, 0.3);
        assert!(close(nb2.pmf(3), 0.110_960_031_985_585_6, 1e-12));
        assert!(close(nb2.cdf(5), 0.556_183_734_708_268_1, 1e-12));
        // r = 1 is Geometric shifted to failures: pmf(k) = p(1−p)^k.
        let nb1 = NegativeBinomial::new(1.0, 0.25);
        assert!(close(nb1.pmf(2), 0.25 * 0.75 * 0.75, 1e-14));
        // p = 1: point mass at zero failures.
        let sure = NegativeBinomial::new(3.0, 1.0);
        assert!(close(sure.pmf(0), 1.0, 1e-15));
        assert_eq!(sure.pmf(1), 0.0);
    }

    #[test]
    fn beta_binomial_matches_scipy() {
        // SciPy betabinom(10, 2, 3).
        let bb = BetaBinomial::new(10, 2.0, 3.0);
        assert!(close(bb.pmf(0), 0.065_934_065_934_065_95, 1e-12));
        assert!(close(bb.pmf(4), 0.139_860_139_860_139_76, 1e-12));
        assert!(close(bb.pmf(10), 0.010_989_010_989_010_992, 1e-12));
        assert!(close(bb.cdf(4), 0.594_405_594_405_594_4, 1e-12));
        assert!(close(bb.sf(4), 0.405_594_405_594_405_6, 1e-12));
        assert_eq!(bb.quantile(0.5), 4);
        assert!(close(bb.mean(), 4.0, 1e-14));
        assert!(close(bb.variance(), 6.0, 1e-13));
        // a = b = 1 is the discrete uniform on 0..=n.
        let u = BetaBinomial::new(6, 1.0, 1.0);
        assert!(close(u.pmf(3), 1.0 / 7.0, 1e-13));
        // Total mass 1.
        let total: f64 = (0..=10).map(|k| bb.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-13));
    }

    #[test]
    fn zipfian_matches_scipy() {
        // SciPy zipfian(1.5, 20), support 1..=20.
        let z = Zipfian::new(1.5, 20);
        assert!(close(z.pmf(1), 0.460_684_691_303_022_2, 1e-13));
        assert!(close(z.pmf(2), 0.162_876_634_604_599_42, 1e-13));
        assert!(close(z.pmf(5), 0.041_204_891_437_882_57, 1e-13));
        assert!(close(z.pmf(20), 0.005_150_611_429_735_321, 1e-13));
        assert_eq!(z.pmf(0), 0.0);
        assert_eq!(z.pmf(21), 0.0);
        assert!(close(z.cdf(5), 0.811_010_613_936_828_4, 1e-13));
        assert!(close(z.sf(5), 0.188_989_386_063_171_64, 1e-13));
        assert_eq!(z.quantile(0.5), 2);
        assert!(close(z.mean(), 3.499_017_716_693_377, 1e-13));
        assert!(close(z.variance(), 16.165_446_970_218_827, 1e-12));
        // s = 0 is the discrete uniform on 1..=n.
        let u = Zipfian::new(0.0, 10);
        assert!(close(u.pmf(7), 0.1, 1e-14));
        assert!(close(u.cdf(10), 1.0, 1e-15));
    }

    #[test]
    fn skellam_matches_scipy() {
        // SciPy skellam(3.2, 1.5).
        let s = Skellam::new(3.2, 1.5);
        assert!(close(s.pmf(-4), 0.004_693_621_474_905_621, 1e-11));
        assert!(close(s.pmf(-1), 0.086_025_279_807_399_57, 1e-11));
        assert!(close(s.pmf(0), 0.143_310_965_409_640_56, 1e-11));
        assert!(close(s.pmf(2), 0.183_382_994_925_598_12, 1e-11));
        assert!(close(s.pmf(6), 0.026_216_209_058_590_838, 1e-11));
        assert!(close(s.cdf(0), 0.291_039_386_736_692_55, 1e-11));
        assert!(close(s.cdf(3), 0.804_942_925_451_844, 1e-11));
        assert!(close(s.sf(3), 0.195_057_074_548_156_02, 1e-11));
        assert!(close(s.mean(), 1.7, 1e-14));
        assert!(close(s.variance(), 4.7, 1e-14));
        // cdf + sf = 1 across ℤ, both tails included.
        for k in -8..=10_i64
        {
            assert!(close(s.cdf(k) + s.sf(k), 1.0, 1e-12), "k = {k}");
        }
        // Equal rates ⇒ symmetric about 0.
        let sym = Skellam::new(2.0, 2.0);
        assert!(close(sym.pmf(0), 0.207_001_921_223_986_64, 1e-11));
        assert!(close(sym.pmf(1), sym.pmf(-1), 1e-13));
        assert!(close(sym.pmf(1), 0.178_750_839_502_435_3, 1e-11));
        // Total mass over a wide window is 1.
        let total: f64 = (-40..=40_i64).map(|k| s.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-11));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn skellam_sampling_is_deterministic_and_plausible() {
        let s = Skellam::new(3.2, 1.5);
        let mut r1 = SplitMix64::new(99);
        let mut r2 = SplitMix64::new(99);
        let a: Vec<i64> = (0..20_000).map(|_| s.sample(&mut r1)).collect();
        let b: Vec<i64> = (0..20_000).map(|_| s.sample(&mut r2)).collect();
        assert_eq!(a, b);
        let m = a.iter().sum::<i64>() as f64 / a.len() as f64;
        assert!((m - 1.7).abs() < 0.06, "mean {m}");
    }

    #[test]
    fn zeta_matches_scipy() {
        // SciPy zipf(2.5) — infinite-support zeta law.
        let z = Zeta::new(2.5);
        assert_eq!(z.pmf(0), 0.0);
        assert!(close(z.pmf(1), 0.745_441_296_288_777, 1e-13));
        assert!(close(z.pmf(2), 0.131_776_648_895_571_14, 1e-13));
        assert!(close(z.pmf(3), 0.047_820_081_453_043_214, 1e-13));
        assert!(close(z.pmf(10), 0.002_357_292_358_220_957_2, 1e-13));
        assert!(close(z.cdf(5), 0.961_667_926_440_313_7, 1e-13));
        assert!(close(z.sf(5), 0.038_332_073_559_686_264, 1e-13));
        assert_eq!(z.quantile(0.9), 3);
        assert_eq!(z.quantile(0.99), 14);
        // mean = ζ(1.5)/ζ(2.5); variance diverges at s ≤ 3.
        assert!(close(z.mean(), 1.947_372_466_316_956, 1e-13));
        assert!(z.variance().is_infinite());
        // s = 4: both moments finite.
        let z4 = Zeta::new(4.0);
        assert!(close(z4.pmf(2), 0.057_746_150_182_599_39, 1e-13));
        assert!(close(z4.mean(), 1.110_626_535_326_148, 1e-13));
        assert!(close(z4.variance(), 0.286_326_453_664_503_4, 1e-12));
        // cdf + sf = 1 across the head/tail split at k = 19/20.
        for k in [1u64, 5, 18, 19, 20, 50, 1000]
        {
            assert!(close(z.cdf(k) + z.sf(k), 1.0, 1e-12), "k = {k}");
        }
        // Far-tail sf stays accurate and O(1): sf(k) ~ k^(1−s)/((s−1)ζ(s)).
        let k = 1_000_000_u64;
        let approx = (k as f64).powf(-1.5) / (1.5 * 1.341_487_257_250_917_3);
        assert!(close(z.sf(k), approx, 1e-3));
    }

    #[test]
    fn poisson_binomial_matches_scipy() {
        // SciPy poisson_binom([0.1, 0.4, 0.75, 0.5, 0.9]).
        let pb = PoissonBinomial::new(&[0.1, 0.4, 0.75, 0.5, 0.9]);
        assert!(close(pb.pmf(0), 0.006_749_999_999_999_999, 1e-13));
        assert!(close(pb.pmf(1), 0.093, 1e-13));
        assert!(close(pb.pmf(2), 0.332, 1e-13));
        assert!(close(pb.pmf(3), 0.393_5, 1e-13));
        assert!(close(pb.pmf(4), 0.161_25, 1e-13));
        assert!(close(pb.pmf(5), 0.013_5, 1e-13));
        assert_eq!(pb.pmf(6), 0.0);
        assert!(close(pb.cdf(2), 0.431_75, 1e-13));
        assert!(close(pb.sf(2), 0.568_25, 1e-13));
        assert_eq!(pb.quantile(0.5), 3);
        assert!(close(pb.mean(), 2.65, 1e-14));
        assert!(close(pb.variance(), 0.857_5, 1e-14));
        assert_eq!(pb.trials(), 5);
        // Homogeneous probabilities collapse to the Binomial.
        let pb_h = PoissonBinomial::new(&[0.3; 20]);
        let b = Binomial::new(20, 0.3);
        for k in [0u64, 3, 6, 10, 20]
        {
            assert!(close(pb_h.pmf(k), b.pmf(k), 1e-12), "k = {k}");
        }
        // Total mass 1.
        let total: f64 = (0..=5).map(|k| pb.pmf(k)).sum();
        assert!(close(total, 1.0, 1e-14));
    }

    #[test]
    fn multinomial_matches_scipy() {
        // SciPy multinomial(8, [0.2, 0.3, 0.5]).
        let m = Multinomial::new(8, &[0.2, 0.3, 0.5]);
        assert!(close(m.pmf(&[2, 3, 3]), 0.075_599_999_999_999_96, 1e-12));
        assert!(close(m.pmf(&[1, 2, 5]), 0.094_500_000_000_000_2, 1e-12));
        assert!(close(m.pmf(&[8, 0, 0]), 2.560_000_000_000_001_7e-6, 1e-12));
        assert!(close(m.ln_pmf(&[2, 3, 3]), -2.582_298_995_796_650_2, 1e-12));
        // Wrong total ⇒ impossible outcome.
        assert_eq!(m.pmf(&[2, 3, 4]), 0.0);
        // Moments.
        let mean = m.mean();
        assert!(close(mean[0], 1.6, 1e-14) && close(mean[1], 2.4, 1e-14));
        let cov = m.covariance();
        assert!(close(cov[0][0], 1.28, 1e-14));
        assert!(close(cov[0][1], -0.48, 1e-14));
        assert!(close(cov[2][2], 2.0, 1e-14));
        // Zero-probability category: only zero counts allowed.
        let z = Multinomial::new(4, &[0.5, 0.5, 0.0]);
        assert_eq!(z.pmf(&[2, 1, 1]), 0.0);
        assert!(z.pmf(&[2, 2, 0]) > 0.0);
        // Two categories degenerate to the Binomial.
        let m2 = Multinomial::new(20, &[0.3, 0.7]);
        let b = Binomial::new(20, 0.3);
        assert!(close(m2.pmf(&[6, 14]), b.pmf(6), 1e-12));
    }

    #[test]
    fn multivariate_hypergeometric_matches_scipy() {
        // SciPy multivariate_hypergeom(m=[10, 5, 15], n=8);
        // exact pmf([3,1,4]) = C(10,3)·C(5,1)·C(15,4)/C(30,8) = 280/2001.
        let mh = MultivariateHypergeometric::new(&[10, 5, 15], 8);
        assert!(close(mh.pmf(&[3, 1, 4]), 280.0 / 2001.0, 1e-12));
        assert!(close(
            mh.pmf(&[0, 0, 8]),
            0.001_099_450_274_862_565_7,
            1e-12
        ));
        // Wrong total or over-drawing a color ⇒ impossible.
        assert_eq!(mh.pmf(&[3, 1, 3]), 0.0);
        assert_eq!(mh.pmf(&[0, 6, 2]), 0.0);
        let mean = mh.mean();
        assert!(close(mean[0], 8.0 / 3.0, 1e-14));
        assert!(close(mean[1], 4.0 / 3.0, 1e-14));
        assert!(close(mean[2], 4.0, 1e-14));
        // Two colors degenerate to the univariate Hypergeometric.
        let mh2 = MultivariateHypergeometric::new(&[6, 43], 6);
        let h = Hypergeometric::new(49, 6, 6);
        for k in 0..=6u64
        {
            assert!(close(mh2.pmf(&[k, 6 - k]), h.pmf(k), 1e-12), "k = {k}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn vector_sampling_is_deterministic_and_consistent() {
        // Multinomial draws always sum to n and reproduce bit-for-bit.
        let m = Multinomial::new(50, &[0.2, 0.3, 0.5]);
        let mut r1 = SplitMix64::new(11);
        let mut r2 = SplitMix64::new(11);
        let mut totals = [0u64; 3];
        for _ in 0..2_000
        {
            let a = m.sample(&mut r1);
            let b = m.sample(&mut r2);
            assert_eq!(a, b);
            assert_eq!(a.iter().sum::<u64>(), 50);
            for (t, &x) in totals.iter_mut().zip(&a)
            {
                *t += x;
            }
        }
        // Empirical means near n·p = [10, 15, 25].
        assert!((totals[0] as f64 / 2_000.0 - 10.0).abs() < 0.2);
        assert!((totals[2] as f64 / 2_000.0 - 25.0).abs() < 0.3);
        // Multivariate hypergeometric draws sum to `draws` and respect caps.
        let mh = MultivariateHypergeometric::new(&[10, 5, 15], 8);
        let mut r = SplitMix64::new(23);
        for _ in 0..2_000
        {
            let d = mh.sample(&mut r);
            assert_eq!(d.iter().sum::<u64>(), 8);
            assert!(d[0] <= 10 && d[1] <= 5 && d[2] <= 15);
        }
    }
}
