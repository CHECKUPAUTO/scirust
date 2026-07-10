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
//! `k ≥ 1`, SciPy's `geom`), not the number of failures (R's `dgeom`).

use crate::comb::{ln_binomial, ln_factorial};
use crate::rng::SplitMix64;
use scirust_special::{regularized_gamma_p, regularized_gamma_q, regularized_incomplete_beta};

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
}
