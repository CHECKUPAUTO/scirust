//! Deterministic pseudo-random numbers for the stochastic models: the
//! [`SplitMix64`] generator (Vigna's published algorithm), uniform and
//! Gaussian variates.
//!
//! There is no ambient randomness anywhere in this crate — every stochastic
//! model takes an explicit `seed: u64`, in line with the workspace-wide
//! reproducibility convention, so a simulation is a pure function of its
//! inputs.

/// SplitMix64 pseudo-random generator (Steele, Lea & Flood; Vigna's public
/// domain reference implementation), validated against the published output
/// sequence in the tests.
///
/// Statistically solid for simulation purposes, one `u64` of state, and the
/// same generator the rest of the workspace already uses inline to seed
/// synthetic test data. Not cryptographic.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitMix64 {
    state: u64,
    /// Spare Gaussian variate from the last Box–Muller pair.
    gauss_spare: Option<f64>,
}

impl SplitMix64 {
    /// Create a generator from an explicit seed. Equal seeds yield equal
    /// output sequences; any seed (including 0) is valid.
    pub fn new(seed: u64) -> Self {
        SplitMix64 {
            state: seed,
            gauss_spare: None,
        }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform variate in `[0, 1)` with the full 53 bits of `f64` precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal variate (mean 0, variance 1) by the Box–Muller
    /// transform; the second variate of each pair is cached, so consecutive
    /// calls consume uniforms two at a time.
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(z) = self.gauss_spare.take()
        {
            return z;
        }
        // u1 in (0, 1] so that ln(u1) is finite.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        self.gauss_spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    /// Exponential variate with the given rate (mean `1/rate`), by inversion.
    ///
    /// Returns `None` when `rate` is not finite and positive.
    pub fn next_exponential(&mut self, rate: f64) -> Option<f64> {
        if !rate.is_finite() || rate <= 0.0
        {
            return None;
        }
        // 1 - U is in (0, 1] so the logarithm is finite.
        Some(-(1.0 - self.next_f64()).ln() / rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_published_reference_sequence() {
        // Oracle: Vigna's splitmix64.c reference, cross-checked with an
        // independent Python implementation of the published algorithm.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
        assert_eq!(rng.next_u64(), 0xf88b_b8a8_724c_81ec);

        let mut rng = SplitMix64::new(1_234_567);
        assert_eq!(rng.next_u64(), 0x599e_d017_fb08_fc85);
        assert_eq!(rng.next_u64(), 0x2c73_f084_5854_0fa5);
    }

    #[test]
    fn same_seed_same_sequence_different_seed_different_sequence() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        let mut c = SplitMix64::new(43);
        let seq_a: Vec<u64> = (0..100).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..100).map(|_| b.next_u64()).collect();
        let seq_c: Vec<u64> = (0..100).map(|_| c.next_u64()).collect();
        assert_eq!(seq_a, seq_b);
        assert_ne!(seq_a, seq_c);
    }

    #[test]
    fn uniforms_stay_in_unit_interval_and_fill_it() {
        let mut rng = SplitMix64::new(7);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for _ in 0..10_000
        {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u));
            lo = lo.min(u);
            hi = hi.max(u);
        }
        assert!(lo < 0.01 && hi > 0.99, "range [{lo}, {hi}] too narrow");
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn gaussian_moments_match_standard_normal() {
        let mut rng = SplitMix64::new(2024);
        let n = 20_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn exponential_mean_matches_rate_and_bad_rates_are_rejected() {
        let mut rng = SplitMix64::new(99);
        let n = 20_000;
        let rate = 2.5;
        let mean = (0..n)
            .map(|_| rng.next_exponential(rate).unwrap())
            .sum::<f64>()
            / f64::from(n);
        assert!((mean - 1.0 / rate).abs() < 0.01, "mean {mean}");
        assert_eq!(rng.next_exponential(0.0), None);
        assert_eq!(rng.next_exponential(-1.0), None);
        assert_eq!(rng.next_exponential(f64::NAN), None);
    }
}
