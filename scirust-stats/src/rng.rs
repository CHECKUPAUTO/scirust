//! A small, seeded, deterministic uniform random source for distribution
//! sampling — no external crates, no global state, reproducible across runs and
//! platforms (consistent with SciRust's bit-exact discipline).

/// SplitMix64: a fast, well-distributed seeded generator (Steele et al., 2014).
/// Used only as a uniform `[0, 1)` source that distributions transform via their
/// inverse CDF, so the entire sampling path is deterministic given the seed.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from a fixed seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next `f64` uniformly in `[0, 1)` with 53 bits of resolution.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits → [0, 1); the classic `/ 2^53` construction.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_in_unit_interval_and_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000
        {
            let x = a.next_f64();
            assert!((0.0..1.0).contains(&x));
            assert_eq!(x.to_bits(), b.next_f64().to_bits());
        }
    }

    #[test]
    fn mean_is_near_one_half() {
        let mut r = SplitMix64::new(7);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| r.next_f64()).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean = {mean}");
    }
}
