//! Deterministic, explicitly seeded random-number stream.
//!
//! Structural evolution must be bit-exactly reproducible from a seed. This
//! module provides a small, self-contained [`SplitMix64`] generator so that a
//! given seed reproduces the exact same decisions on every platform and across
//! dependency versions, independent of the `rand` crate. It never consults the
//! operating system or a thread-local generator.

/// A deterministic pseudo-random stream based on the SplitMix64 algorithm.
///
/// The generator is intentionally simple and stable: the constants and update
/// rule are fixed, so `DeterministicRng::new(seed)` yields an identical
/// sequence forever. Distinct seeds produce independent streams.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Create a stream seeded by `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next 64-bit value and advance the stream.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniformly select an index in `[0, bound)`.
    ///
    /// Returns `0` when `bound` is `0`; callers that require a non-empty range
    /// must ensure `bound > 0`. Uses Lemire's multiply-shift reduction to avoid
    /// modulo bias.
    pub fn below(&mut self, bound: usize) -> usize {
        if bound <= 1
        {
            return 0;
        }

        let product = (self.next_u64() as u128).wrapping_mul(bound as u128);
        (product >> 64) as usize
    }

    /// Draw a finite `f32` in `[-magnitude, magnitude]`.
    ///
    /// The result is always finite. A non-finite or non-positive `magnitude`
    /// is treated as `1.0` so that generated scale factors can never be
    /// non-finite.
    pub fn finite_factor(&mut self, magnitude: f32) -> f32 {
        let magnitude = if magnitude.is_finite() && magnitude > 0.0
        {
            magnitude
        }
        else
        {
            1.0
        };

        // 24 uniformly distributed bits map exactly into f32's mantissa.
        let bits = (self.next_u64() >> 40) as u32;
        let unit = bits as f32 / (1u32 << 24) as f32;
        let signed = unit.mul_add(2.0, -1.0);
        let factor = signed * magnitude;

        // Defensive clamp: the arithmetic above is finite for finite
        // `magnitude`, but guarantee the contract regardless.
        if factor.is_finite() { factor } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_seeds_reproduce_identical_streams() {
        let mut a = DeterministicRng::new(0xABCD_1234);
        let mut b = DeterministicRng::new(0xABCD_1234);

        for _ in 0..64
        {
            assert_eq!(a.below(1000), b.below(1000));
        }
    }

    #[test]
    fn distinct_seeds_diverge() {
        let mut a = DeterministicRng::new(1);
        let mut b = DeterministicRng::new(2);

        let stream_a: Vec<usize> = (0..32).map(|_| a.below(1000)).collect();
        let stream_b: Vec<usize> = (0..32).map(|_| b.below(1000)).collect();

        assert_ne!(stream_a, stream_b);
    }

    #[test]
    fn below_respects_bound() {
        let mut rng = DeterministicRng::new(7);
        for _ in 0..10_000
        {
            assert!(rng.below(5) < 5);
        }
        assert_eq!(rng.below(0), 0);
        assert_eq!(rng.below(1), 0);
    }

    #[test]
    fn finite_factor_is_always_finite_and_bounded() {
        let mut rng = DeterministicRng::new(99);
        for _ in 0..10_000
        {
            let factor = rng.finite_factor(4.0);
            assert!(factor.is_finite());
            assert!((-4.0..=4.0).contains(&factor));
        }

        // Degenerate magnitudes fall back to a finite range.
        assert!(rng.finite_factor(f32::INFINITY).is_finite());
        assert!(rng.finite_factor(f32::NAN).is_finite());
        assert!(rng.finite_factor(-1.0).is_finite());
    }
}
