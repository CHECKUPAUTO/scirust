//! Drawing many independent realisations of a stochastic capability, and
//! summarising them without holding them all in memory.
//!
//! A single seeded path answers "what does one realisation look like". Most
//! questions worth asking about a stochastic model are distributional — where
//! does the process sit on average, how wide is the spread, does the sample
//! mean converge where the theory says it should — and none of those can be
//! answered from one sample. See
//! `docs/studio/adr/0008-ensembles.md`.
//!
//! Two things in here are less obvious than they look: how the replicates'
//! seeds are derived, and why the summary is accumulated in one pass.

use scirust_sim::rng::SplitMix64;

/// The largest ensemble a scenario may request.
///
/// This is a guard against a typo, not a resource guarantee. The cost of an
/// ensemble is `replicates x steps`, and this bounds only the first factor —
/// four thousand replicates of a ten-step path is trivial, and four thousand
/// replicates of a ten-million-step path is not something this constant
/// saves anyone from. The error message says so rather than implying the
/// limit means the rest is safe.
pub const MAX_REPLICATES: u32 = 4_096;

/// How many individual realisations are kept in the result.
///
/// The ensemble's *summary* is what the run is for, and it costs one path's
/// worth of memory however many replicates were drawn. Individual paths are
/// kept only so a reader can see that the realisations genuinely differ and
/// that the band is not decoration — eight is enough to show that and few
/// enough to leave the result small and the chart legible.
///
/// Retaining a subset is a real loss of information, so it is *reported*:
/// [`EnsembleSummary::retained`] and [`EnsembleSummary::replicates`] are both
/// in the result, and the interfaces show both. A truncation nobody is told
/// about reads as "here is the ensemble".
pub const MAX_RETAINED_MEMBERS: usize = 8;

/// Derive the seeds of `count` independent replicates from one scenario seed.
///
/// # Why the seeds are scrambled rather than counted
///
/// The obvious derivation — `base`, `base + 1`, `base + 2` … — is not
/// obviously safe here, and the reason is specific to the generator this
/// project uses. [`SplitMix64`] keeps a 64-bit state that advances by a fixed
/// odd constant `y = 0x9e3779b97f4a7c15` on every draw, and `SplitMix64::new`
/// takes the seed *as* the initial state. Every seed therefore lands on the
/// same single cycle of length 2^64, at a different offset. Two replicates
/// whose seeds differ by `m * y (mod 2^64)` produce the *same noise stream*
/// shifted by `m` draws: for small `m` that is two "independent" realisations
/// sharing most of their randomness, which would quietly bias every
/// across-replicate statistic while looking perfectly fine.
///
/// Counting seeds does not actually trigger that — for a small difference
/// `d = j - i` to equal `m * y (mod 2^64)` needs a large `m` — but it makes
/// the safety of the scheme depend on an arithmetic accident rather than on
/// anything stated. Passing the base seed through SplitMix64's output mixer
/// instead makes the pairwise offsets uniform over the whole cycle, so the
/// probability that any of `n` replicates overlaps another within `L` draws
/// is about `n^2 * L / 2^64` — negligible for any ensemble this crate will
/// run, and negligible for a stated reason.
///
/// This is also what SplitMix64 was published for: expanding one seed into
/// seeds for other generators.
///
/// # Why the first replicate keeps the base seed
///
/// The first realisation uses `base` unchanged, and only the rest are drawn.
/// That buys two properties at once:
///
/// * A run of one replicate produces exactly what the same scenario produced
///   before ensembles existed. Introducing this module did not silently
///   change what `seed = 42` means, so the outputs quoted in the tutorial and
///   in `docs/studio/adr/0007-seeded-stochastic-capabilities.md` still stand.
/// * `replicate_seeds(base, 4)` is a prefix of `replicate_seeds(base, 64)`,
///   all the way down to a single replicate. Enlarging an ensemble adds
///   realisations rather than replacing the ones already reported, so the
///   member paths in a small run are the same curves as in a large one.
///
/// It costs nothing in independence: `base` sits at a uniformly-distributed
/// offset from each drawn seed on the shared cycle, which is the same
/// argument that covers the drawn seeds among themselves.
///
/// The derivation is deterministic, so an ensemble is reproducible from the
/// same single number a lone run is.
pub fn replicate_seeds(base: u64, count: usize) -> Vec<u64> {
    let mut rng = SplitMix64::new(base);
    (0..count)
        .map(|k| if k == 0 { base } else { rng.next_u64() })
        .collect()
}

/// A replicate's path had a different length from the ensemble's.
///
/// Every replicate of a run integrates the same span with the same step, so
/// this is an internal inconsistency rather than anything a scenario can
/// cause. It is returned rather than asserted so the adapter can report it as
/// an internal error with the two lengths in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthMismatch {
    /// The length the ensemble was opened with.
    pub expected: usize,
    /// The length the offending replicate had.
    pub found: usize,
}

impl std::fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "a replicate produced {} samples where the ensemble expects {}",
            self.found, self.expected
        )
    }
}

impl std::error::Error for LengthMismatch {}

/// Accumulates the across-replicate mean and variance at every axis
/// coordinate, one realisation at a time.
///
/// # Why one pass
///
/// The natural implementation collects `Vec<Vec<f64>>` and reduces at the
/// end. At the sizes an ensemble is *for* that is the wrong shape: 4 096
/// replicates of a 40 001-sample path is 1.3 GiB of `f64`, to produce two
/// vectors of 40 001. Accumulating instead costs `O(samples)` regardless of
/// the replicate count, which is what makes a large ensemble a question of
/// patience rather than of memory.
///
/// The update is Welford's, not the sum-of-squares shortcut. `E[x^2] -
/// E[x]^2` loses catastrophically when the variance is small next to the
/// mean — exactly the case here, since an OU ensemble at stationarity sits at
/// `mu` with a spread of `sigma/sqrt(2*theta)` that is routinely orders of
/// magnitude smaller. It can even return a negative variance. Welford's form
/// costs one extra subtraction per sample and cannot.
#[derive(Debug, Clone)]
pub struct EnsembleAccumulator {
    count: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl EnsembleAccumulator {
    /// Open an accumulator for paths of `samples` length.
    pub fn new(samples: usize) -> Self {
        EnsembleAccumulator {
            count: 0,
            mean: vec![0.0; samples],
            m2: vec![0.0; samples],
        }
    }

    /// Fold one realisation in.
    pub fn push(&mut self, path: &[f64]) -> Result<(), LengthMismatch> {
        if path.len() != self.mean.len()
        {
            return Err(LengthMismatch {
                expected: self.mean.len(),
                found: path.len(),
            });
        }
        self.count += 1;
        let n = self.count as f64;
        for (i, &x) in path.iter().enumerate()
        {
            let delta = x - self.mean[i];
            self.mean[i] += delta / n;
            self.m2[i] += delta * (x - self.mean[i]);
        }
        Ok(())
    }

    /// How many realisations have been folded in.
    pub fn replicates(&self) -> usize {
        self.count
    }

    /// How many samples each realisation has.
    pub fn samples(&self) -> usize {
        self.mean.len()
    }

    /// The across-replicate mean at each coordinate.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The across-replicate sample variance (`n - 1` denominator) at each
    /// coordinate.
    ///
    /// A single replicate has no spread to report, so this is all zeros
    /// rather than a division by zero.
    pub fn variance(&self) -> Vec<f64> {
        if self.count < 2
        {
            return vec![0.0; self.mean.len()];
        }
        let denom = (self.count - 1) as f64;
        // Welford's `m2` is non-negative by construction, but rounding can
        // leave a value a few ulps below zero when every replicate agreed;
        // `max(0.0)` keeps `sqrt` real without hiding anything meaningful.
        self.m2.iter().map(|&s| (s / denom).max(0.0)).collect()
    }

    /// The across-replicate standard deviation at each coordinate.
    pub fn stddev(&self) -> Vec<f64> {
        self.variance().into_iter().map(f64::sqrt).collect()
    }

    /// The band `mean -/+ sigmas * stddev`, as `(lower, upper)`.
    pub fn band(&self, sigmas: f64) -> (Vec<f64>, Vec<f64>) {
        let sd = self.stddev();
        let lower = self
            .mean
            .iter()
            .zip(&sd)
            .map(|(m, s)| m - sigmas * s)
            .collect();
        let upper = self
            .mean
            .iter()
            .zip(&sd)
            .map(|(m, s)| m + sigmas * s)
            .collect();
        (lower, upper)
    }

    /// The standard error of the ensemble mean at each coordinate,
    /// `stddev / sqrt(n)`.
    ///
    /// This is the quantity that makes an ensemble worth drawing. The
    /// replicates are independent by construction, so — unlike the
    /// autocorrelated samples along a single path — `n` realisations really
    /// do carry `n` realisations' worth of information, and no effective
    /// sample size correction applies.
    pub fn standard_error(&self) -> Vec<f64> {
        let n = (self.count.max(1)) as f64;
        self.stddev().into_iter().map(|s| s / n.sqrt()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The multiplicative inverse of `x` modulo 2^64, by Newton's iteration.
    /// Defined for odd `x`, which SplitMix64's increment is.
    fn inverse_mod_2_64(x: u64) -> u64 {
        assert!(x % 2 == 1, "only odd numbers are invertible mod 2^64");
        // `x` itself is correct to 3 bits; each step doubles the correct bits,
        // so five steps cover 64.
        let mut inv = x;
        for _ in 0..5
        {
            inv = inv.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(inv)));
        }
        debug_assert_eq!(x.wrapping_mul(inv), 1);
        inv
    }

    /// SplitMix64's fixed state increment.
    const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

    #[test]
    fn the_seeds_are_a_deterministic_function_of_the_base() {
        assert_eq!(replicate_seeds(42, 16), replicate_seeds(42, 16));
        assert_ne!(replicate_seeds(42, 16), replicate_seeds(43, 16));
    }

    /// A single-replicate run must be the run this capability already did, or
    /// adding ensembles would have quietly changed what every existing
    /// scenario's seed means.
    #[test]
    fn the_first_replicate_uses_the_base_seed_unchanged() {
        assert_eq!(replicate_seeds(42, 1), vec![42]);
        assert_eq!(replicate_seeds(42, 256)[0], 42);
    }

    /// A prefix of a longer ensemble must be the shorter ensemble, or adding
    /// replicates would silently re-draw the ones already reported.
    #[test]
    fn a_larger_ensemble_extends_a_smaller_one() {
        let few = replicate_seeds(7, 4);
        let many = replicate_seeds(7, 64);
        assert_eq!(&many[..4], &few[..]);
    }

    #[test]
    fn the_seeds_are_distinct() {
        let seeds = replicate_seeds(0, 1_000);
        let unique: std::collections::BTreeSet<_> = seeds.iter().collect();
        assert_eq!(unique.len(), seeds.len());
    }

    /// The property the derivation exists for.
    ///
    /// Two SplitMix64 streams overlap when their seeds differ by a multiple
    /// of the state increment: seeds `s` and `s + m*GAMMA` produce the same
    /// draws offset by `m`. This recovers `m` for every pair and asserts none
    /// is small enough for the streams to meet within a realistic path.
    #[test]
    fn no_two_replicate_streams_overlap_within_any_plausible_path() {
        let seeds = replicate_seeds(2_026, 256);
        let g_inv = inverse_mod_2_64(GAMMA);
        // Far beyond `MAX_REPLICATES` x any path this crate will integrate.
        const HORIZON: u64 = 1 << 40;

        let mut closest = u64::MAX;
        for (i, &a) in seeds.iter().enumerate()
        {
            for &b in &seeds[i + 1..]
            {
                // Draws separating the two streams, in either direction.
                let m = b.wrapping_sub(a).wrapping_mul(g_inv);
                let distance = m.min(m.wrapping_neg());
                closest = closest.min(distance);
            }
        }
        assert!(
            closest > HORIZON,
            "two replicate streams meet after {closest} draws, which is within a \
             path this crate could integrate; the seed derivation is not safe"
        );
    }

    /// The counting derivation this module deliberately does not use is the
    /// one this test characterises: it happens to be fine, and it is fine for
    /// a reason nobody stated. Kept so the claim in `replicate_seeds`'s
    /// documentation is checked rather than asserted.
    #[test]
    fn counting_seeds_would_have_been_safe_but_only_by_accident() {
        let g_inv = inverse_mod_2_64(GAMMA);
        let mut closest = u64::MAX;
        for d in 1u64..512
        {
            let m = d.wrapping_mul(g_inv);
            closest = closest.min(m.min(m.wrapping_neg()));
        }
        // Large — but nothing about `base + i` says it had to be.
        assert!(closest > (1 << 40));
    }

    #[test]
    fn a_constant_ensemble_has_the_constant_as_its_mean_and_no_spread() {
        let mut acc = EnsembleAccumulator::new(3);
        for _ in 0..10
        {
            acc.push(&[1.0, 2.0, 3.0]).unwrap();
        }
        assert_eq!(acc.replicates(), 10);
        assert_eq!(acc.mean(), &[1.0, 2.0, 3.0]);
        assert_eq!(acc.variance(), vec![0.0, 0.0, 0.0]);
        assert_eq!(acc.standard_error(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_mean_and_variance_match_the_textbook_definitions() {
        let mut acc = EnsembleAccumulator::new(1);
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        {
            acc.push(&[x]).unwrap();
        }
        assert!((acc.mean()[0] - 5.0).abs() < 1e-12);
        // Sample variance with the n-1 denominator: 32 / 7.
        assert!((acc.variance()[0] - 32.0 / 7.0).abs() < 1e-12);
        assert!((acc.standard_error()[0] - (32.0 / 7.0f64).sqrt() / 8.0f64.sqrt()).abs() < 1e-12);
    }

    /// The reason Welford's form is used rather than `E[x^2] - E[x]^2`.
    ///
    /// These values have a variance of 1, sitting on a mean of 1e9. The
    /// shortcut computes it as the difference of two numbers around 1e18,
    /// where an `f64` has no resolution left below about 100 — so it returns
    /// something between badly wrong and negative.
    #[test]
    fn a_small_spread_on_a_large_mean_survives() {
        let offset = 1e9;
        let mut acc = EnsembleAccumulator::new(1);
        for x in [-1.0, 0.0, 1.0]
        {
            acc.push(&[offset + x]).unwrap();
        }
        assert!((acc.mean()[0] - offset).abs() < 1e-6);
        assert!(
            (acc.variance()[0] - 1.0).abs() < 1e-6,
            "variance came out as {}",
            acc.variance()[0]
        );
    }

    #[test]
    fn one_replicate_reports_no_spread_rather_than_dividing_by_zero() {
        let mut acc = EnsembleAccumulator::new(2);
        acc.push(&[1.0, 2.0]).unwrap();
        assert_eq!(acc.variance(), vec![0.0, 0.0]);
        assert_eq!(acc.stddev(), vec![0.0, 0.0]);
        let (lo, hi) = acc.band(2.0);
        assert_eq!(lo, vec![1.0, 2.0]);
        assert_eq!(hi, vec![1.0, 2.0]);
    }

    #[test]
    fn the_band_is_symmetric_about_the_mean() {
        let mut acc = EnsembleAccumulator::new(1);
        for x in [1.0, 3.0]
        {
            acc.push(&[x]).unwrap();
        }
        let (lo, hi) = acc.band(2.0);
        assert!((acc.mean()[0] - lo[0] - (hi[0] - acc.mean()[0])).abs() < 1e-12);
    }

    #[test]
    fn a_replicate_of_the_wrong_length_is_refused_with_both_lengths() {
        let mut acc = EnsembleAccumulator::new(4);
        let err = acc.push(&[1.0, 2.0]).unwrap_err();
        assert_eq!(
            err,
            LengthMismatch {
                expected: 4,
                found: 2
            }
        );
        assert!(err.to_string().contains('4') && err.to_string().contains('2'));
        // The rejected replicate did not land.
        assert_eq!(acc.replicates(), 0);
    }

    /// The standard error is the point of the whole exercise, so its scaling
    /// is pinned: four times the replicates halves it.
    #[test]
    fn the_standard_error_falls_as_one_over_root_n() {
        let seeds = replicate_seeds(11, 1_600);
        let mut small = EnsembleAccumulator::new(1);
        let mut large = EnsembleAccumulator::new(1);
        for (k, &s) in seeds.iter().enumerate()
        {
            let x = SplitMix64::new(s).next_gaussian();
            if k < 400
            {
                small.push(&[x]).unwrap();
            }
            large.push(&[x]).unwrap();
        }
        let ratio = small.standard_error()[0] / large.standard_error()[0];
        // Exactly 2 up to the sampling error in the two variance estimates.
        assert!(
            (ratio - 2.0).abs() < 0.2,
            "standard error ratio was {ratio}, expected about 2"
        );
    }
}
