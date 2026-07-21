//! Deterministic robust descriptive statistics.
//!
//! These are the order-statistic and trimming primitives that a robust
//! scientific/industrial pipeline needs and that later phases of the SRCC robust
//! structural-intelligence program (scale-aware geometry, robust regression,
//! contamination models, benchmarking) build on. Everything here is a location or
//! scale summary that resists a *minority* of grossly aberrant observations —
//! never a claim of robustness to an arbitrary majority.
//!
//! # What's here
//!
//! - [`median_absolute_deviation`] — the canonical robust scale, optionally scaled
//!   to be normal-consistent with the standard deviation.
//! - [`interquartile_range`] — the `Q3 − Q1` robust scale, using the exact same
//!   `type-7` quantile convention as [`crate::describe::quantile`].
//! - [`weighted_median`] — the deterministic weighted 0.5-quantile.
//! - [`trimmed_mean`] / [`winsorized_mean`] — symmetric-trimming robust locations.
//! - [`median_of_means`] — a block estimator with either contiguous or a
//!   deterministic seeded-permutation partition.
//!
//! # Reuse, not reinvention
//!
//! The median and the quantile rule come from [`crate::describe`]; this module
//! does not define a second, incompatible median or quantile. Any pseudo-random
//! partitioning uses the crate's seeded [`crate::rng::SplitMix64`] — there is no
//! hidden global RNG.
//!
//! # Breakdown limitations
//!
//! The median, MAD and IQR have a 50 % breakdown point *in the limit*; a symmetric
//! `α`-trimmed or `α`-winsorized mean tolerates up to a fraction `α` of
//! contamination on each tail. None of these estimators can recover a signal once
//! a numerical majority of the sample is adversarial — that requires the explicit
//! identifiability assumptions introduced in a later phase, not a descriptive
//! statistic. The median-of-means guarantee, in particular, is contingent on the
//! contamination touching only a minority of blocks; a contiguous partition is
//! additionally sensitive to the order in which samples are supplied.
//!
//! # Errors and non-finite input
//!
//! Every entry point validates its input and returns a typed
//! [`RobustStatsError`]: an empty slice, a non-finite value, an invalid weight or
//! trim fraction, or an out-of-range block count is reported explicitly rather
//! than being silently turned into a `NaN`.

use core::fmt;

use crate::describe::{mean, median, quantile};
use crate::rng::SplitMix64;

/// Normal-consistency scaling factor for the MAD: `1 / Φ⁻¹(3/4)`, the multiplier
/// that makes the median absolute deviation a consistent estimator of the
/// standard deviation of a normal sample. Numerically `≈ 1.4826`. It is applied
/// only when [`MadConsistency::Normal`] is requested — never silently. A unit test
/// cross-checks this literal against the crate's own audited normal quantile.
const NORMAL_CONSISTENCY: f64 = 1.482_602_218_505_602;

/// Errors returned by the robust descriptive statistics in this module.
///
/// The [`f64`] payloads mean this type is deliberately only [`PartialEq`], not
/// [`Eq`]; match on the variant (or use [`matches!`]) when a payload may be `NaN`.
#[derive(Clone, Debug, PartialEq)]
pub enum RobustStatsError {
    /// The input slice was empty.
    EmptyInput,
    /// A sample value was not finite (`NaN` or `±∞`).
    NonFiniteValue {
        /// Index of the offending sample.
        index: usize,
        /// The non-finite value encountered.
        value: f64,
    },
    /// A weight was not finite (`NaN` or `±∞`).
    NonFiniteWeight {
        /// Index of the offending weight.
        index: usize,
        /// The non-finite weight encountered.
        weight: f64,
    },
    /// A weight was negative.
    NegativeWeight {
        /// Index of the offending weight.
        index: usize,
        /// The negative weight encountered.
        weight: f64,
    },
    /// The weights summed to a non-positive total, so no median is defined.
    ZeroTotalWeight,
    /// `values` and `weights` had different lengths.
    LengthMismatch {
        /// Number of values supplied.
        values: usize,
        /// Number of weights supplied.
        weights: usize,
    },
    /// The trim fraction was not in the half-open interval `[0, 0.5)` (this also
    /// rejects non-finite fractions).
    InvalidTrimFraction {
        /// The rejected trim fraction.
        fraction: f64,
    },
    /// The requested block count was zero.
    InvalidBlockCount {
        /// The rejected block count.
        block_count: usize,
    },
    /// More blocks were requested than there are samples, which would create at
    /// least one empty block.
    TooManyBlocks {
        /// The requested block count.
        block_count: usize,
        /// The available sample count.
        sample_count: usize,
    },
}

impl fmt::Display for RobustStatsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyInput => formatter.write_str("input slice is empty"),
            Self::NonFiniteValue { index, value } =>
            {
                write!(formatter, "sample {index} is not finite (value = {value})")
            },
            Self::NonFiniteWeight { index, weight } =>
            {
                write!(
                    formatter,
                    "weight {index} is not finite (weight = {weight})"
                )
            },
            Self::NegativeWeight { index, weight } =>
            {
                write!(formatter, "weight {index} is negative (weight = {weight})")
            },
            Self::ZeroTotalWeight => formatter
                .write_str("weights sum to a non-positive total; weighted median undefined"),
            Self::LengthMismatch { values, weights } => write!(
                formatter,
                "values length {values} does not match weights length {weights}"
            ),
            Self::InvalidTrimFraction { fraction } =>
            {
                write!(formatter, "trim fraction {fraction} is not in [0, 0.5)")
            },
            Self::InvalidBlockCount { block_count } =>
            {
                write!(formatter, "block count {block_count} must be at least 1")
            },
            Self::TooManyBlocks {
                block_count,
                sample_count,
            } => write!(
                formatter,
                "block count {block_count} exceeds sample count {sample_count}"
            ),
        }
    }
}

impl std::error::Error for RobustStatsError {}

/// How the [`median_absolute_deviation`] result is scaled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MadConsistency {
    /// Return the raw MAD, `median(|xᵢ − median(x)|)`, with no scaling.
    Raw,
    /// Scale by `1 / Φ⁻¹(3/4) ≈ 1.4826` so the result is a consistent estimator of
    /// the standard deviation for normally distributed data.
    Normal,
}

/// How [`median_of_means`] assigns samples to blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MedianOfMeansPartition {
    /// Contiguous blocks in the input order. Deterministic and allocation-light,
    /// but sensitive to the order in which samples are supplied.
    Contiguous,
    /// A seeded pseudo-random permutation (deterministic [`SplitMix64`]
    /// Fisher–Yates) applied before contiguous blocking, decoupling the blocks
    /// from the input order for a fixed seed.
    SeededPermutation,
}

/// Configuration for [`median_of_means`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MedianOfMeansConfig {
    /// Number of blocks the sample is partitioned into. Must be at least `1` and
    /// at most the sample count (no empty blocks).
    pub block_count: usize,
    /// Seed for [`MedianOfMeansPartition::SeededPermutation`]. Ignored by
    /// [`MedianOfMeansPartition::Contiguous`].
    pub seed: u64,
    /// The partition strategy.
    pub partition: MedianOfMeansPartition,
}

/// The median absolute deviation, `median(|xᵢ − median(x)|)`.
///
/// This is the canonical robust scale estimator: it has a 50 % breakdown point in
/// the limit, so a minority of arbitrarily large outliers cannot inflate it. With
/// [`MadConsistency::Normal`] the raw MAD is multiplied by
/// [`NORMAL_CONSISTENCY`](self) so that, for normally distributed data, it
/// estimates the standard deviation.
///
/// The center and the deviation-median are computed with [`crate::describe::median`]
/// — the same median used everywhere else in the crate.
///
/// # Errors
///
/// [`RobustStatsError::EmptyInput`] for an empty slice, or
/// [`RobustStatsError::NonFiniteValue`] if any sample is `NaN` or `±∞`.
pub fn median_absolute_deviation(
    values: &[f64],
    consistency: MadConsistency,
) -> Result<f64, RobustStatsError> {
    ensure_finite_nonempty(values)?;
    let center = median(values);
    let deviations: Vec<f64> = values.iter().map(|&x| (x - center).abs()).collect();
    let raw = median(&deviations);
    Ok(match consistency
    {
        MadConsistency::Raw => raw,
        MadConsistency::Normal => raw * NORMAL_CONSISTENCY,
    })
}

/// The interquartile range, `Q3 − Q1`.
///
/// The quartiles use [`crate::describe::quantile`] (the linear-interpolated
/// `type-7` rule), so this shares one quantile convention with the rest of the
/// crate rather than introducing a second, incompatible one. A constant or
/// single-element sample has an IQR of exactly `0`.
///
/// # Errors
///
/// [`RobustStatsError::EmptyInput`] for an empty slice, or
/// [`RobustStatsError::NonFiniteValue`] if any sample is `NaN` or `±∞`.
pub fn interquartile_range(values: &[f64]) -> Result<f64, RobustStatsError> {
    ensure_finite_nonempty(values)?;
    Ok(quantile(values, 0.75) - quantile(values, 0.25))
}

/// The weighted median: the value at which the cumulative weight first reaches
/// half of the total weight.
///
/// Ties among equal values are broken by original index, so the ordering — and
/// therefore the result — is fully deterministic. Weights are accumulated in a
/// single fixed order (ascending value, index tie-break) so the computation is
/// reproducible bit-for-bit.
///
/// Behaviour at exactly 50 %: if the cumulative weight equals half the total
/// exactly at some element, the half-weight boundary lies between that element and
/// the next in sorted order, and their arithmetic midpoint is returned (the
/// lower/upper weighted-median average). Otherwise the unique crossing element is
/// returned. With non-dyadic weights the exact-equality branch is, as always,
/// subject to floating-point representation.
///
/// # Errors
///
/// - [`RobustStatsError::EmptyInput`] for an empty slice.
/// - [`RobustStatsError::LengthMismatch`] if `values` and `weights` differ in
///   length.
/// - [`RobustStatsError::NonFiniteValue`] / [`RobustStatsError::NonFiniteWeight`]
///   for a non-finite sample or weight.
/// - [`RobustStatsError::NegativeWeight`] for a negative weight.
/// - [`RobustStatsError::ZeroTotalWeight`] if the weights sum to zero.
pub fn weighted_median(values: &[f64], weights: &[f64]) -> Result<f64, RobustStatsError> {
    if values.is_empty()
    {
        return Err(RobustStatsError::EmptyInput);
    }
    if values.len() != weights.len()
    {
        return Err(RobustStatsError::LengthMismatch {
            values: values.len(),
            weights: weights.len(),
        });
    }
    for (index, &value) in values.iter().enumerate()
    {
        if !value.is_finite()
        {
            return Err(RobustStatsError::NonFiniteValue { index, value });
        }
    }
    for (index, &weight) in weights.iter().enumerate()
    {
        if !weight.is_finite()
        {
            return Err(RobustStatsError::NonFiniteWeight { index, weight });
        }
        if weight < 0.0
        {
            return Err(RobustStatsError::NegativeWeight { index, weight });
        }
    }

    // Canonical order: ascending value, ties broken by original index.
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));

    // Total accumulated in the same order the running sum uses below, so the two
    // are bit-identical and the exact-half comparison is meaningful.
    let total: f64 = order.iter().map(|&i| weights[i]).sum();
    if total <= 0.0
    {
        return Err(RobustStatsError::ZeroTotalWeight);
    }
    let half = total / 2.0;

    let mut cumulative = 0.0_f64;
    for position in 0..order.len()
    {
        cumulative += weights[order[position]];
        if cumulative > half
        {
            return Ok(values[order[position]]);
        }
        if cumulative == half
        {
            // `total > half`, so positive weight remains and `position + 1` is a
            // valid sorted index: average this element with the next one.
            let next = order[position + 1];
            return Ok((values[order[position]] + values[next]) / 2.0);
        }
    }

    // Unreachable: the running sum equals `total > half` at the last element, so
    // the strict-crossing branch above has already returned. Kept as a total,
    // deterministic fallback rather than a panic.
    Ok(values[order[order.len() - 1]])
}

/// The symmetric `α`-trimmed mean: discard the `⌊n·α⌋` smallest and `⌊n·α⌋`
/// largest values, then average the rest.
///
/// The trim count uses `floor`, so no fractional sample is ever partially removed
/// or interpolated. Because `α < 0.5`, at least one value always survives.
///
/// # Errors
///
/// [`RobustStatsError::EmptyInput`], [`RobustStatsError::NonFiniteValue`], or
/// [`RobustStatsError::InvalidTrimFraction`] if `trim_fraction` is not in
/// `[0, 0.5)`.
pub fn trimmed_mean(values: &[f64], trim_fraction: f64) -> Result<f64, RobustStatsError> {
    ensure_finite_nonempty(values)?;
    let count = floor_trim_count(values.len(), trim_fraction)?;
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let kept = &sorted[count..values.len() - count];
    Ok(mean(kept))
}

/// The symmetric `α`-winsorized mean: clamp the `⌊n·α⌋` smallest values up to the
/// smallest retained value and the `⌊n·α⌋` largest down to the largest retained
/// value, then average all `n` clamped values.
///
/// This uses the exact same `floor` trim convention as [`trimmed_mean`]; the
/// difference is that winsorizing *replaces* the extremes rather than dropping
/// them, so every sample still contributes.
///
/// # Errors
///
/// [`RobustStatsError::EmptyInput`], [`RobustStatsError::NonFiniteValue`], or
/// [`RobustStatsError::InvalidTrimFraction`] if `trim_fraction` is not in
/// `[0, 0.5)`.
pub fn winsorized_mean(values: &[f64], trim_fraction: f64) -> Result<f64, RobustStatsError> {
    ensure_finite_nonempty(values)?;
    let count = floor_trim_count(values.len(), trim_fraction)?;
    let n = values.len();
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let lower = sorted[count];
    let upper = sorted[n - 1 - count];
    for value in sorted.iter_mut().take(count)
    {
        *value = lower;
    }
    for value in sorted.iter_mut().skip(n - count)
    {
        *value = upper;
    }
    Ok(mean(&sorted))
}

/// The median-of-means estimator: partition the sample into `block_count` blocks,
/// take each block's mean, and return the median of those block means.
///
/// With [`MedianOfMeansPartition::Contiguous`] the blocks are consecutive runs in
/// the input order (the first `n mod block_count` blocks take one extra sample),
/// which makes the result sensitive to input order. With
/// [`MedianOfMeansPartition::SeededPermutation`] a deterministic
/// [`SplitMix64`]-driven Fisher–Yates shuffle of the indices is applied first, so
/// for a fixed seed the block contents no longer depend on the caller's ordering.
///
/// Median-of-means only sharpens a location estimate when contamination is
/// confined to a minority of blocks; it is not a route to majority-corruption
/// robustness, and the choice of `block_count` trades bias against that tolerance.
///
/// # Errors
///
/// [`RobustStatsError::EmptyInput`], [`RobustStatsError::NonFiniteValue`],
/// [`RobustStatsError::InvalidBlockCount`] if `block_count == 0`, or
/// [`RobustStatsError::TooManyBlocks`] if `block_count` exceeds the sample count.
pub fn median_of_means(
    values: &[f64],
    config: MedianOfMeansConfig,
) -> Result<f64, RobustStatsError> {
    ensure_finite_nonempty(values)?;
    let n = values.len();
    if config.block_count == 0
    {
        return Err(RobustStatsError::InvalidBlockCount { block_count: 0 });
    }
    if config.block_count > n
    {
        return Err(RobustStatsError::TooManyBlocks {
            block_count: config.block_count,
            sample_count: n,
        });
    }
    let order: Vec<usize> = match config.partition
    {
        MedianOfMeansPartition::Contiguous => (0..n).collect(),
        MedianOfMeansPartition::SeededPermutation => seeded_permutation(n, config.seed),
    };
    let means = block_means(values, &order, config.block_count);
    Ok(median(&means))
}

/// Validate that `values` is non-empty and entirely finite.
fn ensure_finite_nonempty(values: &[f64]) -> Result<(), RobustStatsError> {
    if values.is_empty()
    {
        return Err(RobustStatsError::EmptyInput);
    }
    for (index, &value) in values.iter().enumerate()
    {
        if !value.is_finite()
        {
            return Err(RobustStatsError::NonFiniteValue { index, value });
        }
    }
    Ok(())
}

/// Validate a trim fraction and return the symmetric per-tail trim count `⌊n·α⌋`.
/// The range check on `[0, 0.5)` also rejects any non-finite fraction.
fn floor_trim_count(n: usize, trim_fraction: f64) -> Result<usize, RobustStatsError> {
    if !(0.0..0.5).contains(&trim_fraction)
    {
        return Err(RobustStatsError::InvalidTrimFraction {
            fraction: trim_fraction,
        });
    }
    Ok((n as f64 * trim_fraction).floor() as usize)
}

/// A deterministic Fisher–Yates permutation of `0..n` seeded by `SplitMix64`.
///
/// Indices are drawn with `next_u64() % (i + 1)`; the modulo introduces a
/// negligible, fully deterministic bias that does not affect the median-of-means
/// estimate and keeps the whole path reproducible for a fixed seed.
fn seeded_permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(seed);
    let mut i = n;
    while i > 1
    {
        i -= 1;
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// The mean of each contiguous block of `order`, gathering values through the
/// `order` index map. Block sizes are as even as possible: the first
/// `n mod block_count` blocks take one extra element. Every block is non-empty
/// because `block_count ≤ n` is enforced by the caller. Each block mean uses the
/// same `sum / count` convention as [`crate::describe::mean`].
fn block_means(values: &[f64], order: &[usize], block_count: usize) -> Vec<f64> {
    let n = order.len();
    let base = n / block_count;
    let remainder = n % block_count;
    let mut means = Vec::with_capacity(block_count);
    let mut start = 0;
    for block in 0..block_count
    {
        let size = base + usize::from(block < remainder);
        let end = start + size;
        let sum: f64 = order[start..end].iter().map(|&i| values[i]).sum();
        means.push(sum / size as f64);
        start = end;
    }
    means
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{Distribution, Normal};

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual} (tol {tol})"
        );
    }

    #[test]
    fn mad_raw_hand_computed() {
        // median = 2; |x - 2| = [1,1,0,0,2,4,7]; median of those = 1.
        let data = [1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0];
        let raw = median_absolute_deviation(&data, MadConsistency::Raw).unwrap();
        assert_close(raw, 1.0, 1e-12);
    }

    #[test]
    fn mad_normal_scales_by_documented_factor() {
        let data = [1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0];
        let raw = median_absolute_deviation(&data, MadConsistency::Raw).unwrap();
        let normal = median_absolute_deviation(&data, MadConsistency::Normal).unwrap();
        assert_close(normal, raw * NORMAL_CONSISTENCY, 1e-12);
    }

    #[test]
    fn mad_normal_factor_matches_crate_normal_quantile() {
        // The normal-consistency factor is, by definition, 1 / Φ⁻¹(3/4). Cross-check
        // the literal against the crate's own audited normal quantile.
        let phi_inv_three_quarter = Normal::standard().quantile(0.75);
        assert_close(NORMAL_CONSISTENCY, 1.0 / phi_inv_three_quarter, 1e-9);
    }

    #[test]
    fn iqr_hand_computed() {
        // type-7 quantiles of [1,2,3,4,5]: Q1 = 2, Q3 = 4, IQR = 2.
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_close(interquartile_range(&data).unwrap(), 2.0, 1e-12);
    }

    #[test]
    fn iqr_is_zero_for_constant_sample() {
        assert_close(interquartile_range(&[7.0, 7.0, 7.0]).unwrap(), 0.0, 1e-12);
        assert_close(interquartile_range(&[42.0]).unwrap(), 0.0, 1e-12);
    }

    #[test]
    fn weighted_median_equal_weights_equals_median() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = [1.0; 5];
        assert_close(weighted_median(&values, &weights).unwrap(), 3.0, 1e-12);
    }

    #[test]
    fn weighted_median_exact_half_averages_neighbours() {
        // total = 4, half = 2, cumulative hits 2 exactly after the second value.
        let values = [1.0, 2.0, 3.0, 4.0];
        let weights = [1.0; 4];
        assert_close(weighted_median(&values, &weights).unwrap(), 2.5, 1e-12);
    }

    #[test]
    fn weighted_median_weight_shifts_result() {
        let values = [1.0, 2.0, 3.0];
        let weights = [1.0, 1.0, 5.0];
        assert_close(weighted_median(&values, &weights).unwrap(), 3.0, 1e-12);
    }

    #[test]
    fn weighted_median_ignores_zero_weight_outlier() {
        // The huge value carries no weight, so it must not move the estimate.
        let values = [1.0, 2.0, 3.0, 1.0e12];
        let weights = [1.0, 1.0, 1.0, 0.0];
        assert_close(weighted_median(&values, &weights).unwrap(), 2.0, 1e-12);
    }

    #[test]
    fn trimmed_mean_discards_extremes() {
        // n = 5, α = 0.2 ⇒ trim 1 each side ⇒ mean of [2,3,4] = 3.
        let data = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_close(trimmed_mean(&data, 0.2).unwrap(), 3.0, 1e-12);
    }

    #[test]
    fn trimmed_mean_zero_fraction_equals_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_close(trimmed_mean(&data, 0.0).unwrap(), mean(&data), 1e-12);
    }

    #[test]
    fn winsorized_mean_clamps_extremes() {
        // n = 5, α = 0.2 ⇒ clamp to [2, 4] ⇒ mean of [2,2,3,4,4] = 3.
        let data = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_close(winsorized_mean(&data, 0.2).unwrap(), 3.0, 1e-12);
    }

    #[test]
    fn winsorized_mean_zero_fraction_equals_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_close(winsorized_mean(&data, 0.0).unwrap(), mean(&data), 1e-12);
    }

    #[test]
    fn median_of_means_contiguous_hand_computed() {
        // 3 blocks of 2 ⇒ block means [1.5, 3.5, 52.5] ⇒ median 3.5.
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let config = MedianOfMeansConfig {
            block_count: 3,
            seed: 0,
            partition: MedianOfMeansPartition::Contiguous,
        };
        assert_close(median_of_means(&data, config).unwrap(), 3.5, 1e-12);
    }

    #[test]
    fn median_of_means_uneven_blocks() {
        // n = 7, 3 blocks ⇒ sizes [3,2,2] ⇒ means [2, 4.5, 6.5] ⇒ median 4.5.
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = MedianOfMeansConfig {
            block_count: 3,
            seed: 0,
            partition: MedianOfMeansPartition::Contiguous,
        };
        assert_close(median_of_means(&data, config).unwrap(), 4.5, 1e-12);
    }

    #[test]
    fn median_of_means_seeded_is_reproducible() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let config = MedianOfMeansConfig {
            block_count: 7,
            seed: 0xC0FF_EE00,
            partition: MedianOfMeansPartition::SeededPermutation,
        };
        let first = median_of_means(&data, config).unwrap();
        let second = median_of_means(&data, config).unwrap();
        assert_eq!(first.to_bits(), second.to_bits());
    }

    #[test]
    fn seeded_permutation_is_a_permutation() {
        let order = seeded_permutation(11, 12345);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..11).collect::<Vec<_>>());
    }

    #[test]
    fn empty_input_is_rejected() {
        let empty: [f64; 0] = [];
        assert_eq!(
            median_absolute_deviation(&empty, MadConsistency::Raw),
            Err(RobustStatsError::EmptyInput)
        );
        assert_eq!(
            interquartile_range(&empty),
            Err(RobustStatsError::EmptyInput)
        );
        assert_eq!(trimmed_mean(&empty, 0.1), Err(RobustStatsError::EmptyInput));
        assert_eq!(
            winsorized_mean(&empty, 0.1),
            Err(RobustStatsError::EmptyInput)
        );
        assert_eq!(
            weighted_median(&empty, &empty),
            Err(RobustStatsError::EmptyInput)
        );
        let config = MedianOfMeansConfig {
            block_count: 1,
            seed: 0,
            partition: MedianOfMeansPartition::Contiguous,
        };
        assert_eq!(
            median_of_means(&empty, config),
            Err(RobustStatsError::EmptyInput)
        );
    }

    #[test]
    fn non_finite_value_is_reported_with_index() {
        let data = [1.0, f64::NAN, 3.0];
        assert!(matches!(
            median_absolute_deviation(&data, MadConsistency::Raw),
            Err(RobustStatsError::NonFiniteValue { index: 1, .. })
        ));
        let data = [1.0, 2.0, f64::INFINITY];
        assert!(matches!(
            interquartile_range(&data),
            Err(RobustStatsError::NonFiniteValue { index: 2, .. })
        ));
    }

    #[test]
    fn weighted_median_rejects_bad_weights() {
        assert!(matches!(
            weighted_median(&[1.0, 2.0], &[1.0]),
            Err(RobustStatsError::LengthMismatch {
                values: 2,
                weights: 1
            })
        ));
        assert!(matches!(
            weighted_median(&[1.0, 2.0], &[1.0, f64::NAN]),
            Err(RobustStatsError::NonFiniteWeight { index: 1, .. })
        ));
        assert!(matches!(
            weighted_median(&[1.0, 2.0], &[1.0, -1.0]),
            Err(RobustStatsError::NegativeWeight { index: 1, .. })
        ));
        assert_eq!(
            weighted_median(&[1.0, 2.0], &[0.0, 0.0]),
            Err(RobustStatsError::ZeroTotalWeight)
        );
    }

    #[test]
    fn invalid_trim_fraction_is_rejected() {
        for bad in [0.5, 0.6, -0.1, f64::NAN, f64::INFINITY]
        {
            assert!(matches!(
                trimmed_mean(&[1.0, 2.0, 3.0], bad),
                Err(RobustStatsError::InvalidTrimFraction { .. })
            ));
            assert!(matches!(
                winsorized_mean(&[1.0, 2.0, 3.0], bad),
                Err(RobustStatsError::InvalidTrimFraction { .. })
            ));
        }
    }

    #[test]
    fn median_of_means_rejects_bad_block_counts() {
        let data = [1.0, 2.0, 3.0];
        let zero = MedianOfMeansConfig {
            block_count: 0,
            seed: 0,
            partition: MedianOfMeansPartition::Contiguous,
        };
        assert_eq!(
            median_of_means(&data, zero),
            Err(RobustStatsError::InvalidBlockCount { block_count: 0 })
        );
        let too_many = MedianOfMeansConfig {
            block_count: 4,
            seed: 0,
            partition: MedianOfMeansPartition::Contiguous,
        };
        assert_eq!(
            median_of_means(&data, too_many),
            Err(RobustStatsError::TooManyBlocks {
                block_count: 4,
                sample_count: 3
            })
        );
    }

    #[test]
    fn scale_estimators_are_translation_invariant() {
        let data = [1.0, 2.0, 4.0, 8.0, 16.0];
        let shifted: Vec<f64> = data.iter().map(|x| x + 1000.0).collect();
        assert_close(
            median_absolute_deviation(&data, MadConsistency::Raw).unwrap(),
            median_absolute_deviation(&shifted, MadConsistency::Raw).unwrap(),
            1e-9,
        );
        assert_close(
            interquartile_range(&data).unwrap(),
            interquartile_range(&shifted).unwrap(),
            1e-9,
        );
    }

    #[test]
    fn location_estimators_are_translation_equivariant() {
        let data = [1.0, 2.0, 4.0, 8.0, 16.0];
        let shift = 1000.0;
        let shifted: Vec<f64> = data.iter().map(|x| x + shift).collect();
        assert_close(
            trimmed_mean(&shifted, 0.2).unwrap(),
            trimmed_mean(&data, 0.2).unwrap() + shift,
            1e-9,
        );
        assert_close(
            winsorized_mean(&shifted, 0.2).unwrap(),
            winsorized_mean(&data, 0.2).unwrap() + shift,
            1e-9,
        );
        assert_close(
            weighted_median(&shifted, &[1.0; 5]).unwrap(),
            weighted_median(&data, &[1.0; 5]).unwrap() + shift,
            1e-9,
        );
    }

    #[test]
    fn estimators_are_positive_scale_equivariant() {
        let data = [1.0, 2.0, 4.0, 8.0, 16.0];
        let scale = 2.5;
        let scaled: Vec<f64> = data.iter().map(|x| x * scale).collect();
        assert_close(
            median_absolute_deviation(&scaled, MadConsistency::Raw).unwrap(),
            median_absolute_deviation(&data, MadConsistency::Raw).unwrap() * scale,
            1e-9,
        );
        assert_close(
            interquartile_range(&scaled).unwrap(),
            interquartile_range(&data).unwrap() * scale,
            1e-9,
        );
        assert_close(
            trimmed_mean(&scaled, 0.2).unwrap(),
            trimmed_mean(&data, 0.2).unwrap() * scale,
            1e-9,
        );
        assert_close(
            weighted_median(&scaled, &[1.0; 5]).unwrap(),
            weighted_median(&data, &[1.0; 5]).unwrap() * scale,
            1e-9,
        );
    }

    #[test]
    fn constant_sample_has_zero_scale_and_constant_location() {
        let data = [5.0; 8];
        assert_close(
            median_absolute_deviation(&data, MadConsistency::Normal).unwrap(),
            0.0,
            1e-12,
        );
        assert_close(interquartile_range(&data).unwrap(), 0.0, 1e-12);
        assert_close(trimmed_mean(&data, 0.25).unwrap(), 5.0, 1e-12);
        assert_close(winsorized_mean(&data, 0.25).unwrap(), 5.0, 1e-12);
        assert_close(weighted_median(&data, &[1.0; 8]).unwrap(), 5.0, 1e-12);
    }
}

/// Property-based checks: invariants that must hold for arbitrary samples rather
/// than a handful of hand-picked ones.
// Excluded from Miri: proptest runs hundreds of randomized cases per property —
// impractically slow under the interpreter — and its harness is not designed to
// run under Miri. The native Build & Test jobs exercise these properties.
#[cfg(all(test, not(miri)))]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn finite_vec() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-1.0e6f64..1.0e6, 1..64)
    }

    proptest! {
        /// The raw MAD is a non-negative, finite scale.
        #[test]
        fn mad_is_nonnegative(data in finite_vec()) {
            let mad = median_absolute_deviation(&data, MadConsistency::Raw).unwrap();
            prop_assert!(mad.is_finite() && mad >= 0.0);
        }

        /// The MAD is invariant to a common additive shift.
        #[test]
        fn mad_translation_invariant(data in finite_vec(), shift in -1.0e5f64..1.0e5) {
            let shifted: Vec<f64> = data.iter().map(|x| x + shift).collect();
            let base = median_absolute_deviation(&data, MadConsistency::Raw).unwrap();
            let moved = median_absolute_deviation(&shifted, MadConsistency::Raw).unwrap();
            prop_assert!((base - moved).abs() <= 1e-6 * (1.0 + base.abs()));
        }

        /// The IQR scales with a positive multiplier.
        #[test]
        fn iqr_positive_scale_equivariant(data in finite_vec(), scale in 1.0e-3f64..1.0e3) {
            let scaled: Vec<f64> = data.iter().map(|x| x * scale).collect();
            let base = interquartile_range(&data).unwrap();
            let moved = interquartile_range(&scaled).unwrap();
            prop_assert!((moved - base * scale).abs() <= 1e-6 * (1.0 + (base * scale).abs()));
        }

        /// The trimmed mean shifts by a common additive constant.
        #[test]
        fn trimmed_mean_translation_equivariant(data in finite_vec(), shift in -1.0e5f64..1.0e5) {
            let shifted: Vec<f64> = data.iter().map(|x| x + shift).collect();
            let base = trimmed_mean(&data, 0.2).unwrap();
            let moved = trimmed_mean(&shifted, 0.2).unwrap();
            prop_assert!((moved - (base + shift)).abs() <= 1e-6 * (1.0 + (base + shift).abs()));
        }

        /// The trimmed mean lies between the sample minimum and maximum.
        #[test]
        fn trimmed_mean_within_range(data in finite_vec()) {
            let lo = data.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let m = trimmed_mean(&data, 0.25).unwrap();
            prop_assert!(m >= lo - 1e-9 && m <= hi + 1e-9);
        }

        /// The weighted median is invariant to a permutation of the (value, weight)
        /// pairs — reordering the input must not change the result.
        #[test]
        fn weighted_median_permutation_invariant(data in finite_vec(), seed in any::<u64>()) {
            let weights: Vec<f64> = (0..data.len()).map(|i| 1.0 + (i % 5) as f64).collect();
            let base = weighted_median(&data, &weights).unwrap();

            let mut idx: Vec<usize> = (0..data.len()).collect();
            let mut rng = SplitMix64::new(seed);
            let mut i = idx.len();
            while i > 1 {
                i -= 1;
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                idx.swap(i, j);
            }
            let pv: Vec<f64> = idx.iter().map(|&k| data[k]).collect();
            let pw: Vec<f64> = idx.iter().map(|&k| weights[k]).collect();
            let permuted = weighted_median(&pv, &pw).unwrap();
            prop_assert_eq!(base.to_bits(), permuted.to_bits());
        }

        /// Median-of-means with a fixed seed is bit-for-bit reproducible.
        #[test]
        fn median_of_means_seeded_reproducible(data in finite_vec(), seed in any::<u64>()) {
            let blocks = 1 + data.len() / 4;
            let config = MedianOfMeansConfig {
                block_count: blocks.min(data.len()).max(1),
                seed,
                partition: MedianOfMeansPartition::SeededPermutation,
            };
            let a = median_of_means(&data, config).unwrap();
            let b = median_of_means(&data, config).unwrap();
            prop_assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
