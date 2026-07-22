//! Deterministic paired statistical comparison.
//!
//! Aggregate means hide per-unit structure, so comparisons here are
//! **paired**: both methods are evaluated on the same units (samples,
//! machines, splits), and inference runs on the per-unit differences. The
//! machinery is a seeded percentile bootstrap over the difference vector:
//!
//! - resampling uses `SplitMix64` with an explicit recorded seed, index
//!   draws in fixed order — identical inputs give identical intervals on
//!   every platform;
//! - the percentile rule is the empirical quantile at
//!   `floor(q · (resamples − 1))` on the `total_cmp`-sorted resample means —
//!   an exact, documented convention (no interpolation ambiguity);
//! - the effect size is the paired Cohen's d: mean difference divided by the
//!   sample standard deviation of the differences (`n − 1` denominator),
//!   reported as `None` when the differences are constant (zero variance) —
//!   never a division by zero smuggled into a number.
//!
//! A bootstrap over fewer than two units, or a confidence level outside
//! `(0, 1)`, is a typed error, not a degenerate interval.

use core::fmt;

use scirust_bench_schema::ConfidenceInterval;
use scirust_stats::SplitMix64;

/// A deterministic paired-bootstrap summary of `method_a − method_b`
/// differences.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedBootstrapReport {
    /// Mean of the observed per-unit differences.
    pub mean_difference: f64,
    /// Percentile bootstrap interval for the mean difference.
    pub confidence_interval: ConfidenceInterval,
    /// Paired Cohen's d (`None` when the differences have zero variance).
    pub effect_size: Option<f64>,
    /// Number of per-unit differences.
    pub unit_count: usize,
    /// Number of bootstrap resamples drawn.
    pub resamples: usize,
    /// The seed consumed by the resampler.
    pub seed: u64,
}

/// Typed paired-comparison errors.
#[derive(Clone, Debug, PartialEq)]
pub enum PairedComparisonError {
    /// Fewer than two per-unit differences.
    TooFewUnits {
        /// The number supplied.
        found: usize,
    },
    /// The two per-unit metric vectors have different lengths.
    LengthMismatch {
        /// Length of the first vector.
        left: usize,
        /// Length of the second vector.
        right: usize,
    },
    /// A difference is `NaN` or `±∞`.
    NonFiniteDifference {
        /// Index of the offending unit.
        index: usize,
    },
    /// The confidence level is not in `(0, 1)`.
    InvalidConfidenceLevel {
        /// The offending level.
        level: f64,
    },
    /// Zero bootstrap resamples were requested.
    ZeroResamples,
}

impl fmt::Display for PairedComparisonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::TooFewUnits { found } =>
            {
                write!(
                    formatter,
                    "paired comparison needs at least 2 units, found {found}"
                )
            },
            Self::LengthMismatch { left, right } =>
            {
                write!(formatter, "paired vectors have lengths {left} and {right}")
            },
            Self::NonFiniteDifference { index } =>
            {
                write!(formatter, "difference at unit {index} is not finite")
            },
            Self::InvalidConfidenceLevel { level } =>
            {
                write!(formatter, "confidence level {level} is not in (0, 1)")
            },
            Self::ZeroResamples => formatter.write_str("bootstrap needs at least one resample"),
        }
    }
}

impl std::error::Error for PairedComparisonError {}

/// Pairs two per-unit metric vectors into differences `left − right`.
pub fn paired_differences(left: &[f64], right: &[f64]) -> Result<Vec<f64>, PairedComparisonError> {
    if left.len() != right.len()
    {
        return Err(PairedComparisonError::LengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }

    let differences: Vec<f64> = left.iter().zip(right).map(|(a, b)| a - b).collect();

    for (index, difference) in differences.iter().enumerate()
    {
        if !difference.is_finite()
        {
            return Err(PairedComparisonError::NonFiniteDifference { index });
        }
    }

    Ok(differences)
}

/// Seeded percentile bootstrap over per-unit differences.
pub fn paired_bootstrap(
    differences: &[f64],
    resamples: usize,
    level: f64,
    seed: u64,
) -> Result<PairedBootstrapReport, PairedComparisonError> {
    let unit_count = differences.len();

    if unit_count < 2
    {
        return Err(PairedComparisonError::TooFewUnits { found: unit_count });
    }

    for (index, difference) in differences.iter().enumerate()
    {
        if !difference.is_finite()
        {
            return Err(PairedComparisonError::NonFiniteDifference { index });
        }
    }

    if !level.is_finite() || level <= 0.0 || level >= 1.0
    {
        return Err(PairedComparisonError::InvalidConfidenceLevel { level });
    }

    if resamples == 0
    {
        return Err(PairedComparisonError::ZeroResamples);
    }

    let mean_difference = differences.iter().sum::<f64>() / unit_count as f64;

    let variance = differences
        .iter()
        .map(|difference| (difference - mean_difference).powi(2))
        .sum::<f64>()
        / (unit_count - 1) as f64;

    let standard_deviation = variance.sqrt();

    let effect_size = if standard_deviation == 0.0
    {
        None
    }
    else
    {
        Some(mean_difference / standard_deviation)
    };

    let mut rng = SplitMix64::new(seed);
    let mut resample_means = Vec::with_capacity(resamples);

    for _ in 0..resamples
    {
        let mut sum = 0.0;

        for _ in 0..unit_count
        {
            let draw = (rng.next_f64() * unit_count as f64) as usize;
            sum += differences[draw.min(unit_count - 1)];
        }

        resample_means.push(sum / unit_count as f64);
    }

    resample_means.sort_by(f64::total_cmp);

    let quantile = |q: f64| -> f64 {
        let position = (q * (resamples - 1) as f64).floor() as usize;

        resample_means[position.min(resamples - 1)]
    };

    let tail = (1.0 - level) / 2.0;

    Ok(PairedBootstrapReport {
        mean_difference,
        confidence_interval: ConfidenceInterval {
            lo: quantile(tail),
            hi: quantile(1.0 - tail),
            level,
        },
        effect_size,
        unit_count,
        resamples,
        seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_is_deterministic() {
        let differences = vec![0.5, 0.7, 0.4, 0.9, 0.6, 0.55, 0.65, 0.8];

        let first = paired_bootstrap(&differences, 999, 0.95, 0x0000_B007).unwrap();
        let second = paired_bootstrap(&differences, 999, 0.95, 0x0000_B007).unwrap();

        assert_eq!(first, second);

        let other_seed = paired_bootstrap(&differences, 999, 0.95, 0x0000_B008).unwrap();

        assert_ne!(
            (first.confidence_interval.lo, first.confidence_interval.hi),
            (
                other_seed.confidence_interval.lo,
                other_seed.confidence_interval.hi,
            ),
        );
    }

    #[test]
    fn interval_brackets_the_mean_for_a_clear_effect() {
        let differences: Vec<f64> = (0..40).map(|i| 1.0 + 0.01 * (i % 5) as f64).collect();

        let report = paired_bootstrap(&differences, 2000, 0.95, 42).unwrap();

        assert!(report.confidence_interval.lo <= report.mean_difference);
        assert!(report.mean_difference <= report.confidence_interval.hi);
        // Every difference is near 1.0; the interval must exclude zero.
        assert!(report.confidence_interval.lo > 0.9);
        assert!(report.effect_size.unwrap() > 0.0);
    }

    #[test]
    fn constant_differences_have_no_effect_size_and_a_degenerate_interval() {
        let differences = vec![0.25; 10];

        let report = paired_bootstrap(&differences, 500, 0.9, 7).unwrap();

        assert_eq!(report.effect_size, None);
        assert_eq!(report.mean_difference, 0.25);
        assert_eq!(report.confidence_interval.lo, 0.25);
        assert_eq!(report.confidence_interval.hi, 0.25);
    }

    #[test]
    fn paired_differences_validate_shape_and_finiteness() {
        assert_eq!(
            paired_differences(&[1.0, 2.0], &[1.0]),
            Err(PairedComparisonError::LengthMismatch { left: 2, right: 1 }),
        );

        assert_eq!(
            paired_differences(&[f64::MAX, 0.0], &[-f64::MAX, 0.0]),
            Err(PairedComparisonError::NonFiniteDifference { index: 0 }),
        );

        assert_eq!(
            paired_differences(&[3.0, 2.0], &[1.0, 0.5]).unwrap(),
            vec![2.0, 1.5],
        );
    }

    #[test]
    fn invalid_requests_are_typed_errors() {
        assert_eq!(
            paired_bootstrap(&[1.0], 100, 0.95, 0),
            Err(PairedComparisonError::TooFewUnits { found: 1 }),
        );

        assert_eq!(
            paired_bootstrap(&[1.0, 2.0], 100, 1.0, 0),
            Err(PairedComparisonError::InvalidConfidenceLevel { level: 1.0 }),
        );

        assert_eq!(
            paired_bootstrap(&[1.0, 2.0], 0, 0.95, 0),
            Err(PairedComparisonError::ZeroResamples),
        );

        assert_eq!(
            paired_bootstrap(&[1.0, f64::NAN], 100, 0.95, 0),
            Err(PairedComparisonError::NonFiniteDifference { index: 1 }),
        );
    }
}
