//! Deterministic robust profile synthesis from ElasticTokenizer timings.
//!
//! The fitter partitions ordered probe lengths into exactly six contiguous
//! execution classes (S/M/L/XL/XXL/XXXL). Correctness is a hard gate: any
//! `(piece length, kernel)` pair that ever failed canonical token-id parity is
//! excluded. Valid samples are Tukey-IQR filtered and each segment is optimized
//! lexicographically by the sum of robust median costs, then the sum of p95
//! costs. No weighted score can compensate for semantic failure.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use scirust_stats::describe::{median, quantile};

use crate::elastic_calibration::CalibrationMeasurement;
use crate::elastic_tokenizer::{BpeKernel, ElasticProfile, ElasticThresholds, ThresholdError};

const PROFILE_CLASSES: usize = 6;
const PROFILE_KERNELS: [BpeKernel; 4] = [
    BpeKernel::Reference,
    BpeKernel::TinyScan,
    BpeKernel::Indexed,
    BpeKernel::Heap,
];

type TimingKey = (usize, u8);
type RobustCosts = BTreeMap<TimingKey, RobustTimingCost>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct RobustTimingCost {
    median_nanos: u64,
    p95_nanos: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AggregatedTimings {
    lengths: Vec<usize>,
    costs: RobustCosts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ProfileCost {
    median_sum: u128,
    p95_sum: u128,
}

impl ProfileCost {
    const ZERO: Self = Self {
        median_sum: 0,
        p95_sum: 0,
    };

    const fn saturating_add(self, other: Self) -> Self {
        Self {
            median_sum: self.median_sum.saturating_add(other.median_sum),
            p95_sum: self.p95_sum.saturating_add(other.p95_sum),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct State {
    cost: ProfileCost,
    start: usize,
    kernel: BpeKernel,
}

/// Builds one six-class hardware-local execution profile from robust measured costs.
pub struct ElasticProfileFitter;

impl ElasticProfileFitter {
    pub fn fit(measurements: &[CalibrationMeasurement]) -> Result<ElasticProfile, ProfileFitError> {
        let aggregated = aggregate_robust_costs(measurements)?;
        let lengths = aggregated.lengths;
        let costs = aggregated.costs;
        if lengths.len() < PROFILE_CLASSES
        {
            return Err(ProfileFitError::NotEnoughProbeLengths {
                found: lengths.len(),
            });
        }

        let n = lengths.len();
        let mut dp = vec![vec![None::<State>; n + 1]; PROFILE_CLASSES + 1];
        dp[0][0] = Some(State {
            cost: ProfileCost::ZERO,
            start: 0,
            kernel: BpeKernel::Reference,
        });

        for segments in 1..=PROFILE_CLASSES
        {
            let min_end = segments;
            let max_end = n - (PROFILE_CLASSES - segments);
            for end in min_end..=max_end
            {
                for start in segments - 1..end
                {
                    let Some(previous) = dp[segments - 1][start]
                    else
                    {
                        continue;
                    };
                    for kernel in PROFILE_KERNELS
                    {
                        let Some(segment_cost) =
                            segment_cost(&lengths, &costs, start, end, kernel)
                        else
                        {
                            continue;
                        };
                        let candidate = State {
                            cost: previous.cost.saturating_add(segment_cost),
                            start,
                            kernel,
                        };
                        if dp[segments][end]
                            .is_none_or(|current| candidate.cost < current.cost)
                        {
                            dp[segments][end] = Some(candidate);
                        }
                    }
                }
            }
        }

        let Some(_) = dp[PROFILE_CLASSES][n]
        else
        {
            return Err(ProfileFitError::NoFeasibleSixClassProfile);
        };

        let mut segments = Vec::with_capacity(PROFILE_CLASSES);
        let mut end = n;
        for count in (1..=PROFILE_CLASSES).rev()
        {
            let state = dp[count][end].expect("final DP path is complete");
            segments.push((state.start, end, state.kernel));
            end = state.start;
        }
        segments.reverse();

        let mut boundaries = [0usize; PROFILE_CLASSES - 1];
        for index in 0..PROFILE_CLASSES - 1
        {
            let left = lengths[segments[index].1 - 1];
            let right = lengths[segments[index + 1].0];
            boundaries[index] = left + (right - left) / 2;
        }

        let thresholds = ElasticThresholds::new(
            boundaries[0],
            boundaries[1],
            boundaries[2],
            boundaries[3],
            boundaries[4],
        )?;
        let kernels = [
            segments[0].2,
            segments[1].2,
            segments[2].2,
            segments[3].2,
            segments[4].2,
            segments[5].2,
        ];
        Ok(ElasticProfile::new(thresholds, kernels))
    }
}

fn aggregate_robust_costs(
    measurements: &[CalibrationMeasurement],
) -> Result<AggregatedTimings, ProfileFitError> {
    if measurements.is_empty()
    {
        return Err(ProfileFitError::NoMeasurements);
    }

    let lengths: Vec<usize> = measurements
        .iter()
        .map(|measurement| measurement.piece_len)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let disqualified: BTreeSet<TimingKey> = measurements
        .iter()
        .filter(|measurement| !measurement.semantic_match)
        .map(|measurement| (measurement.piece_len, kernel_order(measurement.kernel)))
        .collect();

    let mut grouped: BTreeMap<TimingKey, Vec<f64>> = BTreeMap::new();
    for measurement in measurements
    {
        let key = (measurement.piece_len, kernel_order(measurement.kernel));
        if !measurement.semantic_match || disqualified.contains(&key)
        {
            continue;
        }
        grouped
            .entry(key)
            .or_default()
            .push(measurement.elapsed_nanos as f64);
    }

    let mut costs = BTreeMap::new();
    for (key, samples) in grouped
    {
        let clean = reject_tukey_outliers(&samples);
        if clean.is_empty()
        {
            continue;
        }
        costs.insert(
            key,
            RobustTimingCost {
                median_nanos: finite_nanos_to_u64(median(&clean)),
                p95_nanos: finite_nanos_to_u64(quantile(&clean, 0.95)),
            },
        );
    }
    Ok(AggregatedTimings { lengths, costs })
}

fn reject_tukey_outliers(samples: &[f64]) -> Vec<f64> {
    if samples.len() < 4
    {
        return samples.to_vec();
    }
    let q1 = quantile(samples, 0.25);
    let q3 = quantile(samples, 0.75);
    let iqr = q3 - q1;
    let low = q1 - 1.5 * iqr;
    let high = q3 + 1.5 * iqr;
    samples
        .iter()
        .copied()
        .filter(|sample| *sample >= low && *sample <= high)
        .collect()
}

fn finite_nanos_to_u64(value: f64) -> u64 {
    if !value.is_finite() || value <= 0.0
    {
        0
    }
    else if value >= u64::MAX as f64
    {
        u64::MAX
    }
    else
    {
        value.round() as u64
    }
}

fn segment_cost(
    lengths: &[usize],
    costs: &RobustCosts,
    start: usize,
    end: usize,
    kernel: BpeKernel,
) -> Option<ProfileCost> {
    let mut cost = ProfileCost::ZERO;
    for &piece_len in &lengths[start..end]
    {
        let timing = costs.get(&(piece_len, kernel_order(kernel)))?;
        cost = cost.saturating_add(ProfileCost {
            median_sum: u128::from(timing.median_nanos),
            p95_sum: u128::from(timing.p95_nanos),
        });
    }
    Some(cost)
}

const fn kernel_order(kernel: BpeKernel) -> u8 {
    match kernel
    {
        BpeKernel::Reference => 0,
        BpeKernel::TinyScan => 1,
        BpeKernel::Indexed => 2,
        BpeKernel::Heap => 3,
    }
}

/// Measurements cannot be converted into a valid six-class profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileFitError {
    NoMeasurements,
    NotEnoughProbeLengths { found: usize },
    NoFeasibleSixClassProfile,
    InvalidThresholds(ThresholdError),
}

impl fmt::Display for ProfileFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NoMeasurements => f.write_str("elastic profile fit received no measurements"),
            Self::NotEnoughProbeLengths { found } => write!(
                f,
                "elastic profile fit needs at least six distinct probe lengths, found {found}"
            ),
            Self::NoFeasibleSixClassProfile =>
            {
                f.write_str("no semantics-safe six-class elastic profile is feasible")
            },
            Self::InvalidThresholds(error) => write!(f, "invalid fitted thresholds: {error}"),
        }
    }
}

impl std::error::Error for ProfileFitError {}

impl From<ThresholdError> for ProfileFitError {
    fn from(value: ThresholdError) -> Self {
        Self::InvalidThresholds(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(
        piece_len: usize,
        kernel: BpeKernel,
        elapsed_nanos: u64,
        semantic_match: bool,
    ) -> CalibrationMeasurement {
        CalibrationMeasurement {
            piece_len,
            kernel,
            elapsed_nanos,
            semantic_match,
        }
    }

    #[test]
    fn six_probe_lengths_map_directly_to_six_execution_classes() {
        let lengths = [8, 32, 128, 512, 2048, 8192];
        let winners = [
            BpeKernel::TinyScan,
            BpeKernel::TinyScan,
            BpeKernel::Indexed,
            BpeKernel::Indexed,
            BpeKernel::Heap,
            BpeKernel::Heap,
        ];
        let mut measurements = Vec::new();
        for (&piece_len, &winner) in lengths.iter().zip(winners.iter())
        {
            for kernel in PROFILE_KERNELS
            {
                measurements.push(m(
                    piece_len,
                    kernel,
                    if kernel == winner { 10 } else { 100 },
                    true,
                ));
            }
        }

        let profile = ElasticProfileFitter::fit(&measurements).unwrap();
        assert_eq!(profile.kernels(), winners);
        assert_eq!(profile.thresholds().s_max, 20);
        assert_eq!(profile.thresholds().m_max, 80);
        assert_eq!(profile.thresholds().l_max, 320);
        assert_eq!(profile.thresholds().xl_max, 1280);
        assert_eq!(profile.thresholds().xxl_max, 5120);
    }

    #[test]
    fn semantic_mismatch_disqualifies_otherwise_fast_kernel() {
        let lengths = [8, 16, 32, 64, 128, 256];
        let mut measurements = Vec::new();
        for piece_len in lengths
        {
            measurements.push(m(piece_len, BpeKernel::Reference, 100, true));
            measurements.push(m(piece_len, BpeKernel::Indexed, 10, true));
        }
        measurements.push(m(64, BpeKernel::Indexed, 1, false));

        let profile = ElasticProfileFitter::fit(&measurements).unwrap();
        assert_eq!(profile.kernel_for(64), BpeKernel::Reference);
    }

    #[test]
    fn tukey_spike_does_not_change_robust_profile_winner() {
        let lengths = [8, 16, 32, 64, 128, 256];
        let mut measurements = Vec::new();
        for piece_len in lengths
        {
            for value in [10, 10, 11, 9, 10_000]
            {
                measurements.push(m(piece_len, BpeKernel::Indexed, value, true));
            }
            for value in [20, 20, 21, 19, 20]
            {
                measurements.push(m(piece_len, BpeKernel::Reference, value, true));
            }
        }
        let profile = ElasticProfileFitter::fit(&measurements).unwrap();
        assert_eq!(profile.kernels(), [BpeKernel::Indexed; PROFILE_CLASSES]);
    }

    #[test]
    fn fitter_rejects_too_few_distinct_probe_lengths() {
        let measurements = [m(8, BpeKernel::Reference, 1, true)];
        assert_eq!(
            ElasticProfileFitter::fit(&measurements),
            Err(ProfileFitError::NotEnoughProbeLengths { found: 1 })
        );
    }
}
