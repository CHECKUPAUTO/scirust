//! Deterministic profile synthesis from ElasticTokenizer timing measurements.
//!
//! The fitter partitions ordered probe lengths into exactly six contiguous
//! execution classes (S/M/L/XL/XXL/XXXL). For each segment it chooses one
//! kernel and minimizes the sum of median nanosecond costs. Any
//! `(piece length, kernel)` pair that ever failed canonical token-id parity is
//! excluded before optimization.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::elastic_calibration::CalibrationMeasurement;
use crate::elastic_tokenizer::{BpeKernel, ElasticProfile, ElasticThresholds, ThresholdError};

const PROFILE_CLASSES: usize = 6;
const PROFILE_KERNELS: [BpeKernel; 4] = [
    BpeKernel::Reference,
    BpeKernel::TinyScan,
    BpeKernel::Indexed,
    BpeKernel::Heap,
];

type MedianKey = (usize, u8);
type MedianCosts = BTreeMap<MedianKey, u64>;

#[derive(Clone, Debug, Eq, PartialEq)]
struct AggregatedTimings {
    lengths: Vec<usize>,
    medians: MedianCosts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct State {
    cost: u128,
    start: usize,
    kernel: BpeKernel,
}

/// Builds one six-class hardware-local execution profile from measured costs.
pub struct ElasticProfileFitter;

impl ElasticProfileFitter {
    pub fn fit(measurements: &[CalibrationMeasurement]) -> Result<ElasticProfile, ProfileFitError> {
        let aggregated = aggregate_medians(measurements)?;
        let lengths = aggregated.lengths;
        let medians = aggregated.medians;
        if lengths.len() < PROFILE_CLASSES
        {
            return Err(ProfileFitError::NotEnoughProbeLengths {
                found: lengths.len(),
            });
        }

        let n = lengths.len();
        let mut dp = vec![vec![None::<State>; n + 1]; PROFILE_CLASSES + 1];
        dp[0][0] = Some(State {
            cost: 0,
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
                            segment_cost(&lengths, &medians, start, end, kernel)
                        else
                        {
                            continue;
                        };
                        let candidate = State {
                            cost: previous.cost.saturating_add(segment_cost),
                            start,
                            kernel,
                        };
                        if dp[segments][end].is_none_or(|current| candidate.cost < current.cost)
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

fn aggregate_medians(
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
    let disqualified: BTreeSet<MedianKey> = measurements
        .iter()
        .filter(|measurement| !measurement.semantic_match)
        .map(|measurement| (measurement.piece_len, kernel_order(measurement.kernel)))
        .collect();

    let mut grouped: BTreeMap<MedianKey, Vec<u64>> = BTreeMap::new();
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
            .push(measurement.elapsed_nanos);
    }

    let mut medians = BTreeMap::new();
    for (key, mut samples) in grouped
    {
        samples.sort_unstable();
        medians.insert(key, integer_median(&samples));
    }
    Ok(AggregatedTimings { lengths, medians })
}

fn segment_cost(
    lengths: &[usize],
    medians: &MedianCosts,
    start: usize,
    end: usize,
    kernel: BpeKernel,
) -> Option<u128> {
    let mut cost = 0u128;
    for &piece_len in &lengths[start..end]
    {
        let median = medians.get(&(piece_len, kernel_order(kernel)))?;
        cost = cost.saturating_add(u128::from(*median));
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

fn integer_median(sorted: &[u64]) -> u64 {
    debug_assert!(!sorted.is_empty());
    let middle = sorted.len() / 2;
    if !sorted.len().is_multiple_of(2)
    {
        sorted[middle]
    }
    else
    {
        let low = sorted[middle - 1];
        let high = sorted[middle];
        low + (high - low) / 2
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
    fn fitter_rejects_too_few_distinct_probe_lengths() {
        let measurements = [m(8, BpeKernel::Reference, 1, true)];
        assert_eq!(
            ElasticProfileFitter::fit(&measurements),
            Err(ProfileFitError::NotEnoughProbeLengths { found: 1 })
        );
    }
}
