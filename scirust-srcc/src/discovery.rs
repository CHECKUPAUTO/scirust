//! Deterministic discovery of SRCC transport operators.

use core::fmt;

use crate::{LinearMap16, SRCC_DIMENSION, Vector16, squared_norm};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccTransportSample {
    pub source: Vector16,
    pub target: Vector16,
}

impl SrccTransportSample {
    #[must_use]
    pub const fn new(source: Vector16, target: Vector16) -> Self {
        Self { source, target }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrccDiscoveryError {
    EmptySamples,
    InvalidViewCount,
    TooManyViews,
    InsufficientDistinctViews,
    InvalidEnergyFloor,
    NonFiniteSample { index: usize },
    ZeroSource { index: usize },
}

impl fmt::Display for SrccDiscoveryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptySamples => formatter.write_str("transport samples must not be empty"),
            Self::InvalidViewCount => formatter.write_str("view count must be strictly positive"),
            Self::TooManyViews => formatter.write_str("view count must not exceed sample count"),
            Self::InsufficientDistinctViews =>
            {
                formatter.write_str("view count must not exceed distinct sample count")
            },
            Self::InvalidEnergyFloor =>
            {
                formatter.write_str("energy floor must be finite and positive")
            },
            Self::NonFiniteSample { index } =>
            {
                write!(formatter, "transport sample {index} is non-finite",)
            },
            Self::ZeroSource { index } =>
            {
                write!(formatter, "transport sample {index} has zero source energy",)
            },
        }
    }
}

impl std::error::Error for SrccDiscoveryError {}

/// Learns independent real-linear transport views.
///
/// Samples are assigned deterministically using:
///
/// `view_index = sample_index mod view_count`.
///
/// Each sample contributes the normalized rank-one action:
///
/// `target * source^T / ||source||^2`.
pub fn learn_interleaved_transport_views(
    samples: &[SrccTransportSample],
    view_count: usize,
    energy_floor: f64,
) -> Result<Vec<LinearMap16>, SrccDiscoveryError> {
    if samples.is_empty()
    {
        return Err(SrccDiscoveryError::EmptySamples);
    }

    if view_count == 0
    {
        return Err(SrccDiscoveryError::InvalidViewCount);
    }

    if view_count > samples.len()
    {
        return Err(SrccDiscoveryError::TooManyViews);
    }

    if !energy_floor.is_finite() || energy_floor <= 0.0
    {
        return Err(SrccDiscoveryError::InvalidEnergyFloor);
    }

    for (sample_index, sample) in samples.iter().enumerate()
    {
        if sample
            .source
            .iter()
            .chain(sample.target.iter())
            .any(|value| !value.is_finite())
        {
            return Err(SrccDiscoveryError::NonFiniteSample {
                index: sample_index,
            });
        }

        let source_energy = squared_norm(&sample.source);

        if source_energy <= energy_floor
        {
            return Err(SrccDiscoveryError::ZeroSource {
                index: sample_index,
            });
        }
    }

    let mut ordered_samples = samples.to_vec();
    ordered_samples.sort_by(compare_samples);

    let mut groups: Vec<Vec<SrccTransportSample>> = Vec::new();

    for sample in ordered_samples
    {
        match groups.last_mut()
        {
            Some(group) if group[0] == sample => group.push(sample),
            _ => groups.push(vec![sample]),
        }
    }

    if groups.len() < view_count
    {
        return Err(SrccDiscoveryError::InsufficientDistinctViews);
    }

    let mut transports = vec![[[0.0; SRCC_DIMENSION]; SRCC_DIMENSION]; view_count];

    let mut counts = vec![0_usize; view_count];

    for (group_index, group) in groups.iter().enumerate()
    {
        let view_index = group_index % view_count;

        for sample in group
        {
            let inverse_energy = 1.0 / squared_norm(&sample.source);

            for (target_value, row) in sample.target.iter().zip(transports[view_index].iter_mut())
            {
                let scaled_target = target_value * inverse_energy;

                for (entry, source_value) in row.iter_mut().zip(sample.source.iter())
                {
                    *entry += scaled_target * source_value;
                }
            }

            counts[view_index] += 1;
        }
    }

    for (transport, count) in transports.iter_mut().zip(counts)
    {
        let inverse_count = 1.0 / count as f64;

        for row in transport
        {
            for value in row
            {
                *value *= inverse_count;
            }
        }
    }

    Ok(transports)
}

fn compare_samples(left: &SrccTransportSample, right: &SrccTransportSample) -> core::cmp::Ordering {
    compare_vectors(&left.source, &right.source)
        .then_with(|| compare_vectors(&left.target, &right.target))
}

fn compare_vectors(left: &Vector16, right: &Vector16) -> core::cmp::Ordering {
    left.iter()
        .zip(right.iter())
        .find_map(|(left_value, right_value)| {
            let ordering = left_value.total_cmp(right_value);

            ordering.is_ne().then_some(ordering)
        })
        .unwrap_or(core::cmp::Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SrccClosure, SrccConfig, basis_vector, dot};

    #[test]
    fn independent_views_generate_consensus() {
        let source = basis_vector(1).unwrap();
        let positive = basis_vector(2).unwrap();
        let negative = positive.map(|value| -value);

        let samples = [
            SrccTransportSample::new(source, positive),
            SrccTransportSample::new(source, negative),
            SrccTransportSample::new(source, positive),
            SrccTransportSample::new(source, negative),
        ];

        let transports = learn_interleaved_transport_views(&samples, 2, 1.0e-30).unwrap();

        let closure = SrccClosure::build(&[source], &transports, SrccConfig::default()).unwrap();

        assert_eq!(closure.dimension(), 2);

        assert!(dot(&closure.basis()[1], &basis_vector(2).unwrap(),).abs() > 1.0 - 1.0e-12);
    }

    #[test]
    fn disagreement_between_views_is_rejected() {
        let source = basis_vector(1).unwrap();

        let samples = [
            SrccTransportSample::new(source, basis_vector(2).unwrap()),
            SrccTransportSample::new(source, basis_vector(3).unwrap()),
        ];

        let transports = learn_interleaved_transport_views(&samples, 2, 1.0e-30).unwrap();

        let closure = SrccClosure::build(&[source], &transports, SrccConfig::default()).unwrap();

        assert_eq!(closure.dimension(), 1);
    }

    #[test]
    fn discovery_is_deterministic() {
        let source = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let samples = [
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, target.map(|value| -value)),
        ];

        let first = learn_interleaved_transport_views(&samples, 2, 1.0e-30).unwrap();

        let second = learn_interleaved_transport_views(&samples, 2, 1.0e-30).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn discovery_is_invariant_to_sample_order() {
        let source = basis_vector(1).unwrap();
        let positive = basis_vector(2).unwrap();
        let negative = positive.map(|value| -value);

        let samples = [
            SrccTransportSample::new(source, positive),
            SrccTransportSample::new(source, negative),
            SrccTransportSample::new(source, positive),
            SrccTransportSample::new(source, negative),
        ];

        let reordered = [samples[3], samples[0], samples[2], samples[1]];

        let first = learn_interleaved_transport_views(&samples, 2, 1.0e-30).unwrap();

        let second = learn_interleaved_transport_views(&reordered, 2, 1.0e-30).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn invalid_discovery_inputs_are_rejected() {
        assert_eq!(
            learn_interleaved_transport_views(&[], 1, 1.0e-30,),
            Err(SrccDiscoveryError::EmptySamples),
        );

        let samples = [SrccTransportSample::new(
            [0.0; SRCC_DIMENSION],
            basis_vector(2).unwrap(),
        )];

        assert_eq!(
            learn_interleaved_transport_views(&samples, 1, 1.0e-30,),
            Err(SrccDiscoveryError::ZeroSource { index: 0 }),
        );
    }
}
