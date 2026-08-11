//! Reproducible measurement protocol and deterministic timing summaries.
//!
//! The tuner deliberately does not read clocks. A benchmark/device harness owns
//! synchronization and timestamp collection, then submits integer nanosecond
//! samples under this explicit protocol.

use crate::ElasticMeasurement;

pub const ELASTIC_MEASUREMENT_PROTOCOL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElasticTimingSource {
    HostWallClock,
    DeviceTimestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElasticResidenceMode {
    Resident,
    TransferInclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElasticSynchronizationBoundary {
    PerIteration,
    BatchEnd,
}

/// Measurement contract that must be fixed before timing starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElasticMeasurementProtocol {
    pub schema_version: u32,
    pub warmup_iterations: u32,
    pub measured_iterations: u32,
    pub timing_source: ElasticTimingSource,
    pub residence_mode: ElasticResidenceMode,
    pub synchronization: ElasticSynchronizationBoundary,
}

impl ElasticMeasurementProtocol {
    pub const fn new(
        warmup_iterations: u32,
        measured_iterations: u32,
        timing_source: ElasticTimingSource,
        residence_mode: ElasticResidenceMode,
        synchronization: ElasticSynchronizationBoundary,
    ) -> Self {
        Self {
            schema_version: ELASTIC_MEASUREMENT_PROTOCOL_SCHEMA_VERSION,
            warmup_iterations,
            measured_iterations,
            timing_source,
            residence_mode,
            synchronization,
        }
    }

    pub fn validate(self) -> Result<(), ElasticMeasurementProtocolError> {
        if self.schema_version != ELASTIC_MEASUREMENT_PROTOCOL_SCHEMA_VERSION
        {
            return Err(ElasticMeasurementProtocolError::UnsupportedSchema {
                expected: ELASTIC_MEASUREMENT_PROTOCOL_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.measured_iterations == 0
        {
            return Err(ElasticMeasurementProtocolError::ZeroMeasuredIterations);
        }
        Ok(())
    }

    /// Summarize caller-collected nanosecond samples according to this protocol.
    ///
    /// The sample count must exactly match `measured_iterations`. Caller-owned
    /// scratch keeps allocation policy explicit. Quantiles use nearest-rank and
    /// MAD is the median absolute deviation from the sample median.
    pub fn summarize(
        self,
        samples_ns: &[u64],
        scratch: &mut [u64],
    ) -> Result<ElasticMeasurement, ElasticMeasurementProtocolError> {
        self.validate()?;
        let expected = usize::try_from(self.measured_iterations)
            .map_err(|_| ElasticMeasurementProtocolError::IterationCountTooLarge)?;
        if samples_ns.len() != expected
        {
            return Err(ElasticMeasurementProtocolError::SampleCountMismatch {
                expected,
                actual: samples_ns.len(),
            });
        }
        if scratch.len() < expected
        {
            return Err(ElasticMeasurementProtocolError::ScratchTooSmall {
                required: expected,
                actual: scratch.len(),
            });
        }

        let work = &mut scratch[..expected];
        work.copy_from_slice(samples_ns);
        work.sort_unstable();

        let median_ns = median_sorted(work);
        let p95_ns = nearest_rank_sorted(work, 95);
        let p99_ns = nearest_rank_sorted(work, 99);

        for (slot, &sample) in work.iter_mut().zip(samples_ns)
        {
            *slot = sample.abs_diff(median_ns);
        }
        work.sort_unstable();
        let mad_ns = median_sorted(work);

        Ok(ElasticMeasurement {
            sample_count: self.measured_iterations,
            median_ns,
            p95_ns,
            p99_ns,
            mad_ns,
        })
    }

    /// Stable, dependency-free binary identity for persistence/evidence records.
    pub fn canonical_bytes(self) -> Result<[u8; 12], ElasticMeasurementProtocolError> {
        self.validate()?;
        let mut out = [0_u8; 12];
        out[0..4].copy_from_slice(&self.schema_version.to_le_bytes());
        out[4..8].copy_from_slice(&self.warmup_iterations.to_le_bytes());
        out[8..12].copy_from_slice(&self.measured_iterations.to_le_bytes());
        out[3] ^= timing_code(self.timing_source) << 4;
        out[7] ^= residence_code(self.residence_mode) << 4;
        out[11] ^= synchronization_code(self.synchronization) << 4;
        Ok(out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticMeasurementProtocolError {
    UnsupportedSchema { expected: u32, actual: u32 },
    ZeroMeasuredIterations,
    IterationCountTooLarge,
    SampleCountMismatch { expected: usize, actual: usize },
    ScratchTooSmall { required: usize, actual: usize },
}

impl core::fmt::Display for ElasticMeasurementProtocolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::UnsupportedSchema { expected, actual } => write!(
                f,
                "measurement protocol schema mismatch: expected {expected}, got {actual}"
            ),
            Self::ZeroMeasuredIterations =>
            {
                write!(f, "measurement protocol requires at least one measured iteration")
            },
            Self::IterationCountTooLarge =>
            {
                write!(f, "measurement iteration count does not fit this target")
            },
            Self::SampleCountMismatch { expected, actual } => write!(
                f,
                "measurement sample count mismatch: expected {expected}, got {actual}"
            ),
            Self::ScratchTooSmall { required, actual } => write!(
                f,
                "measurement scratch too small: required {required}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for ElasticMeasurementProtocolError {}

fn median_sorted(values: &[u64]) -> u64 {
    let middle = values.len() / 2;
    if values.len() % 2 == 1
    {
        values[middle]
    }
    else
    {
        let left = values[middle - 1];
        let right = values[middle];
        left / 2 + right / 2 + (left % 2 + right % 2) / 2
    }
}

fn nearest_rank_sorted(values: &[u64], percentile: usize) -> u64 {
    debug_assert!((1..=100).contains(&percentile));
    let rank = percentile
        .saturating_mul(values.len())
        .div_ceil(100)
        .max(1);
    values[rank - 1]
}

const fn timing_code(source: ElasticTimingSource) -> u8 {
    match source {
        ElasticTimingSource::HostWallClock => 0,
        ElasticTimingSource::DeviceTimestamp => 1,
    }
}

const fn residence_code(mode: ElasticResidenceMode) -> u8 {
    match mode {
        ElasticResidenceMode::Resident => 0,
        ElasticResidenceMode::TransferInclusive => 1,
    }
}

const fn synchronization_code(boundary: ElasticSynchronizationBoundary) -> u8 {
    match boundary {
        ElasticSynchronizationBoundary::PerIteration => 0,
        ElasticSynchronizationBoundary::BatchEnd => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol(iterations: u32) -> ElasticMeasurementProtocol {
        ElasticMeasurementProtocol::new(
            3,
            iterations,
            ElasticTimingSource::HostWallClock,
            ElasticResidenceMode::Resident,
            ElasticSynchronizationBoundary::PerIteration,
        )
    }

    #[test]
    fn summary_is_order_independent() {
        let samples = [11_u64, 9, 10, 100, 12, 8, 10, 9, 11, 10];
        let reversed = [10_u64, 11, 9, 10, 8, 12, 100, 10, 9, 11];
        let mut scratch_a = [0_u64; 10];
        let mut scratch_b = [0_u64; 10];
        let a = protocol(10).summarize(&samples, &mut scratch_a).unwrap();
        let b = protocol(10).summarize(&reversed, &mut scratch_b).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.median_ns, 10);
        assert_eq!(a.p95_ns, 100);
        assert_eq!(a.p99_ns, 100);
        assert_eq!(a.mad_ns, 1);
    }

    #[test]
    fn sample_count_must_match_protocol() {
        let mut scratch = [0_u64; 3];
        assert_eq!(
            protocol(3).summarize(&[1, 2], &mut scratch),
            Err(ElasticMeasurementProtocolError::SampleCountMismatch {
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn protocol_identity_changes_with_semantics() {
        let baseline = protocol(5).canonical_bytes().unwrap();
        let changed = ElasticMeasurementProtocol::new(
            3,
            5,
            ElasticTimingSource::DeviceTimestamp,
            ElasticResidenceMode::Resident,
            ElasticSynchronizationBoundary::PerIteration,
        )
        .canonical_bytes()
        .unwrap();
        assert_ne!(baseline, changed);
    }

    #[test]
    fn zero_measured_iterations_fail_closed() {
        assert_eq!(
            protocol(0).validate(),
            Err(ElasticMeasurementProtocolError::ZeroMeasuredIterations)
        );
    }
}
