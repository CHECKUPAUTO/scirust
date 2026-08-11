//! Deterministic summaries for out-of-band SGEMM timing samples.
//!
//! This module does not read clocks. Measurement harnesses own synchronization
//! and timing; this layer only validates and summarizes integer nanosecond
//! samples in the exact shape required by an autotuning evidence pipeline.

/// Stable timing summary matching ElasticAutoTuner measurement semantics without
/// creating a dependency edge from `scirust-simd` to `elastic-autotuner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmMeasurementSummary {
    pub sample_count: u32,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub mad_ns: u64,
}

/// Validate and deterministically summarize caller-collected nanosecond samples.
///
/// The input order is irrelevant. Quantiles use nearest-rank semantics and MAD
/// is the median absolute deviation from the sample median. Caller-owned scratch
/// keeps allocation policy explicit; it must be at least `samples.len()` long.
pub fn summarize_gemm_measurements(
    samples: &[u64],
    scratch: &mut [u64],
) -> Result<GemmMeasurementSummary, GemmMeasurementError> {
    if samples.is_empty()
    {
        return Err(GemmMeasurementError::NoSamples);
    }
    let sample_count =
        u32::try_from(samples.len()).map_err(|_| GemmMeasurementError::TooManySamples)?;
    if scratch.len() < samples.len()
    {
        return Err(GemmMeasurementError::ScratchTooSmall {
            required: samples.len(),
            actual: scratch.len(),
        });
    }

    let work = &mut scratch[..samples.len()];
    work.copy_from_slice(samples);
    work.sort_unstable();

    let median_ns = median_sorted(work);
    let p95_ns = nearest_rank_sorted(work, 95);
    let p99_ns = nearest_rank_sorted(work, 99);

    for (slot, &sample) in work.iter_mut().zip(samples.iter())
    {
        *slot = sample.abs_diff(median_ns);
    }
    work.sort_unstable();
    let mad_ns = median_sorted(work);

    Ok(GemmMeasurementSummary {
        sample_count,
        median_ns,
        p95_ns,
        p99_ns,
        mad_ns,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmMeasurementError {
    NoSamples,
    TooManySamples,
    ScratchTooSmall { required: usize, actual: usize },
}

impl core::fmt::Display for GemmMeasurementError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::NoSamples => write!(f, "at least one GEMM timing sample is required"),
            Self::TooManySamples => write!(f, "GEMM timing sample count exceeds u32"),
            Self::ScratchTooSmall { required, actual } => write!(
                f,
                "GEMM measurement scratch too small: required {required}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for GemmMeasurementError {}

fn median_sorted(values: &[u64]) -> u64 {
    let middle = values.len() / 2;
    if values.len() % 2 == 1
    {
        values[middle]
    }
    else
    {
        // Overflow-safe floor average, deterministic for integer nanoseconds.
        let left = values[middle - 1];
        let right = values[middle];
        left / 2 + right / 2 + (left % 2 + right % 2) / 2
    }
}

fn nearest_rank_sorted(values: &[u64], percentile: usize) -> u64 {
    debug_assert!((1..=100).contains(&percentile));
    // nearest-rank index = ceil(p*N/100)-1, computed without floating point.
    let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
    values[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_is_order_independent_and_exact() {
        let samples = [11_u64, 9, 10, 100, 12, 8, 10, 9, 11, 10];
        let reversed = [10_u64, 11, 9, 10, 8, 12, 100, 10, 9, 11];
        let mut scratch_a = [0_u64; 10];
        let mut scratch_b = [0_u64; 10];
        let a = summarize_gemm_measurements(&samples, &mut scratch_a).unwrap();
        let b = summarize_gemm_measurements(&reversed, &mut scratch_b).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.sample_count, 10);
        assert_eq!(a.median_ns, 10);
        assert_eq!(a.p95_ns, 100);
        assert_eq!(a.p99_ns, 100);
        assert_eq!(a.mad_ns, 1);
    }

    #[test]
    fn even_median_uses_overflow_safe_integer_average() {
        let samples = [u64::MAX - 1, u64::MAX];
        let mut scratch = [0_u64; 2];
        let summary = summarize_gemm_measurements(&samples, &mut scratch).unwrap();
        assert_eq!(summary.median_ns, u64::MAX - 1);
    }

    #[test]
    fn caller_must_provide_sufficient_scratch() {
        let samples = [1_u64, 2, 3];
        let mut scratch = [0_u64; 2];
        assert_eq!(
            summarize_gemm_measurements(&samples, &mut scratch),
            Err(GemmMeasurementError::ScratchTooSmall {
                required: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn empty_measurement_is_rejected() {
        let mut scratch = [];
        assert_eq!(
            summarize_gemm_measurements(&[], &mut scratch),
            Err(GemmMeasurementError::NoSamples)
        );
    }
}
