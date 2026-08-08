//! Runtime timing harness for ElasticTokenizer auto-calibration.
//!
//! The harness measures complete pre-tokenized pieces on every compatible
//! kernel, compares every measured output to the canonical oracle, and only
//! then hands integer nanosecond samples to [`CalibrationReport`]. It never
//! invents chunk boundaries and never treats a faster semantic mismatch as a
//! valid optimization.

use std::fmt;
use std::hint::black_box;
use std::time::Instant;

use crate::elastic_calibration::{CalibrationError, CalibrationMeasurement, CalibrationReport};
use crate::elastic_heap::HeapBpe;
use crate::elastic_indexed::IndexedBpe;
use crate::elastic_tiny::TinyScanBpe;
use crate::elastic_tokenizer::{BpeKernel, CanonicalBpeOracle, DuplicateMergeRule, TokenId};

const CALIBRATION_KERNELS: [BpeKernel; 4] = [
    BpeKernel::Reference,
    BpeKernel::TinyScan,
    BpeKernel::Indexed,
    BpeKernel::Heap,
];

/// One representative complete BPE piece used during auto-calibration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationCase {
    /// Original byte length used by the ElasticTokenizer router.
    pub piece_len: usize,
    /// Already pre-tokenized base ids for the complete piece.
    pub input_ids: Vec<TokenId>,
}

impl CalibrationCase {
    pub fn new(piece_len: usize, input_ids: Vec<TokenId>) -> Self {
        Self {
            piece_len,
            input_ids,
        }
    }
}

/// Warm-up and measured repetition counts for one calibration session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutotuneConfig {
    pub warmup_runs: usize,
    pub measured_runs: usize,
}

impl AutotuneConfig {
    pub fn new(warmup_runs: usize, measured_runs: usize) -> Result<Self, AutotuneError> {
        if measured_runs == 0
        {
            return Err(AutotuneError::ZeroMeasuredRuns);
        }
        Ok(Self {
            warmup_runs,
            measured_runs,
        })
    }
}

impl Default for AutotuneConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 2,
            measured_runs: 7,
        }
    }
}

/// Full output of one hardware-local calibration session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutotuneResult {
    measurements: Vec<CalibrationMeasurement>,
    report: CalibrationReport,
}

impl AutotuneResult {
    pub fn measurements(&self) -> &[CalibrationMeasurement] {
        &self.measurements
    }

    pub const fn report(&self) -> &CalibrationReport {
        &self.report
    }
}

/// Runtime auto-calibrator backed by the same four semantic kernels as the
/// production elastic router.
#[derive(Clone, Debug)]
pub struct ElasticAutotuner {
    reference: CanonicalBpeOracle,
    tiny_scan: TinyScanBpe,
    indexed: IndexedBpe,
    heap: HeapBpe,
    config: AutotuneConfig,
}

impl ElasticAutotuner {
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
        config: AutotuneConfig,
    ) -> Result<Self, DuplicateMergeRule> {
        Ok(Self {
            reference: CanonicalBpeOracle::from_ordered_merges(merges)?,
            tiny_scan: TinyScanBpe::from_ordered_merges(merges)?,
            indexed: IndexedBpe::from_ordered_merges(merges)?,
            heap: HeapBpe::from_ordered_merges(merges)?,
            config,
        })
    }

    /// Measures all compatible kernels and produces a semantics-gated report.
    pub fn calibrate(&self, cases: &[CalibrationCase]) -> Result<AutotuneResult, AutotuneError> {
        if cases.is_empty()
        {
            return Err(AutotuneError::NoCases);
        }

        let mut measurements = Vec::new();
        for case in cases
        {
            let expected = self.reference.encode_ids(black_box(&case.input_ids));
            for kernel in CALIBRATION_KERNELS
            {
                for _ in 0..self.config.warmup_runs
                {
                    let Some(output) = self.run_kernel(kernel, black_box(&case.input_ids))
                    else
                    {
                        break;
                    };
                    black_box(output);
                }

                for _ in 0..self.config.measured_runs
                {
                    let start = Instant::now();
                    let Some(output) = self.run_kernel(kernel, black_box(&case.input_ids))
                    else
                    {
                        break;
                    };
                    let elapsed_nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    let semantic_match = output == expected;
                    black_box(&output);
                    measurements.push(CalibrationMeasurement {
                        piece_len: case.piece_len,
                        kernel,
                        elapsed_nanos,
                        semantic_match,
                    });
                }
            }
        }

        let report = CalibrationReport::from_measurements(&measurements)?;
        Ok(AutotuneResult {
            measurements,
            report,
        })
    }

    fn run_kernel(&self, kernel: BpeKernel, input: &[TokenId]) -> Option<Vec<TokenId>> {
        match kernel
        {
            BpeKernel::Reference => Some(self.reference.encode_ids(input)),
            BpeKernel::TinyScan => self.tiny_scan.try_encode_ids(input),
            BpeKernel::Indexed => Some(self.indexed.encode_ids(input)),
            BpeKernel::Heap => Some(self.heap.encode_ids(input)),
        }
    }
}

/// Auto-calibration cannot produce a safe report from the supplied inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AutotuneError {
    ZeroMeasuredRuns,
    NoCases,
    Calibration(CalibrationError),
}

impl fmt::Display for AutotuneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroMeasuredRuns => f.write_str("elastic autotune requires measured_runs > 0"),
            Self::NoCases => f.write_str("elastic autotune received no calibration cases"),
            Self::Calibration(error) => write!(f, "elastic autotune calibration failed: {error}"),
        }
    }
}

impl std::error::Error for AutotuneError {}

impl From<CalibrationError> for AutotuneError {
    fn from(value: CalibrationError) -> Self {
        Self::Calibration(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elastic_tiny::TINY_SCAN_CAPACITY;

    #[test]
    fn autotune_config_rejects_zero_measured_runs() {
        assert_eq!(
            AutotuneConfig::new(1, 0),
            Err(AutotuneError::ZeroMeasuredRuns)
        );
    }

    #[test]
    fn autotune_requires_at_least_one_case() {
        let tuner = ElasticAutotuner::from_ordered_merges(
            &[(1, 1, 2)],
            AutotuneConfig::new(0, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(tuner.calibrate(&[]), Err(AutotuneError::NoCases));
    }

    #[test]
    fn autotune_measures_every_compatible_kernel_and_preserves_semantics() {
        let tuner = ElasticAutotuner::from_ordered_merges(
            &[(2, 3, 10), (1, 2, 11), (1, 10, 12)],
            AutotuneConfig::new(0, 2).unwrap(),
        )
        .unwrap();
        let cases = [
            CalibrationCase::new(32, vec![1, 2, 3]),
            CalibrationCase::new(512, vec![1; TINY_SCAN_CAPACITY + 1]),
        ];

        let result = tuner.calibrate(&cases).unwrap();
        // 4 kernels for the small case, 3 for the oversized TinyScan case,
        // with two measured repetitions each.
        assert_eq!(result.measurements().len(), 14);
        assert!(result.measurements().iter().all(|m| m.semantic_match));
        assert_eq!(result.report().winners().len(), 2);
        assert_eq!(result.report().rejected_semantic_measurements(), 0);
    }
}
