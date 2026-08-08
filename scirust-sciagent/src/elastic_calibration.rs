//! Deterministic selection layer for ElasticTokenizer auto-calibration.
//!
//! Timing collection is intentionally kept outside this module. This layer
//! consumes integer timing samples, rejects measurements whose output did not
//! match the canonical BPE oracle, aggregates repeated runs by median, and
//! deterministically selects the fastest valid kernel at every probed length.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::elastic_tokenizer::BpeKernel;

/// One timing observation produced by the calibration harness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CalibrationMeasurement {
    pub piece_len: usize,
    pub kernel: BpeKernel,
    pub elapsed_nanos: u64,
    /// Must be true only after exact token-id comparison with the canonical
    /// oracle for the measured input.
    pub semantic_match: bool,
}

/// Fastest semantically valid kernel for one probed piece length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CalibrationWinner {
    pub piece_len: usize,
    pub kernel: BpeKernel,
    pub median_nanos: u64,
}

/// Deterministic result of one calibration data set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationReport {
    winners: Vec<CalibrationWinner>,
    rejected_semantic_measurements: usize,
}

impl CalibrationReport {
    pub fn from_measurements(
        measurements: &[CalibrationMeasurement],
    ) -> Result<Self, CalibrationError> {
        if measurements.is_empty()
        {
            return Err(CalibrationError::NoMeasurements);
        }

        let lengths: BTreeSet<usize> = measurements.iter().map(|m| m.piece_len).collect();
        let rejected_semantic_measurements = measurements
            .iter()
            .filter(|m| !m.semantic_match)
            .count();

        let mut grouped: BTreeMap<(usize, u8), (BpeKernel, Vec<u64>)> = BTreeMap::new();
        for measurement in measurements
        {
            if !measurement.semantic_match
            {
                continue;
            }
            let key = (measurement.piece_len, kernel_order(measurement.kernel));
            grouped
                .entry(key)
                .or_insert_with(|| (measurement.kernel, Vec::new()))
                .1
                .push(measurement.elapsed_nanos);
        }

        let mut candidates: BTreeMap<usize, Vec<(BpeKernel, u64)>> = BTreeMap::new();
        for ((piece_len, _), (kernel, mut samples)) in grouped
        {
            samples.sort_unstable();
            let median = integer_median(&samples);
            candidates
                .entry(piece_len)
                .or_default()
                .push((kernel, median));
        }

        let mut winners = Vec::with_capacity(lengths.len());
        for piece_len in lengths
        {
            let Some(per_kernel) = candidates.get(&piece_len) else
            {
                return Err(CalibrationError::NoSemanticallyValidKernel { piece_len });
            };
            let Some(&(kernel, median_nanos)) = per_kernel
                .iter()
                .min_by_key(|(kernel, nanos)| (*nanos, kernel_order(*kernel)))
            else
            {
                return Err(CalibrationError::NoSemanticallyValidKernel { piece_len });
            };
            winners.push(CalibrationWinner {
                piece_len,
                kernel,
                median_nanos,
            });
        }

        Ok(Self {
            winners,
            rejected_semantic_measurements,
        })
    }

    pub fn winners(&self) -> &[CalibrationWinner] {
        &self.winners
    }

    pub const fn rejected_semantic_measurements(&self) -> usize {
        self.rejected_semantic_measurements
    }
}

/// Calibration input was insufficient to make a semantics-safe decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationError {
    NoMeasurements,
    NoSemanticallyValidKernel { piece_len: usize },
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NoMeasurements => f.write_str("elastic calibration received no measurements"),
            Self::NoSemanticallyValidKernel { piece_len } => write!(
                f,
                "elastic calibration has no semantically valid kernel for piece length {piece_len}"
            ),
        }
    }
}

impl std::error::Error for CalibrationError {}

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
    fn calibration_selects_median_fastest_kernel_per_length() {
        let report = CalibrationReport::from_measurements(&[
            m(32, BpeKernel::Reference, 120, true),
            m(32, BpeKernel::Reference, 100, true),
            m(32, BpeKernel::Reference, 110, true),
            m(32, BpeKernel::TinyScan, 70, true),
            m(32, BpeKernel::TinyScan, 80, true),
            m(32, BpeKernel::TinyScan, 75, true),
            m(512, BpeKernel::Reference, 900, true),
            m(512, BpeKernel::Heap, 300, true),
        ])
        .unwrap();

        assert_eq!(
            report.winners(),
            &[
                CalibrationWinner {
                    piece_len: 32,
                    kernel: BpeKernel::TinyScan,
                    median_nanos: 75,
                },
                CalibrationWinner {
                    piece_len: 512,
                    kernel: BpeKernel::Heap,
                    median_nanos: 300,
                },
            ]
        );
    }

    #[test]
    fn faster_semantic_mismatch_can_never_win() {
        let report = CalibrationReport::from_measurements(&[
            m(64, BpeKernel::Reference, 100, true),
            m(64, BpeKernel::TinyScan, 1, false),
        ])
        .unwrap();

        assert_eq!(report.winners()[0].kernel, BpeKernel::Reference);
        assert_eq!(report.rejected_semantic_measurements(), 1);
    }

    #[test]
    fn calibration_fails_if_length_has_no_valid_kernel() {
        let err = CalibrationReport::from_measurements(&[
            m(16, BpeKernel::Reference, 100, true),
            m(32, BpeKernel::TinyScan, 10, false),
        ])
        .unwrap_err();

        assert_eq!(
            err,
            CalibrationError::NoSemanticallyValidKernel { piece_len: 32 }
        );
    }

    #[test]
    fn calibration_tie_break_is_stable() {
        let report = CalibrationReport::from_measurements(&[
            m(32, BpeKernel::TinyScan, 50, true),
            m(32, BpeKernel::Reference, 50, true),
        ])
        .unwrap();

        assert_eq!(report.winners()[0].kernel, BpeKernel::Reference);
    }

    #[test]
    fn integer_median_handles_even_sample_count_without_overflow() {
        assert_eq!(integer_median(&[10, 20]), 15);
        assert_eq!(integer_median(&[u64::MAX - 1, u64::MAX]), u64::MAX - 1);
    }
}
