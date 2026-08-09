//! Robust, semantics-first model selection for ElasticTokenizer kernels.
//!
//! This layer consumes the raw timings produced by [`crate::ElasticAutotuner`]
//! and uses SciRust's statistics crate to distinguish repeatable speedups from
//! benchmark noise. Correctness is a hard gate: any semantic mismatch removes
//! that kernel at the affected piece length before statistics are computed.

use std::collections::{BTreeMap, BTreeSet};

use scirust_stats::describe::{mean, median, quantile, std_dev};
use scirust_stats::htest::{Tail, t_test_two_sample};

use crate::{BpeKernel, CalibrationMeasurement};

/// Robust summary for one semantically valid `(piece_len, kernel)` timing group.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelTimingSummary {
    pub piece_len: usize,
    pub kernel: BpeKernel,
    pub raw_samples: usize,
    pub clean_samples: usize,
    pub dropped_outliers: usize,
    pub mean_nanos: f64,
    pub std_dev_nanos: f64,
    /// Standard deviation divided by mean after Tukey filtering.
    pub coefficient_of_variation: f64,
    pub median_nanos: f64,
    pub p95_nanos: f64,
    pub q1_nanos: f64,
    pub q3_nanos: f64,
}

/// Strength of the evidence that the selected kernel is faster than the runner-up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionConfidence {
    /// Welch p < 0.01.
    Strong,
    /// Welch p < 0.05.
    Significant,
    /// The median is best but the available samples do not distinguish it from noise.
    Provisional,
    /// There is no second semantically valid kernel to compare against.
    Uncontested,
}

/// Model-selection decision for one piece length.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSelection {
    pub piece_len: usize,
    pub winner: KernelTimingSummary,
    pub runner_up: Option<KernelTimingSummary>,
    /// Runner-up median divided by winner median. Values above one favor the winner.
    pub median_speedup: Option<f64>,
    pub welch_p_value: Option<f64>,
    pub confidence: SelectionConfidence,
}

/// Full robust selection report across every probed length.
#[derive(Clone, Debug, PartialEq)]
pub struct ElasticModelSelectionReport {
    selections: Vec<KernelSelection>,
    rejected_semantic_measurements: usize,
}

impl ElasticModelSelectionReport {
    /// Build a robust report from raw autotune measurements.
    pub fn from_measurements(
        measurements: &[CalibrationMeasurement],
    ) -> Result<Self, ModelSelectionError> {
        if measurements.is_empty()
        {
            return Err(ModelSelectionError::NoMeasurements);
        }

        let rejected_semantic_measurements =
            measurements.iter().filter(|m| !m.semantic_match).count();
        let disqualified: BTreeSet<(usize, u8)> = measurements
            .iter()
            .filter(|m| !m.semantic_match)
            .map(|m| (m.piece_len, kernel_order(m.kernel)))
            .collect();
        let lengths: BTreeSet<usize> = measurements.iter().map(|m| m.piece_len).collect();

        let mut grouped: BTreeMap<(usize, u8), (BpeKernel, Vec<f64>)> = BTreeMap::new();
        for measurement in measurements
        {
            let key = (measurement.piece_len, kernel_order(measurement.kernel));
            if !measurement.semantic_match || disqualified.contains(&key)
            {
                continue;
            }
            grouped
                .entry(key)
                .or_insert_with(|| (measurement.kernel, Vec::new()))
                .1
                .push(measurement.elapsed_nanos as f64);
        }

        let mut by_length: BTreeMap<usize, Vec<(KernelTimingSummary, Vec<f64>)>> = BTreeMap::new();
        for ((piece_len, _), (kernel, raw)) in grouped
        {
            let clean = reject_tukey_outliers(&raw);
            if clean.is_empty()
            {
                continue;
            }
            let summary = summarize(piece_len, kernel, raw.len(), &clean);
            by_length
                .entry(piece_len)
                .or_default()
                .push((summary, clean));
        }

        let mut selections = Vec::with_capacity(lengths.len());
        for piece_len in lengths
        {
            let Some(groups) = by_length.get_mut(&piece_len)
            else
            {
                return Err(ModelSelectionError::NoSemanticallyValidKernel { piece_len });
            };
            groups.sort_by(|(left, _), (right, _)| {
                left.median_nanos
                    .total_cmp(&right.median_nanos)
                    .then_with(|| left.p95_nanos.total_cmp(&right.p95_nanos))
                    .then_with(|| {
                        left.coefficient_of_variation
                            .total_cmp(&right.coefficient_of_variation)
                    })
                    .then_with(|| kernel_order(left.kernel).cmp(&kernel_order(right.kernel)))
            });
            let (winner, winner_samples) = groups[0].clone();
            let runner = groups.get(1).cloned();
            let (runner_up, median_speedup, welch_p_value, confidence) =
                if let Some((runner, runner_samples)) = runner
                {
                    let speedup = if winner.median_nanos > 0.0
                    {
                        Some(runner.median_nanos / winner.median_nanos)
                    }
                    else
                    {
                        None
                    };
                    let p =
                        t_test_two_sample(&winner_samples, &runner_samples, false, Tail::TwoSided)
                            .map(|result| result.p_value);
                    let confidence = match p
                    {
                        Some(value) if value < 0.01 => SelectionConfidence::Strong,
                        Some(value) if value < 0.05 => SelectionConfidence::Significant,
                        _ => SelectionConfidence::Provisional,
                    };
                    (Some(runner), speedup, p, confidence)
                }
                else
                {
                    (None, None, None, SelectionConfidence::Uncontested)
                };
            selections.push(KernelSelection {
                piece_len,
                winner,
                runner_up,
                median_speedup,
                welch_p_value,
                confidence,
            });
        }

        Ok(Self {
            selections,
            rejected_semantic_measurements,
        })
    }

    pub fn selections(&self) -> &[KernelSelection] {
        &self.selections
    }

    pub const fn rejected_semantic_measurements(&self) -> usize {
        self.rejected_semantic_measurements
    }
}

/// Invalid or insufficient model-selection evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSelectionError {
    NoMeasurements,
    NoSemanticallyValidKernel { piece_len: usize },
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

fn summarize(
    piece_len: usize,
    kernel: BpeKernel,
    raw_samples: usize,
    clean: &[f64],
) -> KernelTimingSummary {
    let mean_nanos = mean(clean);
    let std_dev_nanos = std_dev(clean);
    let coefficient_of_variation = if mean_nanos > 0.0 && std_dev_nanos.is_finite()
    {
        std_dev_nanos / mean_nanos
    }
    else
    {
        f64::INFINITY
    };
    KernelTimingSummary {
        piece_len,
        kernel,
        raw_samples,
        clean_samples: clean.len(),
        dropped_outliers: raw_samples.saturating_sub(clean.len()),
        mean_nanos,
        std_dev_nanos,
        coefficient_of_variation,
        median_nanos: median(clean),
        p95_nanos: quantile(clean, 0.95),
        q1_nanos: quantile(clean, 0.25),
        q3_nanos: quantile(clean, 0.75),
    }
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
    fn semantic_mismatch_disqualifies_fast_but_wrong_kernel() {
        let report = ElasticModelSelectionReport::from_measurements(&[
            m(32, BpeKernel::Reference, 100, true),
            m(32, BpeKernel::Reference, 101, true),
            m(32, BpeKernel::TinyScan, 10, true),
            m(32, BpeKernel::TinyScan, 9, false),
        ])
        .unwrap();
        assert_eq!(report.selections()[0].winner.kernel, BpeKernel::Reference);
        assert_eq!(report.rejected_semantic_measurements(), 1);
    }

    #[test]
    fn tukey_filter_removes_timing_spike_before_selection() {
        let mut measurements = Vec::new();
        for value in [100, 101, 99, 100, 5000]
        {
            measurements.push(m(64, BpeKernel::Reference, value, true));
        }
        for value in [120, 121, 119, 120, 122]
        {
            measurements.push(m(64, BpeKernel::Heap, value, true));
        }
        let report = ElasticModelSelectionReport::from_measurements(&measurements).unwrap();
        let selection = &report.selections()[0];
        assert_eq!(selection.winner.kernel, BpeKernel::Reference);
        assert_eq!(selection.winner.dropped_outliers, 1);
        assert!(selection.winner.coefficient_of_variation.is_finite());
    }

    #[test]
    fn report_exposes_runner_up_speedup_and_welch_evidence() {
        let mut measurements = Vec::new();
        for value in 90..110
        {
            measurements.push(m(128, BpeKernel::Indexed, value, true));
            measurements.push(m(128, BpeKernel::Heap, value + 100, true));
        }
        let report = ElasticModelSelectionReport::from_measurements(&measurements).unwrap();
        let selection = &report.selections()[0];
        assert_eq!(selection.winner.kernel, BpeKernel::Indexed);
        assert!(selection.median_speedup.unwrap() > 1.5);
        assert!(matches!(selection.confidence, SelectionConfidence::Strong));
        assert!(selection.winner.std_dev_nanos > 0.0);
    }
}
