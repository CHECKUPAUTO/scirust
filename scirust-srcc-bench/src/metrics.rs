//! Deterministic evaluation metrics, with capability honesty.
//!
//! Every function validates shape and finiteness and returns typed errors —
//! a metric that cannot be computed is an error or a typed absence, never a
//! silent `0.0`. Conventions, stated precisely:
//!
//! - **AUROC** is rank-based (Mann–Whitney) with average ranks on ties, and
//!   exists only for *score-producing* detectors; label-only detectors
//!   cannot get one (the type system upstream enforces this, this module
//!   additionally refuses degenerate label sets — all-positive or
//!   all-negative — as [`MetricError::DegenerateLabels`]);
//! - **binary classification counts** use the caller's explicit threshold
//!   convention: predicted positive iff `score > threshold`;
//! - **detection delay** is `first alarm at or after the onset − onset` in
//!   *steps of the evaluation stream*; a run with no such alarm is a typed
//!   [`DetectionOutcome::Missed`], not a sentinel number. Alarms strictly
//!   before the onset are false alarms and are counted separately;
//! - **cluster recovery** is the pair-counting Rand index and the adjusted
//!   Rand index (permutation-invariant by construction; the ARI of a
//!   single-cluster-vs-anything comparison has a zero denominator and is a
//!   typed [`MetricError::DegeneratePartition`]);
//! - quantiles use the same sorted-midpoint convention as the rest of the
//!   program (`total_cmp` sort; even-length median is the mean of the two
//!   central order statistics).

use core::fmt;

/// Typed metric errors.
#[derive(Clone, Debug, PartialEq)]
pub enum MetricError {
    /// The input is empty.
    EmptyInput,
    /// Two aligned vectors have different lengths.
    LengthMismatch {
        /// Left length.
        left: usize,
        /// Right length.
        right: usize,
    },
    /// A value is `NaN` or `±∞`.
    NonFiniteValue {
        /// Index of the offending value.
        index: usize,
    },
    /// A label is neither `0.0` nor `1.0`.
    NonBinaryLabel {
        /// Index of the offending label.
        index: usize,
    },
    /// AUROC / balanced accuracy need both classes present.
    DegenerateLabels,
    /// The adjusted Rand index is undefined for this pair of partitions.
    DegeneratePartition,
    /// The onset index lies outside the evaluation stream.
    OnsetOutOfRange {
        /// The requested onset.
        onset: usize,
        /// The stream length.
        length: usize,
    },
}

impl fmt::Display for MetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyInput => formatter.write_str("metric input is empty"),
            Self::LengthMismatch { left, right } =>
            {
                write!(formatter, "aligned inputs have lengths {left} and {right}")
            },
            Self::NonFiniteValue { index } =>
            {
                write!(formatter, "value at index {index} is not finite")
            },
            Self::NonBinaryLabel { index } =>
            {
                write!(formatter, "label at index {index} is neither 0 nor 1")
            },
            Self::DegenerateLabels =>
            {
                formatter.write_str("metric needs both classes present in the labels")
            },
            Self::DegeneratePartition =>
            {
                formatter.write_str("adjusted Rand index is undefined for these partitions")
            },
            Self::OnsetOutOfRange { onset, length } =>
            {
                write!(formatter, "onset {onset} outside stream of length {length}")
            },
        }
    }
}

impl std::error::Error for MetricError {}

fn check_aligned_finite(left: &[f64], right: &[f64]) -> Result<(), MetricError> {
    if left.is_empty()
    {
        return Err(MetricError::EmptyInput);
    }

    if left.len() != right.len()
    {
        return Err(MetricError::LengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }

    for (index, value) in left.iter().chain(right).enumerate()
    {
        if !value.is_finite()
        {
            return Err(MetricError::NonFiniteValue {
                index: index % left.len(),
            });
        }
    }

    Ok(())
}

fn check_binary(labels: &[f64]) -> Result<(), MetricError> {
    for (index, &label) in labels.iter().enumerate()
    {
        if label != 0.0 && label != 1.0
        {
            return Err(MetricError::NonBinaryLabel { index });
        }
    }

    Ok(())
}

/// Root-mean-square error of predictions against references.
pub fn rmse(predictions: &[f64], references: &[f64]) -> Result<f64, MetricError> {
    check_aligned_finite(predictions, references)?;

    let sum: f64 = predictions
        .iter()
        .zip(references)
        .map(|(p, r)| (p - r).powi(2))
        .sum();

    Ok((sum / predictions.len() as f64).sqrt())
}

/// Mean absolute error.
pub fn mean_absolute_error(predictions: &[f64], references: &[f64]) -> Result<f64, MetricError> {
    check_aligned_finite(predictions, references)?;

    let sum: f64 = predictions
        .iter()
        .zip(references)
        .map(|(p, r)| (p - r).abs())
        .sum();

    Ok(sum / predictions.len() as f64)
}

/// Median absolute error (sorted-midpoint median of `|p − r|`).
pub fn median_absolute_error(predictions: &[f64], references: &[f64]) -> Result<f64, MetricError> {
    check_aligned_finite(predictions, references)?;

    let mut errors: Vec<f64> = predictions
        .iter()
        .zip(references)
        .map(|(p, r)| (p - r).abs())
        .collect();

    errors.sort_by(f64::total_cmp);

    let n = errors.len();

    if n % 2 == 1
    {
        Ok(errors[n / 2])
    }
    else
    {
        Ok((errors[n / 2 - 1] + errors[n / 2]) / 2.0)
    }
}

/// Largest absolute error.
pub fn worst_absolute_error(predictions: &[f64], references: &[f64]) -> Result<f64, MetricError> {
    check_aligned_finite(predictions, references)?;

    Ok(predictions
        .iter()
        .zip(references)
        .map(|(p, r)| (p - r).abs())
        .fold(0.0, f64::max))
}

/// The four confusion counts under the `score > threshold` convention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfusionCounts {
    /// Anomalies flagged as anomalies.
    pub true_positives: usize,
    /// Clean rows flagged as anomalies.
    pub false_positives: usize,
    /// Clean rows passed as clean.
    pub true_negatives: usize,
    /// Anomalies passed as clean.
    pub false_negatives: usize,
}

/// Confusion counts of scores against `{0, 1}` labels at a threshold.
pub fn confusion_counts(
    scores: &[f64],
    labels: &[f64],
    threshold: f64,
) -> Result<ConfusionCounts, MetricError> {
    check_aligned_finite(scores, labels)?;
    check_binary(labels)?;

    if !threshold.is_finite()
    {
        return Err(MetricError::NonFiniteValue { index: 0 });
    }

    let mut counts = ConfusionCounts {
        true_positives: 0,
        false_positives: 0,
        true_negatives: 0,
        false_negatives: 0,
    };

    for (&score, &label) in scores.iter().zip(labels)
    {
        let predicted_positive = score > threshold;
        let actual_positive = label == 1.0;

        match (predicted_positive, actual_positive)
        {
            (true, true) => counts.true_positives += 1,
            (true, false) => counts.false_positives += 1,
            (false, false) => counts.true_negatives += 1,
            (false, true) => counts.false_negatives += 1,
        }
    }

    Ok(counts)
}

impl ConfusionCounts {
    /// Precision (`None` when nothing was predicted positive).
    #[must_use]
    pub fn precision(&self) -> Option<f64> {
        let predicted = self.true_positives + self.false_positives;

        (predicted > 0).then(|| self.true_positives as f64 / predicted as f64)
    }

    /// Recall (`None` when there are no actual positives).
    #[must_use]
    pub fn recall(&self) -> Option<f64> {
        let actual = self.true_positives + self.false_negatives;

        (actual > 0).then(|| self.true_positives as f64 / actual as f64)
    }

    /// F1 (`None` when precision or recall is undefined or both are zero).
    #[must_use]
    pub fn f1(&self) -> Option<f64> {
        let precision = self.precision()?;
        let recall = self.recall()?;
        let sum = precision + recall;

        (sum > 0.0).then(|| 2.0 * precision * recall / sum)
    }

    /// False-alarm rate (`None` when there are no actual negatives).
    #[must_use]
    pub fn false_alarm_rate(&self) -> Option<f64> {
        let actual_negatives = self.false_positives + self.true_negatives;

        (actual_negatives > 0).then(|| self.false_positives as f64 / actual_negatives as f64)
    }

    /// Missed-detection rate (`None` when there are no actual positives).
    #[must_use]
    pub fn missed_detection_rate(&self) -> Option<f64> {
        let actual_positives = self.true_positives + self.false_negatives;

        (actual_positives > 0).then(|| self.false_negatives as f64 / actual_positives as f64)
    }

    /// Balanced accuracy (`None` unless both classes are present).
    #[must_use]
    pub fn balanced_accuracy(&self) -> Option<f64> {
        let recall = self.recall()?;
        let actual_negatives = self.false_positives + self.true_negatives;
        let specificity =
            (actual_negatives > 0).then(|| self.true_negatives as f64 / actual_negatives as f64)?;

        Some((recall + specificity) / 2.0)
    }
}

/// Rank-based AUROC with average ranks on ties. Requires both classes.
pub fn auroc(scores: &[f64], labels: &[f64]) -> Result<f64, MetricError> {
    check_aligned_finite(scores, labels)?;
    check_binary(labels)?;

    let positives = labels.iter().filter(|&&label| label == 1.0).count();
    let negatives = labels.len() - positives;

    if positives == 0 || negatives == 0
    {
        return Err(MetricError::DegenerateLabels);
    }

    // Sort indices by score ascending; assign average ranks to tied runs.
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));

    let mut ranks = vec![0.0; scores.len()];
    let mut start = 0usize;

    while start < order.len()
    {
        let mut end = start;

        while end + 1 < order.len() && scores[order[end + 1]] == scores[order[start]]
        {
            end += 1;
        }

        // Ranks are 1-based; a tied run [start, end] shares the average rank.
        let average = ((start + 1 + end + 1) as f64) / 2.0;

        for &index in &order[start..=end]
        {
            ranks[index] = average;
        }

        start = end + 1;
    }

    let positive_rank_sum: f64 = labels
        .iter()
        .zip(&ranks)
        .filter(|(label, _)| **label == 1.0)
        .map(|(_, rank)| *rank)
        .sum();

    let u = positive_rank_sum - (positives * (positives + 1)) as f64 / 2.0;

    Ok(u / (positives * negatives) as f64)
}

/// Outcome of a detection-delay evaluation: a typed state, never a sentinel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectionOutcome {
    /// First alarm at or after the onset, `delay` steps after it.
    Detected {
        /// Steps between onset and first at-or-after alarm.
        delay: usize,
    },
    /// No alarm at or after the onset.
    Missed,
}

/// Detection-delay report over one evaluation stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectionReport {
    /// The typed detection outcome.
    pub outcome: DetectionOutcome,
    /// Alarms strictly before the onset (false alarms).
    pub false_alarms: usize,
    /// Steps strictly before the onset (the false-alarm opportunity count).
    pub pre_onset_steps: usize,
}

/// Evaluates alarm positions against a known onset in a stream of
/// `stream_length` steps.
pub fn detection_report(
    alarm_steps: &[usize],
    onset: usize,
    stream_length: usize,
) -> Result<DetectionReport, MetricError> {
    if onset >= stream_length
    {
        return Err(MetricError::OnsetOutOfRange {
            onset,
            length: stream_length,
        });
    }

    let false_alarms = alarm_steps.iter().filter(|&&step| step < onset).count();

    let outcome = alarm_steps
        .iter()
        .filter(|&&step| step >= onset && step < stream_length)
        .min()
        .map_or(DetectionOutcome::Missed, |&first| {
            DetectionOutcome::Detected {
                delay: first - onset,
            }
        });

    Ok(DetectionReport {
        outcome,
        false_alarms,
        pre_onset_steps: onset,
    })
}

/// Pair-counting Rand index of two partitions (labels as arbitrary ids).
pub fn rand_index(left: &[usize], right: &[usize]) -> Result<f64, MetricError> {
    if left.is_empty()
    {
        return Err(MetricError::EmptyInput);
    }

    if left.len() != right.len()
    {
        return Err(MetricError::LengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }

    if left.len() == 1
    {
        return Ok(1.0);
    }

    let n = left.len();
    let mut agreements = 0usize;

    for i in 0..n
    {
        for j in (i + 1)..n
        {
            let together_left = left[i] == left[j];
            let together_right = right[i] == right[j];

            if together_left == together_right
            {
                agreements += 1;
            }
        }
    }

    Ok(agreements as f64 / (n * (n - 1) / 2) as f64)
}

/// Adjusted Rand index (Hubert–Arabie). Typed error when the expected-index
/// denominator vanishes (e.g. both partitions are single clusters).
pub fn adjusted_rand_index(left: &[usize], right: &[usize]) -> Result<f64, MetricError> {
    if left.is_empty()
    {
        return Err(MetricError::EmptyInput);
    }

    if left.len() != right.len()
    {
        return Err(MetricError::LengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }

    let n = left.len();

    let distinct = |labels: &[usize]| -> Vec<usize> {
        let mut seen: Vec<usize> = Vec::new();

        for &label in labels
        {
            if !seen.contains(&label)
            {
                seen.push(label);
            }
        }

        seen
    };

    let left_ids = distinct(left);
    let right_ids = distinct(right);

    let choose2 = |count: usize| -> f64 { (count * count.saturating_sub(1)) as f64 / 2.0 };

    let mut sum_cells = 0.0;

    for &a in &left_ids
    {
        for &b in &right_ids
        {
            let cell = (0..n).filter(|&i| left[i] == a && right[i] == b).count();
            sum_cells += choose2(cell);
        }
    }

    let sum_left: f64 = left_ids
        .iter()
        .map(|&a| choose2(left.iter().filter(|&&label| label == a).count()))
        .sum();

    let sum_right: f64 = right_ids
        .iter()
        .map(|&b| choose2(right.iter().filter(|&&label| label == b).count()))
        .sum();

    let total = choose2(n);
    let expected = sum_left * sum_right / total;
    let maximum = (sum_left + sum_right) / 2.0;
    let denominator = maximum - expected;

    if denominator == 0.0
    {
        return Err(MetricError::DegeneratePartition);
    }

    Ok((sum_cells - expected) / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_metrics_match_hand_calculations() {
        let predictions = [1.0, 2.0, 4.0, 8.0];
        let references = [1.0, 1.0, 5.0, 4.0];
        // Absolute errors: 0, 1, 1, 4.

        assert_eq!(
            rmse(&predictions, &references).unwrap(),
            ((0.0 + 1.0 + 1.0 + 16.0) / 4.0f64).sqrt(),
        );
        assert_eq!(mean_absolute_error(&predictions, &references).unwrap(), 1.5);
        assert_eq!(
            median_absolute_error(&predictions, &references).unwrap(),
            1.0,
        );
        assert_eq!(
            worst_absolute_error(&predictions, &references).unwrap(),
            4.0,
        );
    }

    #[test]
    fn confusion_counts_and_derived_rates_are_exact() {
        let scores = [0.9, 0.8, 0.3, 0.2, 0.7, 0.1];
        let labels = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let counts = confusion_counts(&scores, &labels, 0.5).unwrap();

        assert_eq!(
            counts,
            ConfusionCounts {
                true_positives: 2,
                false_positives: 1,
                true_negatives: 2,
                false_negatives: 1,
            },
        );

        assert_eq!(counts.precision(), Some(2.0 / 3.0));
        assert_eq!(counts.recall(), Some(2.0 / 3.0));
        assert_eq!(counts.f1(), Some(2.0 / 3.0));
        assert_eq!(counts.false_alarm_rate(), Some(1.0 / 3.0));
        assert_eq!(counts.missed_detection_rate(), Some(1.0 / 3.0));
        assert_eq!(counts.balanced_accuracy(), Some(2.0 / 3.0));
    }

    #[test]
    fn degenerate_confusion_rates_are_typed_absences() {
        let counts = ConfusionCounts {
            true_positives: 0,
            false_positives: 0,
            true_negatives: 3,
            false_negatives: 0,
        };

        assert_eq!(counts.precision(), None);
        assert_eq!(counts.recall(), None);
        assert_eq!(counts.f1(), None);
        assert_eq!(counts.missed_detection_rate(), None);
        assert_eq!(counts.balanced_accuracy(), None);
        assert_eq!(counts.false_alarm_rate(), Some(0.0));
    }

    #[test]
    fn auroc_matches_hand_calculation_and_handles_ties() {
        // Perfect separation.
        assert_eq!(
            auroc(&[0.1, 0.2, 0.8, 0.9], &[0.0, 0.0, 1.0, 1.0]).unwrap(),
            1.0,
        );

        // Anti-separation.
        assert_eq!(
            auroc(&[0.9, 0.8, 0.2, 0.1], &[0.0, 0.0, 1.0, 1.0]).unwrap(),
            0.0,
        );

        // All scores tied: average ranks force exactly 1/2.
        assert_eq!(
            auroc(&[0.5, 0.5, 0.5, 0.5], &[1.0, 0.0, 1.0, 0.0]).unwrap(),
            0.5,
        );

        // Hand-computed mixed case: scores 0.4(+), 0.3(−), 0.4(−), 0.2(+).
        // Pairs (+ vs −): (0.4,0.3)=win, (0.4,0.4)=tie=0.5, (0.2,0.3)=loss,
        // (0.2,0.4)=loss → (1 + 0.5) / 4 = 0.375.
        assert_eq!(
            auroc(&[0.4, 0.3, 0.4, 0.2], &[1.0, 0.0, 0.0, 1.0]).unwrap(),
            0.375,
        );
    }

    #[test]
    fn auroc_requires_both_classes_and_binary_labels() {
        assert_eq!(
            auroc(&[0.1, 0.2], &[1.0, 1.0]),
            Err(MetricError::DegenerateLabels),
        );

        assert_eq!(
            auroc(&[0.1, 0.2], &[1.0, 0.5]),
            Err(MetricError::NonBinaryLabel { index: 1 }),
        );
    }

    #[test]
    fn detection_reports_are_typed() {
        // Onset at step 10 in a 20-step stream.
        let report = detection_report(&[3, 12, 15], 10, 20).unwrap();

        assert_eq!(report.outcome, DetectionOutcome::Detected { delay: 2 });
        assert_eq!(report.false_alarms, 1);
        assert_eq!(report.pre_onset_steps, 10);

        let missed = detection_report(&[3, 7], 10, 20).unwrap();

        assert_eq!(missed.outcome, DetectionOutcome::Missed);
        assert_eq!(missed.false_alarms, 2);

        assert_eq!(
            detection_report(&[1], 20, 20),
            Err(MetricError::OnsetOutOfRange {
                onset: 20,
                length: 20,
            }),
        );
    }

    #[test]
    fn rand_indices_match_hand_calculations() {
        let left = [0, 0, 1, 1];
        let identical = [5, 5, 9, 9];
        let crossed = [0, 1, 0, 1];

        assert_eq!(rand_index(&left, &identical).unwrap(), 1.0);
        assert_eq!(adjusted_rand_index(&left, &identical).unwrap(), 1.0);

        // Crossed 2×2: agreements = pairs split in both = (0,3),(1,2) → 2 of 6.
        assert_eq!(rand_index(&left, &crossed).unwrap(), 2.0 / 6.0);

        // ARI of the crossed case: sum_cells = 0, expected = 2·2/6 = 2/3,
        // maximum = 2 → (0 − 2/3) / (2 − 2/3) = −1/2 in exact arithmetic;
        // 2/3 rounds in IEEE, so compare within one ulp of the quotient.
        let crossed_ari = adjusted_rand_index(&left, &crossed).unwrap();
        assert!((crossed_ari - (-0.5)).abs() < 1.0e-15, "ARI {crossed_ari}");

        // Permutation invariance of label ids.
        assert_eq!(adjusted_rand_index(&[1, 1, 0, 0], &identical).unwrap(), 1.0,);
    }

    #[test]
    fn degenerate_partitions_are_typed() {
        assert_eq!(
            adjusted_rand_index(&[0, 0, 0], &[0, 0, 0]),
            Err(MetricError::DegeneratePartition),
        );

        assert_eq!(rand_index(&[], &[]), Err(MetricError::EmptyInput),);

        assert_eq!(
            rand_index(&[0, 1], &[0]),
            Err(MetricError::LengthMismatch { left: 2, right: 1 }),
        );
    }

    #[test]
    fn shape_and_finiteness_are_enforced() {
        assert_eq!(rmse(&[], &[]), Err(MetricError::EmptyInput));

        assert_eq!(
            rmse(&[1.0], &[1.0, 2.0]),
            Err(MetricError::LengthMismatch { left: 1, right: 2 }),
        );

        assert!(matches!(
            rmse(&[f64::NAN], &[1.0]),
            Err(MetricError::NonFiniteValue { .. })
        ));

        assert!(matches!(
            confusion_counts(&[0.5], &[1.0], f64::NAN),
            Err(MetricError::NonFiniteValue { .. })
        ));
    }
}
