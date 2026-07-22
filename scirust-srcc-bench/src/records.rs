//! `BenchRecord` emission and run metadata.
//!
//! One place encodes which metrics exist for which output shape — the
//! capability-honesty rule as code:
//!
//! - predictions → error metrics (RMSE, MAE, median/worst absolute error);
//! - anomaly **scores** + labels → AUROC plus threshold-derived counts;
//! - anomaly **labels** only → threshold-derived counts, **no AUROC row at
//!   all** (a metric that cannot exist emits nothing, never `0.0`);
//! - alarms + onset → `detected` (0/1), `detection_delay_steps` (only when
//!   detected), `false_alarm_count`, and `false_alarm_rate` when there are
//!   pre-onset steps to rate.
//!
//! Undefined threshold-derived rates (no predicted positives, one-class
//! label sets) are simply absent from the emitted rows.
//!
//! [`RunMetadata`] carries the environment identity (git commit, toolchain,
//! configuration hash) of a published result set. It is serialized **next
//! to** result files, never inside hashed scientific content — environment
//! identity may vary without changing the science.

use scirust_bench_schema::BenchRecord;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::metrics::{
    DetectionOutcome, MetricError, auroc, confusion_counts, detection_report, mean_absolute_error,
    median_absolute_error, rmse, worst_absolute_error,
};

/// Environment identity of a published result set (kept outside hashed
/// scientific content).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunMetadata {
    /// Git commit SHA of the code that produced the results.
    pub git_commit: String,
    /// Canonical checksum of the evaluated dataset.
    pub dataset_sha256: String,
    /// Checksum of the canonical configuration string ([`sha256_hex`]).
    pub configuration_sha256: String,
    /// Toolchain identifier (e.g. `"nightly-2026-07-02"`).
    pub toolchain: String,
    /// Cargo feature flags in effect.
    pub feature_flags: Vec<String>,
}

/// Lowercase-hex SHA-256 of arbitrary bytes (for configuration strings and
/// other canonical content).
#[must_use]
pub fn sha256_hex(content: &[u8]) -> String {
    let digest = Sha256::digest(content);

    let mut hex = String::with_capacity(64);

    for byte in digest
    {
        use core::fmt::Write;

        write!(hex, "{byte:02x}").expect("writing to a String is infallible");
    }

    hex
}

/// Identifies one benchmark cell: everything but the metric and value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordKey {
    /// The computation benchmarked (e.g. `"industrial_demo/regression"`).
    pub kernel: String,
    /// The workload (e.g. `"synthetic_plant/coherent_cluster_0.2"`).
    pub dataset: String,
    /// The method (an adapter's stable name).
    pub method: String,
    /// The seed behind the randomness of this cell.
    pub seed: u64,
}

impl RecordKey {
    fn record(&self, metric: &str, value: f64) -> BenchRecord {
        BenchRecord::new(
            self.kernel.clone(),
            self.dataset.clone(),
            self.method.clone(),
            self.seed,
            metric,
            value,
        )
    }
}

/// Error-metric rows for regression predictions.
pub fn regression_records(
    key: &RecordKey,
    predictions: &[f64],
    references: &[f64],
) -> Result<Vec<BenchRecord>, MetricError> {
    Ok(vec![
        key.record("rmse", rmse(predictions, references)?),
        key.record("mae", mean_absolute_error(predictions, references)?),
        key.record(
            "median_absolute_error",
            median_absolute_error(predictions, references)?,
        ),
        key.record(
            "worst_absolute_error",
            worst_absolute_error(predictions, references)?,
        ),
    ])
}

fn threshold_records(
    key: &RecordKey,
    scores: &[f64],
    labels: &[f64],
    threshold: f64,
) -> Result<Vec<BenchRecord>, MetricError> {
    let counts = confusion_counts(scores, labels, threshold)?;
    let mut records = Vec::new();

    for (metric, value) in [
        ("precision", counts.precision()),
        ("recall", counts.recall()),
        ("f1", counts.f1()),
        ("balanced_accuracy", counts.balanced_accuracy()),
        ("false_alarm_rate", counts.false_alarm_rate()),
        ("missed_detection_rate", counts.missed_detection_rate()),
    ]
    {
        if let Some(value) = value
        {
            records.push(key.record(metric, value));
        }
    }

    Ok(records)
}

/// AUROC plus threshold-derived rows for score-producing detectors.
pub fn anomaly_score_records(
    key: &RecordKey,
    scores: &[f64],
    labels: &[f64],
    threshold: f64,
) -> Result<Vec<BenchRecord>, MetricError> {
    let mut records = vec![key.record("auroc", auroc(scores, labels)?)];

    records.extend(threshold_records(key, scores, labels, threshold)?);

    Ok(records)
}

/// Threshold-derived rows only, for label-only detectors — no AUROC row is
/// ever fabricated.
pub fn anomaly_label_records(
    key: &RecordKey,
    flags: &[bool],
    labels: &[f64],
) -> Result<Vec<BenchRecord>, MetricError> {
    let scores: Vec<f64> = flags
        .iter()
        .map(|&flag| f64::from(u8::from(flag)))
        .collect();

    threshold_records(key, &scores, labels, 0.5)
}

/// Detection rows for stream alarms against a known onset.
pub fn alarm_records(
    key: &RecordKey,
    alarm_steps: &[usize],
    onset: usize,
    stream_length: usize,
) -> Result<Vec<BenchRecord>, MetricError> {
    let report = detection_report(alarm_steps, onset, stream_length)?;

    let mut records = Vec::new();

    match report.outcome
    {
        DetectionOutcome::Detected { delay } =>
        {
            records.push(key.record("detected", 1.0));
            records.push(key.record("detection_delay_steps", delay as f64));
        },
        DetectionOutcome::Missed =>
        {
            records.push(key.record("detected", 0.0));
        },
    }

    records.push(key.record("false_alarm_count", report.false_alarms as f64));

    if report.pre_onset_steps > 0
    {
        records.push(key.record(
            "false_alarm_rate",
            report.false_alarms as f64 / report.pre_onset_steps as f64,
        ));
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> RecordKey {
        RecordKey {
            kernel: "demo/kernel".into(),
            dataset: "demo/data".into(),
            method: "demo_method".into(),
            seed: 7,
        }
    }

    #[test]
    fn regression_records_carry_all_four_error_metrics() {
        let records = regression_records(&key(), &[1.0, 2.0], &[1.5, 2.0]).unwrap();

        let metrics: Vec<&str> = records.iter().map(|r| r.metric.as_str()).collect();

        assert_eq!(
            metrics,
            vec![
                "rmse",
                "mae",
                "median_absolute_error",
                "worst_absolute_error"
            ],
        );

        assert!(
            records
                .iter()
                .all(|r| r.seed == 7 && r.method == "demo_method")
        );
    }

    #[test]
    fn score_records_include_auroc_label_records_do_not() {
        let labels = [1.0, 0.0, 1.0, 0.0];

        let with_scores =
            anomaly_score_records(&key(), &[0.9, 0.1, 0.8, 0.2], &labels, 0.5).unwrap();

        assert!(with_scores.iter().any(|r| r.metric == "auroc"));

        let label_only =
            anomaly_label_records(&key(), &[true, false, true, false], &labels).unwrap();

        assert!(
            !label_only.iter().any(|r| r.metric == "auroc"),
            "label-only detectors must not get an AUROC row",
        );

        assert!(label_only.iter().any(|r| r.metric == "balanced_accuracy"));
    }

    #[test]
    fn undefined_rates_are_absent_not_zero() {
        // Nothing predicted positive: precision/f1 undefined and absent.
        let records =
            anomaly_score_records(&key(), &[0.1, 0.2, 0.3, 0.4], &[1.0, 0.0, 1.0, 0.0], 0.9)
                .unwrap();

        assert!(!records.iter().any(|r| r.metric == "precision"));
        assert!(!records.iter().any(|r| r.metric == "f1"));
        assert!(records.iter().any(|r| r.metric == "recall"));
    }

    #[test]
    fn alarm_records_distinguish_detected_from_missed() {
        let detected = alarm_records(&key(), &[2, 12], 10, 20).unwrap();

        assert!(
            detected
                .iter()
                .any(|r| r.metric == "detected" && r.value == 1.0)
        );
        assert!(
            detected
                .iter()
                .any(|r| r.metric == "detection_delay_steps" && r.value == 2.0)
        );
        assert!(
            detected
                .iter()
                .any(|r| r.metric == "false_alarm_count" && r.value == 1.0)
        );

        let missed = alarm_records(&key(), &[2], 10, 20).unwrap();

        assert!(
            missed
                .iter()
                .any(|r| r.metric == "detected" && r.value == 0.0)
        );
        assert!(
            !missed.iter().any(|r| r.metric == "detection_delay_steps"),
            "a missed detection has no delay row",
        );
    }

    #[test]
    fn sha256_hex_matches_the_known_empty_digest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        );

        assert_ne!(sha256_hex(b"a"), sha256_hex(b"b"));
    }

    #[test]
    fn run_metadata_round_trips_through_json() {
        let metadata = RunMetadata {
            git_commit: "0123456789abcdef".into(),
            dataset_sha256: sha256_hex(b"data"),
            configuration_sha256: sha256_hex(b"config"),
            toolchain: "nightly-2026-07-02".into(),
            feature_flags: vec!["default".into()],
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let back: RunMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(back, metadata);
    }
}
