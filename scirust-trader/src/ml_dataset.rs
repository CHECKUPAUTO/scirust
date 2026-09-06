//! Time-ordered market-ML dataset and leakage checks.
//!
//! The trading roadmap requires model comparisons to preserve time ordering and
//! feature provenance. This module provides that contract independently of any
//! particular learner. A row records when its features were actually available
//! and when its target becomes known; invalid temporal relationships are
//! rejected before a model is fit.

use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FeatureProvenance {
    pub name: String,
    pub source: String,
    pub transformation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MlRow {
    /// Observation timestamp represented by the feature vector.
    pub ts_ms: i64,
    /// Latest timestamp of any raw datum contributing to these features.
    pub feature_available_ts_ms: i64,
    /// Timestamp at which the supervised target becomes observable.
    pub target_ts_ms: i64,
    pub features: Vec<f32>,
    pub target: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimeSeriesMlDataset {
    pub feature_provenance: Vec<FeatureProvenance>,
    pub rows: Vec<MlRow>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TimeSplit {
    pub train_start: usize,
    pub train_end: usize,
    pub validation_start: usize,
    pub validation_end: usize,
    pub test_start: usize,
    pub test_end: usize,
}

impl TimeSplit {
    pub fn train(&self) -> Range<usize> {
        self.train_start..self.train_end
    }

    pub fn validation(&self) -> Range<usize> {
        self.validation_start..self.validation_end
    }

    pub fn test(&self) -> Range<usize> {
        self.test_start..self.test_end
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MlDatasetError {
    NoFeatures,
    EmptyDataset,
    DuplicateFeatureName(String),
    FeatureDimensionMismatch { row: usize },
    NonFiniteFeature { row: usize, feature: usize },
    NonFiniteTarget { row: usize },
    NonMonotonicObservationTime { row: usize },
    FeatureFromFuture { row: usize },
    TargetNotInFuture { row: usize },
    TargetOverlapsLaterPartition,
    InvalidSplitFractions,
    DatasetTooSmallForThreeWaySplit,
}

impl TimeSeriesMlDataset {
    pub fn validate(&self) -> Result<(), MlDatasetError> {
        if self.feature_provenance.is_empty() {
            return Err(MlDatasetError::NoFeatures);
        }
        if self.rows.is_empty() {
            return Err(MlDatasetError::EmptyDataset);
        }
        let mut names = std::collections::BTreeSet::new();
        for feature in &self.feature_provenance {
            if feature.name.is_empty() || !names.insert(feature.name.clone()) {
                return Err(MlDatasetError::DuplicateFeatureName(feature.name.clone()));
            }
        }
        let width = self.feature_provenance.len();
        for (row_index, row) in self.rows.iter().enumerate() {
            if row.features.len() != width {
                return Err(MlDatasetError::FeatureDimensionMismatch { row: row_index });
            }
            for (feature_index, value) in row.features.iter().enumerate() {
                if !value.is_finite() {
                    return Err(MlDatasetError::NonFiniteFeature {
                        row: row_index,
                        feature: feature_index,
                    });
                }
            }
            if !row.target.is_finite() {
                return Err(MlDatasetError::NonFiniteTarget { row: row_index });
            }
            if row.feature_available_ts_ms > row.ts_ms {
                return Err(MlDatasetError::FeatureFromFuture { row: row_index });
            }
            if row.target_ts_ms <= row.ts_ms {
                return Err(MlDatasetError::TargetNotInFuture { row: row_index });
            }
            if row_index > 0 && row.ts_ms <= self.rows[row_index - 1].ts_ms {
                return Err(MlDatasetError::NonMonotonicObservationTime { row: row_index });
            }
        }
        Ok(())
    }

    /// Build chronological train/validation/test partitions.
    ///
    /// `train_fraction + validation_fraction` must be strictly below 1. The
    /// remainder is the test set. Each partition receives at least one row.
    /// The boundary check also rejects label overlap: a training target may not
    /// become known at or after the first validation observation, and a
    /// validation target may not become known at or after the first test
    /// observation.
    pub fn time_split(
        &self,
        train_fraction: f32,
        validation_fraction: f32,
    ) -> Result<TimeSplit, MlDatasetError> {
        self.validate()?;
        if !train_fraction.is_finite()
            || !validation_fraction.is_finite()
            || train_fraction <= 0.0
            || validation_fraction <= 0.0
            || train_fraction + validation_fraction >= 1.0
        {
            return Err(MlDatasetError::InvalidSplitFractions);
        }
        if self.rows.len() < 3 {
            return Err(MlDatasetError::DatasetTooSmallForThreeWaySplit);
        }
        let n = self.rows.len();
        let train_end = ((n as f32 * train_fraction).floor() as usize).clamp(1, n - 2);
        let validation_len = ((n as f32 * validation_fraction).floor() as usize).max(1);
        let validation_end = (train_end + validation_len).min(n - 1);
        if validation_end <= train_end {
            return Err(MlDatasetError::DatasetTooSmallForThreeWaySplit);
        }
        let split = TimeSplit {
            train_start: 0,
            train_end,
            validation_start: train_end,
            validation_end,
            test_start: validation_end,
            test_end: n,
        };

        let validation_first_ts = self.rows[split.validation_start].ts_ms;
        if self.rows[split.train_end - 1].target_ts_ms >= validation_first_ts {
            return Err(MlDatasetError::TargetOverlapsLaterPartition);
        }
        let test_first_ts = self.rows[split.test_start].ts_ms;
        if self.rows[split.validation_end - 1].target_ts_ms >= test_first_ts {
            return Err(MlDatasetError::TargetOverlapsLaterPartition);
        }
        Ok(split)
    }

    pub fn features_targets(&self, range: Range<usize>) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut features = Vec::with_capacity(range.len());
        let mut targets = Vec::with_capacity(range.len());
        for row in &self.rows[range] {
            features.push(row.features.clone());
            targets.push(row.target);
        }
        (features, targets)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RegressionMetrics {
    pub n: usize,
    pub mse: f32,
    pub mae: f32,
    /// Fraction whose predicted and realized target signs agree. Zero targets
    /// count as correct only when both prediction and target are zero.
    pub sign_accuracy: f32,
}

pub fn regression_metrics(targets: &[f32], predictions: &[f32]) -> Option<RegressionMetrics> {
    if targets.is_empty() || targets.len() != predictions.len() {
        return None;
    }
    let mut squared = 0.0f32;
    let mut absolute = 0.0f32;
    let mut sign_hits = 0usize;
    for (&target, &prediction) in targets.iter().zip(predictions) {
        if !target.is_finite() || !prediction.is_finite() {
            return None;
        }
        let error = prediction - target;
        squared += error * error;
        absolute += error.abs();
        let same_sign = if target == 0.0 || prediction == 0.0 {
            target == prediction
        } else {
            target.is_sign_positive() == prediction.is_sign_positive()
        };
        sign_hits += usize::from(same_sign);
    }
    let n = targets.len();
    Some(RegressionMetrics {
        n,
        mse: squared / n as f32,
        mae: absolute / n as f32,
        sign_accuracy: sign_hits as f32 / n as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> TimeSeriesMlDataset {
        TimeSeriesMlDataset {
            feature_provenance: vec![FeatureProvenance {
                name: "return_1".to_string(),
                source: "close".to_string(),
                transformation: "lagged return".to_string(),
            }],
            rows: (0..12)
                .map(|i| MlRow {
                    ts_ms: i * 10,
                    feature_available_ts_ms: i * 10,
                    target_ts_ms: i * 10 + 5,
                    features: vec![i as f32],
                    target: i as f32 * 0.1,
                })
                .collect(),
        }
    }

    #[test]
    fn valid_dataset_splits_chronologically() {
        let d = dataset();
        let split = d.time_split(0.5, 0.25).unwrap();
        assert_eq!(split.train(), 0..6);
        assert_eq!(split.validation(), 6..9);
        assert_eq!(split.test(), 9..12);
    }

    #[test]
    fn feature_from_future_is_rejected() {
        let mut d = dataset();
        d.rows[4].feature_available_ts_ms = d.rows[4].ts_ms + 1;
        assert!(matches!(
            d.validate(),
            Err(MlDatasetError::FeatureFromFuture { row: 4 })
        ));
    }

    #[test]
    fn target_overlap_across_split_is_rejected() {
        let mut d = dataset();
        d.rows[5].target_ts_ms = d.rows[6].ts_ms;
        assert!(matches!(
            d.time_split(0.5, 0.25),
            Err(MlDatasetError::TargetOverlapsLaterPartition)
        ));
    }

    #[test]
    fn metrics_are_deterministic() {
        let targets = [1.0, -2.0, 3.0];
        let predictions = [1.5, -1.0, -1.0];
        let a = regression_metrics(&targets, &predictions).unwrap();
        let b = regression_metrics(&targets, &predictions).unwrap();
        assert_eq!(a.mse.to_bits(), b.mse.to_bits());
        assert!((a.sign_accuracy - 2.0 / 3.0).abs() < 1e-6);
    }
}
