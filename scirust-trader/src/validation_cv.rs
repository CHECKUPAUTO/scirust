//! Purged and embargoed cross-validation for forward-labelled market data.
//!
//! Each [`MlRow`](crate::ml_dataset::MlRow) represents a label interval from
//! `ts_ms` until `target_ts_ms`.  A training row is purged whenever that interval
//! overlaps the aggregate label interval of the held-out fold.  An explicit
//! number of observation rows immediately following the test fold is embargoed.

use serde::{Deserialize, Serialize};

use crate::ml_dataset::{MlDatasetError, TimeSeriesMlDataset};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PurgedCvConfig {
    pub n_splits: usize,
    /// Number of observations immediately after each test fold excluded from
    /// training.  This is expressed in rows rather than wall-clock time so the
    /// policy remains explicit for irregularly sampled datasets.
    pub embargo_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PurgedFold {
    pub fold_index: usize,
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub purged_indices: Vec<usize>,
    pub embargoed_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PurgedCvError {
    Dataset(MlDatasetError),
    InvalidSplitCount,
    DatasetTooSmall,
}

impl From<MlDatasetError> for PurgedCvError {
    fn from(value: MlDatasetError) -> Self {
        Self::Dataset(value)
    }
}

#[inline]
fn intervals_overlap(a_start: i64, a_end: i64, b_start: i64, b_end: i64) -> bool {
    a_start <= b_end && a_end >= b_start
}

/// Build contiguous K-fold test blocks with interval purging and post-test
/// embargo.  No shuffling is performed.
pub fn purged_kfold(
    dataset: &TimeSeriesMlDataset,
    config: PurgedCvConfig,
) -> Result<Vec<PurgedFold>, PurgedCvError> {
    dataset.validate()?;
    let n = dataset.rows.len();
    if config.n_splits < 2 || config.n_splits > n
    {
        return Err(PurgedCvError::InvalidSplitCount);
    }
    if n < config.n_splits
    {
        return Err(PurgedCvError::DatasetTooSmall);
    }

    let base = n / config.n_splits;
    let remainder = n % config.n_splits;
    let mut folds = Vec::with_capacity(config.n_splits);
    let mut start = 0usize;

    for fold_index in 0..config.n_splits
    {
        let fold_len = base + usize::from(fold_index < remainder);
        let end = start + fold_len;
        let test_indices: Vec<usize> = (start..end).collect();
        let test_start_ts = dataset.rows[start].ts_ms;
        let test_label_end_ts = dataset.rows[start..end]
            .iter()
            .map(|row| row.target_ts_ms)
            .max()
            .expect("test fold is non-empty");
        let embargo_end = end.saturating_add(config.embargo_rows).min(n);

        let mut train_indices = Vec::new();
        let mut purged_indices = Vec::new();
        let mut embargoed_indices = Vec::new();

        for (index, row) in dataset.rows.iter().enumerate()
        {
            if (start..end).contains(&index)
            {
                continue;
            }
            if (end..embargo_end).contains(&index)
            {
                embargoed_indices.push(index);
                continue;
            }
            if intervals_overlap(
                row.ts_ms,
                row.target_ts_ms,
                test_start_ts,
                test_label_end_ts,
            )
            {
                purged_indices.push(index);
                continue;
            }
            train_indices.push(index);
        }

        folds.push(PurgedFold {
            fold_index,
            train_indices,
            test_indices,
            purged_indices,
            embargoed_indices,
        });
        start = end;
    }
    Ok(folds)
}

/// Verify that none of the returned training-label intervals overlaps its test
/// fold's aggregate label interval.
pub fn fold_is_leakage_free(dataset: &TimeSeriesMlDataset, fold: &PurgedFold) -> bool {
    let Some(&first_test) = fold.test_indices.first()
    else
    {
        return false;
    };
    let test_start_ts = dataset.rows[first_test].ts_ms;
    let Some(test_label_end_ts) = fold
        .test_indices
        .iter()
        .map(|&i| dataset.rows[i].target_ts_ms)
        .max()
    else
    {
        return false;
    };
    fold.train_indices.iter().all(|&i| {
        let row = &dataset.rows[i];
        !intervals_overlap(
            row.ts_ms,
            row.target_ts_ms,
            test_start_ts,
            test_label_end_ts,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_dataset::{FeatureProvenance, MlRow};

    fn dataset(target_horizon_ms: i64) -> TimeSeriesMlDataset {
        TimeSeriesMlDataset {
            feature_provenance: vec![FeatureProvenance {
                name: "x".into(),
                source: "synthetic".into(),
                transformation: "identity".into(),
            }],
            rows: (0..12)
                .map(|i| MlRow {
                    ts_ms: i * 10,
                    feature_available_ts_ms: i * 10,
                    target_ts_ms: i * 10 + target_horizon_ms,
                    features: vec![i as f32],
                    target: i as f32,
                })
                .collect(),
        }
    }

    #[test]
    fn contiguous_folds_cover_each_observation_once_as_test() {
        let d = dataset(5);
        let folds = purged_kfold(
            &d,
            PurgedCvConfig {
                n_splits: 3,
                embargo_rows: 1,
            },
        )
        .unwrap();
        let mut all_test: Vec<usize> = folds
            .iter()
            .flat_map(|fold| fold.test_indices.iter().copied())
            .collect();
        all_test.sort_unstable();
        assert_eq!(all_test, (0..12).collect::<Vec<_>>());
        assert!(folds.iter().all(|fold| fold_is_leakage_free(&d, fold)));
    }

    #[test]
    fn overlapping_forward_label_is_purged() {
        let d = dataset(15);
        let folds = purged_kfold(
            &d,
            PurgedCvConfig {
                n_splits: 3,
                embargo_rows: 0,
            },
        )
        .unwrap();
        // Fold 1 tests rows 4..8 (starts at t=40). Row 3 labels through t=45.
        assert!(folds[1].purged_indices.contains(&3));
        assert!(!folds[1].train_indices.contains(&3));
    }

    #[test]
    fn embargo_removes_rows_after_test_block() {
        let d = dataset(5);
        let folds = purged_kfold(
            &d,
            PurgedCvConfig {
                n_splits: 3,
                embargo_rows: 2,
            },
        )
        .unwrap();
        assert_eq!(folds[0].embargoed_indices, vec![4, 5]);
        assert!(!folds[0].train_indices.contains(&4));
    }
}
