//! In-memory tabular dataset representation and canonical content hashing.
//!
//! The harness operates on one rectangular table: `features` (rows ×
//! columns), one `targets` value per row (a regression target or a `{0, 1}`
//! anomaly label — the task defines the meaning), an optional per-row
//! `groups` key (machine / run / unit identity, the leakage boundary for
//! grouped splits) and an optional per-row `time_index` (the ordering key for
//! temporal splits and burst contamination).
//!
//! # Canonical hash
//!
//! [`TabularDataset::content_sha256`] hashes a versioned, explicitly
//! little-endian byte serialization: the format tag, the dimensions, every
//! feature value as IEEE-754 bits, the targets, then presence-tagged groups
//! and time indices. Two datasets hash equal **iff** they are bit-identical
//! in every field the harness can observe — the hash is the identity that
//! split manifests and contamination manifests record. No timestamps, no
//! environment data.

use core::fmt;

use sha2::{Digest, Sha256};

/// One rectangular benchmark table.
#[derive(Clone, Debug, PartialEq)]
pub struct TabularDataset {
    /// Row-major feature rows; every row must have the same length.
    pub features: Vec<Vec<f64>>,
    /// One target per row (task-defined meaning).
    pub targets: Vec<f64>,
    /// Optional per-row group key (the grouped-split leakage boundary).
    pub groups: Option<Vec<u64>>,
    /// Optional per-row temporal ordering key.
    pub time_index: Option<Vec<u64>>,
}

/// Typed dataset-shape errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DatasetError {
    /// The dataset has zero rows.
    EmptyDataset,
    /// The dataset has zero feature columns.
    EmptyFeatureRow,
    /// A feature row's length differs from the first row's.
    RaggedFeatureRow {
        /// Index of the offending row.
        row: usize,
        /// Length of the first row.
        expected: usize,
        /// Length of the offending row.
        found: usize,
    },
    /// `targets` length differs from the number of feature rows.
    TargetLengthMismatch {
        /// Number of feature rows.
        rows: usize,
        /// `targets.len()`.
        found: usize,
    },
    /// `groups` is present with a length differing from the row count.
    GroupLengthMismatch {
        /// Number of feature rows.
        rows: usize,
        /// `groups.len()`.
        found: usize,
    },
    /// `time_index` is present with a length differing from the row count.
    TimeIndexLengthMismatch {
        /// Number of feature rows.
        rows: usize,
        /// `time_index.len()`.
        found: usize,
    },
    /// A feature value is `NaN` or `±∞`.
    NonFiniteFeature {
        /// Row of the offending value.
        row: usize,
        /// Column of the offending value.
        column: usize,
    },
    /// A target value is `NaN` or `±∞`.
    NonFiniteTarget {
        /// Row of the offending value.
        row: usize,
    },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyDataset => formatter.write_str("dataset has zero rows"),
            Self::EmptyFeatureRow => formatter.write_str("dataset has zero feature columns"),
            Self::RaggedFeatureRow {
                row,
                expected,
                found,
            } => write!(
                formatter,
                "feature row {row} has {found} values, expected {expected}"
            ),
            Self::TargetLengthMismatch { rows, found } =>
            {
                write!(formatter, "targets has {found} values for {rows} rows")
            },
            Self::GroupLengthMismatch { rows, found } =>
            {
                write!(formatter, "groups has {found} values for {rows} rows")
            },
            Self::TimeIndexLengthMismatch { rows, found } =>
            {
                write!(formatter, "time_index has {found} values for {rows} rows")
            },
            Self::NonFiniteFeature { row, column } =>
            {
                write!(formatter, "feature ({row}, {column}) is not finite")
            },
            Self::NonFiniteTarget { row } => write!(formatter, "target {row} is not finite"),
        }
    }
}

impl std::error::Error for DatasetError {}

impl TabularDataset {
    /// Number of rows.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.features.len()
    }

    /// Number of feature columns (0 for an empty dataset).
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.features.first().map_or(0, Vec::len)
    }

    /// Validates rectangularity, aligned lengths and finiteness.
    pub fn validate(&self) -> Result<(), DatasetError> {
        let rows = self.features.len();

        if rows == 0
        {
            return Err(DatasetError::EmptyDataset);
        }

        let columns = self.features[0].len();

        if columns == 0
        {
            return Err(DatasetError::EmptyFeatureRow);
        }

        for (row, values) in self.features.iter().enumerate()
        {
            if values.len() != columns
            {
                return Err(DatasetError::RaggedFeatureRow {
                    row,
                    expected: columns,
                    found: values.len(),
                });
            }

            for (column, value) in values.iter().enumerate()
            {
                if !value.is_finite()
                {
                    return Err(DatasetError::NonFiniteFeature { row, column });
                }
            }
        }

        if self.targets.len() != rows
        {
            return Err(DatasetError::TargetLengthMismatch {
                rows,
                found: self.targets.len(),
            });
        }

        for (row, target) in self.targets.iter().enumerate()
        {
            if !target.is_finite()
            {
                return Err(DatasetError::NonFiniteTarget { row });
            }
        }

        if let Some(groups) = &self.groups
            && groups.len() != rows
        {
            return Err(DatasetError::GroupLengthMismatch {
                rows,
                found: groups.len(),
            });
        }

        if let Some(time_index) = &self.time_index
            && time_index.len() != rows
        {
            return Err(DatasetError::TimeIndexLengthMismatch {
                rows,
                found: time_index.len(),
            });
        }

        Ok(())
    }

    /// Materializes the sub-dataset of the given rows, in the given order
    /// (splits pass ascending indices, so subsets stay canonically ordered).
    ///
    /// # Panics
    ///
    /// Panics if any index is out of range — split indices come from
    /// [`crate::splits::split_dataset`] on the same dataset, so an
    /// out-of-range index is harness misuse, not data corruption.
    #[must_use]
    pub fn select_rows(&self, rows: &[usize]) -> Self {
        Self {
            features: rows.iter().map(|&row| self.features[row].clone()).collect(),
            targets: rows.iter().map(|&row| self.targets[row]).collect(),
            groups: self
                .groups
                .as_ref()
                .map(|groups| rows.iter().map(|&row| groups[row]).collect()),
            time_index: self
                .time_index
                .as_ref()
                .map(|time_index| rows.iter().map(|&row| time_index[row]).collect()),
        }
    }

    /// Canonical SHA-256 of the dataset content (see the module docs for the
    /// exact byte layout). Validation is a precondition: hash only datasets
    /// that pass [`TabularDataset::validate`].
    #[must_use]
    pub fn content_sha256(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(b"scirust-srcc-bench:tabular:v1");
        hasher.update((self.features.len() as u64).to_le_bytes());
        hasher.update((self.feature_count() as u64).to_le_bytes());

        for row in &self.features
        {
            for value in row
            {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }

        for target in &self.targets
        {
            hasher.update(target.to_bits().to_le_bytes());
        }

        match &self.groups
        {
            None => hasher.update([0u8]),
            Some(groups) =>
            {
                hasher.update([1u8]);

                for group in groups
                {
                    hasher.update(group.to_le_bytes());
                }
            },
        }

        match &self.time_index
        {
            None => hasher.update([0u8]),
            Some(time_index) =>
            {
                hasher.update([1u8]);

                for time in time_index
                {
                    hasher.update(time.to_le_bytes());
                }
            },
        }

        let digest = hasher.finalize();

        let mut hex = String::with_capacity(64);

        for byte in digest
        {
            use core::fmt::Write;

            write!(hex, "{byte:02x}").expect("writing to a String is infallible");
        }

        hex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> TabularDataset {
        TabularDataset {
            features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            targets: vec![0.5, -0.5],
            groups: Some(vec![7, 8]),
            time_index: None,
        }
    }

    #[test]
    fn valid_dataset_passes_and_hashes_deterministically() {
        let dataset = small();

        assert_eq!(dataset.validate(), Ok(()));
        assert_eq!(dataset.content_sha256(), dataset.content_sha256());
        assert_eq!(dataset.content_sha256().len(), 64);
    }

    #[test]
    fn every_observable_field_changes_the_hash() {
        let base = small();

        let mut feature_changed = base.clone();
        feature_changed.features[1][0] = 3.0000000001;

        let mut target_changed = base.clone();
        target_changed.targets[0] = 0.75;

        let mut group_changed = base.clone();
        group_changed.groups = Some(vec![7, 9]);

        let mut group_dropped = base.clone();
        group_dropped.groups = None;

        let mut time_added = base.clone();
        time_added.time_index = Some(vec![0, 1]);

        for variant in [
            feature_changed,
            target_changed,
            group_changed,
            group_dropped,
            time_added,
        ]
        {
            assert_ne!(base.content_sha256(), variant.content_sha256());
        }
    }

    #[test]
    fn negative_zero_and_positive_zero_hash_differently() {
        // The hash is bit-level identity, stricter than `==` on floats.
        let mut positive = small();
        positive.targets[0] = 0.0;

        let mut negative = small();
        negative.targets[0] = -0.0;

        assert_ne!(positive.content_sha256(), negative.content_sha256());
    }

    #[test]
    fn shape_violations_are_typed() {
        assert_eq!(
            TabularDataset {
                features: vec![],
                targets: vec![],
                groups: None,
                time_index: None,
            }
            .validate(),
            Err(DatasetError::EmptyDataset),
        );

        assert_eq!(
            TabularDataset {
                features: vec![vec![1.0], vec![1.0, 2.0]],
                targets: vec![0.0, 0.0],
                groups: None,
                time_index: None,
            }
            .validate(),
            Err(DatasetError::RaggedFeatureRow {
                row: 1,
                expected: 1,
                found: 2
            }),
        );

        assert_eq!(
            TabularDataset {
                features: vec![vec![1.0]],
                targets: vec![],
                groups: None,
                time_index: None,
            }
            .validate(),
            Err(DatasetError::TargetLengthMismatch { rows: 1, found: 0 }),
        );

        assert_eq!(
            TabularDataset {
                features: vec![vec![f64::NAN]],
                targets: vec![0.0],
                groups: None,
                time_index: None,
            }
            .validate(),
            Err(DatasetError::NonFiniteFeature { row: 0, column: 0 }),
        );

        assert_eq!(
            TabularDataset {
                features: vec![vec![1.0]],
                targets: vec![f64::INFINITY],
                groups: None,
                time_index: None,
            }
            .validate(),
            Err(DatasetError::NonFiniteTarget { row: 0 }),
        );

        assert_eq!(
            TabularDataset {
                features: vec![vec![1.0]],
                targets: vec![0.0],
                groups: Some(vec![1, 2]),
                time_index: None,
            }
            .validate(),
            Err(DatasetError::GroupLengthMismatch { rows: 1, found: 2 }),
        );

        assert_eq!(
            TabularDataset {
                features: vec![vec![1.0]],
                targets: vec![0.0],
                groups: None,
                time_index: Some(vec![]),
            }
            .validate(),
            Err(DatasetError::TimeIndexLengthMismatch { rows: 1, found: 0 }),
        );
    }
}
