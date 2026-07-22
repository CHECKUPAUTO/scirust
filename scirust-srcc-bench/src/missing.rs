//! Train-fitted missing-value policy.
//!
//! Real process data (SECOM) has missing sensor readings. The preregistered
//! pipeline handles them with a policy **fitted on the training split
//! only** and applied unchanged to validation and test — the same
//! no-leakage discipline as every other fitted component:
//!
//! - a feature column is **dropped** when its training missing fraction
//!   exceeds `maximum_missing_fraction`, or when its observed training
//!   values are all identical (zero spread carries no information and
//!   breaks scale estimates);
//! - every remaining missing value (in any split) is **imputed with the
//!   training median** of its column (observed training values only);
//! - the fitted policy records exactly what it did: kept columns, per-kept-
//!   column training medians, dropped columns with the reason — the counts
//!   the preregistration requires published.
//!
//! The transform refuses (typed error) datasets whose feature count does
//! not match the fit, and never produces a non-finite output from finite
//! observed inputs.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::dataset::TabularDataset;

/// Configuration of the missing-value policy.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct MissingValuePolicy {
    /// Maximum tolerated fraction of missing training values per column,
    /// in `[0, 1)`.
    pub maximum_missing_fraction: f64,
}

/// Why a column was dropped.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DropReason {
    /// Training missing fraction exceeded the policy maximum.
    TooManyMissing,
    /// All observed training values are identical (zero spread).
    ConstantColumn,
    /// The column has no observed training value at all.
    AllMissing,
}

/// The fitted, train-only imputation model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FittedImputer {
    /// Feature count the imputer was fitted on.
    pub input_feature_count: usize,
    /// Kept column indices (ascending, into the original feature space).
    pub kept_columns: Vec<usize>,
    /// Training median per kept column (aligned with `kept_columns`).
    pub training_medians: Vec<f64>,
    /// Dropped columns with reasons (ascending).
    pub dropped_columns: Vec<(usize, DropReason)>,
}

/// Typed missing-policy errors.
#[derive(Clone, Debug, PartialEq)]
pub enum MissingPolicyError {
    /// The policy fraction is not a finite value in `[0, 1)`.
    InvalidPolicy {
        /// The offending fraction.
        maximum_missing_fraction: f64,
    },
    /// The training split is empty.
    EmptyTrain,
    /// Fitting dropped every column.
    NoUsableColumns,
    /// Transform input feature count differs from the fit.
    FeatureCountMismatch {
        /// Feature count at fit time.
        fitted: usize,
        /// Feature count of the transform input.
        found: usize,
    },
    /// An input value is infinite (missing values must be `NaN`; infinities
    /// are data corruption, not missingness).
    InfiniteValue {
        /// Row of the offending value.
        row: usize,
        /// Column of the offending value.
        column: usize,
    },
}

impl fmt::Display for MissingPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidPolicy {
                maximum_missing_fraction,
            } => write!(
                formatter,
                "maximum missing fraction {maximum_missing_fraction} is not in [0, 1)"
            ),
            Self::EmptyTrain => formatter.write_str("training split is empty"),
            Self::NoUsableColumns => formatter.write_str("the policy dropped every feature column"),
            Self::FeatureCountMismatch { fitted, found } => write!(
                formatter,
                "imputer fitted on {fitted} features, input has {found}"
            ),
            Self::InfiniteValue { row, column } => write!(
                formatter,
                "value at ({row}, {column}) is infinite — corruption, not missingness"
            ),
        }
    }
}

impl std::error::Error for MissingPolicyError {}

fn check_no_infinities(features: &[Vec<f64>]) -> Result<(), MissingPolicyError> {
    for (row, values) in features.iter().enumerate()
    {
        for (column, value) in values.iter().enumerate()
        {
            if value.is_infinite()
            {
                return Err(MissingPolicyError::InfiniteValue { row, column });
            }
        }
    }

    Ok(())
}

/// Median of the observed (non-NaN) values via the program's canonical
/// sorted-midpoint convention.
fn observed_median(values: &[f64]) -> Option<f64> {
    let mut observed: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();

    if observed.is_empty()
    {
        return None;
    }

    observed.sort_by(f64::total_cmp);

    let n = observed.len();

    Some(
        if n % 2 == 1
        {
            observed[n / 2]
        }
        else
        {
            (observed[n / 2 - 1] + observed[n / 2]) / 2.0
        },
    )
}

impl FittedImputer {
    /// Fits the policy on the training features (rows × columns; missing =
    /// `NaN`).
    pub fn fit(
        train_features: &[Vec<f64>],
        policy: MissingValuePolicy,
    ) -> Result<Self, MissingPolicyError> {
        if !policy.maximum_missing_fraction.is_finite()
            || !(0.0..1.0).contains(&policy.maximum_missing_fraction)
        {
            return Err(MissingPolicyError::InvalidPolicy {
                maximum_missing_fraction: policy.maximum_missing_fraction,
            });
        }

        if train_features.is_empty()
        {
            return Err(MissingPolicyError::EmptyTrain);
        }

        check_no_infinities(train_features)?;

        let rows = train_features.len();
        let columns = train_features[0].len();

        let mut kept_columns = Vec::new();
        let mut training_medians = Vec::new();
        let mut dropped_columns = Vec::new();

        for column in 0..columns
        {
            let values: Vec<f64> = train_features.iter().map(|row| row[column]).collect();

            let missing = values.iter().filter(|v| v.is_nan()).count();

            if missing == rows
            {
                dropped_columns.push((column, DropReason::AllMissing));
                continue;
            }

            if missing as f64 / rows as f64 > policy.maximum_missing_fraction
            {
                dropped_columns.push((column, DropReason::TooManyMissing));
                continue;
            }

            let observed: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();

            let first = observed[0];

            if observed.iter().all(|&v| v == first)
            {
                dropped_columns.push((column, DropReason::ConstantColumn));
                continue;
            }

            let median = observed_median(&values).expect("column has observed values");

            kept_columns.push(column);
            training_medians.push(median);
        }

        if kept_columns.is_empty()
        {
            return Err(MissingPolicyError::NoUsableColumns);
        }

        Ok(Self {
            input_feature_count: columns,
            kept_columns,
            training_medians,
            dropped_columns,
        })
    }

    /// Applies the fitted policy: keeps the fitted columns, imputes missing
    /// values with the training medians. Targets, groups and time index
    /// pass through unchanged.
    pub fn transform(
        &self,
        dataset: &TabularDataset,
    ) -> Result<TabularDataset, MissingPolicyError> {
        if dataset.feature_count() != self.input_feature_count
        {
            return Err(MissingPolicyError::FeatureCountMismatch {
                fitted: self.input_feature_count,
                found: dataset.feature_count(),
            });
        }

        check_no_infinities(&dataset.features)?;

        let features: Vec<Vec<f64>> = dataset
            .features
            .iter()
            .map(|row| {
                self.kept_columns
                    .iter()
                    .zip(&self.training_medians)
                    .map(|(&column, &median)| {
                        let value = row[column];

                        if value.is_nan() { median } else { value }
                    })
                    .collect()
            })
            .collect();

        Ok(TabularDataset {
            features,
            targets: dataset.targets.clone(),
            groups: dataset.groups.clone(),
            time_index: dataset.time_index.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nan() -> f64 {
        f64::NAN
    }

    #[test]
    fn fit_drops_and_imputes_per_the_policy() {
        // Column 0: complete, varying — kept, median 2.0.
        // Column 1: 50% missing — dropped (policy max 0.4).
        // Column 2: constant where observed — dropped.
        // Column 3: all missing — dropped.
        // Column 4: one missing — kept, median of {1,3,5} = 3.
        let train = vec![
            vec![1.0, nan(), 7.0, nan(), 1.0],
            vec![2.0, 5.0, 7.0, nan(), 3.0],
            vec![3.0, nan(), 7.0, nan(), 5.0],
            vec![2.0, 6.0, 7.0, nan(), nan()],
        ];

        let imputer = FittedImputer::fit(
            &train,
            MissingValuePolicy {
                maximum_missing_fraction: 0.4,
            },
        )
        .unwrap();

        assert_eq!(imputer.kept_columns, vec![0, 4]);
        assert_eq!(imputer.training_medians, vec![2.0, 3.0]);
        assert_eq!(
            imputer.dropped_columns,
            vec![
                (1, DropReason::TooManyMissing),
                (2, DropReason::ConstantColumn),
                (3, DropReason::AllMissing),
            ],
        );

        let evaluation = TabularDataset {
            features: vec![vec![9.0, 1.0, 2.0, 3.0, nan()]],
            targets: vec![1.0],
            groups: None,
            time_index: None,
        };

        let transformed = imputer.transform(&evaluation).unwrap();

        assert_eq!(transformed.features, vec![vec![9.0, 3.0]]);
        assert_eq!(transformed.targets, vec![1.0]);
        assert_eq!(transformed.validate(), Ok(()));
    }

    #[test]
    fn imputation_uses_training_medians_not_evaluation_values() {
        let train = vec![vec![1.0], vec![3.0]];

        let imputer = FittedImputer::fit(
            &train,
            MissingValuePolicy {
                maximum_missing_fraction: 0.0,
            },
        )
        .unwrap();

        // Evaluation values are far away; the imputed value must still be
        // the TRAINING median 2.0.
        let evaluation = TabularDataset {
            features: vec![vec![100.0], vec![nan()]],
            targets: vec![0.0, 0.0],
            groups: None,
            time_index: None,
        };

        let transformed = imputer.transform(&evaluation).unwrap();

        assert_eq!(transformed.features[1][0], 2.0);
    }

    #[test]
    fn typed_errors_cover_policy_shape_and_corruption() {
        assert!(matches!(
            FittedImputer::fit(
                &[vec![1.0]],
                MissingValuePolicy {
                    maximum_missing_fraction: 1.0,
                },
            ),
            Err(MissingPolicyError::InvalidPolicy { .. })
        ));

        assert_eq!(
            FittedImputer::fit(
                &[],
                MissingValuePolicy {
                    maximum_missing_fraction: 0.5,
                },
            ),
            Err(MissingPolicyError::EmptyTrain),
        );

        assert_eq!(
            FittedImputer::fit(
                &[vec![nan()], vec![nan()]],
                MissingValuePolicy {
                    maximum_missing_fraction: 0.5,
                },
            ),
            Err(MissingPolicyError::NoUsableColumns),
        );

        assert_eq!(
            FittedImputer::fit(
                &[vec![f64::INFINITY]],
                MissingValuePolicy {
                    maximum_missing_fraction: 0.5,
                },
            ),
            Err(MissingPolicyError::InfiniteValue { row: 0, column: 0 }),
        );

        let imputer = FittedImputer::fit(
            &[vec![1.0, 2.0], vec![2.0, 4.0]],
            MissingValuePolicy {
                maximum_missing_fraction: 0.5,
            },
        )
        .unwrap();

        let narrow = TabularDataset {
            features: vec![vec![1.0]],
            targets: vec![0.0],
            groups: None,
            time_index: None,
        };

        assert_eq!(
            imputer.transform(&narrow),
            Err(MissingPolicyError::FeatureCountMismatch {
                fitted: 2,
                found: 1
            }),
        );
    }

    #[test]
    fn fit_and_transform_are_deterministic() {
        let train = vec![
            vec![1.0, nan(), 5.0],
            vec![2.0, 7.0, nan()],
            vec![3.0, 8.0, 6.0],
        ];

        let policy = MissingValuePolicy {
            maximum_missing_fraction: 0.5,
        };

        let first = FittedImputer::fit(&train, policy).unwrap();
        let second = FittedImputer::fit(&train, policy).unwrap();

        assert_eq!(first, second);
    }
}
