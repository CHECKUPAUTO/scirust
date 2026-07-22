//! Deterministic, leakage-aware dataset splitting.
//!
//! Every split is a pure function of `(dataset, strategy, seed)` and carries
//! a [`SplitManifest`] recording exactly that. Leakage prevention is
//! **structural**, not advisory:
//!
//! - [`SplitStrategy::GroupedHoldout`] assigns whole *groups* (machines,
//!   runs) to one side — a group never straddles splits;
//! - [`SplitStrategy::Temporal`] orders rows by `time_index` (ties broken by
//!   row index, canonically) and cuts contiguous prefixes — training data
//!   never postdates evaluation data;
//! - [`SplitStrategy::LeaveOneGroupOut`] holds out one entire group.
//!
//! Randomized strategies use `scirust-stats`' `SplitMix64` with an explicit
//! seed and a Fisher–Yates shuffle in fixed index order; identical inputs
//! produce identical assignments on every platform.

use core::fmt;

use scirust_stats::SplitMix64;
use serde::{Deserialize, Serialize};

use crate::dataset::TabularDataset;

/// How to partition rows into train / validation / test.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum SplitStrategy {
    /// Row-level seeded shuffle, then fraction cuts. **No leakage protection
    /// across groups or time** — use only for i.i.d.-style tables.
    RandomHoldout {
        /// Fraction of rows assigned to training, in (0, 1).
        train_fraction: f64,
        /// Fraction of rows assigned to validation, in [0, 1).
        validation_fraction: f64,
    },
    /// Group-level seeded shuffle, then fraction cuts over *groups*; every
    /// row of a group lands on the same side. Requires `groups`.
    GroupedHoldout {
        /// Fraction of groups assigned to training, in (0, 1).
        train_fraction: f64,
        /// Fraction of groups assigned to validation, in [0, 1).
        validation_fraction: f64,
    },
    /// Contiguous cuts in `time_index` order (ties by row index): the
    /// earliest rows train, the latest test. Requires `time_index`. The seed
    /// is recorded but unused (there is no randomness to reproduce).
    Temporal {
        /// Fraction of rows assigned to training, in (0, 1).
        train_fraction: f64,
        /// Fraction of rows assigned to validation, in [0, 1).
        validation_fraction: f64,
    },
    /// Train on every other group, test on `held_out_group`; validation is
    /// empty by construction (hyperparameters must come from preregistration
    /// or nested selection inside the training groups). Requires `groups`.
    LeaveOneGroupOut {
        /// The group key to hold out entirely.
        held_out_group: u64,
    },
}

/// The reproducibility record every split carries.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SplitManifest {
    /// Seed consumed by randomized strategies (recorded verbatim for all).
    pub seed: u64,
    /// The exact strategy, with its parameters.
    pub strategy: SplitStrategy,
    /// Human name of the grouping key when one is in play (`"machine_id"`);
    /// `None` for row-level and temporal strategies.
    pub grouping_key: Option<String>,
    /// Canonical checksum of the dataset that was split.
    pub dataset_sha256: String,
}

/// Row indices per side, plus the manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitAssignment {
    /// Training row indices, ascending.
    pub train: Vec<usize>,
    /// Validation row indices, ascending (may be empty).
    pub validation: Vec<usize>,
    /// Test row indices, ascending.
    pub test: Vec<usize>,
    /// The reproducibility record.
    pub manifest: SplitManifest,
}

/// Typed split errors.
#[derive(Clone, Debug, PartialEq)]
pub enum SplitError {
    /// A fraction is non-finite, out of range, or the fractions leave no
    /// test rows.
    InvalidFractions {
        /// The offending train fraction.
        train_fraction: f64,
        /// The offending validation fraction.
        validation_fraction: f64,
    },
    /// The strategy needs `groups` but the dataset has none.
    MissingGroups,
    /// The strategy needs `time_index` but the dataset has none.
    MissingTimeIndex,
    /// `held_out_group` does not occur in `groups`.
    UnknownGroup {
        /// The requested group key.
        group: u64,
    },
    /// A side that must be non-empty came out empty (dataset too small for
    /// the requested fractions, or the held-out group covers everything).
    EmptySide {
        /// `"train"` or `"test"`.
        side: &'static str,
    },
}

impl fmt::Display for SplitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidFractions {
                train_fraction,
                validation_fraction,
            } => write!(
                formatter,
                "invalid split fractions: train {train_fraction}, validation \
{validation_fraction} (train in (0,1), validation in [0,1), sum < 1)"
            ),
            Self::MissingGroups =>
            {
                formatter.write_str("strategy requires per-row groups, dataset has none")
            },
            Self::MissingTimeIndex =>
            {
                formatter.write_str("strategy requires per-row time_index, dataset has none")
            },
            Self::UnknownGroup { group } =>
            {
                write!(formatter, "held-out group {group} does not occur in groups")
            },
            Self::EmptySide { side } =>
            {
                write!(formatter, "split leaves the {side} side empty")
            },
        }
    }
}

impl std::error::Error for SplitError {}

fn validate_fractions(train_fraction: f64, validation_fraction: f64) -> Result<(), SplitError> {
    let invalid = !train_fraction.is_finite()
        || !validation_fraction.is_finite()
        || train_fraction <= 0.0
        || train_fraction >= 1.0
        || validation_fraction < 0.0
        || validation_fraction >= 1.0
        || train_fraction + validation_fraction >= 1.0;

    if invalid
    {
        return Err(SplitError::InvalidFractions {
            train_fraction,
            validation_fraction,
        });
    }

    Ok(())
}

/// Seeded Fisher–Yates over `0..count` in fixed order.
fn shuffled_indices(count: usize, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    let mut indices: Vec<usize> = (0..count).collect();

    for position in (1..count).rev()
    {
        let draw = (rng.next_f64() * (position + 1) as f64) as usize;
        let swap = draw.min(position);
        indices.swap(position, swap);
    }

    indices
}

/// Cuts a shuffled unit list into (train, validation, test) unit sets by
/// fractions; counts are floors, with the guarantee that train gets at least
/// one unit and test gets the remainder.
fn fraction_cut(
    units: &[usize],
    train_fraction: f64,
    validation_fraction: f64,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let count = units.len();
    let train_count = ((count as f64) * train_fraction).floor().max(1.0) as usize;
    let validation_count = ((count as f64) * validation_fraction).floor() as usize;
    let train_end = train_count.min(count);
    let validation_end = (train_count + validation_count).min(count);

    (
        units[..train_end].to_vec(),
        units[train_end..validation_end].to_vec(),
        units[validation_end..].to_vec(),
    )
}

/// Distinct group keys in first-occurrence order (deterministic).
fn distinct_groups(groups: &[u64]) -> Vec<u64> {
    let mut distinct: Vec<u64> = Vec::new();

    for &group in groups
    {
        if !distinct.contains(&group)
        {
            distinct.push(group);
        }
    }

    distinct
}

/// Splits a validated dataset. The returned index vectors are sorted
/// ascending; their disjoint union is exactly `0..sample_count` for every
/// strategy except `LeaveOneGroupOut` (whose validation side is empty by
/// construction — the union still covers all rows).
pub fn split_dataset(
    dataset: &TabularDataset,
    strategy: &SplitStrategy,
    seed: u64,
    grouping_key: Option<&str>,
) -> Result<SplitAssignment, SplitError> {
    let rows = dataset.sample_count();

    let (mut train, mut validation, mut test) = match strategy
    {
        SplitStrategy::RandomHoldout {
            train_fraction,
            validation_fraction,
        } =>
        {
            validate_fractions(*train_fraction, *validation_fraction)?;

            let shuffled = shuffled_indices(rows, seed);

            fraction_cut(&shuffled, *train_fraction, *validation_fraction)
        },
        SplitStrategy::GroupedHoldout {
            train_fraction,
            validation_fraction,
        } =>
        {
            validate_fractions(*train_fraction, *validation_fraction)?;

            let groups = dataset.groups.as_ref().ok_or(SplitError::MissingGroups)?;
            let distinct = distinct_groups(groups);

            let group_order = shuffled_indices(distinct.len(), seed);
            let (train_units, validation_units, _test_units) =
                fraction_cut(&group_order, *train_fraction, *validation_fraction);

            let side_of_group = |group: u64| -> u8 {
                let position = distinct
                    .iter()
                    .position(|&candidate| candidate == group)
                    .expect("group came from this dataset");

                if train_units.contains(&position)
                {
                    0
                }
                else if validation_units.contains(&position)
                {
                    1
                }
                else
                {
                    2
                }
            };

            let mut train = Vec::new();
            let mut validation = Vec::new();
            let mut test = Vec::new();

            for (row, &group) in groups.iter().enumerate()
            {
                match side_of_group(group)
                {
                    0 => train.push(row),
                    1 => validation.push(row),
                    _ => test.push(row),
                }
            }

            (train, validation, test)
        },
        SplitStrategy::Temporal {
            train_fraction,
            validation_fraction,
        } =>
        {
            validate_fractions(*train_fraction, *validation_fraction)?;

            let time_index = dataset
                .time_index
                .as_ref()
                .ok_or(SplitError::MissingTimeIndex)?;

            let mut order: Vec<usize> = (0..rows).collect();
            order.sort_by_key(|&row| (time_index[row], row));

            fraction_cut(&order, *train_fraction, *validation_fraction)
        },
        SplitStrategy::LeaveOneGroupOut { held_out_group } =>
        {
            let groups = dataset.groups.as_ref().ok_or(SplitError::MissingGroups)?;

            if !groups.contains(held_out_group)
            {
                return Err(SplitError::UnknownGroup {
                    group: *held_out_group,
                });
            }

            let mut train = Vec::new();
            let mut test = Vec::new();

            for (row, &group) in groups.iter().enumerate()
            {
                if group == *held_out_group
                {
                    test.push(row);
                }
                else
                {
                    train.push(row);
                }
            }

            (train, Vec::new(), test)
        },
    };

    train.sort_unstable();
    validation.sort_unstable();
    test.sort_unstable();

    if train.is_empty()
    {
        return Err(SplitError::EmptySide { side: "train" });
    }

    if test.is_empty()
    {
        return Err(SplitError::EmptySide { side: "test" });
    }

    Ok(SplitAssignment {
        train,
        validation,
        test,
        manifest: SplitManifest {
            seed,
            strategy: strategy.clone(),
            grouping_key: grouping_key.map(str::to_owned),
            dataset_sha256: dataset.content_sha256(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(rows: usize, groups: Option<Vec<u64>>, time: Option<Vec<u64>>) -> TabularDataset {
        TabularDataset {
            features: (0..rows).map(|row| vec![row as f64, 1.0]).collect(),
            targets: (0..rows).map(|row| row as f64).collect(),
            groups,
            time_index: time,
        }
    }

    fn coverage(assignment: &SplitAssignment, rows: usize) {
        let mut all: Vec<usize> = assignment
            .train
            .iter()
            .chain(&assignment.validation)
            .chain(&assignment.test)
            .copied()
            .collect();

        all.sort_unstable();

        assert_eq!(all, (0..rows).collect::<Vec<_>>());
    }

    #[test]
    fn random_holdout_is_deterministic_and_covering() {
        let data = dataset(20, None, None);

        let strategy = SplitStrategy::RandomHoldout {
            train_fraction: 0.6,
            validation_fraction: 0.2,
        };

        let first = split_dataset(&data, &strategy, 0x5EED, None).unwrap();
        let second = split_dataset(&data, &strategy, 0x5EED, None).unwrap();
        let other_seed = split_dataset(&data, &strategy, 0x5EED + 1, None).unwrap();

        assert_eq!(first, second);
        assert_ne!(first.train, other_seed.train);
        coverage(&first, 20);
        assert_eq!(first.train.len(), 12);
        assert_eq!(first.validation.len(), 4);
        assert_eq!(first.test.len(), 4);
    }

    #[test]
    fn grouped_holdout_never_straddles_a_group() {
        let groups: Vec<u64> = (0..30).map(|row| (row % 6) as u64).collect();
        let data = dataset(30, Some(groups.clone()), None);

        let assignment = split_dataset(
            &data,
            &SplitStrategy::GroupedHoldout {
                train_fraction: 0.5,
                validation_fraction: 0.2,
            },
            7,
            Some("machine_id"),
        )
        .unwrap();

        coverage(&assignment, 30);

        let side_of_row = |row: usize| -> u8 {
            if assignment.train.contains(&row)
            {
                0
            }
            else if assignment.validation.contains(&row)
            {
                1
            }
            else
            {
                2
            }
        };

        for row in 0..30
        {
            for other in 0..30
            {
                if groups[row] == groups[other]
                {
                    assert_eq!(
                        side_of_row(row),
                        side_of_row(other),
                        "group {} straddles sides",
                        groups[row],
                    );
                }
            }
        }

        assert_eq!(
            assignment.manifest.grouping_key.as_deref(),
            Some("machine_id")
        );
    }

    #[test]
    fn temporal_split_never_trains_on_the_future() {
        // Deliberately scrambled time order across rows.
        let time: Vec<u64> = vec![9, 3, 7, 1, 5, 0, 8, 2, 6, 4];
        let data = dataset(10, None, Some(time.clone()));

        let assignment = split_dataset(
            &data,
            &SplitStrategy::Temporal {
                train_fraction: 0.5,
                validation_fraction: 0.2,
            },
            0,
            None,
        )
        .unwrap();

        coverage(&assignment, 10);

        let latest_train = assignment.train.iter().map(|&row| time[row]).max().unwrap();

        let earliest_validation = assignment
            .validation
            .iter()
            .map(|&row| time[row])
            .min()
            .unwrap_or(u64::MAX);

        let earliest_test = assignment.test.iter().map(|&row| time[row]).min().unwrap();

        assert!(latest_train < earliest_validation.min(earliest_test));
        assert!(
            assignment
                .validation
                .iter()
                .map(|&row| time[row])
                .max()
                .unwrap_or(0)
                < earliest_test
        );
    }

    #[test]
    fn temporal_ties_break_by_row_index_canonically() {
        let data = dataset(4, None, Some(vec![1, 0, 1, 0]));

        let assignment = split_dataset(
            &data,
            &SplitStrategy::Temporal {
                train_fraction: 0.5,
                validation_fraction: 0.0,
            },
            0,
            None,
        )
        .unwrap();

        // Sorted (time, row): (0,1), (0,3), (1,0), (1,2) — train = rows {1,3}.
        assert_eq!(assignment.train, vec![1, 3]);
        assert_eq!(assignment.test, vec![0, 2]);
    }

    #[test]
    fn leave_one_group_out_holds_out_exactly_that_group() {
        let groups: Vec<u64> = vec![1, 1, 2, 2, 3, 3];
        let data = dataset(6, Some(groups), None);

        let assignment = split_dataset(
            &data,
            &SplitStrategy::LeaveOneGroupOut { held_out_group: 2 },
            0,
            Some("run_id"),
        )
        .unwrap();

        assert_eq!(assignment.train, vec![0, 1, 4, 5]);
        assert!(assignment.validation.is_empty());
        assert_eq!(assignment.test, vec![2, 3]);
    }

    #[test]
    fn invalid_configurations_are_typed_errors() {
        let data = dataset(10, None, None);

        assert!(matches!(
            split_dataset(
                &data,
                &SplitStrategy::RandomHoldout {
                    train_fraction: 0.9,
                    validation_fraction: 0.2,
                },
                0,
                None,
            ),
            Err(SplitError::InvalidFractions { .. })
        ));

        assert_eq!(
            split_dataset(
                &data,
                &SplitStrategy::GroupedHoldout {
                    train_fraction: 0.5,
                    validation_fraction: 0.0,
                },
                0,
                None,
            ),
            Err(SplitError::MissingGroups),
        );

        assert_eq!(
            split_dataset(
                &data,
                &SplitStrategy::Temporal {
                    train_fraction: 0.5,
                    validation_fraction: 0.0,
                },
                0,
                None,
            ),
            Err(SplitError::MissingTimeIndex),
        );

        let grouped = dataset(4, Some(vec![1, 1, 1, 1]), None);

        assert_eq!(
            split_dataset(
                &grouped,
                &SplitStrategy::LeaveOneGroupOut { held_out_group: 9 },
                0,
                None,
            ),
            Err(SplitError::UnknownGroup { group: 9 }),
        );

        assert_eq!(
            split_dataset(
                &grouped,
                &SplitStrategy::LeaveOneGroupOut { held_out_group: 1 },
                0,
                None,
            ),
            Err(SplitError::EmptySide { side: "train" }),
        );
    }

    #[test]
    fn manifests_record_seed_strategy_and_checksum() {
        let data = dataset(10, None, None);

        let strategy = SplitStrategy::RandomHoldout {
            train_fraction: 0.6,
            validation_fraction: 0.0,
        };

        let assignment = split_dataset(&data, &strategy, 42, None).unwrap();

        assert_eq!(assignment.manifest.seed, 42);
        assert_eq!(assignment.manifest.strategy, strategy);
        assert_eq!(assignment.manifest.dataset_sha256, data.content_sha256());
    }
}
