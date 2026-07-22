//! Deterministic, manifest-recorded contamination generators.
//!
//! Every contamination is a pure function of `(dataset, kind, fraction,
//! seed)` and returns both the contaminated dataset and a
//! [`ContaminationManifest`] recording **exactly** what was done: the kind
//! with its parameters, the seed, the affected row indices, how many rows
//! were appended (duplication only), and the content checksums before and
//! after. A benchmark row without its contamination manifest is not
//! reproducible; the manifest is the record.
//!
//! Selection semantics per kind, stated precisely:
//!
//! - scattered kinds ([`AdditiveNoise`], [`CoordinateScaleShift`],
//!   [`TargetFlip`], [`SourceDuplication`], [`CoherentAlternativeCluster`],
//!   [`SensorBias`], [`SensorDropout`]): the affected rows are the first
//!   `⌊fraction · rows⌋` of a seeded Fisher–Yates shuffle of all rows;
//! - [`BurstAttack`]: requires `time_index`; the affected rows are a
//!   **contiguous window** of `⌊fraction · rows⌋` rows in time order (ties by
//!   row index), starting at a seeded offset — a temporally coherent attack;
//! - [`ViewConcentratedAttack`]: the affected rows are the first
//!   `⌊fraction · group_size⌋` of a seeded shuffle **within the named
//!   group** — corruption concentrated in one unit, invisible from others.
//!
//! `fraction = 0` is a legitimate no-op (benchmark sweeps include the clean
//! level); an empty affected set is recorded, not an error.
//!
//! [`AdditiveNoise`]: ContaminationKind::AdditiveNoise
//! [`CoordinateScaleShift`]: ContaminationKind::CoordinateScaleShift
//! [`TargetFlip`]: ContaminationKind::TargetFlip
//! [`SourceDuplication`]: ContaminationKind::SourceDuplication
//! [`CoherentAlternativeCluster`]: ContaminationKind::CoherentAlternativeCluster
//! [`SensorBias`]: ContaminationKind::SensorBias
//! [`SensorDropout`]: ContaminationKind::SensorDropout
//! [`BurstAttack`]: ContaminationKind::BurstAttack
//! [`ViewConcentratedAttack`]: ContaminationKind::ViewConcentratedAttack
//! [`LeveragePoint`]: ContaminationKind::LeveragePoint

use core::fmt;

use scirust_stats::{Distribution, Normal, SplitMix64};
use serde::{Deserialize, Serialize};

use crate::dataset::{DatasetError, TabularDataset};

/// What corruption to apply. Every parameter is explicit; nothing is
/// implied by defaults.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub enum ContaminationKind {
    /// Seeded Gaussian noise of the given standard deviation added to every
    /// feature of the affected rows.
    AdditiveNoise {
        /// Noise standard deviation (finite, non-negative).
        standard_deviation: f64,
    },
    /// One feature column multiplied by `factor` on the affected rows — the
    /// unit-change / miscalibration scenario.
    CoordinateScaleShift {
        /// The column to rescale.
        column: usize,
        /// The multiplicative factor (finite, nonzero).
        factor: f64,
    },
    /// Target corruption `y ↦ 1 − y` on the affected rows: an exact,
    /// involutive label flip for `{0, 1}` anomaly labels. Applied verbatim
    /// to continuous targets too, where it is deterministic but **not**
    /// exactly involutive (`1 − (1 − y)` can differ from `y` by rounding).
    TargetFlip,
    /// Each affected row appended again `copies` times (groups and time
    /// index duplicated alongside) — repeated-state support inflation.
    SourceDuplication {
        /// How many extra copies of each affected row (at least 1).
        copies: usize,
    },
    /// A **coherent** fake structure: the affected rows all receive the same
    /// constant feature offset on every column and the same target offset —
    /// the adversarial alternative-cluster scenario, maximally consistent
    /// with itself.
    CoherentAlternativeCluster {
        /// Constant added to every feature of the affected rows.
        feature_offset: f64,
        /// Constant added to the target of the affected rows.
        target_offset: f64,
    },
    /// Constant additive bias on one column of the affected rows — the
    /// drifting / miscalibrated sensor.
    SensorBias {
        /// The column to bias.
        column: usize,
        /// The additive bias (finite).
        bias: f64,
    },
    /// One column of the affected rows replaced by a constant fill value —
    /// the dead / saturated sensor.
    SensorDropout {
        /// The column to overwrite.
        column: usize,
        /// The fill value (finite).
        fill_value: f64,
    },
    /// Target shift on a temporally **contiguous** window of rows — the
    /// burst attack of the trust-model scenarios. Requires `time_index`.
    BurstAttack {
        /// Constant added to the targets inside the burst window (finite).
        target_shift: f64,
    },
    /// Target shift concentrated **within one group** — the
    /// view-concentrated attack of the trust-model scenarios.
    ViewConcentratedAttack {
        /// The group whose rows are attacked.
        group: u64,
        /// Constant added to the affected targets (finite).
        target_shift: f64,
    },
    /// A **high-leverage bad point**: every feature of an affected row is
    /// pushed to `feature_shift_mads` train-MADs beyond its column median (a
    /// fixed positive direction), and its target is overwritten with
    /// `corrupt_target`. Unlike [`CoherentAlternativeCluster`], which the
    /// phase-728 diagnostic showed barely moved the least-squares fit (a
    /// low-leverage block), a point far out in feature space with a wrong
    /// target exerts large influence on ordinary least squares — the classic
    /// bad-leverage case robust regression is meant to reject. The per-column
    /// median and MAD are computed over the input dataset's rows; a
    /// near-constant column (MAD 0) receives no shift (no leverage there),
    /// which is stated rather than hidden.
    ///
    /// [`CoherentAlternativeCluster`]: ContaminationKind::CoherentAlternativeCluster
    LeveragePoint {
        /// How many train-MADs beyond each column's median to push the
        /// affected rows' features (finite, positive).
        feature_shift_mads: f64,
        /// The corrupt target value written to affected rows (finite).
        corrupt_target: f64,
    },
}

/// A contamination request: kind, share of rows, seed.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContaminationConfig {
    /// The corruption to apply.
    pub kind: ContaminationKind,
    /// Share of eligible rows to affect, in `[0, 1]`.
    pub fraction: f64,
    /// Seed for every random choice this contamination makes.
    pub seed: u64,
}

/// The exact record of an applied contamination.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContaminationManifest {
    /// The kind, with its parameters, verbatim.
    pub kind: ContaminationKind,
    /// The seed consumed.
    pub seed: u64,
    /// The requested fraction.
    pub requested_fraction: f64,
    /// The affected row indices (in the input dataset), ascending.
    pub affected_rows: Vec<usize>,
    /// Rows appended by duplication (0 for every other kind).
    pub appended_rows: usize,
    /// Content checksum of the input dataset.
    pub input_sha256: String,
    /// Content checksum of the contaminated dataset.
    pub output_sha256: String,
}

/// Typed contamination errors.
#[derive(Clone, Debug, PartialEq)]
pub enum ContaminationError {
    /// `fraction` is not a finite value in `[0, 1]`.
    InvalidFraction {
        /// The offending fraction.
        fraction: f64,
    },
    /// A kind parameter is non-finite or out of its documented domain.
    InvalidParameter {
        /// Which parameter.
        parameter: &'static str,
    },
    /// The kind addresses a feature column the dataset does not have.
    ColumnOutOfRange {
        /// Requested column.
        column: usize,
        /// Available columns.
        feature_count: usize,
    },
    /// `BurstAttack` requires `time_index`.
    MissingTimeIndex,
    /// `ViewConcentratedAttack` requires `groups`.
    MissingGroups,
    /// The named group does not occur in `groups`.
    UnknownGroup {
        /// The requested group key.
        group: u64,
    },
    /// The input dataset failed validation.
    InvalidInput(DatasetError),
    /// The contamination produced a non-finite value (e.g. overflow from an
    /// extreme factor) — reported, never silently emitted.
    NonFiniteResult(DatasetError),
}

impl fmt::Display for ContaminationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidFraction { fraction } =>
            {
                write!(
                    formatter,
                    "fraction {fraction} is not a finite value in [0, 1]"
                )
            },
            Self::InvalidParameter { parameter } =>
            {
                write!(formatter, "invalid contamination parameter: {parameter}")
            },
            Self::ColumnOutOfRange {
                column,
                feature_count,
            } => write!(
                formatter,
                "column {column} out of range for {feature_count} features"
            ),
            Self::MissingTimeIndex =>
            {
                formatter.write_str("burst attack requires per-row time_index")
            },
            Self::MissingGroups =>
            {
                formatter.write_str("view-concentrated attack requires per-row groups")
            },
            Self::UnknownGroup { group } =>
            {
                write!(formatter, "group {group} does not occur in groups")
            },
            Self::InvalidInput(error) => write!(formatter, "invalid input dataset: {error}"),
            Self::NonFiniteResult(error) =>
            {
                write!(
                    formatter,
                    "contamination produced a non-finite value: {error}"
                )
            },
        }
    }
}

impl std::error::Error for ContaminationError {}

fn validate_kind(
    kind: &ContaminationKind,
    dataset: &TabularDataset,
) -> Result<(), ContaminationError> {
    let feature_count = dataset.feature_count();

    let check_column = |column: usize| -> Result<(), ContaminationError> {
        if column >= feature_count
        {
            return Err(ContaminationError::ColumnOutOfRange {
                column,
                feature_count,
            });
        }

        Ok(())
    };

    match kind
    {
        ContaminationKind::AdditiveNoise { standard_deviation } =>
        {
            if !standard_deviation.is_finite() || *standard_deviation < 0.0
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "standard_deviation must be finite and non-negative",
                });
            }
        },
        ContaminationKind::CoordinateScaleShift { column, factor } =>
        {
            check_column(*column)?;

            if !factor.is_finite() || *factor == 0.0
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "factor must be finite and nonzero",
                });
            }
        },
        ContaminationKind::TargetFlip =>
        {},
        ContaminationKind::SourceDuplication { copies } =>
        {
            if *copies == 0
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "copies must be at least 1",
                });
            }
        },
        ContaminationKind::CoherentAlternativeCluster {
            feature_offset,
            target_offset,
        } =>
        {
            if !feature_offset.is_finite() || !target_offset.is_finite()
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "cluster offsets must be finite",
                });
            }
        },
        ContaminationKind::SensorBias { column, bias } =>
        {
            check_column(*column)?;

            if !bias.is_finite()
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "bias must be finite",
                });
            }
        },
        ContaminationKind::SensorDropout { column, fill_value } =>
        {
            check_column(*column)?;

            if !fill_value.is_finite()
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "fill_value must be finite",
                });
            }
        },
        ContaminationKind::BurstAttack { target_shift } =>
        {
            if dataset.time_index.is_none()
            {
                return Err(ContaminationError::MissingTimeIndex);
            }

            if !target_shift.is_finite()
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "target_shift must be finite",
                });
            }
        },
        ContaminationKind::ViewConcentratedAttack {
            group,
            target_shift,
        } =>
        {
            let groups = dataset
                .groups
                .as_ref()
                .ok_or(ContaminationError::MissingGroups)?;

            if !groups.contains(group)
            {
                return Err(ContaminationError::UnknownGroup { group: *group });
            }

            if !target_shift.is_finite()
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "target_shift must be finite",
                });
            }
        },
        ContaminationKind::LeveragePoint {
            feature_shift_mads,
            corrupt_target,
        } =>
        {
            if !feature_shift_mads.is_finite() || *feature_shift_mads <= 0.0
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "feature_shift_mads must be finite and positive",
                });
            }

            if !corrupt_target.is_finite()
            {
                return Err(ContaminationError::InvalidParameter {
                    parameter: "corrupt_target must be finite",
                });
            }
        },
    }

    Ok(())
}

/// Per-column (median, normal-consistency MAD) over a set of rows, matching
/// the program's sorted-midpoint convention.
fn column_median_mads(features: &[Vec<f64>]) -> Vec<(f64, f64)> {
    let columns = features.first().map_or(0, Vec::len);

    let midpoint_median = |sorted: &[f64]| -> f64 {
        let n = sorted.len();

        if n % 2 == 1
        {
            sorted[n / 2]
        }
        else
        {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    };

    (0..columns)
        .map(|column| {
            let mut values: Vec<f64> = features.iter().map(|row| row[column]).collect();
            values.sort_by(f64::total_cmp);
            let median = midpoint_median(&values);

            let mut deviations: Vec<f64> =
                values.iter().map(|value| (value - median).abs()).collect();
            deviations.sort_by(f64::total_cmp);
            let mad = midpoint_median(&deviations) * 1.482_602_218_505_602;

            (median, mad)
        })
        .collect()
}

/// Seeded Fisher–Yates over the given candidate rows; returns the first
/// `count` as the affected set, sorted ascending.
fn scattered_selection(candidates: &[usize], count: usize, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    let mut pool = candidates.to_vec();

    for position in (1..pool.len()).rev()
    {
        let draw = (rng.next_f64() * (position + 1) as f64) as usize;
        pool.swap(position, draw.min(position));
    }

    let mut affected: Vec<usize> = pool.into_iter().take(count).collect();
    affected.sort_unstable();
    affected
}

/// Applies a contamination and records exactly what happened.
pub fn apply_contamination(
    dataset: &TabularDataset,
    config: &ContaminationConfig,
) -> Result<(TabularDataset, ContaminationManifest), ContaminationError> {
    dataset
        .validate()
        .map_err(ContaminationError::InvalidInput)?;

    if !config.fraction.is_finite() || !(0.0..=1.0).contains(&config.fraction)
    {
        return Err(ContaminationError::InvalidFraction {
            fraction: config.fraction,
        });
    }

    validate_kind(&config.kind, dataset)?;

    let rows = dataset.sample_count();
    let input_sha256 = dataset.content_sha256();

    // Affected-row selection, per the module-documented semantics.
    let affected: Vec<usize> = match &config.kind
    {
        ContaminationKind::BurstAttack { .. } =>
        {
            let time_index = dataset
                .time_index
                .as_ref()
                .expect("validated: burst requires time_index");

            let mut order: Vec<usize> = (0..rows).collect();
            order.sort_by_key(|&row| (time_index[row], row));

            let window = ((rows as f64) * config.fraction).floor() as usize;

            if window == 0
            {
                Vec::new()
            }
            else
            {
                let mut rng = SplitMix64::new(config.seed);
                let latest_start = rows - window;
                let start = (rng.next_f64() * (latest_start + 1) as f64) as usize;
                let start = start.min(latest_start);

                let mut selected: Vec<usize> = order[start..start + window].to_vec();
                selected.sort_unstable();
                selected
            }
        },
        ContaminationKind::ViewConcentratedAttack { group, .. } =>
        {
            let groups = dataset
                .groups
                .as_ref()
                .expect("validated: view attack requires groups");

            let members: Vec<usize> = (0..rows).filter(|&row| groups[row] == *group).collect();
            let count = ((members.len() as f64) * config.fraction).floor() as usize;

            scattered_selection(&members, count, config.seed)
        },
        _ =>
        {
            let candidates: Vec<usize> = (0..rows).collect();
            let count = ((rows as f64) * config.fraction).floor() as usize;

            scattered_selection(&candidates, count, config.seed)
        },
    };

    let mut output = dataset.clone();
    let mut appended_rows = 0usize;

    match &config.kind
    {
        ContaminationKind::AdditiveNoise { standard_deviation } =>
        {
            let standard = Normal::standard();
            // A dedicated stream, decoupled from the selection stream, so the
            // noise at a given row depends only on (seed, affected set).
            let mut rng = SplitMix64::new(config.seed ^ 0x00A0_0A5E);

            for &row in &affected
            {
                for value in &mut output.features[row]
                {
                    let uniform = 1.0e-6 + rng.next_f64() * (1.0 - 2.0e-6);
                    *value += standard_deviation * standard.quantile(uniform);
                }
            }
        },
        ContaminationKind::CoordinateScaleShift { column, factor } =>
        {
            for &row in &affected
            {
                output.features[row][*column] *= factor;
            }
        },
        ContaminationKind::TargetFlip =>
        {
            for &row in &affected
            {
                output.targets[row] = 1.0 - output.targets[row];
            }
        },
        ContaminationKind::SourceDuplication { copies } =>
        {
            for &row in &affected
            {
                for _ in 0..*copies
                {
                    output.features.push(dataset.features[row].clone());
                    output.targets.push(dataset.targets[row]);

                    if let Some(groups) = &mut output.groups
                    {
                        let group = groups[row];
                        groups.push(group);
                    }

                    if let Some(time_index) = &mut output.time_index
                    {
                        let time = time_index[row];
                        time_index.push(time);
                    }

                    appended_rows += 1;
                }
            }
        },
        ContaminationKind::CoherentAlternativeCluster {
            feature_offset,
            target_offset,
        } =>
        {
            for &row in &affected
            {
                for value in &mut output.features[row]
                {
                    *value += feature_offset;
                }

                output.targets[row] += target_offset;
            }
        },
        ContaminationKind::SensorBias { column, bias } =>
        {
            for &row in &affected
            {
                output.features[row][*column] += bias;
            }
        },
        ContaminationKind::SensorDropout { column, fill_value } =>
        {
            for &row in &affected
            {
                output.features[row][*column] = *fill_value;
            }
        },
        ContaminationKind::BurstAttack { target_shift }
        | ContaminationKind::ViewConcentratedAttack { target_shift, .. } =>
        {
            for &row in &affected
            {
                output.targets[row] += target_shift;
            }
        },
        ContaminationKind::LeveragePoint {
            feature_shift_mads,
            corrupt_target,
        } =>
        {
            // Reference spread from the original (unmutated) rows.
            let spreads = column_median_mads(&dataset.features);

            for &row in &affected
            {
                for (column, (median, mad)) in spreads.iter().enumerate()
                {
                    output.features[row][column] = median + feature_shift_mads * mad;
                }

                output.targets[row] = *corrupt_target;
            }
        },
    }

    output
        .validate()
        .map_err(ContaminationError::NonFiniteResult)?;

    let manifest = ContaminationManifest {
        kind: config.kind.clone(),
        seed: config.seed,
        requested_fraction: config.fraction,
        affected_rows: affected,
        appended_rows,
        input_sha256,
        output_sha256: output.content_sha256(),
    };

    Ok((output, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> TabularDataset {
        TabularDataset {
            features: (0..10).map(|row| vec![row as f64, 100.0]).collect(),
            targets: (0..10).map(|row| row as f64 * 0.1).collect(),
            groups: Some(vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            time_index: Some((0..10).collect()),
        }
    }

    fn config(kind: ContaminationKind, fraction: f64) -> ContaminationConfig {
        ContaminationConfig {
            kind,
            fraction,
            seed: 0x00C0_FFEE,
        }
    }

    #[test]
    fn contamination_is_deterministic_and_manifested() {
        let data = dataset();

        let request = config(
            ContaminationKind::AdditiveNoise {
                standard_deviation: 1.0,
            },
            0.3,
        );

        let (first, manifest_a) = apply_contamination(&data, &request).unwrap();
        let (second, manifest_b) = apply_contamination(&data, &request).unwrap();

        assert_eq!(first, second);
        assert_eq!(manifest_a, manifest_b);
        assert_eq!(manifest_a.affected_rows.len(), 3);
        assert_eq!(manifest_a.input_sha256, data.content_sha256());
        assert_eq!(manifest_a.output_sha256, first.content_sha256());
        assert_ne!(manifest_a.input_sha256, manifest_a.output_sha256);

        // Unaffected rows are untouched.
        for row in 0..10
        {
            if !manifest_a.affected_rows.contains(&row)
            {
                assert_eq!(first.features[row], data.features[row]);
            }
        }
    }

    #[test]
    fn zero_fraction_is_a_recorded_no_op() {
        let data = dataset();

        let (output, manifest) =
            apply_contamination(&data, &config(ContaminationKind::TargetFlip, 0.0)).unwrap();

        assert_eq!(output, data);
        assert!(manifest.affected_rows.is_empty());
        assert_eq!(manifest.input_sha256, manifest.output_sha256);
    }

    #[test]
    fn target_flip_is_exactly_involutive_on_binary_labels() {
        // The involution claim holds exactly for {0, 1} labels only;
        // continuous targets round (1 − (1 − y) ≠ y in general), which the
        // kind's documentation states rather than hides.
        let mut data = dataset();
        data.targets = (0..10)
            .map(|row| f64::from(u8::from(row % 3 == 0)))
            .collect();

        let request = config(ContaminationKind::TargetFlip, 1.0);

        let (flipped, manifest) = apply_contamination(&data, &request).unwrap();
        let (back, _) = apply_contamination(&flipped, &request).unwrap();

        assert_eq!(manifest.affected_rows.len(), 10);
        assert_ne!(flipped.targets, data.targets);
        assert!(
            flipped
                .targets
                .iter()
                .all(|&label| label == 0.0 || label == 1.0)
        );
        assert_eq!(back.targets, data.targets);
    }

    #[test]
    fn duplication_appends_aligned_rows() {
        let data = dataset();

        let (output, manifest) = apply_contamination(
            &data,
            &config(ContaminationKind::SourceDuplication { copies: 2 }, 0.2),
        )
        .unwrap();

        assert_eq!(manifest.affected_rows.len(), 2);
        assert_eq!(manifest.appended_rows, 4);
        assert_eq!(output.sample_count(), 14);
        assert_eq!(output.validate(), Ok(()));

        // Every appended row is a bit-exact copy of an affected source row,
        // with aligned group and time values.
        let groups = output.groups.as_ref().unwrap();
        let time_index = output.time_index.as_ref().unwrap();
        let mut appended = 10usize;

        for &source in &manifest.affected_rows
        {
            for _ in 0..2
            {
                assert_eq!(output.features[appended], data.features[source]);
                assert_eq!(output.targets[appended], data.targets[source]);
                assert_eq!(groups[appended], data.groups.as_ref().unwrap()[source]);
                assert_eq!(
                    time_index[appended],
                    data.time_index.as_ref().unwrap()[source],
                );
                appended += 1;
            }
        }
    }

    #[test]
    fn burst_attack_is_temporally_contiguous() {
        // Scrambled time order: contiguity must hold in TIME, not row index.
        let mut data = dataset();
        data.time_index = Some(vec![9, 3, 7, 1, 5, 0, 8, 2, 6, 4]);

        let (output, manifest) = apply_contamination(
            &data,
            &config(ContaminationKind::BurstAttack { target_shift: 50.0 }, 0.4),
        )
        .unwrap();

        assert_eq!(manifest.affected_rows.len(), 4);

        let time = data.time_index.as_ref().unwrap();
        let mut affected_times: Vec<u64> = manifest
            .affected_rows
            .iter()
            .map(|&row| time[row])
            .collect();
        affected_times.sort_unstable();

        for pair in affected_times.windows(2)
        {
            assert_eq!(
                pair[1],
                pair[0] + 1,
                "burst window must be contiguous in time"
            );
        }

        for &row in &manifest.affected_rows
        {
            assert_eq!(output.targets[row], data.targets[row] + 50.0);
        }
    }

    #[test]
    fn view_concentrated_attack_stays_inside_the_group() {
        let data = dataset();

        let (output, manifest) = apply_contamination(
            &data,
            &config(
                ContaminationKind::ViewConcentratedAttack {
                    group: 2,
                    target_shift: 25.0,
                },
                1.0,
            ),
        )
        .unwrap();

        assert_eq!(manifest.affected_rows, vec![5, 6, 7, 8, 9]);

        for row in 0..5
        {
            assert_eq!(output.targets[row], data.targets[row]);
        }

        for row in 5..10
        {
            assert_eq!(output.targets[row], data.targets[row] + 25.0);
        }
    }

    #[test]
    fn leverage_point_pushes_features_out_and_corrupts_targets() {
        // Column 0 over rows {0..10}: values 0..9 → median 4.5, MAD ≈ 2.5·k.
        // Column 1 is constant (100) → MAD 0 → no shift there.
        let data = dataset();

        let (output, manifest) = apply_contamination(
            &data,
            &config(
                ContaminationKind::LeveragePoint {
                    feature_shift_mads: 10.0,
                    corrupt_target: -999.0,
                },
                0.2,
            ),
        )
        .unwrap();

        assert_eq!(manifest.affected_rows.len(), 2);

        let spreads = column_median_mads(&data.features);

        for &row in &manifest.affected_rows
        {
            // Column 0 pushed far out; column 1 (MAD 0) untouched in value.
            assert_eq!(output.features[row][0], spreads[0].0 + 10.0 * spreads[0].1,);
            assert!(output.features[row][0] > data.features[row][0] + 10.0);
            assert_eq!(output.features[row][1], spreads[1].0); // MAD 0 → median
            assert_eq!(output.targets[row], -999.0);
        }

        // Unaffected rows are untouched.
        for row in 0..10
        {
            if !manifest.affected_rows.contains(&row)
            {
                assert_eq!(output.features[row], data.features[row]);
                assert_eq!(output.targets[row], data.targets[row]);
            }
        }
    }

    #[test]
    fn leverage_point_rejects_bad_parameters() {
        let data = dataset();

        assert!(matches!(
            apply_contamination(
                &data,
                &config(
                    ContaminationKind::LeveragePoint {
                        feature_shift_mads: 0.0,
                        corrupt_target: 1.0,
                    },
                    0.5,
                ),
            ),
            Err(ContaminationError::InvalidParameter { .. })
        ));

        assert!(matches!(
            apply_contamination(
                &data,
                &config(
                    ContaminationKind::LeveragePoint {
                        feature_shift_mads: 5.0,
                        corrupt_target: f64::NAN,
                    },
                    0.5,
                ),
            ),
            Err(ContaminationError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn invalid_requests_are_typed_errors() {
        let data = dataset();

        assert!(matches!(
            apply_contamination(&data, &config(ContaminationKind::TargetFlip, 1.5)),
            Err(ContaminationError::InvalidFraction { .. })
        ));

        assert!(matches!(
            apply_contamination(
                &data,
                &config(
                    ContaminationKind::SensorBias {
                        column: 9,
                        bias: 1.0,
                    },
                    0.5,
                ),
            ),
            Err(ContaminationError::ColumnOutOfRange {
                column: 9,
                feature_count: 2,
            })
        ));

        assert!(matches!(
            apply_contamination(
                &data,
                &config(
                    ContaminationKind::CoordinateScaleShift {
                        column: 0,
                        factor: 0.0,
                    },
                    0.5,
                ),
            ),
            Err(ContaminationError::InvalidParameter { .. })
        ));

        assert!(matches!(
            apply_contamination(
                &data,
                &config(
                    ContaminationKind::ViewConcentratedAttack {
                        group: 99,
                        target_shift: 1.0,
                    },
                    0.5,
                ),
            ),
            Err(ContaminationError::UnknownGroup { group: 99 })
        ));

        let mut no_time = data.clone();
        no_time.time_index = None;

        assert_eq!(
            apply_contamination(
                &no_time,
                &config(ContaminationKind::BurstAttack { target_shift: 1.0 }, 0.5),
            ),
            Err(ContaminationError::MissingTimeIndex),
        );

        assert!(matches!(
            apply_contamination(
                &data,
                &config(ContaminationKind::SourceDuplication { copies: 0 }, 0.5),
            ),
            Err(ContaminationError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn overflow_to_non_finite_is_reported_not_emitted() {
        let mut data = dataset();
        data.features[0][0] = f64::MAX;

        let result = apply_contamination(
            &data,
            &config(
                ContaminationKind::CoordinateScaleShift {
                    column: 0,
                    factor: 2.0,
                },
                1.0,
            ),
        );

        assert!(matches!(
            result,
            Err(ContaminationError::NonFiniteResult(_))
        ));
    }
}
