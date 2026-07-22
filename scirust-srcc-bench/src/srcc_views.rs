//! Transport-view construction for SRCC evaluation (preregistered).
//!
//! SRCC consumes *transport views* — per-unit sequences of `(source,
//! target)` 16-dimensional pairs — not tabular rows. The preregistration
//! (`docs/research/SRCC_INDUSTRIAL_BENCHMARK_PREREGISTRATION.md`, §4) fixes
//! this construction to prevent post-hoc choice, and this module implements
//! exactly that:
//!
//! - each **group** (machine / run / trajectory) is one view, in ascending
//!   group-id order;
//! - within a view, rows are ordered by `(time_index, row)` — the canonical
//!   temporal order used everywhere in the harness;
//! - the **source** at position `i` is the preregistered channel columns
//!   (at most 16, zero-padded) of row `i`; the **target** is the same
//!   channels at position `i + horizon`;
//! - when `center_per_trajectory` is set, each channel is centered by its
//!   **own trajectory's median** (computed over that trajectory only — a
//!   per-unit normalization with no cross-unit leakage, and no use of any
//!   other split's data).
//!
//! Every failure mode is typed: missing groups or time index, more than 16
//! channels, out-of-range columns, or a trajectory too short for the
//! horizon. Nothing is silently dropped.

use core::fmt;

use scirust_srcc::{SRCC_DIMENSION, SrccTransportSample, Vector16};
use scirust_stats::describe::median;

use crate::dataset::TabularDataset;

/// The preregistered view-construction parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportViewSpec {
    /// Feature columns used as channels, in order (at most 16; fewer are
    /// zero-padded).
    pub channel_columns: Vec<usize>,
    /// Steps ahead (in canonical temporal order) of the target relative to
    /// the source. Must be at least 1.
    pub horizon: usize,
    /// Center each channel by its own trajectory's median.
    pub center_per_trajectory: bool,
}

/// One view per group, in ascending group order.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportViews {
    /// The views (aligned with `group_ids`).
    pub views: Vec<Vec<SrccTransportSample>>,
    /// Ascending group ids, one per view.
    pub group_ids: Vec<u64>,
}

/// Typed view-construction errors.
#[derive(Clone, Debug, PartialEq)]
pub enum ViewError {
    /// The dataset has no `groups`.
    MissingGroups,
    /// The dataset has no `time_index`.
    MissingTimeIndex,
    /// No channel columns were requested.
    NoChannels,
    /// More than 16 channels were requested.
    TooManyChannels {
        /// The requested count.
        requested: usize,
    },
    /// A channel column is out of range.
    ColumnOutOfRange {
        /// The offending column.
        column: usize,
        /// Available feature count.
        feature_count: usize,
    },
    /// The horizon is zero.
    ZeroHorizon,
    /// A trajectory has too few rows for the horizon (it would produce an
    /// empty view).
    TrajectoryTooShort {
        /// The group id.
        group: u64,
        /// Rows in that trajectory.
        rows: usize,
        /// The requested horizon.
        horizon: usize,
    },
}

impl fmt::Display for ViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::MissingGroups => formatter.write_str("view construction requires per-row groups"),
            Self::MissingTimeIndex =>
            {
                formatter.write_str("view construction requires per-row time_index")
            },
            Self::NoChannels => formatter.write_str("at least one channel column is required"),
            Self::TooManyChannels { requested } =>
            {
                write!(
                    formatter,
                    "at most {SRCC_DIMENSION} channels are supported, {requested} requested"
                )
            },
            Self::ColumnOutOfRange {
                column,
                feature_count,
            } => write!(
                formatter,
                "channel column {column} out of range for {feature_count} features"
            ),
            Self::ZeroHorizon => formatter.write_str("horizon must be at least 1"),
            Self::TrajectoryTooShort {
                group,
                rows,
                horizon,
            } => write!(
                formatter,
                "trajectory of group {group} has {rows} rows, too short for horizon {horizon}"
            ),
        }
    }
}

impl std::error::Error for ViewError {}

/// Builds one transport view per group, per the preregistered construction.
pub fn build_transport_views(
    dataset: &TabularDataset,
    spec: &TransportViewSpec,
) -> Result<TransportViews, ViewError> {
    let groups = dataset.groups.as_ref().ok_or(ViewError::MissingGroups)?;
    let time_index = dataset
        .time_index
        .as_ref()
        .ok_or(ViewError::MissingTimeIndex)?;

    if spec.channel_columns.is_empty()
    {
        return Err(ViewError::NoChannels);
    }

    if spec.channel_columns.len() > SRCC_DIMENSION
    {
        return Err(ViewError::TooManyChannels {
            requested: spec.channel_columns.len(),
        });
    }

    let feature_count = dataset.feature_count();

    for &column in &spec.channel_columns
    {
        if column >= feature_count
        {
            return Err(ViewError::ColumnOutOfRange {
                column,
                feature_count,
            });
        }
    }

    if spec.horizon == 0
    {
        return Err(ViewError::ZeroHorizon);
    }

    // Ascending distinct group ids.
    let mut group_ids: Vec<u64> = Vec::new();

    for &group in groups
    {
        if !group_ids.contains(&group)
        {
            group_ids.push(group);
        }
    }

    group_ids.sort_unstable();

    let mut views = Vec::with_capacity(group_ids.len());

    for &group in &group_ids
    {
        // Canonical temporal order within the trajectory.
        let mut rows: Vec<usize> = (0..dataset.sample_count())
            .filter(|&row| groups[row] == group)
            .collect();

        rows.sort_by_key(|&row| (time_index[row], row));

        if rows.len() <= spec.horizon
        {
            return Err(ViewError::TrajectoryTooShort {
                group,
                rows: rows.len(),
                horizon: spec.horizon,
            });
        }

        // Per-trajectory channel medians (this trajectory's rows only).
        let centers: Vec<f64> = if spec.center_per_trajectory
        {
            spec.channel_columns
                .iter()
                .map(|&column| {
                    let values: Vec<f64> = rows
                        .iter()
                        .map(|&row| dataset.features[row][column])
                        .collect();

                    median(&values)
                })
                .collect()
        }
        else
        {
            vec![0.0; spec.channel_columns.len()]
        };

        let embed = |row: usize| -> Vector16 {
            let mut vector = [0.0; SRCC_DIMENSION];

            for (slot, (&column, &center)) in spec.channel_columns.iter().zip(&centers).enumerate()
            {
                vector[slot] = dataset.features[row][column] - center;
            }

            vector
        };

        let view: Vec<SrccTransportSample> = (0..rows.len() - spec.horizon)
            .map(|position| {
                SrccTransportSample::new(
                    embed(rows[position]),
                    embed(rows[position + spec.horizon]),
                )
            })
            .collect();

        views.push(view);
    }

    Ok(TransportViews { views, group_ids })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> TabularDataset {
        // Two trajectories with deliberately scrambled time order.
        TabularDataset {
            features: vec![
                vec![10.0, 100.0], // g1 t2
                vec![8.0, 80.0],   // g1 t0
                vec![9.0, 90.0],   // g1 t1
                vec![21.0, 210.0], // g2 t1
                vec![20.0, 200.0], // g2 t0
                vec![22.0, 220.0], // g2 t2
            ],
            targets: vec![0.0; 6],
            groups: Some(vec![1, 1, 1, 2, 2, 2]),
            time_index: Some(vec![2, 0, 1, 1, 0, 2]),
        }
    }

    #[test]
    fn views_follow_canonical_time_order_and_group_order() {
        let views = build_transport_views(
            &dataset(),
            &TransportViewSpec {
                channel_columns: vec![0, 1],
                horizon: 1,
                center_per_trajectory: false,
            },
        )
        .unwrap();

        assert_eq!(views.group_ids, vec![1, 2]);
        assert_eq!(views.views[0].len(), 2);

        // Group 1 temporal order: rows 1 (t0) → 2 (t1) → 0 (t2).
        assert_eq!(views.views[0][0].source[0], 8.0);
        assert_eq!(views.views[0][0].target[0], 9.0);
        assert_eq!(views.views[0][1].source[0], 9.0);
        assert_eq!(views.views[0][1].target[0], 10.0);

        // Padding beyond the two channels is exactly zero.
        assert_eq!(views.views[0][0].source[2..], [0.0; 14]);
    }

    #[test]
    fn per_trajectory_median_centering_uses_only_that_trajectory() {
        let views = build_transport_views(
            &dataset(),
            &TransportViewSpec {
                channel_columns: vec![0],
                horizon: 1,
                center_per_trajectory: true,
            },
        )
        .unwrap();

        // Medians: group 1 → 9.0, group 2 → 21.0 (their own rows only).
        assert_eq!(views.views[0][0].source[0], 8.0 - 9.0);
        assert_eq!(views.views[1][0].source[0], 20.0 - 21.0);
    }

    #[test]
    fn horizon_two_pairs_across_two_steps() {
        let views = build_transport_views(
            &dataset(),
            &TransportViewSpec {
                channel_columns: vec![0],
                horizon: 2,
                center_per_trajectory: false,
            },
        )
        .unwrap();

        assert_eq!(views.views[0].len(), 1);
        assert_eq!(views.views[0][0].source[0], 8.0);
        assert_eq!(views.views[0][0].target[0], 10.0);
    }

    #[test]
    fn construction_errors_are_typed() {
        let data = dataset();

        let spec = |columns: Vec<usize>, horizon: usize| TransportViewSpec {
            channel_columns: columns,
            horizon,
            center_per_trajectory: false,
        };

        let mut no_groups = data.clone();
        no_groups.groups = None;
        assert_eq!(
            build_transport_views(&no_groups, &spec(vec![0], 1)),
            Err(ViewError::MissingGroups),
        );

        let mut no_time = data.clone();
        no_time.time_index = None;
        assert_eq!(
            build_transport_views(&no_time, &spec(vec![0], 1)),
            Err(ViewError::MissingTimeIndex),
        );

        assert_eq!(
            build_transport_views(&data, &spec(vec![], 1)),
            Err(ViewError::NoChannels),
        );

        assert_eq!(
            build_transport_views(&data, &spec((0..17).collect(), 1)),
            Err(ViewError::TooManyChannels { requested: 17 }),
        );

        assert_eq!(
            build_transport_views(&data, &spec(vec![5], 1)),
            Err(ViewError::ColumnOutOfRange {
                column: 5,
                feature_count: 2,
            }),
        );

        assert_eq!(
            build_transport_views(&data, &spec(vec![0], 0)),
            Err(ViewError::ZeroHorizon),
        );

        assert_eq!(
            build_transport_views(&data, &spec(vec![0], 3)),
            Err(ViewError::TrajectoryTooShort {
                group: 1,
                rows: 3,
                horizon: 3,
            }),
        );
    }

    #[test]
    fn construction_is_deterministic() {
        let spec = TransportViewSpec {
            channel_columns: vec![0, 1],
            horizon: 1,
            center_per_trajectory: true,
        };

        assert_eq!(
            build_transport_views(&dataset(), &spec).unwrap(),
            build_transport_views(&dataset(), &spec).unwrap(),
        );
    }
}
