//! The structured, versioned result model every [`crate::CapabilityAdapter`]
//! returns from `execute()`.
//!
//! Terminal formatting (colour, column widths, layout) lives outside this
//! module — in `scirust-cli`'s formatter — so the same [`RunResult`] can be
//! rendered as text today and fed to a chart or a desktop UI later without
//! re-running anything. JSON serialization lives here (via
//! [`RunResult::to_json_pretty`]), the same way
//! `scirust_studio_registry::CapabilityRegistry::to_json` keeps its
//! serialization in the crate that owns the type, so callers such as
//! `scirust-cli` do not need a direct `serde_json` dependency just to expose
//! `--format json`.
//!
//! ## Schema version 2: axes carry their coordinates
//!
//! Version 1 described axes but did not store their values, which silently
//! forced every consumer into the same wrong assumption — that samples are
//! uniformly spaced. That is false for any adaptive solver: Robertson's
//! Rosenbrock-W integrator chooses its own steps, spanning several decades of
//! time in a single run, and a chart drawn against a regenerated linear axis
//! would be a picture of a different experiment.
//!
//! Version 2 therefore stores the integrator's real coordinates in
//! [`Axis::values`], and every [`Series`] names the axis it belongs to
//! through [`Series::axis_id`]. Nothing infers spacing, and nothing
//! reconstructs a time vector. See
//! `docs/studio/adr/0006-result-axis-coordinates.md`.

use serde::{Deserialize, Serialize};

use scirust_studio_registry::DeterminismClass;

/// The schema version of [`RunResult`] itself, independent of the scenario
/// schema version (`scirust_studio_schema::CURRENT_SCHEMA_VERSION`) and of
/// any individual capability's own versioning.
pub const RESULT_SCHEMA_VERSION: u32 = 2;

/// What an axis promises about the ordering of its coordinates.
///
/// Declared by the capability that produced the axis — the adapter is the
/// capability's own code — and enforced by [`validate_result`]. A time axis
/// from a forward integration is strictly increasing; an axis that merely
/// enumerates samples promises nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisMonotonicity {
    /// Each coordinate is strictly greater than the one before it.
    StrictlyIncreasing,
    /// Each coordinate is greater than or equal to the one before it.
    NonDecreasing,
    /// No ordering is claimed, and none is checked.
    Unspecified,
}

/// One shared axis, *including its coordinates*.
///
/// `values` are the integrator's own coordinates, carried through
/// unmodified. They are never regenerated from a start, an end and a count —
/// doing so would be correct only for fixed-step solvers and quietly wrong
/// for every adaptive one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Axis {
    /// Stable id, e.g. `"t"`, referenced by [`Series::axis_id`].
    pub id: String,
    /// Human-facing label, e.g. `"time"`.
    pub display_name: String,
    /// Unit symbol, e.g. `"s"`.
    pub unit: String,
    /// What this axis promises about its ordering.
    pub monotonicity: AxisMonotonicity,
    /// The coordinates themselves, in order.
    pub values: Vec<f64>,
}

/// What a [`Series`] *is*, so a consumer does not have to infer it from the
/// id.
///
/// Before ensembles every series was one computed trajectory and this
/// question had one answer, so it did not need asking. An ensemble result
/// carries several kinds of curve at once — individual realisations, their
/// mean, the edges of a spread band — and a chart that drew all of them
/// identically would present a summary statistic and a single noisy sample as
/// the same kind of evidence. That is a misleading picture, not merely an
/// ugly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesRole {
    /// One computed trajectory. The default, and what every deterministic
    /// capability produces.
    #[default]
    Trajectory,
    /// A line drawn for comparison that was not computed by the solver —
    /// an analytic mean, an equilibrium level, a threshold.
    Reference,
    /// One realisation out of an ensemble, retained as an exemplar.
    EnsembleMember,
    /// The across-replicate mean at each axis coordinate.
    EnsembleMean,
    /// The lower edge of the ensemble's spread band.
    EnsembleBandLower,
    /// The upper edge of the ensemble's spread band.
    EnsembleBandUpper,
}

impl SeriesRole {
    /// Whether this role is part of an ensemble's presentation.
    pub fn is_ensemble(self) -> bool {
        matches!(
            self,
            SeriesRole::EnsembleMember
                | SeriesRole::EnsembleMean
                | SeriesRole::EnsembleBandLower
                | SeriesRole::EnsembleBandUpper
        )
    }
}

/// One named output time-course, bound to the axis it is plotted against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Series {
    /// Stable id, e.g. `"position"`.
    pub id: String,
    /// Human-facing label.
    pub display_name: String,
    /// Unit symbol of the values.
    pub unit: String,
    /// The [`Axis::id`] these values are plotted against. Required: a
    /// consumer must never have to guess, and a run with more than one axis
    /// would make guessing wrong.
    pub axis_id: String,
    /// What this curve represents.
    ///
    /// Defaulted, so every result written before this field existed reads
    /// back as [`SeriesRole::Trajectory`] — which is what those series were.
    #[serde(default)]
    pub role: SeriesRole,
    /// The values, aligned one-to-one with that axis's coordinates.
    pub values: Vec<f64>,
}

/// A quantity that varies over **two** axes at once — a field.
///
/// # Why a [`Series`] could not do this
///
/// `REPOSITORY_AUDIT.md` said for two phases that schema v2 could already
/// express a spatially discretised field, because `axes` is a vector and
/// every series names its axis. Reading it again while adapting one showed
/// that is not so. A `Series` is aligned *one-to-one* with a single axis, so
/// the most it can hold is a slice: one node's history, or one instant's
/// profile. Representing `u(x, t)` meant `n` series and losing the fact they
/// are one quantity, or `m` series and losing it the other way.
///
/// So the field is its own thing. Nothing else in the result model changes,
/// and a capability that has no field carries an empty vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    /// Stable id, e.g. `"temperature"`.
    pub id: String,
    /// Human-facing label.
    pub display_name: String,
    /// Unit symbol of the values.
    pub unit: String,
    /// The [`Axis::id`] the **rows** run along — time, for an evolving field.
    pub row_axis_id: String,
    /// The [`Axis::id`] the **columns** run along — position, for a rod.
    pub column_axis_id: String,
    /// How many columns each row has.
    ///
    /// Redundant with the column axis's length, and deliberately so:
    /// [`validate_result`] checks the two agree, and in exchange a consumer
    /// holding only the field — a downsampler, a renderer — knows its shape
    /// without being handed the axes as well. A shape that can be checked is
    /// worth more than one that must be inferred.
    pub columns: usize,
    /// The values, row-major: `values[row * columns + column]`.
    pub values: Vec<f64>,
}

impl Field {
    /// How many rows this field has.
    pub fn rows(&self) -> usize {
        // A zero-column field is a defect `validate_result` reports; this
        // just declines to divide by it.
        self.values.len().checked_div(self.columns).unwrap_or(0)
    }

    /// The value at `(row, column)`, or `None` when either is out of range.
    pub fn at(&self, row: usize, column: usize) -> Option<f64> {
        if column >= self.columns || row >= self.rows()
        {
            return None;
        }
        self.values.get(row * self.columns + column).copied()
    }

    /// One row, i.e. the field's profile at a single row coordinate.
    pub fn row(&self, row: usize) -> Option<&[f64]> {
        if row >= self.rows()
        {
            return None;
        }
        self.values
            .get(row * self.columns..(row + 1) * self.columns)
    }
}

/// The full, versioned result of one capability execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    /// [`RESULT_SCHEMA_VERSION`] at the time this result was produced.
    pub schema_version: u32,
    /// The capability that produced this result.
    pub capability_id: String,
    /// Human-facing summary.
    pub summary: RunSummary,
    /// Shared axes the series are plotted against, with their coordinates.
    pub axes: Vec<Axis>,
    /// Named output series, each bound to an axis by id.
    pub series: Vec<Series>,
    /// Named output fields, each spanning two axes.
    ///
    /// Defaulted, so every result written before fields existed reads back
    /// with none — which is what those results had.
    #[serde(default)]
    pub fields: Vec<Field>,
    /// Named scalar/derived metrics.
    pub metrics: Vec<Metric>,
    /// Non-fatal warnings raised during validation or execution.
    pub warnings: Vec<RunWarning>,
    /// Scientific verification checks and their outcomes.
    pub verifications: Vec<VerificationResult>,
    /// What produced this result and when.
    pub provenance: RunProvenance,
}

impl RunResult {
    /// Serialize to pretty-printed, stable-field-order JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// The axis a series is plotted against, if it exists.
    pub fn axis_for(&self, series: &Series) -> Option<&Axis> {
        self.axes.iter().find(|a| a.id == series.axis_id)
    }

    /// The primary time axis (`id == "t"`), if this result has one.
    pub fn time_axis(&self) -> Option<&Axis> {
        self.axes.iter().find(|a| a.id == TIME_AXIS_ID)
    }
}

/// The id every capability in this crate uses for its time axis.
pub const TIME_AXIS_ID: &str = "t";

/// Human-facing summary of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    /// The capability's display name at the time of the run.
    pub capability_display_name: String,
    /// The scenario's `experiment.name`.
    pub scenario_name: String,
    /// Number of integration steps taken (excluding the initial condition).
    pub steps: usize,
    /// Start time, in the axis's unit. Must equal the time axis's first
    /// coordinate exactly — both come from the same trajectory.
    pub t_start: f64,
    /// End time actually reached, in the axis's unit. Must equal the time
    /// axis's last coordinate exactly.
    pub t_end: f64,
}

/// A metric's value. Three kinds cover every metric an adapter in this
/// crate currently produces: a plain number, an integer count, and a
/// text classification (e.g. an RLC damping regime).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricValue {
    /// A finite floating-point value. Never `NaN`/infinite — see
    /// [`validate_result`].
    Scalar(f64),
    /// An integer count.
    Integer(i64),
    /// A text classification.
    Text(String),
}

/// One named scalar or derived metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    /// Stable id, e.g. `"peak_infected"`.
    pub id: String,
    /// Human-facing label.
    pub display_name: String,
    /// The value.
    pub value: MetricValue,
    /// Unit symbol, when the metric has one (a classification like a
    /// damping regime does not).
    pub unit: Option<String>,
}

/// What kind of thing a warning is about — the brief's rule that "a warning
/// is not an error unless the operation cannot produce a meaningful result"
/// only works if warnings are categorised, not one undifferentiated bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningCategory {
    /// About the scenario's input, short of a hard validation failure.
    Input,
    /// About the numerics of the run itself.
    Numerical,
    /// About solver convergence.
    Convergence,
    /// About a resource limit approached or exceeded.
    Resource,
    /// About the execution backend.
    Backend,
    /// Raised by a verification check that did not fully pass.
    Verification,
}

/// A non-fatal warning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunWarning {
    /// The warning's category.
    pub category: WarningCategory,
    /// Human-facing message.
    pub message: String,
}

/// The outcome of one verification check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    /// The check passed within its threshold.
    Passed,
    /// The check did not fully pass but the result is still usable.
    Warning,
    /// The check failed.
    Failed,
    /// The check does not apply to this run's configuration (e.g. an
    /// energy-conservation check when the model configuration is dissipative
    /// by design).
    NotApplicable,
}

/// The result of one scientific verification check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Stable id, matching a
    /// `scirust_studio_registry::VerificationCheckDescriptor::id`.
    pub id: String,
    /// The outcome.
    pub status: VerificationStatus,
    /// The measured quantity the check was based on, if numeric.
    pub measured: Option<f64>,
    /// The threshold the measured quantity was compared against, if any.
    pub threshold: Option<f64>,
    /// Human-facing explanation of the check and its outcome.
    pub explanation: String,
}

/// What produced a [`RunResult`] and when.
///
/// Content hashing, the run manifest and the environment fingerprint live in
/// `scirust-studio-store`, which owns storage; this is what the *execution*
/// can report about itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunProvenance {
    /// The capability that produced this result.
    pub capability_id: String,
    /// The capability's determinism classification.
    pub determinism: DeterminismClass,
    /// The crate that implemented the adapter.
    pub adapter_crate: String,
    /// That crate's version at build time (`env!("CARGO_PKG_VERSION")` of
    /// `scirust-studio-runtime` itself — the one version this crate can
    /// stamp robustly; a per-dependency version for `scirust-sim` would
    /// need a build script to stay honest as that crate's version moves, so
    /// it is not included here).
    pub adapter_version: String,
    /// Wall-clock start time, RFC 3339 UTC.
    pub started_at_rfc3339: String,
    /// Wall-clock completion time, RFC 3339 UTC.
    pub completed_at_rfc3339: String,
    /// Monotonic elapsed wall-clock duration, in seconds.
    pub elapsed_seconds: f64,
    /// The seed the computation actually **consumed**, when it consumed one.
    ///
    /// `None` for a capability whose result does not depend on a seed — which
    /// is every capability whose determinism is
    /// [`DeterminismClass::StrictSameBinarySameTarget`]. Recording a scenario's
    /// `experiment.seed` there would imply the number mattered, and it did
    /// not; the scenario source is stored verbatim beside the result, so
    /// nothing is lost by leaving this empty.
    ///
    /// For a stochastic capability this is the whole reproducibility story:
    /// the result is one sample from a distribution, and this is the only
    /// input that makes it the *same* sample twice. Validation refuses to run
    /// such a capability without one.
    ///
    /// Added after schema v2 shipped, as an optional field with a default
    /// rather than as v3. That is sound in both directions here: a v2 file
    /// written before this field simply has no `seed` and reads back as
    /// `None`, and no type in this crate or the store uses
    /// `deny_unknown_fields`, so a v2 reader built before the field ignores
    /// it rather than failing. A version bump would have bought a second
    /// compatibility path and no additional guarantee.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// A way in which a [`RunResult`] is internally inconsistent.
///
/// Every variant names the specific thing that is wrong, because "invalid
/// result" tells whoever is debugging an adapter nothing at all.
#[derive(Debug, Clone, PartialEq)]
pub enum ResultDefect {
    /// An axis coordinate is `NaN` or infinite.
    NonFiniteAxisValue {
        /// The axis.
        axis_id: String,
        /// Index of the offending coordinate.
        index: usize,
        /// The offending value, rendered.
        value: String,
    },
    /// A series value is `NaN` or infinite.
    NonFiniteSeriesValue {
        /// The series.
        series_id: String,
        /// Index of the offending value.
        index: usize,
        /// The offending value, rendered.
        value: String,
    },
    /// A `Scalar` metric is `NaN` or infinite.
    NonFiniteMetric {
        /// The metric.
        metric_id: String,
        /// The offending value, rendered.
        value: String,
    },
    /// A series names an axis that does not exist.
    MissingAxis {
        /// The series.
        series_id: String,
        /// The axis it asked for.
        axis_id: String,
    },
    /// Two axes share an id.
    DuplicateAxisId {
        /// The repeated id.
        axis_id: String,
    },
    /// Two series share an id.
    DuplicateSeriesId {
        /// The repeated id.
        series_id: String,
    },
    /// A series and its axis have different lengths.
    LengthMismatch {
        /// The series.
        series_id: String,
        /// How many values the series has.
        series_len: usize,
        /// The axis.
        axis_id: String,
        /// How many coordinates the axis has.
        axis_len: usize,
    },
    /// An axis has no coordinates although a series depends on it.
    EmptyAxisWithData {
        /// The axis.
        axis_id: String,
        /// A series that depends on it.
        series_id: String,
    },
    /// An axis broke the ordering it declared.
    NotMonotonic {
        /// The axis.
        axis_id: String,
        /// Index of the coordinate that broke the order.
        index: usize,
        /// The previous coordinate, rendered.
        previous: String,
        /// The offending coordinate, rendered.
        value: String,
    },
    /// `summary.t_start` disagrees with the time axis's first coordinate.
    SummaryStartMismatch {
        /// What the summary says.
        summary: String,
        /// What the axis says.
        axis: String,
    },
    /// `summary.t_end` disagrees with the time axis's last coordinate.
    SummaryEndMismatch {
        /// What the summary says.
        summary: String,
        /// What the axis says.
        axis: String,
    },
    /// More than one series claims the same singular ensemble role.
    DuplicateEnsembleRole {
        /// The role claimed twice.
        role: String,
        /// The second series to claim it.
        series_id: String,
    },
    /// An ensemble band edge is present with no mean to be an edge of.
    EnsembleBandWithoutMean {
        /// The band edge.
        series_id: String,
    },
    /// An ensemble band edge crosses the mean it brackets.
    EnsembleBandDoesNotBracketMean {
        /// The band edge.
        series_id: String,
        /// Where it crosses.
        index: usize,
        /// The edge's value there.
        edge: String,
        /// The mean's value there.
        mean: String,
    },
    /// Two fields share an id.
    DuplicateFieldId {
        /// The repeated id.
        field_id: String,
    },
    /// A field names an axis that does not exist.
    FieldMissingAxis {
        /// The field.
        field_id: String,
        /// The axis it asked for.
        axis_id: String,
    },
    /// A field's declared column count disagrees with its column axis.
    FieldShapeMismatch {
        /// The field.
        field_id: String,
        /// What the field says.
        declared: usize,
        /// What the axis says.
        axis_len: usize,
    },
    /// A field's value count is not `rows x columns`.
    FieldValueCountMismatch {
        /// The field.
        field_id: String,
        /// How many values it has.
        values: usize,
        /// How many it should have.
        expected: usize,
    },
    /// A field value is `NaN` or infinite.
    NonFiniteFieldValue {
        /// The field.
        field_id: String,
        /// The row.
        row: usize,
        /// The column.
        column: usize,
        /// The offending value, rendered.
        value: String,
    },
    /// Two series of one ensemble are plotted against different axes.
    EnsembleAxisMismatch {
        /// The series that disagrees.
        series_id: String,
        /// The axis it names.
        axis_id: String,
        /// The axis the rest of the ensemble uses.
        expected_axis_id: String,
    },
}

impl std::fmt::Display for ResultDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            ResultDefect::NonFiniteAxisValue {
                axis_id,
                index,
                value,
            } => write!(
                f,
                "axis `{axis_id}`[{index}] is {value}, not a finite coordinate"
            ),
            ResultDefect::NonFiniteSeriesValue {
                series_id,
                index,
                value,
            } => write!(
                f,
                "series `{series_id}`[{index}] is {value}, not a valid result value"
            ),
            ResultDefect::NonFiniteMetric { metric_id, value } =>
            {
                write!(
                    f,
                    "metric `{metric_id}` is {value}, not a valid result value"
                )
            },
            ResultDefect::MissingAxis { series_id, axis_id } => write!(
                f,
                "series `{series_id}` is plotted against axis `{axis_id}`, which this result does not contain"
            ),
            ResultDefect::DuplicateAxisId { axis_id } =>
            {
                write!(f, "two axes share the id `{axis_id}`")
            },
            ResultDefect::DuplicateSeriesId { series_id } =>
            {
                write!(f, "two series share the id `{series_id}`")
            },
            ResultDefect::LengthMismatch {
                series_id,
                series_len,
                axis_id,
                axis_len,
            } => write!(
                f,
                "series `{series_id}` has {series_len} values but its axis `{axis_id}` has {axis_len} coordinates"
            ),
            ResultDefect::EmptyAxisWithData { axis_id, series_id } => write!(
                f,
                "axis `{axis_id}` has no coordinates although series `{series_id}` depends on it"
            ),
            ResultDefect::NotMonotonic {
                axis_id,
                index,
                previous,
                value,
            } => write!(
                f,
                "axis `{axis_id}` declares an increasing order but [{index}] = {value} follows {previous}"
            ),
            ResultDefect::SummaryStartMismatch { summary, axis } => write!(
                f,
                "summary.t_start is {summary} but the time axis starts at {axis}"
            ),
            ResultDefect::SummaryEndMismatch { summary, axis } => write!(
                f,
                "summary.t_end is {summary} but the time axis ends at {axis}"
            ),
            ResultDefect::DuplicateEnsembleRole { role, series_id } => write!(
                f,
                "series `{series_id}` is a second `{role}`; an ensemble has one of each"
            ),
            ResultDefect::EnsembleBandWithoutMean { series_id } => write!(
                f,
                "series `{series_id}` is an ensemble band edge, but the result has no ensemble \
                 mean for it to be an edge of"
            ),
            ResultDefect::EnsembleBandDoesNotBracketMean {
                series_id,
                index,
                edge,
                mean,
            } => write!(
                f,
                "ensemble band edge `{series_id}` is {edge} at index {index}, on the wrong side \
                 of the mean's {mean}"
            ),
            ResultDefect::DuplicateFieldId { field_id } =>
            {
                write!(f, "two fields share the id `{field_id}`")
            },
            ResultDefect::FieldMissingAxis { field_id, axis_id } => write!(
                f,
                "field `{field_id}` names axis `{axis_id}`, which this result does not have"
            ),
            ResultDefect::FieldShapeMismatch {
                field_id,
                declared,
                axis_len,
            } => write!(
                f,
                "field `{field_id}` declares {declared} columns but its column axis has \
                 {axis_len} coordinates"
            ),
            ResultDefect::FieldValueCountMismatch {
                field_id,
                values,
                expected,
            } => write!(
                f,
                "field `{field_id}` has {values} values where its two axes require {expected}"
            ),
            ResultDefect::NonFiniteFieldValue {
                field_id,
                row,
                column,
                value,
            } => write!(
                f,
                "field `{field_id}` value at row {row}, column {column} is {value}"
            ),
            ResultDefect::EnsembleAxisMismatch {
                series_id,
                axis_id,
                expected_axis_id,
            } => write!(
                f,
                "ensemble series `{series_id}` is plotted against axis `{axis_id}` while the \
                 rest of the ensemble uses `{expected_axis_id}`"
            ),
        }
    }
}

/// Check that a [`RunResult`] is internally consistent, returning every
/// defect found rather than only the first.
///
/// Called by every adapter immediately before returning a successful result,
/// so a malformed one can never reach a caller. The checks exist because
/// each corresponds to a way a consumer would otherwise be silently
/// misled — a non-finite value becomes JSON `null`, a missing axis becomes
/// an unplottable series, a length mismatch becomes points drawn against
/// the wrong coordinates, and a non-monotonic time axis becomes a chart
/// that doubles back on itself.
///
/// `summary.t_start`/`t_end` are compared to the time axis with **exact**
/// equality, because both are read out of the same stored trajectory: any
/// difference at all means one of them was recomputed rather than carried
/// through, which is precisely the bug this schema version exists to
/// prevent.
pub fn validate_result(result: &RunResult) -> Result<(), Vec<ResultDefect>> {
    let mut defects = Vec::new();

    let mut seen_axes = std::collections::BTreeSet::new();
    for axis in &result.axes
    {
        if !seen_axes.insert(axis.id.as_str())
        {
            defects.push(ResultDefect::DuplicateAxisId {
                axis_id: axis.id.clone(),
            });
        }
        for (index, value) in axis.values.iter().enumerate()
        {
            if !value.is_finite()
            {
                defects.push(ResultDefect::NonFiniteAxisValue {
                    axis_id: axis.id.clone(),
                    index,
                    value: value.to_string(),
                });
            }
        }
        check_monotonicity(axis, &mut defects);
    }

    let mut seen_series = std::collections::BTreeSet::new();
    for series in &result.series
    {
        if !seen_series.insert(series.id.as_str())
        {
            defects.push(ResultDefect::DuplicateSeriesId {
                series_id: series.id.clone(),
            });
        }
        for (index, value) in series.values.iter().enumerate()
        {
            if !value.is_finite()
            {
                defects.push(ResultDefect::NonFiniteSeriesValue {
                    series_id: series.id.clone(),
                    index,
                    value: value.to_string(),
                });
            }
        }

        match result.axes.iter().find(|a| a.id == series.axis_id)
        {
            None => defects.push(ResultDefect::MissingAxis {
                series_id: series.id.clone(),
                axis_id: series.axis_id.clone(),
            }),
            Some(axis) =>
            {
                if axis.values.is_empty() && !series.values.is_empty()
                {
                    defects.push(ResultDefect::EmptyAxisWithData {
                        axis_id: axis.id.clone(),
                        series_id: series.id.clone(),
                    });
                }
                else if axis.values.len() != series.values.len()
                {
                    defects.push(ResultDefect::LengthMismatch {
                        series_id: series.id.clone(),
                        series_len: series.values.len(),
                        axis_id: axis.id.clone(),
                        axis_len: axis.values.len(),
                    });
                }
            },
        }
    }

    for metric in &result.metrics
    {
        if let MetricValue::Scalar(v) = &metric.value
        {
            if !v.is_finite()
            {
                defects.push(ResultDefect::NonFiniteMetric {
                    metric_id: metric.id.clone(),
                    value: v.to_string(),
                });
            }
        }
    }

    check_fields(result, &mut defects);
    check_ensemble(result, &mut defects);

    if let Some(axis) = result.time_axis()
    {
        if let (Some(first), Some(last)) = (axis.values.first(), axis.values.last())
        {
            if result.summary.t_start != *first
            {
                defects.push(ResultDefect::SummaryStartMismatch {
                    summary: result.summary.t_start.to_string(),
                    axis: first.to_string(),
                });
            }
            if result.summary.t_end != *last
            {
                defects.push(ResultDefect::SummaryEndMismatch {
                    summary: result.summary.t_end.to_string(),
                    axis: last.to_string(),
                });
            }
        }
    }

    if defects.is_empty()
    {
        Ok(())
    }
    else
    {
        Err(defects)
    }
}

/// The internal consistency of every field.
///
/// A field is the one thing in a result whose shape can be wrong in a way
/// that is invisible in the numbers: a row-major buffer read with the wrong
/// column count is still a buffer of finite doubles, and would render as a
/// plausible-looking picture of nothing. So the shape is checked twice —
/// against the declared column count and against both axes — rather than
/// trusted from either.
fn check_fields(result: &RunResult, defects: &mut Vec<ResultDefect>) {
    let mut seen = std::collections::BTreeSet::new();
    for field in &result.fields
    {
        if !seen.insert(field.id.as_str())
        {
            defects.push(ResultDefect::DuplicateFieldId {
                field_id: field.id.clone(),
            });
        }

        let row_axis = result.axes.iter().find(|a| a.id == field.row_axis_id);
        let column_axis = result.axes.iter().find(|a| a.id == field.column_axis_id);
        for (axis, id) in [
            (row_axis, &field.row_axis_id),
            (column_axis, &field.column_axis_id),
        ]
        {
            if axis.is_none()
            {
                defects.push(ResultDefect::FieldMissingAxis {
                    field_id: field.id.clone(),
                    axis_id: id.clone(),
                });
            }
        }
        let (Some(row_axis), Some(column_axis)) = (row_axis, column_axis)
        else
        {
            continue;
        };

        if field.columns != column_axis.values.len()
        {
            defects.push(ResultDefect::FieldShapeMismatch {
                field_id: field.id.clone(),
                declared: field.columns,
                axis_len: column_axis.values.len(),
            });
            // Every later check derives from the column count, so reporting
            // them too would be one mistake described five ways.
            continue;
        }

        let expected = row_axis.values.len() * column_axis.values.len();
        if field.values.len() != expected
        {
            defects.push(ResultDefect::FieldValueCountMismatch {
                field_id: field.id.clone(),
                values: field.values.len(),
                expected,
            });
            continue;
        }

        for (index, value) in field.values.iter().enumerate()
        {
            if !value.is_finite()
            {
                defects.push(ResultDefect::NonFiniteFieldValue {
                    field_id: field.id.clone(),
                    row: index / field.columns.max(1),
                    column: index % field.columns.max(1),
                    value: value.to_string(),
                });
                // One report per field: a diverged field is usually
                // non-finite from some point onward, and thousands of
                // identical defects would bury every other problem.
                break;
            }
        }
    }
}

/// The internal consistency of an ensemble's presentation.
///
/// Nothing here is about the *statistics* — whether the mean is in the right
/// place is a question for a capability's own verification checks, which know
/// the distribution being sampled. These are the structural claims that hold
/// for any ensemble whatever it is sampling: there is one mean, a band edge
/// is an edge of something, the edges are on the sides they say they are on,
/// and the whole ensemble is drawn against one axis.
///
/// A result with no ensemble series passes trivially, which is every result
/// from every deterministic capability.
fn check_ensemble(result: &RunResult, defects: &mut Vec<ResultDefect>) {
    let mut mean: Option<&Series> = None;
    let mut lower: Option<&Series> = None;
    let mut upper: Option<&Series> = None;
    let mut axis: Option<&str> = None;

    for series in result.series.iter().filter(|s| s.role.is_ensemble())
    {
        match axis
        {
            None => axis = Some(&series.axis_id),
            Some(expected) if expected != series.axis_id =>
            {
                defects.push(ResultDefect::EnsembleAxisMismatch {
                    series_id: series.id.clone(),
                    axis_id: series.axis_id.clone(),
                    expected_axis_id: expected.to_string(),
                });
            },
            Some(_) =>
            {},
        }

        let slot = match series.role
        {
            SeriesRole::EnsembleMean => Some((&mut mean, "ensemble mean")),
            SeriesRole::EnsembleBandLower => Some((&mut lower, "lower band edge")),
            SeriesRole::EnsembleBandUpper => Some((&mut upper, "upper band edge")),
            // Members are the one ensemble role there can be many of.
            _ => None,
        };
        if let Some((slot, name)) = slot
        {
            if slot.is_some()
            {
                defects.push(ResultDefect::DuplicateEnsembleRole {
                    role: name.to_string(),
                    series_id: series.id.clone(),
                });
            }
            else
            {
                *slot = Some(series);
            }
        }
    }

    for (edge, is_lower) in [(lower, true), (upper, false)]
    {
        let Some(edge) = edge
        else
        {
            continue;
        };
        let Some(mean) = mean
        else
        {
            defects.push(ResultDefect::EnsembleBandWithoutMean {
                series_id: edge.id.clone(),
            });
            continue;
        };
        for (index, (e, m)) in edge.values.iter().zip(mean.values.iter()).enumerate()
        {
            // `partial_cmp` rather than `>` / `<` so a NaN edge is a defect
            // rather than something that compares false against everything and
            // passes: an edge that cannot be ordered against the mean has not
            // been shown to bracket it.
            let wrong_side = match e.partial_cmp(m)
            {
                None => true,
                Some(ordering) =>
                {
                    if is_lower
                    {
                        ordering == std::cmp::Ordering::Greater
                    }
                    else
                    {
                        ordering == std::cmp::Ordering::Less
                    }
                },
            };
            if wrong_side
            {
                defects.push(ResultDefect::EnsembleBandDoesNotBracketMean {
                    series_id: edge.id.clone(),
                    index,
                    edge: e.to_string(),
                    mean: m.to_string(),
                });
                // One report per edge: a band that has crossed once has
                // usually crossed everywhere, and thousands of identical
                // defects would bury every other problem in the list.
                break;
            }
        }
    }
}

fn check_monotonicity(axis: &Axis, defects: &mut Vec<ResultDefect>) {
    let strict = match axis.monotonicity
    {
        AxisMonotonicity::StrictlyIncreasing => true,
        AxisMonotonicity::NonDecreasing => false,
        AxisMonotonicity::Unspecified => return,
    };
    for (index, pair) in axis.values.windows(2).enumerate()
    {
        // A non-finite coordinate is already reported separately; comparing
        // it here would add a second, less useful defect for one cause.
        if !pair[0].is_finite() || !pair[1].is_finite()
        {
            continue;
        }
        let ok = if strict
        {
            pair[1] > pair[0]
        }
        else
        {
            pair[1] >= pair[0]
        };
        if !ok
        {
            defects.push(ResultDefect::NotMonotonic {
                axis_id: axis.id.clone(),
                index: index + 1,
                previous: pair[0].to_string(),
                value: pair[1].to_string(),
            });
        }
    }
}

/// Render a defect list as one message, for an error type that carries a
/// string.
pub fn describe_defects(defects: &[ResultDefect]) -> String {
    defects
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A three-point ensemble on the standard fixture's axis.
    fn ensemble_result() -> RunResult {
        let mut result = sample_result();
        let s = |id: &str, role: SeriesRole, values: Vec<f64>| Series {
            id: id.to_string(),
            display_name: id.to_string(),
            unit: "m".to_string(),
            axis_id: TIME_AXIS_ID.to_string(),
            role,
            values,
        };
        result.series = vec![
            s(
                "ensemble_mean",
                SeriesRole::EnsembleMean,
                vec![1.0, 0.9, 0.8],
            ),
            s(
                "ensemble_band_lower",
                SeriesRole::EnsembleBandLower,
                vec![0.5, 0.4, 0.3],
            ),
            s(
                "ensemble_band_upper",
                SeriesRole::EnsembleBandUpper,
                vec![1.5, 1.4, 1.3],
            ),
            s("member_0", SeriesRole::EnsembleMember, vec![1.1, 0.7, 0.9]),
            s("member_1", SeriesRole::EnsembleMember, vec![0.9, 1.1, 0.7]),
        ];
        result
    }

    fn sample_result() -> RunResult {
        RunResult {
            schema_version: RESULT_SCHEMA_VERSION,
            capability_id: "sim.mechanics.spring_mass_damper".to_string(),
            summary: RunSummary {
                capability_display_name: "Spring-mass-damper".to_string(),
                scenario_name: "test".to_string(),
                steps: 2,
                t_start: 0.0,
                t_end: 2.0,
            },
            axes: vec![Axis {
                id: TIME_AXIS_ID.to_string(),
                display_name: "time".to_string(),
                unit: "s".to_string(),
                monotonicity: AxisMonotonicity::StrictlyIncreasing,
                values: vec![0.0, 1.0, 2.0],
            }],
            series: vec![Series {
                id: "position".to_string(),
                display_name: "Position".to_string(),
                unit: "m".to_string(),
                axis_id: TIME_AXIS_ID.to_string(),
                role: SeriesRole::Trajectory,
                values: vec![1.0, 0.9, 0.8],
            }],
            fields: vec![],
            metrics: vec![Metric {
                id: "final_position".to_string(),
                display_name: "Final position".to_string(),
                value: MetricValue::Scalar(0.8),
                unit: Some("m".to_string()),
            }],
            warnings: vec![],
            verifications: vec![VerificationResult {
                id: "energy_drift".to_string(),
                status: VerificationStatus::Passed,
                measured: Some(1e-14),
                threshold: Some(1e-6),
                explanation: "relative energy drift within tolerance".to_string(),
            }],
            provenance: RunProvenance {
                capability_id: "sim.mechanics.spring_mass_damper".to_string(),
                determinism: DeterminismClass::StrictSameBinarySameTarget,
                adapter_crate: "scirust-studio-runtime".to_string(),
                adapter_version: "0.1.0".to_string(),
                started_at_rfc3339: "2026-01-01T00:00:00Z".to_string(),
                completed_at_rfc3339: "2026-01-01T00:00:01Z".to_string(),
                elapsed_seconds: 1.0,
                seed: None,
            },
        }
    }

    fn defects_of(result: &RunResult) -> Vec<ResultDefect> {
        validate_result(result).unwrap_err()
    }

    #[test]
    fn the_schema_version_is_two() {
        assert_eq!(RESULT_SCHEMA_VERSION, 2);
        assert_eq!(sample_result().schema_version, 2);
    }

    #[test]
    fn a_well_formed_result_validates() {
        assert!(validate_result(&sample_result()).is_ok());
    }

    #[test]
    fn a_well_formed_ensemble_validates() {
        assert!(validate_result(&ensemble_result()).is_ok());
    }

    /// The ensemble rules must not fire on the nine capabilities that have
    /// no ensemble, which is what `sample_result` is.
    #[test]
    fn a_result_with_no_ensemble_is_not_subject_to_the_ensemble_rules() {
        let mut defects = Vec::new();
        check_ensemble(&sample_result(), &mut defects);
        assert!(defects.is_empty(), "{defects:?}");
    }

    #[test]
    fn a_second_ensemble_mean_is_rejected() {
        let mut result = ensemble_result();
        let mut twin = result.series[0].clone();
        twin.id = "ensemble_mean_again".to_string();
        result.series.push(twin);
        let defects = validate_result(&result).unwrap_err();
        assert!(matches!(
            defects.as_slice(),
            [ResultDefect::DuplicateEnsembleRole { .. }]
        ));
    }

    #[test]
    fn a_band_edge_with_no_mean_is_rejected() {
        let mut result = ensemble_result();
        result.series.retain(|s| s.role != SeriesRole::EnsembleMean);
        let defects = validate_result(&result).unwrap_err();
        assert_eq!(
            defects
                .iter()
                .filter(|d| matches!(d, ResultDefect::EnsembleBandWithoutMean { .. }))
                .count(),
            2,
            "both edges are orphaned: {defects:?}"
        );
    }

    #[test]
    fn a_band_edge_on_the_wrong_side_of_the_mean_is_rejected() {
        let mut result = ensemble_result();
        // Push the lower edge above the mean at one coordinate.
        result.series[1].values[1] = 99.0;
        let defects = validate_result(&result).unwrap_err();
        assert!(
            matches!(
                defects.as_slice(),
                [ResultDefect::EnsembleBandDoesNotBracketMean { index: 1, .. }]
            ),
            "{defects:?}"
        );
    }

    /// A crossed band is reported once, not once per coordinate: a defect
    /// list thousands long buries every other problem in it.
    #[test]
    fn a_band_that_is_wrong_everywhere_is_reported_once_per_edge() {
        let mut result = ensemble_result();
        result.series[1].values = vec![9.0, 9.0, 9.0];
        result.series[2].values = vec![-9.0, -9.0, -9.0];
        let defects = validate_result(&result).unwrap_err();
        assert_eq!(defects.len(), 2, "{defects:?}");
    }

    #[test]
    fn an_ensemble_split_across_two_axes_is_rejected() {
        let mut result = ensemble_result();
        result.axes.push(Axis {
            id: "other".to_string(),
            display_name: "other".to_string(),
            unit: "s".to_string(),
            monotonicity: AxisMonotonicity::StrictlyIncreasing,
            values: vec![0.0, 1.0, 2.0],
        });
        result.series[3].axis_id = "other".to_string();
        let defects = validate_result(&result).unwrap_err();
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::EnsembleAxisMismatch { .. })),
            "{defects:?}"
        );
    }

    /// A NaN edge must not slip past by comparing false against everything.
    #[test]
    fn a_band_edge_that_is_not_a_number_is_rejected() {
        let mut result = ensemble_result();
        result.series[1].values[0] = f64::NAN;
        let defects = validate_result(&result).unwrap_err();
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::EnsembleBandDoesNotBracketMean { .. })),
            "a NaN band edge was accepted as bracketing the mean: {defects:?}"
        );
    }

    #[test]
    fn a_non_finite_axis_coordinate_is_rejected() {
        let mut r = sample_result();
        r.axes[0].values[1] = f64::NAN;
        // Rebuild the summary so the only defect under test is the axis.
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::NonFiniteAxisValue { axis_id, index: 1, .. } if axis_id == "t")),
            "{defects:?}"
        );
    }

    #[test]
    fn a_non_finite_series_value_is_rejected() {
        let mut r = sample_result();
        r.series[0].values[1] = f64::INFINITY;
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::NonFiniteSeriesValue { series_id, .. } if series_id == "position")),
            "{defects:?}"
        );
    }

    #[test]
    fn an_infinite_metric_is_rejected() {
        let mut r = sample_result();
        r.metrics[0].value = MetricValue::Scalar(f64::INFINITY);
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::NonFiniteMetric { .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn a_text_metric_is_not_checked_for_finiteness() {
        let mut r = sample_result();
        r.metrics[0].value = MetricValue::Text("underdamped".to_string());
        assert!(validate_result(&r).is_ok());
    }

    #[test]
    fn a_series_naming_a_missing_axis_is_rejected() {
        let mut r = sample_result();
        r.series[0].axis_id = "no_such_axis".to_string();
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::MissingAxis { axis_id, .. } if axis_id == "no_such_axis")),
            "{defects:?}"
        );
    }

    #[test]
    fn duplicate_axis_ids_are_rejected() {
        let mut r = sample_result();
        let duplicate = r.axes[0].clone();
        r.axes.push(duplicate);
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::DuplicateAxisId { .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn duplicate_series_ids_are_rejected() {
        let mut r = sample_result();
        let duplicate = r.series[0].clone();
        r.series.push(duplicate);
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::DuplicateSeriesId { .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn a_series_shorter_than_its_axis_is_rejected() {
        let mut r = sample_result();
        r.series[0].values.pop();
        let defects = defects_of(&r);
        assert!(
            defects.iter().any(|d| matches!(
                d,
                ResultDefect::LengthMismatch {
                    series_len: 2,
                    axis_len: 3,
                    ..
                }
            )),
            "{defects:?}"
        );
    }

    #[test]
    fn an_empty_axis_with_a_populated_series_is_rejected() {
        let mut r = sample_result();
        r.axes[0].values.clear();
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::EmptyAxisWithData { .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn a_time_axis_that_goes_backwards_is_rejected() {
        let mut r = sample_result();
        r.axes[0].values = vec![0.0, 2.0, 1.0];
        r.summary.t_end = 1.0;
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::NotMonotonic { index: 2, .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn a_repeated_coordinate_breaks_strict_but_not_non_decreasing() {
        let mut r = sample_result();
        r.axes[0].values = vec![0.0, 1.0, 1.0];
        r.summary.t_end = 1.0;
        assert!(
            defects_of(&r)
                .iter()
                .any(|d| matches!(d, ResultDefect::NotMonotonic { .. })),
            "strictly-increasing must reject a repeat"
        );

        r.axes[0].monotonicity = AxisMonotonicity::NonDecreasing;
        assert!(
            validate_result(&r).is_ok(),
            "non-decreasing must accept a repeat"
        );
    }

    #[test]
    fn an_unspecified_axis_is_not_order_checked() {
        let mut r = sample_result();
        r.axes[0].monotonicity = AxisMonotonicity::Unspecified;
        r.axes[0].values = vec![5.0, 1.0, 3.0];
        r.summary.t_start = 5.0;
        r.summary.t_end = 3.0;
        assert!(validate_result(&r).is_ok());
    }

    /// The check that makes "the summary was carried through, not
    /// recomputed" enforceable: exact equality, not a tolerance.
    #[test]
    fn a_summary_start_one_ulp_off_the_axis_is_rejected() {
        let mut r = sample_result();
        r.summary.t_start = f64::from_bits(r.axes[0].values[0].to_bits() + 1);
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::SummaryStartMismatch { .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn a_summary_end_that_disagrees_with_the_axis_is_rejected() {
        let mut r = sample_result();
        r.summary.t_end = 2.5;
        let defects = defects_of(&r);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::SummaryEndMismatch { .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn every_defect_is_reported_not_just_the_first() {
        let mut r = sample_result();
        r.series[0].values[0] = f64::NAN;
        r.metrics[0].value = MetricValue::Scalar(f64::NAN);
        r.summary.t_end = 99.0;
        let defects = defects_of(&r);
        assert!(defects.len() >= 3, "{defects:?}");
    }

    #[test]
    fn round_trips_through_json() {
        let result = sample_result();
        let json = serde_json::to_string(&result).unwrap();
        let parsed: RunResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, parsed);
    }

    /// Adaptive solvers produce coordinates that no linear reconstruction
    /// could reproduce; they must survive serialization exactly.
    #[test]
    fn adaptive_axis_coordinates_survive_json_bit_for_bit() {
        let mut r = sample_result();
        r.axes[0].values = vec![0.0, 1.234_567_890_123_456_7e-8, 4.9e-3, 0.4];
        r.series[0].values = vec![1.0, 0.999, 0.5, 0.25];
        r.summary.t_start = 0.0;
        r.summary.t_end = 0.4;
        r.summary.steps = 3;
        validate_result(&r).expect("valid");

        let json = serde_json::to_string(&r).unwrap();
        let parsed: RunResult = serde_json::from_str(&json).unwrap();
        for (a, b) in r.axes[0].values.iter().zip(parsed.axes[0].values.iter())
        {
            assert_eq!(a.to_bits(), b.to_bits(), "{a:?} became {b:?}");
        }
    }

    #[test]
    fn json_uses_stable_snake_case_variant_names() {
        let json = serde_json::to_string(&sample_result()).unwrap();
        assert!(json.contains("\"passed\""), "{json}");
        assert!(json.contains("\"strictly_increasing\""), "{json}");
        assert!(
            json.contains("\"scalar\":0.8") || json.contains("\"scalar\": 0.8"),
            "{json}"
        );
    }

    #[test]
    fn axis_for_resolves_a_series_to_its_axis() {
        let r = sample_result();
        let axis = r.axis_for(&r.series[0]).expect("resolves");
        assert_eq!(axis.id, TIME_AXIS_ID);
        assert_eq!(r.time_axis().map(|a| a.id.as_str()), Some(TIME_AXIS_ID));
    }

    #[test]
    fn describe_defects_joins_every_message() {
        let mut r = sample_result();
        r.series[0].values[0] = f64::NAN;
        r.summary.t_end = 99.0;
        let text = describe_defects(&defects_of(&r));
        assert!(text.contains("position"), "{text}");
        assert!(text.contains("t_end"), "{text}");
    }

    fn field_result() -> RunResult {
        let mut result = sample_result();
        result.axes.push(Axis {
            id: "x".to_string(),
            display_name: "position".to_string(),
            unit: "m".to_string(),
            monotonicity: AxisMonotonicity::StrictlyIncreasing,
            values: vec![0.0, 0.5],
        });
        result.fields = vec![Field {
            id: "temperature".to_string(),
            display_name: "Temperature".to_string(),
            unit: "K".to_string(),
            row_axis_id: TIME_AXIS_ID.to_string(),
            column_axis_id: "x".to_string(),
            columns: 2,
            // 3 rows (the time axis) x 2 columns.
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }];
        result
    }

    #[test]
    fn a_well_formed_field_validates() {
        assert!(validate_result(&field_result()).is_ok());
    }

    /// The accessors have to agree with the packing, or every consumer reads
    /// a transposed picture that still looks like data.
    #[test]
    fn the_field_reads_back_row_major() {
        let result = field_result();
        let field = &result.fields[0];
        assert_eq!(field.rows(), 3);
        assert_eq!(field.at(0, 0), Some(1.0));
        assert_eq!(field.at(0, 1), Some(2.0));
        assert_eq!(field.at(1, 0), Some(3.0));
        assert_eq!(field.at(2, 1), Some(6.0));
        assert_eq!(field.row(1), Some(&[3.0, 4.0][..]));
        assert_eq!(field.at(3, 0), None);
        assert_eq!(field.at(0, 2), None);
        assert_eq!(field.row(3), None);
    }

    /// A shape error is the one field defect that leaves every number finite
    /// and plausible, so it is the one most worth catching.
    #[test]
    fn a_column_count_disagreeing_with_its_axis_is_rejected() {
        let mut result = field_result();
        result.fields[0].columns = 3;
        let defects = validate_result(&result).unwrap_err();
        assert!(
            matches!(
                defects.as_slice(),
                [ResultDefect::FieldShapeMismatch {
                    declared: 3,
                    axis_len: 2,
                    ..
                }]
            ),
            "{defects:?}"
        );
    }

    #[test]
    fn a_field_with_the_wrong_number_of_values_is_rejected() {
        let mut result = field_result();
        result.fields[0].values.pop();
        let defects = validate_result(&result).unwrap_err();
        assert!(
            matches!(
                defects.as_slice(),
                [ResultDefect::FieldValueCountMismatch {
                    values: 5,
                    expected: 6,
                    ..
                }]
            ),
            "{defects:?}"
        );
    }

    #[test]
    fn a_field_naming_a_missing_axis_is_rejected() {
        let mut result = field_result();
        result.fields[0].column_axis_id = "nowhere".to_string();
        let defects = validate_result(&result).unwrap_err();
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::FieldMissingAxis { .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn a_non_finite_field_value_is_rejected_with_its_position() {
        let mut result = field_result();
        result.fields[0].values[3] = f64::NAN;
        let defects = validate_result(&result).unwrap_err();
        assert!(
            matches!(
                defects.as_slice(),
                [ResultDefect::NonFiniteFieldValue {
                    row: 1,
                    column: 1,
                    ..
                }]
            ),
            "{defects:?}"
        );
    }

    #[test]
    fn duplicate_field_ids_are_rejected() {
        let mut result = field_result();
        let twin = result.fields[0].clone();
        result.fields.push(twin);
        let defects = validate_result(&result).unwrap_err();
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, ResultDefect::DuplicateFieldId { .. })),
            "{defects:?}"
        );
    }

    /// Results written before fields existed have no `fields` key at all and
    /// must keep reading back.
    #[test]
    fn a_result_stored_before_fields_existed_decodes_with_none() {
        let mut json: serde_json::Value =
            serde_json::to_value(sample_result()).expect("serialises");
        json.as_object_mut().expect("an object").remove("fields");
        let decoded: RunResult = serde_json::from_value(json).expect("decodes without `fields`");
        assert!(decoded.fields.is_empty());
        assert!(validate_result(&decoded).is_ok());
    }

    #[test]
    fn a_field_round_trips_through_json_bit_for_bit() {
        let result = field_result();
        let text = result.to_json_pretty().expect("serialises");
        let back: RunResult = serde_json::from_str(&text).expect("decodes");
        assert_eq!(back.fields, result.fields);
    }
}
