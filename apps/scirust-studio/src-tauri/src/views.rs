//! The wire types the frontend receives.
//!
//! Deliberately separate from the app service's own types: the frontend is a
//! WASM module reached over a JSON bridge, and giving it a view model rather
//! than the internal representation means the internal one can change
//! without breaking the bridge, and the bridge carries only what a UI
//! actually needs.

use serde::{Deserialize, Serialize};

use scirust_studio_app_service::{
    AppServiceError, JobSnapshot, RunVerificationReport, StoredRunSummary, ValidationOutcome,
};
use scirust_studio_registry::CapabilityDescriptor;
use scirust_studio_runtime::{Metric, MetricValue, RunWarning, VerificationResult};
use scirust_studio_store::LoadedRunResult;

/// A structured error the interface can render as a real card.
///
/// Every field exists because an error card needs it. There is deliberately
/// no path by which the UI ends up with only "something went wrong".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorView {
    /// Stable machine-readable code.
    pub code: String,
    /// Heading.
    pub title: String,
    /// Full explanation.
    pub explanation: String,
    /// Whether the user can carry on.
    pub recoverable: bool,
    /// What to try next, when there is something concrete.
    pub suggested_action: Option<String>,
    /// Structured validation problems, when this was a validation failure.
    pub validation: Option<ValidationOutcome>,
}

impl From<AppServiceError> for ErrorView {
    fn from(e: AppServiceError) -> Self {
        let validation = match &e
        {
            AppServiceError::Validation(outcome) => Some((**outcome).clone()),
            _ => None,
        };
        ErrorView {
            code: format!("{:?}", e.code()),
            title: e.title().to_string(),
            explanation: e.to_string(),
            recoverable: e.recoverable(),
            suggested_action: e.suggested_action().map(str::to_string),
            validation,
        }
    }
}

/// What the application knows at start-up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapView {
    /// The application's version.
    pub app_version: String,
    /// The worker's version, when one is running.
    pub worker_version: Option<String>,
    /// Whether a worker is up. When false, Run must be disabled and the
    /// catalogue and help must still work.
    pub worker_running: bool,
    /// Where runs are stored, for the "reveal" action and the first-launch
    /// notice.
    pub store_path: String,
    /// How many capabilities can actually be executed.
    pub capability_count: usize,
    /// Runs whose writer never finished, discovered at start-up.
    pub interrupted_runs: Vec<String>,
    /// Why the worker is not running, when it is not.
    pub worker_problem: Option<ErrorView>,
}

/// One capability, as the catalogue shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityView {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Category.
    pub category: String,
    /// The crate that implements the model.
    pub source_crate: String,
    /// One-line description.
    pub summary: String,
    /// Maturity.
    pub maturity: String,
    /// Determinism classification.
    pub determinism: String,
    /// Solver ids this capability accepts.
    pub solvers: Vec<String>,
    /// Model parameters.
    pub parameters: Vec<FieldView>,
    /// Initial-state fields.
    pub initial_state: Vec<FieldView>,
    /// Output series it produces.
    pub outputs: Vec<OutputView>,
    /// Scientific checks it performs.
    pub checks: Vec<CheckView>,
    /// Whether it can report genuine progress. False means the UI must show
    /// indeterminate activity, never a percentage.
    pub supports_progress: bool,
    /// Whether a tested tutorial ships for it.
    pub has_tutorial: bool,
}

/// One parameter or initial-state field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldView {
    /// Canonical name in the scenario file.
    pub name: String,
    /// Display name.
    pub display_name: String,
    /// Whether it must be present.
    pub required: bool,
    /// Units it accepts.
    pub units: Vec<String>,
    /// Description.
    pub description: String,
}

/// One output series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputView {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Unit.
    pub unit: String,
    /// Description.
    pub description: String,
}

/// One scientific check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckView {
    /// Stable id.
    pub id: String,
    /// What it checks.
    pub description: String,
}

impl From<&CapabilityDescriptor> for CapabilityView {
    fn from(d: &CapabilityDescriptor) -> Self {
        CapabilityView {
            id: d.id.0.to_string(),
            display_name: d.display_name.to_string(),
            category: format!("{:?}", d.category),
            source_crate: d.source_crate.to_string(),
            summary: d.summary.to_string(),
            maturity: format!("{:?}", d.maturity),
            determinism: format!("{:?}", d.determinism),
            solvers: d
                .supported_solvers
                .iter()
                .map(|s| s.id.to_string())
                .collect(),
            parameters: d.parameters.iter().map(FieldView::from).collect(),
            initial_state: d.initial_state.iter().map(FieldView::from).collect(),
            outputs: d
                .outputs
                .iter()
                .map(|o| OutputView {
                    id: o.id.to_string(),
                    display_name: o.display_name.to_string(),
                    unit: o.unit.to_string(),
                    description: o.description.to_string(),
                })
                .collect(),
            checks: d
                .verification
                .checks
                .iter()
                .map(|c| CheckView {
                    id: c.id.to_string(),
                    description: c.description.to_string(),
                })
                .collect(),
            supports_progress: scirust_studio_app_service::capability_supports_progress(d),
            has_tutorial: scirust_studio_runtime::tutorial_scenario_for(d.id.0).is_some(),
        }
    }
}

impl From<&scirust_studio_registry::FieldDescriptor> for FieldView {
    fn from(f: &scirust_studio_registry::FieldDescriptor) -> Self {
        FieldView {
            name: f.canonical_name.to_string(),
            display_name: f.display_name.to_string(),
            required: f.required,
            units: f.accepted_units.iter().map(|u| u.to_string()).collect(),
            description: f.description.to_string(),
        }
    }
}

/// A scenario opened into the editor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioView {
    /// The source text.
    pub source: String,
    /// Where it came from, when it came from a file.
    pub path: Option<String>,
    /// The capability it targets, when known.
    pub capability_id: Option<String>,
}

/// A quantity sampled on a two-axis grid — what
/// `scirust_studio_runtime::Field` carries.
///
/// Named `Grid`, not `Field`, because `FieldView`/`FieldWire` were already
/// taken by a *form* field: the description of one `model.*` parameter a
/// scenario may set. Two unrelated meanings of the word met here, and the
/// newcomer yields — renaming the older one would touch the catalogue panel,
/// the bridge contract and every capability view for a naming preference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridView {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Unit.
    pub unit: String,
    /// The axis the rows run along.
    pub row_axis_id: String,
    /// The axis the columns run along.
    pub column_axis_id: String,
    /// How many columns each row has.
    pub columns: usize,
    /// Row-major values.
    pub values: Vec<f64>,
}

/// A scenario file the user chose, as text.
///
/// Deliberately carries **no path**. The frontend gets the contents and a
/// bare name to put in a title; it cannot reconstruct a location from either,
/// and so cannot ask for the same file again without the user picking it
/// again. See `commands::studio_open_scenario`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioFileView {
    /// The file's own name, for display only.
    pub file_name: String,
    /// Its contents.
    pub source: String,
}

/// A run in progress or settled.
pub type JobView = JobSnapshot;

/// What a chart may claim its horizontal axis represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XAxisKind {
    /// Real coordinates from the integrator.
    PhysicalCoordinates,
    /// Only sample ordinals are known — a legacy result. The chart must say
    /// so and must not label the axis as time.
    SampleIndex,
}

/// What a plotted curve represents, mirroring
/// `scirust_studio_runtime::SeriesRole`.
///
/// Crossing the bridge as a typed enum rather than a string: the interface
/// *branches* on this to decide how to draw, and a mistyped string would
/// silently fall through to the default — drawing a summary statistic and one
/// noisy realisation with the same weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesRoleView {
    /// One computed trajectory.
    #[default]
    Trajectory,
    /// A comparison line the solver did not compute.
    Reference,
    /// One realisation out of an ensemble.
    EnsembleMember,
    /// The across-replicate mean.
    EnsembleMean,
    /// The lower edge of the spread band.
    EnsembleBandLower,
    /// The upper edge of the spread band.
    EnsembleBandUpper,
}

impl From<scirust_studio_runtime::SeriesRole> for SeriesRoleView {
    fn from(role: scirust_studio_runtime::SeriesRole) -> Self {
        use scirust_studio_runtime::SeriesRole as R;
        match role
        {
            R::Trajectory => SeriesRoleView::Trajectory,
            R::Reference => SeriesRoleView::Reference,
            R::EnsembleMember => SeriesRoleView::EnsembleMember,
            R::EnsembleMean => SeriesRoleView::EnsembleMean,
            R::EnsembleBandLower => SeriesRoleView::EnsembleBandLower,
            R::EnsembleBandUpper => SeriesRoleView::EnsembleBandUpper,
        }
    }
}

/// One plottable series with the axis it belongs to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesView {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Unit.
    pub unit: String,
    /// What this curve represents. Defaulted, so a run stored before roles
    /// existed reads back as a trajectory — which is what it was.
    #[serde(default)]
    pub role: SeriesRoleView,
    /// The values.
    pub values: Vec<f64>,
}

/// A stored run, loaded for display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunView {
    /// The run's id.
    pub run_id: String,
    /// The capability that produced it.
    pub capability_id: String,
    /// The experiment name.
    pub scenario_name: String,
    /// Terminal status.
    pub status: String,
    /// The result schema version it was stored under.
    pub result_schema_version: u32,
    /// What the horizontal axis genuinely represents.
    pub x_axis_kind: XAxisKind,
    /// The axis label — never "time" for a legacy result.
    pub x_axis_label: String,
    /// The axis unit, empty for a sample index.
    pub x_axis_unit: String,
    /// The axis coordinates. For a legacy result these are `0, 1, 2, …`
    /// sample ordinals, and `x_axis_kind` says so.
    pub x_values: Vec<f64>,
    /// The series.
    pub series: Vec<SeriesView>,
    /// Fields, each spanning two axes. Empty for every capability whose
    /// outputs are curves.
    #[serde(default)]
    pub fields: Vec<GridView>,
    /// Metrics.
    pub metrics: Vec<MetricView>,
    /// Scientific checks.
    pub verifications: Vec<VerificationView>,
    /// Warnings.
    pub warnings: Vec<WarningView>,
    /// Provenance.
    pub provenance: ProvenanceView,
    /// The scenario that produced it.
    pub scenario_source: String,
    /// Whether the stored bytes still match their hashes. **Separate** from
    /// the scientific checks: integrity and correctness are different
    /// questions.
    pub integrity: VerificationReportView,
}

/// A metric, pre-rendered for display but keeping its raw value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricView {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// The value, rendered.
    pub value: String,
    /// The numeric value, when it is one — so the UI can format it by locale
    /// without re-parsing a string.
    pub numeric: Option<f64>,
    /// Unit, when it has one.
    pub unit: Option<String>,
}

impl From<&Metric> for MetricView {
    fn from(m: &Metric) -> Self {
        let (value, numeric) = match &m.value
        {
            MetricValue::Scalar(v) => (format!("{v}"), Some(*v)),
            MetricValue::Integer(v) => (v.to_string(), Some(*v as f64)),
            MetricValue::Text(t) => (t.clone(), None),
        };
        MetricView {
            id: m.id.clone(),
            display_name: m.display_name.clone(),
            value,
            numeric,
            unit: m.unit.clone(),
        }
    }
}

/// One scientific check's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationView {
    /// Stable id.
    pub id: String,
    /// `passed` / `warning` / `failed` / `not_applicable`.
    pub status: String,
    /// What was measured.
    pub measured: Option<f64>,
    /// What it was compared against.
    pub threshold: Option<f64>,
    /// Explanation.
    pub explanation: String,
}

impl From<&VerificationResult> for VerificationView {
    fn from(v: &VerificationResult) -> Self {
        VerificationView {
            id: v.id.clone(),
            status: format!("{:?}", v.status).to_lowercase(),
            measured: v.measured,
            threshold: v.threshold,
            explanation: v.explanation.clone(),
        }
    }
}

/// A non-fatal warning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarningView {
    /// Category.
    pub category: String,
    /// Message.
    pub message: String,
}

impl From<&RunWarning> for WarningView {
    fn from(w: &RunWarning) -> Self {
        WarningView {
            category: format!("{:?}", w.category).to_lowercase(),
            message: w.message.clone(),
        }
    }
}

/// What produced a result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceView {
    /// The capability.
    pub capability_id: String,
    /// Determinism classification.
    pub determinism: String,
    /// The adapter crate.
    pub adapter_crate: String,
    /// Its version.
    pub adapter_version: String,
    /// Result schema version.
    pub result_schema_version: u32,
    /// The target it ran on.
    pub target: String,
    /// Start time.
    pub started_at: String,
    /// Completion time.
    pub completed_at: String,
    /// Elapsed seconds.
    pub elapsed_seconds: f64,
    /// The seed the computation consumed, when it consumed one. `None` for
    /// every capability whose result does not depend on a seed — showing a
    /// number there would imply it mattered.
    pub seed: Option<u64>,
}

/// The outcome of a store-integrity check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationReportView {
    /// The run checked.
    pub run_id: String,
    /// Whether every stored hash still matches.
    pub intact: bool,
    /// What changed, when something did.
    pub detail: Option<String>,
}

impl From<RunVerificationReport> for VerificationReportView {
    fn from(r: RunVerificationReport) -> Self {
        VerificationReportView {
            run_id: r.run_id,
            intact: r.intact,
            detail: r.detail,
        }
    }
}

/// A stored run in the history list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRunView {
    /// The run's id.
    pub run_id: String,
    /// The capability.
    pub capability_id: String,
    /// Terminal status.
    pub status: String,
    /// When it started.
    pub started_at: String,
    /// When it settled.
    pub finished_at: String,
    /// The result schema version.
    pub result_schema_version: u32,
}

impl From<StoredRunSummary> for StoredRunView {
    fn from(s: StoredRunSummary) -> Self {
        StoredRunView {
            run_id: s.run_id,
            capability_id: s.capability_id,
            status: s.status,
            started_at: s.started_at_rfc3339,
            finished_at: s.finished_at_rfc3339,
            result_schema_version: s.result_schema_version,
        }
    }
}

/// Build a [`RunView`] from a loaded run.
///
/// This is where the v1/v2 distinction becomes visible to the user. A v2
/// result is plotted against its stored coordinates; a v1 result is plotted
/// against sample ordinals and **labelled as such**, because nothing in a v1
/// file records when its samples were taken and inventing an axis would be a
/// picture of a different experiment.
pub fn run_view(
    run_id: &str,
    manifest: &scirust_studio_store::RunManifest,
    result: &LoadedRunResult,
    scenario_source: String,
    integrity: VerificationReportView,
) -> RunView {
    let target = format!("{}-{}", manifest.environment.os, manifest.environment.arch);

    let (x_axis_kind, x_axis_label, x_axis_unit, x_values, series, fields) = match result
    {
        LoadedRunResult::V2(r) =>
        {
            let axis = r.time_axis().or_else(|| r.axes.first());
            let (label, unit, values) = match axis
            {
                Some(a) => (a.display_name.clone(), a.unit.clone(), a.values.clone()),
                None => (String::new(), String::new(), Vec::new()),
            };
            let series = r
                .series
                .iter()
                .map(|s| SeriesView {
                    id: s.id.clone(),
                    display_name: s.display_name.clone(),
                    unit: s.unit.clone(),
                    role: s.role.into(),
                    values: s.values.clone(),
                })
                .collect();
            // The field is the *result* for a capability like the heat rod;
            // the series beside it are summaries of it. Dropping it here
            // would be the interface silently showing less than the run
            // produced.
            let fields = r
                .fields
                .iter()
                .map(|f| GridView {
                    id: f.id.clone(),
                    display_name: f.display_name.clone(),
                    unit: f.unit.clone(),
                    row_axis_id: f.row_axis_id.clone(),
                    column_axis_id: f.column_axis_id.clone(),
                    columns: f.columns,
                    values: f.values.clone(),
                })
                .collect();
            (
                XAxisKind::PhysicalCoordinates,
                label,
                unit,
                values,
                series,
                fields,
            )
        },
        LoadedRunResult::V1(r) =>
        {
            let length = r.series.first().map(|s| s.values.len()).unwrap_or(0);
            let series = r
                .series
                .iter()
                .map(|s| SeriesView {
                    id: s.id.clone(),
                    display_name: s.display_name.clone(),
                    unit: s.unit.clone(),
                    // Schema v1 predates roles entirely; every v1 series is a
                    // plain trajectory, which is what the default says.
                    role: SeriesRoleView::Trajectory,
                    values: s.values.clone(),
                })
                .collect();
            (
                XAxisKind::SampleIndex,
                "Sample index".to_string(),
                String::new(),
                (0..length).map(|i| i as f64).collect(),
                series,
                // Schema v1 has no notion of a field.
                Vec::new(),
            )
        },
    };

    RunView {
        run_id: run_id.to_string(),
        capability_id: result.capability_id().to_string(),
        scenario_name: result.summary().scenario_name.clone(),
        status: manifest.status.label().to_string(),
        result_schema_version: result.schema_version(),
        x_axis_kind,
        x_axis_label,
        x_axis_unit,
        x_values,
        series,
        fields,
        metrics: result.metrics().iter().map(MetricView::from).collect(),
        verifications: result
            .verifications()
            .iter()
            .map(VerificationView::from)
            .collect(),
        warnings: result.warnings().iter().map(WarningView::from).collect(),
        provenance: ProvenanceView {
            capability_id: result.provenance().capability_id.clone(),
            determinism: format!("{:?}", result.provenance().determinism),
            adapter_crate: result.provenance().adapter_crate.clone(),
            adapter_version: result.provenance().adapter_version.clone(),
            result_schema_version: result.schema_version(),
            target,
            started_at: result.provenance().started_at_rfc3339.clone(),
            completed_at: result.provenance().completed_at_rfc3339.clone(),
            elapsed_seconds: result.provenance().elapsed_seconds,
            seed: result.provenance().seed,
        },
        scenario_source,
        integrity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalogue_view_carries_everything_the_ui_shows() {
        let registry = scirust_studio_runtime::build_registry();
        for descriptor in registry.iter()
        {
            let view = CapabilityView::from(descriptor);
            assert!(!view.id.is_empty());
            assert!(!view.display_name.is_empty());
            assert!(!view.summary.is_empty());
            assert!(!view.solvers.is_empty());
            assert!(!view.outputs.is_empty());
            assert!(!view.checks.is_empty());
            assert!(view.has_tutorial, "{} ships a tutorial", view.id);

            match view.id.as_str()
            {
                "sim.chemistry.robertson" => assert!(
                    !view.supports_progress,
                    "the adaptive path must be flagged as indeterminate"
                ),
                _ => assert!(view.supports_progress),
            }
        }
    }

    #[test]
    fn an_error_view_always_has_a_code_and_a_title() {
        let view = ErrorView::from(AppServiceError::Busy {
            active_job_id: "job-1".to_string(),
        });
        assert_eq!(view.code, "Busy");
        assert!(!view.title.is_empty());
        assert!(!view.explanation.is_empty());
        assert!(view.recoverable);
        assert!(view.suggested_action.is_some());
    }

    /// A validation failure must reach the editor as structured problems,
    /// not as a formatted blob.
    #[test]
    fn a_validation_error_carries_its_problems() {
        let outcome = scirust_studio_app_service::validate_source("not = = toml");
        let view = ErrorView::from(AppServiceError::Validation(Box::new(outcome)));
        let problems = view.validation.expect("problems").problems;
        assert!(!problems.is_empty());
    }

    #[test]
    fn metric_views_keep_the_numeric_value_for_locale_formatting() {
        let m = Metric {
            id: "x".into(),
            display_name: "X".into(),
            value: MetricValue::Scalar(1.5),
            unit: Some("m".into()),
        };
        let view = MetricView::from(&m);
        assert_eq!(view.numeric, Some(1.5));

        let text = Metric {
            id: "regime".into(),
            display_name: "Regime".into(),
            value: MetricValue::Text("underdamped".into()),
            unit: None,
        };
        assert_eq!(MetricView::from(&text).numeric, None);
    }
}
