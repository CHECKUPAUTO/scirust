//! The types that cross the Tauri bridge.
//!
//! These mirror `src-tauri/src/views.rs` field for field. They are declared
//! separately rather than shared because the frontend compiles to
//! `wasm32-unknown-unknown` and the shell's view module transitively pulls in
//! `tauri`, the registry, the store and the worker supervisor — none of which
//! can or should be compiled into a WebAssembly module.
//!
//! The two sides are kept honest by `src-tauri/tests/bridge_contract.rs`,
//! which serialises a real view value with `serde_json` and deserialises it
//! into the type here. A field renamed on one side and not the other fails
//! that test rather than producing a silently empty panel at runtime.
//!
//! Everything here is pure data: it compiles on the host, so the decoding is
//! unit tested without a browser.

use serde::{Deserialize, Serialize};

/// A structured error, as the shell reports it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorWire {
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
    pub validation: Option<ValidationWire>,
}

/// The outcome of validating scenario source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationWire {
    /// Whether the scenario can be run.
    pub valid: bool,
    /// The capability it targets, when that could be determined.
    pub capability_id: Option<String>,
    /// The experiment name, when parsing got that far.
    pub scenario_name: Option<String>,
    /// Everything found, errors and warnings alike.
    pub problems: Vec<ProblemWire>,
}

/// One problem found in a scenario.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProblemWire {
    /// Stable `SRST-…` code, when the underlying error carries one.
    pub code: Option<String>,
    /// Short heading.
    pub title: String,
    /// Full explanation.
    pub explanation: String,
    /// What to do about it, when known.
    pub suggested_action: Option<String>,
    /// The field this is about, when identifiable.
    pub field: Option<String>,
    /// Where in the source, when determinable.
    pub location: Option<LocationWire>,
    /// Which stage rejected it: `parse`, `schema` or `capability`.
    pub stage: String,
    /// `error` or `warning`.
    pub severity: String,
}

/// A position in the scenario source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocationWire {
    /// 1-based line.
    pub line: usize,
    /// 1-based column.
    pub column: usize,
}

/// What the application knows at start-up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapWire {
    /// The application's version.
    pub app_version: String,
    /// The worker's version, when one is running.
    pub worker_version: Option<String>,
    /// Whether a worker is up.
    pub worker_running: bool,
    /// Where runs are stored.
    pub store_path: String,
    /// How many capabilities can actually be executed.
    pub capability_count: usize,
    /// Runs whose writer never finished, discovered at start-up.
    pub interrupted_runs: Vec<String>,
    /// Why the worker is not running, when it is not.
    pub worker_problem: Option<ErrorWire>,
}

/// One capability, as the catalogue shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityWire {
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
    pub parameters: Vec<FieldWire>,
    /// Initial-state fields.
    pub initial_state: Vec<FieldWire>,
    /// Output series it produces.
    pub outputs: Vec<OutputWire>,
    /// Scientific checks it performs.
    pub checks: Vec<CheckWire>,
    /// Whether it can report genuine progress.
    pub supports_progress: bool,
    /// Whether a tested tutorial ships for it.
    pub has_tutorial: bool,
}

/// One parameter or initial-state field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldWire {
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

/// One output series a capability produces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputWire {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Unit.
    pub unit: String,
    /// Description.
    pub description: String,
}

/// One scientific check a capability performs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckWire {
    /// Stable id.
    pub id: String,
    /// What it checks.
    pub description: String,
}

/// A scenario opened into the editor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioWire {
    /// The source text.
    pub source: String,
    /// Where it came from, when it came from a file.
    pub path: Option<String>,
    /// The capability it targets, when known.
    pub capability_id: Option<String>,
}

/// How far along a run is.
///
/// Internally tagged on `state`, matching
/// `scirust_studio_app_service::JobState`. The tag name is part of the
/// contract and is asserted by the bridge contract test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobStateWire {
    /// Accepted, not yet started by the worker.
    Queued,
    /// Running, with genuine measurable progress.
    Running {
        /// Fraction of the time span integrated, `0.0..=1.0`.
        fraction: f64,
        /// The simulation time reached.
        t: f64,
    },
    /// Running with no fraction available.
    RunningIndeterminate,
    /// The user asked to cancel; the worker has not settled yet.
    Cancelling,
    /// Stopped because the user asked.
    Cancelled,
    /// Finished and stored.
    Completed {
        /// The stored run's id.
        run_id: String,
    },
    /// Integration failed numerically.
    FailedNumerical {
        /// The engine's explanation.
        message: String,
    },
    /// The scenario was rejected.
    FailedValidation {
        /// The explanation.
        message: String,
    },
    /// A bug in the worker or an adapter.
    FailedInternal {
        /// The explanation.
        message: String,
    },
    /// The worker died while this job was running.
    Interrupted {
        /// What is known about the exit.
        detail: String,
    },
}

/// Everything the interface needs about one job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobWire {
    /// This session's id for the job.
    pub job_id: String,
    /// The capability being run.
    pub capability_id: String,
    /// The experiment name from the scenario.
    pub scenario_name: String,
    /// Current state.
    pub state: JobStateWire,
    /// Whether this capability can report progress at all.
    pub supports_progress: bool,
    /// When the job started, RFC 3339 UTC.
    pub started_at_rfc3339: String,
    /// Elapsed wall-clock seconds.
    pub elapsed_seconds: f64,
    /// Warnings raised so far.
    pub warnings: Vec<WarningWire>,
    /// The run id this job was recorded under, once it has been recorded.
    pub run_id: Option<String>,
}

/// What a chart may claim its horizontal axis represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XAxisKindWire {
    /// Real coordinates from the integrator.
    PhysicalCoordinates,
    /// Sample ordinals only — a legacy result.
    SampleIndex,
}

/// What a plotted curve represents, mirroring `SeriesRoleView` in the shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesRoleWire {
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

impl SeriesRoleWire {
    /// Whether this curve is part of an ensemble's presentation.
    pub fn is_ensemble(self) -> bool {
        matches!(
            self,
            SeriesRoleWire::EnsembleMember
                | SeriesRoleWire::EnsembleMean
                | SeriesRoleWire::EnsembleBandLower
                | SeriesRoleWire::EnsembleBandUpper
        )
    }
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
pub struct GridWire {
    /// Stable id.
    pub id: String,
    /// Label.
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

/// A scenario file the user picked, as text.
///
/// No path, by design — the shell hands over contents and a display name and
/// nothing that could be used to reach the file again. See
/// `src-tauri/src/commands.rs`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioFileWire {
    /// The file's own name, for the title only.
    pub file_name: String,
    /// Its contents.
    pub source: String,
}

/// One plottable series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesWire {
    /// Stable id.
    pub id: String,
    /// Label for the legend.
    pub display_name: String,
    /// Unit.
    pub unit: String,
    /// What this curve represents. Defaulted, so a run stored before roles
    /// existed reads back as a trajectory — which is what it was.
    #[serde(default)]
    pub role: SeriesRoleWire,
    /// The values.
    pub values: Vec<f64>,
}

/// A stored run, loaded for display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunWire {
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
    pub x_axis_kind: XAxisKindWire,
    /// The axis label — never "time" for a legacy result.
    pub x_axis_label: String,
    /// The axis unit, empty for a sample index.
    pub x_axis_unit: String,
    /// The axis coordinates.
    pub x_values: Vec<f64>,
    /// The series.
    pub series: Vec<SeriesWire>,
    /// Fields, each spanning two axes. Empty for every capability whose
    /// outputs are curves.
    #[serde(default)]
    pub fields: Vec<GridWire>,
    /// Metrics.
    pub metrics: Vec<MetricWire>,
    /// Scientific checks.
    pub verifications: Vec<VerificationWire>,
    /// Warnings.
    pub warnings: Vec<WarningWire>,
    /// Provenance.
    pub provenance: ProvenanceWire,
    /// The scenario that produced it.
    pub scenario_source: String,
    /// Whether the stored bytes still match their hashes.
    pub integrity: IntegrityWire,
}

/// A metric, pre-rendered but keeping its raw value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricWire {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// The value, rendered.
    pub value: String,
    /// The numeric value, when it is one.
    pub numeric: Option<f64>,
    /// Unit, when it has one.
    pub unit: Option<String>,
}

/// One scientific check's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationWire {
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

/// A non-fatal warning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarningWire {
    /// Category.
    pub category: String,
    /// Message.
    pub message: String,
}

/// What produced a result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceWire {
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
    /// The seed the computation consumed, when it consumed one.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// The outcome of a store-integrity check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegrityWire {
    /// The run checked.
    pub run_id: String,
    /// Whether every stored hash still matches.
    pub intact: bool,
    /// What changed, when something did.
    pub detail: Option<String>,
}

/// A stored run in the history list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRunWire {
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

/// Something the application wants the interface to know about.
///
/// Internally tagged on `event`, matching
/// `scirust_studio_app_service::AppEvent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventWire {
    /// A worker process is up and has completed the handshake.
    WorkerStarted {
        /// The worker's crate version.
        worker_version: String,
        /// Where it was launched from.
        path: String,
    },
    /// The worker process is gone.
    WorkerExited {
        /// Why.
        class: ExitClassWire,
    },
    /// A job was accepted.
    JobStarted {
        /// The job.
        job_id: String,
        /// The capability being run.
        capability_id: String,
        /// Whether genuine progress will be reported for it.
        supports_progress: bool,
    },
    /// A job's state changed.
    JobState {
        /// The job.
        job_id: String,
        /// Its new state.
        state: JobStateWire,
    },
    /// A job raised a non-fatal warning.
    JobWarning {
        /// The job.
        job_id: String,
        /// The warning.
        warning: WarningWire,
    },
    /// Something worth showing that is not job- or worker-lifecycle.
    Diagnostic {
        /// Severity: `info`, `pass`, `warning` or `error`.
        level: String,
        /// The message.
        message: String,
    },
}

/// Why the worker process is gone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitClassWire {
    /// It was asked to shut down and did.
    Requested,
    /// It exited on its own while the application still expected it.
    Unexpected {
        /// The exit status, rendered.
        status: String,
    },
    /// The supervisor terminated it.
    Terminated,
}

/// A batch of events, plus how many were lost before it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventBatchWire {
    /// The events, oldest first.
    pub events: Vec<EventWire>,
    /// How many events were discarded because the buffer was full.
    pub dropped: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact JSON the shell emits for a determinate run. Written out by
    /// hand so that a change to the tag name or a field is caught here as
    /// well as in the cross-crate contract test.
    #[test]
    fn a_running_job_decodes_from_the_shell_s_json() {
        let json = r#"{
            "job_id": "job-1",
            "capability_id": "sim.mechanics.spring_mass_damper",
            "scenario_name": "Underdamped",
            "state": { "state": "running", "fraction": 0.25, "t": 2.5 },
            "supports_progress": true,
            "started_at_rfc3339": "2026-07-28T21:51:09Z",
            "elapsed_seconds": 0.5,
            "warnings": [],
            "run_id": null
        }"#;
        let job: JobWire = serde_json::from_str(json).expect("a job");
        assert_eq!(
            job.state,
            JobStateWire::Running {
                fraction: 0.25,
                t: 2.5
            }
        );
        assert!(job.supports_progress);
        assert_eq!(job.run_id, None);
    }

    /// A unit variant of an internally tagged enum is an object carrying
    /// only the tag — the shape most easily got wrong.
    #[test]
    fn a_unit_job_state_is_an_object_with_only_the_tag() {
        let value = serde_json::to_value(JobStateWire::RunningIndeterminate).unwrap();
        assert_eq!(
            value,
            serde_json::json!({ "state": "running_indeterminate" })
        );

        let decoded: JobStateWire =
            serde_json::from_value(serde_json::json!({ "state": "cancelled" })).unwrap();
        assert_eq!(decoded, JobStateWire::Cancelled);
    }

    #[test]
    fn an_event_batch_decodes_every_variant() {
        let json = r#"{
            "events": [
                { "event": "worker_started", "worker_version": "0.1.0", "path": "/opt/w" },
                { "event": "worker_exited", "class": "requested" },
                { "event": "worker_exited", "class": { "unexpected": { "status": "exit 1" } } },
                { "event": "job_started", "job_id": "j", "capability_id": "c",
                  "supports_progress": false },
                { "event": "job_state", "job_id": "j",
                  "state": { "state": "completed", "run_id": "r" } },
                { "event": "job_warning", "job_id": "j",
                  "warning": { "category": "numerical", "message": "m" } },
                { "event": "diagnostic", "level": "warning", "message": "m" }
            ],
            "dropped": 3
        }"#;
        let batch: EventBatchWire = serde_json::from_str(json).expect("a batch");
        assert_eq!(batch.events.len(), 7);
        assert_eq!(batch.dropped, 3);
        assert_eq!(
            batch.events[1],
            EventWire::WorkerExited {
                class: ExitClassWire::Requested
            }
        );
        assert_eq!(
            batch.events[2],
            EventWire::WorkerExited {
                class: ExitClassWire::Unexpected {
                    status: "exit 1".to_string()
                }
            }
        );
    }

    /// The bridge must not lose a bit of a scientific value. `f64` survives
    /// the JSON text form exactly, and that is worth asserting rather than
    /// assuming: a chart of a result that differs in its last bit is a chart
    /// of a different result.
    #[test]
    fn series_values_survive_the_json_round_trip_bit_for_bit() {
        let values = vec![
            0.1,
            1.0 / 3.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            -1.234_567_890_123_456_7e-17,
            6.999_999_999_999_999e-15,
        ];
        let series = SeriesWire {
            id: "x".to_string(),
            display_name: "Position".to_string(),
            unit: "m".to_string(),
            role: SeriesRoleWire::Trajectory,
            values: values.clone(),
        };
        let text = serde_json::to_string(&series).unwrap();
        let back: SeriesWire = serde_json::from_str(&text).unwrap();
        for (original, decoded) in values.iter().zip(back.values.iter())
        {
            assert_eq!(
                original.to_bits(),
                decoded.to_bits(),
                "{original} changed crossing the bridge"
            );
        }
    }

    #[test]
    fn a_legacy_run_declares_a_sample_index_axis() {
        let kind: XAxisKindWire = serde_json::from_str("\"sample_index\"").unwrap();
        assert_eq!(kind, XAxisKindWire::SampleIndex);
        let kind: XAxisKindWire = serde_json::from_str("\"physical_coordinates\"").unwrap();
        assert_eq!(kind, XAxisKindWire::PhysicalCoordinates);
    }
}
