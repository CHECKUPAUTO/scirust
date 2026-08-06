//! The two halves of the bridge must agree.
//!
//! The shell serialises `crate::views::*`; the interface deserialises
//! `scirust_studio_ui::backend::wire::*`. They are separate declarations —
//! the interface compiles to WebAssembly and cannot link the registry, the
//! store or Tauri — so nothing in the type system stops one side from being
//! renamed while the other is not.
//!
//! This is what stops it. Every test here takes a **real** value produced by
//! the shell's own code paths, serialises it exactly as Tauri's IPC does,
//! and deserialises it into the type the interface actually uses. A renamed
//! field, a changed enum tag or a dropped variant fails here, at build time,
//! rather than showing up as a mysteriously empty panel in a running
//! application.

use scirust_studio_app_service::{
    AppServiceError, JobSnapshot, JobState, ValidationOutcome, validate_source,
};
use scirust_studio_desktop_lib::views::{
    BootstrapView, CapabilityView, DistributionView, ErrorView, GridView, MetricView, RunView,
    SeriesRoleView, StoredRunView, VerificationReportView, VerificationView, WarningView,
    XAxisKind,
};
use scirust_studio_runtime::{Metric, MetricValue, RunWarning, WarningCategory};
use scirust_studio_ui::backend::wire::{
    BootstrapWire, CapabilityWire, ErrorWire, EventBatchWire, JobStateWire, JobWire, RunWire,
    SeriesRoleWire, StoredRunWire, ValidationWire, XAxisKindWire,
};

/// Serialise as the IPC does, then decode as the interface does.
fn cross<T: serde::Serialize, U: serde::de::DeserializeOwned>(value: &T) -> U {
    let text = serde_json::to_string(value).expect("the shell can serialise its own view");
    match serde_json::from_str(&text)
    {
        Ok(decoded) => decoded,
        Err(e) => panic!("the interface cannot decode what the shell sends: {e}\n{text}"),
    }
}

#[test]
fn every_catalogue_entry_crosses_intact() {
    let registry = scirust_studio_runtime::build_registry();
    assert!(
        registry.iter().count() >= 5,
        "the contract is only proved if there is a real catalogue to cross"
    );

    for descriptor in registry.iter()
    {
        let view = CapabilityView::from(descriptor);
        let wire: CapabilityWire = cross(&view);

        assert_eq!(wire.id, view.id);
        assert_eq!(wire.display_name, view.display_name);
        assert_eq!(wire.category, view.category);
        assert_eq!(wire.source_crate, view.source_crate);
        assert_eq!(wire.summary, view.summary);
        assert_eq!(wire.maturity, view.maturity);
        assert_eq!(wire.determinism, view.determinism);
        assert_eq!(wire.solvers, view.solvers);
        assert_eq!(wire.supports_progress, view.supports_progress);
        assert_eq!(wire.has_tutorial, view.has_tutorial);
        assert_eq!(wire.parameters.len(), view.parameters.len());
        assert_eq!(wire.initial_state.len(), view.initial_state.len());
        assert_eq!(wire.outputs.len(), view.outputs.len());
        assert_eq!(wire.checks.len(), view.checks.len());

        // The catalogue is useless if a field arrives empty, so the crossing
        // is checked to carry content and not merely to parse.
        for (from, to) in view.parameters.iter().zip(wire.parameters.iter())
        {
            assert_eq!(from.name, to.name);
            assert_eq!(from.units, to.units);
            assert!(!to.description.is_empty(), "{}", to.name);
        }
    }
}

/// Every job state, including the ones a UI must not confuse.
#[test]
fn every_job_state_crosses_with_its_tag() {
    let states = [
        JobState::Queued,
        JobState::Running {
            fraction: 0.375,
            t: 1.25,
        },
        JobState::RunningIndeterminate,
        JobState::Cancelling,
        JobState::Cancelled,
        JobState::Completed {
            run_id: "20260728T215109Z-ef820514594aa333".to_string(),
        },
        JobState::FailedNumerical {
            message: "the step size underflowed".to_string(),
        },
        JobState::FailedValidation {
            message: "unknown field".to_string(),
        },
        JobState::FailedInternal {
            message: "adapter panicked".to_string(),
        },
        JobState::Interrupted {
            detail: "the worker exited with status 1".to_string(),
        },
    ];

    for state in states
    {
        let snapshot = JobSnapshot {
            job_id: "job-1".to_string(),
            capability_id: "sim.mechanics.spring_mass_damper".to_string(),
            scenario_name: "Underdamped".to_string(),
            state: state.clone(),
            supports_progress: true,
            started_at_rfc3339: "2026-07-28T21:51:09Z".to_string(),
            elapsed_seconds: 0.25,
            warnings: vec![RunWarning {
                category: WarningCategory::Numerical,
                message: "energy drift approached its threshold".to_string(),
            }],
            run_id: None,
        };
        let wire: JobWire = cross(&snapshot);

        let expected = match &state
        {
            JobState::Queued => JobStateWire::Queued,
            JobState::Running { fraction, t } => JobStateWire::Running {
                fraction: *fraction,
                t: *t,
            },
            JobState::RunningIndeterminate => JobStateWire::RunningIndeterminate,
            JobState::Cancelling => JobStateWire::Cancelling,
            JobState::Cancelled => JobStateWire::Cancelled,
            JobState::Completed { run_id } => JobStateWire::Completed {
                run_id: run_id.clone(),
            },
            JobState::FailedNumerical { message } => JobStateWire::FailedNumerical {
                message: message.clone(),
            },
            JobState::FailedValidation { message } => JobStateWire::FailedValidation {
                message: message.clone(),
            },
            JobState::FailedInternal { message } => JobStateWire::FailedInternal {
                message: message.clone(),
            },
            JobState::Interrupted { detail } => JobStateWire::Interrupted {
                detail: detail.clone(),
            },
        };
        assert_eq!(wire.state, expected, "{state:?} did not cross intact");
        assert_eq!(wire.warnings.len(), 1);
        assert_eq!(wire.warnings[0].category, "numerical");
    }
}

/// Cancelled and interrupted are different facts and must stay different
/// across the bridge, because the interface presents them differently and a
/// user acts on them differently.
#[test]
fn cancelled_and_interrupted_do_not_collapse_into_each_other() {
    let cancelled: JobStateWire = cross(&JobState::Cancelled);
    let interrupted: JobStateWire = cross(&JobState::Interrupted {
        detail: "the worker exited".to_string(),
    });
    assert_ne!(cancelled, interrupted);
    assert!(matches!(cancelled, JobStateWire::Cancelled));
    assert!(matches!(interrupted, JobStateWire::Interrupted { .. }));
}

#[test]
fn a_validation_failure_crosses_with_its_problems_and_locations() {
    let outcome: ValidationOutcome = validate_source(
        "schema_version = 1\n\
         [experiment]\n\
         name = \"x\"\n\
         capability = \"sim.mechanics.spring_mass_damper\"\n\
         [parameters]\n\
         mas = 1.0\n",
    );
    assert!(!outcome.valid, "the fixture must actually fail");

    let view = ErrorView::from(AppServiceError::Validation(Box::new(outcome.clone())));
    let wire: ErrorWire = cross(&view);

    let problems = wire.validation.expect("the problems must cross").problems;
    assert_eq!(problems.len(), outcome.problems.len());
    for (from, to) in outcome.problems.iter().zip(problems.iter())
    {
        assert_eq!(from.title, to.title);
        assert_eq!(from.explanation, to.explanation);
        assert_eq!(from.field, to.field);
        assert_eq!(from.location.map(|l| l.line), to.location.map(|l| l.line));
        assert_eq!(
            format!("{:?}", from.stage).to_lowercase(),
            to.stage,
            "the stage tag must cross as its snake_case name"
        );
    }
}

/// A `ValidationOutcome` also crosses on its own, which is the shape
/// `studio_validate_scenario` returns.
#[test]
fn a_clean_validation_outcome_crosses() {
    let source = scirust_studio_runtime::tutorial_scenario_for("sim.mechanics.spring_mass_damper")
        .expect("the tutorial ships");
    let outcome = validate_source(source);
    assert!(outcome.valid, "the shipped tutorial must validate");

    let wire: ValidationWire = cross(&outcome);
    assert!(wire.valid);
    assert_eq!(wire.capability_id, outcome.capability_id);
    assert_eq!(wire.scenario_name, outcome.scenario_name);
    assert!(wire.problems.is_empty());
}

#[test]
fn the_bootstrap_view_crosses() {
    let view = BootstrapView {
        app_version: "0.1.0".to_string(),
        worker_version: Some("0.1.0".to_string()),
        worker_running: true,
        store_path: "/home/user/.local/share/Memorithm/SciRust Studio/runs".to_string(),
        capability_count: 5,
        interrupted_runs: vec!["20260728T215109Z-ef820514594aa333".to_string()],
        worker_problem: None,
    };
    let wire: BootstrapWire = cross(&view);
    assert_eq!(wire.store_path, view.store_path);
    assert_eq!(wire.capability_count, 5);
    assert_eq!(wire.interrupted_runs, view.interrupted_runs);
    assert!(wire.worker_problem.is_none());
}

#[test]
fn an_error_view_crosses_with_everything_a_card_needs() {
    let view = ErrorView::from(AppServiceError::Busy {
        active_job_id: "job-1".to_string(),
    });
    let wire: ErrorWire = cross(&view);
    assert_eq!(wire.code, view.code);
    assert_eq!(wire.title, view.title);
    assert_eq!(wire.explanation, view.explanation);
    assert_eq!(wire.recoverable, view.recoverable);
    assert_eq!(wire.suggested_action, view.suggested_action);
}

/// The whole point of schema v2: a chart must receive the coordinates the
/// integrator produced, unchanged, and a legacy result must arrive labelled
/// as sample ordinals.
#[test]
fn a_run_view_crosses_with_its_coordinates_bit_for_bit() {
    let x_values = vec![
        0.0,
        1.0 / 3.0,
        // The next representable f64 above 1.0: a value that survives only
        // if the crossing is exact.
        1.000_000_000_000_000_2,
        1.234_567_890_123_456_7e-9,
        9.999_999_999_999_998e5,
    ];
    let series_values = vec![
        1.0,
        -0.333_333_333_333_333_3,
        6.999e-15,
        f64::MIN_POSITIVE,
        0.0,
    ];

    let view = RunView {
        run_id: "20260728T215109Z-ef820514594aa333".to_string(),
        capability_id: "sim.mechanics.spring_mass_damper".to_string(),
        scenario_name: "Underdamped".to_string(),
        status: "completed".to_string(),
        result_schema_version: 2,
        x_axis_kind: XAxisKind::PhysicalCoordinates,
        x_axis_label: "time".to_string(),
        x_axis_unit: "s".to_string(),
        x_values: x_values.clone(),
        series: vec![scirust_studio_desktop_lib::views::SeriesView {
            id: "x".to_string(),
            display_name: "Displacement".to_string(),
            unit: "m".to_string(),
            role: SeriesRoleView::Trajectory,
            values: series_values.clone(),
        }],
        fields: vec![GridView {
            id: "temperature".to_string(),
            display_name: "Temperature".to_string(),
            unit: "K".to_string(),
            row_axis_id: "t".to_string(),
            column_axis_id: "x".to_string(),
            columns: 2,
            values: vec![1.0, 2.0, 3.0, 4.0],
        }],
        // Edge counts, under- and overflow all cross the bridge: a histogram
        // whose range was chosen badly must be able to say so in the
        // interface, not only on the command line.
        distributions: vec![DistributionView {
            id: "queue_length".to_string(),
            display_name: "Time-average number in system".to_string(),
            unit: "1".to_string(),
            edges: vec![0.0, 1.0, 2.0, 3.0],
            counts: vec![2, 5, 1],
            underflow: 1,
            overflow: 3,
        }],
        metrics: vec![MetricView::from(&Metric {
            id: "energy_drift".to_string(),
            display_name: "Energy drift".to_string(),
            value: MetricValue::Scalar(6.99e-15),
            unit: Some("J".to_string()),
        })],
        verifications: vec![VerificationView {
            id: "energy_drift".to_string(),
            status: "passed".to_string(),
            measured: Some(6.99e-15),
            threshold: Some(1e-9),
            explanation: "within threshold".to_string(),
        }],
        warnings: vec![WarningView {
            category: "numerical".to_string(),
            message: "none".to_string(),
        }],
        provenance: scirust_studio_desktop_lib::views::ProvenanceView {
            capability_id: "sim.mechanics.spring_mass_damper".to_string(),
            determinism: "BitExact".to_string(),
            adapter_crate: "scirust-studio-runtime".to_string(),
            adapter_version: "0.1.0".to_string(),
            result_schema_version: 2,
            target: "linux-x86_64".to_string(),
            started_at: "2026-07-28T21:51:09Z".to_string(),
            completed_at: "2026-07-28T21:51:10Z".to_string(),
            elapsed_seconds: 1.5,
            seed: None,
        },
        scenario_source: "schema_version = 1\n".to_string(),
        integrity: VerificationReportView {
            run_id: "20260728T215109Z-ef820514594aa333".to_string(),
            intact: true,
            detail: None,
        },
    };

    let wire: RunWire = cross(&view);
    assert_eq!(wire.x_axis_kind, XAxisKindWire::PhysicalCoordinates);
    assert_eq!(wire.x_axis_label, "time");
    assert_eq!(wire.x_axis_unit, "s");
    assert_eq!(wire.result_schema_version, 2);

    for (original, decoded) in x_values.iter().zip(wire.x_values.iter())
    {
        assert_eq!(
            original.to_bits(),
            decoded.to_bits(),
            "coordinate {original} changed crossing the bridge"
        );
    }
    for (original, decoded) in series_values.iter().zip(wire.series[0].values.iter())
    {
        assert_eq!(
            original.to_bits(),
            decoded.to_bits(),
            "value {original} changed crossing the bridge"
        );
    }

    assert_eq!(wire.metrics[0].numeric, Some(6.99e-15));
    assert_eq!(wire.verifications[0].status, "passed");
    assert!(wire.integrity.intact);
    assert_eq!(wire.series[0].role, SeriesRoleWire::Trajectory);

    // The field is the whole result for a capability like the heat rod, so
    // it has to survive the crossing shape-intact — a transposed or
    // truncated grid still renders as a convincing picture.
    assert_eq!(wire.fields.len(), 1);
    assert_eq!(wire.fields[0].columns, 2);
    assert_eq!(wire.fields[0].values, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(wire.fields[0].row_axis_id, "t");
    assert_eq!(wire.fields[0].column_axis_id, "x");
}

/// A stochastic run's seed must reach the interface: it is the only thing
/// that makes the result reproducible, and a provenance panel that omits it
/// is showing a run nobody can repeat.
///
/// The `None` case matters just as much. A deterministic capability consumed
/// no seed, and a number displayed there would claim it had.
#[test]
fn a_consumed_seed_crosses_and_an_unconsumed_one_stays_absent() {
    use scirust_studio_desktop_lib::views::ProvenanceView;
    use scirust_studio_ui::backend::wire::ProvenanceWire;

    let base = ProvenanceView {
        capability_id: "sim.stochastic.ornstein_uhlenbeck".to_string(),
        determinism: "InherentlyStochasticRecordedSeed".to_string(),
        adapter_crate: "scirust-studio-runtime".to_string(),
        adapter_version: "0.1.0".to_string(),
        result_schema_version: 2,
        target: "linux-x86_64".to_string(),
        started_at: "2026-07-29T00:00:00Z".to_string(),
        completed_at: "2026-07-29T00:00:01Z".to_string(),
        elapsed_seconds: 1.0,
        seed: Some(u64::MAX),
    };
    // u64::MAX on purpose: a seed routed through a JSON number that some
    // reader turns into an f64 would come back as 18446744073709552000.
    let wire: ProvenanceWire = cross(&base);
    assert_eq!(wire.seed, Some(u64::MAX));

    let unseeded = ProvenanceView { seed: None, ..base };
    let wire: ProvenanceWire = cross(&unseeded);
    assert_eq!(wire.seed, None);
}

/// And a provenance written before the field existed must still decode, since
/// the field was added to schema v2 rather than bumping to v3.
#[test]
fn a_provenance_without_a_seed_field_still_decodes() {
    use scirust_studio_ui::backend::wire::ProvenanceWire;

    let json = r#"{
        "capability_id": "sim.mechanics.spring_mass_damper",
        "determinism": "StrictSameBinarySameTarget",
        "adapter_crate": "scirust-studio-runtime",
        "adapter_version": "0.1.0",
        "result_schema_version": 2,
        "target": "linux-x86_64",
        "started_at": "2026-07-28T21:51:09Z",
        "completed_at": "2026-07-28T21:51:10Z",
        "elapsed_seconds": 1.5
    }"#;
    let wire: ProvenanceWire = serde_json::from_str(json).expect("an older provenance");
    assert_eq!(wire.seed, None);
}

#[test]
fn a_legacy_run_crosses_as_sample_indices() {
    let kind: XAxisKindWire = cross(&XAxisKind::SampleIndex);
    assert_eq!(kind, XAxisKindWire::SampleIndex);
}

#[test]
fn a_stored_run_summary_crosses() {
    let view = StoredRunView {
        run_id: "20260728T215109Z-ef820514594aa333".to_string(),
        capability_id: "sim.chemistry.robertson".to_string(),
        status: "completed".to_string(),
        started_at: "2026-07-28T21:51:09Z".to_string(),
        finished_at: "2026-07-28T21:51:10Z".to_string(),
        result_schema_version: 1,
    };
    let wire: StoredRunWire = cross(&view);
    assert_eq!(wire.run_id, view.run_id);
    assert_eq!(wire.result_schema_version, 1);
}

/// The event batch is the interface's only source of worker-lifecycle news;
/// every variant has to decode or the activity log silently loses lines.
#[test]
fn every_application_event_crosses() {
    use scirust_studio_app_service::{AppEvent, DiagnosticLevel, EventBatch, WorkerExitClass};
    use scirust_studio_ui::backend::wire::{EventWire, ExitClassWire};

    let batch = EventBatch {
        events: vec![
            AppEvent::WorkerStarted {
                worker_version: "0.1.0".to_string(),
                path: "/opt/worker".to_string(),
            },
            AppEvent::WorkerExited {
                class: WorkerExitClass::Requested,
            },
            AppEvent::WorkerExited {
                class: WorkerExitClass::Unexpected {
                    status: "exit status: 1".to_string(),
                },
            },
            AppEvent::WorkerExited {
                class: WorkerExitClass::Terminated,
            },
            AppEvent::JobStarted {
                job_id: "job-1".to_string(),
                capability_id: "sim.chemistry.robertson".to_string(),
                supports_progress: false,
            },
            AppEvent::JobState {
                job_id: "job-1".to_string(),
                state: JobState::RunningIndeterminate,
            },
            AppEvent::JobWarning {
                job_id: "job-1".to_string(),
                warning: RunWarning {
                    category: WarningCategory::Convergence,
                    message: "the solver reduced its step".to_string(),
                },
            },
            AppEvent::Diagnostic {
                level: DiagnosticLevel::Pass,
                message: "integrity verified".to_string(),
            },
        ],
        dropped: 7,
    };

    let wire: EventBatchWire = cross(&batch);
    assert_eq!(wire.events.len(), batch.events.len());
    assert_eq!(wire.dropped, 7);
    assert_eq!(
        wire.events[1],
        EventWire::WorkerExited {
            class: ExitClassWire::Requested
        }
    );
    assert_eq!(
        wire.events[2],
        EventWire::WorkerExited {
            class: ExitClassWire::Unexpected {
                status: "exit status: 1".to_string()
            }
        }
    );
    assert!(matches!(
        wire.events[3],
        EventWire::WorkerExited {
            class: ExitClassWire::Terminated
        }
    ));
    assert_eq!(
        wire.events[5],
        EventWire::JobState {
            job_id: "job-1".to_string(),
            state: JobStateWire::RunningIndeterminate
        }
    );
    match &wire.events[7]
    {
        EventWire::Diagnostic { level, message } =>
        {
            assert_eq!(level, "pass");
            assert_eq!(message, "integrity verified");
        },
        other => panic!("the diagnostic did not cross: {other:?}"),
    }
}

/// The interface's error type must be constructible from what the shell
/// rejects with, without losing the parts an error card renders.
#[test]
fn a_rejection_becomes_a_complete_frontend_error() {
    use scirust_studio_ui::backend::FrontendError;

    let view = ErrorView::from(AppServiceError::WorkerUnavailable {
        reason: "no worker binary beside the application".to_string(),
    });
    let wire: ErrorWire = cross(&view);
    let error = FrontendError::from_wire(wire);

    assert_eq!(error.code, view.code);
    assert_eq!(error.title, view.title);
    assert!(!error.explanation.is_empty());
    assert_eq!(error.recoverable, view.recoverable);
    assert_eq!(error.suggested_action, view.suggested_action);
}

/// Every series role must survive the crossing, because the interface
/// *branches* on it: a role that decoded to the default would draw a summary
/// statistic and one noisy realisation with the same weight, which is a
/// misleading picture rather than a cosmetic one.
#[test]
fn every_series_role_crosses_the_bridge_intact() {
    let pairs = [
        (SeriesRoleView::Trajectory, SeriesRoleWire::Trajectory),
        (SeriesRoleView::Reference, SeriesRoleWire::Reference),
        (
            SeriesRoleView::EnsembleMember,
            SeriesRoleWire::EnsembleMember,
        ),
        (SeriesRoleView::EnsembleMean, SeriesRoleWire::EnsembleMean),
        (
            SeriesRoleView::EnsembleBandLower,
            SeriesRoleWire::EnsembleBandLower,
        ),
        (
            SeriesRoleView::EnsembleBandUpper,
            SeriesRoleWire::EnsembleBandUpper,
        ),
    ];
    for (view, expected) in pairs
    {
        let decoded: SeriesRoleWire = cross(&view);
        assert_eq!(
            decoded, expected,
            "role {view:?} did not cross the bridge as {expected:?}"
        );
    }
}

/// A run recorded before roles existed has no `role` key at all. It must
/// decode as a trajectory rather than failing, because that is what those
/// series were — the store holds results going back to schema v1 and they
/// are meant to stay readable.
#[test]
fn a_series_stored_before_roles_existed_decodes_as_a_trajectory() {
    let legacy = serde_json::json!({
        "id": "x",
        "display_name": "Displacement",
        "unit": "m",
        "values": [1.0, 0.5]
    });
    let decoded: scirust_studio_ui::backend::wire::SeriesWire =
        serde_json::from_value(legacy).expect("a series without a role still decodes");
    assert_eq!(decoded.role, SeriesRoleWire::Trajectory);
    assert!(!decoded.role.is_ensemble());
}

/// The two enums are mirrors, so a role added to one and forgotten on the
/// other must not compile away quietly. `cross` goes through JSON, so this
/// fails loudly on a name that does not match rather than on a count.
#[test]
fn the_shell_and_the_interface_agree_on_the_ensemble_roles() {
    let ensemble = [
        SeriesRoleView::EnsembleMember,
        SeriesRoleView::EnsembleMean,
        SeriesRoleView::EnsembleBandLower,
        SeriesRoleView::EnsembleBandUpper,
    ];
    for role in ensemble
    {
        let decoded: SeriesRoleWire = cross(&role);
        assert!(
            decoded.is_ensemble(),
            "{role:?} crossed as {decoded:?}, which the interface does not treat as an ensemble \
             curve"
        );
    }
    let decoded: SeriesRoleWire = cross(&SeriesRoleView::Reference);
    assert!(!decoded.is_ensemble());
}
