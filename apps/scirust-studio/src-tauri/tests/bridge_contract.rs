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
    BootstrapView, CapabilityView, ErrorView, MetricView, RunView, StoredRunView,
    VerificationReportView, VerificationView, WarningView, XAxisKind,
};
use scirust_studio_runtime::{Metric, MetricValue, RunWarning, WarningCategory};
use scirust_studio_ui::backend::wire::{
    BootstrapWire, CapabilityWire, ErrorWire, EventBatchWire, JobStateWire, JobWire, RunWire,
    StoredRunWire, ValidationWire, XAxisKindWire,
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
        0.100_000_000_000_000_005,
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
            values: series_values.clone(),
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
