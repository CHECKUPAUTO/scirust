//! Application-level supervision, tested against the **real** worker binary.
//!
//! Every test here launches the compiled `scirust-studio-worker` as a genuine
//! child process. Nothing is stubbed: the handshake, the job lifecycle, the
//! cancellation paths, the store commit and the interruption handling are all
//! exercised end to end, because those are exactly the behaviours that only
//! exist once there is a second process to lose.
//!
//! The worker is located beside the test executable in Cargo's target
//! directory. It is not a path dependency because it is a binary-only crate,
//! so the gate command builds it first:
//!
//! ```text
//! cargo build -p scirust-studio-worker
//! cargo test  -p scirust-studio-app-service
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use scirust_studio_app_service::{
    AppEvent, AppServiceConfig, AppServiceError, JobState, StudioAppService, WorkerExitClass,
    WorkerLaunchConfig,
};

const SPRING_MASS: &str =
    include_str!("../../docs/studio/tutorials/spring_mass_damper.scirust.toml");
const SIR: &str = include_str!("../../docs/studio/tutorials/sir_epidemic.scirust.toml");
const ROBERTSON: &str = include_str!("../../docs/studio/tutorials/robertson_stiff.scirust.toml");

/// The compiled worker, beside this test executable.
fn worker_path() -> PathBuf {
    let mut dir = std::env::current_exe().expect("the test executable has a path");
    dir.pop(); // the test binary's own name
    if dir.ends_with("deps")
    {
        dir.pop();
    }
    let path = dir.join(WorkerLaunchConfig::executable_name());
    assert!(
        path.is_file(),
        "the worker binary is missing at {}.\n\
         Build it first:  cargo build -p scirust-studio-worker",
        path.display()
    );
    path
}

fn service_with_store() -> (tempfile::TempDir, StudioAppService) {
    let dir = tempfile::tempdir().expect("temp dir");
    let config = AppServiceConfig {
        worker: WorkerLaunchConfig::at(worker_path()),
        store_root: dir.path().to_path_buf(),
        event_buffer_capacity: 4096,
    };
    let service = StudioAppService::bootstrap(config).expect("bootstraps");
    (dir, service)
}

/// A scenario long enough to be cancelled, but inside `scirust-sim`'s
/// 10 000 000-step budget — past that the engine rejects the run outright
/// and there would be nothing running to cancel.
fn long_running() -> String {
    SIR.replace(
        r#"end = { value = 60.0, unit = "s" }"#,
        r#"end = { value = 4000.0, unit = "s" }"#,
    )
    .replace(
        r#"step = { value = 0.05, unit = "s" }"#,
        r#"step = { value = 0.001, unit = "s" }"#,
    )
}

/// Poll until `predicate` holds, or fail with the last state seen.
///
/// Polling rather than sleeping a fixed amount: the assertion is on the
/// outcome, and a slow machine gets more time instead of a flaky failure.
fn wait_for(
    service: &StudioAppService,
    job_id: &str,
    what: &str,
    predicate: impl Fn(&JobState) -> bool,
) -> JobState {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut last = None;
    while Instant::now() < deadline
    {
        let state = service.job_snapshot(job_id).expect("job exists").state;
        if predicate(&state)
        {
            return state;
        }
        last = Some(state);
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {what}; last state was {last:?}");
}

fn wait_for_terminal(service: &StudioAppService, job_id: &str) -> JobState {
    wait_for(service, job_id, "a terminal state", |s| s.is_terminal())
}

#[test]
fn bootstrap_starts_a_worker_and_reports_its_version() {
    let (_dir, service) = service_with_store();
    assert!(service.worker_is_running());
    let version = service.worker_version().expect("a version");
    assert!(!version.is_empty(), "the handshake must report a version");

    let events = service.drain_events();
    assert!(
        events
            .events
            .iter()
            .any(|e| matches!(e, AppEvent::WorkerStarted { .. })),
        "{:?}",
        events.events
    );
    service.shutdown().expect("shuts down");
}

#[test]
fn a_missing_worker_is_reported_not_panicked() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppServiceConfig {
        worker: WorkerLaunchConfig::at("/definitely/not/a/worker"),
        store_root: dir.path().to_path_buf(),
        event_buffer_capacity: 16,
    };
    match StudioAppService::bootstrap(config)
    {
        Err(AppServiceError::WorkerNotFound { searched }) => assert_eq!(searched.len(), 1),
        Err(other) => panic!("expected WorkerNotFound, got {other:?}"),
        Ok(_) => panic!("bootstrapping without a worker must not succeed"),
    }
}

/// The whole point of the layer: a scenario goes in, a verified stored run
/// comes out, with the real worker doing the arithmetic.
#[test]
fn a_run_completes_and_is_committed_to_the_store() {
    let (_dir, service) = service_with_store();

    let job = service.start_run(SPRING_MASS).expect("starts");
    let state = wait_for_terminal(&service, &job.job_id);

    let run_id = match state
    {
        JobState::Completed { run_id } => run_id,
        other => panic!("expected Completed, got {other:?}"),
    };

    let runs = service.list_runs().expect("lists");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].run_id, run_id);
    assert_eq!(runs[0].status, "completed");
    assert_eq!(runs[0].result_schema_version, 2);

    let report = service.verify_run(&run_id).expect("verifies");
    assert!(report.intact, "{report:?}");

    let loaded = service.load_run(&run_id).expect("loads");
    let result = loaded.result.as_v2().expect("a fresh run is v2");
    let axis = result.time_axis().expect("time axis");
    assert!(!axis.values.is_empty(), "the axis carries its coordinates");
    assert_eq!(result.summary.t_start, axis.values[0]);
    assert!(!loaded.scenario_source.is_empty());

    service.shutdown().expect("shuts down");
}

#[test]
fn a_fixed_step_run_reports_genuine_progress() {
    let (_dir, service) = service_with_store();

    let job = service.start_run(&long_running()).expect("starts");
    assert!(
        job.supports_progress,
        "SIR integrates with a fixed step and can report progress"
    );

    let state = wait_for(&service, &job.job_id, "measurable progress", |s| {
        matches!(s, JobState::Running { .. })
    });
    match state
    {
        JobState::Running { fraction, .. } =>
        {
            assert!((0.0..=1.0).contains(&fraction), "{fraction}");
        },
        other => panic!("{other:?}"),
    }

    service.cancel_run(&job.job_id).expect("cancels");
    wait_for_terminal(&service, &job.job_id);
    service.shutdown().expect("shuts down");
}

/// Robertson's adaptive stiff solver has no per-step callback, so it must be
/// shown as indeterminate — never as a fabricated percentage.
#[test]
fn the_adaptive_capability_runs_indeterminate() {
    let (_dir, service) = service_with_store();

    let job = service.start_run(ROBERTSON).expect("starts");
    assert!(
        !job.supports_progress,
        "the adaptive stiff path cannot report intermediate progress"
    );
    assert_eq!(job.state, JobState::RunningIndeterminate);
    assert_eq!(
        job.state.fraction(),
        None,
        "an indeterminate run must not offer a number to draw"
    );

    let state = wait_for_terminal(&service, &job.job_id);
    assert!(matches!(state, JobState::Completed { .. }), "got {state:?}");
    service.shutdown().expect("shuts down");
}

#[test]
fn cancelling_a_fixed_step_run_is_cooperative_and_reports_cancelled() {
    let (_dir, service) = service_with_store();

    let job = service.start_run(&long_running()).expect("starts");
    wait_for(&service, &job.job_id, "the run to start", |s| {
        matches!(s, JobState::Running { .. })
    });

    service.cancel_run(&job.job_id).expect("cancels");
    let state = wait_for_terminal(&service, &job.job_id);
    assert_eq!(state, JobState::Cancelled, "got {state:?}");

    // Cooperative cancellation must leave the worker alive and usable.
    assert!(
        service.worker_is_running(),
        "a cooperative cancellation must not kill the engine"
    );
    let next = service
        .start_run(SPRING_MASS)
        .expect("the service is usable");
    wait_for_terminal(&service, &next.job_id);

    service.shutdown().expect("shuts down");
}

/// A cancelled run is recorded as cancelled, not as a failure: the user
/// asked for it.
#[test]
fn a_cancelled_run_is_stored_as_cancelled() {
    let (_dir, service) = service_with_store();

    let job = service.start_run(&long_running()).expect("starts");
    wait_for(&service, &job.job_id, "the run to start", |s| {
        matches!(s, JobState::Running { .. })
    });
    service.cancel_run(&job.job_id).expect("cancels");
    wait_for_terminal(&service, &job.job_id);

    let runs = service.list_runs().expect("lists");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].status, "cancelled");
    assert!(
        service
            .verify_run(&runs[0].run_id)
            .expect("verifies")
            .intact,
        "a cancelled run still verifies"
    );
    service.shutdown().expect("shuts down");
}

/// The application runs one scientific job at a time.
#[test]
fn a_second_run_is_refused_while_one_is_active() {
    let (_dir, service) = service_with_store();

    let first = service.start_run(&long_running()).expect("starts");
    match service.start_run(SPRING_MASS)
    {
        Err(AppServiceError::Busy { active_job_id }) =>
        {
            assert_eq!(active_job_id, first.job_id);
        },
        other => panic!("expected Busy, got {other:?}"),
    }

    service.cancel_run(&first.job_id).expect("cancels");
    wait_for_terminal(&service, &first.job_id);
    service.shutdown().expect("shuts down");
}

/// The rule that matters most after a crash: the lost calculation is
/// reported as interrupted and is **not** silently re-run.
#[test]
fn an_unexpected_worker_exit_interrupts_the_job_without_rerunning_it() {
    let (dir, service) = service_with_store();

    let job = service.start_run(&long_running()).expect("starts");
    wait_for(&service, &job.job_id, "the run to start", |s| {
        matches!(s, JobState::Running { .. })
    });

    // Kill the worker out from under the application, which is what a crash
    // looks like from here.
    kill_worker_process(&service);

    let state = wait_for_terminal(&service, &job.job_id);
    assert!(
        matches!(state, JobState::Interrupted { .. }),
        "expected Interrupted, got {state:?}"
    );

    // Give any (incorrect) automatic rerun a chance to happen.
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        matches!(
            service.job_snapshot(&job.job_id).unwrap().state,
            JobState::Interrupted { .. }
        ),
        "an interrupted job must stay interrupted"
    );
    assert!(
        service.list_runs().expect("lists").is_empty(),
        "an interrupted run must not be finalized into the store"
    );

    // The evidence is preserved instead.
    let interrupted = service.interrupted_runs().expect("lists interrupted");
    assert_eq!(
        interrupted.len(),
        1,
        "the partial run directory is the record that this was attempted"
    );
    assert!(dir.path().join("runs").is_dir());

    let events = service.drain_events();
    assert!(
        events.events.iter().any(|e| matches!(
            e,
            AppEvent::WorkerExited {
                class: WorkerExitClass::Unexpected { .. }
            }
        )),
        "the exit must be classified as unexpected: {:?}",
        events.events
    );
}

/// After a crash the application must become usable again — but only for
/// new work.
#[test]
fn the_worker_can_be_restarted_for_a_future_job() {
    let (_dir, service) = service_with_store();

    let lost = service.start_run(&long_running()).expect("starts");
    wait_for(&service, &lost.job_id, "the run to start", |s| {
        matches!(s, JobState::Running { .. })
    });
    kill_worker_process(&service);
    wait_for_terminal(&service, &lost.job_id);
    assert!(!service.worker_is_running());

    // A run cannot be started without an engine...
    assert!(matches!(
        service.start_run(SPRING_MASS),
        Err(AppServiceError::WorkerUnavailable { .. })
    ));

    // ...until one is restarted, and then only the new job runs.
    service.start_worker().expect("restarts");
    assert!(service.worker_is_running());

    let fresh = service.start_run(SPRING_MASS).expect("starts");
    let state = wait_for_terminal(&service, &fresh.job_id);
    assert!(matches!(state, JobState::Completed { .. }), "{state:?}");

    assert!(
        matches!(
            service.job_snapshot(&lost.job_id).unwrap().state,
            JobState::Interrupted { .. }
        ),
        "restarting the engine must not resurrect the lost job"
    );
    // Exactly one finalized run: the new one.
    assert_eq!(service.list_runs().expect("lists").len(), 1);

    service.shutdown().expect("shuts down");
}

#[test]
fn an_invalid_scenario_is_refused_before_the_worker_is_involved() {
    let (_dir, service) = service_with_store();

    let broken = SIR.replace(r#"beta = { value = 0.6, unit = "1/s" }"#, "");
    match service.start_run(&broken)
    {
        Err(AppServiceError::Validation(outcome)) =>
        {
            assert!(!outcome.valid);
            assert!(!outcome.problems.is_empty());
        },
        other => panic!("expected Validation, got {other:?}"),
    }

    assert!(
        service.list_runs().expect("lists").is_empty(),
        "a scenario that never ran must not create a stored run"
    );
    assert!(
        service.active_job().is_none(),
        "a refused scenario must not occupy the active slot"
    );
    service.shutdown().expect("shuts down");
}

#[test]
fn events_are_ordered_and_the_buffer_is_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let service = StudioAppService::bootstrap(AppServiceConfig {
        worker: WorkerLaunchConfig::at(worker_path()),
        store_root: dir.path().to_path_buf(),
        // Deliberately tiny, so a real run overflows it.
        event_buffer_capacity: 4,
    })
    .expect("bootstraps");

    let job = service.start_run(&long_running()).expect("starts");
    wait_for(&service, &job.job_id, "the run to start", |s| {
        matches!(s, JobState::Running { .. })
    });
    service.cancel_run(&job.job_id).expect("cancels");
    wait_for_terminal(&service, &job.job_id);

    let batch = service.drain_events();
    assert!(batch.events.len() <= 4, "the bound must hold");
    assert!(
        batch.dropped > 0,
        "overflow must be reported, not silent: {batch:?}"
    );
    service.shutdown().expect("shuts down");
}

#[test]
fn tutorials_and_the_catalogue_come_from_the_real_registry() {
    let (_dir, service) = service_with_store();

    // Against the registry rather than a number written here: a hard-coded
    // count turns adding a capability into a chore, and a chore gets updated
    // without anyone asking whether the change was intended. What matters is
    // that the service exposes exactly what the runtime can execute.
    let catalog = service.catalog();
    let registry = scirust_studio_runtime::build_registry();
    assert_eq!(
        catalog.len(),
        registry.len(),
        "the service must expose every executable capability and nothing else"
    );
    for descriptor in registry.iter()
    {
        assert!(
            catalog.iter().any(|c| c.id == descriptor.id),
            "{} is executable but the service does not expose it",
            descriptor.id
        );
    }

    let tutorials = service.tutorials();
    assert_eq!(tutorials.len(), catalog.len());
    for tutorial in &tutorials
    {
        let doc = service
            .load_tutorial(&tutorial.capability_id)
            .expect("loads");
        let outcome = service.validate_source(&doc.source);
        assert!(
            outcome.valid,
            "the shipped tutorial for {} must validate: {:?}",
            tutorial.capability_id, outcome.problems
        );
    }

    assert!(matches!(
        service.load_tutorial("no.such.capability"),
        Err(AppServiceError::NoSuchTutorial { .. })
    ));
    service.shutdown().expect("shuts down");
}

#[test]
fn shutdown_stops_the_worker_and_refuses_further_work() {
    let (_dir, service) = service_with_store();
    service.shutdown().expect("shuts down");
    assert!(!service.worker_is_running());
    assert!(matches!(
        service.start_run(SPRING_MASS),
        Err(AppServiceError::ShuttingDown) | Err(AppServiceError::WorkerUnavailable { .. })
    ));
}

#[test]
fn shutdown_with_an_active_run_cancels_it_rather_than_leaving_it_running() {
    let (_dir, service) = service_with_store();

    let job = service.start_run(&long_running()).expect("starts");
    wait_for(&service, &job.job_id, "the run to start", |s| {
        matches!(s, JobState::Running { .. })
    });

    service.shutdown().expect("shuts down");
    assert!(!service.worker_is_running());

    let state = service.job_snapshot(&job.job_id).unwrap().state;
    assert!(
        state.is_terminal(),
        "an active job must be settled by shutdown, got {state:?}"
    );
}

#[test]
fn verifying_a_tampered_run_reports_it_without_erroring() {
    let (_dir, service) = service_with_store();

    let job = service.start_run(SPRING_MASS).expect("starts");
    let run_id = match wait_for_terminal(&service, &job.job_id)
    {
        JobState::Completed { run_id } => run_id,
        other => panic!("{other:?}"),
    };

    let path = service
        .store_root()
        .join("runs")
        .join(&run_id)
        .join("result.json");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replacen("0.", "1.", 1)).unwrap();

    let report = service
        .verify_run(&run_id)
        .expect("reports rather than errors");
    assert!(!report.intact);
    assert!(report.detail.is_some(), "it must say what changed");

    service.shutdown().expect("shuts down");
}

/// Kill this service's worker process, by pid.
///
/// Targeting the exact pid rather than matching a command line: a pattern
/// like `pkill -f scirust-studio-worker` also matches the `cargo test -p
/// scirust-studio-worker …` invocation that launched the harness, and
/// promptly kills the test runner itself.
fn kill_worker_process(service: &StudioAppService) {
    let pid = service
        .worker_pid()
        .expect("a running worker has a process id");

    #[cfg(unix)]
    {
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status();
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .status();
    }
    // Give the reader thread a moment to observe the closed pipe.
    std::thread::sleep(Duration::from_millis(100));
}
