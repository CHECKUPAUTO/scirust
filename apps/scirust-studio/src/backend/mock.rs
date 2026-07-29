//! A deterministic stand-in backend, for tests and interface previews only.
//!
//! # What this is not
//!
//! It is **not** a second implementation of the simulation runtime. It
//! integrates nothing, solves nothing and verifies nothing. Every number it
//! returns is a hard-coded literal chosen to make a chart draw, and the data
//! is labelled as fabricated everywhere it appears. Building a second
//! integrator here — even a simple one — would create exactly the thing this
//! project refuses to have: two answers to the same scientific question.
//!
//! # Why it exists
//!
//! Two reasons, both about being able to test the interface rather than
//! click through it:
//!
//! 1. The reducers and the whole run lifecycle can be driven end to end on
//!    the host, with a predictable sequence of states.
//! 2. `dx serve` renders the interface outside a Tauri host, which is how
//!    layout and styling get iterated on without a full native rebuild.
//!
//! # Why it cannot ship
//!
//! `super`'s `compile_error!` refuses a release build with this feature on,
//! and [`super::USES_MOCK_DATA`] makes the interface show a permanent
//! banner. Both would have to be deliberately removed for a user to ever see
//! one of these numbers.

use core::cell::RefCell;

use super::wire::{
    BootstrapWire, CapabilityWire, CheckWire, DistributionWire, ErrorWire, EventBatchWire,
    EventWire, FieldWire, IntegrityWire, JobStateWire, JobWire, LocationWire, MetricWire,
    OutputWire, ProblemWire, ProvenanceWire, RunWire, ScenarioFileWire, ScenarioWire,
    SeriesRoleWire, SeriesWire, StoredRunWire, ValidationWire, VerificationWire, XAxisKindWire,
};
use super::{FrontendError, StudioBackend};

/// The fabricated horizontal coordinates. Nine points, unevenly spaced, so
/// the chart's handling of a non-uniform axis is exercised.
const MOCK_X: [f64; 9] = [0.0, 0.1, 0.25, 0.4, 0.55, 0.8, 1.2, 1.6, 2.0];

/// The fabricated series. Not a solution to anything.
const MOCK_Y: [f64; 9] = [1.0, 0.82, 0.51, 0.18, -0.09, -0.31, -0.22, -0.05, 0.03];

/// The job id the mock always issues.
const MOCK_JOB: &str = "mock-job-1";

/// The run id the mock always records.
const MOCK_RUN: &str = "20260101T000000Z-mockmockmockmock";

/// A legacy run, so the sample-index path is visible in a preview.
const MOCK_LEGACY_RUN: &str = "20250101T000000Z-legacylegacyleg1";

#[derive(Debug, Default)]
struct MockState {
    job: Option<JobWire>,
    /// How many times the job has been polled. The lifecycle advances one
    /// step per poll, so a test drives it by polling rather than by sleeping.
    polls: u32,
    cancel_requested: bool,
    events: Vec<EventWire>,
    completed_runs: Vec<StoredRunWire>,
}

/// The deterministic stand-in backend.
#[derive(Debug, Default)]
pub struct MockBackend {
    state: RefCell<MockState>,
}

impl MockBackend {
    /// A backend with nothing having happened yet.
    pub fn new() -> Self {
        MockBackend::default()
    }

    /// The tutorial source the mock hands back, and the only source it
    /// accepts as valid.
    pub fn tutorial_source() -> &'static str {
        "schema_version = 1\n\
         \n\
         [experiment]\n\
         name = \"Mock experiment\"\n\
         capability = \"sim.mechanics.spring_mass_damper\"\n"
    }

    fn capability() -> CapabilityWire {
        CapabilityWire {
            id: "sim.mechanics.spring_mass_damper".to_string(),
            display_name: "Spring–mass–damper (MOCK DATA)".to_string(),
            category: "Mechanics".to_string(),
            source_crate: "mock".to_string(),
            summary: "Fabricated catalogue entry for interface previews.".to_string(),
            maturity: "Stable".to_string(),
            determinism: "BitExact".to_string(),
            solvers: vec!["rk4".to_string()],
            parameters: vec![FieldWire {
                name: "mass".to_string(),
                display_name: "Mass".to_string(),
                required: true,
                units: vec!["kg".to_string()],
                description: "Fabricated.".to_string(),
            }],
            initial_state: vec![FieldWire {
                name: "x".to_string(),
                display_name: "Displacement".to_string(),
                required: true,
                units: vec!["m".to_string()],
                description: "Fabricated.".to_string(),
            }],
            outputs: vec![OutputWire {
                id: "x".to_string(),
                display_name: "Displacement".to_string(),
                unit: "m".to_string(),
                description: "Fabricated.".to_string(),
            }],
            checks: vec![CheckWire {
                id: "energy_drift".to_string(),
                description: "Fabricated.".to_string(),
            }],
            supports_progress: true,
            has_tutorial: true,
        }
    }

    fn run_view(run_id: &str, legacy: bool) -> RunWire {
        let (kind, label, unit, x) = if legacy
        {
            (
                XAxisKindWire::SampleIndex,
                "Sample index".to_string(),
                String::new(),
                (0..MOCK_Y.len()).map(|i| i as f64).collect::<Vec<_>>(),
            )
        }
        else
        {
            (
                XAxisKindWire::PhysicalCoordinates,
                "time".to_string(),
                "s".to_string(),
                MOCK_X.to_vec(),
            )
        };
        RunWire {
            run_id: run_id.to_string(),
            capability_id: "sim.mechanics.spring_mass_damper".to_string(),
            scenario_name: "Mock experiment".to_string(),
            status: "completed".to_string(),
            result_schema_version: if legacy { 1 } else { 2 },
            x_axis_kind: kind,
            x_axis_label: label,
            x_axis_unit: unit,
            x_values: x,
            series: vec![SeriesWire {
                id: "x".to_string(),
                display_name: "Displacement (MOCK)".to_string(),
                unit: "m".to_string(),
                // The fabricated capability is a spring-mass-damper, which
                // draws no sample; there is no ensemble to be a member of.
                role: SeriesRoleWire::Trajectory,
                values: MOCK_Y.to_vec(),
            }],
            // The fabricated capability produces curves, not a field.
            fields: Vec::new(),
            // A histogram, though — deliberately, and not because the
            // fabricated spring-mass-damper has one. The mock exists so the
            // interface can be developed and demonstrated without a backend,
            // and a rendering path it never exercises is a rendering path
            // nobody notices breaking. The overflow count is non-zero for the
            // same reason: the "samples outside the binned range" line is
            // exactly the sort of thing that is only ever seen when it is
            // already too late.
            distributions: vec![DistributionWire {
                id: "mock_distribution".to_string(),
                display_name: "Fabricated distribution (MOCK)".to_string(),
                unit: "1".to_string(),
                edges: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                counts: vec![3, 11, 24, 9, 2],
                underflow: 0,
                overflow: 1,
            }],
            metrics: vec![MetricWire {
                id: "mock".to_string(),
                display_name: "Fabricated metric".to_string(),
                value: "0".to_string(),
                numeric: Some(0.0),
                unit: None,
            }],
            verifications: vec![VerificationWire {
                id: "energy_drift".to_string(),
                status: "not_applicable".to_string(),
                measured: None,
                threshold: None,
                explanation: "Mock data is not verified and never can be.".to_string(),
            }],
            warnings: Vec::new(),
            provenance: ProvenanceWire {
                capability_id: "sim.mechanics.spring_mass_damper".to_string(),
                determinism: "BitExact".to_string(),
                adapter_crate: "mock".to_string(),
                adapter_version: "0.0.0".to_string(),
                result_schema_version: if legacy { 1 } else { 2 },
                target: "mock".to_string(),
                started_at: "2026-01-01T00:00:00Z".to_string(),
                completed_at: "2026-01-01T00:00:01Z".to_string(),
                elapsed_seconds: 1.0,
                // The fabricated capability is a spring-mass-damper, which
                // consumes no seed. A number here would render a Seed row in
                // the inspector for a run that never drew a sample.
                seed: None,
            },
            scenario_source: MockBackend::tutorial_source().to_string(),
            integrity: IntegrityWire {
                run_id: run_id.to_string(),
                intact: true,
                detail: None,
            },
        }
    }

    fn stored(run_id: &str, legacy: bool) -> StoredRunWire {
        StoredRunWire {
            run_id: run_id.to_string(),
            capability_id: "sim.mechanics.spring_mass_damper".to_string(),
            status: "completed".to_string(),
            started_at: "2026-01-01T00:00:00Z".to_string(),
            finished_at: "2026-01-01T00:00:01Z".to_string(),
            result_schema_version: if legacy { 1 } else { 2 },
        }
    }

    fn snapshot(state: &JobStateWire) -> JobWire {
        JobWire {
            job_id: MOCK_JOB.to_string(),
            capability_id: "sim.mechanics.spring_mass_damper".to_string(),
            scenario_name: "Mock experiment".to_string(),
            state: state.clone(),
            supports_progress: true,
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_string(),
            elapsed_seconds: 0.0,
            warnings: Vec::new(),
            run_id: match state
            {
                JobStateWire::Completed { run_id } => Some(run_id.clone()),
                _ => None,
            },
        }
    }

    /// The state the job reaches after `polls` polls.
    ///
    /// A pure function of the poll count, so a test that drives the mock
    /// gets the same sequence every time and can assert on the whole
    /// lifecycle rather than on whatever a timer happened to produce.
    fn state_after(polls: u32, cancel_requested: bool) -> JobStateWire {
        if cancel_requested
        {
            return if polls >= 1
            {
                JobStateWire::Cancelled
            }
            else
            {
                JobStateWire::Cancelling
            };
        }
        match polls
        {
            0 => JobStateWire::Queued,
            1 => JobStateWire::Running {
                fraction: 0.25,
                t: 0.5,
            },
            2 => JobStateWire::Running {
                fraction: 0.5,
                t: 1.0,
            },
            3 => JobStateWire::Running {
                fraction: 0.75,
                t: 1.5,
            },
            _ => JobStateWire::Completed {
                run_id: MOCK_RUN.to_string(),
            },
        }
    }
}

/// Whether a job is still occupying the single active-run slot.
fn is_active(state: &JobStateWire) -> bool {
    matches!(
        state,
        JobStateWire::Queued
            | JobStateWire::Running { .. }
            | JobStateWire::RunningIndeterminate
            | JobStateWire::Cancelling
    )
}

/// The mock rejects any source that is not the tutorial, so that the
/// validation path is exercised without pretending to understand TOML.
fn validate(source: &str) -> ValidationWire {
    if source.trim() == MockBackend::tutorial_source().trim()
    {
        ValidationWire {
            valid: true,
            capability_id: Some("sim.mechanics.spring_mass_damper".to_string()),
            scenario_name: Some("Mock experiment".to_string()),
            problems: Vec::new(),
        }
    }
    else
    {
        ValidationWire {
            valid: false,
            capability_id: None,
            scenario_name: None,
            problems: vec![ProblemWire {
                code: Some("MOCK-0001".to_string()),
                title: "The mock backend only accepts its own tutorial".to_string(),
                explanation: "This build serves fabricated data and does not parse scenarios."
                    .to_string(),
                suggested_action: Some("Load the tutorial from the catalogue.".to_string()),
                field: None,
                location: Some(LocationWire { line: 1, column: 1 }),
                stage: "parse".to_string(),
                severity: "error".to_string(),
            }],
        }
    }
}

impl StudioBackend for MockBackend {
    async fn bootstrap(&self) -> Result<BootstrapWire, FrontendError> {
        Ok(BootstrapWire {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            worker_version: Some("mock".to_string()),
            worker_running: true,
            store_path: "/mock/store".to_string(),
            capability_count: 1,
            interrupted_runs: Vec::new(),
            worker_problem: None,
        })
    }

    async fn catalog(&self) -> Result<Vec<CapabilityWire>, FrontendError> {
        Ok(vec![MockBackend::capability()])
    }

    async fn tutorials(&self) -> Result<Vec<CapabilityWire>, FrontendError> {
        Ok(vec![MockBackend::capability()])
    }

    async fn load_tutorial(&self, capability_id: &str) -> Result<ScenarioWire, FrontendError> {
        if capability_id != "sim.mechanics.spring_mass_damper"
        {
            return Err(FrontendError::from_wire(ErrorWire {
                code: "UnknownCapability".to_string(),
                title: "No such capability".to_string(),
                explanation: format!("The mock backend knows nothing about `{capability_id}`."),
                recoverable: true,
                suggested_action: None,
                validation: None,
            }));
        }
        Ok(ScenarioWire {
            source: MockBackend::tutorial_source().to_string(),
            path: None,
            capability_id: Some(capability_id.to_string()),
        })
    }

    /// The mock never shows a dialog — there is no host to show one — so it
    /// reports the outcome a user who cancelled would produce. That keeps the
    /// preview honest: `dx serve` cannot open a real file, and pretending it
    /// had would mean fabricating a scenario the user did not choose.
    async fn open_scenario(&self) -> Result<Option<ScenarioFileWire>, FrontendError> {
        Ok(None)
    }

    async fn save_scenario(&self, _source: &str) -> Result<Option<String>, FrontendError> {
        Ok(None)
    }

    async fn validate_scenario(&self, source: &str) -> Result<ValidationWire, FrontendError> {
        Ok(validate(source))
    }

    async fn start_run(&self, source: &str) -> Result<JobWire, FrontendError> {
        let outcome = validate(source);
        if !outcome.valid
        {
            return Err(FrontendError::from_wire(ErrorWire {
                code: "Validation".to_string(),
                title: "The scenario was rejected".to_string(),
                explanation: "The mock backend only accepts its own tutorial.".to_string(),
                recoverable: true,
                suggested_action: Some("Load the tutorial from the catalogue.".to_string()),
                validation: Some(outcome),
            }));
        }

        let mut state = self.state.borrow_mut();
        if state.job.as_ref().is_some_and(|j| is_active(&j.state))
        {
            return Err(FrontendError::from_wire(ErrorWire {
                code: "Busy".to_string(),
                title: "A run is already in progress".to_string(),
                explanation: "One run at a time.".to_string(),
                recoverable: true,
                suggested_action: Some("Wait for it to finish, or cancel it.".to_string()),
                validation: None,
            }));
        }

        state.polls = 0;
        state.cancel_requested = false;
        let job = MockBackend::snapshot(&JobStateWire::Queued);
        state.job = Some(job.clone());
        state.events.push(EventWire::JobStarted {
            job_id: MOCK_JOB.to_string(),
            capability_id: "sim.mechanics.spring_mass_damper".to_string(),
            supports_progress: true,
        });
        Ok(job)
    }

    async fn cancel_run(&self, job_id: &str) -> Result<(), FrontendError> {
        let mut state = self.state.borrow_mut();
        if state.job.as_ref().map(|j| j.job_id.as_str()) != Some(job_id)
        {
            return Err(FrontendError::from_wire(ErrorWire {
                code: "UnknownJob".to_string(),
                title: "No such run".to_string(),
                explanation: format!("`{job_id}` is not a job this session started."),
                recoverable: true,
                suggested_action: None,
                validation: None,
            }));
        }
        state.cancel_requested = true;
        state.polls = 0;
        Ok(())
    }

    async fn job_snapshot(&self, job_id: &str) -> Result<JobWire, FrontendError> {
        let mut state = self.state.borrow_mut();
        if state.job.as_ref().map(|j| j.job_id.as_str()) != Some(job_id)
        {
            return Err(FrontendError::from_wire(ErrorWire {
                code: "UnknownJob".to_string(),
                title: "No such run".to_string(),
                explanation: format!("`{job_id}` is not a job this session started."),
                recoverable: true,
                suggested_action: None,
                validation: None,
            }));
        }
        let next = MockBackend::state_after(state.polls, state.cancel_requested);
        state.polls = state.polls.saturating_add(1);
        if let JobStateWire::Completed { run_id } = &next
        {
            let already = state.completed_runs.iter().any(|r| &r.run_id == run_id);
            if !already
            {
                state
                    .completed_runs
                    .push(MockBackend::stored(run_id, false));
            }
        }
        state.events.push(EventWire::JobState {
            job_id: job_id.to_string(),
            state: next.clone(),
        });
        let snapshot = MockBackend::snapshot(&next);
        state.job = Some(snapshot.clone());
        Ok(snapshot)
    }

    async fn active_job(&self) -> Result<Option<JobWire>, FrontendError> {
        let state = self.state.borrow();
        Ok(state.job.clone().filter(|j| {
            !matches!(
                j.state,
                JobStateWire::Completed { .. }
                    | JobStateWire::Cancelled
                    | JobStateWire::Interrupted { .. }
                    | JobStateWire::FailedNumerical { .. }
                    | JobStateWire::FailedValidation { .. }
                    | JobStateWire::FailedInternal { .. }
            )
        }))
    }

    async fn list_runs(&self) -> Result<Vec<StoredRunWire>, FrontendError> {
        let state = self.state.borrow();
        let mut runs = vec![MockBackend::stored(MOCK_LEGACY_RUN, true)];
        runs.extend(state.completed_runs.iter().cloned());
        Ok(runs)
    }

    async fn load_run(&self, run_id: &str) -> Result<RunWire, FrontendError> {
        match run_id
        {
            MOCK_RUN => Ok(MockBackend::run_view(MOCK_RUN, false)),
            MOCK_LEGACY_RUN => Ok(MockBackend::run_view(MOCK_LEGACY_RUN, true)),
            other => Err(FrontendError::from_wire(ErrorWire {
                code: "UnknownRun".to_string(),
                title: "No such run".to_string(),
                explanation: format!("`{other}` is not in the mock store."),
                recoverable: true,
                suggested_action: None,
                validation: None,
            })),
        }
    }

    async fn verify_run(&self, run_id: &str) -> Result<IntegrityWire, FrontendError> {
        Ok(IntegrityWire {
            run_id: run_id.to_string(),
            intact: true,
            detail: None,
        })
    }

    async fn poll_events(&self) -> Result<EventBatchWire, FrontendError> {
        let mut state = self.state.borrow_mut();
        Ok(EventBatchWire {
            events: core::mem::take(&mut state.events),
            dropped: 0,
        })
    }

    async fn store_path(&self) -> Result<String, FrontendError> {
        Ok("/mock/store".to_string())
    }

    async fn restart_worker(&self) -> Result<BootstrapWire, FrontendError> {
        self.bootstrap().await
    }

    async fn worker_diagnostics(&self) -> Result<Vec<String>, FrontendError> {
        Ok(vec!["mock backend: no worker process exists".to_string()])
    }

    async fn frontend_ready(&self) -> Result<BootstrapWire, FrontendError> {
        self.bootstrap().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives a future to completion on the host.
    ///
    /// The mock never yields — every method returns a value immediately — so
    /// a single poll always finishes it, and no executor dependency is
    /// needed to test the whole lifecycle.
    fn block_on<F: core::future::Future>(future: F) -> F::Output {
        use core::task::{Context, Poll, Waker};
        // `Waker::noop` needs no executor and no `unsafe`, which keeps this
        // crate's `forbid(unsafe_code)` intact.
        let mut context = Context::from_waker(Waker::noop());
        let mut future = core::pin::pin!(future);
        match future.as_mut().poll(&mut context)
        {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("the mock backend must never yield"),
        }
    }

    #[test]
    fn a_run_progresses_deterministically_to_completion() {
        let backend = MockBackend::new();
        let job = block_on(backend.start_run(MockBackend::tutorial_source())).expect("started");
        assert_eq!(job.state, JobStateWire::Queued);

        let states: Vec<JobStateWire> = (0..5)
            .map(|_| {
                block_on(backend.job_snapshot(MOCK_JOB))
                    .expect("a snapshot")
                    .state
            })
            .collect();
        assert_eq!(states[0], JobStateWire::Queued);
        assert_eq!(
            states[1],
            JobStateWire::Running {
                fraction: 0.25,
                t: 0.5
            }
        );
        assert_eq!(
            states[4],
            JobStateWire::Completed {
                run_id: MOCK_RUN.to_string()
            }
        );
    }

    #[test]
    fn cancelling_settles_as_cancelled_not_as_a_failure() {
        let backend = MockBackend::new();
        block_on(backend.start_run(MockBackend::tutorial_source())).expect("started");
        block_on(backend.job_snapshot(MOCK_JOB)).expect("a snapshot");
        block_on(backend.cancel_run(MOCK_JOB)).expect("cancelled");

        assert_eq!(
            block_on(backend.job_snapshot(MOCK_JOB)).unwrap().state,
            JobStateWire::Cancelling
        );
        assert_eq!(
            block_on(backend.job_snapshot(MOCK_JOB)).unwrap().state,
            JobStateWire::Cancelled
        );
    }

    #[test]
    fn a_completed_run_becomes_loadable_and_charts_against_real_coordinates() {
        let backend = MockBackend::new();
        block_on(backend.start_run(MockBackend::tutorial_source())).expect("started");
        for _ in 0..5
        {
            block_on(backend.job_snapshot(MOCK_JOB)).expect("a snapshot");
        }
        let runs = block_on(backend.list_runs()).expect("runs");
        assert!(runs.iter().any(|r| r.run_id == MOCK_RUN));

        let run = block_on(backend.load_run(MOCK_RUN)).expect("a run");
        assert_eq!(run.x_axis_kind, XAxisKindWire::PhysicalCoordinates);
        assert_eq!(run.x_values.len(), run.series[0].values.len());
    }

    /// The legacy fixture must never claim coordinates it does not have.
    #[test]
    fn the_legacy_run_is_labelled_as_sample_indices() {
        let backend = MockBackend::new();
        let run = block_on(backend.load_run(MOCK_LEGACY_RUN)).expect("a run");
        assert_eq!(run.x_axis_kind, XAxisKindWire::SampleIndex);
        assert_eq!(run.x_axis_label, "Sample index");
        assert!(run.x_axis_unit.is_empty());
        assert_eq!(run.result_schema_version, 1);
    }

    #[test]
    fn an_unacceptable_scenario_fails_validation_with_a_location() {
        let backend = MockBackend::new();
        let error = block_on(backend.start_run("nonsense")).expect_err("rejected");
        assert_eq!(error.code, "Validation");
        assert_eq!(error.problems().len(), 1);
        assert_eq!(error.problems()[0].location.map(|l| l.line), Some(1));
    }

    /// Everything the mock returns must be visibly fabricated. A preview
    /// that looks like a result is the failure mode this guards against.
    #[test]
    fn every_mock_value_announces_itself() {
        let backend = MockBackend::new();
        let capability = &block_on(backend.catalog()).unwrap()[0];
        assert!(capability.display_name.contains("MOCK"));
        assert_eq!(capability.source_crate, "mock");

        let run = block_on(backend.load_run(MOCK_RUN)).unwrap();
        assert!(run.series[0].display_name.contains("MOCK"));
        assert_eq!(run.verifications[0].status, "not_applicable");
        assert_eq!(run.provenance.adapter_crate, "mock");
    }

    /// The mock must not invent a file the user did not pick.
    ///
    /// `dx serve` runs the interface outside a Tauri host, so there is no
    /// native dialog to show. Reporting the cancelled outcome is the honest
    /// answer; fabricating a scenario here would make the preview show a
    /// file-open working when the shipped path is what actually has to work.
    #[test]
    fn the_mock_reports_a_cancelled_dialog_rather_than_inventing_a_file() {
        let backend = MockBackend::new();
        let opened = block_on(backend.open_scenario());
        assert_eq!(opened.unwrap(), None);
        let saved = block_on(backend.save_scenario("schema_version = 1"));
        assert_eq!(saved.unwrap(), None);
    }
}
