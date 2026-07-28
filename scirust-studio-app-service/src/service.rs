//! [`StudioAppService`] — the application's single orchestration object.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use scirust_studio_ipc::{ClientMessage, FailureKind, RequestId, WorkerMessage};
use scirust_studio_registry::CapabilityDescriptor;
use scirust_studio_runtime::{build_registry, tutorial_scenario_for};
use scirust_studio_store::{LoadedRunResult, RunManifest, RunStore};

use crate::error::AppServiceError;
use crate::events::{AppEvent, DiagnosticLevel, EventBatch, EventBus};
use crate::job::{JobSnapshot, JobState, capability_supports_progress};
use crate::paths::AppDirectories;
use crate::validation::ValidationOutcome;
use crate::worker::{
    RequestIds, WorkerExitClass, WorkerLaunchConfig, WorkerProcess, read_worker_message,
    spawn_and_handshake,
};

/// How the service is configured at bootstrap.
#[derive(Debug, Clone)]
pub struct AppServiceConfig {
    /// How to find and launch the worker.
    pub worker: WorkerLaunchConfig,
    /// Where runs are stored.
    pub store_root: PathBuf,
    /// How many events to buffer before discarding the oldest.
    pub event_buffer_capacity: usize,
}

impl AppServiceConfig {
    /// The platform default: the worker beside this executable, runs under
    /// the application data directory.
    pub fn platform_default() -> Result<Self, AppServiceError> {
        let dirs = AppDirectories::platform_default()?;
        dirs.create_all()?;
        Ok(AppServiceConfig {
            worker: WorkerLaunchConfig::beside_current_executable(),
            store_root: dirs.root.clone(),
            event_buffer_capacity: 2048,
        })
    }
}

/// One tutorial the application can offer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TutorialDescriptor {
    /// The capability it demonstrates.
    pub capability_id: String,
    /// The capability's display name.
    pub display_name: String,
    /// The file name it ships under.
    pub file_name: String,
    /// One-line description, from the capability.
    pub summary: String,
}

/// A scenario's text plus what is known about it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioDocument {
    /// The `.scirust.toml` source.
    pub source: String,
    /// Where it came from, when it came from a file.
    pub path: Option<String>,
    /// The capability it targets, when determinable.
    pub capability_id: Option<String>,
}

/// A stored run, summarised for a list.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StoredRunSummary {
    /// The run's id.
    pub run_id: String,
    /// The capability it ran.
    pub capability_id: String,
    /// Terminal status label.
    pub status: String,
    /// When it started, RFC 3339.
    pub started_at_rfc3339: String,
    /// When it settled, RFC 3339.
    pub finished_at_rfc3339: String,
    /// The result schema version it was written under.
    pub result_schema_version: u32,
}

impl From<&RunManifest> for StoredRunSummary {
    fn from(m: &RunManifest) -> Self {
        StoredRunSummary {
            run_id: m.run_id.clone(),
            capability_id: m.capability_id.clone(),
            status: m.status.label().to_string(),
            started_at_rfc3339: m.started_at_rfc3339.clone(),
            finished_at_rfc3339: m.finished_at_rfc3339.clone(),
            result_schema_version: m.result_schema_version,
        }
    }
}

/// A loaded run: its manifest, its result, and whether it still verifies.
#[derive(Debug)]
pub struct LoadedStoredRun {
    /// The manifest.
    pub manifest: RunManifest,
    /// The result, tagged with the schema version it was stored under.
    pub result: LoadedRunResult,
    /// The scenario that produced it.
    pub scenario_source: String,
}

/// The outcome of re-checking a stored run's hashes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunVerificationReport {
    /// The run checked.
    pub run_id: String,
    /// Whether every stored hash still matches.
    pub intact: bool,
    /// What failed, when something did.
    pub detail: Option<String>,
}

/// Everything the service knows that a thread might touch.
struct Shared {
    worker: Option<WorkerProcess>,
    /// Jobs by id, in insertion order.
    jobs: BTreeMap<String, JobRecord>,
    /// The single active job, if any.
    active: Option<String>,
    /// Maps a worker request id back to a job.
    by_request: BTreeMap<u64, String>,
    request_ids: RequestIds,
    next_job: u64,
    shutting_down: bool,
    /// Set when the user asked to cancel a job the solver cannot interrupt,
    /// so the worker's death is classified as a cancellation rather than a
    /// crash.
    terminating_for_cancel: bool,
}

struct JobRecord {
    snapshot: JobSnapshot,
    started: Instant,
    /// The store entry opened before execution, so an interrupted run leaves
    /// evidence. Taken when the job settles.
    pending: Option<scirust_studio_store::PendingRun>,
}

/// The application's orchestration object.
///
/// Owns the worker, the job lifecycle, the event bus and the run store. It
/// contains no scientific code: capabilities, validation, IPC framing and
/// store verification all live in the crates that already implement them.
pub struct StudioAppService {
    config: AppServiceConfig,
    store: RunStore,
    events: EventBus,
    shared: Arc<Mutex<Shared>>,
    reader: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl StudioAppService {
    /// Start the service: open the store, launch the worker, handshake, and
    /// begin reading its output.
    pub fn bootstrap(config: AppServiceConfig) -> Result<Self, AppServiceError> {
        let store = RunStore::open(&config.store_root)?;
        let events = EventBus::new(config.event_buffer_capacity);

        let shared = Arc::new(Mutex::new(Shared {
            worker: None,
            jobs: BTreeMap::new(),
            active: None,
            by_request: BTreeMap::new(),
            request_ids: RequestIds::default(),
            next_job: 0,
            shutting_down: false,
            terminating_for_cancel: false,
        }));

        let service = StudioAppService {
            config,
            store,
            events,
            shared,
            reader: Mutex::new(None),
        };
        service.start_worker()?;
        Ok(service)
    }

    /// Launch a worker and spawn the thread that reads its output.
    ///
    /// Public because a user whose worker died should be able to get one
    /// back without restarting the application.
    pub fn start_worker(&self) -> Result<(), AppServiceError> {
        {
            let shared = self.lock()?;
            if shared.shutting_down
            {
                return Err(AppServiceError::ShuttingDown);
            }
            if shared.worker.is_some()
            {
                return Ok(());
            }
        }

        let (worker, mut reader) = spawn_and_handshake(&self.config.worker)?;
        let version = worker.worker_version.clone();
        let path = worker.path().display().to_string();

        {
            let mut shared = self.lock()?;
            shared.worker = Some(worker);
        }
        self.events.publish(AppEvent::WorkerStarted {
            worker_version: version,
            path,
        });

        let shared = Arc::clone(&self.shared);
        let events = self.events.clone();
        let store_root = self.config.store_root.clone();
        let handle = std::thread::spawn(move || {
            loop
            {
                match read_worker_message(&mut reader)
                {
                    Ok(Some(message)) =>
                    {
                        handle_worker_message(&shared, &events, &store_root, message);
                    },
                    Ok(None) =>
                    {
                        handle_worker_gone(&shared, &events, None);
                        return;
                    },
                    Err(e) =>
                    {
                        handle_worker_gone(&shared, &events, Some(e.to_string()));
                        return;
                    },
                }
            }
        });
        if let Ok(mut slot) = self.reader.lock()
        {
            *slot = Some(handle);
        }
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Shared>, AppServiceError> {
        self.shared
            .lock()
            .map_err(|_| AppServiceError::WorkerChannel {
                reason: "internal state was poisoned by a panic".to_string(),
            })
    }

    /// The event bus, for a caller that wants to forward events onward.
    pub fn events(&self) -> &EventBus {
        &self.events
    }

    /// Take everything buffered on the event bus.
    pub fn drain_events(&self) -> EventBatch {
        self.events.drain()
    }

    /// Where runs are stored.
    pub fn store_root(&self) -> &std::path::Path {
        self.store.root()
    }

    /// Whether a worker is currently running.
    pub fn worker_is_running(&self) -> bool {
        self.lock().map(|s| s.worker.is_some()).unwrap_or(false)
    }

    /// The worker's version, when one is running.
    pub fn worker_version(&self) -> Option<String> {
        self.lock()
            .ok()
            .and_then(|s| s.worker.as_ref().map(|w| w.worker_version.clone()))
    }

    /// The worker process's id, when one is running.
    pub fn worker_pid(&self) -> Option<u32> {
        self.lock()
            .ok()
            .and_then(|s| s.worker.as_ref().map(|w| w.pid()))
    }

    /// Lines the worker wrote to stderr, bounded.
    pub fn worker_diagnostics(&self) -> Vec<String> {
        self.lock()
            .ok()
            .and_then(|s| {
                s.worker
                    .as_ref()
                    .and_then(|w| w.stderr.lock().ok().map(|l| l.lines()))
            })
            .unwrap_or_default()
    }

    /// Every capability this build can run.
    pub fn catalog(&self) -> Vec<&'static CapabilityDescriptor> {
        build_registry().iter().collect()
    }

    /// Every shipped tutorial.
    pub fn tutorials(&self) -> Vec<TutorialDescriptor> {
        build_registry()
            .iter()
            .filter_map(|d| {
                let file_name = scirust_studio_runtime::tutorial_file_name_for(d.id.0)?;
                Some(TutorialDescriptor {
                    capability_id: d.id.0.to_string(),
                    display_name: d.display_name.to_string(),
                    file_name: file_name.to_string(),
                    summary: d.summary.to_string(),
                })
            })
            .collect()
    }

    /// The tutorial scenario for a capability.
    pub fn load_tutorial(&self, capability_id: &str) -> Result<ScenarioDocument, AppServiceError> {
        let source = tutorial_scenario_for(capability_id).ok_or_else(|| {
            AppServiceError::NoSuchTutorial {
                capability_id: capability_id.to_string(),
            }
        })?;
        Ok(ScenarioDocument {
            source: source.to_string(),
            path: None,
            capability_id: Some(capability_id.to_string()),
        })
    }

    /// Validate scenario source without running it.
    pub fn validate_source(&self, source: &str) -> ValidationOutcome {
        crate::validate::validate_source(source)
    }

    /// Validate and start a run.
    ///
    /// Validation happens here, in-process, before the worker is involved:
    /// a scenario with a typo should not cost a process round-trip, and the
    /// editor gets its problems back synchronously.
    pub fn start_run(&self, source: &str) -> Result<JobSnapshot, AppServiceError> {
        let outcome = self.validate_source(source);
        if !outcome.valid
        {
            return Err(AppServiceError::Validation(Box::new(outcome)));
        }
        let capability_id = outcome
            .capability_id
            .clone()
            .expect("a valid scenario names its capability");
        let scenario_name = outcome.scenario_name.clone().unwrap_or_default();

        let descriptor = *build_registry().find(&capability_id).ok_or_else(|| {
            AppServiceError::NoSuchTutorial {
                capability_id: capability_id.clone(),
            }
        })?;
        let supports_progress = capability_supports_progress(&descriptor);

        // Open the store entry *before* executing, so a run that is
        // interrupted leaves recoverable evidence of the attempt.
        let pending = self.store.begin(&capability_id, source)?;

        let mut shared = self.lock()?;
        if shared.shutting_down
        {
            return Err(AppServiceError::ShuttingDown);
        }
        if let Some(active) = &shared.active
        {
            return Err(AppServiceError::Busy {
                active_job_id: active.clone(),
            });
        }
        if shared.worker.is_none()
        {
            return Err(AppServiceError::WorkerUnavailable {
                reason: "no worker process is running".to_string(),
            });
        }

        shared.next_job += 1;
        let job_id = format!("job-{}", shared.next_job);
        let request_id = shared.request_ids.next();

        let snapshot = JobSnapshot {
            job_id: job_id.clone(),
            capability_id: capability_id.clone(),
            scenario_name,
            state: JobState::Queued,
            supports_progress,
            started_at_rfc3339: now_rfc3339(),
            elapsed_seconds: 0.0,
            warnings: Vec::new(),
            run_id: None,
        };

        shared.jobs.insert(
            job_id.clone(),
            JobRecord {
                snapshot: snapshot.clone(),
                started: Instant::now(),
                pending: Some(pending),
            },
        );
        shared.active = Some(job_id.clone());
        shared.by_request.insert(request_id.0, job_id.clone());

        let worker = shared.worker.as_mut().expect("checked above");
        worker.send(&ClientMessage::Run {
            request_id,
            scenario_toml: source.to_string(),
        })?;
        drop(shared);

        self.events.publish(AppEvent::JobStarted {
            job_id: job_id.clone(),
            capability_id,
            supports_progress,
        });
        // An indeterminate capability has no first Progress event to move it
        // out of Queued, so say up front what kind of run this is.
        let initial = if supports_progress
        {
            JobState::Queued
        }
        else
        {
            JobState::RunningIndeterminate
        };
        self.set_state(&job_id, initial.clone())?;

        Ok(JobSnapshot {
            state: initial,
            ..snapshot
        })
    }

    /// Ask the worker to cancel a job.
    ///
    /// For a capability whose solver checks cancellation between steps this
    /// is a cooperative stop. For one that cannot be interrupted — the
    /// adaptive stiff path — the worker process is terminated instead, which
    /// is the only mechanism that actually works there. Either way the job
    /// is classified as **cancelled**, because that is what the user asked
    /// for; calling it a failure would blame them for their own decision.
    pub fn cancel_run(&self, job_id: &str) -> Result<(), AppServiceError> {
        let mut shared = self.lock()?;
        let record = shared
            .jobs
            .get(job_id)
            .ok_or_else(|| AppServiceError::NoSuchJob {
                job_id: job_id.to_string(),
            })?;
        if record.snapshot.state.is_terminal()
        {
            return Ok(());
        }
        let cooperative = record.snapshot.supports_progress;
        let request_id = shared
            .by_request
            .iter()
            .find(|(_, j)| j.as_str() == job_id)
            .map(|(r, _)| RequestId(*r));

        if let Some(record) = shared.jobs.get_mut(job_id)
        {
            record.snapshot.state = JobState::Cancelling;
        }
        drop(shared);
        self.events.publish(AppEvent::JobState {
            job_id: job_id.to_string(),
            state: JobState::Cancelling,
        });

        let mut shared = self.lock()?;
        if cooperative
        {
            if let (Some(worker), Some(request_id)) = (shared.worker.as_mut(), request_id)
            {
                worker.send(&ClientMessage::Cancel { request_id })?;
            }
        }
        else
        {
            // Mark first, so the reader thread classifies the resulting exit
            // as a cancellation rather than a crash.
            shared.terminating_for_cancel = true;
            if let Some(worker) = shared.worker.as_mut()
            {
                worker.terminate();
            }
        }
        Ok(())
    }

    fn set_state(&self, job_id: &str, state: JobState) -> Result<(), AppServiceError> {
        {
            let mut shared = self.lock()?;
            match shared.jobs.get_mut(job_id)
            {
                Some(record) => record.snapshot.state = state.clone(),
                None =>
                {
                    return Err(AppServiceError::NoSuchJob {
                        job_id: job_id.to_string(),
                    });
                },
            }
        }
        self.events.publish(AppEvent::JobState {
            job_id: job_id.to_string(),
            state,
        });
        Ok(())
    }

    /// The current state of a job.
    pub fn job_snapshot(&self, job_id: &str) -> Result<JobSnapshot, AppServiceError> {
        let shared = self.lock()?;
        let record = shared
            .jobs
            .get(job_id)
            .ok_or_else(|| AppServiceError::NoSuchJob {
                job_id: job_id.to_string(),
            })?;
        let mut snapshot = record.snapshot.clone();
        snapshot.elapsed_seconds = record.started.elapsed().as_secs_f64();
        Ok(snapshot)
    }

    /// The active job, if one is running.
    pub fn active_job(&self) -> Option<JobSnapshot> {
        let shared = self.lock().ok()?;
        let id = shared.active.clone()?;
        let record = shared.jobs.get(&id)?;
        let mut snapshot = record.snapshot.clone();
        snapshot.elapsed_seconds = record.started.elapsed().as_secs_f64();
        Some(snapshot)
    }

    /// Every run recorded in the store, oldest first.
    pub fn list_runs(&self) -> Result<Vec<StoredRunSummary>, AppServiceError> {
        Ok(self
            .store
            .list()?
            .iter()
            .map(StoredRunSummary::from)
            .collect())
    }

    /// Runs whose writer never finished — evidence of an interruption.
    pub fn interrupted_runs(&self) -> Result<Vec<String>, AppServiceError> {
        Ok(self
            .store
            .interrupted()?
            .into_iter()
            .map(|r| r.run_id)
            .collect())
    }

    /// Load one stored run in full.
    pub fn load_run(&self, run_id: &str) -> Result<LoadedStoredRun, AppServiceError> {
        Ok(LoadedStoredRun {
            manifest: self.store.load_manifest(run_id)?,
            result: self.store.load_result(run_id)?,
            scenario_source: self.store.load_scenario(run_id)?,
        })
    }

    /// Re-check a stored run's hashes.
    pub fn verify_run(&self, run_id: &str) -> Result<RunVerificationReport, AppServiceError> {
        match self.store.verify(run_id)
        {
            Ok(()) => Ok(RunVerificationReport {
                run_id: run_id.to_string(),
                intact: true,
                detail: None,
            }),
            Err(scirust_studio_store::StoreError::HashMismatch {
                file,
                expected,
                actual,
                ..
            }) => Ok(RunVerificationReport {
                run_id: run_id.to_string(),
                intact: false,
                detail: Some(format!(
                    "{file} hashes to {actual} but its manifest records {expected}"
                )),
            }),
            Err(e) => Err(e.into()),
        }
    }

    /// Stop the worker and refuse further work.
    ///
    /// An active job is cancelled first: leaving a worker computing after
    /// the application has gone would be a background process nobody asked
    /// for and nobody can see.
    pub fn shutdown(&self) -> Result<(), AppServiceError> {
        let active = {
            let mut shared = self.lock()?;
            shared.shutting_down = true;
            shared.active.clone()
        };
        if let Some(job_id) = active
        {
            let _ = self.cancel_run(&job_id);
        }

        {
            let mut shared = self.lock()?;
            if let Some(worker) = shared.worker.as_mut()
            {
                let _ = worker.send(&ClientMessage::Shutdown);
                worker.close_input();
            }
        }

        // Give the worker a moment to exit on its own, then insist.
        if let Ok(mut slot) = self.reader.lock()
        {
            if let Some(handle) = slot.take()
            {
                let _ = handle.join();
            }
        }
        let mut shared = self.lock()?;
        if let Some(mut worker) = shared.worker.take()
        {
            if worker.has_exited().is_none()
            {
                worker.terminate();
            }
        }
        drop(shared);
        self.events.publish(AppEvent::WorkerExited {
            class: WorkerExitClass::Requested,
        });
        Ok(())
    }
}

impl Drop for StudioAppService {
    fn drop(&mut self) {
        // Never leave an orphaned worker behind, however the application
        // ended.
        if let Ok(mut shared) = self.shared.lock()
        {
            shared.shutting_down = true;
            if let Some(mut worker) = shared.worker.take()
            {
                worker.close_input();
                if worker.has_exited().is_none()
                {
                    worker.terminate();
                }
            }
        }
    }
}

fn now_rfc3339() -> String {
    // Formatted here rather than pulled from another crate's clock so the
    // service has no opinion about time beyond "when did this start".
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", HumanTime(now.as_secs(), now.subsec_nanos()))
}

/// Minimal RFC 3339 rendering of a Unix timestamp, UTC.
struct HumanTime(u64, u32);

impl std::fmt::Display for HumanTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.0;
        let days = secs / 86_400;
        let rem = secs % 86_400;
        let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

        // Civil-from-days, Howard Hinnant's algorithm.
        let z = days as i64 + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let mth = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if mth <= 2 { y + 1 } else { y };

        write!(
            f,
            "{y:04}-{mth:02}-{d:02}T{h:02}:{m:02}:{s:02}.{:09}Z",
            self.1
        )
    }
}

/// Handle one message from the worker, on the reader thread.
fn handle_worker_message(
    shared: &Arc<Mutex<Shared>>,
    events: &EventBus,
    store_root: &std::path::Path,
    message: WorkerMessage,
) {
    let job_for = |request_id: RequestId| -> Option<String> {
        shared
            .lock()
            .ok()
            .and_then(|s| s.by_request.get(&request_id.0).cloned())
    };

    match message
    {
        WorkerMessage::Progress {
            request_id,
            fraction,
            t,
        } =>
        {
            if let Some(job_id) = job_for(request_id)
            {
                let state = JobState::Running { fraction, t };
                if let Ok(mut s) = shared.lock()
                {
                    if let Some(record) = s.jobs.get_mut(&job_id)
                    {
                        if record.snapshot.state.is_terminal()
                        {
                            return;
                        }
                        record.snapshot.state = state.clone();
                    }
                }
                events.publish(AppEvent::JobState { job_id, state });
            }
        },
        WorkerMessage::Warning {
            request_id,
            warning,
        } =>
        {
            if let Some(job_id) = job_for(request_id)
            {
                if let Ok(mut s) = shared.lock()
                {
                    if let Some(record) = s.jobs.get_mut(&job_id)
                    {
                        record.snapshot.warnings.push(warning.clone());
                    }
                }
                events.publish(AppEvent::JobWarning { job_id, warning });
            }
        },
        WorkerMessage::Completed { request_id, result } =>
        {
            if let Some(job_id) = job_for(request_id)
            {
                settle(shared, events, &job_id, Settlement::Completed(result));
            }
        },
        WorkerMessage::Cancelled { request_id } =>
        {
            if let Some(job_id) = job_for(request_id)
            {
                settle(shared, events, &job_id, Settlement::Cancelled);
            }
        },
        WorkerMessage::Failed {
            request_id,
            kind,
            message,
        } =>
        {
            if let Some(job_id) = job_for(request_id)
            {
                settle(
                    shared,
                    events,
                    &job_id,
                    Settlement::Failed { kind, message },
                );
            }
        },
        WorkerMessage::ProtocolError { message } =>
        {
            events.publish(AppEvent::Diagnostic {
                level: DiagnosticLevel::Error,
                message: format!("worker protocol error: {message}"),
            });
        },
        WorkerMessage::Ready { .. }
        | WorkerMessage::Catalog { .. }
        | WorkerMessage::Validated { .. }
        | WorkerMessage::ShuttingDown =>
        {
            // The application asks for the catalogue and validates
            // in-process, so these are not part of its normal flow; a
            // second `Ready` would mean a protocol bug, not data to act on.
            let _ = store_root;
        },
    }
}

enum Settlement {
    /// Boxed: a `RunResult` carrying long time-courses dwarfs every other
    /// variant, and an unboxed one would make each of them that size.
    Completed(Box<scirust_studio_runtime::RunResult>),
    Cancelled,
    Failed {
        kind: FailureKind,
        message: String,
    },
    Interrupted(String),
}

/// Record a job's terminal outcome, commit it to the store, and free the
/// active slot.
fn settle(shared: &Arc<Mutex<Shared>>, events: &EventBus, job_id: &str, settlement: Settlement) {
    let pending = {
        let Ok(mut s) = shared.lock()
        else
        {
            return;
        };
        let Some(record) = s.jobs.get_mut(job_id)
        else
        {
            return;
        };
        if record.snapshot.state.is_terminal()
        {
            return;
        }
        record.pending.take()
    };

    let (state, run_id) = match settlement
    {
        Settlement::Completed(result) => match pending.map(|p| p.complete(&result))
        {
            Some(Ok(run_id)) => (
                JobState::Completed {
                    run_id: run_id.clone(),
                },
                Some(run_id),
            ),
            Some(Err(e)) => (
                JobState::FailedInternal {
                    message: format!("the run finished but could not be stored: {e}"),
                },
                None,
            ),
            None => (
                JobState::FailedInternal {
                    message: "the run finished but had no store entry".to_string(),
                },
                None,
            ),
        },
        Settlement::Cancelled =>
        {
            if let Some(p) = pending
            {
                let _ = p.cancel();
            }
            (JobState::Cancelled, None)
        },
        Settlement::Failed { kind, message } =>
        {
            if let Some(p) = pending
            {
                let _ = p.fail(&message);
            }
            let state = match kind
            {
                FailureKind::Numerical => JobState::FailedNumerical { message },
                FailureKind::Validation => JobState::FailedValidation { message },
                FailureKind::Internal => JobState::FailedInternal { message },
            };
            (state, None)
        },
        Settlement::Interrupted(detail) =>
        {
            // Deliberately do NOT finalize the store entry. Dropping the
            // pending run leaves its `.partial-` directory in place, which
            // is exactly the evidence the store was designed to preserve.
            drop(pending);
            (JobState::Interrupted { detail }, None)
        },
    };

    if let Ok(mut s) = shared.lock()
    {
        if let Some(record) = s.jobs.get_mut(job_id)
        {
            record.snapshot.state = state.clone();
            record.snapshot.run_id = run_id;
            record.snapshot.elapsed_seconds = record.started.elapsed().as_secs_f64();
        }
        if s.active.as_deref() == Some(job_id)
        {
            s.active = None;
        }
        s.by_request.retain(|_, j| j != job_id);
    }

    events.publish(AppEvent::JobState {
        job_id: job_id.to_string(),
        state,
    });
}

/// The worker's output ended: classify why, and mark any active job.
fn handle_worker_gone(shared: &Arc<Mutex<Shared>>, events: &EventBus, detail: Option<String>) {
    let (active, class, shutting_down) = {
        let Ok(mut s) = shared.lock()
        else
        {
            return;
        };
        let class = if s.terminating_for_cancel
        {
            WorkerExitClass::Terminated
        }
        else if s.shutting_down
        {
            WorkerExitClass::Requested
        }
        else
        {
            WorkerExitClass::Unexpected {
                status: detail
                    .clone()
                    .unwrap_or_else(|| "the worker closed its output".to_string()),
            }
        };
        let terminating = s.terminating_for_cancel;
        s.terminating_for_cancel = false;
        // Drop the handle: the process is gone, and a stale one would make
        // `worker_is_running` lie.
        s.worker = None;
        (s.active.clone(), class, s.shutting_down || terminating)
    };

    if let Some(job_id) = active
    {
        match &class
        {
            // A termination we performed on the user's behalf is a
            // cancellation, not a crash.
            WorkerExitClass::Terminated => settle(shared, events, &job_id, Settlement::Cancelled),
            WorkerExitClass::Requested if shutting_down =>
            {
                settle(shared, events, &job_id, Settlement::Cancelled)
            },
            other =>
            {
                let detail = match other
                {
                    WorkerExitClass::Unexpected { status } => status.clone(),
                    _ => "the worker exited".to_string(),
                };
                settle(shared, events, &job_id, Settlement::Interrupted(detail));
            },
        }
    }

    events.publish(AppEvent::WorkerExited { class });
}
