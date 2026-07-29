//! The typed bridge between the interface and the native shell.
//!
//! The frontend has exactly one way to reach the outside world: the
//! [`StudioBackend`] trait. There is no `fetch`, no WebSocket, no
//! `localStorage`-backed cache of scientific data and no shell access — the
//! webview is granted none of those (see
//! `src-tauri/capabilities/main-window.json`), and this trait is the whole
//! surface it does have.
//!
//! # Two implementations, one of which can never ship
//!
//! * [`bridge::TauriBridge`] — the real one. Calls the shell's typed
//!   commands over Tauri's IPC and decodes the JSON.
//! * [`mock::MockBackend`] — a deterministic in-memory stand-in for frontend
//!   tests and for `dx serve` outside a Tauri host. It is behind the
//!   `mock-backend` feature and **a release build that enables it does not
//!   compile** (see the `compile_error!` below).
//!
//! That guard is not a nicety. A scientific application that can silently
//! display invented numbers is worse than one that fails to start, so the
//! failure is moved to build time where nobody can miss it, and the mock —
//! when it is active at all — also raises a permanent banner through
//! [`USES_MOCK_DATA`].

pub mod wire;

#[cfg(feature = "mock-backend")]
pub mod mock;

#[cfg(target_arch = "wasm32")]
pub mod bridge;

pub use wire::{
    BootstrapWire, CapabilityWire, CheckWire, ErrorWire, EventBatchWire, EventWire, ExitClassWire,
    FieldWire, IntegrityWire, JobStateWire, JobWire, LocationWire, MetricWire, OutputWire,
    ProblemWire, ProvenanceWire, RunWire, ScenarioFileWire, ScenarioWire, SeriesRoleWire,
    SeriesWire, StoredRunWire, ValidationWire, VerificationWire, WarningWire, XAxisKindWire,
};

// A shipped application must never be able to display invented data. The
// mock exists for tests and for previewing the interface without a Tauri
// host; compiling it into an optimised build is a mistake that must not be
// possible to make quietly.
#[cfg(all(feature = "mock-backend", not(debug_assertions)))]
compile_error!(
    "the `mock-backend` feature is enabled in a build with debug assertions off. \
     The mock serves fabricated results and must never be compiled into a shipped \
     application. Build without `--features mock-backend`, or use a debug profile \
     for interface previews."
);

/// Whether this build serves fabricated data.
///
/// Read by the interface, which shows a permanent, unmissable banner when it
/// is true. A user must never be able to mistake a preview for a result.
pub const USES_MOCK_DATA: bool = cfg!(feature = "mock-backend");

// The flag and the feature must agree. Checked at compile time rather than
// in a test, because a test can be filtered out and this cannot.
#[cfg(not(feature = "mock-backend"))]
const _: () = assert!(
    !USES_MOCK_DATA,
    "a build without `mock-backend` must never report mock data"
);
#[cfg(feature = "mock-backend")]
const _: () = assert!(
    USES_MOCK_DATA,
    "a build with `mock-backend` must declare itself"
);

/// Everything that can go wrong on this side of the bridge.
///
/// Shaped like the shell's error view because that is what an error card
/// needs, and because a bridge failure and a service failure look the same
/// to a user: something did not happen, here is why, here is what to try.
/// There is deliberately no path that produces only "an error occurred".
#[derive(Debug, Clone, PartialEq)]
pub struct FrontendError {
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

impl FrontendError {
    /// The shell rejected the call and said why.
    pub fn from_wire(wire: ErrorWire) -> Self {
        FrontendError {
            code: wire.code,
            title: wire.title,
            explanation: wire.explanation,
            recoverable: wire.recoverable,
            suggested_action: wire.suggested_action,
            validation: wire.validation,
        }
    }

    /// There is no Tauri host: the page is open somewhere it cannot work.
    ///
    /// Recoverable is false because nothing the user does inside this page
    /// can produce a backend; the honest advice is to launch the
    /// application.
    pub fn bridge_unavailable(detail: impl Into<String>) -> Self {
        FrontendError {
            code: "BRIDGE_UNAVAILABLE".to_string(),
            title: "The application backend is not reachable".to_string(),
            explanation: format!(
                "This interface must run inside the SciRust Studio desktop application: {}",
                detail.into()
            ),
            recoverable: false,
            suggested_action: Some(
                "Start the SciRust Studio application rather than opening this page directly."
                    .to_string(),
            ),
            validation: None,
        }
    }

    /// The shell answered with something this build cannot decode.
    ///
    /// Kept distinct from a service error on purpose: a decoding failure is
    /// a version mismatch between the interface and the shell, and telling
    /// the user "the simulation failed" would send them looking in the
    /// wrong place entirely.
    pub fn malformed_response(command: &str, detail: impl Into<String>) -> Self {
        FrontendError {
            code: "BRIDGE_DECODE_FAILED".to_string(),
            title: "The application sent an unexpected answer".to_string(),
            explanation: format!(
                "`{command}` returned data this interface could not read: {}",
                detail.into()
            ),
            recoverable: true,
            suggested_action: Some(
                "This usually means the interface and the application were built from \
                 different versions. Reinstall the application."
                    .to_string(),
            ),
            validation: None,
        }
    }

    /// The call was rejected with something that is not an error view.
    pub fn opaque_rejection(command: &str, detail: impl Into<String>) -> Self {
        FrontendError {
            code: "BRIDGE_REJECTED".to_string(),
            title: "The application refused the request".to_string(),
            explanation: format!("`{command}` failed: {}", detail.into()),
            recoverable: true,
            suggested_action: None,
            validation: None,
        }
    }

    /// The blocking validation problems, when this was a validation failure.
    pub fn problems(&self) -> &[ProblemWire] {
        match &self.validation
        {
            Some(v) => &v.problems,
            None => &[],
        }
    }
}

impl core::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.title, self.explanation)
    }
}

/// Everything the interface can ask the native shell to do.
///
/// One method per Tauri command, each narrow and typed. Adding a capability
/// to the interface means adding a method here and a command there — there
/// is no general-purpose escape hatch to reach for instead.
///
/// `async fn` in a trait is used without a `Send` bound because this runs on
/// a WebAssembly module's single thread, where nothing is sent anywhere.
#[allow(async_fn_in_trait)]
pub trait StudioBackend {
    /// What the interface needs at start-up.
    async fn bootstrap(&self) -> Result<BootstrapWire, FrontendError>;

    /// The capability catalogue, from the real registry.
    async fn catalog(&self) -> Result<Vec<CapabilityWire>, FrontendError>;

    /// Every capability that ships a tested tutorial.
    async fn tutorials(&self) -> Result<Vec<CapabilityWire>, FrontendError>;

    /// Load a capability's tutorial scenario.
    async fn load_tutorial(&self, capability_id: &str) -> Result<ScenarioWire, FrontendError>;

    /// Ask the user for a scenario file and return its **contents**.
    ///
    /// `None` means the user cancelled, which is an outcome and not an error.
    /// No path crosses in either direction: this build cannot re-open a file
    /// without the user choosing it again, and that is deliberate.
    async fn open_scenario(&self) -> Result<Option<ScenarioFileWire>, FrontendError>;

    /// Ask the user where to save `source`, returning the chosen file's name.
    ///
    /// `None` means the user cancelled.
    async fn save_scenario(&self, source: &str) -> Result<Option<String>, FrontendError>;

    /// Validate scenario source without running it.
    async fn validate_scenario(&self, source: &str) -> Result<ValidationWire, FrontendError>;

    /// Validate and start a run.
    async fn start_run(&self, source: &str) -> Result<JobWire, FrontendError>;

    /// Ask a run to stop.
    async fn cancel_run(&self, job_id: &str) -> Result<(), FrontendError>;

    /// The current state of a run.
    async fn job_snapshot(&self, job_id: &str) -> Result<JobWire, FrontendError>;

    /// The run occupying the single active slot, when there is one.
    async fn active_job(&self) -> Result<Option<JobWire>, FrontendError>;

    /// Everything recorded in the run store.
    async fn list_runs(&self) -> Result<Vec<StoredRunWire>, FrontendError>;

    /// Load one stored run, including its integrity check.
    async fn load_run(&self, run_id: &str) -> Result<RunWire, FrontendError>;

    /// Re-check a stored run's hashes.
    async fn verify_run(&self, run_id: &str) -> Result<IntegrityWire, FrontendError>;

    /// Drain buffered application events.
    async fn poll_events(&self) -> Result<EventBatchWire, FrontendError>;

    /// Where runs are stored.
    async fn store_path(&self) -> Result<String, FrontendError>;

    /// Start a worker after one has exited.
    async fn restart_worker(&self) -> Result<BootstrapWire, FrontendError>;

    /// Bounded worker stderr, for the diagnostics panel only.
    async fn worker_diagnostics(&self) -> Result<Vec<String>, FrontendError>;

    /// Tell the shell the interface has rendered.
    async fn frontend_ready(&self) -> Result<BootstrapWire, FrontendError>;
}

/// The backend this build uses.
///
/// Resolved at compile time, so there is no runtime switch an attacker or a
/// stray configuration file could flip.
#[cfg(all(target_arch = "wasm32", not(feature = "mock-backend")))]
pub type ActiveBackend = bridge::TauriBridge;

/// The backend this build uses: the mock, because `mock-backend` is on.
#[cfg(all(target_arch = "wasm32", feature = "mock-backend"))]
pub type ActiveBackend = mock::MockBackend;

/// The backend this build uses.
#[cfg(target_arch = "wasm32")]
pub fn active_backend() -> ActiveBackend {
    ActiveBackend::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interface must be able to tell the user which kind of build this
    /// is, and the answer must come from the feature rather than from a
    /// value someone could set.
    #[test]
    fn the_build_reports_its_own_data_source() {
        assert_eq!(USES_MOCK_DATA, cfg!(feature = "mock-backend"));
    }

    #[test]
    fn a_validation_failure_keeps_its_problems() {
        let wire = ErrorWire {
            code: "Validation".to_string(),
            title: "The scenario was rejected".to_string(),
            explanation: "…".to_string(),
            recoverable: true,
            suggested_action: None,
            validation: Some(ValidationWire {
                valid: false,
                capability_id: Some("sim.mechanics.spring_mass_damper".to_string()),
                scenario_name: None,
                problems: vec![ProblemWire {
                    code: Some("SRST-0201".to_string()),
                    title: "Unknown field".to_string(),
                    explanation: "`mas` is not a parameter".to_string(),
                    suggested_action: Some("Did you mean `mass`?".to_string()),
                    field: Some("mass".to_string()),
                    location: Some(LocationWire { line: 7, column: 1 }),
                    stage: "capability".to_string(),
                    severity: "error".to_string(),
                }],
            }),
        };
        let error = FrontendError::from_wire(wire);
        assert_eq!(error.problems().len(), 1);
        assert_eq!(error.problems()[0].location.map(|l| l.line), Some(7));
    }

    /// A missing host is not a scientific failure and must not read like
    /// one.
    #[test]
    fn a_missing_host_is_reported_as_a_bridge_problem_not_a_run_failure() {
        let error = FrontendError::bridge_unavailable("`window.__TAURI__` is undefined");
        assert_eq!(error.code, "BRIDGE_UNAVAILABLE");
        assert!(!error.recoverable);
        assert!(error.suggested_action.is_some());
        assert!(error.problems().is_empty());
    }

    /// A version mismatch must say so rather than blaming the simulation.
    #[test]
    fn a_decode_failure_points_at_the_version_not_at_the_science() {
        let error =
            FrontendError::malformed_response("studio_load_run", "missing field `x_values`");
        assert_eq!(error.code, "BRIDGE_DECODE_FAILED");
        assert!(error.explanation.contains("studio_load_run"));
        assert!(
            error
                .suggested_action
                .as_deref()
                .unwrap_or_default()
                .contains("version"),
            "the advice must name the real cause"
        );
    }

    #[test]
    fn every_error_renders_as_a_full_card() {
        let errors = [
            FrontendError::bridge_unavailable("x"),
            FrontendError::malformed_response("c", "d"),
            FrontendError::opaque_rejection("c", "d"),
        ];
        for error in errors
        {
            assert!(!error.code.is_empty());
            assert!(!error.title.is_empty());
            assert!(!error.explanation.is_empty());
            assert!(!format!("{error}").is_empty());
        }
    }
}
