//! `scirust-studio-app-service` — application orchestration for SciRust
//! Studio, with no opinion about how it is displayed.
//!
//! This is Phase 3A of the SciRust Studio effort (see
//! `docs/studio/APP_SERVICE.md`). Phase 2B produced a well-behaved worker
//! process and an immutable run store but deliberately left *supervision* to
//! the application — when to restart a worker, what a dead one means for a
//! job in flight, how many runs may be active at once. Those are product
//! decisions, and this is where they are made.
//!
//! ## What this crate does not do
//!
//! It contains no scientific code. Capabilities, integrators, scenario
//! validation rules, IPC framing and store verification all live in the
//! crates that already implement them, and are called, not re-implemented.
//! It also has no dependency on Tauri, Dioxus, or any other UI toolkit: the
//! desktop shell is one caller, the smoke-test binary is another, and the
//! integration tests are a third.
//!
//! ## The policies it does encode
//!
//! - **One worker, one active job.** Adapters are CPU-bound; a second
//!   concurrent run finishes neither sooner, and two progress bars competing
//!   for the same cores is worse than one honest one.
//! - **An interrupted job is never re-run automatically.** Re-executing
//!   someone's calculation without being asked produces a result they did
//!   not request and cannot tell apart from the one they lost.
//! - **Cancellation means cancelled.** For a capability whose solver checks
//!   between steps, cancelling is cooperative. For the adaptive stiff path,
//!   which has no per-step callback to interrupt, the worker process is
//!   terminated — and the job is still classified as *cancelled*, because
//!   that is what the user asked for.
//! - **Storage opens before execution.** A run killed halfway leaves a
//!   `.partial-` directory recording what was attempted, which is the whole
//!   reason the store separates "begin" from "finalize".

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod events;
mod job;
mod paths;
mod service;
mod validate;
mod validation;

pub use error::{AppErrorCode, AppServiceError};
pub use events::{AppEvent, DiagnosticLevel, EventBatch, EventBus};
pub use job::{JobSnapshot, JobState, capability_supports_progress};
pub use paths::{APPLICATION, AppDirectories, VENDOR, application_data_dir};
pub use service::{
    AppServiceConfig, LoadedStoredRun, RunVerificationReport, ScenarioDocument, StoredRunSummary,
    StudioAppService, TutorialDescriptor,
};
pub use validate::validate_source;
pub use validation::{Problem, ProblemSeverity, ProblemStage, SourceLocation, ValidationOutcome};
pub use worker_api::{StderrLog, WorkerExitClass, WorkerLaunchConfig};

/// Worker supervision types, re-exported at the crate root.
mod worker_api {
    pub use crate::worker::{StderrLog, WorkerExitClass, WorkerLaunchConfig};
}

mod worker;

#[cfg(test)]
mod tests {
    use super::*;

    /// The crate must stay free of GUI dependencies: the desktop shell is
    /// one caller of this, not its owner. Asserted against the manifest so
    /// adding one is a deliberate act with a failing test attached.
    #[test]
    fn the_manifest_declares_no_gui_dependency() {
        let manifest = include_str!("../Cargo.toml");
        for forbidden in ["tauri", "dioxus", "wry", "webkit", "gtk", "winit"]
        {
            assert!(
                !manifest.contains(forbidden),
                "`{forbidden}` must not be a dependency of the app service"
            );
        }
    }

    #[test]
    fn the_platform_config_points_at_a_vendor_qualified_store() {
        // Skip where the environment provides no home directory.
        let Ok(config) = AppServiceConfig::platform_default()
        else
        {
            return;
        };
        let text = config.store_root.to_string_lossy().into_owned();
        assert!(text.contains(VENDOR), "{text}");
        assert!(text.contains(APPLICATION), "{text}");
        assert!(config.event_buffer_capacity > 0);
    }
}
