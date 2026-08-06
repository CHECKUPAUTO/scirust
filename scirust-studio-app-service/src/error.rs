//! Application-level failures.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Something the application could not do.
///
/// Every variant carries enough to build a real error card — a stable code,
/// a title, an explanation and, where one exists, a suggested action. A UI
/// that can only say "something went wrong" is a UI that was handed an error
/// type like that.
#[derive(Debug, Clone, PartialEq)]
pub enum AppServiceError {
    /// No suitable directory for application data on this platform.
    NoApplicationDirectory {
        /// Why not.
        reason: String,
    },
    /// A filesystem operation failed.
    Io {
        /// The path involved.
        path: PathBuf,
        /// The underlying message.
        source: String,
    },
    /// The worker executable could not be found.
    WorkerNotFound {
        /// Where the service looked.
        searched: Vec<PathBuf>,
    },
    /// The worker could not be started.
    WorkerLaunchFailed {
        /// The executable that was tried.
        path: PathBuf,
        /// Why it failed.
        reason: String,
    },
    /// The worker is running but does not speak this build's protocol.
    WorkerVersionMismatch {
        /// The version this build speaks.
        ours: u32,
        /// What the worker said, when it said anything intelligible.
        theirs: Option<u32>,
        /// The worker's own explanation.
        detail: String,
    },
    /// The worker is not currently available — it exited, or never started.
    WorkerUnavailable {
        /// What is known about why.
        reason: String,
    },
    /// The channel to the worker failed.
    WorkerChannel {
        /// What went wrong.
        reason: String,
    },
    /// A run was requested while another is still active.
    ///
    /// The application deliberately runs one scientific job at a time; see
    /// `docs/studio/APP_SERVICE.md`.
    Busy {
        /// The job already running.
        active_job_id: String,
    },
    /// The scenario did not validate. Carries the structured report so the
    /// editor can point at fields rather than printing a blob.
    Validation(Box<crate::validation::ValidationOutcome>),
    /// No such job in this session.
    NoSuchJob {
        /// The id asked for.
        job_id: String,
    },
    /// No tutorial with this capability id.
    NoSuchTutorial {
        /// The id asked for.
        capability_id: String,
    },
    /// The run store failed.
    Store {
        /// The store's own message.
        reason: String,
    },
    /// The service has been shut down.
    ShuttingDown,
}

/// A stable, machine-readable code for an [`AppServiceError`].
///
/// Stable across releases: a UI keys its help topics and its telemetry-free
/// diagnostics off these, so renaming one is a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AppErrorCode {
    /// [`AppServiceError::NoApplicationDirectory`]
    NoApplicationDirectory,
    /// [`AppServiceError::Io`]
    Io,
    /// [`AppServiceError::WorkerNotFound`]
    WorkerNotFound,
    /// [`AppServiceError::WorkerLaunchFailed`]
    WorkerLaunchFailed,
    /// [`AppServiceError::WorkerVersionMismatch`]
    WorkerVersionMismatch,
    /// [`AppServiceError::WorkerUnavailable`]
    WorkerUnavailable,
    /// [`AppServiceError::WorkerChannel`]
    WorkerChannel,
    /// [`AppServiceError::Busy`]
    Busy,
    /// [`AppServiceError::Validation`]
    Validation,
    /// [`AppServiceError::NoSuchJob`]
    NoSuchJob,
    /// [`AppServiceError::NoSuchTutorial`]
    NoSuchTutorial,
    /// [`AppServiceError::Store`]
    Store,
    /// [`AppServiceError::ShuttingDown`]
    ShuttingDown,
}

impl AppServiceError {
    /// This error's stable code.
    pub fn code(&self) -> AppErrorCode {
        match self
        {
            AppServiceError::NoApplicationDirectory { .. } => AppErrorCode::NoApplicationDirectory,
            AppServiceError::Io { .. } => AppErrorCode::Io,
            AppServiceError::WorkerNotFound { .. } => AppErrorCode::WorkerNotFound,
            AppServiceError::WorkerLaunchFailed { .. } => AppErrorCode::WorkerLaunchFailed,
            AppServiceError::WorkerVersionMismatch { .. } => AppErrorCode::WorkerVersionMismatch,
            AppServiceError::WorkerUnavailable { .. } => AppErrorCode::WorkerUnavailable,
            AppServiceError::WorkerChannel { .. } => AppErrorCode::WorkerChannel,
            AppServiceError::Busy { .. } => AppErrorCode::Busy,
            AppServiceError::Validation(_) => AppErrorCode::Validation,
            AppServiceError::NoSuchJob { .. } => AppErrorCode::NoSuchJob,
            AppServiceError::NoSuchTutorial { .. } => AppErrorCode::NoSuchTutorial,
            AppServiceError::Store { .. } => AppErrorCode::Store,
            AppServiceError::ShuttingDown => AppErrorCode::ShuttingDown,
        }
    }

    /// A short title, suitable as an error card's heading.
    pub fn title(&self) -> &'static str {
        match self
        {
            AppServiceError::NoApplicationDirectory { .. } => "No place to store application data",
            AppServiceError::Io { .. } => "A file could not be read or written",
            AppServiceError::WorkerNotFound { .. } => "The calculation engine is missing",
            AppServiceError::WorkerLaunchFailed { .. } => "The calculation engine did not start",
            AppServiceError::WorkerVersionMismatch { .. } =>
            {
                "The calculation engine is a different version"
            },
            AppServiceError::WorkerUnavailable { .. } => "The calculation engine is not running",
            AppServiceError::WorkerChannel { .. } => "Lost contact with the calculation engine",
            AppServiceError::Busy { .. } => "A run is already in progress",
            AppServiceError::Validation(_) => "The scenario is not valid",
            AppServiceError::NoSuchJob { .. } => "No such run in this session",
            AppServiceError::NoSuchTutorial { .. } => "No example for that capability",
            AppServiceError::Store { .. } => "The run store could not be used",
            AppServiceError::ShuttingDown => "The application is closing",
        }
    }

    /// Whether the user can plausibly carry on after this.
    pub fn recoverable(&self) -> bool {
        match self
        {
            AppServiceError::Busy { .. }
            | AppServiceError::Validation(_)
            | AppServiceError::NoSuchJob { .. }
            | AppServiceError::NoSuchTutorial { .. }
            | AppServiceError::WorkerUnavailable { .. }
            | AppServiceError::WorkerChannel { .. } => true,
            AppServiceError::NoApplicationDirectory { .. }
            | AppServiceError::Io { .. }
            | AppServiceError::WorkerNotFound { .. }
            | AppServiceError::WorkerLaunchFailed { .. }
            | AppServiceError::WorkerVersionMismatch { .. }
            | AppServiceError::Store { .. }
            | AppServiceError::ShuttingDown => false,
        }
    }

    /// What the user could try next, when there is something concrete.
    pub fn suggested_action(&self) -> Option<&'static str> {
        match self
        {
            AppServiceError::WorkerNotFound { .. } => Some(
                "Reinstall the application: the calculation engine is part of the installation.",
            ),
            AppServiceError::WorkerVersionMismatch { .. } => Some(
                "Reinstall the application so the interface and the calculation engine come from the same build.",
            ),
            AppServiceError::WorkerUnavailable { .. } | AppServiceError::WorkerChannel { .. } =>
            {
                Some("Restart the calculation engine, then run the scenario again.")
            },
            AppServiceError::Busy { .. } =>
            {
                Some("Wait for the current run to finish, or cancel it first.")
            },
            AppServiceError::Validation(_) =>
            {
                Some("Correct the reported fields and validate again.")
            },
            AppServiceError::Io { .. } | AppServiceError::Store { .. } =>
            {
                Some("Check that the storage location exists and is writable.")
            },
            _ => None,
        }
    }
}

impl std::fmt::Display for AppServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            AppServiceError::NoApplicationDirectory { reason } =>
            {
                write!(f, "no application data directory: {reason}")
            },
            AppServiceError::Io { path, source } =>
            {
                write!(f, "{}: {source}", path.display())
            },
            AppServiceError::WorkerNotFound { searched } =>
            {
                let list = searched
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "the worker executable was not found (looked in: {list})")
            },
            AppServiceError::WorkerLaunchFailed { path, reason } =>
            {
                write!(f, "could not start {}: {reason}", path.display())
            },
            AppServiceError::WorkerVersionMismatch {
                ours,
                theirs,
                detail,
            } => match theirs
            {
                Some(t) => write!(
                    f,
                    "the worker speaks protocol v{t}, this build speaks v{ours}: {detail}"
                ),
                None => write!(f, "the worker did not accept protocol v{ours}: {detail}"),
            },
            AppServiceError::WorkerUnavailable { reason } =>
            {
                write!(f, "the worker is not available: {reason}")
            },
            AppServiceError::WorkerChannel { reason } =>
            {
                write!(f, "the worker channel failed: {reason}")
            },
            AppServiceError::Busy { active_job_id } =>
            {
                write!(f, "run {active_job_id} is still active")
            },
            AppServiceError::Validation(outcome) =>
            {
                write!(f, "{} problem(s) in the scenario", outcome.problems.len())
            },
            AppServiceError::NoSuchJob { job_id } => write!(f, "no job `{job_id}`"),
            AppServiceError::NoSuchTutorial { capability_id } =>
            {
                write!(f, "no tutorial for capability `{capability_id}`")
            },
            AppServiceError::Store { reason } => write!(f, "run store: {reason}"),
            AppServiceError::ShuttingDown => write!(f, "the application is shutting down"),
        }
    }
}

impl std::error::Error for AppServiceError {}

impl From<scirust_studio_store::StoreError> for AppServiceError {
    fn from(e: scirust_studio_store::StoreError) -> Self {
        AppServiceError::Store {
            reason: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_a_distinct_code_and_a_title() {
        let errors = vec![
            AppServiceError::NoApplicationDirectory { reason: "x".into() },
            AppServiceError::Io {
                path: PathBuf::from("/x"),
                source: "x".into(),
            },
            AppServiceError::WorkerNotFound { searched: vec![] },
            AppServiceError::WorkerLaunchFailed {
                path: PathBuf::from("/x"),
                reason: "x".into(),
            },
            AppServiceError::WorkerVersionMismatch {
                ours: 1,
                theirs: Some(2),
                detail: "x".into(),
            },
            AppServiceError::WorkerUnavailable { reason: "x".into() },
            AppServiceError::WorkerChannel { reason: "x".into() },
            AppServiceError::Busy {
                active_job_id: "j".into(),
            },
            AppServiceError::NoSuchJob { job_id: "j".into() },
            AppServiceError::NoSuchTutorial {
                capability_id: "c".into(),
            },
            AppServiceError::Store { reason: "x".into() },
            AppServiceError::ShuttingDown,
        ];

        let mut codes = std::collections::BTreeSet::new();
        for e in &errors
        {
            assert!(!e.title().is_empty(), "{e:?}");
            assert!(!e.to_string().is_empty(), "{e:?}");
            assert!(
                codes.insert(format!("{:?}", e.code())),
                "duplicate code for {e:?}"
            );
        }
    }

    #[test]
    fn a_version_mismatch_names_both_versions() {
        let text = AppServiceError::WorkerVersionMismatch {
            ours: 1,
            theirs: Some(9),
            detail: "refused".into(),
        }
        .to_string();
        assert!(text.contains("v1") && text.contains("v9"), "{text}");
    }

    #[test]
    fn a_missing_worker_lists_where_it_looked() {
        let text = AppServiceError::WorkerNotFound {
            searched: vec![PathBuf::from("/a/worker"), PathBuf::from("/b/worker")],
        }
        .to_string();
        assert!(
            text.contains("/a/worker") && text.contains("/b/worker"),
            "{text}"
        );
    }

    /// Recoverable errors are the ones a user can act on without
    /// reinstalling; getting this backwards produces an application that
    /// tells people to reinstall over a typo in their scenario.
    #[test]
    fn recoverability_matches_what_the_user_can_actually_do() {
        assert!(
            AppServiceError::Busy {
                active_job_id: "j".into()
            }
            .recoverable()
        );
        assert!(
            !AppServiceError::WorkerNotFound { searched: vec![] }.recoverable(),
            "a missing engine is not something the user can fix in the editor"
        );
    }

    #[test]
    fn actionable_errors_suggest_an_action() {
        for e in [
            AppServiceError::WorkerNotFound { searched: vec![] },
            AppServiceError::Busy {
                active_job_id: "j".into(),
            },
            AppServiceError::WorkerUnavailable { reason: "x".into() },
        ]
        {
            assert!(e.suggested_action().is_some(), "{e:?}");
        }
    }
}
