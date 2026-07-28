//! Job lifecycle state, as the application (not the worker) sees it.

use serde::{Deserialize, Serialize};

/// How far along a run is.
///
/// The distinctions here are the ones a user can act on, and each is
/// reachable by a different real path — nothing is a synonym for anything
/// else. In particular **cancelled** and **interrupted** are separate:
/// cancelled is what the user asked for, interrupted is the engine dying
/// underneath them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobState {
    /// Accepted, not yet started by the worker.
    Queued,
    /// Running, with genuine measurable progress.
    Running {
        /// Fraction of the time span integrated, `0.0..=1.0`.
        fraction: f64,
        /// The simulation time reached.
        t: f64,
    },
    /// Running, but this capability's solver cannot report intermediate
    /// steps, so there is no fraction to show. A UI must show indeterminate
    /// activity here, never a made-up percentage.
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
    /// The scenario was rejected. Should be rare here — validation runs
    /// before a job is ever started — but the model remains the final word.
    FailedValidation {
        /// The explanation.
        message: String,
    },
    /// A bug in the worker or an adapter.
    FailedInternal {
        /// The explanation.
        message: String,
    },
    /// The worker died while this job was running. The run's partial
    /// evidence remains in the store.
    Interrupted {
        /// What is known about the exit.
        detail: String,
    },
}

impl JobState {
    /// Whether this job has stopped for good.
    pub fn is_terminal(&self) -> bool {
        !matches!(
            self,
            JobState::Queued
                | JobState::Running { .. }
                | JobState::RunningIndeterminate
                | JobState::Cancelling
        )
    }

    /// Whether the job is occupying the single active-run slot.
    pub fn is_active(&self) -> bool {
        !self.is_terminal()
    }

    /// A short, stable label for a status chip.
    pub fn label(&self) -> &'static str {
        match self
        {
            JobState::Queued => "queued",
            JobState::Running { .. } => "running",
            JobState::RunningIndeterminate => "running",
            JobState::Cancelling => "cancelling",
            JobState::Cancelled => "cancelled",
            JobState::Completed { .. } => "completed",
            JobState::FailedNumerical { .. } => "failed",
            JobState::FailedValidation { .. } => "failed",
            JobState::FailedInternal { .. } => "failed",
            JobState::Interrupted { .. } => "interrupted",
        }
    }

    /// The progress fraction, when one genuinely exists.
    ///
    /// `None` for an indeterminate run: a caller that wants a number is made
    /// to handle its absence rather than defaulting to zero and drawing an
    /// empty bar that looks stuck.
    pub fn fraction(&self) -> Option<f64> {
        match self
        {
            JobState::Running { fraction, .. } => Some(*fraction),
            JobState::Completed { .. } => Some(1.0),
            _ => None,
        }
    }
}

/// Everything the UI needs about one job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSnapshot {
    /// This session's id for the job.
    pub job_id: String,
    /// The capability being run.
    pub capability_id: String,
    /// The experiment name from the scenario.
    pub scenario_name: String,
    /// Current state.
    pub state: JobState,
    /// Whether this capability can report progress at all. Known up front
    /// from the capability, so the UI can pick the right control before the
    /// first event rather than switching styles mid-run.
    pub supports_progress: bool,
    /// When the job started, RFC 3339 UTC.
    pub started_at_rfc3339: String,
    /// Elapsed wall-clock seconds, measured monotonically.
    pub elapsed_seconds: f64,
    /// Warnings raised so far.
    pub warnings: Vec<scirust_studio_runtime::RunWarning>,
    /// The run id this job was recorded under, once it has been recorded.
    pub run_id: Option<String>,
}

/// Whether a capability can report genuine intermediate progress.
///
/// Derived from the solvers the capability declares: a fixed-step solver is
/// driven through `scirust-sim`'s per-step observer and reports real
/// fractions, while the adaptive stiff path has no per-step callback to hook
/// (see `docs/studio/RUNTIME_CONTRACT.md`). Read from the registry rather
/// than hard-coded, so adding a capability does not require editing a list
/// here.
pub fn capability_supports_progress(
    descriptor: &scirust_studio_registry::CapabilityDescriptor,
) -> bool {
    descriptor.supported_solvers.iter().any(|s| s.fixed_step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_settled_states_are_terminal() {
        assert!(!JobState::Queued.is_terminal());
        assert!(
            !JobState::Running {
                fraction: 0.5,
                t: 1.0
            }
            .is_terminal()
        );
        assert!(!JobState::RunningIndeterminate.is_terminal());
        assert!(!JobState::Cancelling.is_terminal());

        assert!(JobState::Cancelled.is_terminal());
        assert!(JobState::Completed { run_id: "r".into() }.is_terminal());
        assert!(JobState::Interrupted { detail: "x".into() }.is_terminal());
    }

    #[test]
    fn active_is_the_complement_of_terminal() {
        for state in [
            JobState::Queued,
            JobState::Cancelling,
            JobState::Cancelled,
            JobState::Completed { run_id: "r".into() },
        ]
        {
            assert_eq!(state.is_active(), !state.is_terminal(), "{state:?}");
        }
    }

    /// Cancelled and interrupted must not collapse into one another: one is
    /// the user's decision, the other is a failure.
    #[test]
    fn cancelled_and_interrupted_are_distinct() {
        let cancelled = JobState::Cancelled;
        let interrupted = JobState::Interrupted {
            detail: "worker exited".into(),
        };
        assert_ne!(cancelled, interrupted);
        assert_ne!(cancelled.label(), interrupted.label());
    }

    #[test]
    fn an_indeterminate_run_reports_no_fraction() {
        assert_eq!(JobState::RunningIndeterminate.fraction(), None);
        assert_eq!(
            JobState::Running {
                fraction: 0.25,
                t: 1.0
            }
            .fraction(),
            Some(0.25)
        );
    }

    #[test]
    fn states_round_trip_through_json_with_stable_tags() {
        for state in [
            JobState::Queued,
            JobState::Running {
                fraction: 0.5,
                t: 2.0,
            },
            JobState::RunningIndeterminate,
            JobState::Cancelling,
            JobState::Cancelled,
            JobState::Completed { run_id: "r".into() },
            JobState::FailedNumerical {
                message: "boom".into(),
            },
            JobState::Interrupted { detail: "x".into() },
        ]
        {
            let json = serde_json::to_string(&state).unwrap();
            let back: JobState = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state);
            assert!(json.contains("\"state\""), "{json}");
        }
    }

    /// The progress capability must come from the registry, and must match
    /// what `RUNTIME_CONTRACT.md` documents per capability.
    #[test]
    fn progress_support_is_derived_from_the_real_registry() {
        let registry = scirust_studio_runtime::build_registry();
        for descriptor in registry.iter()
        {
            let supports = capability_supports_progress(descriptor);
            match descriptor.id.0
            {
                "sim.chemistry.robertson" => assert!(
                    !supports,
                    "the adaptive stiff solver cannot report intermediate progress"
                ),
                other => assert!(supports, "{other} integrates with a fixed step"),
            }
        }
    }
}
