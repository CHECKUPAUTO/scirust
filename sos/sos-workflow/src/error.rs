//! Workflow-engine error type.

use sos_registry::RegistryError;
use thiserror::Error;

use crate::plan::StageId;

/// Errors produced while building or running a workflow plan.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WorkflowError {
    /// Two stages share the same [`StageId`].
    #[error("duplicate stage id: {0}")]
    DuplicateStage(StageId),
    /// A stage lists a dependency that is not a stage in the plan.
    #[error("stage {stage} depends on unknown stage {dep}")]
    MissingDependency {
        /// The stage with the bad dependency.
        stage: StageId,
        /// The dependency that does not resolve.
        dep: StageId,
    },
    /// The plan's dependency graph contains a cycle (it is not a DAG).
    #[error("workflow plan has a dependency cycle")]
    Cycle,
    /// A stage executor failed while running a cache-missed stage.
    #[error("stage {stage} failed: {reason}")]
    StageFailed {
        /// The stage that failed.
        stage: StageId,
        /// A human-readable reason from the executor.
        reason: String,
    },
    /// A stage could not be **bound** to an implementation: the plugin it names
    /// is not registered, no registered version satisfies it, the registered
    /// implementation's content hash **drifted** from the one the stage pins, or
    /// it needs capabilities the study did not grant. See
    /// [`crate::Dispatch`].
    #[error("stage {stage} could not be bound: {source}")]
    Unbound {
        /// The stage that could not be bound.
        stage: StageId,
        /// Why the registry refused to bind it.
        #[source]
        source: RegistryError,
    },
    /// A stage's plugin resolved in the registry, but no handler was registered
    /// with [`Dispatch::register`](crate::Dispatch::register) to actually run
    /// it — the descriptor is known, the code is missing.
    #[error("stage {stage} resolved plugin `{plugin}`, but no handler is registered for it")]
    NoHandler {
        /// The stage with no runnable implementation.
        stage: StageId,
        /// The plugin name that resolved but cannot run.
        plugin: String,
    },
}

/// Convenience alias for workflow results.
pub type Result<T> = core::result::Result<T, WorkflowError>;
