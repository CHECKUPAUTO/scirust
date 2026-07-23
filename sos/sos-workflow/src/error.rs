//! Workflow-engine error type.

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
}

/// Convenience alias for workflow results.
pub type Result<T> = core::result::Result<T, WorkflowError>;
