//! Reproducibility-engine error type.

use thiserror::Error;

/// Errors produced by the reproducibility engine.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReproError {
    /// A verification was given a different number of claims and reproduced
    /// outcomes — every declared node must have exactly one reproduction.
    #[error("verify inputs disagree: {claims} claims vs {reproduced} reproduced outcomes")]
    LengthMismatch {
        /// Number of declared node claims.
        claims: usize,
        /// Number of reproduced outcomes supplied.
        reproduced: usize,
    },
    /// A workflow error occurred while re-realizing (`rerun`) a plan.
    #[error("workflow error during rerun: {0}")]
    Workflow(#[from] sos_workflow::WorkflowError),
}

/// Convenience alias for reproducibility results.
pub type Result<T> = core::result::Result<T, ReproError>;
