//! Planning-engine error type.

use thiserror::Error;

/// Errors produced by the planning engine.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PlannerError {
    /// [`recommend`](crate::Planner::recommend) was called with no candidate
    /// designs — there is nothing to rank or recommend.
    #[error("no candidate designs to plan over")]
    NoCandidates,
}

/// Convenience alias for planning results.
pub type Result<T> = core::result::Result<T, PlannerError>;
