//! Simulation-engine error type.

use thiserror::Error;

/// Errors produced by a simulation backend.
///
/// The set is `#[non_exhaustive]` so a backend can motivate new variants without
/// a breaking change; the deterministic core here defines the two every backend
/// needs.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SimError {
    /// The configuration was rejected before any compute began (out-of-range
    /// parameters, an unsatisfiable spec).
    #[error("invalid simulation config: {0}")]
    InvalidConfig(String),
    /// The backend solver failed while running (non-convergence, a numerical
    /// breakdown, an unavailable resource).
    #[error("simulation backend failed: {0}")]
    Backend(String),
}

/// Convenience alias for simulation results.
pub type Result<T> = core::result::Result<T, SimError>;
