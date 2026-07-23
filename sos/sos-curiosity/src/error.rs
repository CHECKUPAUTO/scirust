//! Curiosity-engine error type.

use thiserror::Error;

/// Errors produced by the curiosity engine.
///
/// The deterministic core is largely infallible — a sweep reads an already-built
/// [`sos_knowledge::KnowledgeGraph`] and returns questions. This type gives the
/// crate a stable error surface for its fallible boundaries (e.g. a budget that
/// admits no work, or a future graph-loading path).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CuriosityError {
    /// A sweep was asked for zero questions, so it could do no useful work.
    #[error("curiosity budget allows zero questions")]
    EmptyBudget,
}

/// Convenience alias for curiosity results.
pub type Result<T> = core::result::Result<T, CuriosityError>;
