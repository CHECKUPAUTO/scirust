//! Theory-engine error type.

use sos_core::ObjectId;
use thiserror::Error;

/// Errors produced by the theory engine.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TheoryError {
    /// A referenced theory is not present in the store (or is not a `Theory`).
    #[error("unknown theory: {0}")]
    UnknownTheory(ObjectId),
    /// The underlying store failed (I/O, or an integrity check).
    #[error("store error: {0}")]
    Store(#[from] sos_store::StoreError),
    /// A discrimination problem needs at least two rivals — there is nothing
    /// to tell apart otherwise.
    #[error("a discrimination problem needs at least two rivals (got {0})")]
    TooFewRivals(usize),
    /// The same theory was listed twice as a rival. No experiment separates a
    /// theory from itself.
    #[error("a theory cannot be its own rival")]
    DuplicateRival,
    /// No candidate designs were submitted.
    #[error("no candidate designs were submitted for discrimination")]
    NoDiscriminatingCandidates,
    /// A design claims to bear on a theory that is not one of this problem's
    /// rivals — informative, perhaps, but not about *these* theories.
    #[error("{0} is not a rival in this discrimination problem")]
    NotARival(sos_core::ObjectId),
    /// A design claims to separate a theory from itself.
    #[error("a design cannot separate {0} from itself")]
    SelfSeparation(sos_core::ObjectId),
    /// The Planning Engine rejected the submitted designs. Ranking,
    /// admission and stopping are its call, not this crate's.
    #[error("planner rejected the discrimination candidates: {0}")]
    Planner(#[from] sos_planner::PlannerError),
}

/// Convenience alias for theory results.
pub type Result<T> = core::result::Result<T, TheoryError>;
