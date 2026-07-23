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
}

/// Convenience alias for theory results.
pub type Result<T> = core::result::Result<T, TheoryError>;
