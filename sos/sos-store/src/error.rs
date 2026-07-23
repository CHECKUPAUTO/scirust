//! Storage-layer error type.

use sos_core::SosError;
use sos_core::kind::Kind;
use thiserror::Error;

/// Errors produced by the storage layer.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// A kernel error — most importantly an
    /// [`SosError::IdMismatch`](sos_core::SosError::IdMismatch) raised when an
    /// object's content address does not match its content (tamper/corruption).
    #[error(transparent)]
    Core(#[from] SosError),

    /// A typed read requested one [`Kind`] but the stored object is another —
    /// e.g. asking for a `Hypothesis` at an id that holds a `Theory`.
    #[error("kind mismatch: requested {requested}, stored {stored}")]
    KindMismatch {
        /// The kind the caller asked to deserialize.
        requested: Kind,
        /// The kind actually stored at that id.
        stored: Kind,
    },

    /// (De)serialization of an object's interchange form failed.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Convenience alias for storage results.
pub type Result<T> = core::result::Result<T, StoreError>;
