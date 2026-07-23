//! Provenance-engine error type.

use thiserror::Error;

/// Errors produced while querying the provenance graph.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProvError {
    /// A stored object's bytes could not be parsed as an object header while
    /// walking the graph — i.e. it was not written via
    /// [`sos_store::TypedStore::put_object`].
    #[error("malformed object header in store: {0}")]
    MalformedHeader(#[from] serde_json::Error),
}

/// Convenience alias for provenance results.
pub type Result<T> = core::result::Result<T, ProvError>;
