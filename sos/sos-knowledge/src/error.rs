//! Knowledge-engine error type.

use thiserror::Error;

/// Errors produced while building or querying the knowledge graph.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KnowledgeError {
    /// A stored object tagged as an [`crate::Edge`] could not be deserialized —
    /// i.e. its bytes are not a valid `Object<Edge>`.
    #[error("malformed edge object in store: {0}")]
    MalformedEdge(#[from] serde_json::Error),
}

/// Convenience alias for knowledge-engine results.
pub type Result<T> = core::result::Result<T, KnowledgeError>;
