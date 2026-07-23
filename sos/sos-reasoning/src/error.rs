//! Reasoning-engine error type.

use thiserror::Error;

/// Errors produced by the reasoning engine.
///
/// The deterministic core is largely infallible (it reads an already-built
/// [`sos_knowledge::KnowledgeGraph`]); this type exists for the fallible
/// boundaries (e.g. a future graph-loading path) and to give the crate a stable
/// error surface.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReasoningError {
    /// A goal referenced an object that is not present in the knowledge graph.
    #[error("unknown object in reasoning goal: {0}")]
    UnknownObject(sos_core::ObjectId),
}

/// Convenience alias for reasoning results.
pub type Result<T> = core::result::Result<T, ReasoningError>;
