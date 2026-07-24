//! [`CliError`] — every way a command can fail, and its [`Result`] alias.

use thiserror::Error;

/// Errors the `sos` command surface can produce.
///
/// This is a thin porcelain over already-tested engines, so most variants wrap
/// an underlying engine error rather than defining new failure semantics.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CliError {
    /// The command line itself was malformed (missing/extra arguments, an
    /// unknown subcommand, an unparseable value).
    #[error("{0}")]
    Usage(String),

    /// The named object was not found in the store.
    #[error("object not found: {0}")]
    NotFound(sos_core::ObjectId),

    /// A filesystem operation failed.
    #[error("filesystem error: {0}")]
    Io(#[from] std::io::Error),

    /// (De)serialization of a command's input or output failed.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// A storage-layer error.
    #[error(transparent)]
    Store(#[from] sos_store::StoreError),

    /// A knowledge-graph query error.
    #[error(transparent)]
    Knowledge(#[from] sos_knowledge::KnowledgeError),

    /// A provenance-graph query error.
    #[error(transparent)]
    Provenance(#[from] sos_provenance::ProvError),

    /// A planning error.
    #[error(transparent)]
    Planner(#[from] sos_planner::PlannerError),

    /// A publication-engine error.
    #[error(transparent)]
    Publication(#[from] sos_publication::PublicationError),

    /// A dependency-graph read error (used by `sos diff`).
    #[error(transparent)]
    Source(#[from] sos_publication::SourceError),

    /// A registry error.
    #[error(transparent)]
    Registry(#[from] sos_registry::RegistryError),
}

/// Convenience alias for command results.
pub type Result<T> = core::result::Result<T, CliError>;
