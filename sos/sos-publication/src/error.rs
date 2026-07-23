//! [`PublicationError`] (operational failures), its [`Result`] alias, and the
//! [`SourceError`] a [`PublicationObjectSource`](crate::source::PublicationObjectSource)
//! raises.
//!
//! ## Errors vs. findings
//!
//! A deliberate line runs through this crate: **operational failures are
//! `Err`; scientific outcomes are report data.** A missing dependency, a
//! contradicted claim, an unmet reproducibility bar, or a structurally
//! incomplete document are *findings* — surfaced in the
//! [`VerificationReport`](crate::verify::VerificationReport), never as an
//! `Err`. Collapsing them into an error would hide the rest of the report and
//! violate the mandate that verification *never hides contradictions*. Only a
//! genuine operational fault (an unrenderable format, a backend read failure,
//! a serialization failure) is an error here.

use thiserror::Error;

use crate::render::Format;

/// Operational failures from building, rendering, or verifying a publication.
///
/// Scientific outcomes (unsupported/contradicted/unresolved claims, missing
/// dependencies, reproducibility gaps) are **not** here — they are fields of
/// the [`VerificationReport`](crate::verify::VerificationReport).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PublicationError {
    /// A render target this engine does not emit (e.g. [`Format::Latex`] /
    /// [`Format::Pdf`], which need a typesetting backend deferred per Invariant
    /// VIII).
    ///
    /// [`Format::Latex`]: crate::render::Format::Latex
    /// [`Format::Pdf`]: crate::render::Format::Pdf
    #[error("unsupported render format: {}", .0.code())]
    UnsupportedFormat(Format),

    /// A read from the underlying object source failed — a backend fault or a
    /// stored object whose header could not be decoded. Distinct from "the
    /// object is absent", which is a finding, not an error.
    #[error("object source error: {0}")]
    Source(#[from] SourceError),

    /// Serialization of a rendered JSON artifact failed. Serialization of this
    /// crate's value types is total, so this is reserved for a genuine serde
    /// fault.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// A failure reading the object graph through a
/// [`PublicationObjectSource`](crate::source::PublicationObjectSource).
///
/// **Absence is not a failure.** `source.facts(id) == Ok(None)` means "no such
/// object" — a finding the verifier records as a missing dependency. A
/// `SourceError` is a genuine fault: the backend could not answer, or the bytes
/// it holds could not be decoded into the header the verifier needs.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SourceError {
    /// The stored bytes for an object could not be decoded into an object header
    /// (missing/renamed fields, corrupt record). The object exists but cannot be
    /// interpreted.
    #[error("could not decode object header: {0}")]
    Decode(String),

    /// A backend-specific failure (I/O, transport, lock). The deterministic core
    /// never produces this itself; adapters over a fallible backend do.
    #[error("object source backend error: {0}")]
    Backend(String),
}

/// Convenience alias for publication results.
pub type Result<T> = core::result::Result<T, PublicationError>;
