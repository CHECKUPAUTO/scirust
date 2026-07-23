//! Kernel error type.
//!
//! `sos-core` never panics on recoverable conditions; fallible operations
//! return [`Result`]. Parsing and identity-verification are the only fallible
//! paths in the kernel today.

use thiserror::Error;

/// Errors produced by the SOS kernel.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum SosError {
    /// A hexadecimal digest string was malformed (wrong length, or a
    /// non-hex character). Carries a human-readable reason.
    #[error("invalid digest hex: {0}")]
    InvalidDigestHex(String),

    /// A [`crate::SemVer`] string could not be parsed as `major.minor.patch`.
    #[error("invalid semantic version: {0}")]
    InvalidSemVer(String),

    /// An object's stored [`crate::ObjectId`] did not match the id recomputed
    /// from its content. The object has been tampered with, corrupted, or was
    /// produced by an incompatible schema. Carries `(stored, recomputed)`.
    #[error("object id mismatch: stored {stored} != recomputed {recomputed}")]
    IdMismatch {
        /// The id carried by the object.
        stored: crate::ObjectId,
        /// The id recomputed from the object's current content.
        recomputed: crate::ObjectId,
    },
}

/// Convenience alias for kernel results.
pub type Result<T> = core::result::Result<T, SosError>;
