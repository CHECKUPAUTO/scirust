//! [`CcosError`] and the crate [`Result`] alias.

use thiserror::Error;

/// Errors from the cognitive adapter's deterministic core.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum CcosError {
    /// A proposal was constructed (or admitted) that grounds in **no** object.
    /// A cognitive suggestion that concerns nothing in the graph is not a
    /// scientific proposal and is refused (RFC-0002 §06.4 grounding rule).
    #[error("ungrounded proposal: a proposal must concern at least one object")]
    Ungrounded,

    /// A capability-scoped cognitive act was attempted without the capability
    /// having been granted — refused by default (least privilege).
    #[error("cognitive act denied: missing capability {capability}")]
    Denied {
        /// The capability the act required.
        capability: String,
    },

    /// The attestation hash-chain failed verification at this sequence number —
    /// an entry was altered, reordered, or its link to the previous entry broken.
    #[error("attestation chain broken at seq {seq}")]
    ChainBroken {
        /// The 0-based sequence number of the first bad entry.
        seq: u64,
    },
}

/// Convenience alias for cognitive-adapter results.
pub type Result<T> = core::result::Result<T, CcosError>;
