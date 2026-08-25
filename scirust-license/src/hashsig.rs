//! Backward-compatible hash-signature namespace for `scirust-license`.
//!
//! The implementation now lives in the neutral `scirust-hashsig` crate so
//! licensing, provenance, SciCapsule and verification products can share the
//! same audited Lamport/Merkle primitive without depending on licensing.

pub use scirust_hashsig::*;
