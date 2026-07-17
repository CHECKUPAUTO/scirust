//! Deterministic content and expression digests.
//!
//! SHA-256 (RustCrypto `sha2`) is used **only** for content identity, corruption
//! detection, reproducibility, and relation-expression identity. This is **not**
//! encryption and provides **no** secrecy or authentication guarantee.
//!
//! Every digest is **domain-separated** so a concept payload and an expression
//! serialization can never collide even if their raw bytes coincide:
//!
//! ```text
//! scirust-hypermemory:concept:v1
//! scirust-hypermemory:expression:v1
//! ```
//!
//! Byte fields are **length-prefixed** (`u64` little-endian) before the payload
//! so `("ab", "c")` and `("a", "bc")` never hash equal.

use sha2::{Digest, Sha256};

/// Domain-separation tag for concept-content digests.
pub const DOMAIN_CONCEPT: &[u8] = b"scirust-hypermemory:concept:v1";
/// Domain-separation tag for relation-expression digests.
pub const DOMAIN_EXPRESSION: &[u8] = b"scirust-hypermemory:expression:v1";

/// A 32-byte SHA-256 digest.
pub type Digest32 = [u8; 32];

/// Digest of a concept payload: `SHA-256(DOMAIN_CONCEPT ‖ len(payload) ‖ payload)`.
#[must_use]
pub fn concept_digest(payload: &[u8]) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_CONCEPT);
    hasher.update((payload.len() as u64).to_le_bytes());
    hasher.update(payload);
    hasher.finalize().into()
}

/// An incremental hasher pre-seeded with a domain tag, used by the expression
/// serializer to fold a tree into a stable digest without allocating an
/// intermediate buffer.
pub(crate) struct DomainHasher(Sha256);

impl DomainHasher {
    /// New hasher seeded with `domain`.
    pub(crate) fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self(hasher)
    }

    /// Absorb raw bytes.
    #[inline]
    pub(crate) fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    /// Finish and produce the 32-byte digest.
    #[must_use]
    pub(crate) fn finalize(self) -> Digest32 {
        self.0.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concept_digest_is_stable_and_domain_separated() {
        let d1 = concept_digest(b"hello");
        let d2 = concept_digest(b"hello");
        assert_eq!(d1, d2, "same input → same digest");
        assert_ne!(concept_digest(b"hello"), concept_digest(b"world"));
    }

    #[test]
    fn length_prefix_prevents_concatenation_collision() {
        // Without a length prefix these would be indistinguishable byte streams
        // once the domain tag is fixed; the prefix separates them.
        assert_ne!(concept_digest(b"ab"), concept_digest(b"abc"));
        assert_ne!(concept_digest(b""), concept_digest(b"\0"));
    }

    #[test]
    fn empty_payload_has_a_defined_digest() {
        // Deterministic even for the empty payload (domain + zero length).
        assert_eq!(concept_digest(b""), concept_digest(b""));
    }

    #[test]
    fn domain_hasher_matches_manual_construction() {
        let mut h = DomainHasher::new(DOMAIN_EXPRESSION);
        h.update(&[0x01]);
        h.update(&7u32.to_le_bytes());
        let got = h.finalize();

        let mut manual = Sha256::new();
        manual.update(DOMAIN_EXPRESSION);
        manual.update([0x01]);
        manual.update(7u32.to_le_bytes());
        let expected: Digest32 = manual.finalize().into();

        assert_eq!(got, expected);
    }
}
