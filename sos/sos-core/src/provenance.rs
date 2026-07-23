//! Provenance metadata carried by every object: who/what produced it.
//!
//! * [`ProducerRef`] — the engine or plugin that produced the object, pinned by
//!   name + [`crate::SemVer`] + content hash, so a rerun that resolves a
//!   different implementation is a detected drift.
//! * [`Author`] — the principal (human, agent, or engine) that initiated the
//!   work.
//! * [`Signature`] — an optional detached attestation slot (filled by
//!   `sos-provenance`, which wraps `scirust-provenance`'s Merkle/Lamport
//!   signing). It is **excluded from the object id**, because a signature
//!   cannot be part of the content it signs.

use serde::{Deserialize, Serialize};

use crate::canonical::{Canonical, CanonicalEncoder};
use crate::hash::Digest;
use crate::version::SemVer;

/// The engine or plugin that produced an object, content-pinned.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProducerRef {
    /// Producer name, e.g. `"sos-reasoning"` or `"sos-scirust/symreg"`.
    pub name: String,
    /// Producer semantic version.
    pub version: SemVer,
    /// Content hash of the producer artifact itself (its plugin digest), so the
    /// exact code that produced the object is pinned, not just its name.
    pub content_hash: Digest,
}

impl ProducerRef {
    /// Construct a producer reference.
    #[must_use]
    pub fn new(name: impl Into<String>, version: SemVer, content_hash: Digest) -> Self {
        Self {
            name: name.into(),
            version,
            content_hash,
        }
    }
}

impl Canonical for ProducerRef {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.name);
        enc.value(&self.version);
        enc.bytes(self.content_hash.as_bytes());
    }
}

/// The principal that initiated an object's creation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "principal", content = "id", rename_all = "lowercase")]
pub enum Author {
    /// A human, identified by a stable handle.
    Human(String),
    /// A software agent (e.g. an LLM/cognitive backend acting as a proposer).
    Agent(String),
    /// An SOS engine acting autonomously (e.g. the Curiosity Engine).
    Engine(String),
}

impl Author {
    /// A human author from a handle.
    #[must_use]
    pub fn human(handle: impl Into<String>) -> Self {
        Self::Human(handle.into())
    }
    /// An agent author from a name.
    #[must_use]
    pub fn agent(name: impl Into<String>) -> Self {
        Self::Agent(name.into())
    }
    /// An engine author from a name.
    #[must_use]
    pub fn engine(name: impl Into<String>) -> Self {
        Self::Engine(name.into())
    }

    /// A stable discriminant used in the canonical encoding.
    const fn discriminant(&self) -> u64 {
        match self
        {
            Self::Human(_) => 0,
            Self::Agent(_) => 1,
            Self::Engine(_) => 2,
        }
    }

    /// The principal's identifier string.
    #[must_use]
    pub fn id(&self) -> &str {
        match self
        {
            Self::Human(s) | Self::Agent(s) | Self::Engine(s) => s,
        }
    }
}

impl Canonical for Author {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
        enc.str(self.id());
    }
}

/// A detached cryptographic attestation over an object's id.
///
/// The kernel only carries the typed container; producing and verifying
/// signatures is `sos-provenance`'s job (it wraps `scirust-provenance`). A
/// signature is **not** part of the object's id (§03.3), so adding one to an
/// object never changes its content address.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Signature {
    /// Name of the signing scheme, e.g. `"merkle-lamport-sha256"`.
    pub algo: String,
    /// The raw signature bytes.
    pub bytes: Vec<u8>,
}

impl Signature {
    /// Construct a signature container.
    #[must_use]
    pub fn new(algo: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            algo: algo.into(),
            bytes,
        }
    }
}

impl Canonical for Signature {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.algo);
        enc.bytes(&self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::HashAlgo;

    fn digest() -> Digest {
        HashAlgo::Sha256.hash(b"t", b"producer")
    }

    #[test]
    fn author_variants_do_not_collide() {
        assert_ne!(
            Author::human("x").canonical_bytes(),
            Author::agent("x").canonical_bytes()
        );
        assert_ne!(
            Author::agent("x").canonical_bytes(),
            Author::engine("x").canonical_bytes()
        );
    }

    #[test]
    fn author_id_accessor() {
        assert_eq!(Author::human("ada").id(), "ada");
        assert_eq!(Author::engine("curiosity").id(), "curiosity");
    }

    #[test]
    fn producer_ref_canonical_is_stable() {
        let p = ProducerRef::new("sos-reasoning", SemVer::new(0, 1, 0), digest());
        assert_eq!(p.canonical_bytes(), p.clone().canonical_bytes());
    }

    #[test]
    fn author_serde_roundtrips() {
        let a = Author::agent("ccos");
        let json = serde_json::to_string(&a).unwrap();
        let back: Author = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }
}
