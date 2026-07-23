//! [`ObjectId`] — the content address of a scientific object.
//!
//! An `ObjectId` is a [`Digest`] over an object's [`crate::canonical`] bytes,
//! taken under a per-kind domain-separation prefix. Because an object's
//! canonical form includes the ids of its parents, an `ObjectId` is a Merkle
//! hash over the object's entire lineage: change any ancestor and every
//! descendant id changes, so tampering anywhere in the graph is detectable.

use core::fmt;
use serde::{Deserialize, Serialize};

use crate::canonical::Canonical;
use crate::hash::{Digest, HashAlgo};

/// The content address of an [`crate::Object`].
///
/// Displayed and serialized as `sos1:<hex>` — the `sos1:` prefix names the
/// address scheme so ids are self-describing in logs and on the wire.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId(Digest);

/// The human/wire prefix identifying an SOS content address.
pub const OBJECT_ID_PREFIX: &str = "sos1:";

impl ObjectId {
    /// Wrap a raw [`Digest`] as an [`ObjectId`].
    #[must_use]
    pub const fn from_digest(d: Digest) -> Self {
        Self(d)
    }

    /// The underlying [`Digest`].
    #[must_use]
    pub const fn digest(&self) -> &Digest {
        &self.0
    }

    /// Compute an id by hashing `canonical` bytes under `domain` with `algo`.
    ///
    /// Callers normally use [`crate::Object::seal`] rather than this directly;
    /// it is public so that non-envelope content (e.g. a side-blob) can be
    /// addressed with the same scheme.
    #[must_use]
    pub fn compute(algo: HashAlgo, domain: &[u8], canonical: &[u8]) -> Self {
        Self(algo.hash(domain, canonical))
    }

    /// Compute an id for any [`Canonical`] value under `domain`.
    #[must_use]
    pub fn of<T: Canonical + ?Sized>(algo: HashAlgo, domain: &[u8], value: &T) -> Self {
        Self::compute(algo, domain, &value.canonical_bytes())
    }

    /// Render as `sos1:<hex>`.
    #[must_use]
    pub fn to_prefixed_hex(&self) -> String {
        format!("{OBJECT_ID_PREFIX}{}", self.0.to_hex())
    }

    /// Parse a `sos1:<hex>` string (the `sos1:` prefix is optional).
    ///
    /// # Errors
    /// Returns [`crate::SosError::InvalidDigestHex`] if the hex body is not a
    /// valid 32-byte digest.
    pub fn parse(s: &str) -> crate::Result<Self> {
        let hex = s.strip_prefix(OBJECT_ID_PREFIX).unwrap_or(s);
        Ok(Self(Digest::from_hex(hex)?))
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{OBJECT_ID_PREFIX}{}", self.0.to_hex())
    }
}

impl fmt::Debug for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectId({}{})", OBJECT_ID_PREFIX, self.0.to_hex())
    }
}

// An `ObjectId` is itself `Canonical` (it appears inside objects, e.g. as a
// parent link), encoded as its raw digest bytes.
impl Canonical for ObjectId {
    fn encode(&self, enc: &mut crate::canonical::CanonicalEncoder) {
        enc.bytes(self.0.as_bytes());
    }
}

impl Serialize for ObjectId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_prefixed_hex())
    }
}

impl<'de> Deserialize<'de> for ObjectId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        ObjectId::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn of_is_deterministic() {
        let a = ObjectId::of(HashAlgo::Sha256, b"dom", &String::from("x"));
        let b = ObjectId::of(HashAlgo::Sha256, b"dom", &String::from("x"));
        assert_eq!(a, b);
    }

    #[test]
    fn different_domain_or_value_changes_id() {
        let base = ObjectId::of(HashAlgo::Sha256, b"dom", &1u64);
        assert_ne!(base, ObjectId::of(HashAlgo::Sha256, b"dom2", &1u64));
        assert_ne!(base, ObjectId::of(HashAlgo::Sha256, b"dom", &2u64));
    }

    #[test]
    fn prefixed_hex_roundtrips() {
        let id = ObjectId::of(HashAlgo::Sha256, b"dom", &7u64);
        let s = id.to_prefixed_hex();
        assert!(s.starts_with(OBJECT_ID_PREFIX));
        assert_eq!(ObjectId::parse(&s).unwrap(), id);
        // The prefix is optional on parse.
        assert_eq!(ObjectId::parse(id.digest().to_hex().as_str()).unwrap(), id);
    }

    #[test]
    fn serde_roundtrips() {
        let id = ObjectId::of(HashAlgo::Sha256, b"dom", &"hello");
        let json = serde_json::to_string(&id).unwrap();
        let back: ObjectId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn ids_are_canonical_as_parent_links() {
        // Two different ids must encode differently when nested.
        let a = ObjectId::of(HashAlgo::Sha256, b"d", &1u64);
        let b = ObjectId::of(HashAlgo::Sha256, b"d", &2u64);
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }
}
