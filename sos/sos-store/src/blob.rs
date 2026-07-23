//! [`BlobRef`] — the content address of a large side-payload.
//!
//! Bulk numeric payloads (tensors, datasets, checkpoints) do not belong inline
//! in the object graph; they are stored as content-addressed **blobs** and
//! referenced by hash (RFC-0002 §09.1, safetensors-style). A [`BlobRef`] is the
//! digest of the blob's bytes under a blob-specific domain, so it can never be
//! confused with an [`sos_core::ObjectId`]. It is [`Canonical`], so a domain
//! body may embed a blob reference and have it hashed into the object's id.

use core::fmt;

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Digest, HashAlgo};

/// The human/wire prefix identifying a blob content address.
pub const BLOB_REF_PREFIX: &str = "blob1:";

/// A content address for a stored blob (32-byte digest of its bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlobRef(Digest);

impl BlobRef {
    /// The content address of `bytes` (SHA-256 under the `sos-blob:v1` domain).
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(HashAlgo::default().hash(b"sos-blob:v1", bytes))
    }

    /// Wrap a raw [`Digest`] as a blob reference.
    #[must_use]
    pub const fn from_digest(d: Digest) -> Self {
        Self(d)
    }

    /// The underlying digest.
    #[must_use]
    pub const fn digest(&self) -> &Digest {
        &self.0
    }

    /// Render as `blob1:<hex>`.
    #[must_use]
    pub fn to_prefixed_hex(&self) -> String {
        format!("{BLOB_REF_PREFIX}{}", self.0.to_hex())
    }

    /// Parse a `blob1:<hex>` string (the `blob1:` prefix is optional).
    ///
    /// # Errors
    /// Returns [`sos_core::SosError::InvalidDigestHex`] if the hex body is not a
    /// valid 32-byte digest.
    pub fn parse(s: &str) -> sos_core::Result<Self> {
        let hex = s.strip_prefix(BLOB_REF_PREFIX).unwrap_or(s);
        Ok(Self(Digest::from_hex(hex)?))
    }
}

impl fmt::Display for BlobRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{BLOB_REF_PREFIX}{}", self.0.to_hex())
    }
}

impl fmt::Debug for BlobRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlobRef({}{})", BLOB_REF_PREFIX, self.0.to_hex())
    }
}

impl Canonical for BlobRef {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.bytes(self.0.as_bytes());
    }
}

impl Serialize for BlobRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_prefixed_hex())
    }
}

impl<'de> Deserialize<'de> for BlobRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        BlobRef::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_ref_is_content_addressed() {
        assert_eq!(BlobRef::of(b"abc"), BlobRef::of(b"abc"));
        assert_ne!(BlobRef::of(b"abc"), BlobRef::of(b"abd"));
    }

    #[test]
    fn blob_ref_domain_differs_from_object_domain() {
        // A blob of some bytes must not share an address with an object id
        // computed over the same bytes — different domains guarantee this.
        let blob = BlobRef::of(b"payload");
        let oid = sos_core::ObjectId::compute(HashAlgo::default(), b"sos-obj:X:v1", b"payload");
        assert_ne!(blob.digest(), oid.digest());
    }

    #[test]
    fn prefixed_hex_roundtrips() {
        let r = BlobRef::of(b"data");
        let s = r.to_prefixed_hex();
        assert!(s.starts_with(BLOB_REF_PREFIX));
        assert_eq!(BlobRef::parse(&s).unwrap(), r);
        assert_eq!(BlobRef::parse(r.digest().to_hex().as_str()).unwrap(), r);
    }

    #[test]
    fn serde_roundtrips() {
        let r = BlobRef::of(b"data");
        let json = serde_json::to_string(&r).unwrap();
        let back: BlobRef = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
