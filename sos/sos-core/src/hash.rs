//! Content hashing: the [`Digest`] type and the versioned [`HashAlgo`].
//!
//! SOS content addresses are SHA-256 digests, computed over
//! [`crate::canonical`] bytes with an explicit **domain-separation** prefix so
//! that a digest of one kind of thing can never be mistaken for a digest of
//! another (the discipline used by `scirust-provenance` and
//! `scirust-sciagent::CcosLog`). SHA-256 is provided by the pure-Rust `sha2`
//! crate — no FFI.
//!
//! The algorithm is *versioned* ([`HashAlgo`]) rather than hard-coded, so a
//! future addition is an explicit, non-breaking tag rather than a silent
//! change that would invalidate every existing id.

use core::fmt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};

use crate::error::SosError;

/// Length of a [`Digest`] in bytes (SHA-256 = 32).
pub const DIGEST_LEN: usize = 32;

/// A fixed-size content digest (32 bytes / 256 bits).
///
/// Serialized as a lowercase hex string for human-readable interchange, and
/// ordered lexicographically by bytes so digests can be used as stable,
/// deterministic sort keys (SOS relies on this for canonical tie-breaking).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Digest([u8; DIGEST_LEN]);

impl Digest {
    /// Wrap raw digest bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; DIGEST_LEN]) -> Self {
        Self(bytes)
    }

    /// Borrow the raw digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; DIGEST_LEN] {
        &self.0
    }

    /// Render as a lowercase hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(DIGEST_LEN * 2);
        for b in &self.0
        {
            // Two lowercase hex digits per byte; `write!` to a String is
            // infallible, but we avoid the macro/format machinery on the hot
            // path by indexing a static table.
            const HEX: &[u8; 16] = b"0123456789abcdef";
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 0x0f) as usize] as char);
        }
        s
    }

    /// Parse a lowercase-or-uppercase 64-character hex string into a digest.
    ///
    /// # Errors
    /// Returns [`SosError::InvalidDigestHex`] if the string is not exactly
    /// `2 * DIGEST_LEN` hex characters.
    pub fn from_hex(s: &str) -> crate::Result<Self> {
        if s.len() != DIGEST_LEN * 2
        {
            return Err(SosError::InvalidDigestHex(format!(
                "expected {} hex chars, got {}",
                DIGEST_LEN * 2,
                s.len()
            )));
        }
        let mut out = [0u8; DIGEST_LEN];
        let bytes = s.as_bytes();
        for (i, slot) in out.iter_mut().enumerate()
        {
            let hi =
                hex_val(bytes[i * 2]).ok_or_else(|| SosError::InvalidDigestHex(s.to_string()))?;
            let lo = hex_val(bytes[i * 2 + 1])
                .ok_or_else(|| SosError::InvalidDigestHex(s.to_string()))?;
            *slot = (hi << 4) | lo;
        }
        Ok(Self(out))
    }
}

/// Convert one ASCII hex digit to its value, or `None` if not a hex digit.
fn hex_val(c: u8) -> Option<u8> {
    match c
    {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest({})", self.to_hex())
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Digest::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// The content-hash algorithm used to form a [`Digest`].
///
/// Versioned on purpose: the variant is (or will be) part of an object's
/// [`crate::Kind`] so that changing it is an explicit, non-breaking extension
/// rather than a silent invalidation of every existing content address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum HashAlgo {
    /// SHA-256 (the default; pure-Rust `sha2`, matching `scirust-provenance`).
    #[default]
    Sha256,
}

impl HashAlgo {
    /// Hash `data` under a domain-separation `domain` prefix.
    ///
    /// The digest is `SHA256(domain ‖ 0x00 ‖ data)`. The `0x00` separator makes
    /// the boundary between domain and data unambiguous, so two different
    /// `(domain, data)` splits can never produce the same pre-image.
    #[must_use]
    pub fn hash(self, domain: &[u8], data: &[u8]) -> Digest {
        match self
        {
            HashAlgo::Sha256 =>
            {
                let mut h = Sha256::new();
                h.update(domain);
                h.update([0u8]);
                h.update(data);
                let out = h.finalize();
                let mut bytes = [0u8; DIGEST_LEN];
                bytes.copy_from_slice(&out);
                Digest(bytes)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrips() {
        let d = HashAlgo::Sha256.hash(b"sos-test", b"payload");
        let hex = d.to_hex();
        assert_eq!(hex.len(), DIGEST_LEN * 2);
        assert_eq!(Digest::from_hex(&hex).unwrap(), d);
    }

    #[test]
    fn hashing_is_deterministic() {
        let a = HashAlgo::Sha256.hash(b"dom", b"data");
        let b = HashAlgo::Sha256.hash(b"dom", b"data");
        assert_eq!(a, b);
    }

    #[test]
    fn domain_separation_changes_the_digest() {
        let a = HashAlgo::Sha256.hash(b"dom1", b"data");
        let b = HashAlgo::Sha256.hash(b"dom2", b"data");
        assert_ne!(a, b);
    }

    #[test]
    fn separator_prevents_boundary_collision() {
        // Without the 0x00 separator, ("ab","c") and ("a","bc") would collide.
        let a = HashAlgo::Sha256.hash(b"ab", b"c");
        let b = HashAlgo::Sha256.hash(b"a", b"bc");
        assert_ne!(a, b);
    }

    #[test]
    fn bad_hex_is_rejected() {
        assert!(Digest::from_hex("xyz").is_err());
        assert!(Digest::from_hex(&"z".repeat(64)).is_err());
        assert!(matches!(
            Digest::from_hex("00"),
            Err(SosError::InvalidDigestHex(_))
        ));
    }

    #[test]
    fn serde_is_hex_string() {
        let d = HashAlgo::Sha256.hash(b"dom", b"data");
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, format!("\"{}\"", d.to_hex()));
        let back: Digest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }
}
