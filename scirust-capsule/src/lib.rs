//! Deterministic `.scicap` container encoding for SciRust.
//!
//! This crate is deliberately narrower than a runtime or signing layer. It
//! turns a validated [`CapsuleManifestV1`] plus payload bytes into one stable
//! binary representation and verifies that representation on decode.
//!
//! Container v1 layout:
//!
//! ```text
//! +----------------------+-----------------------------------------------+
//! | bytes                | meaning                                       |
//! +======================+===============================================+
//! | 0..8                 | magic: `SCICAP\0\x01`                         |
//! | 8..16                | canonical manifest length, little-endian u64  |
//! | 16..16+manifest_len  | canonical compact UTF-8 JSON manifest         |
//! | remaining bytes      | payload bytes in manifest path order          |
//! +----------------------+-----------------------------------------------+
//! ```
//!
//! Payload paths are not duplicated in the binary envelope: the manifest is
//! the single source of ordering and lengths. Decoding rejects non-canonical
//! manifest JSON, truncation, trailing bytes, size mismatches and SHA-256
//! mismatches. The SHA-256 stored by the schema is the ordinary raw SHA-256 of
//! the payload bytes, not SciRust's domain-separated `scirust-digest` value.
//!
//! File I/O, streaming, signatures, provenance, licensing and execution are
//! intentionally left to later layers.

#![forbid(unsafe_code)]

use core::fmt;
use scirust_capsule_schema::{
    CapsuleManifestV1, CapsulePath, CapsuleSchemaError, PayloadDescriptor, Sha256Hex,
};
use sha2::{Digest as _, Sha256};

/// Binary envelope discriminator for `.scicap` container version 1.
pub const CAPSULE_MAGIC_V1: [u8; 8] = *b"SCICAP\0\x01";

/// Fixed bytes before the canonical manifest.
pub const CAPSULE_HEADER_LEN: usize = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapsuleError {
    Schema(CapsuleSchemaError),
    InvalidMagic,
    TruncatedHeader {
        actual_bytes: usize,
    },
    LengthOverflow(&'static str),
    TruncatedManifest {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    InvalidManifestJson(String),
    NonCanonicalManifest,
    PayloadCountMismatch {
        expected: usize,
        actual: usize,
    },
    PayloadPathMismatch {
        index: usize,
        expected: String,
        actual: String,
    },
    PayloadSizeMismatch {
        path: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
    PayloadDigestMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    TruncatedPayload {
        path: String,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    TrailingBytes(usize),
}

impl fmt::Display for CapsuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(error) => write!(f, "invalid capsule schema: {error}"),
            Self::InvalidMagic => f.write_str("invalid SciCapsule container magic"),
            Self::TruncatedHeader { actual_bytes } => write!(
                f,
                "truncated SciCapsule header: expected {CAPSULE_HEADER_LEN} bytes, got {actual_bytes}"
            ),
            Self::LengthOverflow(what) => write!(f, "{what} does not fit in this address space"),
            Self::TruncatedManifest {
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "truncated capsule manifest: expected {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::InvalidManifestJson(error) => {
                write!(f, "invalid capsule manifest JSON: {error}")
            }
            Self::NonCanonicalManifest => {
                f.write_str("capsule manifest JSON is valid but not in canonical v1 encoding")
            }
            Self::PayloadCountMismatch { expected, actual } => write!(
                f,
                "capsule payload count mismatch: manifest expects {expected}, got {actual}"
            ),
            Self::PayloadPathMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "capsule payload path mismatch at index {index}: expected {expected:?}, got {actual:?}"
            ),
            Self::PayloadSizeMismatch {
                path,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "capsule payload {path:?} size mismatch: expected {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::PayloadDigestMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "capsule payload {path:?} SHA-256 mismatch: expected {expected}, got {actual}"
            ),
            Self::TruncatedPayload {
                path,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "truncated capsule payload {path:?}: expected {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::TrailingBytes(bytes) => {
                write!(f, "capsule contains {bytes} trailing byte(s) after declared payloads")
            }
        }
    }
}

impl std::error::Error for CapsuleError {}

impl From<CapsuleSchemaError> for CapsuleError {
    fn from(error: CapsuleSchemaError) -> Self {
        Self::Schema(error)
    }
}

/// One owned payload bound to a validated portable path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapsulePayload {
    path: CapsulePath,
    bytes: Vec<u8>,
}

impl CapsulePayload {
    #[must_use]
    pub fn new(path: CapsulePath, bytes: Vec<u8>) -> Self {
        Self { path, bytes }
    }

    #[must_use]
    pub fn path(&self) -> &CapsulePath {
        &self.path
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Fully validated in-memory capsule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Capsule {
    manifest: CapsuleManifestV1,
    payloads: Vec<CapsulePayload>,
}

impl Capsule {
    /// Builds a capsule from raw payload bytes.
    ///
    /// Input payload order is irrelevant: payloads are sorted by path before
    /// the manifest is constructed, making the resulting binary deterministic.
    pub fn new(
        name: impl Into<String>,
        entrypoint: CapsulePath,
        mut payloads: Vec<CapsulePayload>,
    ) -> Result<Self, CapsuleError> {
        payloads.sort_by(|left, right| left.path.cmp(&right.path));

        let descriptors = payloads
            .iter()
            .map(descriptor_for_payload)
            .collect::<Result<Vec<_>, _>>()?;
        let manifest = CapsuleManifestV1::new(name, entrypoint, descriptors)?;
        Self::from_parts(manifest, payloads)
    }

    /// Validates an existing manifest against owned payload bytes.
    ///
    /// Unlike [`Self::new`], this function does not reorder payloads: callers
    /// supplying an existing manifest must provide bytes in manifest order.
    pub fn from_parts(
        manifest: CapsuleManifestV1,
        payloads: Vec<CapsulePayload>,
    ) -> Result<Self, CapsuleError> {
        validate_parts(&manifest, &payloads)?;
        Ok(Self { manifest, payloads })
    }

    #[must_use]
    pub fn manifest(&self) -> &CapsuleManifestV1 {
        &self.manifest
    }

    #[must_use]
    pub fn payloads(&self) -> &[CapsulePayload] {
        &self.payloads
    }

    /// Encodes this capsule into the deterministic v1 binary envelope.
    pub fn encode(&self) -> Result<Vec<u8>, CapsuleError> {
        validate_parts(&self.manifest, &self.payloads)?;
        let manifest_bytes = canonical_manifest_json(&self.manifest)?;
        let manifest_len = u64::try_from(manifest_bytes.len())
            .map_err(|_| CapsuleError::LengthOverflow("manifest length"))?;

        let mut total_len = CAPSULE_HEADER_LEN
            .checked_add(manifest_bytes.len())
            .ok_or(CapsuleError::LengthOverflow("capsule length"))?;
        for payload in &self.payloads {
            total_len = total_len
                .checked_add(payload.bytes.len())
                .ok_or(CapsuleError::LengthOverflow("capsule length"))?;
        }

        let mut encoded = Vec::with_capacity(total_len);
        encoded.extend_from_slice(&CAPSULE_MAGIC_V1);
        encoded.extend_from_slice(&manifest_len.to_le_bytes());
        encoded.extend_from_slice(&manifest_bytes);
        for payload in &self.payloads {
            encoded.extend_from_slice(&payload.bytes);
        }
        Ok(encoded)
    }

    /// Decodes and fully verifies an in-memory v1 capsule.
    ///
    /// The input slice already bounds all reads, so this parser performs no
    /// attacker-controlled allocation from an untrusted length field. A later
    /// streaming/file API can add explicit resource limits independently.
    pub fn decode(encoded: &[u8]) -> Result<Self, CapsuleError> {
        if encoded.len() < CAPSULE_HEADER_LEN {
            return Err(CapsuleError::TruncatedHeader {
                actual_bytes: encoded.len(),
            });
        }
        if encoded[..CAPSULE_MAGIC_V1.len()] != CAPSULE_MAGIC_V1 {
            return Err(CapsuleError::InvalidMagic);
        }

        let mut manifest_len_bytes = [0_u8; 8];
        manifest_len_bytes.copy_from_slice(&encoded[8..16]);
        let manifest_len = usize::try_from(u64::from_le_bytes(manifest_len_bytes))
            .map_err(|_| CapsuleError::LengthOverflow("manifest length"))?;
        let manifest_end = CAPSULE_HEADER_LEN
            .checked_add(manifest_len)
            .ok_or(CapsuleError::LengthOverflow("manifest end offset"))?;
        if manifest_end > encoded.len() {
            return Err(CapsuleError::TruncatedManifest {
                expected_bytes: manifest_len,
                actual_bytes: encoded.len() - CAPSULE_HEADER_LEN,
            });
        }

        let manifest_bytes = &encoded[CAPSULE_HEADER_LEN..manifest_end];
        let manifest: CapsuleManifestV1 = serde_json::from_slice(manifest_bytes)
            .map_err(|error| CapsuleError::InvalidManifestJson(error.to_string()))?;
        if canonical_manifest_json(&manifest)?.as_slice() != manifest_bytes {
            return Err(CapsuleError::NonCanonicalManifest);
        }

        let mut cursor = manifest_end;
        let mut payloads = Vec::with_capacity(manifest.payloads().len());
        for descriptor in manifest.payloads() {
            let payload_len = usize::try_from(descriptor.size_bytes)
                .map_err(|_| CapsuleError::LengthOverflow("payload length"))?;
            let payload_end = cursor
                .checked_add(payload_len)
                .ok_or(CapsuleError::LengthOverflow("payload end offset"))?;
            if payload_end > encoded.len() {
                return Err(CapsuleError::TruncatedPayload {
                    path: descriptor.path.to_string(),
                    expected_bytes: payload_len,
                    actual_bytes: encoded.len() - cursor,
                });
            }
            payloads.push(CapsulePayload::new(
                descriptor.path.clone(),
                encoded[cursor..payload_end].to_vec(),
            ));
            cursor = payload_end;
        }

        if cursor != encoded.len() {
            return Err(CapsuleError::TrailingBytes(encoded.len() - cursor));
        }

        Self::from_parts(manifest, payloads)
    }
}

/// Returns the canonical v1 manifest bytes used inside the binary envelope.
///
/// Canonical v1 is the compact UTF-8 JSON produced from the schema's fixed
/// struct field order and strictly ordered payload vector. The schema contains
/// no maps, so there is no unordered object supplied by callers.
pub fn canonical_manifest_json(
    manifest: &CapsuleManifestV1,
) -> Result<Vec<u8>, CapsuleError> {
    manifest.validate()?;
    serde_json::to_vec(manifest)
        .map_err(|error| CapsuleError::InvalidManifestJson(error.to_string()))
}

fn descriptor_for_payload(payload: &CapsulePayload) -> Result<PayloadDescriptor, CapsuleError> {
    let size_bytes = u64::try_from(payload.bytes.len())
        .map_err(|_| CapsuleError::LengthOverflow("payload length"))?;
    let sha256 = Sha256Hex::new(raw_sha256_hex(&payload.bytes))?;
    Ok(PayloadDescriptor::new(
        payload.path.clone(),
        sha256,
        size_bytes,
    ))
}

fn validate_parts(
    manifest: &CapsuleManifestV1,
    payloads: &[CapsulePayload],
) -> Result<(), CapsuleError> {
    manifest.validate()?;
    if manifest.payloads().len() != payloads.len() {
        return Err(CapsuleError::PayloadCountMismatch {
            expected: manifest.payloads().len(),
            actual: payloads.len(),
        });
    }

    for (index, (descriptor, payload)) in manifest
        .payloads()
        .iter()
        .zip(payloads.iter())
        .enumerate()
    {
        if descriptor.path != payload.path {
            return Err(CapsuleError::PayloadPathMismatch {
                index,
                expected: descriptor.path.to_string(),
                actual: payload.path.to_string(),
            });
        }

        let actual_bytes = u64::try_from(payload.bytes.len())
            .map_err(|_| CapsuleError::LengthOverflow("payload length"))?;
        if descriptor.size_bytes != actual_bytes {
            return Err(CapsuleError::PayloadSizeMismatch {
                path: descriptor.path.to_string(),
                expected_bytes: descriptor.size_bytes,
                actual_bytes,
            });
        }

        let actual_digest = raw_sha256_hex(&payload.bytes);
        if descriptor.sha256.as_str() != actual_digest {
            return Err(CapsuleError::PayloadDigestMismatch {
                path: descriptor.path.to_string(),
                expected: descriptor.sha256.to_string(),
                actual: actual_digest,
            });
        }
    }
    Ok(())
}

fn raw_sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(value: &str) -> CapsulePath {
        CapsulePath::new(value).unwrap()
    }

    fn sample_capsule(reverse_input: bool) -> Capsule {
        let run = CapsulePayload::new(path("bin/run"), b"run".to_vec());
        let input = CapsulePayload::new(path("data/input.bin"), b"input".to_vec());
        let payloads = if reverse_input {
            vec![input, run]
        } else {
            vec![run, input]
        };
        Capsule::new("demo", path("bin/run"), payloads).unwrap()
    }

    #[test]
    fn construction_and_encoding_are_order_independent() {
        let forward = sample_capsule(false);
        let reversed = sample_capsule(true);
        assert_eq!(forward, reversed);
        assert_eq!(forward.encode().unwrap(), reversed.encode().unwrap());
        assert_eq!(forward.payloads()[0].path().as_str(), "bin/run");
        assert_eq!(forward.payloads()[1].path().as_str(), "data/input.bin");
    }

    #[test]
    fn canonical_manifest_bytes_are_pinned() {
        let capsule = sample_capsule(false);
        let actual = String::from_utf8(canonical_manifest_json(capsule.manifest()).unwrap()).unwrap();
        let expected = concat!(
            "{\"schema_version\":1,\"name\":\"demo\",\"entrypoint\":\"bin/run\",\"payloads\":[",
            "{\"path\":\"bin/run\",\"sha256\":\"acba25512100f80b56fc3ccd14c65be55d94800cda77585c5f41a887e398f9be\",\"size_bytes\":3},",
            "{\"path\":\"data/input.bin\",\"sha256\":\"c96c6d5be8d08a12e7b5cdc1b207fa6b2430974c86803d8891675e76fd992c20\",\"size_bytes\":5}]}"
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn encode_decode_roundtrip_verifies_payloads() {
        let capsule = sample_capsule(false);
        let encoded = capsule.encode().unwrap();
        assert_eq!(Capsule::decode(&encoded).unwrap(), capsule);
    }

    #[test]
    fn payload_tampering_is_rejected() {
        let capsule = sample_capsule(false);
        let mut encoded = capsule.encode().unwrap();
        *encoded.last_mut().unwrap() ^= 0x01;
        let error = Capsule::decode(&encoded).unwrap_err();
        assert!(matches!(error, CapsuleError::PayloadDigestMismatch { .. }));
    }

    #[test]
    fn truncation_is_rejected_before_integrity_check() {
        let capsule = sample_capsule(false);
        let mut encoded = capsule.encode().unwrap();
        encoded.pop();
        let error = Capsule::decode(&encoded).unwrap_err();
        assert!(matches!(error, CapsuleError::TruncatedPayload { .. }));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let capsule = sample_capsule(false);
        let mut encoded = capsule.encode().unwrap();
        encoded.push(0);
        assert_eq!(
            Capsule::decode(&encoded).unwrap_err(),
            CapsuleError::TrailingBytes(1)
        );
    }

    #[test]
    fn invalid_magic_is_rejected() {
        let capsule = sample_capsule(false);
        let mut encoded = capsule.encode().unwrap();
        encoded[0] = b'X';
        assert_eq!(Capsule::decode(&encoded).unwrap_err(), CapsuleError::InvalidMagic);
    }

    #[test]
    fn valid_but_noncanonical_manifest_json_is_rejected() {
        let capsule = sample_capsule(false);
        let canonical = canonical_manifest_json(capsule.manifest()).unwrap();
        let mut noncanonical = Vec::with_capacity(canonical.len() + 1);
        noncanonical.push(b'{');
        noncanonical.push(b' ');
        noncanonical.extend_from_slice(&canonical[1..]);

        let manifest_len = u64::try_from(noncanonical.len()).unwrap();
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&CAPSULE_MAGIC_V1);
        encoded.extend_from_slice(&manifest_len.to_le_bytes());
        encoded.extend_from_slice(&noncanonical);
        for payload in capsule.payloads() {
            encoded.extend_from_slice(payload.bytes());
        }

        assert_eq!(
            Capsule::decode(&encoded).unwrap_err(),
            CapsuleError::NonCanonicalManifest
        );
    }

    #[test]
    fn from_parts_rejects_manifest_digest_mismatch() {
        let payload = CapsulePayload::new(path("bin/run"), b"run".to_vec());
        let descriptor = PayloadDescriptor::new(
            path("bin/run"),
            Sha256Hex::new("00".repeat(32)).unwrap(),
            3,
        );
        let manifest = CapsuleManifestV1::new("demo", path("bin/run"), vec![descriptor]).unwrap();

        let error = Capsule::from_parts(manifest, vec![payload]).unwrap_err();
        assert!(matches!(error, CapsuleError::PayloadDigestMismatch { .. }));
    }
}
