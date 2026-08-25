//! Validation-first schema types for SciRust `.scicap` execution capsules.
//!
//! This crate defines the shared, versioned manifest contract only. It does not
//! implement archive I/O, signing, provenance, licensing, or execution. Those
//! concerns belong to higher layers so the schema remains a small leaf crate.
//!
//! The v1 manifest deliberately starts with the minimum immutable bundle
//! contract required by later layers:
//! - one schema version;
//! - one human-readable capsule name;
//! - one validated relative entrypoint path;
//! - a strictly ordered list of payloads, each bound to a lowercase SHA-256 and
//!   an exact byte length.
//!
//! Payload ordering is part of the v1 invariant. Constructors sort by path and
//! deserialization rejects unsorted input, giving serializers a deterministic
//! field/list order without claiming that arbitrary JSON text is a canonical
//! byte encoding.

#![forbid(unsafe_code)]

use core::fmt;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

pub const CAPSULE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapsuleSchemaError {
    UnsupportedSchemaVersion(u16),
    EmptyName,
    EmptyPayloads,
    InvalidPath { value: String, reason: &'static str },
    InvalidSha256(String),
    DuplicatePayloadPath(String),
    UnsortedPayloads,
    MissingEntrypoint(String),
}

impl fmt::Display for CapsuleSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::UnsupportedSchemaVersion(version) =>
            {
                write!(f, "unsupported SciCapsule schema version {version}")
            },
            Self::EmptyName => f.write_str("capsule name must not be empty"),
            Self::EmptyPayloads => f.write_str("capsule must contain at least one payload"),
            Self::InvalidPath { value, reason } =>
            {
                write!(f, "invalid capsule path {value:?}: {reason}")
            },
            Self::InvalidSha256(value) => write!(
                f,
                "invalid SHA-256 {value:?}: expected exactly 64 lowercase hexadecimal characters"
            ),
            Self::DuplicatePayloadPath(path) =>
            {
                write!(f, "duplicate capsule payload path {path:?}")
            },
            Self::UnsortedPayloads =>
            {
                f.write_str("capsule payloads must be strictly ordered by path")
            },
            Self::MissingEntrypoint(path) =>
            {
                write!(f, "capsule entrypoint {path:?} is not present in payloads")
            },
        }
    }
}

impl std::error::Error for CapsuleSchemaError {}

/// Portable path stored in a capsule manifest.
///
/// Paths are relative, forward-slash separated, and may not contain empty,
/// current-directory, or parent-directory components. Backslashes and colons
/// are rejected so one manifest cannot change meaning between Unix and Windows
/// path parsers.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct CapsulePath(String);

impl CapsulePath {
    pub fn new(value: impl Into<String>) -> Result<Self, CapsuleSchemaError> {
        let value = value.into();
        validate_capsule_path(&value)?;
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CapsulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for CapsulePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

/// Strict lowercase hexadecimal SHA-256 interchange value.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Sha256Hex(String);

impl Sha256Hex {
    pub fn new(value: impl Into<String>) -> Result<Self, CapsuleSchemaError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(CapsuleSchemaError::InvalidSha256(value));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256Hex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Sha256Hex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PayloadDescriptor {
    pub path: CapsulePath,
    pub sha256: Sha256Hex,
    pub size_bytes: u64,
}

impl PayloadDescriptor {
    #[must_use]
    pub const fn new(path: CapsulePath, sha256: Sha256Hex, size_bytes: u64) -> Self {
        Self {
            path,
            sha256,
            size_bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CapsuleManifestV1 {
    schema_version: u16,
    name: String,
    entrypoint: CapsulePath,
    payloads: Vec<PayloadDescriptor>,
}

impl CapsuleManifestV1 {
    pub fn new(
        name: impl Into<String>,
        entrypoint: CapsulePath,
        mut payloads: Vec<PayloadDescriptor>,
    ) -> Result<Self, CapsuleSchemaError> {
        payloads.sort_by(|left, right| left.path.cmp(&right.path));
        let manifest = Self {
            schema_version: CAPSULE_SCHEMA_VERSION,
            name: name.into(),
            entrypoint,
            payloads,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), CapsuleSchemaError> {
        if self.schema_version != CAPSULE_SCHEMA_VERSION
        {
            return Err(CapsuleSchemaError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.name.trim().is_empty()
        {
            return Err(CapsuleSchemaError::EmptyName);
        }
        if self.payloads.is_empty()
        {
            return Err(CapsuleSchemaError::EmptyPayloads);
        }

        for pair in self.payloads.windows(2)
        {
            match pair[0].path.cmp(&pair[1].path)
            {
                core::cmp::Ordering::Less =>
                {},
                core::cmp::Ordering::Equal =>
                {
                    return Err(CapsuleSchemaError::DuplicatePayloadPath(
                        pair[0].path.to_string(),
                    ));
                },
                core::cmp::Ordering::Greater =>
                {
                    return Err(CapsuleSchemaError::UnsortedPayloads);
                },
            }
        }

        if !self
            .payloads
            .iter()
            .any(|payload| payload.path == self.entrypoint)
        {
            return Err(CapsuleSchemaError::MissingEntrypoint(
                self.entrypoint.to_string(),
            ));
        }

        Ok(())
    }

    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn entrypoint(&self) -> &CapsulePath {
        &self.entrypoint
    }

    #[must_use]
    pub fn payloads(&self) -> &[PayloadDescriptor] {
        &self.payloads
    }
}

#[derive(Deserialize)]
struct RawCapsuleManifestV1 {
    schema_version: u16,
    name: String,
    entrypoint: CapsulePath,
    payloads: Vec<PayloadDescriptor>,
}

impl<'de> Deserialize<'de> for CapsuleManifestV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawCapsuleManifestV1::deserialize(deserializer)?;
        let manifest = Self {
            schema_version: raw.schema_version,
            name: raw.name,
            entrypoint: raw.entrypoint,
            payloads: raw.payloads,
        };
        manifest.validate().map_err(D::Error::custom)?;
        Ok(manifest)
    }
}

fn validate_capsule_path(value: &str) -> Result<(), CapsuleSchemaError> {
    if value.is_empty()
    {
        return Err(CapsuleSchemaError::InvalidPath {
            value: value.to_string(),
            reason: "path must not be empty",
        });
    }
    if value.starts_with('/')
    {
        return Err(CapsuleSchemaError::InvalidPath {
            value: value.to_string(),
            reason: "path must be relative",
        });
    }
    if value.contains('\\')
    {
        return Err(CapsuleSchemaError::InvalidPath {
            value: value.to_string(),
            reason: "path must use forward slashes",
        });
    }
    if value.contains(':')
    {
        return Err(CapsuleSchemaError::InvalidPath {
            value: value.to_string(),
            reason: "colon is not portable across supported path syntaxes",
        });
    }
    if value.contains('\0')
    {
        return Err(CapsuleSchemaError::InvalidPath {
            value: value.to_string(),
            reason: "path must not contain NUL",
        });
    }
    if value
        .split('/')
        .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(CapsuleSchemaError::InvalidPath {
            value: value.to_string(),
            reason: "path contains a forbidden component",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> Sha256Hex {
        Sha256Hex::new(byte.to_string().repeat(64)).unwrap()
    }

    fn payload(path: &str, byte: char, size_bytes: u64) -> PayloadDescriptor {
        PayloadDescriptor::new(CapsulePath::new(path).unwrap(), digest(byte), size_bytes)
    }

    #[test]
    fn constructor_sorts_payloads_and_roundtrips() {
        let manifest = CapsuleManifestV1::new(
            "demo",
            CapsulePath::new("bin/run").unwrap(),
            vec![
                payload("data/input.bin", 'b', 7),
                payload("bin/run", 'a', 11),
            ],
        )
        .unwrap();

        assert_eq!(manifest.schema_version(), CAPSULE_SCHEMA_VERSION);
        assert_eq!(manifest.payloads()[0].path.as_str(), "bin/run");
        assert_eq!(manifest.payloads()[1].path.as_str(), "data/input.bin");

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: CapsuleManifestV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn deserialize_rejects_unsorted_payloads() {
        let json = r#"{
            "schema_version":1,
            "name":"demo",
            "entrypoint":"bin/run",
            "payloads":[
                {"path":"data/input.bin","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","size_bytes":7},
                {"path":"bin/run","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size_bytes":11}
            ]
        }"#;
        let error = serde_json::from_str::<CapsuleManifestV1>(json).unwrap_err();
        assert!(error.to_string().contains("strictly ordered"));
    }

    #[test]
    fn duplicate_payloads_are_rejected() {
        let entrypoint = CapsulePath::new("bin/run").unwrap();
        let error = CapsuleManifestV1::new(
            "demo",
            entrypoint.clone(),
            vec![
                PayloadDescriptor::new(entrypoint.clone(), digest('a'), 11),
                PayloadDescriptor::new(entrypoint, digest('b'), 12),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CapsuleSchemaError::DuplicatePayloadPath(path) if path == "bin/run"
        ));
    }

    #[test]
    fn entrypoint_must_be_a_payload() {
        let error = CapsuleManifestV1::new(
            "demo",
            CapsulePath::new("bin/missing").unwrap(),
            vec![payload("bin/run", 'a', 11)],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CapsuleSchemaError::MissingEntrypoint(path) if path == "bin/missing"
        ));
    }

    #[test]
    fn path_validation_blocks_nonportable_or_escaping_paths() {
        for path in [
            "",
            "/bin/run",
            "../bin/run",
            "bin/../run",
            "bin//run",
            "bin/./run",
            "bin\\run",
            "C:/bin/run",
        ]
        {
            assert!(CapsulePath::new(path).is_err(), "accepted {path:?}");
        }
        assert!(CapsulePath::new("bin/run").is_ok());
    }

    #[test]
    fn sha256_is_strict_lowercase_hex() {
        assert!(Sha256Hex::new("ab".repeat(32)).is_ok());
        assert!(Sha256Hex::new("AB".repeat(32)).is_err());
        assert!(Sha256Hex::new("g0".repeat(32)).is_err());
        assert!(Sha256Hex::new("0".repeat(63)).is_err());
    }

    #[test]
    fn deserialization_rechecks_schema_version() {
        let json = r#"{
            "schema_version":2,
            "name":"demo",
            "entrypoint":"bin/run",
            "payloads":[
                {"path":"bin/run","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size_bytes":11}
            ]
        }"#;
        let error = serde_json::from_str::<CapsuleManifestV1>(json).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported SciCapsule schema version 2")
        );
    }
}
