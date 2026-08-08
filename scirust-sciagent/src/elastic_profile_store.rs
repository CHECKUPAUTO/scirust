//! Persistent, compatibility-checked ElasticTokenizer execution profiles.
//!
//! A calibrated profile is a hardware-local optimization artifact, never part
//! of tokenizer semantics. The stored envelope therefore binds the thresholds
//! and kernel choices to both a canonical tokenizer fingerprint and a hardware
//! identity before it can be reused.

use std::fmt;
use std::fs;
use std::path::Path;

use crate::elastic_tokenizer::{BpeKernel, ElasticProfile, ElasticThresholds, ThresholdError, TokenId};
use crate::sha256::sha256_hex;

pub const ELASTIC_PROFILE_SCHEMA_V1: u32 = 1;
pub const CANONICAL_BPE_SEMANTICS_V1: &str = "canonical-rank-v1";

/// Stable identity of the host characteristics used for calibration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ElasticHardwareIdentity {
    pub arch: String,
    pub os: String,
    /// Optional caller-supplied CPU/device discriminator. This may contain a
    /// stable CPU model, topology class, or deployment-defined hardware key.
    pub device: String,
}

impl ElasticHardwareIdentity {
    pub fn new(arch: impl Into<String>, os: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            arch: arch.into(),
            os: os.into(),
            device: device.into(),
        }
    }

    /// Conservative native identity available without platform-specific FFI.
    pub fn native() -> Self {
        Self::new(std::env::consts::ARCH, std::env::consts::OS, "generic")
    }

    pub fn fingerprint(&self) -> String {
        let mut bytes = Vec::new();
        append_len_prefixed(&mut bytes, self.arch.as_bytes());
        append_len_prefixed(&mut bytes, self.os.as_bytes());
        append_len_prefixed(&mut bytes, self.device.as_bytes());
        sha256_hex(&bytes)
    }
}

/// Canonical fingerprint of the ordered BPE merge semantics consumed by the
/// ElasticTokenizer oracle and optimized kernels.
pub fn ordered_merges_fingerprint(merges: &[(TokenId, TokenId, TokenId)]) -> String {
    let mut bytes = Vec::with_capacity(16 + merges.len().saturating_mul(24));
    append_len_prefixed(&mut bytes, CANONICAL_BPE_SEMANTICS_V1.as_bytes());
    bytes.extend_from_slice(&(merges.len() as u64).to_le_bytes());
    for &(left, right, output) in merges
    {
        bytes.extend_from_slice(&(left as u64).to_le_bytes());
        bytes.extend_from_slice(&(right as u64).to_le_bytes());
        bytes.extend_from_slice(&(output as u64).to_le_bytes());
    }
    sha256_hex(&bytes)
}

/// Versioned envelope for one calibrated execution profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredElasticProfile {
    pub schema_version: u32,
    pub bpe_semantics: String,
    pub tokenizer_fingerprint: String,
    pub hardware: ElasticHardwareIdentity,
    pub profile: ElasticProfile,
}

impl StoredElasticProfile {
    pub fn new(
        merges: &[(TokenId, TokenId, TokenId)],
        hardware: ElasticHardwareIdentity,
        profile: ElasticProfile,
    ) -> Self {
        Self {
            schema_version: ELASTIC_PROFILE_SCHEMA_V1,
            bpe_semantics: CANONICAL_BPE_SEMANTICS_V1.to_string(),
            tokenizer_fingerprint: ordered_merges_fingerprint(merges),
            hardware,
            profile,
        }
    }

    /// Verifies that this optimization artifact is safe to reuse for the
    /// supplied tokenizer semantics and hardware identity.
    pub fn verify_for(
        &self,
        merges: &[(TokenId, TokenId, TokenId)],
        hardware: &ElasticHardwareIdentity,
    ) -> Result<(), ProfileStoreError> {
        if self.schema_version != ELASTIC_PROFILE_SCHEMA_V1
        {
            return Err(ProfileStoreError::UnsupportedSchema(self.schema_version));
        }
        if self.bpe_semantics != CANONICAL_BPE_SEMANTICS_V1
        {
            return Err(ProfileStoreError::SemanticVersionMismatch);
        }
        if self.tokenizer_fingerprint != ordered_merges_fingerprint(merges)
        {
            return Err(ProfileStoreError::TokenizerFingerprintMismatch);
        }
        if &self.hardware != hardware
        {
            return Err(ProfileStoreError::HardwareMismatch);
        }
        Ok(())
    }

    pub fn to_json_string(&self) -> Result<String, ProfileStoreError> {
        let thresholds = self.profile.thresholds();
        let kernels = self
            .profile
            .kernels()
            .into_iter()
            .map(kernel_name)
            .collect::<Vec<_>>();
        let value = serde_json::json!({
            "schema_version": self.schema_version,
            "bpe_semantics": self.bpe_semantics,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "hardware": {
                "arch": self.hardware.arch,
                "os": self.hardware.os,
                "device": self.hardware.device,
                "fingerprint": self.hardware.fingerprint(),
            },
            "thresholds": {
                "s_max": thresholds.s_max,
                "m_max": thresholds.m_max,
                "l_max": thresholds.l_max,
                "xl_max": thresholds.xl_max,
                "xxl_max": thresholds.xxl_max,
            },
            "kernels": kernels,
        });
        serde_json::to_string_pretty(&value).map_err(ProfileStoreError::Json)
    }

    pub fn from_json_str(input: &str) -> Result<Self, ProfileStoreError> {
        let value: serde_json::Value = serde_json::from_str(input).map_err(ProfileStoreError::Json)?;
        let schema_version = required_u64(&value, "schema_version")?;
        let schema_version = u32::try_from(schema_version)
            .map_err(|_| ProfileStoreError::InvalidField("schema_version"))?;
        if schema_version != ELASTIC_PROFILE_SCHEMA_V1
        {
            return Err(ProfileStoreError::UnsupportedSchema(schema_version));
        }

        let bpe_semantics = required_str(&value, "bpe_semantics")?.to_string();
        let tokenizer_fingerprint = required_str(&value, "tokenizer_fingerprint")?.to_string();
        let hardware_value = value
            .get("hardware")
            .ok_or(ProfileStoreError::MissingField("hardware"))?;
        let hardware = ElasticHardwareIdentity::new(
            required_str(hardware_value, "arch")?,
            required_str(hardware_value, "os")?,
            required_str(hardware_value, "device")?,
        );
        let stored_hardware_fingerprint = required_str(hardware_value, "fingerprint")?;
        if stored_hardware_fingerprint != hardware.fingerprint()
        {
            return Err(ProfileStoreError::HardwareFingerprintCorrupt);
        }

        let threshold_value = value
            .get("thresholds")
            .ok_or(ProfileStoreError::MissingField("thresholds"))?;
        let thresholds = ElasticThresholds::new(
            required_usize(threshold_value, "s_max")?,
            required_usize(threshold_value, "m_max")?,
            required_usize(threshold_value, "l_max")?,
            required_usize(threshold_value, "xl_max")?,
            required_usize(threshold_value, "xxl_max")?,
        )?;

        let kernel_values = value
            .get("kernels")
            .and_then(serde_json::Value::as_array)
            .ok_or(ProfileStoreError::MissingField("kernels"))?;
        if kernel_values.len() != 6
        {
            return Err(ProfileStoreError::InvalidKernelCount(kernel_values.len()));
        }
        let mut kernels = [BpeKernel::Reference; 6];
        for (slot, raw) in kernels.iter_mut().zip(kernel_values)
        {
            let name = raw
                .as_str()
                .ok_or(ProfileStoreError::InvalidField("kernels"))?;
            *slot = parse_kernel(name)?;
        }

        Ok(Self {
            schema_version,
            bpe_semantics,
            tokenizer_fingerprint,
            hardware,
            profile: ElasticProfile::new(thresholds, kernels),
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ProfileStoreError> {
        fs::write(path, self.to_json_string()?).map_err(ProfileStoreError::Io)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ProfileStoreError> {
        let input = fs::read_to_string(path).map_err(ProfileStoreError::Io)?;
        Self::from_json_str(&input)
    }
}

fn append_len_prefixed(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(bytes);
}

const fn kernel_name(kernel: BpeKernel) -> &'static str {
    match kernel
    {
        BpeKernel::Reference => "reference",
        BpeKernel::TinyScan => "tiny_scan",
        BpeKernel::Indexed => "indexed",
        BpeKernel::Heap => "heap",
    }
}

fn parse_kernel(name: &str) -> Result<BpeKernel, ProfileStoreError> {
    match name
    {
        "reference" => Ok(BpeKernel::Reference),
        "tiny_scan" => Ok(BpeKernel::TinyScan),
        "indexed" => Ok(BpeKernel::Indexed),
        "heap" => Ok(BpeKernel::Heap),
        _ => Err(ProfileStoreError::UnknownKernel(name.to_string())),
    }
}

fn required_str<'a>(
    value: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a str, ProfileStoreError> {
    value
        .get(field)
        .ok_or(ProfileStoreError::MissingField(field))?
        .as_str()
        .ok_or(ProfileStoreError::InvalidField(field))
}

fn required_u64(
    value: &serde_json::Value,
    field: &'static str,
) -> Result<u64, ProfileStoreError> {
    value
        .get(field)
        .ok_or(ProfileStoreError::MissingField(field))?
        .as_u64()
        .ok_or(ProfileStoreError::InvalidField(field))
}

fn required_usize(
    value: &serde_json::Value,
    field: &'static str,
) -> Result<usize, ProfileStoreError> {
    usize::try_from(required_u64(value, field)?)
        .map_err(|_| ProfileStoreError::InvalidField(field))
}

#[derive(Debug)]
pub enum ProfileStoreError {
    Io(std::io::Error),
    Json(serde_json::Error),
    MissingField(&'static str),
    InvalidField(&'static str),
    InvalidKernelCount(usize),
    UnknownKernel(String),
    UnsupportedSchema(u32),
    SemanticVersionMismatch,
    TokenizerFingerprintMismatch,
    HardwareMismatch,
    HardwareFingerprintCorrupt,
    InvalidThresholds(ThresholdError),
}

impl fmt::Display for ProfileStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Io(error) => write!(f, "elastic profile I/O failed: {error}"),
            Self::Json(error) => write!(f, "elastic profile JSON failed: {error}"),
            Self::MissingField(field) => write!(f, "elastic profile missing field `{field}`"),
            Self::InvalidField(field) => write!(f, "elastic profile invalid field `{field}`"),
            Self::InvalidKernelCount(count) => {
                write!(f, "elastic profile must contain six kernels, found {count}")
            },
            Self::UnknownKernel(name) => write!(f, "elastic profile unknown kernel `{name}`"),
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported elastic profile schema version {version}")
            },
            Self::SemanticVersionMismatch => {
                f.write_str("elastic profile BPE semantic version mismatch")
            },
            Self::TokenizerFingerprintMismatch => {
                f.write_str("elastic profile tokenizer fingerprint mismatch")
            },
            Self::HardwareMismatch => f.write_str("elastic profile hardware identity mismatch"),
            Self::HardwareFingerprintCorrupt => {
                f.write_str("elastic profile hardware fingerprint is corrupt")
            },
            Self::InvalidThresholds(error) => write!(f, "elastic profile thresholds invalid: {error}"),
        }
    }
}

impl std::error::Error for ProfileStoreError {}

impl From<ThresholdError> for ProfileStoreError {
    fn from(value: ThresholdError) -> Self {
        Self::InvalidThresholds(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> ElasticProfile {
        ElasticProfile::new(
            ElasticThresholds::new(16, 64, 256, 1024, 4096).unwrap(),
            [
                BpeKernel::TinyScan,
                BpeKernel::TinyScan,
                BpeKernel::Indexed,
                BpeKernel::Indexed,
                BpeKernel::Heap,
                BpeKernel::Heap,
            ],
        )
    }

    #[test]
    fn ordered_merge_fingerprint_depends_on_order_and_content() {
        let a = ordered_merges_fingerprint(&[(1, 2, 3), (3, 4, 5)]);
        let b = ordered_merges_fingerprint(&[(3, 4, 5), (1, 2, 3)]);
        let c = ordered_merges_fingerprint(&[(1, 2, 3), (3, 4, 6)]);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, ordered_merges_fingerprint(&[(1, 2, 3), (3, 4, 5)]));
    }

    #[test]
    fn stored_profile_roundtrips_and_verifies() {
        let merges = [(1, 2, 3), (3, 4, 5)];
        let hardware = ElasticHardwareIdentity::new("aarch64", "linux", "thor-class");
        let stored = StoredElasticProfile::new(&merges, hardware.clone(), sample_profile());
        let json = stored.to_json_string().unwrap();
        let loaded = StoredElasticProfile::from_json_str(&json).unwrap();
        assert_eq!(stored, loaded);
        loaded.verify_for(&merges, &hardware).unwrap();
    }

    #[test]
    fn tokenizer_mismatch_is_rejected() {
        let hardware = ElasticHardwareIdentity::new("x86_64", "linux", "cpu-a");
        let stored = StoredElasticProfile::new(&[(1, 2, 3)], hardware.clone(), sample_profile());
        assert!(matches!(
            stored.verify_for(&[(1, 2, 4)], &hardware),
            Err(ProfileStoreError::TokenizerFingerprintMismatch)
        ));
    }

    #[test]
    fn hardware_mismatch_is_rejected() {
        let merges = [(1, 2, 3)];
        let stored = StoredElasticProfile::new(
            &merges,
            ElasticHardwareIdentity::new("x86_64", "linux", "cpu-a"),
            sample_profile(),
        );
        let other = ElasticHardwareIdentity::new("aarch64", "linux", "cpu-b");
        assert!(matches!(
            stored.verify_for(&merges, &other),
            Err(ProfileStoreError::HardwareMismatch)
        ));
    }

    #[test]
    fn corrupted_hardware_fingerprint_is_rejected_on_load() {
        let merges = [(1, 2, 3)];
        let stored = StoredElasticProfile::new(
            &merges,
            ElasticHardwareIdentity::new("x86_64", "linux", "cpu-a"),
            sample_profile(),
        );
        let mut value: serde_json::Value =
            serde_json::from_str(&stored.to_json_string().unwrap()).unwrap();
        value["hardware"]["fingerprint"] = serde_json::Value::String("bad".to_string());
        let json = serde_json::to_string(&value).unwrap();
        assert!(matches!(
            StoredElasticProfile::from_json_str(&json),
            Err(ProfileStoreError::HardwareFingerprintCorrupt)
        ));
    }
}
