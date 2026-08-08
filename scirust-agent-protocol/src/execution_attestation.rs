use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const EXECUTION_PROFILE_SCHEMA_VERSION: u16 = 1;
const EXECUTION_PROFILE_HASH_DOMAIN: &[u8] = b"scirust.execution-profile.v1\0";
const MAX_SEMANTIC_TEXT_BYTES: usize = 128;

/// SHA-256 digest encoded as 64 lowercase hexadecimal characters.
///
/// Deserialization is intentionally followed by [`ExecutionProfile::validate`]
/// or [`ExecutionAttestation::verify`], so untrusted wire data cannot bypass the
/// canonical lowercase representation required by the profile fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn parse(value: impl Into<String>) -> Result<Self, ExecutionAttestationError> {
        let value = value.into();
        if !is_lower_hex_sha256(&value)
        {
            return Err(ExecutionAttestationError::InvalidSha256);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn from_bytes(bytes: [u8; 32]) -> Self {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(64);
        for byte in bytes
        {
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 0x0f) as usize] as char);
        }
        Self(encoded)
    }

    fn validate(&self) -> Result<(), ExecutionAttestationError> {
        if is_lower_hex_sha256(&self.0)
        {
            Ok(())
        }
        else
        {
            Err(ExecutionAttestationError::InvalidSha256)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBackendKind {
    Reference,
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionArchitectureFamily {
    Unknown,
    X86_64,
    Aarch64,
    RiscV64,
    LoongArch64,
    Wasm32,
    NvidiaGpu,
    AmdGpu,
    IntelGpu,
    AppleGpu,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionArchitecture {
    pub family: ExecutionArchitectureFamily,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionReproducibility {
    Unknown,
    BitExact,
    Deterministic,
    NumericallyEquivalent,
    FastApproximate,
}

/// Semantic execution identity consumed by SciAgent/COGNO-1.
///
/// The profile intentionally contains no ISA-feature list. Backend selection can
/// use low-level capabilities internally, but the attestation records only the
/// semantic execution contract and the fingerprints of the capability/topology
/// snapshots that justified that selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProfile {
    pub schema_version: u16,
    pub backend: ExecutionBackendKind,
    pub device_ordinal: u32,
    pub architecture: ExecutionArchitecture,
    pub capability_profile_sha256: Sha256Digest,
    pub topology_profile_sha256: Sha256Digest,
    pub numeric_mode: String,
    pub reproducibility: ExecutionReproducibility,
    pub kernel_semantic_version: String,
    pub sampler_semantic_version: Option<String>,
    pub model_sha256: Sha256Digest,
    pub tokenizer_sha256: Sha256Digest,
}

impl ExecutionProfile {
    pub fn validate(&self) -> Result<(), ExecutionAttestationError> {
        if self.schema_version != EXECUTION_PROFILE_SCHEMA_VERSION
        {
            return Err(ExecutionAttestationError::UnsupportedSchema(
                self.schema_version,
            ));
        }

        self.capability_profile_sha256.validate()?;
        self.topology_profile_sha256.validate()?;
        self.model_sha256.validate()?;
        self.tokenizer_sha256.validate()?;

        validate_semantic_id("numeric_mode", &self.numeric_mode)?;
        validate_semantic_id("kernel_semantic_version", &self.kernel_semantic_version)?;
        if let Some(version) = &self.sampler_semantic_version
        {
            validate_semantic_id("sampler_semantic_version", version)?;
        }

        if let Some(name) = &self.architecture.name
        {
            validate_architecture_name(name)?;
        }
        if self.architecture.family == ExecutionArchitectureFamily::Other
            && self.architecture.name.is_none()
        {
            return Err(ExecutionAttestationError::OtherArchitectureRequiresName);
        }

        Ok(())
    }

    /// Canonical, versioned byte representation used only for fingerprinting.
    ///
    /// This encoding is independent of serde/JSON formatting: fixed-order scalar
    /// tags, little-endian integers, and length-prefixed UTF-8 text are hashed
    /// under an explicit domain separator.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ExecutionAttestationError> {
        self.validate()?;

        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(EXECUTION_PROFILE_HASH_DOMAIN);
        put_u16(&mut out, self.schema_version);
        out.push(backend_tag(self.backend));
        put_u32(&mut out, self.device_ordinal);
        out.push(architecture_tag(self.architecture.family));
        put_optional_text(&mut out, self.architecture.name.as_deref());
        put_text(&mut out, self.capability_profile_sha256.as_str());
        put_text(&mut out, self.topology_profile_sha256.as_str());
        put_text(&mut out, &self.numeric_mode);
        out.push(reproducibility_tag(self.reproducibility));
        put_text(&mut out, &self.kernel_semantic_version);
        put_optional_text(&mut out, self.sampler_semantic_version.as_deref());
        put_text(&mut out, self.model_sha256.as_str());
        put_text(&mut out, self.tokenizer_sha256.as_str());
        Ok(out)
    }

    pub fn fingerprint(&self) -> Result<Sha256Digest, ExecutionAttestationError> {
        let digest = Sha256::digest(self.canonical_bytes()?);
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        Ok(Sha256Digest::from_bytes(bytes))
    }
}

/// Self-checking execution profile envelope.
///
/// The digest detects accidental or malicious profile mutation. It is not a
/// signature and does not establish who produced the profile; authenticity is a
/// separate trust/provenance concern at the agent protocol layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionAttestation {
    pub profile: ExecutionProfile,
    pub profile_sha256: Sha256Digest,
}

impl ExecutionAttestation {
    pub fn new(profile: ExecutionProfile) -> Result<Self, ExecutionAttestationError> {
        let profile_sha256 = profile.fingerprint()?;
        Ok(Self {
            profile,
            profile_sha256,
        })
    }

    pub fn verify(&self) -> Result<(), ExecutionAttestationError> {
        self.profile.validate()?;
        self.profile_sha256.validate()?;
        if self.profile.fingerprint()? != self.profile_sha256
        {
            return Err(ExecutionAttestationError::DigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionAttestationError {
    UnsupportedSchema(u16),
    InvalidSha256,
    InvalidSemanticId(&'static str),
    InvalidArchitectureName,
    OtherArchitectureRequiresName,
    DigestMismatch,
}

fn validate_semantic_id(
    field: &'static str,
    value: &str,
) -> Result<(), ExecutionAttestationError> {
    let valid = !value.is_empty()
        && value.len() <= MAX_SEMANTIC_TEXT_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b'+' | b':' | b'/')
        });
    if valid
    {
        Ok(())
    }
    else
    {
        Err(ExecutionAttestationError::InvalidSemanticId(field))
    }
}

fn validate_architecture_name(value: &str) -> Result<(), ExecutionAttestationError> {
    let valid = !value.is_empty()
        && value.len() <= MAX_SEMANTIC_TEXT_BYTES
        && value.trim() == value
        && value.bytes().all(|byte| byte.is_ascii_graphic() || byte == b' ');
    if valid
    {
        Ok(())
    }
    else
    {
        Err(ExecutionAttestationError::InvalidArchitectureName)
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_text(out: &mut Vec<u8>, value: &str) {
    let len = u32::try_from(value.len()).expect("validated execution-profile text length fits u32");
    put_u32(out, len);
    out.extend_from_slice(value.as_bytes());
}

fn put_optional_text(out: &mut Vec<u8>, value: Option<&str>) {
    match value
    {
        Some(value) =>
        {
            out.push(1);
            put_text(out, value);
        },
        None => out.push(0),
    }
}

fn backend_tag(value: ExecutionBackendKind) -> u8 {
    match value
    {
        ExecutionBackendKind::Reference => 0,
        ExecutionBackendKind::Cpu => 1,
        ExecutionBackendKind::Wgpu => 2,
        ExecutionBackendKind::Cuda => 3,
    }
}

fn architecture_tag(value: ExecutionArchitectureFamily) -> u8 {
    match value
    {
        ExecutionArchitectureFamily::Unknown => 0,
        ExecutionArchitectureFamily::X86_64 => 1,
        ExecutionArchitectureFamily::Aarch64 => 2,
        ExecutionArchitectureFamily::RiscV64 => 3,
        ExecutionArchitectureFamily::LoongArch64 => 4,
        ExecutionArchitectureFamily::Wasm32 => 5,
        ExecutionArchitectureFamily::NvidiaGpu => 6,
        ExecutionArchitectureFamily::AmdGpu => 7,
        ExecutionArchitectureFamily::IntelGpu => 8,
        ExecutionArchitectureFamily::AppleGpu => 9,
        ExecutionArchitectureFamily::Other => 10,
    }
}

fn reproducibility_tag(value: ExecutionReproducibility) -> u8 {
    match value
    {
        ExecutionReproducibility::Unknown => 0,
        ExecutionReproducibility::BitExact => 1,
        ExecutionReproducibility::Deterministic => 2,
        ExecutionReproducibility::NumericallyEquivalent => 3,
        ExecutionReproducibility::FastApproximate => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn profile() -> ExecutionProfile {
        ExecutionProfile {
            schema_version: EXECUTION_PROFILE_SCHEMA_VERSION,
            backend: ExecutionBackendKind::Cuda,
            device_ordinal: 0,
            architecture: ExecutionArchitecture {
                family: ExecutionArchitectureFamily::NvidiaGpu,
                name: Some("sm_110".to_string()),
            },
            capability_profile_sha256: hash(0x11),
            topology_profile_sha256: hash(0x22),
            numeric_mode: "bf16_tensor_core".to_string(),
            reproducibility: ExecutionReproducibility::Deterministic,
            kernel_semantic_version: "sciagent.decode.v1".to_string(),
            sampler_semantic_version: Some("resident_sampler.v1".to_string()),
            model_sha256: hash(0x33),
            tokenizer_sha256: hash(0x44),
        }
    }

    #[test]
    fn canonical_fingerprint_is_deterministic_and_json_independent() {
        let profile = profile();
        let first = profile.fingerprint().unwrap();
        let json = serde_json::to_string_pretty(&profile).unwrap();
        let decoded: ExecutionProfile = serde_json::from_str(&json).unwrap();
        let second = decoded.fingerprint().unwrap();

        assert_eq!(first, second);
        assert_eq!(profile.canonical_bytes().unwrap(), decoded.canonical_bytes().unwrap());
    }

    #[test]
    fn attestation_detects_profile_mutation() {
        let mut attestation = ExecutionAttestation::new(profile()).unwrap();
        assert_eq!(attestation.verify(), Ok(()));

        attestation.profile.device_ordinal = 1;
        assert_eq!(
            attestation.verify(),
            Err(ExecutionAttestationError::DigestMismatch)
        );
    }

    #[test]
    fn hashes_must_be_canonical_lowercase_sha256() {
        assert_eq!(
            Sha256Digest::parse("AA".repeat(32)),
            Err(ExecutionAttestationError::InvalidSha256)
        );
        assert_eq!(
            Sha256Digest::parse("00".repeat(31)),
            Err(ExecutionAttestationError::InvalidSha256)
        );
    }

    #[test]
    fn low_level_isa_features_are_absent_from_wire_schema() {
        let value = serde_json::to_value(profile()).unwrap();
        let object = value.as_object().unwrap();
        assert!(!object.contains_key("isa"));
        assert!(!object.contains_key("isa_features"));
        assert!(!object.contains_key("vector_model"));
    }

    #[test]
    fn other_architecture_requires_a_semantic_name() {
        let mut profile = profile();
        profile.architecture = ExecutionArchitecture {
            family: ExecutionArchitectureFamily::Other,
            name: None,
        };
        assert_eq!(
            profile.validate(),
            Err(ExecutionAttestationError::OtherArchitectureRequiresName)
        );
    }

    #[test]
    fn unsupported_schema_fails_closed() {
        let mut profile = profile();
        profile.schema_version += 1;
        assert_eq!(
            profile.validate(),
            Err(ExecutionAttestationError::UnsupportedSchema(2))
        );
    }
}
