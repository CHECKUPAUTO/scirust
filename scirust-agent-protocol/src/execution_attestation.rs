use serde::{Deserialize, Serialize};

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
    /// Caller-provided memory ceiling that constrained implementation selection.
    /// `None` means no explicit caller budget was applied; `Some(0)` is a real
    /// zero-byte constraint and is intentionally distinct in the fingerprint.
    pub memory_budget_bytes: Option<u64>,
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
        put_optional_u64(&mut out, self.memory_budget_bytes);
        put_text(&mut out, &self.numeric_mode);
        out.push(reproducibility_tag(self.reproducibility));
        put_text(&mut out, &self.kernel_semantic_version);
        put_optional_text(&mut out, self.sampler_semantic_version.as_deref());
        put_text(&mut out, self.model_sha256.as_str());
        put_text(&mut out, self.tokenizer_sha256.as_str());
        Ok(out)
    }

    pub fn fingerprint(&self) -> Result<Sha256Digest, ExecutionAttestationError> {
        Ok(Sha256Digest::from_bytes(sha256(&self.canonical_bytes()?)))
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

fn validate_semantic_id(field: &'static str, value: &str) -> Result<(), ExecutionAttestationError> {
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
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() || byte == b' ');
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

fn put_u64(out: &mut Vec<u8>, value: u64) {
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

fn put_optional_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value
    {
        Some(value) =>
        {
            out.push(1);
            put_u64(out, value);
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

// Self-contained SHA-256 (FIPS 180-4), matching the implementation already used
// by SciAgent's deterministic attestation chain. Keeping the profile fingerprint
// independent from DefaultHasher and platform state makes it stable across Rust
// releases and architectures without adding a new dependency edge to the wire
// protocol crate.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56
    {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut w = [0_u32; 64];
    for block in msg.as_chunks::<64>().0
    {
        for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate()
        {
            w[i] = u32::from_be_bytes(*chunk);
        }
        for i in 16..64
        {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64
        {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0_u8; 32];
    for (i, word) in h.iter().enumerate()
    {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
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
            memory_budget_bytes: Some(8 * 1024 * 1024 * 1024),
            numeric_mode: "bf16_tensor_core".to_string(),
            reproducibility: ExecutionReproducibility::Deterministic,
            kernel_semantic_version: "sciagent.decode.v1".to_string(),
            sampler_semantic_version: Some("resident_sampler.v1".to_string()),
            model_sha256: hash(0x33),
            tokenizer_sha256: hash(0x44),
        }
    }

    #[test]
    fn sha256_matches_nist_vectors() {
        assert_eq!(
            Sha256Digest::from_bytes(sha256(b"")).as_str(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            Sha256Digest::from_bytes(sha256(b"abc")).as_str(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn canonical_fingerprint_is_deterministic_and_json_independent() {
        let profile = profile();
        let first = profile.fingerprint().unwrap();
        let json = serde_json::to_string_pretty(&profile).unwrap();
        let decoded: ExecutionProfile = serde_json::from_str(&json).unwrap();
        let second = decoded.fingerprint().unwrap();

        assert_eq!(first, second);
        assert_eq!(
            profile.canonical_bytes().unwrap(),
            decoded.canonical_bytes().unwrap()
        );
    }

    #[test]
    fn memory_budget_is_part_of_the_execution_identity() {
        let profile = profile();
        let baseline = profile.fingerprint().unwrap();

        let mut unconstrained = profile.clone();
        unconstrained.memory_budget_bytes = None;
        assert_ne!(unconstrained.fingerprint().unwrap(), baseline);

        let mut zero_budget = profile;
        zero_budget.memory_budget_bytes = Some(0);
        assert_ne!(zero_budget.fingerprint().unwrap(), baseline);
        assert_ne!(
            zero_budget.fingerprint().unwrap(),
            unconstrained.fingerprint().unwrap()
        );
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
