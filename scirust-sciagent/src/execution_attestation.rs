use scirust_agent_protocol::{
    EXECUTION_PROFILE_SCHEMA_VERSION, ExecutionArchitecture, ExecutionAttestation,
    ExecutionAttestationError, ExecutionBackendKind, ExecutionProfile, ExecutionReproducibility,
    Sha256Digest,
};

use crate::sha256::sha256_hex;

/// Explicit runtime facts required to construct a semantic execution
/// attestation.
///
/// The capability/topology byte slices must come from the canonical v1 encoders
/// in `scirust-compute`. This builder deliberately accepts bytes instead of
/// compute-layer structs so SciAgent's core crate does not gain a new direct
/// dependency edge merely for attestation.
pub struct RuntimeExecutionAttestationInputs<'a> {
    pub backend: ExecutionBackendKind,
    pub device_ordinal: u32,
    pub architecture: ExecutionArchitecture,
    pub capability_profile_bytes: &'a [u8],
    pub topology_profile_bytes: &'a [u8],
    pub memory_budget_bytes: Option<u64>,
    pub numeric_mode: &'a str,
    pub reproducibility: ExecutionReproducibility,
    pub kernel_semantic_version: &'a str,
    pub sampler_semantic_version: Option<&'a str>,
    pub model_sha256: Sha256Digest,
    pub tokenizer_sha256: Sha256Digest,
}

/// Build a self-checking execution attestation from already selected runtime
/// facts.
///
/// This function performs no backend selection, device probing, file I/O, or
/// benchmark. Model/tokenizer hashes are required inputs and must be computed at
/// their load/provenance boundary rather than during token generation.
pub fn build_runtime_execution_attestation(
    inputs: RuntimeExecutionAttestationInputs<'_>,
) -> Result<ExecutionAttestation, ExecutionAttestationError> {
    let profile = ExecutionProfile {
        schema_version: EXECUTION_PROFILE_SCHEMA_VERSION,
        backend: inputs.backend,
        device_ordinal: inputs.device_ordinal,
        architecture: inputs.architecture,
        capability_profile_sha256: digest_canonical_profile(inputs.capability_profile_bytes),
        topology_profile_sha256: digest_canonical_profile(inputs.topology_profile_bytes),
        memory_budget_bytes: inputs.memory_budget_bytes,
        numeric_mode: inputs.numeric_mode.to_string(),
        reproducibility: inputs.reproducibility,
        kernel_semantic_version: inputs.kernel_semantic_version.to_string(),
        sampler_semantic_version: inputs.sampler_semantic_version.map(str::to_string),
        model_sha256: inputs.model_sha256,
        tokenizer_sha256: inputs.tokenizer_sha256,
    };

    ExecutionAttestation::new(profile)
}

/// Convert arbitrary bytes into the protocol's canonical lowercase SHA-256
/// wrapper using SciAgent's architecture-independent FIPS-180-4 implementation.
pub fn sha256_digest(data: &[u8]) -> Sha256Digest {
    Sha256Digest::parse(sha256_hex(data)).expect("SciAgent SHA-256 always emits 64 lowercase hex")
}

fn digest_canonical_profile(bytes: &[u8]) -> Sha256Digest {
    sha256_digest(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_agent_protocol::ExecutionArchitectureFamily;

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn inputs<'a>(
        capability_profile_bytes: &'a [u8],
        topology_profile_bytes: &'a [u8],
    ) -> RuntimeExecutionAttestationInputs<'a> {
        RuntimeExecutionAttestationInputs {
            backend: ExecutionBackendKind::Cuda,
            device_ordinal: 0,
            architecture: ExecutionArchitecture {
                family: ExecutionArchitectureFamily::NvidiaGpu,
                name: Some("sm_110".to_string()),
            },
            capability_profile_bytes,
            topology_profile_bytes,
            memory_budget_bytes: Some(8 * 1024 * 1024 * 1024),
            numeric_mode: "bf16_tensor_core",
            reproducibility: ExecutionReproducibility::Deterministic,
            kernel_semantic_version: "sciagent.decode.v1",
            sampler_semantic_version: Some("resident_sampler.v1"),
            model_sha256: digest(0x33),
            tokenizer_sha256: digest(0x44),
        }
    }

    #[test]
    fn builder_hashes_canonical_profiles_and_verifies() {
        let capability = b"canonical-capability-profile";
        let topology = b"canonical-topology-profile";
        let attestation =
            build_runtime_execution_attestation(inputs(capability, topology)).unwrap();

        assert_eq!(attestation.verify(), Ok(()));
        assert_eq!(
            attestation.profile.capability_profile_sha256,
            sha256_digest(capability)
        );
        assert_eq!(
            attestation.profile.topology_profile_sha256,
            sha256_digest(topology)
        );
    }

    #[test]
    fn capability_or_topology_change_changes_execution_identity() {
        let base = build_runtime_execution_attestation(inputs(b"cap-v1", b"top-v1"))
            .unwrap()
            .profile_sha256;
        let changed_cap = build_runtime_execution_attestation(inputs(b"cap-v2", b"top-v1"))
            .unwrap()
            .profile_sha256;
        let changed_top = build_runtime_execution_attestation(inputs(b"cap-v1", b"top-v2"))
            .unwrap()
            .profile_sha256;

        assert_ne!(base, changed_cap);
        assert_ne!(base, changed_top);
    }

    #[test]
    fn builder_preserves_memory_budget_selection_input() {
        let mut constrained = inputs(b"cap", b"top");
        constrained.memory_budget_bytes = Some(1024);
        let constrained = build_runtime_execution_attestation(constrained).unwrap();

        let mut unconstrained = inputs(b"cap", b"top");
        unconstrained.memory_budget_bytes = None;
        let unconstrained = build_runtime_execution_attestation(unconstrained).unwrap();

        assert_ne!(constrained.profile_sha256, unconstrained.profile_sha256);
    }

    #[test]
    fn invalid_semantic_mode_fails_closed() {
        let mut invalid = inputs(b"cap", b"top");
        invalid.numeric_mode = "contains whitespace";
        assert_eq!(
            build_runtime_execution_attestation(invalid),
            Err(ExecutionAttestationError::InvalidSemanticId("numeric_mode"))
        );
    }
}
