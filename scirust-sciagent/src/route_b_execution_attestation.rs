use scirust_agent_protocol::{
    ExecutionArchitecture, ExecutionArchitectureFamily, ExecutionAttestation,
    ExecutionAttestationError, ExecutionBackendKind, ExecutionReproducibility, Sha256Digest,
};

use crate::execution_attestation::{
    RuntimeExecutionAttestationInputs, build_runtime_execution_attestation,
};

/// Stable semantic identifier for the numeric contract executed by SciAgent's
/// resident Route-B CUDA path: bf16 storage/operands with fp32 accumulation.
pub const ROUTE_B_CUDA_NUMERIC_MODE_V1: &str = "bf16-fp32-accum-v1";

/// Versioned semantic identity of the Route-B CUDA kernel contract attested by
/// this bridge. Kernel changes that alter the execution contract must bump this
/// identifier rather than silently reusing the old attestation identity.
pub const ROUTE_B_CUDA_KERNEL_SEMANTICS_V1: &str = "sciagent.route-b.cuda-kernel-v1";

/// Inputs that remain execution-specific after the invariant Route-B semantics
/// have been fixed.
///
/// Capability and topology bytes must already be canonical compute-profile
/// encodings. `architecture_name` is optional on purpose: callers may attach a
/// structured driver-provided name such as `sm_110`, but this layer never parses
/// a product string or guesses an architecture generation.
pub struct RouteBCudaExecutionAttestationInputs<'a> {
    pub architecture_name: Option<&'a str>,
    pub capability_profile_bytes: &'a [u8],
    pub topology_profile_bytes: &'a [u8],
    pub memory_budget_bytes: Option<u64>,
    pub sampler_semantic_version: Option<&'a str>,
    pub model_sha256: Sha256Digest,
    pub tokenizer_sha256: Sha256Digest,
}

/// Build an execution attestation for SciAgent's resident Route-B CUDA path.
///
/// This helper fixes only facts that are invariant in the current Route-B
/// implementation:
/// - CUDA backend;
/// - logical device ordinal 0 (`CudaChain::new` acquires device 0);
/// - NVIDIA GPU architecture family after successful CUDA acquisition;
/// - bf16 operands/storage with fp32 accumulation;
/// - numerical-equivalence reproducibility rather than a cross-device bit-exact
///   claim.
///
/// Physical capability/topology identity remains supplied by canonical profile
/// bytes. No CUDA context is opened here, and no device name, ISA, memory fabric
/// or compute capability is inferred.
pub fn build_route_b_cuda_execution_attestation(
    inputs: RouteBCudaExecutionAttestationInputs<'_>,
) -> Result<ExecutionAttestation, ExecutionAttestationError> {
    build_runtime_execution_attestation(RuntimeExecutionAttestationInputs {
        backend: ExecutionBackendKind::Cuda,
        device_ordinal: 0,
        architecture: ExecutionArchitecture {
            family: ExecutionArchitectureFamily::NvidiaGpu,
            name: inputs.architecture_name.map(str::to_string),
        },
        capability_profile_bytes: inputs.capability_profile_bytes,
        topology_profile_bytes: inputs.topology_profile_bytes,
        memory_budget_bytes: inputs.memory_budget_bytes,
        numeric_mode: ROUTE_B_CUDA_NUMERIC_MODE_V1,
        reproducibility: ExecutionReproducibility::NumericallyEquivalent,
        kernel_semantic_version: ROUTE_B_CUDA_KERNEL_SEMANTICS_V1,
        sampler_semantic_version: inputs.sampler_semantic_version,
        model_sha256: inputs.model_sha256,
        tokenizer_sha256: inputs.tokenizer_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256_digest;

    fn inputs<'a>(
        capability: &'a [u8],
        topology: &'a [u8],
    ) -> RouteBCudaExecutionAttestationInputs<'a> {
        RouteBCudaExecutionAttestationInputs {
            architecture_name: None,
            capability_profile_bytes: capability,
            topology_profile_bytes: topology,
            memory_budget_bytes: Some(6 * 1024 * 1024 * 1024),
            sampler_semantic_version: Some("resident-sampler-v1"),
            model_sha256: sha256_digest(b"model"),
            tokenizer_sha256: sha256_digest(b"tokenizer"),
        }
    }

    #[test]
    fn route_b_bridge_pins_only_proven_semantic_facts() {
        let attestation = build_route_b_cuda_execution_attestation(inputs(b"cap", b"top")).unwrap();
        let profile = &attestation.profile;

        assert_eq!(profile.backend, ExecutionBackendKind::Cuda);
        assert_eq!(profile.device_ordinal, 0);
        assert_eq!(
            profile.architecture.family,
            ExecutionArchitectureFamily::NvidiaGpu
        );
        assert_eq!(profile.architecture.name, None);
        assert_eq!(profile.numeric_mode, ROUTE_B_CUDA_NUMERIC_MODE_V1);
        assert_eq!(
            profile.reproducibility,
            ExecutionReproducibility::NumericallyEquivalent
        );
        assert_eq!(
            profile.kernel_semantic_version,
            ROUTE_B_CUDA_KERNEL_SEMANTICS_V1
        );
        assert_eq!(attestation.verify(), Ok(()));
    }

    #[test]
    fn structured_architecture_name_is_preserved_without_inference() {
        let mut route = inputs(b"cap", b"top");
        route.architecture_name = Some("sm_110");
        let attestation = build_route_b_cuda_execution_attestation(route).unwrap();

        assert_eq!(
            attestation.profile.architecture.name.as_deref(),
            Some("sm_110")
        );
    }

    #[test]
    fn compute_profile_changes_still_change_route_b_identity() {
        let base = build_route_b_cuda_execution_attestation(inputs(b"cap-a", b"top"))
            .unwrap()
            .profile_sha256;
        let changed = build_route_b_cuda_execution_attestation(inputs(b"cap-b", b"top"))
            .unwrap()
            .profile_sha256;

        assert_ne!(base, changed);
    }
}
