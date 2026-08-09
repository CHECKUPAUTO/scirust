use scirust_agent_protocol::{
    ExecutionArchitecture, ExecutionArchitectureFamily, ExecutionAttestation,
    ExecutionAttestationError, ExecutionBackendKind, ExecutionReproducibility, Sha256Digest,
};

use crate::cuda_decode::{
    CudaDecodeDownMode, CudaDecodeFfnMode, CudaDecodeLmHeadMode, CudaDecodeModes,
};
use crate::execution_attestation::{
    RuntimeExecutionAttestationInputs, build_runtime_execution_attestation,
};

/// Numeric contract of the latency-oriented batch-one CUDA decoder.
pub const CUDA_DECODE_NUMERIC_MODE_V1: &str = "bf16-fp32-accum-v1";

const KERNEL_FUSED_CUBLAS_FUSED: &str = "sciagent.cuda-decode.fused-gemv.cublas-down.fused-argmax-v1";
const KERNEL_FUSED_CUBLAS_FULL: &str = "sciagent.cuda-decode.fused-gemv.cublas-down.full-logits-v1";
const KERNEL_FUSED_TILED_FUSED: &str = "sciagent.cuda-decode.fused-gemv.tiled-down.fused-argmax-v1";
const KERNEL_FUSED_TILED_FULL: &str = "sciagent.cuda-decode.fused-gemv.tiled-down.full-logits-v1";
const KERNEL_CUBLAS_CUBLAS_FUSED: &str = "sciagent.cuda-decode.cublas-ffn.cublas-down.fused-argmax-v1";
const KERNEL_CUBLAS_CUBLAS_FULL: &str = "sciagent.cuda-decode.cublas-ffn.cublas-down.full-logits-v1";
const KERNEL_CUBLAS_TILED_FUSED: &str = "sciagent.cuda-decode.cublas-ffn.tiled-down.fused-argmax-v1";
const KERNEL_CUBLAS_TILED_FULL: &str = "sciagent.cuda-decode.cublas-ffn.tiled-down.full-logits-v1";

/// Facts supplied at the point where an already-acquired [`crate::cuda_decode::CudaDecodeModel`]
/// is bound to its exact model/tokenizer provenance and canonical compute profile.
pub struct CudaDecodeExecutionAttestationInputs<'a> {
    pub architecture_name: Option<&'a str>,
    pub capability_profile_bytes: &'a [u8],
    pub topology_profile_bytes: &'a [u8],
    pub memory_budget_bytes: Option<u64>,
    pub sampler_semantic_version: Option<&'a str>,
    pub model_sha256: Sha256Digest,
    pub tokenizer_sha256: Sha256Digest,
}

/// Return the versioned semantic identity of the exact decode implementation modes.
#[must_use]
pub const fn cuda_decode_kernel_semantic_version(modes: CudaDecodeModes) -> &'static str {
    match (modes.ffn, modes.down, modes.lm_head) {
        (CudaDecodeFfnMode::FusedGemv, CudaDecodeDownMode::CublasLt, CudaDecodeLmHeadMode::FusedArgmax) => KERNEL_FUSED_CUBLAS_FUSED,
        (CudaDecodeFfnMode::FusedGemv, CudaDecodeDownMode::CublasLt, CudaDecodeLmHeadMode::FullLogits) => KERNEL_FUSED_CUBLAS_FULL,
        (CudaDecodeFfnMode::FusedGemv, CudaDecodeDownMode::TiledGemv, CudaDecodeLmHeadMode::FusedArgmax) => KERNEL_FUSED_TILED_FUSED,
        (CudaDecodeFfnMode::FusedGemv, CudaDecodeDownMode::TiledGemv, CudaDecodeLmHeadMode::FullLogits) => KERNEL_FUSED_TILED_FULL,
        (CudaDecodeFfnMode::CublasLt, CudaDecodeDownMode::CublasLt, CudaDecodeLmHeadMode::FusedArgmax) => KERNEL_CUBLAS_CUBLAS_FUSED,
        (CudaDecodeFfnMode::CublasLt, CudaDecodeDownMode::CublasLt, CudaDecodeLmHeadMode::FullLogits) => KERNEL_CUBLAS_CUBLAS_FULL,
        (CudaDecodeFfnMode::CublasLt, CudaDecodeDownMode::TiledGemv, CudaDecodeLmHeadMode::FusedArgmax) => KERNEL_CUBLAS_TILED_FUSED,
        (CudaDecodeFfnMode::CublasLt, CudaDecodeDownMode::TiledGemv, CudaDecodeLmHeadMode::FullLogits) => KERNEL_CUBLAS_TILED_FULL,
    }
}

pub(crate) fn build_cuda_decode_execution_attestation(
    modes: CudaDecodeModes,
    inputs: CudaDecodeExecutionAttestationInputs<'_>,
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
        numeric_mode: CUDA_DECODE_NUMERIC_MODE_V1,
        reproducibility: ExecutionReproducibility::NumericallyEquivalent,
        kernel_semantic_version: cuda_decode_kernel_semantic_version(modes),
        sampler_semantic_version: inputs.sampler_semantic_version,
        model_sha256: inputs.model_sha256,
        tokenizer_sha256: inputs.tokenizer_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256_digest;

    fn inputs() -> CudaDecodeExecutionAttestationInputs<'static> {
        CudaDecodeExecutionAttestationInputs {
            architecture_name: Some("sm_110"),
            capability_profile_bytes: b"capability-profile",
            topology_profile_bytes: b"topology-profile",
            memory_budget_bytes: Some(8 * 1024 * 1024 * 1024),
            sampler_semantic_version: Some("greedy-device-feedback-v1"),
            model_sha256: sha256_digest(b"model"),
            tokenizer_sha256: sha256_digest(b"tokenizer"),
        }
    }

    #[test]
    fn default_i250_modes_have_stable_semantic_identity() {
        assert_eq!(
            cuda_decode_kernel_semantic_version(CudaDecodeModes::default()),
            KERNEL_FUSED_CUBLAS_FUSED
        );
    }

    #[test]
    fn changing_decode_mode_changes_attested_kernel_identity() {
        let base = build_cuda_decode_execution_attestation(CudaDecodeModes::default(), inputs())
            .unwrap();
        let changed = build_cuda_decode_execution_attestation(
            CudaDecodeModes {
                lm_head: CudaDecodeLmHeadMode::FullLogits,
                ..CudaDecodeModes::default()
            },
            inputs(),
        )
        .unwrap();

        assert_ne!(base.profile_sha256, changed.profile_sha256);
        assert_eq!(base.verify(), Ok(()));
        assert_eq!(changed.verify(), Ok(()));
    }
}
