use scirust_agent_protocol::{
    ExecutionArchitecture, ExecutionArchitectureFamily, ExecutionAttestation,
    ExecutionAttestationError, ExecutionBackendKind, ExecutionReproducibility, Sha256Digest,
};
use scirust_gpu::CudaComputeAdapter;

use crate::cuda_decode::{
    CudaDecodeDownMode, CudaDecodeFfnMode, CudaDecodeLmHeadMode, CudaDecodeModel, CudaDecodeModes,
};
use crate::execution_attestation::{
    RuntimeExecutionAttestationInputs, build_runtime_execution_attestation,
};

pub const CUDA_DECODE_NUMERIC_MODE_V1: &str = "bf16-fp32-accum-v1";

// CudaDecodeRuntime::new currently acquires CUDA device zero. Keep that fact
// explicit at the attestation boundary so a separate acquired compute adapter
// cannot attest a different CUDA device for this decode execution.
const CUDA_DECODE_DEVICE_ORDINAL: u32 = 0;

const KERNEL_FUSED_CUBLAS_FUSED: &str =
    "sciagent.cuda-decode.fused-gemv.cublas-down.fused-argmax-v1";
const KERNEL_FUSED_CUBLAS_FULL: &str = "sciagent.cuda-decode.fused-gemv.cublas-down.full-logits-v1";
const KERNEL_FUSED_TILED_FUSED: &str = "sciagent.cuda-decode.fused-gemv.tiled-down.fused-argmax-v1";
const KERNEL_FUSED_TILED_FULL: &str = "sciagent.cuda-decode.fused-gemv.tiled-down.full-logits-v1";
const KERNEL_CUBLAS_CUBLAS_FUSED: &str =
    "sciagent.cuda-decode.cublas-ffn.cublas-down.fused-argmax-v1";
const KERNEL_CUBLAS_CUBLAS_FULL: &str =
    "sciagent.cuda-decode.cublas-ffn.cublas-down.full-logits-v1";
const KERNEL_CUBLAS_TILED_FUSED: &str =
    "sciagent.cuda-decode.cublas-ffn.tiled-down.fused-argmax-v1";
const KERNEL_CUBLAS_TILED_FULL: &str = "sciagent.cuda-decode.cublas-ffn.tiled-down.full-logits-v1";

/// Provenance facts bound to an already-acquired CUDA compute adapter.
///
/// Device ordinal, architecture and canonical hardware/topology bytes are
/// derived from `compute_adapter`; they are no longer caller-provided fields.
pub struct CudaDecodeExecutionAttestationInputs<'a> {
    pub compute_adapter: &'a CudaComputeAdapter,
    pub memory_budget_bytes: Option<u64>,
    pub sampler_semantic_version: Option<&'a str>,
    pub model_sha256: Sha256Digest,
    pub tokenizer_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaDecodeExecutionAttestationError {
    ComputeProfile(String),
    DecodeRuntimeDeviceMismatch {
        decode_ordinal: u32,
        profile_ordinal: u32,
    },
    ExecutionAttestation(ExecutionAttestationError),
}

struct CanonicalCudaExecutionProfile {
    device_ordinal: u32,
    architecture_name: Option<String>,
    capability_profile_bytes: Vec<u8>,
    topology_profile_bytes: Vec<u8>,
}

#[must_use]
pub const fn cuda_decode_kernel_semantic_version(modes: CudaDecodeModes) -> &'static str {
    match (modes.ffn, modes.down, modes.lm_head)
    {
        (
            CudaDecodeFfnMode::FusedGemv,
            CudaDecodeDownMode::CublasLt,
            CudaDecodeLmHeadMode::FusedArgmax,
        ) => KERNEL_FUSED_CUBLAS_FUSED,
        (
            CudaDecodeFfnMode::FusedGemv,
            CudaDecodeDownMode::CublasLt,
            CudaDecodeLmHeadMode::FullLogits,
        ) => KERNEL_FUSED_CUBLAS_FULL,
        (
            CudaDecodeFfnMode::FusedGemv,
            CudaDecodeDownMode::TiledGemv,
            CudaDecodeLmHeadMode::FusedArgmax,
        ) => KERNEL_FUSED_TILED_FUSED,
        (
            CudaDecodeFfnMode::FusedGemv,
            CudaDecodeDownMode::TiledGemv,
            CudaDecodeLmHeadMode::FullLogits,
        ) => KERNEL_FUSED_TILED_FULL,
        (
            CudaDecodeFfnMode::CublasLt,
            CudaDecodeDownMode::CublasLt,
            CudaDecodeLmHeadMode::FusedArgmax,
        ) => KERNEL_CUBLAS_CUBLAS_FUSED,
        (
            CudaDecodeFfnMode::CublasLt,
            CudaDecodeDownMode::CublasLt,
            CudaDecodeLmHeadMode::FullLogits,
        ) => KERNEL_CUBLAS_CUBLAS_FULL,
        (
            CudaDecodeFfnMode::CublasLt,
            CudaDecodeDownMode::TiledGemv,
            CudaDecodeLmHeadMode::FusedArgmax,
        ) => KERNEL_CUBLAS_TILED_FUSED,
        (
            CudaDecodeFfnMode::CublasLt,
            CudaDecodeDownMode::TiledGemv,
            CudaDecodeLmHeadMode::FullLogits,
        ) => KERNEL_CUBLAS_TILED_FULL,
    }
}

fn acquired_profile(
    adapter: &CudaComputeAdapter,
) -> Result<CanonicalCudaExecutionProfile, CudaDecodeExecutionAttestationError> {
    let (device_ordinal, architecture_name, capability_profile_bytes, topology_profile_bytes) =
        adapter.canonical_execution_profile().map_err(|error| {
            CudaDecodeExecutionAttestationError::ComputeProfile(error.to_string())
        })?;

    Ok(CanonicalCudaExecutionProfile {
        device_ordinal,
        architecture_name,
        capability_profile_bytes,
        topology_profile_bytes,
    })
}

fn build_cuda_decode_execution_attestation_from_profile(
    modes: CudaDecodeModes,
    profile: CanonicalCudaExecutionProfile,
    memory_budget_bytes: Option<u64>,
    sampler_semantic_version: Option<&str>,
    model_sha256: Sha256Digest,
    tokenizer_sha256: Sha256Digest,
) -> Result<ExecutionAttestation, CudaDecodeExecutionAttestationError> {
    if profile.device_ordinal != CUDA_DECODE_DEVICE_ORDINAL
    {
        return Err(
            CudaDecodeExecutionAttestationError::DecodeRuntimeDeviceMismatch {
                decode_ordinal: CUDA_DECODE_DEVICE_ORDINAL,
                profile_ordinal: profile.device_ordinal,
            },
        );
    }

    build_runtime_execution_attestation(RuntimeExecutionAttestationInputs {
        backend: ExecutionBackendKind::Cuda,
        device_ordinal: profile.device_ordinal,
        architecture: ExecutionArchitecture {
            family: ExecutionArchitectureFamily::NvidiaGpu,
            name: profile.architecture_name,
        },
        capability_profile_bytes: &profile.capability_profile_bytes,
        topology_profile_bytes: &profile.topology_profile_bytes,
        memory_budget_bytes,
        numeric_mode: CUDA_DECODE_NUMERIC_MODE_V1,
        reproducibility: ExecutionReproducibility::NumericallyEquivalent,
        kernel_semantic_version: cuda_decode_kernel_semantic_version(modes),
        sampler_semantic_version,
        model_sha256,
        tokenizer_sha256,
    })
    .map_err(CudaDecodeExecutionAttestationError::ExecutionAttestation)
}

fn build_cuda_decode_execution_attestation(
    modes: CudaDecodeModes,
    inputs: CudaDecodeExecutionAttestationInputs<'_>,
) -> Result<ExecutionAttestation, CudaDecodeExecutionAttestationError> {
    let profile = acquired_profile(inputs.compute_adapter)?;
    build_cuda_decode_execution_attestation_from_profile(
        modes,
        profile,
        inputs.memory_budget_bytes,
        inputs.sampler_semantic_version,
        inputs.model_sha256,
        inputs.tokenizer_sha256,
    )
}

/// Attestation surface available only from an already-constructed CUDA decode model.
///
/// `CudaDecodeModel::from_model` first acquires its decode CUDA runtime. The
/// additional `CudaComputeAdapter` owns an independently acquired CUDA raw runtime
/// used only for structured discovery; its device ordinal must match the decode
/// runtime before an attestation is emitted.
pub trait CudaDecodeExecutionAttestationExt {
    fn execution_attestation(
        &self,
        modes: CudaDecodeModes,
        inputs: CudaDecodeExecutionAttestationInputs<'_>,
    ) -> Result<ExecutionAttestation, CudaDecodeExecutionAttestationError>;
}

impl CudaDecodeExecutionAttestationExt for CudaDecodeModel {
    fn execution_attestation(
        &self,
        modes: CudaDecodeModes,
        inputs: CudaDecodeExecutionAttestationInputs<'_>,
    ) -> Result<ExecutionAttestation, CudaDecodeExecutionAttestationError> {
        build_cuda_decode_execution_attestation(modes, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256_digest;

    fn profile(device_ordinal: u32) -> CanonicalCudaExecutionProfile {
        CanonicalCudaExecutionProfile {
            device_ordinal,
            architecture_name: Some("sm_110".to_string()),
            capability_profile_bytes: b"canonical-capability-profile".to_vec(),
            topology_profile_bytes: b"canonical-topology-profile".to_vec(),
        }
    }

    fn build(
        modes: CudaDecodeModes,
        device_ordinal: u32,
    ) -> Result<ExecutionAttestation, CudaDecodeExecutionAttestationError> {
        build_cuda_decode_execution_attestation_from_profile(
            modes,
            profile(device_ordinal),
            Some(8 * 1024 * 1024 * 1024),
            Some("greedy-device-feedback-v1"),
            sha256_digest(b"model"),
            sha256_digest(b"tokenizer"),
        )
    }

    #[test]
    fn default_i250_modes_have_stable_semantic_identity() {
        assert_eq!(
            cuda_decode_kernel_semantic_version(CudaDecodeModes::default()),
            KERNEL_FUSED_CUBLAS_FUSED
        );
    }

    #[test]
    fn canonical_profile_drives_device_architecture_and_fingerprint() {
        let attestation = build(CudaDecodeModes::default(), CUDA_DECODE_DEVICE_ORDINAL).unwrap();

        assert_eq!(
            attestation.profile.device_ordinal,
            CUDA_DECODE_DEVICE_ORDINAL
        );
        assert_eq!(
            attestation.profile.architecture.family,
            ExecutionArchitectureFamily::NvidiaGpu
        );
        assert_eq!(
            attestation.profile.architecture.name.as_deref(),
            Some("sm_110")
        );
        assert_eq!(attestation.verify(), Ok(()));
    }

    #[test]
    fn changing_decode_mode_changes_attested_kernel_identity() {
        let base = build(CudaDecodeModes::default(), CUDA_DECODE_DEVICE_ORDINAL).unwrap();
        let changed = build(
            CudaDecodeModes {
                lm_head: CudaDecodeLmHeadMode::FullLogits,
                ..CudaDecodeModes::default()
            },
            CUDA_DECODE_DEVICE_ORDINAL,
        )
        .unwrap();

        assert_ne!(base.profile_sha256, changed.profile_sha256);
        assert_eq!(base.verify(), Ok(()));
        assert_eq!(changed.verify(), Ok(()));
    }

    #[test]
    fn rejects_compute_profile_for_different_cuda_device() {
        assert_eq!(
            build(CudaDecodeModes::default(), CUDA_DECODE_DEVICE_ORDINAL + 1),
            Err(
                CudaDecodeExecutionAttestationError::DecodeRuntimeDeviceMismatch {
                    decode_ordinal: CUDA_DECODE_DEVICE_ORDINAL,
                    profile_ordinal: CUDA_DECODE_DEVICE_ORDINAL + 1,
                }
            )
        );
    }

    #[test]
    fn acquired_cuda_adapter_builds_verifiable_profile_when_available() {
        let adapter = match CudaComputeAdapter::new()
        {
            Ok(adapter) => adapter,
            Err(error) =>
            {
                eprintln!("cuda: {error}; skipping acquired-profile test");
                return;
            },
        };

        let attestation = build_cuda_decode_execution_attestation_from_profile(
            CudaDecodeModes::default(),
            acquired_profile(&adapter).unwrap(),
            Some(8 * 1024 * 1024 * 1024),
            Some("greedy-device-feedback-v1"),
            sha256_digest(b"model"),
            sha256_digest(b"tokenizer"),
        )
        .unwrap();

        assert_eq!(attestation.verify(), Ok(()));
        assert_eq!(
            attestation.profile.device_ordinal,
            CUDA_DECODE_DEVICE_ORDINAL
        );
    }
}
