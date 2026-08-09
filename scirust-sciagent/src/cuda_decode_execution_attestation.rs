use scirust_agent_protocol::{
    ExecutionArchitecture, ExecutionArchitectureFamily, ExecutionAttestation,
    ExecutionAttestationError, ExecutionBackendKind, ExecutionReproducibility, Sha256Digest,
};
use scirust_compute::{
    ArchitectureFamily, DeviceKind, HardwareCapabilities, ProfileEncodingError, SystemTopology,
    canonical_hardware_profile_bytes, canonical_topology_profile_bytes,
};

use crate::cuda_decode::{
    CudaDecodeDownMode, CudaDecodeFfnMode, CudaDecodeLmHeadMode, CudaDecodeModel, CudaDecodeModes,
};
use crate::execution_attestation::{
    RuntimeExecutionAttestationInputs, build_runtime_execution_attestation,
};

pub const CUDA_DECODE_NUMERIC_MODE_V1: &str = "bf16-fp32-accum-v1";

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

/// Structured compute facts bound to one already-acquired CUDA decode runtime.
///
/// Capability and topology fingerprints are always derived from SciRust's canonical
/// compute encodings. Callers cannot inject detached byte strings, a separate device
/// ordinal, or a second architecture label into the execution profile.
pub struct CudaDecodeExecutionAttestationInputs<'a> {
    pub hardware: &'a HardwareCapabilities,
    pub topology: &'a SystemTopology,
    pub memory_budget_bytes: Option<u64>,
    pub sampler_semantic_version: Option<&'a str>,
    pub model_sha256: Sha256Digest,
    pub tokenizer_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaDecodeExecutionAttestationError {
    NonCudaDevice(DeviceKind),
    NonNvidiaArchitecture(ArchitectureFamily),
    DeviceMissingFromTopology,
    ProfileEncoding(ProfileEncodingError),
    ExecutionAttestation(ExecutionAttestationError),
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

fn build_cuda_decode_execution_attestation(
    modes: CudaDecodeModes,
    inputs: CudaDecodeExecutionAttestationInputs<'_>,
) -> Result<ExecutionAttestation, CudaDecodeExecutionAttestationError> {
    if inputs.hardware.device.kind() != DeviceKind::Cuda
    {
        return Err(CudaDecodeExecutionAttestationError::NonCudaDevice(
            inputs.hardware.device.kind(),
        ));
    }
    if inputs.hardware.architecture.family != ArchitectureFamily::NvidiaGpu
    {
        return Err(
            CudaDecodeExecutionAttestationError::NonNvidiaArchitecture(
                inputs.hardware.architecture.family,
            ),
        );
    }
    if !inputs
        .topology
        .nodes
        .iter()
        .any(|node| node.device == Some(inputs.hardware.device))
    {
        return Err(CudaDecodeExecutionAttestationError::DeviceMissingFromTopology);
    }

    let capability_profile_bytes = canonical_hardware_profile_bytes(inputs.hardware)
        .map_err(CudaDecodeExecutionAttestationError::ProfileEncoding)?;
    let topology_profile_bytes = canonical_topology_profile_bytes(inputs.topology)
        .map_err(CudaDecodeExecutionAttestationError::ProfileEncoding)?;

    build_runtime_execution_attestation(RuntimeExecutionAttestationInputs {
        backend: ExecutionBackendKind::Cuda,
        device_ordinal: inputs.hardware.device.ordinal(),
        architecture: ExecutionArchitecture {
            family: ExecutionArchitectureFamily::NvidiaGpu,
            name: inputs.hardware.architecture.name.clone(),
        },
        capability_profile_bytes: &capability_profile_bytes,
        topology_profile_bytes: &topology_profile_bytes,
        memory_budget_bytes: inputs.memory_budget_bytes,
        numeric_mode: CUDA_DECODE_NUMERIC_MODE_V1,
        reproducibility: ExecutionReproducibility::NumericallyEquivalent,
        kernel_semantic_version: cuda_decode_kernel_semantic_version(modes),
        sampler_semantic_version: inputs.sampler_semantic_version,
        model_sha256: inputs.model_sha256,
        tokenizer_sha256: inputs.tokenizer_sha256,
    })
    .map_err(CudaDecodeExecutionAttestationError::ExecutionAttestation)
}

/// Attestation surface available only from an already-constructed CUDA decode model.
///
/// `CudaDecodeModel::from_model` returns `Some` only after `CudaDecodeRuntime::new`
/// has acquired the CUDA execution context. Requiring `&self` here prevents callers
/// from producing an I250 runtime attestation without first acquiring that runtime.
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
    use scirust_compute::{
        Architecture, DeviceCapabilities, DeviceId, DType, TopologyNode, TopologyNodeId,
        TopologyNodeKind,
    };

    fn compute_profiles() -> (HardwareCapabilities, SystemTopology) {
        let device = DeviceId::new(DeviceKind::Cuda, 3);
        let capabilities = DeviceCapabilities {
            device,
            name: "synthetic-cuda".to_string(),
            supported_dtypes: vec![DType::Bf16],
            max_buffer_bytes: Some(8 * 1024 * 1024 * 1024),
            max_workgroup_size: [1024, 1024, 64],
            supports_async_execution: true,
        };
        let mut hardware = capabilities.hardware_baseline();
        hardware.architecture = Architecture::named(ArchitectureFamily::NvidiaGpu, "sm_110");

        let mut accelerator = TopologyNode::new(TopologyNodeId::new(7), TopologyNodeKind::Accelerator);
        accelerator.device = Some(device);
        accelerator.name = Some("synthetic-cuda".to_string());
        let topology = SystemTopology {
            nodes: vec![accelerator],
            links: Vec::new(),
        };
        (hardware, topology)
    }

    fn inputs(
        hardware: &HardwareCapabilities,
        topology: &SystemTopology,
    ) -> CudaDecodeExecutionAttestationInputs<'_> {
        CudaDecodeExecutionAttestationInputs {
            hardware,
            topology,
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
    fn canonical_compute_profiles_drive_architecture_device_and_fingerprints() {
        let (hardware, topology) = compute_profiles();
        let attestation = build_cuda_decode_execution_attestation(
            CudaDecodeModes::default(),
            inputs(&hardware, &topology),
        )
        .unwrap();

        assert_eq!(attestation.profile.device_ordinal, 3);
        assert_eq!(
            attestation.profile.architecture.family,
            ExecutionArchitectureFamily::NvidiaGpu
        );
        assert_eq!(attestation.profile.architecture.name.as_deref(), Some("sm_110"));
        assert_eq!(attestation.verify(), Ok(()));
    }

    #[test]
    fn changing_decode_mode_changes_attested_kernel_identity() {
        let (hardware, topology) = compute_profiles();
        let base = build_cuda_decode_execution_attestation(
            CudaDecodeModes::default(),
            inputs(&hardware, &topology),
        )
        .unwrap();
        let changed = build_cuda_decode_execution_attestation(
            CudaDecodeModes {
                lm_head: CudaDecodeLmHeadMode::FullLogits,
                ..CudaDecodeModes::default()
            },
            inputs(&hardware, &topology),
        )
        .unwrap();

        assert_ne!(base.profile_sha256, changed.profile_sha256);
        assert_eq!(base.verify(), Ok(()));
        assert_eq!(changed.verify(), Ok(()));
    }

    #[test]
    fn rejects_detached_non_cuda_compute_profile() {
        let (mut hardware, topology) = compute_profiles();
        hardware.device = DeviceId::cpu();
        assert_eq!(
            build_cuda_decode_execution_attestation(
                CudaDecodeModes::default(),
                inputs(&hardware, &topology),
            ),
            Err(CudaDecodeExecutionAttestationError::NonCudaDevice(
                DeviceKind::Cpu
            ))
        );
    }

    #[test]
    fn rejects_topology_that_does_not_contain_attested_device() {
        let (hardware, mut topology) = compute_profiles();
        topology.nodes[0].device = Some(DeviceId::new(DeviceKind::Cuda, 4));
        assert_eq!(
            build_cuda_decode_execution_attestation(
                CudaDecodeModes::default(),
                inputs(&hardware, &topology),
            ),
            Err(CudaDecodeExecutionAttestationError::DeviceMissingFromTopology)
        );
    }
}
