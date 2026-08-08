use scirust_compute::{
    Architecture, ArchitectureFamily, DType, DeviceCapabilities, HardwareCapabilities, MemorySpace,
    SupportLevel,
};
use scirust_cuda::CudaDeviceInfo;

const BASELINE_ARITHMETIC_DTYPES: [DType; 3] = [DType::U32, DType::I32, DType::F32];

/// Build the rich hardware profile from facts exposed by the CUDA runtime and
/// the current adapter contract.
///
/// The human-readable device name is deliberately ignored. CUDA itself proves
/// that the compute processor is an NVIDIA GPU, while the architecture name is
/// derived only from the runtime-reported compute capability.
pub(crate) fn hardware_capabilities(
    capabilities: &DeviceCapabilities,
    info: &CudaDeviceInfo,
) -> HardwareCapabilities {
    let mut hardware = capabilities.hardware_baseline();

    hardware.architecture = if info.compute_capability.0 >= 0 && info.compute_capability.1 >= 0
    {
        Architecture::named(
            ArchitectureFamily::NvidiaGpu,
            format!(
                "sm_{}{}",
                info.compute_capability.0, info.compute_capability.1
            ),
        )
    }
    else
    {
        Architecture {
            family: ArchitectureFamily::NvidiaGpu,
            name: None,
        }
    };

    // DeviceCapabilities already preserves the adapter's caller-visible storage
    // dtypes. Do not promote every storage width into a generic arithmetic
    // guarantee: the PTX adapter exposes only the baseline scalar contract here.
    for dtype in BASELINE_ARITHMETIC_DTYPES
    {
        hardware
            .numeric
            .arithmetic_dtypes
            .set_support(dtype, SupportLevel::Supported);
        hardware
            .numeric
            .accumulation_dtypes
            .set_support(dtype, SupportLevel::Supported);
    }

    // CudaComputeAdapter::allocate currently exposes only device allocations.
    // These are API-placement facts, not physical-memory-topology claims.
    hardware
        .memory
        .spaces
        .set_support(MemorySpace::Device, SupportLevel::Supported);
    for space in [
        MemorySpace::Host,
        MemorySpace::HostPinned,
        MemorySpace::Unified,
    ]
    {
        hardware
            .memory
            .spaces
            .set_support(space, SupportLevel::Unsupported);
    }

    // The adapter owns one ordered CUDA stream. Launch returns an explicit
    // event, while read() synchronizes before returning. Optional warp/i64
    // atomic semantics remain unknown until separately exposed and tested.
    hardware.execution.async_execution = SupportLevel::Supported;
    hardware.execution.ordered_streams = SupportLevel::Supported;

    // Matrix acceleration, unified addressing/coherence, independently
    // schedulable transfers and global reproducibility remain Unknown. Compute
    // capability alone is not translated into a stronger generic contract.
    hardware
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_compute::{DeviceId, DeviceKind, ReproducibilityLevel};

    fn capabilities(name: &str) -> DeviceCapabilities {
        DeviceCapabilities {
            device: DeviceId::new(DeviceKind::Cuda, 0),
            name: name.into(),
            supported_dtypes: vec![
                DType::U8,
                DType::I8,
                DType::F16,
                DType::Bf16,
                DType::U32,
                DType::I32,
                DType::F32,
                DType::F64,
            ],
            max_buffer_bytes: Some(8 << 30),
            max_workgroup_size: [1024, 1024, 64],
            supports_async_execution: true,
        }
    }

    fn info(name: &str, compute_capability: (i32, i32)) -> CudaDeviceInfo {
        CudaDeviceInfo {
            ordinal: 0,
            name: name.into(),
            total_memory_bytes: 8 << 30,
            compute_capability,
            max_threads_per_block: 1024,
            max_block_size: [1024, 1024, 64],
            max_grid_size: [2_147_483_647, 65_535, 65_535],
            max_shared_memory_per_block: 49_152,
        }
    }

    #[test]
    fn architecture_uses_cuda_semantics_and_compute_capability_not_names() {
        let hardware = hardware_capabilities(
            &capabilities("pretend-amd-adapter"),
            &info("pretend-apple-gpu", (11, 0)),
        );

        assert_eq!(hardware.architecture.family, ArchitectureFamily::NvidiaGpu);
        assert_eq!(hardware.architecture.name.as_deref(), Some("sm_110"));
    }

    #[test]
    fn implausible_compute_capability_does_not_invent_an_architecture_name() {
        let hardware = hardware_capabilities(
            &capabilities("diagnostic-name"),
            &info("another-diagnostic-name", (-1, 0)),
        );

        assert_eq!(hardware.architecture.family, ArchitectureFamily::NvidiaGpu);
        assert_eq!(hardware.architecture.name, None);
    }

    #[test]
    fn profile_keeps_storage_broad_but_arithmetic_conservative() {
        let hardware = hardware_capabilities(&capabilities("cuda"), &info("cuda", (9, 0)));

        assert_eq!(
            hardware.numeric.storage_dtypes.support_level(&DType::F16),
            SupportLevel::Supported
        );
        for dtype in BASELINE_ARITHMETIC_DTYPES
        {
            assert_eq!(
                hardware.numeric.arithmetic_dtypes.support_level(&dtype),
                SupportLevel::Supported
            );
            assert_eq!(
                hardware.numeric.accumulation_dtypes.support_level(&dtype),
                SupportLevel::Supported
            );
        }
        assert_eq!(
            hardware
                .numeric
                .arithmetic_dtypes
                .support_level(&DType::F16),
            SupportLevel::Unknown
        );
        assert_eq!(
            hardware
                .numeric
                .accumulation_dtypes
                .support_level(&DType::Bf16),
            SupportLevel::Unknown
        );
    }

    #[test]
    fn profile_distinguishes_adapter_memory_contract_from_physical_topology() {
        let hardware = hardware_capabilities(&capabilities("cuda"), &info("cuda", (9, 0)));

        assert_eq!(
            hardware.memory.spaces.support_level(&MemorySpace::Device),
            SupportLevel::Supported
        );
        for space in [
            MemorySpace::Host,
            MemorySpace::HostPinned,
            MemorySpace::Unified,
        ]
        {
            assert_eq!(
                hardware.memory.spaces.support_level(&space),
                SupportLevel::Unsupported
            );
        }
        assert_eq!(hardware.memory.coherent_host_device, SupportLevel::Unknown);
        assert_eq!(hardware.memory.unified_addressing, SupportLevel::Unknown);
        assert_eq!(hardware.memory.async_transfers, SupportLevel::Unknown);
    }

    #[test]
    fn profile_reports_only_proven_execution_and_acceleration_semantics() {
        let hardware = hardware_capabilities(&capabilities("cuda"), &info("cuda", (9, 0)));

        assert_eq!(hardware.execution.async_execution, SupportLevel::Supported);
        assert_eq!(hardware.execution.ordered_streams, SupportLevel::Supported);
        assert_eq!(
            hardware.execution.subgroup_operations,
            SupportLevel::Unknown
        );
        assert_eq!(hardware.execution.atomic_i64, SupportLevel::Unknown);
        assert_eq!(hardware.matrix.accelerated, SupportLevel::Unknown);
        assert_eq!(
            hardware
                .reproducibility
                .support_level(ReproducibilityLevel::Deterministic),
            SupportLevel::Unknown
        );
    }
}
