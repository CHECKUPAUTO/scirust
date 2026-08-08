use scirust_compute::{
    Architecture, DType, DeviceCapabilities, HardwareCapabilities, MemorySpace, SupportLevel,
};

const KNOWN_DTYPES: [DType; 13] = [
    DType::Bool,
    DType::U8,
    DType::I8,
    DType::U16,
    DType::I16,
    DType::F16,
    DType::Bf16,
    DType::U32,
    DType::I32,
    DType::F32,
    DType::U64,
    DType::I64,
    DType::F64,
];

const WGSL_BASELINE_DTYPES: [DType; 3] = [DType::U32, DType::I32, DType::F32];

pub(crate) fn hardware_capabilities(capabilities: &DeviceCapabilities) -> HardwareCapabilities {
    let mut hardware = capabilities.hardware_baseline();

    // WGPU is a portable API and may target a discrete GPU, an integrated GPU,
    // a software implementation, or a future backend. The adapter name is not a
    // stable architecture contract, so do not infer a vendor/family from it.
    hardware.architecture = Architecture::unknown();

    for dtype in KNOWN_DTYPES
    {
        let level = if WGSL_BASELINE_DTYPES.contains(&dtype)
        {
            SupportLevel::Supported
        }
        else
        {
            SupportLevel::Unsupported
        };
        hardware.numeric.arithmetic_dtypes.set_support(dtype, level);
        hardware
            .numeric
            .accumulation_dtypes
            .set_support(dtype, level);
    }

    // The current ComputeBackend allocation contract exposes WGPU buffers only
    // as device memory. Staging buffers used internally for readback do not make
    // Host a caller-visible allocation space.
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

    // Physical coherence / unified-addressing properties deliberately remain
    // Unknown. WGPU can sit on both discrete and unified-memory architectures,
    // and this adapter does not retain a reliable hardware topology contract.

    hardware.execution.async_execution = SupportLevel::Supported;
    hardware.execution.ordered_streams = SupportLevel::Supported;

    // The device is requested with no optional WGPU features. The current
    // generic WGSL contract therefore does not expose subgroup or i64 atomic
    // requirements to planner-selected kernels.
    hardware.execution.subgroup_operations = SupportLevel::Unsupported;
    hardware.execution.atomic_i64 = SupportLevel::Unsupported;

    // Do not infer tensor/matrix acceleration or reproducibility guarantees from
    // WGPU itself. Specific kernels can publish stronger semantic contracts
    // separately when they are proven.

    hardware
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_compute::{ArchitectureFamily, DeviceId, DeviceKind};

    fn legacy_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            device: DeviceId::new(DeviceKind::Wgpu, 0),
            name: "scirust-gpu-wgpu: arbitrary-adapter-name".into(),
            supported_dtypes: WGSL_BASELINE_DTYPES.to_vec(),
            max_buffer_bytes: Some(1024),
            max_workgroup_size: [256, 256, 64],
            supports_async_execution: true,
        }
    }

    #[test]
    fn profile_does_not_infer_architecture_from_adapter_name() {
        let hardware = hardware_capabilities(&legacy_capabilities());
        assert_eq!(hardware.architecture.family, ArchitectureFamily::Unknown);
        assert_eq!(hardware.architecture.name, None);
    }

    #[test]
    fn profile_matches_the_current_wgsl_numeric_contract() {
        let hardware = hardware_capabilities(&legacy_capabilities());

        for dtype in WGSL_BASELINE_DTYPES
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
                .support_level(&DType::F64),
            SupportLevel::Unsupported
        );
    }

    #[test]
    fn profile_distinguishes_allocation_contract_from_physical_memory_topology() {
        let hardware = hardware_capabilities(&legacy_capabilities());

        assert_eq!(
            hardware.memory.spaces.support_level(&MemorySpace::Device),
            SupportLevel::Supported
        );
        assert_eq!(
            hardware.memory.spaces.support_level(&MemorySpace::Host),
            SupportLevel::Unsupported
        );
        assert_eq!(hardware.memory.coherent_host_device, SupportLevel::Unknown);
        assert_eq!(hardware.memory.unified_addressing, SupportLevel::Unknown);
        assert_eq!(hardware.memory.async_transfers, SupportLevel::Unknown);
    }

    #[test]
    fn profile_reports_only_execution_semantics_exposed_by_the_adapter() {
        let hardware = hardware_capabilities(&legacy_capabilities());

        assert_eq!(hardware.execution.async_execution, SupportLevel::Supported);
        assert_eq!(hardware.execution.ordered_streams, SupportLevel::Supported);
        assert_eq!(
            hardware.execution.subgroup_operations,
            SupportLevel::Unsupported
        );
        assert_eq!(hardware.execution.atomic_i64, SupportLevel::Unsupported);
        assert_eq!(hardware.matrix.accelerated, SupportLevel::Unknown);
        assert!(hardware.reproducibility.modes.is_empty());
    }
}
