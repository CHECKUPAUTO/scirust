use scirust_compute::{
    DType, DeviceCapabilities, HardwareCapabilities, MemorySpace, ReproducibilityLevel, SupportLevel,
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

pub(crate) fn hardware_capabilities(capabilities: &DeviceCapabilities) -> HardwareCapabilities {
    let mut hardware = capabilities.hardware_baseline();

    #[cfg(feature = "std")]
    {
        let probed = scirust_compute::probe_host_cpu();
        hardware.architecture = probed.architecture;
        hardware.isa = probed.isa;
    }

    for dtype in KNOWN_DTYPES
    {
        let level = if dtype == DType::F32
        {
            SupportLevel::Supported
        }
        else
        {
            SupportLevel::Unsupported
        };
        hardware.numeric.arithmetic_dtypes.set_support(dtype, level);
        hardware.numeric.accumulation_dtypes.set_support(dtype, level);
    }

    hardware.matrix.accelerated = SupportLevel::Unsupported;

    hardware
        .memory
        .spaces
        .set_support(MemorySpace::Host, SupportLevel::Supported);
    for space in [
        MemorySpace::HostPinned,
        MemorySpace::Device,
        MemorySpace::Unified,
    ]
    {
        hardware
            .memory
            .spaces
            .set_support(space, SupportLevel::Unsupported);
    }
    hardware.memory.coherent_host_device = SupportLevel::Unsupported;
    hardware.memory.unified_addressing = SupportLevel::Unsupported;
    hardware.memory.async_transfers = SupportLevel::Unsupported;

    hardware.execution.async_execution = SupportLevel::Unsupported;
    hardware.execution.ordered_streams = SupportLevel::Supported;
    hardware.execution.subgroup_operations = SupportLevel::Unsupported;
    hardware.execution.atomic_i64 = SupportLevel::Unsupported;

    hardware
        .reproducibility
        .modes
        .set_support(ReproducibilityLevel::Deterministic, SupportLevel::Supported);

    hardware
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_compute::{DeviceId, VectorModel};

    fn legacy_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            device: DeviceId::cpu(),
            name: "scirust-gpu-cpu".into(),
            supported_dtypes: vec![DType::F32],
            max_buffer_bytes: None,
            max_workgroup_size: [1, 1, 1],
            supports_async_execution: false,
        }
    }

    #[test]
    fn reference_cpu_profile_states_only_backend_guarantees() {
        let hardware = hardware_capabilities(&legacy_capabilities());

        assert_eq!(
            hardware
                .numeric
                .arithmetic_dtypes
                .support_level(&DType::F32),
            SupportLevel::Supported
        );
        assert_eq!(
            hardware
                .numeric
                .accumulation_dtypes
                .support_level(&DType::F32),
            SupportLevel::Supported
        );
        assert_eq!(
            hardware
                .numeric
                .arithmetic_dtypes
                .support_level(&DType::F64),
            SupportLevel::Unsupported
        );
        assert_eq!(
            hardware.memory.spaces.support_level(&MemorySpace::Host),
            SupportLevel::Supported
        );
        assert_eq!(
            hardware.memory.spaces.support_level(&MemorySpace::Device),
            SupportLevel::Unsupported
        );
        assert_eq!(hardware.matrix.accelerated, SupportLevel::Unsupported);
        assert_eq!(
            hardware.execution.async_execution,
            SupportLevel::Unsupported
        );
        assert_eq!(hardware.execution.ordered_streams, SupportLevel::Supported);
        assert_eq!(
            hardware
                .reproducibility
                .modes
                .support_level(&ReproducibilityLevel::Deterministic),
            SupportLevel::Supported
        );
        assert_eq!(
            hardware
                .reproducibility
                .modes
                .support_level(&ReproducibilityLevel::BitExact),
            SupportLevel::Unknown
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn reference_cpu_profile_preserves_runtime_host_isa_probe() {
        let hardware = hardware_capabilities(&legacy_capabilities());
        let probed = scirust_compute::probe_host_cpu();

        assert_eq!(hardware.architecture, probed.architecture);
        assert_eq!(hardware.isa, probed.isa);
        assert_ne!(hardware.isa.vector_model, VectorModel::Unknown);
    }
}
