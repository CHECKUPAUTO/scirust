use scirust_compute::{
    AcceleratorTopologyDescriptor, AcceleratorTopologyProvider, Architecture, ArchitectureFamily,
    DType, DeviceCapabilities, HardwareCapabilities, MemoryDomainDescriptor, MemorySpace,
    SupportLevel, SystemTopology, augment_accelerator_topology, canonical_hardware_profile_bytes,
    canonical_topology_profile_bytes,
};
use scirust_cuda::CudaDeviceInfo;

const PROVEN_ARITHMETIC_DTYPES: [DType; 1] = [DType::U32];

pub(crate) fn hardware_capabilities(
    capabilities: &DeviceCapabilities,
    info: &CudaDeviceInfo,
) -> HardwareCapabilities {
    let mut hardware = capabilities.hardware_baseline();

    // Acquiring a CUDA context proves the accelerator belongs to NVIDIA's CUDA
    // device family. Compute capability is a structured driver query, so it is
    // suitable as an architecture name; the human-readable device name is not
    // parsed for capability decisions.
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

    // The generic CudaComputeAdapter has an active PTX execution test for U32.
    // Do not promote I32/F32 merely because CUDA or a separate CUDA adapter can
    // execute them: rich planner guarantees belong to this concrete adapter and
    // remain Unknown until this same path has an explicit execution proof.
    for dtype in PROVEN_ARITHMETIC_DTYPES
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

    // Caller-visible allocation currently exposes device memory only. This does
    // not imply anything about the machine's physical coherence or unified-memory
    // topology (notably on integrated CUDA systems), so those properties remain
    // Unknown.
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

    hardware.execution.async_execution = SupportLevel::Supported;
    hardware.execution.ordered_streams = SupportLevel::Supported;

    // Warp/subgroup instructions, i64 atomics, matrix acceleration and global
    // reproducibility modes are not inferred from CUDA presence or compute
    // capability alone. A planner may only rely on them after a concrete backend
    // path publishes and tests those semantic guarantees.

    hardware
}

pub(crate) fn topology_descriptor(
    capabilities: &DeviceCapabilities,
    info: &CudaDeviceInfo,
) -> AcceleratorTopologyDescriptor {
    let mut descriptor = AcceleratorTopologyDescriptor::new(capabilities.device);
    descriptor.name = Some(capabilities.name.clone());
    descriptor.memory = Some(MemoryDomainDescriptor {
        space: MemorySpace::Device,
        capacity_bytes: u64::try_from(info.total_memory_bytes).ok(),
        host_addressable: SupportLevel::Unknown,
    });
    descriptor
}

impl super::CudaComputeAdapter {
    /// Return the canonical execution identity derived from this acquired CUDA runtime.
    ///
    /// The tuple is `(device_ordinal, architecture_name, hardware_profile_bytes,
    /// topology_profile_bytes)`. All fields originate from this adapter's driver-backed
    /// runtime and SciRust's versioned canonical compute encoders.
    pub fn canonical_execution_profile(
        &self,
    ) -> scirust_compute::ComputeResult<(u32, Option<String>, Vec<u8>, Vec<u8>)> {
        let hardware = hardware_capabilities(self.capabilities(), self.runtime().device_info());
        let hardware_bytes = canonical_hardware_profile_bytes(&hardware).map_err(|_| {
            scirust_compute::ComputeError::InvalidArgument("invalid canonical CUDA hardware profile")
        })?;

        let descriptor = topology_descriptor(self.capabilities(), self.runtime().device_info());
        let mut topology = SystemTopology::default();
        augment_accelerator_topology(&mut topology, descriptor).map_err(|_| {
            scirust_compute::ComputeError::InvalidArgument("invalid canonical CUDA topology profile")
        })?;
        let topology_bytes = canonical_topology_profile_bytes(&topology).map_err(|_| {
            scirust_compute::ComputeError::InvalidArgument("invalid canonical CUDA topology profile")
        })?;

        Ok((
            self.capabilities().device.ordinal(),
            hardware.architecture.name,
            hardware_bytes,
            topology_bytes,
        ))
    }
}

impl AcceleratorTopologyProvider for super::CudaComputeAdapter {
    fn accelerator_topology_descriptor(&self) -> AcceleratorTopologyDescriptor {
        topology_descriptor(self.capabilities(), self.runtime().device_info())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_compute::{DeviceId, DeviceKind};

    fn legacy_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            device: DeviceId::new(DeviceKind::Cuda, 0),
            name: "scirust-gpu-cuda: synthetic (sm_90)".into(),
            supported_dtypes: vec![
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
            ],
            max_buffer_bytes: Some(16 * 1024 * 1024),
            max_workgroup_size: [1024, 1024, 64],
            supports_async_execution: true,
        }
    }

    fn device_info() -> CudaDeviceInfo {
        CudaDeviceInfo {
            ordinal: 0,
            name: "Synthetic NVIDIA GPU".into(),
            total_memory_bytes: 16 * 1024 * 1024,
            compute_capability: (9, 0),
            max_threads_per_block: 1024,
            max_block_size: [1024, 1024, 64],
            max_grid_size: [2_147_483_647, 65_535, 65_535],
            max_shared_memory_per_block: 48 * 1024,
        }
    }

    #[test]
    fn profile_uses_structured_cuda_architecture_identity() {
        let hardware = hardware_capabilities(&legacy_capabilities(), &device_info());

        assert_eq!(hardware.architecture.family, ArchitectureFamily::NvidiaGpu);
        assert_eq!(hardware.architecture.name.as_deref(), Some("sm_90"));
    }

    #[test]
    fn profile_does_not_promote_storage_dtypes_to_unproven_arithmetic() {
        let hardware = hardware_capabilities(&legacy_capabilities(), &device_info());

        for dtype in PROVEN_ARITHMETIC_DTYPES
        {
            assert_eq!(
                hardware.numeric.arithmetic_dtypes.support_level(&dtype),
                SupportLevel::Supported
            );
        }
        for dtype in [DType::I32, DType::F32, DType::Bf16]
        {
            assert_eq!(
                hardware.numeric.arithmetic_dtypes.support_level(&dtype),
                SupportLevel::Unknown
            );
        }
        assert_eq!(
            hardware
                .numeric
                .accumulation_dtypes
                .support_level(&DType::F64),
            SupportLevel::Unknown
        );
    }

    #[test]
    fn profile_separates_cuda_allocation_space_from_physical_memory_topology() {
        let hardware = hardware_capabilities(&legacy_capabilities(), &device_info());

        assert_eq!(
            hardware.memory.spaces.support_level(&MemorySpace::Device),
            SupportLevel::Supported
        );
        assert_eq!(
            hardware.memory.spaces.support_level(&MemorySpace::Unified),
            SupportLevel::Unsupported
        );
        assert_eq!(hardware.memory.coherent_host_device, SupportLevel::Unknown);
        assert_eq!(hardware.memory.unified_addressing, SupportLevel::Unknown);
        assert_eq!(hardware.memory.async_transfers, SupportLevel::Unknown);
    }

    #[test]
    fn profile_keeps_unproven_acceleration_and_reproducibility_unknown() {
        let hardware = hardware_capabilities(&legacy_capabilities(), &device_info());

        assert_eq!(hardware.execution.async_execution, SupportLevel::Supported);
        assert_eq!(hardware.execution.ordered_streams, SupportLevel::Supported);
        assert_eq!(
            hardware.execution.subgroup_operations,
            SupportLevel::Unknown
        );
        assert_eq!(hardware.execution.atomic_i64, SupportLevel::Unknown);
        assert_eq!(hardware.matrix.accelerated, SupportLevel::Unknown);
        assert!(hardware.reproducibility.modes.is_empty());
    }

    #[test]
    fn topology_profile_uses_driver_memory_capacity_without_claiming_host_access() {
        let capabilities = legacy_capabilities();
        let info = device_info();
        let descriptor = topology_descriptor(&capabilities, &info);
        let memory = descriptor.memory.expect("logical CUDA device memory");

        assert_eq!(descriptor.device, capabilities.device);
        assert_eq!(descriptor.name.as_deref(), Some(capabilities.name.as_str()));
        assert_eq!(memory.space, MemorySpace::Device);
        assert_eq!(
            memory.capacity_bytes,
            u64::try_from(info.total_memory_bytes).ok()
        );
        assert_eq!(memory.host_addressable, SupportLevel::Unknown);
    }
}
