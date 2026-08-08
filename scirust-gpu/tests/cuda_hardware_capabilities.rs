#![cfg(feature = "cuda")]

use scirust_compute::{
    ArchitectureFamily, ComputeBackend, DType, DeviceKind, MemorySpace, SupportLevel,
};
use scirust_gpu::CudaComputeAdapter;

fn adapter_or_skip() -> Option<CudaComputeAdapter> {
    match CudaComputeAdapter::new()
    {
        Ok(adapter) => Some(adapter),
        Err(error) if std::env::var_os("SCIRUST_REQUIRE_CUDA").is_some() =>
        {
            panic!("SCIRUST_REQUIRE_CUDA is set but CUDA adapter acquisition failed: {error}")
        },
        Err(error) =>
        {
            eprintln!("cuda: {error}; skipping rich-hardware-profile integration test");
            None
        },
    }
}

#[test]
fn runtime_cuda_adapter_publishes_the_conservative_rich_profile() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let hardware = ComputeBackend::hardware_capabilities(&adapter);

    assert_eq!(hardware.device.kind(), DeviceKind::Cuda);
    assert_eq!(hardware.architecture.family, ArchitectureFamily::NvidiaGpu);
    let architecture = hardware
        .architecture
        .name
        .as_deref()
        .expect("a real CUDA runtime reports a non-negative compute capability");
    assert!(architecture.starts_with("sm_"));

    assert_eq!(
        hardware.numeric.storage_dtypes.support_level(&DType::F16),
        SupportLevel::Supported
    );
    assert_eq!(
        hardware.numeric.arithmetic_dtypes.support_level(&DType::F32),
        SupportLevel::Supported
    );
    assert_eq!(
        hardware.numeric.arithmetic_dtypes.support_level(&DType::F16),
        SupportLevel::Unknown
    );

    assert_eq!(
        hardware.memory.spaces.support_level(&MemorySpace::Device),
        SupportLevel::Supported
    );
    assert_eq!(
        hardware.memory.spaces.support_level(&MemorySpace::Host),
        SupportLevel::Unsupported
    );
    assert_eq!(hardware.memory.unified_addressing, SupportLevel::Unknown);
    assert_eq!(hardware.memory.coherent_host_device, SupportLevel::Unknown);

    assert_eq!(hardware.execution.async_execution, SupportLevel::Supported);
    assert_eq!(hardware.execution.ordered_streams, SupportLevel::Supported);
    assert_eq!(hardware.execution.subgroup_operations, SupportLevel::Unknown);
    assert_eq!(hardware.execution.atomic_i64, SupportLevel::Unknown);
    assert_eq!(hardware.matrix.accelerated, SupportLevel::Unknown);
    assert!(hardware.reproducibility.modes.is_empty());
}
