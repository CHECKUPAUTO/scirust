#![cfg(feature = "cuda")]

use scirust_compute::{ArchitectureFamily, ComputeBackend, DeviceKind, MemorySpace, SupportLevel};
use scirust_gpu::CudaComputeAdapter;

#[test]
fn runtime_cuda_adapter_publishes_the_conservative_rich_profile() {
    let adapter = match CudaComputeAdapter::new()
    {
        Ok(adapter) => adapter,
        Err(error) =>
        {
            assert!(
                std::env::var_os("SCIRUST_REQUIRE_CUDA").is_none(),
                "SCIRUST_REQUIRE_CUDA is set, so the CUDA rich-capability profile must be \
                 validated on a real device, but the adapter could not be acquired: {error}"
            );
            eprintln!(
                "cuda: no runtime device available; skipping rich-capability integration test \
                 ({error})"
            );
            return;
        },
    };

    let hardware = ComputeBackend::hardware_capabilities(&adapter);

    assert_eq!(hardware.device.kind(), DeviceKind::Cuda);
    assert_eq!(hardware.architecture.family, ArchitectureFamily::NvidiaGpu);
    assert!(
        hardware
            .architecture
            .name
            .as_deref()
            .is_some_and(|name| name.starts_with("sm_"))
    );
    assert_eq!(
        hardware.memory.spaces.support_level(&MemorySpace::Device),
        SupportLevel::Supported
    );
    assert_eq!(
        hardware.memory.spaces.support_level(&MemorySpace::Unified),
        SupportLevel::Unsupported
    );
    assert_eq!(hardware.execution.async_execution, SupportLevel::Supported);
    assert_eq!(hardware.execution.ordered_streams, SupportLevel::Supported);
    assert_eq!(hardware.matrix.accelerated, SupportLevel::Unknown);
    assert!(hardware.reproducibility.modes.is_empty());
}
