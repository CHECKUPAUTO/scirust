#![cfg(feature = "wgpu")]

use scirust_compute::{ArchitectureFamily, ComputeBackend, DeviceKind, MemorySpace, SupportLevel};
use scirust_gpu::WgpuComputeAdapter;

#[test]
fn runtime_wgpu_adapter_publishes_the_conservative_rich_profile() {
    let Ok(adapter) = WgpuComputeAdapter::new()
    else
    {
        eprintln!("wgpu: no adapter available; skipping rich-capability integration test");
        return;
    };

    let hardware = ComputeBackend::hardware_capabilities(&adapter);

    assert_eq!(hardware.device.kind(), DeviceKind::Wgpu);
    assert_eq!(hardware.architecture.family, ArchitectureFamily::Unknown);
    assert_eq!(
        hardware.memory.spaces.support_level(&MemorySpace::Device),
        SupportLevel::Supported
    );
    assert_eq!(
        hardware.memory.spaces.support_level(&MemorySpace::Host),
        SupportLevel::Unsupported
    );
    assert_eq!(hardware.execution.async_execution, SupportLevel::Supported);
    assert_eq!(hardware.execution.ordered_streams, SupportLevel::Supported);
    assert_eq!(hardware.matrix.accelerated, SupportLevel::Unknown);
    assert!(hardware.reproducibility.modes.is_empty());
}
