#![cfg(any(feature = "wgpu", feature = "cuda"))]

use scirust_compute::{
    AcceleratorTopologyProvider, DeviceKind, MemorySpace, SupportLevel, SystemTopology,
    augment_accelerator_topology,
};

#[cfg(feature = "cuda")]
use scirust_gpu::CudaComputeAdapter;
#[cfg(feature = "wgpu")]
use scirust_gpu::WgpuComputeAdapter;

#[cfg(feature = "wgpu")]
#[test]
fn runtime_wgpu_adapter_augments_topology_without_physical_fabric_claims() {
    let adapter = match WgpuComputeAdapter::new()
    {
        Ok(adapter) => adapter,
        Err(error) if std::env::var_os("SCIRUST_REQUIRE_WGPU").is_some() =>
        {
            panic!("SCIRUST_REQUIRE_WGPU is set but WGPU acquisition failed: {error}")
        },
        Err(error) =>
        {
            eprintln!("wgpu: {error}; skipping accelerator-topology integration test");
            return;
        },
    };

    let descriptor = AcceleratorTopologyProvider::accelerator_topology_descriptor(&adapter);
    assert_eq!(descriptor.device.kind(), DeviceKind::Wgpu);
    let memory = descriptor.memory.as_ref().expect("logical WGPU memory domain");
    assert_eq!(memory.space, MemorySpace::Device);
    assert_eq!(memory.capacity_bytes, None);
    assert_eq!(memory.host_addressable, SupportLevel::Unknown);

    let mut topology = SystemTopology::default();
    let added = augment_accelerator_topology(&mut topology, descriptor).unwrap();
    assert_eq!(topology.validate(), Ok(()));

    let link = topology
        .adjacent_links(added.accelerator)
        .next()
        .expect("WGPU accelerator-memory affinity");
    assert_eq!(link.interconnect, None);
    assert_eq!(link.metrics, None);
}

#[cfg(feature = "cuda")]
#[test]
fn runtime_cuda_adapter_augments_topology_from_driver_facts_only() {
    let adapter = match CudaComputeAdapter::new()
    {
        Ok(adapter) => adapter,
        Err(error) if std::env::var_os("SCIRUST_REQUIRE_CUDA").is_some() =>
        {
            panic!("SCIRUST_REQUIRE_CUDA is set but CUDA acquisition failed: {error}")
        },
        Err(error) =>
        {
            eprintln!("cuda: {error}; skipping accelerator-topology integration test");
            return;
        },
    };

    let expected_capacity = u64::try_from(adapter.runtime().device_info().total_memory_bytes).ok();
    let descriptor = AcceleratorTopologyProvider::accelerator_topology_descriptor(&adapter);
    assert_eq!(descriptor.device.kind(), DeviceKind::Cuda);
    let memory = descriptor.memory.as_ref().expect("logical CUDA memory domain");
    assert_eq!(memory.space, MemorySpace::Device);
    assert_eq!(memory.capacity_bytes, expected_capacity);
    assert_eq!(memory.host_addressable, SupportLevel::Unknown);

    let mut topology = SystemTopology::default();
    let added = augment_accelerator_topology(&mut topology, descriptor).unwrap();
    assert_eq!(topology.validate(), Ok(()));

    let link = topology
        .adjacent_links(added.accelerator)
        .next()
        .expect("CUDA accelerator-memory affinity");
    assert_eq!(link.interconnect, None);
    assert_eq!(link.metrics, None);
}
