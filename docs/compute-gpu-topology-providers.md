# GPU topology providers

`AcceleratorTopologyProvider` bridges an acquired compute backend to the backend-neutral `SystemTopology` augmentation API.

The provider contract is intentionally narrower than `HardwareCapabilities`: it describes the logical accelerator and the logical memory domain exposed by the adapter. It does not infer physical fabric properties.

## WGPU

`WgpuComputeAdapter` publishes:

- its stable logical WGPU `DeviceId`;
- its adapter name as diagnostic text only;
- one logical `MemorySpace::Device` domain;
- unknown memory capacity;
- `host_addressable = Unknown`.

`DeviceCapabilities.max_buffer_bytes` is a maximum single-buffer limit. It is **not** total physical memory and is therefore never used as topology capacity.

No vendor, architecture, PCIe, UMA, coherence, NUMA, peer-access, bandwidth, or latency fact is derived from the WGPU adapter name or API.

## CUDA

`CudaComputeAdapter` publishes:

- its stable logical CUDA `DeviceId`;
- its diagnostic backend/device name;
- one logical `MemorySpace::Device` domain;
- total device memory reported by the acquired CUDA driver as the logical domain capacity;
- `host_addressable = Unknown`.

The capacity fact does not imply discrete VRAM. Integrated CUDA systems can expose device allocations while sharing physical DRAM with the CPU. The provider therefore does not add `SharedMemory`, `OnChip`, `CoherentWith`, PCIe, unified-addressing, or NUMA relations.

## Composition

A caller can collect provider descriptors from all acquired accelerators and pass them together to `augment_accelerator_topologies`. The batch API sorts by stable logical `DeviceId`, so backend discovery order does not affect accelerator node IDs.

Physical interconnect discovery remains a separate layer. It may enrich the graph only when an OS or backend API proves a specific relation.

## Validation

The integration test `accelerator_topology_profiles` calls the public provider trait on real WGPU/CUDA adapters, augments a `SystemTopology`, validates the graph, and asserts that no interconnect class or transfer metric was fabricated.

When `SCIRUST_REQUIRE_WGPU=1` or `SCIRUST_REQUIRE_CUDA=1` is set, backend acquisition is fail-closed rather than skipped.
