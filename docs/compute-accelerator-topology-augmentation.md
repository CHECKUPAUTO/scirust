# Accelerator topology augmentation

`SystemTopology` describes resource placement and locality independently from `HardwareCapabilities`. Host discovery establishes CPU/package/NUMA/cache facts first; accelerator backends can then add only the facts they actually know without forcing every machine into a discrete-GPU topology.

## Generic augmentation contract

`augment_accelerator_topology` is the convenience API for one logical accelerator. `augment_accelerator_topologies` adds a batch and sorts descriptors by stable logical `DeviceId` order before allocating node IDs, so backend discovery order does not perturb the resulting topology snapshot.

Each descriptor adds:

- one `Accelerator` node carrying a stable logical `DeviceId`;
- optionally one `MemoryDomain` node carrying a backend-proven logical memory-space descriptor;
- when memory is present, a bidirectional `AffineTo` relation between the accelerator and that memory domain.

Node IDs use the smallest currently unused `u32`. Batch augmentation is transactional: invalid topology, duplicate logical devices, or node-ID exhaustion leave the input topology unchanged.

## What is deliberately not inferred

Generic augmentation does **not** add any of these facts automatically:

- PCIe;
- on-chip interconnect;
- shared physical memory;
- coherent CPU/GPU memory;
- unified virtual addressing;
- NUMA affinity;
- peer-to-peer access;
- transfer bandwidth or latency.

A backend or OS-specific probe may add those relations later only when it has a reliable source for them. A device-memory allocation API is not sufficient evidence for a discrete physical memory chip or a PCIe path.

## Unified-memory systems

This distinction matters for integrated accelerators and Jetson-class CUDA devices. A backend may expose a logical `MemorySpace::Device` while CPU and accelerator ultimately share physical DRAM. The logical memory domain can therefore be represented without claiming `SharedMemory`, `OnChip`, or `CoherentWith` until those physical properties are independently established.

## Backend use

A conservative backend can augment only the accelerator node if no memory fact is available.

A CUDA backend can additionally attach a logical device-memory domain with driver-reported capacity while leaving `host_addressable` unknown. A WGPU backend can attach a logical device-memory domain with unknown capacity; WGPU buffer limits are not interpreted as total physical memory.

When multiple backends contribute devices, callers should use the batch API whenever discovery order is not itself semantically meaningful. The batch sort ensures that WGPU/CUDA enumeration order cannot change accelerator node IDs or topology fingerprints.

Interconnect and peer-access discovery are separate later phases. This keeps the base augmentation API deterministic, `no_std` compatible, and free of vendor assumptions.
