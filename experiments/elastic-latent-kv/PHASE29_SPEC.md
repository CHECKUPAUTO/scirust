# Phase 29 — Conservative CUDA hardware capability profile

## Status

Phase 29 follows the architecture-neutral planner work merged in Phase 27 and the independent Phase 28 WGPU generation benchmark. It closes the backend-profile symmetry gap across CPU, WGPU and CUDA.

The CUDA adapter already acquires real runtime facts from `CudaRawRuntime`: device ordinal, compute capability, total device memory, maximum block dimensions, maximum grid dimensions and shared memory per block. The rich `HardwareCapabilities` path currently discards most of those facts by falling back to the generic legacy bridge.

Phase 29 publishes only properties that the CUDA backend can prove from its API/runtime contract. It does not parse product names, recognize Jetson/Thor SKUs, consult host architecture, or infer optional accelerator features from marketing identity.

## Scope

### 1. CUDA architecture identity from backend semantics and compute capability

`CudaComputeAdapter` is intrinsically a CUDA backend, so the processor family is `ArchitectureFamily::NvidiaGpu`.

The optional architecture name is derived only from the runtime-provided compute capability, e.g. `sm_110` for compute capability 11.0. The human-readable CUDA device name remains diagnostic metadata and never participates in capability planning.

No table of GPU product names or host machine models is introduced.

### 2. Conservative numeric contract

The existing `DeviceCapabilities` storage-dtype list remains the source of truth for caller-visible storage widths.

For generic arithmetic/accumulation capability, Phase 29 advertises only the baseline scalar types that the current PTX backend can safely treat as portable generic compute primitives:

- `U32`
- `I32`
- `F32`

All other arithmetic/accumulation dtypes remain `Unknown` rather than being inferred from compute capability alone. In particular, Phase 29 does not claim FP16/BF16 tensor arithmetic, FP64 performance, packed integer dot products or tensor cores.

### 3. Memory contract

The current `CudaComputeAdapter::allocate` accepts only `MemorySpace::Device` and byte alignment. Therefore the rich allocation-space contract is:

- Device: Supported
- Host: Unsupported
- HostPinned: Unsupported
- Unified: Unsupported

These are adapter allocation semantics, not claims about physical topology. CUDA unified addressing, coherent host/device memory and managed-memory availability remain `Unknown` because the current runtime probe does not retain those properties.

The current read API synchronizes before returning and the generic backend does not expose an independently schedulable transfer event, so `async_transfers` remains `Unknown`.

### 4. Execution contract

The adapter owns one ordered CUDA stream and kernel launch returns an explicit event. Therefore:

- async execution: Supported
- ordered streams: Supported

Subgroup/warp operations and i64 atomics remain `Unknown` in the generic profile until the adapter exposes and tests a semantic contract for planner-selected kernels.

### 5. Matrix acceleration and reproducibility

`matrix.accelerated` remains `Unknown`.

A compute capability can correlate with physical tensor-core generations, but the generic `ComputeBackend` contract currently accepts arbitrary precompiled PTX and exposes no typed matrix-instruction requirement. Phase 29 therefore does not translate compute capability into a matrix-acceleration claim.

No global reproducibility mode is advertised by the generic CUDA adapter. Individual deterministic/reference kernels keep their stronger, separately tested guarantees.

## Runtime facts deliberately kept outside the profile

`CudaDeviceInfo` also reports maximum grid dimensions and shared memory per block. The current rich capability schema has no semantic fields for those limits. Phase 29 documents them as retained runtime facts but does not shoehorn them into unrelated fields. A later planner extension can add typed launch/shared-memory requirements.

Phase 27 already models maximum block dimensions through `ExecutionLimits::from_device_capabilities()` using the existing `DeviceCapabilities.max_workgroup_size` bridge, so no duplicate CUDA-specific workgroup planner is required.

## Validation

Required proof:

1. unit tests construct synthetic CUDA runtime facts with deliberately misleading device names and prove the profile never parses those names;
2. architecture family is `NvidiaGpu` and architecture name is derived from compute capability only;
3. storage dtype support remains identical to `DeviceCapabilities`;
4. only U32/I32/F32 are marked as baseline arithmetic/accumulation support;
5. only Device allocation space is advertised;
6. async execution and ordered stream support are explicit;
7. matrix acceleration, subgroup operations, i64 atomics, physical memory coherence and global reproducibility remain unknown;
8. real CUDA integration on the self-hosted Thor/native ARM64 gate verifies the acquired adapter publishes the same conservative profile;
9. strict rustfmt, Clippy and Rust 1.89.0 MSRV remain green.

## Non-goals

Phase 29 does not:

- special-case Jetson AGX Thor or any GPU SKU;
- add a CUDA sampler implementation;
- infer tensor cores from compute capability;
- expose managed/pinned host allocations;
- change CUDA kernel math or launch behavior;
- alter the WGPU or CPU profiles;
- introduce timing-based planning.
