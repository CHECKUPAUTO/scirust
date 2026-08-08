# CUDA backend capability profile

This document defines the rich `HardwareCapabilities` contract intended for `CudaComputeAdapter`.

The profile separates facts proven by the acquired CUDA runtime from properties that are merely plausible on a particular NVIDIA GPU. Planner-visible guarantees must come from structured runtime information or from the concrete adapter contract, never from product-name parsing or benchmark timing.

## Architecture identity

An acquired CUDA context proves `ArchitectureFamily::NvidiaGpu`.

`CudaDeviceInfo::compute_capability` is queried through the CUDA driver and is recorded as an architecture name such as `sm_90` or `sm_110`. The human-readable GPU name remains diagnostic text and is never parsed to select a capability.

Compute capability identifies the CUDA architecture target. It does **not** by itself prove that SciRust has a usable tensor-core, matrix-core, subgroup, or specialized low-precision kernel path.

## Numeric contract

The legacy adapter exposes a broad set of storage dtypes. The rich profile does not automatically reinterpret every storage dtype as an arithmetic guarantee.

The current generic `CudaComputeAdapter` has an active PTX execution proof for:

- `U32`

`I32`, `F32`, and the remaining arithmetic/accumulation dtypes stay `Unknown` until this same generic PTX adapter path has explicit execution tests for them. Arithmetic proven by the separate `CudaReferenceAdapter` is not silently transferred to this contract. Likewise, merely listing `F16`, `BF16`, or `F64` as storable data is not sufficient evidence for planner selection of a low-precision or double-precision arithmetic candidate.

## Memory contract

Caller-visible allocation currently accepts only `MemorySpace::Device`.

`Host`, `HostPinned`, and `Unified` are unsupported as allocation spaces of this adapter today. This must not be confused with the machine's physical memory topology. `coherent_host_device` and `unified_addressing` remain `Unknown`, which is essential for integrated CUDA systems where CPU and GPU can share physical memory even though this adapter still exposes device allocations through its current API.

The transfer API also does not publish a separate asynchronous transfer event. Host-to-device writes are ordered on the CUDA stream while device-to-host reads explicitly synchronize before returning, so `async_transfers` remains `Unknown` at the generic hardware-profile level.

## Execution contract

- `async_execution`: `Supported` — kernel launch records and returns a CUDA completion event.
- `ordered_streams`: `Supported` — the raw runtime owns one ordered CUDA stream and serializes host-side submissions.
- `subgroup_operations`: `Unknown` — CUDA warp facilities are not converted into a generic planner guarantee without an explicit backend contract.
- `atomic_i64`: `Unknown` — no device-attribute query or generic adapter test currently proves this semantic capability.

## Matrix acceleration

`matrix.accelerated` remains `Unknown`.

Some CUDA devices contain Tensor Cores and SciRust has specialized CUDA paths elsewhere, but the generic `CudaComputeAdapter` accepts arbitrary precompiled PTX. CUDA presence or a compute-capability number is not enough to claim that a planner-selected kernel can use matrix acceleration through this adapter.

## Reproducibility

No global reproducibility mode is advertised for arbitrary PTX.

The raw runtime contains carefully constrained NVRTC options for its source-compilation path, but `CudaComputeAdapter` itself accepts caller-provided PTX. Determinism or bit-exactness therefore belongs to the concrete kernel contract, not to CUDA as a blanket backend property.

## Planner consequence

A generic CUDA candidate may require NVIDIA GPU architecture, device memory, asynchronous execution, ordered submissions, and the currently proven U32 arithmetic baseline. Candidates requiring I32/F32 arithmetic, BF16/FP16 arithmetic, Tensor Cores, warp/subgroup operations, i64 atomics, unified memory, or a reproducibility level remain ineligible until those capabilities are explicitly established on this adapter path.
