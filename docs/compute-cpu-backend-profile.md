# SciRust CPU Compute Backend Capability Profile

## Scope

This profile applies to `scirust_gpu::CpuComputeAdapter`, the serial Reference-kernel adapter. It does **not** describe every capability of the host CPU.

Host architecture and ISA facts continue to come from `scirust_compute::probe_host_cpu()`. Backend semantics below describe only operations and resources that `CpuComputeAdapter` actually exposes through `ComputeBackend`.

## Numeric contract

The Reference-kernel compiler accepts `F32` element semantics today. Therefore the backend profile may state:

- `F32` storage: supported (already represented by legacy `DeviceCapabilities`);
- `F32` arithmetic: supported;
- `F32` accumulation: supported;
- other currently enumerated scalar types: unsupported for Reference arithmetic/accumulation.

This is an implementation statement, not an assertion that the host CPU lacks those numeric formats.

## Memory contract

`CpuComputeAdapter::allocate()` currently accepts only:

- alignment `1`;
- `MemorySpace::Host`.

It explicitly rejects `HostPinned`, `Device` and `Unified` spaces. Consequently the backend profile may report those spaces as unsupported. It may also report device/host coherence, unified addressing and asynchronous transfers as unsupported because the adapter exposes no device-memory domain or asynchronous transfer mechanism.

## Execution contract

Reference execution is serial and completes before `launch()` returns. The profile may therefore state:

- asynchronous execution: unsupported;
- ordered streams: supported;
- subgroup operations: unsupported;
- 64-bit device atomics: unsupported.

These are adapter guarantees, not physical CPU ISA claims.

## Matrix acceleration

The Reference adapter does not expose a matrix/tensor-acceleration execution path through `ComputeBackend`. `matrix.accelerated` is therefore unsupported for this backend even when the host processor physically contains matrix instructions.

A future optimized CPU backend may advertise a different profile while sharing the same physical host probe.

## Reproducibility

The adapter executes Reference kernels in a fixed serial order and rejects operations such as `Exp` and `Log` when the Reference implementation cannot provide the required reproducible semantics. It is honest to advertise `ReproducibilityLevel::Deterministic`.

The backend profile must **not** advertise global `BitExact` support. Floating-point NaN payload preservation is not guaranteed for arithmetic, and bit identity is a stronger contract than deterministic execution.

`NumericallyEquivalent` and `FastApproximate` remain unknown until SciRust exposes explicit selectable execution modes with those semantics.

## Runtime ISA preservation

An override of `ComputeBackend::hardware_capabilities()` must preserve the runtime architecture/ISA overlay introduced by the host CPU probe. Enriching backend semantics must never regress AVX/NEON/SVE/other runtime facts back to `Unknown`.

Tests should compare the adapter profile's architecture and ISA fields directly with `probe_host_cpu()` when `std` is enabled.
