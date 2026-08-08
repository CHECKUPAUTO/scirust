# SciRust Compute Capability Model v2

## Status

This document defines the architecture direction for hardware portability in
`scirust-compute`. The target is not a particular workstation: SciRust must be
able to describe and select execution paths across x86-64, AArch64, RISC-V,
LoongArch, GPUs and future accelerators without leaking one vendor ISA into the
backend-neutral API.

The first implementation lives in `scirust-compute/src/hardware.rs` and is
additive to the existing `DeviceCapabilities` contract.

## 1. Design goals

The capability model MUST:

1. keep `scirust-compute` backend-neutral, `no_std` capable and free of unsafe
   code;
2. distinguish architecture identity from optional ISA features;
3. represent fixed-width and scalable-vector machines without hard-coding
   128/256/512-bit assumptions into planners;
4. distinguish storage, arithmetic and accumulation formats;
5. represent matrix/tensor acceleration without naming one vendor API as the
   universal abstraction;
6. describe memory and execution semantics required by schedulers;
7. distinguish `Unknown` from `Unsupported` so incomplete probing never becomes
   a false negative claim;
8. make reproducibility an explicit execution contract;
9. permit future architectures and vendor extensions without redesigning the
   public API; and
10. preserve existing compute backends while the richer contract is adopted.

## 2. Separation of concerns

### 2.1 `DeviceCapabilities`: legacy resource limits

The existing structure remains the compatibility layer for facts already used
throughout SciRust:

- logical `DeviceId`;
- human-readable backend/device name;
- supported storage dtypes;
- maximum buffer size;
- workgroup limits; and
- asynchronous execution flag.

It is intentionally not expanded with many mandatory fields in this phase,
because external struct literals exist across the workspace. Expanding the
structure directly would create a broad migration unrelated to the semantic
model itself.

### 2.2 `HardwareCapabilities`: portable semantic profile

The v2 profile is a separate, richer value returned by `ComputeBackend`.
Existing backends receive a conservative default derived from
`DeviceCapabilities`; specialized backends can override it as reliable probes
are implemented.

The profile contains:

- `Architecture`: processor architecture family plus an optional open-ended
  name;
- `IsaCapabilities`: optional ISA features and vector execution model;
- `NumericCapabilities`: storage, arithmetic and accumulation formats;
- `MatrixCapabilities`: matrix/tensor acceleration and supported formats;
- `MemoryCapabilities`: memory spaces, coherency/addressing and async-transfer
  semantics;
- `ExecutionCapabilities`: asynchronous execution, ordered streams, subgroups
  and 64-bit atomics; and
- `ReproducibilityCapabilities`: semantic reproducibility modes the backend can
  explicitly promise.

## 3. Architecture identity and ISA features

Architecture identity and ISA features MUST NOT be conflated.

For example, `x86_64` does not imply AVX2, AVX-512 or AMX. Likewise `aarch64`
does not imply SVE or SME. The architecture family identifies the execution
family; optional features are advertised only after reliable detection.

Initial architecture families are:

- x86-64;
- AArch64;
- RISC-V 64;
- LoongArch64;
- wasm32;
- NVIDIA, AMD, Intel and Apple GPU families;
- `Other`; and
- `Unknown`.

`Other` plus an optional architecture name is the forward-compatibility path for
new processors or accelerators not yet represented by a stable enum variant.

Initial ISA feature identifiers cover the paths SciRust already cares about or
is likely to dispatch soon:

- x86-64: SSE2, AVX2, FMA, AVX-512F, AVX-512 VNNI, AVX-512 BF16/FP16 and AMX;
- AArch64: NEON, dot-product, i8mm, BF16, SVE/SVE2 and SME/SME2;
- RISC-V: vector extension;
- LoongArch: LSX/LASX; and
- an open-ended vendor/future feature variant.

Presence in the type system does not mean SciRust currently has a stable kernel
implementation for every feature. Detection, implementation and conformance are
separate stages.

## 4. Vector model

The planner must reason about vector semantics rather than assuming a fixed
register width.

`VectorModel` therefore distinguishes:

- `Scalar`;
- `FixedWidth`;
- `Scalable`; and
- `Unknown`.

Optional minimum and maximum widths are metadata, not the dispatch identity.
This is important for SVE, SME and RISC-V vector systems where code should be
vector-length agnostic.

## 5. Numeric and matrix capabilities

Storage, arithmetic and accumulation are represented separately. This avoids a
common portability error where support for storing BF16/FP16 is treated as proof
that a device performs native arithmetic or accumulation in the same format.

Matrix/tensor acceleration is similarly separated from vector ISA features.
The universal contract describes whether accelerated matrix execution exists
and which input/accumulation dtypes it supports; CUDA Tensor Cores, AMX, SME or
future NPUs remain backend implementations of that semantic capability.

## 6. Memory and execution semantics

The capability model intentionally avoids PCIe- or NUMA-specific assumptions in
the universal layer. The first memory contract describes:

- accepted memory spaces;
- host/device coherency;
- unified addressing; and
- asynchronous transfers.

Execution describes:

- asynchronous execution;
- ordered streams/queues;
- subgroup operations; and
- 64-bit atomics.

Detailed machine topology (NUMA nodes, caches, interconnects, peer access,
coherency domains and bandwidth/latency measurements) belongs in a later
`SystemTopology` layer rather than in architecture identity.

## 7. Unknown is a first-class state

Boolean capability flags are insufficient for portable probing. A backend may
not know whether a property is supported because:

- the OS does not expose it;
- a stable Rust probe is unavailable;
- an accelerator API was compiled out;
- the backend intentionally did not query it; or
- the architecture is new to SciRust.

`SupportLevel` therefore has three states:

- `Supported`;
- `Unsupported`; and
- `Unknown`.

A planner MUST NOT treat `Unknown` as `Unsupported` when deciding whether the
hardware has the feature. It may choose a conservative fallback, but the reason
must remain distinguishable.

## 8. Reproducibility contract

Hardware portability and deterministic science are separate dimensions.
Different reductions, fused operations and accelerator kernels may produce
numerically valid results without cross-backend bit identity.

The model therefore exposes semantic modes:

- `BitExact`: identical output bits under the declared contract;
- `Deterministic`: repeatable for the same backend/implementation, without
  cross-backend bit identity being implied;
- `NumericallyEquivalent`: tolerance-based equivalence under a separate numeric
  contract; and
- `FastApproximate`: documented approximation is permitted.

These modes are not automatically ordered. A backend/kernel pair advertises the
modes it can actually satisfy.

## 9. Compatibility strategy

Phase 1 is intentionally additive:

- existing `DeviceCapabilities` fields remain unchanged;
- existing `ComputeBackend` implementations compile because
  `hardware_capabilities()` has a conservative default implementation;
- the default bridge promotes only facts already known by the old contract;
- optional ISA, matrix, memory and reproducibility properties stay unknown until
  a backend advertises them explicitly.

This avoids inventing hardware facts and avoids a workspace-wide flag day.

## 10. Planned implementation phases

### Phase 1 — capability vocabulary and compatibility bridge

Implemented by this change:

- architecture family and open-ended architecture identity;
- tri-state support;
- ISA/vector, numeric, matrix, memory and execution capability types;
- reproducibility modes;
- `HardwareCapabilities`;
- conservative bridge from `DeviceCapabilities`; and
- default `ComputeBackend::hardware_capabilities()`.

### Phase 2 — CPU hardware probe

Add a small probe layer, separate from `scirust-compute`, that maps runtime facts
to the v2 contract.

Responsibilities:

- x86-64 runtime feature detection;
- AArch64 runtime feature detection;
- RISC-V/LoongArch detection where the Rust/toolchain/OS combination can state it
  reliably;
- vector-model population;
- no unsafe code in the neutral contract; and
- no global `RUSTFLAGS` requirement as a detection mechanism.

The probe MUST distinguish runtime detection support from availability of stable
Rust intrinsics. A feature may be detectable before SciRust can legally compile
a specialized stable kernel for it.

### Phase 3 — kernel requirements and deterministic planner

Introduce a `KernelRequirements` contract and capability matcher. Kernels should
request semantic requirements (dtype, vector model, matrix capability, memory
space, reproducibility) instead of branching directly on vendor names.

The planner should be deterministic for the same capability profile and policy.
Optional benchmark-driven autotuning, if added later, must be a separate mode
with explicit persistence/provenance.

### Phase 4 — backend enrichment

Populate real profiles in the existing backends:

- CPU/reference backend: host architecture, detected ISA/vector features and
  explicit reproducibility modes;
- WGPU backend: device limits, shader numeric features, memory/execution
  semantics and subgroup support where the API exposes them;
- CUDA backend: NVIDIA architecture/generation, numeric and matrix capabilities,
  stream/memory semantics and supported reproducibility modes.

Backend-specific APIs remain implementation details. The planner consumes only
the neutral profile.

### Phase 5 — topology model

Add `SystemTopology`/`MemoryTopology` separately from device capabilities:

- NUMA nodes and CPU affinity domains;
- cache hierarchy;
- accelerator locality;
- unified/coherent memory domains;
- peer access;
- interconnect class; and
- optional measured/static transfer characteristics.

This is where dual-socket servers, Jetson unified memory and future coherent
accelerators can be described without baking one machine layout into the API.

### Phase 6 — SciAgent decode integration

SciAgent should consume semantic capabilities through a backend-neutral decode
engine. The sampler path should have a scalar/reference oracle and specialized
CPU/WGPU/CUDA implementations with explicit sampler semantics.

The repository already contains device-resident and parallel-sampling work; new
work should integrate that work with the capability planner rather than create a
second sampler architecture.

### Phase 7 — COGNO-1 execution attestation

COGNO-1 should not import ISA intrinsics into its deterministic core. A later
host-provided execution profile can attest:

- compute backend;
- architecture/capability profile hash;
- numeric mode;
- reproducibility mode;
- kernel/sampler semantic version;
- memory budget; and
- model/tokenizer provenance.

The deterministic core should validate a small versioned data contract rather
than depend directly on hardware-probing implementation details.

## 11. Conformance strategy

The portability work is incomplete until each specialized implementation is
checked against a reference oracle.

Required test layers are:

1. unit tests for capability semantics and unknown-state handling;
2. compile checks for supported target triples where workspace dependencies
   permit cross compilation;
3. scalar/reference semantic tests;
4. property/conformance tests comparing specialized kernels with the reference
   contract;
5. deterministic tie-breaking and NaN/Inf tests for samplers/reductions;
6. hardware-gated tests for x86, AArch64 and CUDA/WGPU paths; and
7. performance benchmarks reported separately from correctness.

Performance claims must come from measured benchmarks. The capability model must
never encode assumptions such as "AVX-512 is always faster than AVX2" or fixed
speedup factors.

## 12. Immediate next code changes

After Phase 1 is green, the next PR should implement CPU probing and populate a
real CPU hardware profile without changing planner semantics. The following PR
should introduce kernel requirements/capability matching. Backend enrichment and
SciAgent integration then build on those contracts rather than adding ad-hoc
architecture checks to individual algorithms.
