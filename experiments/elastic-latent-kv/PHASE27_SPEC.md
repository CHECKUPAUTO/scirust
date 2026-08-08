# Phase 27 — Architecture-neutral capability-planned sampler selection

## Status

Phase 27 follows Phase 26's promotion of the exact parallel bounded-top-k sampler. Phase 26 deliberately uses the WGPU adapter's legacy `max_workgroup_size[0]` field directly. Phase 27 removes that local hardware-policy check and routes implementation choice through an architecture-neutral compute planner layer shared across backends.

The branch is initially stacked on Phase 26 so Phase 26's validated head remains frozen while this work proceeds.

## Problem

The Phase 26 policy is deterministic and correct, but the hardware predicate still lives inside the sampled MiniLLM runtime:

- bounded top-k algorithm eligibility is mixed with a direct `max_workgroup_size_x >= 64` check;
- `DeviceCapabilities` already exposes concrete workgroup limits, but the generic planner does not model launch-width requirements;
- `select_candidate()` chooses among hardware candidates for one common `KernelRequirements` value, but it does not choose between multiple implementations with different requirements on the same device.

That is the remaining architecture-policy debt. It should be fixed generically rather than replaced by x86/ARM/NVIDIA-specific branches.

## Scope

### 1. Portable execution limits

Add an architecture-neutral `ExecutionLimits` companion to the compute capability contracts:

- `ExecutionLimits.max_workgroup_size: [Option<u32>; 3]`;
- `ExecutionLimits::from_device_capabilities()` preserves the concrete workgroup dimensions already reported by a backend;
- manually constructed or future partially-probed profiles can leave dimensions as `None`;
- `None` means unknown, never unsupported or zero.

Keeping launch limits as a separate value avoids pretending they are ISA properties while still making them available to generic implementation planning. A later capability-model revision may fold these limits into a richer backend profile without changing sampler policy.

### 2. Semantic implementation requirements

Add `ImplementationRequirements`, composed from:

- the existing `KernelRequirements` semantic contract;
- a portable `WorkgroupRequirement` with minimum dimensions.

Matching semantics are:

- a known device maximum below a required minimum -> `Incompatible`;
- an unknown maximum -> `Indeterminate`;
- a known sufficient maximum -> compatible;
- dimensions with no minimum -> unconstrained.

Diagnostics report the dimension, required minimum and observed maximum where known.

### 3. Implementation selection on one device

Add `select_implementation()` for implementations that each carry their own requirements:

- candidate name;
- explicit static priority;
- candidate-specific semantic and workgroup requirements;
- one shared `HardwareCapabilities` profile and `ExecutionLimits` value;
- the same `PlannerPolicy`, disposition ordering and lexical final tie-break used by the existing device-candidate planner.

No benchmark timing, mutable history, CPU model, architecture family or vendor string may affect the selection.

### 4. Phase 26 sampler integration

Keep algorithm-domain eligibility separate from hardware eligibility.

Parallel bounded-top-k is an algorithm candidate only when:

- temperature is finite and positive;
- `2 <= top_k < vocab_size`;
- `top_k <= PARALLEL_TOP_K_MAX`.

Its implementation requirements include a minimum workgroup X dimension of `PARALLEL_TOP_K_LANES` (64). The sequential oracle candidate has no workgroup-width requirement and lower preference only when both candidates are proven compatible.

Expected behavior:

- known width >= 64 -> parallel candidate selected;
- known width < 64 -> sequential candidate selected;
- unknown width -> parallel candidate is indeterminate and the proven-compatible sequential candidate wins;
- algorithmically ineligible configurations -> sequential candidate only;
- no silent runtime fallback after the planner has selected parallel execution.

## Determinism

Planner behavior remains entirely static and reproducible:

1. explicit incompatibility is never selected;
2. compatible outranks indeterminate;
3. lower static priority wins within one disposition;
4. equal priority uses lexical candidate name;
5. no timing measurement participates in the decision.

Sampling math, PCG state, top-k/top-p ordering and Phase 21 device-feedback semantics are unchanged.

## Architecture neutrality

Phase 27 contains no checks for:

- `target_arch = x86_64` vs `aarch64` in sampler policy;
- NVIDIA/AMD/Intel/Apple device names;
- Jetson/Thor SKU strings;
- CPU model names;
- benchmark-derived runtime thresholds.

The same minimum-workgroup requirement is valid for WGPU on x86_64, ARM/Jetson, Apple Silicon, future RISC-V hosts, discrete GPUs and software Vulkan implementations.

## Validation

Required proof:

1. compute-model tests for known-sufficient, known-insufficient and unknown workgroup limits;
2. implementation-planner tests for deterministic priority/tie ordering and compatible-over-indeterminate behavior;
3. synthetic sampler-policy tests proving >=64 promotion, <64 fallback and unknown fallback without a physical GPU;
4. existing Phase 24 exact parallel sampler lavapipe parity;
5. existing sampled MiniLLM promotion/fallback parity;
6. existing Phase 21 device-feedback parity;
7. strict Clippy and Rust 1.89.0 MSRV;
8. x86_64, aarch64 cross-check and native Jetson Thor gates.

## Non-goals

Phase 27 does not:

- add new sampling math;
- widen `PARALLEL_TOP_K_MAX`;
- introduce timing-based autotuning;
- infer accelerator architecture from names;
- alter CPU SIMD dispatch;
- add multi-device topology scheduling;
- claim subgroup support merely because a workgroup size is available.
