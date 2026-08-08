# Phase 30 — Backend execution-limit contract

## Status

Phase 30 follows the architecture-neutral implementation planner from Phase 27 and the conservative CUDA capability profile from Phase 29. It removes a remaining layering leak: callers currently reconstruct `ExecutionLimits` directly from `DeviceCapabilities` instead of asking the selected backend for its portable launch limits.

The branch is initially stacked on Phase 29 so Phase 29 can remain frozen while this contract is validated.

## Problem

`ExecutionLimits` is already a backend-neutral planner type, but it has no corresponding method on `ComputeBackend`.

As a result, consumers such as the resident sampler currently do this themselves:

`ExecutionLimits::from_device_capabilities(adapter.capabilities())`

That is correct for today's workgroup-width-only model, but it makes every consumer responsible for knowing where limits come from. It also blocks future backend-specific enrichment such as grid and shared-memory limits because callers would bypass any backend override.

## Scope

### 1. `ComputeBackend::execution_limits()`

Add a default method to the generic backend contract:

- returns `ExecutionLimits`;
- default implementation is `ExecutionLimits::from_device_capabilities(self.capabilities())`;
- no architecture, vendor or device-name branching;
- no probing side effects;
- no timing or mutable state.

This preserves current behavior exactly while creating one stable extension point for richer backends.

### 2. Backend conformance proof

Add integration coverage proving that real WGPU and CUDA adapters expose workgroup dimensions through the trait method and that the default result is identical to their current `DeviceCapabilities.max_workgroup_size` facts.

Tests remain fail-closed only when the corresponding `SCIRUST_REQUIRE_WGPU` / `SCIRUST_REQUIRE_CUDA` environment variable is explicitly set. Hosted builds without the physical backend may skip runtime acquisition while still compiling the trait path.

### 3. Consumer migration

Once the trait method itself is green on MSRV/Clippy/backend tests, migrate the resident sampler from manual `ExecutionLimits::from_device_capabilities(...)` reconstruction to `adapter.execution_limits()`.

That migration is intentionally semantics-preserving: Phase 30 does not change which sampler implementation is selected on any current backend.

## Determinism and architecture neutrality

The method is a pure capability query. Selection remains based on semantic limits only.

Phase 30 contains no:

- x86_64/aarch64 policy branch;
- NVIDIA/AMD/Intel/Apple product-name heuristic;
- Jetson/Thor special case;
- benchmark-derived threshold;
- adaptive runtime history.

## Validation

Required proof:

1. Rust 1.89.0 builds the extended trait;
2. strict Clippy passes for `scirust-compute` and GPU adapters;
3. real/synthetic WGPU adapter reports its known workgroup dimensions through `execution_limits()`;
4. real/synthetic CUDA adapter reports the same current workgroup dimensions through `execution_limits()`;
5. after consumer migration, existing Phase 27 sampler-policy tests and resident generation parity remain unchanged;
6. Native ARM64 and Thor gates remain green.

## Non-goals

Phase 30 does not yet add new limit fields for grid dimensions, shared memory, subgroup size or register pressure. Those require typed semantics and backend probes of their own rather than being folded into the workgroup contract opportunistically.
