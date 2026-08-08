# WGPU backend capability profile

This document defines the rich `HardwareCapabilities` contract intended for `WgpuComputeAdapter`.

The profile describes what the concrete SciRust WGPU adapter can prove through its current API contract. It does **not** infer the physical accelerator architecture from an adapter name, operating system, graphics API, or machine family.

## Architecture identity

`Architecture` remains `Unknown`.

WGPU may execute through Vulkan, Metal, DX12, GL, a discrete GPU, an integrated GPU, or a software adapter such as lavapipe. An adapter display name is diagnostic text, not a stable architecture contract. Planner decisions must therefore not turn strings such as `NVIDIA`, `AMD`, `Intel`, `Apple`, or `llvmpipe` into architecture facts.

## Numeric contract

The existing adapter advertises these storage dtypes:

- `U32`
- `I32`
- `F32`

The generic WGSL contract supports arithmetic and accumulation for the same baseline scalar set. Other SciRust dtypes remain unsupported by this adapter until the implementation and tests expose them explicitly.

This does not imply tensor-core, matrix-core, packed-dot-product, FP16, BF16, or 64-bit arithmetic support.

## Memory contract

Caller-visible allocation currently accepts only `MemorySpace::Device`.

`Host`, `HostPinned`, and `Unified` are unsupported as **allocation spaces of this adapter**. Internal staging buffers used for queue uploads and readback do not make host memory a caller-visible WGPU allocation space.

However, the physical properties `coherent_host_device` and `unified_addressing` remain `Unknown`. A WGPU device may sit on either a discrete-memory or unified-memory machine, and SciRust must not infer topology from the portable API. This distinction is particularly important for integrated and unified-memory accelerators.

`async_transfers` also remains `Unknown` in the rich hardware profile. The current `ComputeBackend::read` contract waits for readback completion and does not expose a separate transfer event that a planner can target.

## Execution contract

- `async_execution`: `Supported` — `launch` returns a submission event and completion is explicit through `wait`/`synchronize`.
- `ordered_streams`: `Supported` — logical streams map to the adapter's single ordered WGPU queue.
- `subgroup_operations`: `Unsupported` by the current adapter configuration — the device is acquired with `required_features = wgpu::Features::empty()` and generic planner candidates cannot require optional subgroup features.
- `atomic_i64`: `Unsupported` by the current WGSL scalar contract.

## Matrix acceleration

`matrix.accelerated` remains `Unknown`.

The implementation does not expose a semantic matrix/tensor instruction contract through `ComputeBackend`. A driver may optimize a shader internally, but that is not sufficient evidence for a deterministic capability planner to select a matrix-acceleration-required candidate.

## Reproducibility

No global reproducibility mode is advertised by the generic WGPU adapter.

Specific SciRust WGPU kernels can provide stronger deterministic contracts separately when their algorithm and validation justify it. The generic ability to compile arbitrary WGSL is not itself proof of `BitExact` or `Deterministic` execution across devices and drivers.

## Planner consequence

A portable WGPU candidate should express semantic requirements such as `F32`, device memory, and asynchronous execution. Hardware-specific candidates requiring a known accelerator family, matrix acceleration, subgroup operations, FP16/BF16, or a reproducibility mode remain ineligible until those properties are explicitly proven.
