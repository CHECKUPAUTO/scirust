# RFC-0001 — SciRust Compute Contracts

- **Status:** Draft
- **Crate:** `scirust-compute`
- **MSRV:** Rust 1.89

## Motivation

SciRust already has several compute paths — CPU, SIMD, WGPU and CUDA —
but their contracts are separate and incompatible.

`scirust-compute` introduces a common vocabulary without replacing the
existing implementations.

## Decision

The crate defines only:

- the scalar types and tensor metadata;
- the devices and memory spaces;
- the kernel modules and launch configurations;
- the buffer bindings;
- the common errors;
- the `ComputeBackend` trait.

The crate stays dependency-free and `no_std`-compatible.

## Dependency rule

The backends depend on `scirust-compute`.

`scirust-compute` depends on neither `scirust-core`, nor `scirust-gpu`,
nor `scirust-cuda`, nor `scirust-simd`.

## Non-goals

This phase does not replace CUDA or WGPU, does not create a new tensor
engine, does not modify any existing API, and implements neither autograd
nor a scheduler.
