# Elastic Latent KV — Phase 12 Kernels

Phase 12 introduces an explicit numerical-kernel layer for the latent attention
runtime and validates the same latent projection algebra across the existing
SciRust CPU, portable-SIMD, WGPU and CUDA compute paths.

## Core kernel contract

`LatentKernelDispatch` exposes:

- scalar reference execution;
- stable block-4 execution;
- optional portable-SIMD contiguous dot products through `scirust-simd`.

The scalar path is the oracle. The stable block-4 path preserves the exact
scalar accumulation order and is required to be bit-identical. Under the
`portable-simd` feature, contiguous dot products use the existing safe
`scirust-simd` implementation and are checked against the scalar oracle within a
strict floating-point tolerance. Strided and weighted-accumulation paths retain
the scalar-order block-4 implementation to avoid gather scratch and unintended
association changes.

`dot_strided_with_initial` carries the incoming accumulator explicitly so a
Transformer projection can preserve the existing order `bias + Σ(x_i*w_i)`.

## GPU differential validation

`scirust-gpu/tests/elastic_latent_kv_accel.rs` expresses dense-to-latent and
latent-to-dense projection as ordinary `RawComputeBackend::gemm_f32` calls. This
is deliberately outside `scirust-core`, preserving the existing dependency
direction `scirust-gpu -> scirust-core` and avoiding a cycle.

The same test vectors are exercised through:

- `CpuBackend`, which is the deterministic reference;
- `WgpuBackend` with the `wgpu` feature, required to execute against Mesa
  lavapipe in CI and match the CPU oracle within tolerance;
- `CudaBackend` with the `cuda` feature, compiled in CI and, when a CUDA device
  is available, required to remain within the documented bf16 Tensor-core error
  envelope rather than fabricating success.

CUDA unavailability remains an explicit `BackendError::Unavailable("cuda")` and
is not treated as a fake successful computation.

## Scope boundary

Phase 12 validates the numerical accelerator primitives needed by Elastic Latent
KV. The Phase 13 session runtime still uses the safe Core dispatch for its hot
path; moving complete cache residency and sparse-residual attention state to a
device backend would require device-owned persistent KV buffers and is not
silently claimed here.

No new FFI or `unsafe` code is added to `scirust-core` by Phase 12.
