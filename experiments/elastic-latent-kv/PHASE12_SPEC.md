# Elastic Latent KV — Phase 12 Kernels

Phase 12 introduces an explicit numerical-kernel layer for the latent attention
runtime. `LatentKernelDispatch` exposes scalar, stable block-4 and optional
portable-SIMD execution.

The scalar path is the oracle. The stable block-4 path preserves the exact
scalar accumulation order and is required to be bit-identical. Under the
`portable-simd` feature, contiguous dot products use the existing safe
`scirust-simd` implementation and are checked against the scalar oracle within a
strict floating-point tolerance; strided and weighted-accumulation paths retain
the scalar-order block-4 implementation to avoid gather scratch and association
changes.

No FFI or `unsafe` code is added to `scirust-core`. Device-specific WGPU/CUDA
implementations can plug into this dispatch contract in later optimized kernels
without changing the scalar differential oracle.
