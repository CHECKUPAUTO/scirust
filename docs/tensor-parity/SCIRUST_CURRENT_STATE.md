# SciRust Current State (Tensor Subsystem)

Audited 2026-08-04 against workspace HEAD `25f272a0` (branch `master`,
toolchain `nightly-2026-07-02`). This file is the factual baseline for the
parity program; it is updated by audit PRs only.

## Workspace map (tensor-relevant crates)

| Crate | Role | Notes |
| --- | --- | --- |
| `scirust-tensor-core` | Prototype N-D tensor (`TensorND`, `Vec<f32>` + shape + strides) | f32 only, `Result<_, String>` errors |
| `scirust-tensor-einsum` | `einsum(pattern, inputs)` multi-operand contraction | Uses the prototype `TensorND` |
| `scirust-tensor-contraction` | `ContractionPlan` (pairwise greedy) + `execute` | |
| `scirust-tensor-compile` | `TensorGraph`, `FusedKernel`, `ElementwiseOp` (AddScalar, MulScalar, Relu, Sigmoid, Tanh, Exp, Log) | fusion DSL |
| `scirust-tensor-runtime` | `TensorRuntime` registry + `run_fused`/`run_contraction`/`run_graph` | |
| `scirust-tensor-examples` | examples on the prototype stack | |
| `scirust-core` | Main crate: 2D `Tensor`/`Tape`/`Var` autodiff, `TensorND`, `Tensor3D`, matrices, optimizers, schedulers | errors via `SciRustError` |
| `scirust-autodiff` | Forward-mode `Dual` (f64) + scalar `Var` tape | |
| `scirust-simd` | SIMD kernels: BLAS (`sgemm_f32_*`, `saxpy`, `sdot`, `sscal`), activations, backward kernels, quant, RoPE, transformer, eigen (lanczos), portable fallback | |
| `scirust-gpu` | WGPU engine: `GpuTensor`, `GpuMatrix`, fused ops | |
| `scirust-cuda` | CUDA engine: `Chain`, `CudaMatrix`, matmul/softmax/rmsnorm/rope/swiglu/embed | |
| `scirust-sparse` | `CooMatrix`, `CsrMatrix`, `CscMatrix`, `SparseLu` | f64 dense solve |

## Storage / dtype reality

- Every dense tensor type stores `f32` (or `f64` for `Dual`, sparse, and
  some simd kernels). There is **no `DType` abstraction anywhere**; dtype
  promotion is therefore not implementable today without a new layer.
- Layout: row-major contiguous, plus strided-view helpers
  (`TensorND::transpose`, `permute`, `slice_axis`, `broadcast_to`,
  `MatrixView` strides). No lazy views in the autodiff path (ops copy).

## Autodiff reality

- Two parallel reverse-mode stacks, **non-interoperating**:
  - `autodiff::reverse` — 2D `Tensor`, `Tape`, `Var<'t>` (the main stack).
  - `autodiff::nd` — `NdTape`, `NdVar<'t>` over `TensorND`.
- Both first-order only; `Dual` (f64) is forward mode.
- Ops present on the 2D stack: elementwise (add/sub/mul/div/hadamard,
  neg/reciprocal, exp/log/log10/sqrt/pow, sin/cos/tan/asin/acos/atan/atan2,
  sinh/cosh/tanh, sigmoid/relu/softmax/log_softmax), reductions
  (sum, sum_axis, mean_axis, max_axis, var_axis), matmul/matmul_bt/
  matmul_gpu/matmul_portable, bmm2d, linear, embedding, dropout,
  layer_norm, l2_normalize, cosine_sim_matrix, reshape, transpose,
  broadcast_to, slice_rows/cols, causal_mask, max_pool2d,
  fake_quantize_ste; Ops on the ND stack: add/sub/mul/div, matmul, bmm,
  softmax, transpose_last2, reshape, layernorm, rmsnorm, sigmoid, exp,
  rope(+portable), permute, relu, sum, gather, causal_conv, cat0,
  cross_entropy.

## Error handling reality

- `scirust-core`: `SciRustError` (ShapeMismatch, DimMismatch, DeviceMismatch,
  WrongDevice, InvalidConfig, GpuNotAvailable, GpuError, IoError,
  InvalidFormat, IndexOutOfBounds, RankMismatch, AxisOutOfBounds,
  NumelMismatch, ...) — structured, good.
- Prototype crates (`scirust-tensor-*`): `Result<_, String>` — **not**
  structured; the parity program must not build on these error surfaces.

## CI reality

- `.github/workflows/ci.yml` (fmt, clippy, build, test on pinned nightly),
  plus `native-arm64.yml`, `release.yml`, `sos-ci.yml`.
- Fuzzing targets exist under `fuzz/fuzz_targets/` (safetensors, tensor ND
  ops, tape backward).
- No differential/PyTorch harness exists yet; fixtures are not generated.

## Verified gaps (drivers for the program)

1. No dtype system; f32-only storage everywhere.
2. Two parallel autodiff stacks with duplicated op sets and no interop.
3. Duplicated `TensorND` (prototype crate vs `scirust-core`) with diverged
   semantics.
4. Prototype crate error handling is `String`, not `SciRustError`.
5. No shape-checked `checked_add/mul/sub` discipline on index arithmetic
   across all constructors (present in some, e.g. prototype `try_offset`,
   `TensorND::reshape`).
6. No differential test harness, no fixtures, no oracle wiring.
7. GPU paths unverified against CPU semantics (no cross-device tests).
