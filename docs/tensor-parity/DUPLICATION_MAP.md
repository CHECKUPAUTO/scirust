# Duplication Map

Every conceptual tensor facility implemented more than once in the workspace,
with the decision for the parity program. Source: full audit 2026-08-04.

## 1. `TensorND` — two implementations

| | Prototype `scirust-tensor-core` | Core `scirust-core/src/tensor/tensor_nd.rs` |
| --- | --- | --- |
| Storage | `Vec<f32>` + explicit strides field | `Arc<[f32]>` + shape (row-major) |
| Errors | `Result<_, String>` | `SciRustError` |
| Views | strides stored but no view ops | `transpose`/`slice_axis`/`broadcast_to`/`unfold` copy (no lazy views) |
| Fuzz | `fuzz/fuzz_targets/tensor_nd_ops_from_bytes.rs` | — |

**Decision**: core `TensorND` is the parity host (structured errors, ND
already). Prototype crate ops are a separate *prototype* surface: keep as
experimental, mark in registry as `experimental`, and **do not** route parity
tests through it. Long-term: either core absorbs the prototype's remaining
semantics or the prototype crate is deprecated. The 2D↔ND bridges
(`from_tensor_2d`/`to_tensor_2d`, `input_from_2d`) stay.

## 2. Autodiff — two reverse-mode stacks

`autodiff::reverse` (2D) and `autodiff::nd` (ND) share no code and do not
interoperate. Op overlap: add/sub/mul/div, matmul, softmax, reshape, sigmoid,
exp, relu, sum, rope, norm variants, cross_entropy.

**Decision**: `NdTape`/`NdVar` is the parity-facing stack (N-D, strided
shapes). The 2D stack remains the primary training stack (optimizers,
schedulers, data-parallel, mixed precision hang off it). Parity PRs add ops to
**both** stacks where a 2D row is claimed (2D is what optimizers use), with
shared kernel functions where SIMD backends exist. `DUPLICATION_MAP` does not
mandate merging; it mandates that the registry records *which* implementation
satisfies a row.

## 3. matmul

Implementations: 2D `matmul`/`matmul_portable`, ND `matmul`, SIMD `sgemm_f32*`
(+ SVE, AMX i8/bf16, portable), GPU `matmul`, CUDA `matmul(at/bt)`, plus
`soft_gemm` in matrix/soft.rs and `spmm_dense` (CSR). No shared dispatch.

**Decision**: parity row for `torch.matmul` (2-D f32 CPU) is satisfied by the
SIMD-backed path; GPU/CUDA rows are `experimental`. Add cross-implementation
differential tests (CPU vs GPU vs CUDA) before any GPU row is marked parity.

## 4. softmax

2D, ND, SIMD (`softmax_backward`), GPU (`softmax_resident`), CUDA
(`softmax_rows`/`softmax`) — 5 implementations. Same decision shape as matmul:
CPU row → SIMD-backed; devices experimental; add cross-device tests.

## 5. Reductions (sum/mean/max/var)

2D (`sum_axis`, `mean_axis`, `max_axis`, `var_axis`), ND (`sum`), GPU
(`reduce_sum/mean/max`), CUDA (`deterministic_reduce_sum/mean`). CUDA is
explicitly deterministic; CPU 2D path must document its accumulation order
(see SCOPE.md determinism rule).

**Decision**: parity rows for `sum`/`mean` on CPU use the deterministic order
guarantee; CUDA keeps its own determinism; the registry records
`deterministic = "stricter-than-torch"` where applicable.

## 6. RoPE

ND (`rope`, `rope_portable`), SIMD (`rope_apply_heads`), CUDA (`rope`).
No CPU-2D public op. **Decision**: one registry row (`rope`), ND+SIMD
implementations; CUDA experimental.

## 7. Cross-entropy

ND (`cross_entropy`), SIMD (`softmax_cross_entropy_loss`), GPU
(`cpu_cross_entropy`), CUDA (`cross_entropy_grad`). 4 implementations of one
loss. **Decision**: SIMD is the reference kernel for CPU parity; others
experimental until cross-tested.

## 8. Embedding

2D (`embedding`), SIMD (`cpu_embed`/`embed_backward`), GPU (`embed`),
CUDA (`embed`). Same decision pattern as above.

## 9. Norm kernels (layer/rms)

2D `layer_norm`, ND `layernorm`/`rmsnorm`, SIMD backward kernels, GPU
`rms_norm`, CUDA `rms_norm` + `rms_norm_gain_backward`. Same pattern.

## 10. BLAS primitives (dot/axpy/scal/gemm)

SIMD portable, SVE, AMX, and CUDA each define their own
`dot/axpy/scale/gemm` — plus `SimdBackend` trait in core. **Decision**: keep
`SimdBackend` as the CPU dispatch point; no new backends without a registry row.

## Registry consequence

Every `tensor-operators.toml` row carries `impl = "2d" | "nd" | "simd" |
"gpu" | "cuda" | "prototype"` (list allowed). The coverage matrix and the CI
gate consume this field, so a duplication is not a defect by itself — an
unmapped duplication is.
