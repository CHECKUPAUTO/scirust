# Operator Inventory (SciRust)

Consolidated from a full `pub fn` scan of the workspace (2026-08-04). Groups
operators by family; for each family, which stacks implement it. The registry
`tensor-operators.toml` is the machine-readable form of this document; this
file is the human explanation.

## Conventions

- **2D** = `autodiff::reverse` stack (`Tensor`/`Var<'t>`), `scirust-core`.
- **ND** = `autodiff::nd` stack (`NdVar`/`NdTape`) over core `TensorND`.
- **TND** = prototype `scirust-tensor-core` `TensorND`.
- **SIMD** = `scirust-simd` kernels (slice-level, used as backends).
- **GPU/CUDA** = `scirust-gpu` (wgpu), `scirust-cuda` (`Chain`).
- **SP** = `scirust-sparse`.
- An op "exists" when it has a forward path; autograd presence noted per row.
- "dup" marks families implemented in more than one stack with no shared code.

## Elementwise unary

| Op | 2D | ND | TND | SIMD | GPU | CUDA | Autograd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neg | ✓ | — | — | — | — | — | ✓ 2D |
| reciprocal | ✓ | — | — | — | — | — | ✓ 2D |
| exp | ✓ | ✓ | — | ✓ `exp_inplace` | — | — | ✓ 2D+ND |
| log / ln | ✓ | — | — | — | — | — | ✓ 2D |
| log10 | ✓ | — | — | — | — | — | ✓ 2D |
| sqrt / rsqrt | ✓ | — | — | ✓ | ✓ rsqrt | — | ✓ 2D |
| pow / powf / powi | ✓ (f32) | — | — | — | — | — | ✓ 2D |
| sin / cos / tan / asin / acos / atan / atan2 | ✓ | — | — | — | — | — | ✓ 2D |
| sinh / cosh / tanh | ✓ | — | — | — | — | — | ✓ 2D |
| sigmoid | ✓ | ✓ | — | ✓ scalar | ✓ | — | ✓ 2D+ND |
| silu / swiglu | ✓ simd | — | — | ✓ scalar | ✓ swiglu | ✓ swiglu | via SIMD kernels |
| relu | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ 2D+ND |
| gelu | — | — | — | ✓ scalar | — | — | via SIMD backward |

## Elementwise binary

| Op | 2D | ND | TND | SIMD | GPU | CUDA | Autograd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add (incl. add_assign, add_scaled, add_bias) | ✓ | ✓ | — | ✓ (inplace f32/f64) | ✓ | ✓ | ✓ 2D+ND |
| sub (incl. sub_assign) | ✓ | ✓ | — | — | — | — | ✓ 2D+ND |
| mul / hadamard | ✓ | ✓ | — | — | ✓ | ✓ | ✓ 2D+ND |
| div | ✓ | ✓ | — | — | — | — | ✓ 2D+ND |
| atan2 | ✓ | — | — | — | — | — | ✓ 2D |
| broadcast_add / broadcast_mul | ✓ (free fns) | — | — | — | — | — | via unbroadcast |

## Linear algebra

| Op | 2D | ND | TND | SIMD | GPU | CUDA | Autograd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matmul | ✓ (matmul/matmul_bt/matmul_portable/matmul_gpu) | ✓ | — | ✓ sgemm* | ✓ | ✓ (matmul_at/bt) | ✓ 2D+ND |
| bmm / bmm2d | ✓ bmm2d | ✓ bmm | — | — | — | — | ✓ 2D+ND |
| linear (w, b) | ✓ + try_ | — | — | ✓ linear_backward | — | — | ✓ 2D |
| einsum | — | — | ✓ einsum | — | — | — | none |
| contraction plan (pairwise) | — | — | ✓ ContractionPlan | — | — | — | none |
| spmv / spmm_dense | — | — | — | — | — | — | SP (f64) |
| SparseLu::solve | — | — | — | — | — | — | SP (f64) |
| cholesky | — | — | — | ✓ (backend trait) | — | — | none |
| cosine_sim_matrix | ✓ | — | — | — | — | — | ✓ 2D |
| l2_normalize | ✓ | — | — | — | — | — | ✓ 2D |
| dot (sdot/sdot_f64) | — | — | — | ✓ | — | — | none |

## Reductions

| Op | 2D | ND | TND | SIMD | GPU | CUDA | Autograd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sum / sum_axis | ✓ | ✓ sum | — | — | ✓ reduce_sum | ✓ deterministic_reduce_sum | ✓ 2D+ND |
| mean_axis | ✓ | — | — | — | ✓ reduce_mean | ✓ deterministic_reduce_mean | ✓ 2D |
| max_axis | ✓ | — | — | — | ✓ reduce_max | — | ✓ 2D |
| var_axis | ✓ | — | — | — | — | — | ✓ 2D |
| abs_max | — | — | ✓ (core TND) | — | — | — | none |
| frob_norm | — | — | ✓ (core TND) | — | — | — | none |

## Normalization / activation composites

| Op | 2D | ND | SIMD | GPU | CUDA |
| --- | --- | --- | --- | --- | --- |
| softmax / softmax_portable | ✓ | ✓ | ✓ softmax_backward | ✓ | ✓ |
| log_softmax | ✓ | — | — | — | — |
| layer_norm | ✓ | ✓ | ✓ layernorm_backward | — | — |
| rms_norm | — | ✓ | ✓ rmsnorm_backward | ✓ | ✓ (+gain backward) |
| dropout | ✓ | — | — | — | — |
| batch_norm | — | — | ✓ batch_norm_backward | — | — |
| causal_mask | ✓ | — | — | — | — |

## Indexing / shape / view

| Op | 2D | ND | TND | Core TND |
| --- | --- | --- | --- | --- |
| reshape | ✓ | ✓ | ✓ | ✓ (checked) |
| transpose / transpose_2d / permute | ✓ | ✓ (permute, transpose_last2) | — | ✓ transpose |
| broadcast / broadcast_to | ✓ | — | — | ✓ broadcast_to + shape logic |
| slice_rows / slice_cols | ✓ | — | — | ✓ slice_axis |
| gather | — | ✓ | — | — |
| embedding (indices) | ✓ + try_ | — | — | — (SIMD/CUDA embed + backward) |
| cat0 / cat along 0 | — | ✓ | — | — |
| causal_conv | — | ✓ | — | — |
| unfold | — | — | — | ✓ |
| flatten / flatten_from | — | — | — | ✓ |

## Positional / attention / misc

| Op | Where |
| --- | --- |
| rope / rope_portable / rope_apply_heads | ND (autograd), SIMD, CUDA |
| attention (scaled dot-product kernel) | SIMD `attention_backward`, CUDA `scale_causal_mask` |
| conv1d / conv2d (+pool1d/pool2d backward) | SIMD kernels; 2D `max_pool2d` forward |
| cross_entropy / softmax_cross_entropy | ND (autograd), SIMD loss, GPU, CUDA grad |
| mse_loss | SIMD |
| fake_quantize_ste | 2D |
| quantize_symmetric_i8 / qlinear_i8 / bf16 convert | SIMD, CUDA |
| lanczos_eigen_symmetric | SIMD |
| gamma / ln_gamma / digamma | SIMD |

## Observations

- The **2D stack is the richest** forward+autograd surface; the **ND stack**
  covers a smaller, transformer-oriented set (rope, rmsnorm, causal_conv,
  cross_entropy).
- **No op is implemented in the prototype TND crate beyond construction/index/
  reshape** — it is a data container, not an operator library.
- GPU and CUDA share *semantics* with CPU ops but have **no shared code and no
  differential tests** against CPU or against each other.
- Reductions, matmul and softmax each have **3+ independent implementations**
  (CPU-2D, SIMD, GPU, CUDA) — see `DUPLICATION_MAP.md`.
- The registry `tensor-operators.toml` records which SciRust implementations
  exist for each PyTorch operator, with per-row dtype/layout/device/autograd/
  tolerance metadata. `impls` is inventory only.
- Differential evidence from `scirust_core::tensor::parity` is tracked
  separately as reference parity. It does not prove the 2D/ND/SIMD/GPU/CUDA/
  sparse implementation until the harness calls that implementation directly.
