# Tensor Type Inventory (SciRust)

Audited 2026-08-04. Every tensor-like public type in the workspace, its
storage, dtype, error surface, and device. Source: `SCIRUST_CURRENT_STATE.md`
and direct code audit.

| # | Type | Crate / file | Storage | Dtype | Device | Errors | Public ops surface |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `TensorND` (prototype) | `scirust-tensor-core/src/lib.rs` (236 lines) | `Vec<f32>` + `shape: Vec<usize>` + `strides: Vec<usize>` | f32 | CPU | `Result<_, String>` | `try_new`, `new`, `zeros`, `scalar`, `ndim`, `size`, `is_empty`, `try_offset`, `offset`, `try_get`, `get`, `reshape` |
| 2 | `TensorND` (core) | `scirust-core/src/tensor/tensor_nd.rs` (689 lines) | `Arc<[f32]>` + shape | f32 | CPU | `SciRustError` | `new`, `zeros`, `ones`, `from_vec`, `shape`, `ndim`, `numel`, `is_empty`, `get`, `set`, `data_mut`, `reshape`, `flatten`, `flatten_from`, `transpose`, `slice_axis`, `can_broadcast_to`, `broadcast_shape`, `matmul_shape`, `broadcast_to`, `from_tensor_2d`, `to_tensor_2d`, `unfold`, `abs_max`, `frob_norm`, `from_matrix`, `is_contiguous`, `to_contiguous` |
| 3 | `Tensor` (2D autodiff) | `scirust-core/src/autodiff/reverse.rs` (6892 lines) | `Vec<f32>` (rows×cols) | f32 | CPU (+`matmul_gpu` engine hook) | `SciRustError` | elementwise/reduction/matmul/linear/embedding/etc. (see OPERATOR_INVENTORY.md) |
| 4 | `Tape` / `Var<'t>` | `scirust-core/src/autodiff/reverse.rs` | graph of ops over `Tensor` | f32 | CPU | `SciRustError` | `backward`, `value`, `grad`, `set_seed`, `set_grad_enabled`, ... |
| 5 | `NdTape` / `NdVar<'t>` | `scirust-core/src/autodiff/nd.rs` (1984 lines) | graph of ops over core `TensorND` | f32 | CPU | `SciRustError` | see OPERATOR_INVENTORY.md (nd stack) |
| 6 | `Tensor3D` / `Var3D<'t>` | `scirust-core/src/tensor/tensor3d.rs` | wrapper over 2D `Tensor` (b, s, d) | f32 | CPU | `SciRustError` | `new`, `zeros`, `shape`, `input_3d`, `as_var`, `from_var` |
| 7 | `Dual` | `scirust-autodiff/src/lib.rs` | `(value: f64, deriv: f64)` | f64 | CPU | n/a (asserts) | `powi`, `powf`, `sqrt`, `exp`, `ln`, `sin`, `cos`, `tan`, `abs` |
| 8 | `Var<'a>` (scalar fwd) | `scirust-autodiff/src/lib.rs` | f64 node graph | f64 | CPU | n/a | `powi`, `exp`, `sin`, `cos`, ... |
| 9 | `GpuTensor` | `scirust-gpu/src/tensor.rs` | wgpu buffer | f32 | WGPU | `SciRustError`/`Result` | `add`, `mul`, `relu`, `rsqrt`, `softmax`, `matmul`, `rope`, `swiglu`, `ew`, `concat_rows`, `slice_rows`, `embed`, `embed_backward`, `cross_entropy`, reduce ops, `rms_norm`, `softmax_backward`, `plan_fusion` |
| 10 | `GpuMatrix` / `WgpuContext` | `scirust-gpu/src/tensor.rs` | wgpu buffers | f32 | WGPU | `SciRustError` | same kernels as `GpuTensor` |
| 11 | `Chain` | `scirust-cuda/src/chain.rs` | CUDA buffers (2D `CudaMatrix`) | f32 (+bf16 storage) | CUDA | `Result` | `matmul`(at/bt), `add`, `mul`, `relu`, `softmax`, `rms_norm`, `rms_norm_gain_backward`, `rope`, `swiglu`, `embed`, `embed_backward`, `cross_entropy_grad`, `global_grad_norm`, `sgd_step`, `quantize_symmetric_i8`, `to_bf16`, `slice_cols`, `place_cols`, `scale_causal_mask`, `softmax_backward`, `deterministic_reduce_sum/mean` |
| 12 | `CooMatrix` | `scirust-sparse/src/lib.rs` | coo triplets (f64) | f64 | CPU | `Result` | `from_dense`, `to_csr`, `to_csc`, `nnz` |
| 13 | `CsrMatrix` | `scirust-sparse/src/lib.rs` | csr (f64) | f64 | CPU | `Result` | `indices`, `indptr`, `spmv` |
| 14 | `CscMatrix` | `scirust-sparse/src/lib.rs` | csc (f64) | f64 | CPU | `Result` | `factor` (for `SparseLu`), `to_dense` |
| 15 | `SparseLu` | `scirust-sparse/src/lib.rs` | LU factors (f64) | f64 | CPU | `Result` | `solve` |
| 16 | `MatrixView` / `MatrixViewMut` | `scirust-core/src/matrix/view.rs` | borrowed slices + strides | generic T | CPU | n/a | `row`, `col`, `subview`, `get`, iterators |
| 17 | `PinBuffer` / `PooledBuffer` | `scirust-core/src/tensor/pinned.rs` | pinned allocations | generic T | CPU pinned | `PinError` | `as_ptr`, `as_slice`, `borrow`, `release` |
| 18 | `SimdBackend` (trait) | `scirust-core/src/matrix/backend.rs` | n/a (interface) | f32 | CPU SIMD | n/a | `sgemm_f32`, `saxpy`, `sdot`, cholesky, etc. |

## Observations that drive the parity program

- **Dtype**: no `DType` abstraction; every dense type is f32 (f64 in
  `Dual`/sparse). `bf16` exists only as storage conversion (`scirust-simd`,
  `scirust-cuda::to_bf16`).
- **Duplication**: two `TensorND` (prototype vs core) with different storage
  (`Vec` vs `Arc`) and error surfaces (`String` vs `SciRustError`); two
  autodiff stacks; three device kernels with no shared op vocabulary.
- **Error surfaces**: mixed — `String` (prototype), `SciRustError` (core),
  `Result` with crate errors (gpu/cuda/sparse).
- **No tensor type is PyTorch-shaped**: none carry dtype/device/layout as
  first-class metadata; promotion/device dispatch must be added.
- The natural host for parity work is **core `TensorND` + `NdTape`/`NdVar`**
  (strided views, structured errors, ND shapes already present), with the
  prototype crates either merged or deprecated (see `DUPLICATION_MAP.md`).
