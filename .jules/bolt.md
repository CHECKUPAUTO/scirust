## 2026-05-14 - [Autodiff Efficiency Boost]
**Learning:** Significant performance gains in reverse-mode autodiff can be achieved by minimizing allocations during the backward pass using in-place updates. Row-major iteration for axis reductions (like sum_axis(0)) provides massive speedups (~17x) due to better cache locality compared to column-wise access.
**Action:** Always prefer in-place gradient accumulation and ensure loop orders match memory layout for reduction operations.
## 2026-05-14 - [Matmul & Autodiff Optimization]
**Learning:** Using `matrixmultiply::sgemm` with explicit strides allows for high-performance matrix multiplications and their derivatives without the overhead of explicit transpositions or temporary allocations. This approach yields ~5x speedup in both forward and backward passes for 512x512 matrices compared to naive triple-loop implementations.
**Action:** Prefer `sgemm` with stride manipulation for linear algebra operations and their backward passes in autodiff engines.
