## 2026-05-14 - [Autodiff Efficiency Boost]
**Learning:** Significant performance gains in reverse-mode autodiff can be achieved by minimizing allocations during the backward pass using in-place updates. Row-major iteration for axis reductions (like sum_axis(0)) provides massive speedups (~17x) due to better cache locality compared to column-wise access.
**Action:** Always prefer in-place gradient accumulation and ensure loop orders match memory layout for reduction operations.
