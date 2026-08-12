# Design Specification: scirust-tensor

## 1. Crate/module architecture

The **scirust-tensor** module is structured as a suite of specialized crates, guaranteeing a clear separation between algebra logic, planning and execution.

*   **scirust-tensor-core**: Defines the `TensorND` (N-dimensional) type, the shape types (Shapes) and stride-manipulation primitives. It is the common foundation with no heavy dependencies.
*   **scirust-tensor-einsum**: Contains the Einstein-style signature parser (e.g. `"ij,jk->ik"`) and the logic for reducing to binary contraction operations.
*   **scirust-tensor-contraction**: Implements the **Contraction Planner**. It decides the optimal multiplication order to minimize FLOPs and memory. Contains the base CPU/SIMD kernels.
*   **scirust-tensor-compile**: The "graph compiler". It transforms a sequence of operations into an optimized execution graph (redundancy elimination, operator fusion).
*   **scirust-tensor-runtime**: Lightweight execution engine. It manages buffer allocation and execution of compiled graphs, compatible with the SRT1 format.
*   **scirust-tensor-examples**: Demonstrations (Transformer Multi-Head Attention via einsum).

**Dependencies**: `runtime` -> `compile` -> `contraction` -> `einsum` -> `core`.
**Hardware**: Core/Einsum/Planner are 100% CPU. The `contraction` kernels and the `runtime` are GPU-extensible.

## 2. Main Rust types

```rust
use std::collections::HashMap;

/// N-dimensional tensor with explicit stride management for determinism.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorND {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

/// Representation of a parsed einsum operation.
pub struct EinsumPattern {
    pub inputs: Vec<Vec<char>>,
    pub output: Vec<char>,
}

/// A contraction plan is a sequence of computation steps.
pub struct ContractionPlan {
    pub steps: Vec<ContractionStep>,
}

pub enum ContractionStep {
    /// Multiplication of two tensors with optional reorganization.
    Contract {
        left: usize,
        right: usize,
        indices_left: Vec<char>,
        indices_right: Vec<char>,
        out_indices: Vec<char>,
    },
    /// Unitary operation (e.g. sum over an axis).
    Reduce {
        input: usize,
        axis: usize,
    },
}

/// Node of an optimized operation graph.
pub enum TensorOp {
    MatMul(usize, usize),
    Add(usize, usize),
    ReLU(usize),
    Fused(FusedOp),
}

/// Node of a fused operation graph.
pub enum FusedOp {
    /// MatMul + Add Bias + ReLU fused into a single memory pass.
    LinearReLU {
        input_idx: usize,
        weight_idx: usize,
        bias_idx: usize,
    },
    /// Optimized contraction.
    OptimizedContraction(ContractionPlan),
}

pub struct TensorGraph {
    pub ops: Vec<TensorOp>,
    pub buffers: Vec<TensorND>,
}
```

## 3. Complete tensor algebra pipeline

1.  **Signature parsing**: The string `"bij,bjk->bik"` is transformed into an `EinsumPattern`. Dimension consistency is checked.
2.  **Contraction Planning**: For more than 2 tensors, a greedy algorithm (or exhaustive for small graphs) computes the FLOP cost of each possible order (e.g. `(A*B)*C` vs `A*(B*C)`).
3.  **Graph Construction**: The operations (matmul, transpose, add) are inserted into a `TensorGraph`.
4.  **Optimization and Fusion**:
    *   **Permute Fusion**: If an axis permutation precedes a MatMul, the two are fused by manipulating the strides inside the GEMM kernel.
    *   **Operator Fusion**: `Linear -> Bias -> ReLU` patterns are identified and replaced by a single `FusedOp` kernel.
5.  **CPU Execution**: Use of `scirust-simd` for tiled (blocking) kernels guaranteeing bit-for-bit determinism (fixed summation).
6.  **GPU Execution**: If available, dispatch to WGSL shaders (`wgpu`) or Tensor Core kernels (`cuBLAS`).

## 4. MVP version (v1)

*   **Features**: Binary einsum, automatic transposition, optimized CPU kernels via Rayon (deterministic).
*   **Rust API**:
```rust
use scirust_tensor_einsum::einsum;

let a = TensorND::rand(&[10, 20]);
let b = TensorND::rand(&[20, 30]);

// C[i, k] = sum_j A[i, j] * B[j, k]
let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
```
*   **Expected metrics**: Parsing overhead < 1ms, CPU GEMM latency competitive with `scirust-core`.

## 5. Advanced version (v2)

*   **Automatic Planner**: Support for N-tensor einsum (e.g. `"ij,jk,kl->il"`) with optimal contraction-path search.
*   **XLA-like compilation**: Generation of a reusable static execution plan for inference.
*   **Automatic operator fusion**: Multi-layer fusion heuristic.
*   **Full GPU support**: wgpu kernels optimized for binary contractions.
*   **Kernel JIT**: backend to design and test; no dummy MIR driver is shipped.

## 6. Metrics to track

*   **Performance**: GFLOPS, p50/p99 latency.
*   **Memory**: Number of intermediate buffers saved by fusion.
*   **Optimization**: Number of operations eliminated from the initial graph.
*   **Determinism**: Fingerprint (hash) of the output bit-for-bit identical on 1 and N threads.

## 7. Determinism and SRT1

*   **Reduction order**: All reductions (sums) use a fixed order (generally increasing by index) to avoid float instabilities related to associativity.
*   **Frozen Graph**: The optimized graph is serialized into the **SRT1** format, including the shapes and the chosen kernel types.
*   **Oracle Validation**: Each complex contraction plan is validated by a "naive" oracle (nested loops) during integration tests.

## 8. int8 quantization and QSR1

*   **Tensor quantization**: Storage of scales and zero points in the QSR1 format.
*   **int8 einsum**: Systematic `i32` accumulation to avoid overflow, followed by deterministic fixed requantization.
*   **Validation**: Each quantized operation is compared to its f32 equivalent via oracle with bounded error.

## 9. Technical risks

*   **Complex XLA-like compiler**: Difficulty of handling all fusion cases. *Mitigation: Start with predefined patterns.*
*   **GPU fusion**: Requires manual writing of complex WGSL kernels. *Mitigation: Use kernel templates.*
*   **Cross-architecture determinism**: Out of scope; focus on cross-thread stability on a single machine.

## 10. Validation checklist

*   [ ] Unit tests for einsum (parsing and execution).
*   [ ] Integration tests for contraction planner (FLOPs optimality).
*   [ ] Determinism tests (stable fingerprint).
*   [ ] CPU oracle validation (loop-based reference).
*   [ ] Performance benchmarks (GFLOPS).
