# SciRust Documentation 🦀

Welcome to the documentation for **SciRust**, a Deep Learning and scientific computing framework written entirely in **pure Rust**.

## 1. What is SciRust?

SciRust is a research and development platform for Artificial Intelligence. Unlike many other tools (such as PyTorch or TensorFlow) that rely on complex C++ or Python libraries, SciRust is built from the ground up in Rust.

**Why does this matter?**
- **Total Transparency**: You can read every line of computation code, from the network layer to the mathematical kernel.
- **Security and Reliability**: Benefits from Rust's memory and safety guarantees.
- **Independence**: No complex external dependencies (FFI) are required.

## 2. Philosophy and Key Advantages

SciRust does not try to replace industry giants, but offers a different approach focused on **trust** and **reproducibility**.

### Bit-for-Bit Determinism
In many frameworks, running the same calculation twice can yield slightly different results (due to parallelism). SciRust guarantees **bit-for-bit determinism**: the result will be strictly identical, regardless of the number of processors used. This is crucial for auditability.

### Auditability
Since everything is in Rust, it is easy to verify that the code does exactly what it says. There is no software "black box".

### Validation Oracles
Every mathematical function in SciRust is validated against a "validator oracle" (a trusted reference). We do not assume the result is correct; we measure it.

## 3. Application Domains

SciRust is particularly useful in fields where precision, security, and small software footprint are critical:

- **Embedded Systems (Edge AI)**: Thanks to its low footprint and quantization capabilities (reducing model size), it runs perfectly on small devices.
- **Regulated Sectors (Aerospace, Medical, Finance)**: Where every AI decision must be reproducible and explainable for safety or compliance reasons.
- **Scientific Research**: To discover mathematical laws from data through symbolic regression.
- **Security Audit**: For companies that need to certify their entire computational chain.

## 4. What You Can Achieve

SciRust covers a wide range of modern techniques:

- **Deep Learning**: Building neural networks (MLP, CNN, Transformers) with automatic differentiation (autograd).
- **Symbolic Regression**: Discovering mathematical formulas (e.g., `f(x) = sin(x) + x^2`) from observations.
- **Evolutionary Optimization**: Using nature-inspired algorithms (like NSGA-II) to solve complex problems.
- **int8 Quantization**: Shrinking model size by 4x to fit on small processors without losing accuracy.
- **GPU Acceleration**: Harnessing the power of graphics cards via WebGPU (wgpu) or NVIDIA Tensor Cores (cuBLAS).
- **Physics-Informed Neural Networks (PINN)**: Integration of physical laws (differential equations) directly into the loss function for modeling complex phenomena.
- **Formal Invariant Contracts**: Mathematical guarantees (absence of NaN/Inf, value bounds) for critical applications (medical, aerospace).
- **CSR Tensors and SpMM Kernels**: Memory and computation optimization for sparse models on embedded targets.
- **Secure Enclave Execution (TEE)**: Hardened #![no_std] compatible runtime for isolated execution (TrustZone/SGX) without OS allocator.

## 5. Command Guide

SciRust is primarily used via the terminal with `cargo`, Rust's standard tool.

### Installation
Add this to your `Cargo.toml` file:
```toml
[dependencies]
scirust-core = { path = "..." }
```

### Compile and Test
- **Check the project**: `cargo check --workspace`
- **Run all tests** (over 250 tests validate the framework): `cargo test --workspace`
- **Compile in optimized mode** (recommended for AI): `cargo build --release`
- **Enable GPU support**: Add `--features wgpu` to your commands.

### Execution Examples
- **MNIST Training (handwritten digits)**:
  ```bash
  cargo run --example mnist_classifier --release
  ```
- **Transformer Compression Demo**:
  ```bash
  cargo run -p transformer_compress --release
  ```
- **Matrix Multiplication Benchmark**:
  ```bash
  cargo run -p scirust-core --example bench_matmul --release
  ```

## 6. Code Example (Quick Start)

Here is how to create and train a very simple model in a few lines:

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Sequential, Linear, ReLU, PcgEngine};

fn main() {
    let mut rng = PcgEngine::new(42);

    // Create a simple model
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // Training loop
    for epoch in 0..100 {
        let tape = Tape::new();
        // ... (data loading and gradient calculation)
        println!("Epoch {}: calculation in progress...", epoch);
    }
}
```

## 7. scirust-tensor — Tensor Algebra and Graph Optimization

The `scirust-tensor` module introduces a high-level abstraction layer for manipulating complex tensors while ensuring maximum performance through graph compilation.

### Why use scirust-tensor?
- **Einsum**: Write complex operations (Multi-Head Attention, Contractions) in a single, readable line of code.
- **Operator Fusion**: Reduce memory access by merging activations and biases directly into the computation kernels.
- **Guaranteed Determinism**: Like all of SciRust, every calculation is bit-for-bit reproducible.

### Example: Multi-Head Attention
```rust
use scirust_tensor_einsum::einsum;

// Einstein signature for attention: Batch, Heads, SeqLen, Dim
// (b, h, i, d) , (b, h, j, d) -> (b, h, i, j)
let attention_scores = einsum("bhid,bhjd->bhij", &[&queries, &keys]).unwrap();
```

### Installation
Add this to your `Cargo.toml`:
```toml
[dependencies]
scirust-tensor-core = { path = "scirust-tensor-core" }
scirust-tensor-einsum = { path = "scirust-tensor-einsum" }
```

## 8. Conclusion

SciRust is the framework of choice for those who prioritize **understanding** and **rigor** over raw speed or the ease of Python. It is a powerful tool for building trustworthy AI, from research to embedded systems.

---
*For more technical details, see the full report in `paper/SciRust-technical-report.md`.*
