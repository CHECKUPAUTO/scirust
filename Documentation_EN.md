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
- **Reinforcement Learning (RL)**: Full stack support for Tabular Q-Learning, DQN, and PPO with clipping.
- **Advanced Computer Vision**: ResNet-18/34 architectures and Vision Transformer (ViT) with global pooling.
- **Generative AI (VAE)**: Variational Autoencoders with reparameterization trick for latent generation.
- **Transformers and MoE**: Mixture of Experts layers with Top-k routing for model scalability.
- **Graph Neural Networks (GNN)**: Graph Convolutional Networks (GCN) for structured data.
- **Speech AI and Audio**: Audio encoders and CTC loss function for speech recognition.
- **PEFT Adaptation (LoRA)**: Low-Rank Adaptation for efficient fine-tuning of pre-trained models.
- **Advanced Scientific Computing**: 1D FEM (Finite Element Method) solver for physical equations.
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

## 13. Research → Functions (N-D autograd extensions)

The N-D autograd tape now carries a complete deep-learning stack, every piece
backed by a research paper and a test (gradient check or oracle). See
[`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) (14/20 delivered).

- **Causal decoder LM**, trained end-to-end (token + positional embeddings,
  causal attention, fused softmax cross-entropy); overfits a sequence exactly.
- **LLaMA-family layers**: RMSNorm, SwiGLU, LLaMA block, RoPE, grouped/
  multi-query attention (GQA/MQA).
- **Deterministic optimizers**: Adam, AdamW, Lion, Muon (Newton–Schulz), Schedule-Free, AdEMAMix, and SOAP (Adam in Shampoo's eigenbasis).
- **Certifiable AI**: Interval Bound Propagation **and CROWN** (tighter
  linear-relaxation bounds) — *provable* output bounds and
  a robustness certificate.
- **Reproducible reductions**, order-independent (bit-identical regardless of
  thread count).
- **Exact speculative decoding**; **FlashAttention** (online softmax);
  **DeltaNet** (delta-rule linear attention);
  **Mamba** (selective state-space / selective scan);
  **Neural ODE** (backprop through an RK4 solver); a Physics-Informed Neural Network (PINN) that solves a boundary-value problem with the PDE residual in the loss.
- **Compression**: Wanda (activation-aware) pruning, SmoothQuant, GPTQ (second-order error-feedback int8 weight quantization), AWQ (activation-aware search-based int8 weight quantization).

New CLI commands:
- `scirust certify [--seed N] [--eps E]` — provable ReLU-MLP bounds (IBP **and** CROWN, the tighter linear-relaxation bounds, side by side).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap]` — train the N-D decoder LM.
- `scirust deltanet [--seed N] [--steps S]` — train a single-head DeltaNet (delta-rule linear attention) layer to fit a sequence; reports the MSE reduction.
- `scirust mamba [--seed N] [--steps S]` — train a Mamba selective state-space layer (S6 scan) to fit a sequence; reports the MSE reduction.
- `scirust conformal [--seed N] [--alpha A]` — conformal intervals with a guaranteed, distribution-free coverage level.
- `scirust pinn [--seed N] [--steps S]` — physics-informed network; solve the BVP `u''=−u` (PDE residual in the loss), checked against `sin x`.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — GPTQ int8 weight quantization; reports the calibration-error reduction vs round-to-nearest.
- `scirust awq [--seed N] [--samples S] [--grid G]` — AWQ activation-aware int8 weight quantization; reports the selected scaling exponent and the calibration-error reduction vs round-to-nearest.
