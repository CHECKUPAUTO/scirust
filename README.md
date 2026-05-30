<img width="1273" height="671" alt="image" src="https://github.com/user-attachments/assets/c36c292c-5893-44c2-94d9-3895ec0749e8" />






# SciRust 🦀

> A pure-Rust deep learning framework — SIMD CPU, tiled GPU compute via WGSL,
> reverse-mode autograd, batch normalization, convolutions, and data parallelism.
> No C++, no Python, no FFI — just Rust from top to bottom.

## Why?

Existing Rust ML libraries either wrap libtorch (`tch`), wrap ONNX runtime,
or are research toys. SciRust is the middle path: a real framework with
real ops, but written entirely in Rust so you can read every line of compute,
modify it, and trust it.

It's not the fastest framework on the planet. It's the one you can **fully
understand**, modify safely, and extend without crossing language boundaries.

## What's in it?

```
✓ Reverse-mode autograd        ✓ Conv2d / MaxPool2d / BatchNorm1d
✓ SIMD CPU kernels (AVX2/NEON) ✓ Tiled WGSL GPU compute (wgpu)
✓ Adam / SGD optimizers        ✓ Data parallelism (1 tape per thread)
✓ Lazy graph compilation       ✓ MNIST IDX reader + DataLoader
✓ safetensors persistence      ✓ Pure Rust, no FFI
```

## Quick start (60 seconds)

Train a 2-class classifier on synthetic data:

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::nn::{
    PcgEngine, Module, Sequential, Linear, ReLU,
    KaimingNormal, Zeros,
};
use scirust_core::nn::loss::{Loss, strict::CrossEntropyLoss};

fn main() {
    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // Toy dataset: 2 clusters
    let x = Tensor::from_vec(vec![1.0, 1.0,  -1.0, -1.0,  2.0, 2.0,  -2.0, -2.0], 4, 2);
    let y = Tensor::from_vec(vec![1.0, 0.0,  0.0, 1.0,  1.0, 0.0,  0.0, 1.0], 4, 2);

    let mut opt = Adam::new(0.05);
    for epoch in 0..100 {
        let tape = Tape::new();
        let xv = tape.input(x.clone());
        let yv = tape.input(y.clone());
        let logits = model.forward(&tape, xv);
        let loss = CrossEntropyLoss.forward(logits, yv);
        loss.backward();
        opt.step(&model.parameter_indices(), &tape);
        model.sync(&tape);

        if epoch % 20 == 0 {
            println!("epoch {epoch}: loss = {:.4}", tape.value(loss.idx()).data[0]);
        }
    }
}
```

That's it. No GPU setup, no `unsafe`, no manual memory management.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirust-core = { path = "path/to/scirust-core" }

# Optional: GPU support via wgpu
scirust-gpu  = { path = "path/to/scirust-gpu", optional = true }
```

Build with GPU support:
```bash
cargo build --features wgpu
```

## Architecture

```
scirust-core/    Core compute, autograd, layers (~12k loc)
scirust-simd/    SIMD CPU kernels (AVX2, SSE2, NEON)
scirust-gpu/     WGSL kernels for wgpu (im2col, sgemm, elementwise)
examples/        Quickstart, MNIST training, GPU benchmark
```

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — Train a 2-class classifier in 50 lines
- [`docs/MNIST.md`](docs/MNIST.md) — Real MNIST training with data parallelism
- [`docs/GPU.md`](docs/GPU.md) — Activate GPU routing for Conv2d
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — How the autograd tape works

## Status

| Feature | Status |
|---|---|
| MLP training | ✅ Stable |
| CNN (Conv2d + MaxPool) | ✅ Stable |
| BatchNorm | ✅ Stable |
| Dropout | ✅ Stable |
| Data parallelism (CPU multithread) | ✅ Stable |
| GPU forward (wgpu) | ✅ Stable |
| GPU backward | ✅ Stable (bolt-opt-autodiff) |
| Transformer (MHA, Encoder, Decoder) | ✅ Stable |
| GQA & KV-Cache | ✅ Stable (GQA + infer_step avec cache) |
| RoPE embeddings | ✅ Stable |
| RNN / LSTM | ✅ Stable (module lstm.rs avec forward_sequence, 7 tests) |
| Flash Attention | ✅ Stable (module flash_attention.rs) |
| Conv2dTranspose | ✅ Stable (module conv2d_transpose.rs) |
| Mixed precision (fp16) | ✅ Stable (module mixed_precision.rs, 3 tests) |

## Architecture — Recent Refactoring (2026-05)

### SciRust Workspace Structure

```
scirust/
├── scirust-core/        # Core: autodiff, tensor, nn, compute backend
├── scirust-simd/        # SIMD kernels (portable, AVX2, NEON)
├── scirust-gpu/         # GPU backends (wgpu, CUDA)
├── scirust-learning/    # Learning algorithms (RL, control, finance)
├── scirust-reasoning/   # Symbolic reasoning, prob逻辑
├── scirust-symbolic/    # Symbolic computation
├── scirust-autodiff/    # Autodiff implementations (tape, forward, reverse)
├── scirust-bridge/      # FFI bridges (Python, Ollama)
├── scirust-gpu-macros/ # GPU proc-macros
├── scirust-simd-macros/ # SIMD proc-macros
└── scirust-macros/      # Common proc-macros
```

### Soullink-Node Refactoring

The `soullink-node` crate has been reorganized:

```
soullink-node/
├── soullink-node/       # Main brain implementation (v6.0)
│   ├── src/
│   │   ├── brain.rs             # Core brain (33K neurons, modules)
│   │   ├── neuron.rs            # Neuron struct + dynamics
│   │   ├── synapse.rs           # Synapse struct + STDP
│   │   ├── ssm_cortex.rs        # Mamba SSM encode/forward/decode
│   │   ├── meta_cortex.rs       # Self-model MLP + closed-loop control
│   │   ├── evolution.rs         # Genome, fitness, Pareto, checkpoint
│   │   ├── self_modify/         # Auto-code-modify engine
│   │   ├── script_engine.rs     # Rhai sandbox
│   │   └── ... (other modules)
│   ├── Cargo.toml
│   └── Cargo.lock
├── soullink-ssm/        # Mamba-style SSM implementation
├── hnn/                 # Hybrid Neural Network modules
├── jepa/                # JEPA (Joint Embedding Predictive Architecture)
└── integration-tests/   # E2E integration tests
```

### Deleted Crates (archived or moved)

- `scirust-burn-bridge/` — Burn ↔ SciRust inference bridge
- `scirust-distributed/` — Distributed computing (moved to mesh networking)
- `scirust-genetic/` — Genetic algorithms (integrated into evolution engine)
- `scirust-inference/` — Inference layers (merged into scirust-core)
- `scirust-jit/` — JIT compilation (replaced by scirust-rustc-driver)
- `scirust-probability/` — Probabilistic models (moved to scirust-learning)
- `scirust-quantum/` — Quantum circuits (archived)
- `scirust-web/` — WASM bindings (moved to separate project)

### Migration Notes

If you have code using deleted modules:

1. **Evolution** → Use `soullink-node::evolution::EvolutionEngine`
2. **Self-modify** → Use `soullink-node::self_modify::SelfModifyEngine`
3. **Script engine** → Use `soullink-node::script_engine::ScriptEngine`
4. **Inference** → Use `scirust-core::nn` modules

See the individual crate READMEs for migration guides.

## License

MIT

## Contributing

PRs welcome. Please run the full quality gate before submitting:

```bash
cargo check --workspace --lib
cargo clippy --workspace --lib -- -D warnings
cargo test --workspace --lib
```

For features touching the autograd tape, include a comparison against
PyTorch numerical gradients in your tests.
