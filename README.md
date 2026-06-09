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

## Positioning

A research artifact: a pure-Rust deep-learning and scientific-computing stack built and
validated from scratch — a runtime plus a transpiler layer — rather than a wrapper
over libtorch or ONNX. The guiding discipline is that every primitive is accepted only
after its output matches a reference oracle, with reproducibility measured rather than
assumed (in several cases bit-for-bit). SciRust is not a production competitor to PyTorch,
Burn, or candle; it is a framework you can read, modify, and trust, with its claims backed
by measurements.

## Validated capabilities

Every result below is reproduced by code in this repository and documented in the
technical report ([`paper/SciRust-technical-report.md`](paper/SciRust-technical-report.md)).

- **Deep-learning core + reverse-mode autodiff** — 255 passing tests; an MLP reaches 97.70% on MNIST.
- **Portable GPU / Tensor Core** (NVIDIA Jetson Thor, aarch64) — a cuBLAS-backed BF16 matmul, validated against a CPU oracle, reaches ~63 TFLOPS.
- **Deterministic inference runtime** — bit-exact forward (a 64-bit output fingerprint identical across thread counts and processes), bounded latency (p99/p50 ~1.15), and architecture-agnostic reconstruction from a plain-text manifest plus an SRT1 weight file.
- **Deterministic int8 quantization for embedded** — weight-only int8 is lossless and 4x smaller; a fully-integer calibrated pipeline reproduces the float model bit-for-bit; a true integer convolution and a portable QSR1 / QModel artifact; an aarch64 NEON int8 kernel ~10x faster and bit-exact against the scalar reference; separable depthwise + pointwise convolutions in deterministic int8.
- **Symbolic regression** — a hybrid genetic-gradient engine recovers closed-form laws (structure and constants) from data, fitting constants with the framework's own symbolic differentiation.
- **Evolutionary optimization** (`scirust-evo`) — NSGA-II recovers the ZDT1 Pareto front to within ~1e-3; the simplified single-objective optimizers are honest about their limits (see the report).

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

## Package layout: framework library vs. bundled agent

The `scirust` package exposes the framework as a **library** (`src/lib.rs`), re-exporting
the member crates under `scirust::{core, simd, symbolic, learning, solvers}`. The
deep-learning and scientific-computing capabilities described here live in those crates.

The repository also bundles a small **experimental autonomous-agent binary**,
`openclaw-u` (`src/main.rs`, run with `cargo run --bin openclaw-u`). It is *not* a
component of the framework, is not required to build or use it, and can be ignored
entirely. Parts of the repository were developed with the assistance of **SoulLink**, a
separate agent system maintained outside this repository; like OpenClaw-U, it is not part
of the framework.

## License

SciRust is **source-available** and dual-licensed: free for noncommercial and personal use
under the **PolyForm Noncommercial 1.0.0** license, with a separate **commercial license**
available. See [`LICENSE.md`](LICENSE.md) and [`LICENSING.md`](LICENSING.md). This is not an
OSI-approved open-source license.

## Contributing

PRs welcome. Please run the full quality gate before submitting:

```bash
cargo check --workspace --lib
cargo clippy --workspace --lib -- -D warnings
cargo test --workspace --lib
```

For features touching the autograd tape, include a comparison against
PyTorch numerical gradients in your tests.
