<img width="1273" height="671" alt="image" src="https://github.com/user-attachments/assets/c36c292c-5893-44c2-94d9-3895ec0749e8" />






# SciRust 🦀

> A pure-Rust deep learning framework — SIMD CPU kernels, reverse-mode
> autograd, batch normalization, convolutions, and data parallelism.
> (GPU/WGSL kernels exist in-tree but are archived, not wired — see Status.)
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

- **Deep-learning core + reverse-mode autodiff** — 683 passing workspace tests (0 failures; measured 2026-06-12); an MLP reaches 97.70% on MNIST.
- **(Archived, not in build) Portable GPU / Tensor Core** — a cuBLAS-backed BF16 matmul once reached ~63 TFLOPS on an NVIDIA Jetson Thor (aarch64), validated against a CPU oracle. ⚠ *This is a historical result, not a current capability: the kernels live in [`archive/scirust-gpu/`](archive/scirust-gpu/) outside the workspace and are not reproducible from today's build. Re-wiring is roadmap item P2.2; see `scirust_complete_audit_report.md` §5.*
- **Deterministic inference runtime** — bit-exact forward (a 64-bit output fingerprint identical across thread counts and processes), bounded latency (p99/p50 ~1.15), and architecture-agnostic reconstruction from a plain-text manifest plus an SRT1 weight file.
- **Deterministic int8 quantization for embedded** — weight-only int8 is lossless and 4x smaller; a fully-integer calibrated pipeline reproduces the float model bit-for-bit; a true integer convolution and a portable QSR1 / QModel artifact; an aarch64 NEON int8 kernel ~10x faster and bit-exact against the scalar reference; separable depthwise + pointwise convolutions in deterministic int8.
- **Symbolic regression** — a hybrid genetic-gradient engine recovers closed-form laws (structure and constants) from data, fitting constants with the framework's own symbolic differentiation.
- **Evolutionary optimization** (`scirust-evo`) — NSGA-II recovers the ZDT1 Pareto front to within ~1e-3; the simplified single-objective optimizers are honest about their limits (see the report).

## What's in it?

```
✓ Reverse-mode autograd        ✓ Conv2d / MaxPool2d / BatchNorm1d
✓ SIMD CPU kernels (AVX2/SSE2/NEON) ✓ Deterministic int8 quantization
✓ Adam / SGD optimizers        ✓ Data parallelism (1 tape per thread)
✓ Lazy graph compilation       ✓ MNIST IDX reader + DataLoader
✓ safetensors persistence      ✓ Pure Rust, no FFI
```

## Quick start (60 seconds)

No code to copy. Install the unified `scirust` CLI and run a command:

```bash
git clone https://github.com/CHECKUPAUTO/scirust && cd scirust
cargo install --path scirust-cli      # provides the `scirust` binary

scirust help                          # list every command, grouped
scirust info                          # capabilities & determinism guarantees
scirust quickstart                    # train a demo classifier (deterministic) → 4/4
scirust som train                     # train the ownership model; accuracy vs baseline
scirust evo                           # minimize a function with a seeded genetic algorithm
scirust diff "x^2 + 3*x"              # symbolic derivative → ((2 * x) + 3)
scirust solve "x^2 - 4"               # real roots → { -2, 2 }
scirust eval "2*x + 1" x=3            # evaluate → 7
scirust analyze src/main.rs           # ownership analysis of a real Rust file
scirust analyze src/main.rs --sarif   # same, as SARIF 2.1.0 for CI code scanning
scirust verify emit  model.qsr1 model.proof    # seal an inference certificate
scirust verify verify model.proof model.qsr1   # re-check it bit-for-bit
```

`scirust quickstart` prints a decreasing loss and reaches 4/4 on a
non-linearly-separable task — proof the autograd tape, Adam, and the layers
work together. Same seed ⇒ identical numbers, every run.

No `cargo install`? Run any command in place with
`cargo run -p scirust-cli -- <command>`.

## Library API (for embedding)

The CLI is a thin layer over the crates; embed them directly when you need
full control. The 2→8→2 classifier the quickstart trains, in code:

```rust
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};

let mut rng = PcgEngine::new(42);
let mut model = Sequential::new()
    .add(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
    .add(ReLU::new())
    .add(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));
let loss_fn = CrossEntropyLoss::new();
let mut opt = Adam::new(0.05);

let tape = Tape::new();
let x = tape.input(Tensor::from_vec(vec![0.0, 1.0], 1, 2));
let y = tape.input(Tensor::from_vec(vec![0.0, 1.0], 1, 2)); // one-hot
let logits = model.forward(&tape, x);
let loss = loss_fn.forward(&tape, logits, y);
tape.backward(loss.idx());
opt.step(&model.parameter_indices(), &tape);
model.sync(&tape);
```

Add a single crate to your own `Cargo.toml`:

```toml
[dependencies]
scirust-core = { path = "path/to/scirust-core" }
```

> GPU note: `scirust-gpu` currently exposes CPU-validated stubs only; the
> WGSL/cuBLAS kernels are preserved in `archive/scirust-gpu/` outside the
> build (`--features wgpu` compiles nothing extra). Re-wiring them is
> tracked in `docs/INDUSTRIAL_ROADMAP.md` (P2.2).

## Architecture

```
scirust-core/    Core compute, autograd, layers (~12k loc)
scirust-simd/    SIMD CPU kernels (AVX2, SSE2, NEON)
scirust-gpu/     CPU-validated GPU stubs (kernels archived in archive/)
scirust-som/     Ownership Model: real-Rust analyzer + Transformer pipeline
examples/        Quickstart, MNIST training, benchmarks
```

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — Train a 2-class classifier in 50 lines
- [`docs/MNIST.md`](docs/MNIST.md) — Real MNIST training with data parallelism
- [`docs/GPU.md`](docs/GPU.md) — Activate GPU routing for Conv2d
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — How the autograd tape works
- [`docs/REFERENCE.md`](docs/REFERENCE.md) — Exhaustive command/binary/API reference
- [`scirust-som/README.md`](scirust-som/README.md) — Ownership Model (real-Rust analyzer)

## Status

| Feature | Status |
|---|---|
| MLP training | ✅ Stable |
| CNN (Conv2d + MaxPool) | ✅ Stable |
| BatchNorm | ✅ Stable |
| Dropout | ✅ Stable |
| Data parallelism (CPU multithread) | ✅ Stable |
| Transformer (MHA, Encoder, Decoder) | ✅ Stable |
| GQA & KV-Cache | ✅ Stable (GQA + `infer_step` with cache, 6 tests) |
| RoPE embeddings | ✅ Stable |
| RNN / LSTM | ✅ Stable (`nn/lstm.rs`, `forward_sequence`, 7 tests) |
| Flash Attention | ✅ Stable (`nn/transformer/flash_attention.rs`, 4 tests vs dense-attention oracle) |
| Conv2dTranspose | ✅ Stable (`nn/conv2d_transpose.rs`, 7 tests) |
| Mixed precision (fp16) | ✅ Stable (`autodiff/mixed_precision.rs`, 3 tests) |
| Checkpointing (save/resume training) | ✅ New |
| DataLoader (batching, shuffle, prefetch) | ✅ New |
| ONNX export | ✅ New |
| Automatic Mixed Precision (AMP) | ✅ New |
| Differential Privacy (DP-SGD) | ✅ New |
| Model pruning (magnitude, structured, LTH) | ✅ New |
| Distributed training (all-reduce) | ✅ New |
| TensorBoard / CSV logging | ✅ New |
| Neural Architecture Search (NAS) | ✅ New |
| Advanced optimizers (RMSprop, AdamW, LAMB) | ✅ New |
| Fused ops (matmul+SiLU, matmul+GELU, etc.) | ✅ New |
| HPC im2col (cache-aware) | ✅ New |
| SOM — real-Rust ownership analyzer (`som-analyze`) | ✅ New (type-aware Copy/move; see `scirust-som/README.md`) |

> **Not included yet (no claim).** GPU execution is **not** part of the
> build: `scirust-gpu` ships CPU-validated stubs only, and the WGSL/cuBLAS
> kernels are preserved in `archive/scirust-gpu/` outside the workspace.
> Re-wiring a tested wgpu path is tracked in
> [`docs/INDUSTRIAL_ROADMAP.md`](docs/INDUSTRIAL_ROADMAP.md) (P2.2). The
> table above lists only what ships and is tested today.


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
