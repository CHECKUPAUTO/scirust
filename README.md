<img width="1273" height="671" alt="image" src="https://github.com/user-attachments/assets/c36c292c-5893-44c2-94d9-3895ec0749e8" />






# SciRust 🦀

> A pure-Rust deep learning framework — SIMD CPU kernels, reverse-mode
> autograd, batch normalization, convolutions, and data parallelism.
> (A portable wgpu GEMM is wired behind the optional `wgpu` feature, tested
> against the CPU oracle on software Vulkan — see docs/GPU.md.)
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

- **Deep-learning core + reverse-mode autodiff** — 1718 passing workspace tests (0 failures; measured 2026-06-19); an MLP reaches 97.70% on MNIST.
- **N-D autograd stack, research-backed and gradient-checked** — a complete causal **decoder LM** (token + positional embeddings, causal attention, fused softmax cross-entropy) that trains end-to-end and overfits a sequence *exactly*; LLaMA-family layers (**RMSNorm**, **SwiGLU**, **RoPE**, **grouped/multi-query attention**); a **LoRA** low-rank adapter (frozen base + trainable `B·A`, gradient-checked); deterministic optimizers (**Adam, AdamW, Lion, Muon, Schedule-Free, AdEMAMix, SOAP** — Adam in Shampoo's eigenbasis, with a from-scratch Jacobi eigensolver — plus **Lookahead, LAMB, Adan**); **exact speculative decoding** and **FlashAttention** (online softmax); a **DeltaNet** delta-rule linear-attention layer, a **Mamba** selective state-space layer (S6 input-dependent scan), a **RetNet** retention layer (recurrent form proven equal to the parallel form), a **GLA** gated-linear-attention layer (data-dependent forget gate), and an **HGRN** gated-linear-RNN token mixer, all linear-time recurrences unrolled on the tape; a **Neural ODE** (backprop through an RK4 solver); a **Physics-Informed Neural Network** that solves a boundary-value problem with the PDE residual in the loss (recovers `sin x` to ~4 decimals). Every op is validated by a finite-difference gradient check. See [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md).
- **Certifiable & reproducible** — **Interval Bound Propagation** and **CROWN** (linear-relaxation back-substitution, provably *tighter* than IBP) give *provable* output bounds and robustness certificates for ReLU MLPs, shown side by side (`scirust certify`); **conformal prediction** gives distribution-free prediction sets with a *guaranteed* coverage level (`scirust conformal`); **temperature scaling** recalibrates over-confident probabilities (lowers the expected calibration error without touching accuracy, `scirust calibrate`); **order-independent floating-point reductions** are bit-identical regardless of thread count; **Wanda** pruning, **SmoothQuant**, **GPTQ** (second-order error-feedback weight quantization — beats round-to-nearest on calibration error, `scirust gptq`), **AWQ** (activation-aware search-based per-channel scaling, `scirust awq`), **BitNet b1.58** (ternary `{−1,0,+1}` weights, ~1.58 bit/weight, with a verified multiplication-free matmul, `scirust bitnet`), and **NF4** (QLoRA's 4-bit NormalFloat — quantile-matched levels that beat uniform 4-bit on Gaussian weights) extend the deterministic int8 path.
- **Portable GPU compute (wgpu, optional)** — a real WGSL `f32` GEMM behind the `wgpu` feature, validated against the CPU oracle on a software Vulkan adapter (Mesa lavapipe) in CI, plumbed into the autograd tape (`Var::matmul_gpu` forward + backward via `WgpuEngine`) **and** into Conv2d's im2col GEMMs (forward + backward). ⚠ *Separately, a historical cuBLAS-backed BF16 matmul once reached ~63 TFLOPS on an NVIDIA Jetson Thor (aarch64); that CUDA path is archived in [`archive/scirust-gpu/`](archive/scirust-gpu/), not reproducible from today's build — see `scirust_complete_audit_report.md` §5.*
- **Deterministic inference runtime** — bit-exact forward (a 64-bit output fingerprint identical across thread counts and processes), bounded latency (p99/p50 ~1.15), and architecture-agnostic reconstruction from a plain-text manifest plus an SRT1 weight file.
- **Certified-deterministic multi-thread training** — `DataParallelTrainer::train_batch_threaded` runs workers across N OS threads yet reduces gradients in a fixed worker order, so the aggregate is **bit-identical for 1/2/4/8 threads** and equal to the sequential path (float addition isn't associative — the reduction order is pinned). CI-tested, including a real autograd backward. No mainstream framework ships this guarantee tested.
- **Deterministic int8 quantization for embedded** — weight-only int8 is lossless and 4x smaller; a fully-integer calibrated pipeline reproduces the float model bit-for-bit; a true integer convolution and a portable QSR1 / QModel artifact; an aarch64 NEON int8 kernel ~10x faster and bit-exact against the scalar reference; separable depthwise + pointwise convolutions in deterministic int8.
- **Symbolic regression** — a hybrid genetic-gradient engine recovers closed-form laws (structure and constants) from data, fitting constants with the framework's own symbolic differentiation.
- **Evolutionary optimization** (`scirust-evo`) — NSGA-II recovers the ZDT1 Pareto front to within ~1e-3; the simplified single-objective optimizers are honest about their limits (see the report).
- **Industrial & automotive monitoring** (`scirust-signal`, `scirust-opcua`, `scirust-mqtt`, `scirust-pdm`, `scirust-mlops`, `scirust-func-safety`, `scirust-integration`) — signal processing (FFT, windows, bearing BPFO/BPFI/BSF diagnostics, order analysis), OPC-UA PLC connectivity with 8 simulated sensors, MQTT SparkPlug B publishing, predictive maintenance (Health Index, RUL, CUSUM, 4 fault detectors), industrial MLOps (drift, shadow deploy, signed OTA), ISO 26262 functional safety (ASIL A-D, fault injection, degraded mode, hash-chained audit log), integration kit (unified Backend/Pipeline/Config/templates), and a dedicated `scirust-industrial` CLI (discover, test, gen-config, scaffold, run, doctor — plus the vertical demos below).
- **State estimation, navigation, water & OT security** (`scirust-estimation`, `scirust-nav`, `scirust-water`, `scirust-ids`, `scirust-func-safety`) — Kalman / IMM / **UD square-root** filters (covariance positive-semidefinite by construction), **GNSS/INS fusion** and **TDOA** multilateration, acoustic **leak correlation** and water-hammer physics (Joukowsky/Korteweg), OT **firmware attestation** and **PLC ladder integrity** (Stuxnet write-set detection) on a tamper-evident hash chain, and a **GMP golden-batch** comparator (DTW alignment + hash-chained 21 CFR Part 11 audit). All reachable from the `scirust-industrial` CLI: `nav-tdoa`, `nav-fusion`, `track-imm`, `track-ud`, `water-leak`, `water-surge`, `ot-firmware`, `ot-plc`, `golden-batch`. Validated on x86 and natively on a **Jetson AGX Thor** via [`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md).
- **Pattern detection** (`scirust-vision`, `scirust-audio`, `scirust-graph`, `scirust-sequential`, `scirust-multivariate`, `scirust-unsupervised`, `scirust-seasonal`, `scirust-nlp-advanced`) — computer vision (CNN, HOG, LBP, Canny, Otsu, NMS), audio (MFCC, chroma, pitch YIN, onset detection), graph patterns (subgraph isomorphism, motif discovery, community detection, betweenness), sequential (HMM, CRF, DTW, KMP/Boyer-Moore), multivariate (PCA, ICA, K-Means++, MDS, CCA), unsupervised (autoencoder, isolation forest, DBSCAN, LOF, GMM), seasonal (STL, ACF/PACF, Mann-Kendall, CUSUM), NLP (NER, LDA, TextRank, MinHash, NaiveBayes).
- **Algorithm creation** (`scirust-automl`, `scirust-synthesis`, `scirust-algogen`, `scirust-codetrans`, `scirust-rl-algo`, `scirust-scaffold`) — AutoML (Bayesian optimization, GP surrogate, model selection, ensembles), program synthesis (30+ ops, sketch-based, bottom-up/top-down/GP/beam search), algorithm generation (sort/search/graph/DP/DaC, complexity analysis), code transformation (AST, pattern matching, 20 optimization rules, refactoring, Rust→Python/C transpilation), RL-based discovery (REINFORCE, Actor-Critic, Q-Learning, MCTS, meta-learning), scaffolding (DSL, code gen, 16 templates, docs).

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
scirust lm                            # train a tiny causal decoder LM (N-D tape) → exact recall
scirust lm --opt lion                 # …with a different deterministic optimizer (adam|adamw|lion)
scirust certify --eps 0.02            # prove a ReLU MLP's output bounds over an L∞ box (IBP)
scirust conformal --alpha 0.1         # conformal intervals with a guaranteed coverage level
scirust som train                     # train the ownership model; accuracy vs baseline
scirust evo                           # minimize a function with a seeded genetic algorithm
scirust diff "x^2 + 3*x"              # symbolic derivative → ((2 * x) + 3)
scirust solve "x^2 - 4"               # real roots → { -2, 2 }
scirust integrate "sin(x)" 0 3.14159  # definite integral (Romberg) → 2
scirust linsolve "2,1;1,3" "3,5"      # solve A·x = b → [0.8, 1.4]
scirust ode "y" 1 0 1                 # dy/dt=y, y(0)=1 → y(1) ≈ e
scirust eval "2*x + 1" x=3            # evaluate → 7
scirust analyze src/main.rs           # ownership analysis of a real Rust file
scirust analyze src/main.rs --sarif   # same, as SARIF 2.1.0 for CI code scanning
scirust verify emit  model.qsr1 model.proof    # seal an inference certificate
scirust verify verify model.proof model.qsr1   # re-check it bit-for-bit

# Industrial monitoring (scirust-industrial CLI)
scirust-industrial discover --simulated                  # browse available PLC sensors
scirust-industrial gen-config --template automotive --stations 3
scirust-industrial scaffold --name line3-monitor --template automotive
scirust-industrial run --config config.json --cycles 100
scirust-industrial doctor --config config.json           # diagnose integration issues

# Vertical demos (deterministic, real crate APIs)
scirust-industrial nav-tdoa                 # TDOA: locate an emitter from arrival-time differences
scirust-industrial nav-fusion --outage 10   # GNSS/INS fusion through a 10-step GNSS outage
scirust-industrial track-imm                # IMM filter swings onto the maneuver model
scirust-industrial track-ud                 # UD square-root Kalman ≡ textbook Kalman, covariance PSD
scirust-industrial water-leak               # acoustic leak correlation (locate a known leak)
scirust-industrial water-surge              # Joukowsky surge + Korteweg wave speed
scirust-industrial ot-firmware              # firmware attestation: clean vs tampered image
scirust-industrial ot-plc                   # PLC integrity + Stuxnet critical-write detection
scirust-industrial golden-batch             # GMP golden-batch comparator (DTW + audit chain)
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

> GPU note: `scirust-gpu` ships a deterministic CPU reference backend (always
> built; the bit-tolerant oracle) and a real portable **wgpu GEMM** behind the
> optional `wgpu` feature, validated against that oracle on a software Vulkan
> adapter (Mesa lavapipe) in CI. Without the feature/adapter the `wgpu` path
> returns `BackendError::Unavailable` — never fabricated output. It is also
> plumbed into the autograd tape (`WgpuEngine` + `Var::matmul_gpu`) and into
> Conv2d's im2col GEMMs, forward and backward, validated end-to-end on lavapipe.
> `cuda` stays out of scope until a GPU runner exists. Next: keep activations in
> VRAM across layers — see `docs/GPU.md` (P2.2).

## Architecture

```
scirust-core/          Core compute, autograd, layers (~12k loc)
scirust-simd/          SIMD CPU kernels (AVX2, SSE2, NEON)
scirust-gpu/           CPU reference backend + real wgpu GEMM (feature `wgpu`, tested on lavapipe)
scirust-signal/        Signal processing: FFT, windows, bearing diagnostics, order analysis
scirust-opcua/         OPC-UA connector: trait + 8 simulated industrial sensors
scirust-mqtt/          MQTT publishing: SparkPlug B payloads, severity classification
scirust-pdm/           Predictive maintenance: Health Index, RUL, CUSUM, fault detectors
scirust-mlops/         Industrial MLOps: drift detection, shadow deployment, signed OTA
scirust-func-safety/   Functional safety: ASIL A-D, fault injection, degraded mode, audit log
scirust-integration/   Integration kit: Backend, Pipeline, config, code templates
scirust-som/           Ownership Model: real-Rust analyzer + Transformer pipeline
examples/              Quickstart, MNIST training, industrial_monitor, benchmarks
```

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — Train a 2-class classifier in 50 lines
- [`docs/MNIST.md`](docs/MNIST.md) — Real MNIST training with data parallelism
- [`docs/GPU.md`](docs/GPU.md) — Portable wgpu compute (status, testing, roadmap)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — How the autograd tape works
- [`docs/REFERENCE.md`](docs/REFERENCE.md) — Exhaustive command/binary/API reference
- [`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md) — Functional acceptance protocol: `scripts/test-protocol.sh` runs every gate, every crate's oracle tests, and a cross-process determinism check in one command
- [`docs/sbom/`](docs/sbom/) — CycloneDX SBOM (reproducible, regenerated in CI & attached to releases)
- [`docs/GROWTH_PLAN.md`](docs/GROWTH_PLAN.md) — Vision, non-negotiable fundamentals, and the ambitious growth plan
- [`docs/RELEASING.md`](docs/RELEASING.md) — Release process & branch-protection runbook
- [`SECURITY.md`](SECURITY.md) — Supply-chain posture, SBOM, accepted advisories
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
| Portable GPU compute + autograd + Conv2d (`scirust-gpu`, feature `wgpu`) | ✅ New (WGSL GEMM, `Var::matmul_gpu` + Conv2d fwd/bwd, oracle-validated on lavapipe) |

> **GPU scope (honest).** A portable wgpu GEMM is wired behind the optional
> `wgpu` feature, tested against the CPU oracle on a software Vulkan adapter
> (Mesa lavapipe) in CI, plumbed into the autograd tape (`WgpuEngine` +
> `Var::matmul_gpu`) and Conv2d's im2col GEMMs (forward and backward), with a
> VRAM-resident matmul-chain API (`GpuChain`) that keeps intermediates on the
> device across GEMMs. It is opt-in, so the default bit-exact guarantee is
> unaffected. Still to do: make tape residency transparent and move im2col onto
> the GPU (P2.2).
> **CUDA** remains out of scope until a hardware GPU runner exists
> (`CudaBackend` returns `Unavailable`); archived WGSL/cuBLAS drafts live
> in `archive/scirust-gpu/`. See [`docs/GPU.md`](docs/GPU.md) and
> [`docs/INDUSTRIAL_ROADMAP.md`](docs/INDUSTRIAL_ROADMAP.md).


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








## scirust-sciagent — Deterministic SLM for Rust

The `scirust-sciagent` crate provides a from-scratch transformer trained on Rust source code (672MB from crates.io + The Stack v2).

**Architecture**: GQA + SwiGLU + RoPE + RMSNorm, from 106K to 7B params.

**Configs**:
| Config | Params | Vocab | Layers | Seq Len |
|--------|--------|-------|--------|---------|
| debug  | 106K   | 256   | 2      | 128     |
| small  | 1.6M   | 8192  | 4      | 256     |
| 350M   | 350M   | 32768 | 24     | 8192    |
| 7B     | 7B     | 32768 | 40     | 8192    |

**Pretrained**: `small` checkpoint (2000 steps, loss 9.01→8.90) at `/tmp/scirust_small_2k/final/`. See `scirust-sciagent/README.md`.

```bash
cargo run --release -p scirust-sciagent --bin sciagent -- --model small \\
  --checkpoint /tmp/scirust_small_2k/final ask "fn main()" --max-tokens 100
```

## License

Dual-licensed: [PolyForm Noncommercial 1.0.0](LICENSE.md) for noncommercial and personal use; commercial license required for any commercial use. See [LICENSING.md](LICENSING.md).
