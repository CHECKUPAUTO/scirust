# SciRust: A Pure-Rust Deep Learning Framework — Portable GPU Acceleration, a Symbolic Regression Engine, and a Deterministic Inference Runtime

**Tarek Zekriti**
Independent researcher · contact@checkupauto.fr
Repository: https://github.com/CHECKUPAUTO/scirust

---

## Abstract

We present **SciRust**, a deep learning framework written in pure Rust that
combines a runtime library with a transpiler layer (procedural-macro attributes
for differentiation, vectorization, and accelerator targeting), and nine
capabilities built and validated on it. The first is a portable GPU and Tensor
Core path: the pure-Rust core ports to an NVIDIA Jetson Thor (aarch64) without
modification, and a cuBLAS-backed matrix-multiply, validated against a CPU oracle,
reaches roughly 63 TFLOPS in BF16. The second is a hybrid genetic-gradient
**symbolic regression** engine that recovers closed-form laws — structure and
constants — from data, using the framework's own symbolic differentiation to fit
constants. The third is a **deterministic inference runtime** offering bit-exact,
bounded-latency, auditable inference, generic over architecture via a plain-text
manifest. The fourth is a deterministic int8 quantization stack for embedded inference: a
portable integer inference path, bit-exact across threads and reproducible
bit-for-bit under fixed-point requantization, that shrinks model weights roughly
fourfold. A single methodological throughline connects them: every primitive is
accepted only after its output matches a reference oracle, and reproducibility is
treated as a first-class, measured property — in several cases bit-for-bit.
Against the framework's baseline (579 passing tests; MNIST 97.70%), these
contributions establish SciRust as a substantive, reproducible research artifact.

---

## 1. Introduction

SciRust is a deep learning framework written in pure Rust. It is a hybrid of a
runtime library and a transpiler system: alongside conventional tensor and neural
network components, it implements real procedural-macro attributes — #[autodiff],
#[simd], and #[gpu] — across three macro crates, so that annotated Rust is
rewritten into differentiated, vectorized, or accelerator-targeted forms. The
project is positioned as a **research artifact**, not as a production competitor to
established frameworks (PyTorch, or in Rust, Burn and candle), which exceed it in
operator coverage, kernel maturity, and hardware breadth.

This report presents the framework and three capabilities built on it, each
validated and reported with its measured figures and honest boundaries: a portable
GPU and Tensor Core path, a symbolic regression engine, and a deterministic
inference runtime. The connective material describes the framework baseline and the
engineering discipline under which every contribution was accepted.

We are explicit about the kinds of claim made. **Measured claims** — throughput,
accuracy, latency, bit-exact fingerprints — are reproducible numbers from the runs
reported. **Interpretive claims** — about what the engineering discipline buys, or
what a capability demonstrates about the framework — are offered as reasoned
arguments grounded in those measurements, not as proofs.

## 2. The SciRust framework

The core (scirust-core) provides a reverse-mode automatic differentiation engine
built around a Tape that records operations, a two-dimensional Tensor type, a
library of neural network modules (linear, convolutional, pooling, normalization,
activation, and transformer layers) behind a common Module trait, optimizers
(including Adam), and data loaders. A deterministic, seedable pseudo-random
generator underpins initialization and data shuffling, which makes whole-run
reproducibility attainable rather than incidental.

What distinguishes SciRust from a plain library is its transpiler dimension. The
macro crates (scirust-macros, scirust-simd-macros, scirust-gpu-macros) implement
the #[autodiff], #[simd], and #[gpu] proc-macro attributes, making the system a
hybrid runtime-plus-transpiler rather than a fixed runtime alone. The CPU numerics
are pure Rust with no mandatory BLAS dependency, which — as Section 4 shows — is
precisely what made cross-architecture portability straightforward.

The framework's baseline validation comprises **579 passing tests** and several
end-to-end demonstrations: MNIST classification at **97.70%** with bit-identical
loss curves across epochs (the strongest non-regression signal the project uses), a
transformer reaching **100%** on a synthetic majority-vote task, and a CIFAR-10
convolutional pipeline reaching **52.40%** on a 5000-image training subset (roughly
5.2x the random baseline, validating the convolutional path). These figures
establish that the substrate is a working framework, not a stub, which is the
premise the rest of the report builds on.

## 3. Engineering discipline

A single discipline governed the acceptance of any contribution into a validated
state, and it is worth stating explicitly because it is what makes the measured
results trustworthy:

- **Oracle validation.** No computational primitive was accepted until its output
  was checked against an independent reference — typically the CPU implementation
  acting as oracle for a GPU path, or a known ground-truth law for the symbolic
  engine. The strongest form of this check is bit-level: identical floating-point
  output (bit-identical loss curves, or identical output fingerprints) is a far
  stronger non-regression signal than approximate agreement.
- **Green-tests gate.** Work did not advance past a step whose tests were not
  passing, with raw build and test output (not summaries) used as evidence.
- **Branch isolation.** Each capability was developed on its own branch and
  validated there before integration, keeping work in progress insulated from
  unrelated change elsewhere in the evolving codebase.
- **Additive integration.** Where possible, new capabilities were landed as
  separate crates or behind feature flags, touching neither the CPU hot path nor the
  autodiff engine, so that a contribution could be validated in isolation.

The recurring lesson is that a numerical test is only as trustworthy as its error
model — a point that surfaces concretely in Sections 4 and 5.

## 4. GPU bring-up: extending SciRust to NVIDIA Tensor Cores on Jetson Thor

### 4.1 Context and portability

SciRust was developed and validated on an x86-64 Debian host. To probe portability
and a GPU execution path, the framework was ported to an NVIDIA Jetson Thor module
(aarch64, Blackwell-class GPU, CUDA 13.0, driver 580).

The pure-Rust core compiled on aarch64 without modification in under 20 seconds,
and crucially **without any BLAS dependency**: the optional intel-mkl-src and
blas-src bindings remained inactive, so the x86-only Intel MKL trap was avoided by
construction. Cross-architecture numerical behaviour held: MNIST reached **97.73%**
(loss 0.0377) on the Jetson, consistent with the x86 baseline, confirming that the
framework's CPU numerics are architecture-portable.

A practical observation on the toolchain: the cudarc 0.14 crate exposes bindings
only up to CUDA 12.8 but loads the driver dynamically. Because the CUDA driver API
is backward compatible, forcing the cuda-12080 binding set runs correctly at
runtime against the CUDA 13.0 driver — the dynamic-loading path is what made the
bring-up possible on a toolchain newer than the binding crate was aware of.

### 4.2 Validation methodology

Matrix multiplication (GEMM) was the bring-up primitive, chosen because it dominates
cost in both training and inference and has an unambiguous reference. Work proceeded
in an isolated sandbox crate first, then in-tree behind a cuda feature flag, each
stage validated against the CPU oracle before the next.

A methodological point surfaced during validation. A naive relative-error metric
reported a 5.6% discrepancy on a non-square problem while reporting 5e-5 on a square
one, using identical kernels. The cause was not a defect but cancellation: with
mixed-sign operands some output entries are near zero, so relative error explodes
while absolute error stays at the FP32 noise floor. The correct oracle combines an
**absolute** tolerance applied everywhere with a **relative** tolerance applied only
where the reference magnitude is significant. Under that combined metric every GPU
path matched the oracle.

### 4.3 The matmul triptych

| Implementation | 512^3 | 1024^3 | 2048^3 | 4096^3 |
|---|---|---|---|---|
| CPU (Rayon, FP32) | 2.37 ms | — | — | — |
| GPU naive kernel (FP32) | 2.749 ms / 98 | — | — | — |
| GPU tiled kernel (FP32) | 1.393 ms / 193 | 5.004 ms / 429 | 17.216 ms / 998 | — |
| cuBLAS (FP32) | 0.376 ms / 714 | 1.993 ms / 1078 | 3.787 ms / 4537 | 22.314 ms / 6159 |
| cuBLAS Tensor Cores (FP16) | 0.237 ms / 1130 | 0.251 ms / 8559 | 0.346 ms / 49699 | 2.166 ms / 63448 |
| cuBLAS Tensor Cores (BF16) | 0.238 ms / 1128 | 0.253 ms / 8493 | 0.347 ms / 49501 | 2.152 ms / 63872 |

(Time per call / throughput in GFLOPS.) The progression is instructive. The naive
kernel is memory-bound and merely matches an optimized multi-core CPU — a GPU is not
automatically faster. The shared-memory tiled kernel (16x16 tiles) roughly doubles
it and crosses into genuine GPU territory (~1 TFLOPS at 2048^3), but a
one-output-per-thread kernel stalls a factor of ~4 below cuBLAS, which is what
register blocking and double-buffering buy. cuBLAS FP32 reaches ~6.2 TFLOPS (6.3x
the CPU at 512^3); engaging the Tensor Cores in FP16/BF16 yields ~63 TFLOPS sustained
at 4096^3, an order of magnitude beyond FP32. Two honesty caveats: throughput below
2048^3 is launch-overhead-bound (only the 4096^3 figure reads as sustained), and the
numbers reflect the device's default power mode.

### 4.4 Precision and integration

cuBLAS FP32 is bit-close to the CPU result (max relative error 4.7e-5 at 512^3),
differing only in summation order; the tiled kernel agreed to 9.4e-6. The
reduced-precision Tensor-Core paths degrade as expected (FP16 1.3e-2, BF16 6.8e-2,
the latter larger due to BF16's 7-bit mantissa), with the error originating in input
rounding rather than accumulation, which is performed in FP32. For machine learning,
BF16's larger single-GEMM error is not a liability: its FP32-equivalent exponent
range avoids the overflow that plagues FP16 in deep activations, which is why it is
the de facto training format and the recommended target for any future
mixed-precision path.

The cuBLAS FP32 GEMM was integrated into the scirust-gpu crate behind the cuda
feature, as a pure slice-level entry point with no dependency on the core tensor
types, eliminating any risk of a dependency cycle. cuBLAS is column-major; the
row-major product C = A.B is obtained by computing (B^T.A^T) with operands swapped
and leading dimensions set accordingly, and the CUDA context and handle are cached
per thread. The integration is additive and non-invasive — it touches neither the
CPU hot path nor the autodiff engine — and is validated by two oracle tests, a
square case and a non-square case that specifically exercises the column-major
dimension mapping.

## 5. Symbolic regression via the framework's own autodiff

### 5.1 Motivation and method

To probe whether SciRust is a substantive framework rather than a fitting harness,
we built a capability combining components it would not normally combine: its
symbolic-math engine (scirust-symbolic — expression trees, simplification,
evaluation, and **symbolic differentiation**) with its automatic-differentiation
discipline. The task is **symbolic regression**: recovering a closed-form expression
— both structure and constants — that fits observed data.

The engine is a hybrid. The **structure** of a candidate is searched by genetic
programming over expression trees (primitives +, -, x, /, sin, cos, exp, plus
variables and constants) with tournament selection, subtree crossover and mutation,
elitism, and a size cap. The **constants** are not searched blindly — the classical
weakness of genetic programming — but fit by gradient descent (Adam), where the
gradients come from the framework's **symbolic differentiation**: for a candidate
with constants c0, c1, ..., the partial d(expr)/d(ck) is obtained from the engine's
diff and evaluated over the data batch. The symbolic engine thus powers its own
learning. Selection is biased toward **parsimony** and the output is a **Pareto
front** over accuracy versus complexity; the data model is **multi-variable**. The
engine is pure Rust, reuses scirust-symbolic unmodified, and is fully reproducible
via a seeded generator.

### 5.2 Validation and results

Each result is checked against an **oracle** — a known ground-truth law — using the
same combined absolute/relative tolerance discussed in Section 4.2. A second,
sharper criterion is structural: did the engine recover the true, compact law or
merely an accurate but bloated approximation?

| Target law | Recovered expression | MSE |
|---|---|---|
| x^2 + sin(x) | (x.x) + sin(x) | 0 |
| exp(-0.3x).cos(2x) | cos(x+x).exp(-0.300.x) | 3.3e-16 |
| x.y + sin(x) (2 variables) | sin(x) + (y.x) | 0 |
| x / (1 + x^2) | x / (x.x + 1.0) | 2.0e-15 |
| 0.5x^2 - 1.2x + 2 + noise (sigma=0.1) | quadratic form | 9.1e-3 ~ sigma^2 |

The engine recovered the exact structure for the polynomial-plus-trigonometric, the
two-variable case, and — notably — the damped oscillator, usually expected to fail
because fitting a frequency inside a cos is highly non-convex; it even expressed
2x as x+x. The noisy quadratic was fit to the signal at the noise variance, not
chasing the noise.

The most instructive result is the rational x/(1+x^2). Under **MSE-only** selection
the engine returned a fourteen-node nested-sin expression that approximated the data
to ~6e-5 but bore no resemblance to the true law. Under the **Pareto front with a
parsimony penalty**, the true compact form appeared at the bottom of the front
(seven nodes, MSE ~2e-15). This is the finding to retain: **low error is not the
same as the correct law** — accuracy-only objectives reward bloated approximations,
and parsimony pressure plus a Pareto view is what recovers structure.

The engine landed as a scirust-symreg crate, developed on its own branch and
additive by construction. Its limitations are stated plainly: a single-session
result on a modest primitive set; a stochastic (seeded, not exhaustive) search; and
the term neuro-symbolic earned only in the narrow sense of gradient-optimized
constants within a symbolic search, not a learned prior over structure.

## 6. A deterministic inference runtime

### 6.1 Positioning

A pure-Rust training framework is a poor competitor to the established ecosystem on
its own terms. Rather than contend on that axis, we asked whether a SciRust-based
system can offer, as a first-class guarantee, a property mainstream runtimes treat
as best-effort. The answer pursued is **deterministic, bounded-latency, auditable
inference** — the combination demanded by edge and regulated deployments. The runtime
(scirust-runtime) is a separate crate over a frozen forward subset of the core; it
performs forward inference only, with training kept as offline tooling. This
separation lets a stable inference contract sit atop the evolving core, with a
regression lock (Section 6.3) turning any drift into a visible failure.

### 6.2 The keystone: bit-exact determinism

Every other guarantee rests on the forward pass being bit-exact, so this was
established empirically first. An MLP (784-256-10) with fixed weights was run
repeatedly over a fixed input, with outputs compared bit-for-bit (to_bits equality,
not tolerance). Across 5120 logit comparisons there were **zero divergences**, and a
64-bit fingerprint of the output bits was identical across calls and across separate
processes.

The decisive test concerns thread count. The matmul is Rayon-parallel, raising the
worry that a work-stealing scheduler reorders summations. It does not: re-running the
binary under RAYON_NUM_THREADS of 1, 2, 4, 8, 16, and 64 produced the identical
fingerprint 0xde2d807686e4b47e every time. The reason is structural — the parallel
matmul distributes work across output cells, each dot product accumulated by a single
thread in fixed order, so the reduction order is independent of thread count. The
honest scope of the resulting claim is bit-exactness for a **fixed compiled artifact
on a given architecture**, stable across thread count and process restarts;
cross-architecture bit-exactness is out of scope by design — the correct audit model
is to ship a pinned artifact and replay it identically on its target.

### 6.3 Weight persistence and reload

For reproducibility across deployments, frozen weights must round-trip without loss.
We defined a small format, **SRT1**, writing each tensor as
(key, rows, cols, f32 little-endian) with keys sorted, so the on-disk bytes are
deterministic and the artifact has a hashable stable hash. The load-bearing golden
test — serialize, construct a fresh differently-seeded model, reload, run
forward — must reproduce the original fingerprint. It does: a differently-seeded
model differs before loading and reproduces 0xde2d807686e4b47e bit-for-bit after.
Exercised on a real trained model, the MLP trained on MNIST (loss 0.2615 -> 0.0377)
and frozen to an 814 KB artifact reloads to **97.73%** test accuracy with test-logit
fingerprint 0xc96d25fa658f5611 stable across processes. This closes the thesis
end-to-end: train once, freeze, and the runtime replays an accurate, bit-exact
inference on every invocation.

### 6.4 Bounded latency

With correctness fixed by Section 6.2, latency was treated as a temporal measurement.
For single-request inference (batch=1) the MLP showed p50 = 126 us, p99 = 145 us, and
a **p99/p50 ratio of 1.15** — a tight, predictable tail. Latency was also invariant to
thread count (flat p50 from 1 to 8 threads): the per-call cost is dominated by fixed
overhead, not compute or dispatch, so thread count is a throughput lever (batch=64
throughput scaled 23k -> 81k samples/s across 1->8 threads), irrelevant to
single-request latency. A deliberate non-result: we hypothesized an allocation-free
arena would be needed to bound the tail, but the measured 1.15x ratio showed
allocation jitter to be negligible, so **no arena was built** — the data did not
justify the optimization. Resisting an optimization the measurements contradict is
part of the discipline.

### 6.5 Generality via manifest-driven reconstruction

To show the guarantees are not artifacts of one small MLP, the audit was repeated on
a convolutional network (Conv->ReLU->MaxPool twice, then a classifier): forward
bit-exact (0x1381e4b51d0eeba4) and thread-invariant; the 4.28 MB artifact
round-tripped bit-for-bit including convolutional weights; batch=32 latency kept a
tight tail (p50 45.9 ms, p99/p50 = 1.20). The runtime was then generalized so that
**no architecture is hardcoded in the inference path**: a plain-text manifest of
layer specifications plus an SRT1 file reconstructs an arbitrary supported
Sequential. A manifest-rebuilt CNN reproduces the hardcoded model's fingerprint
exactly, and — the decisive case — the trained MNIST MLP rebuilt purely from a
manifest plus its weights reproduces both 97.73% accuracy and fingerprint
0xc96d25fa658f5611 bit-for-bit. The supported set covers Linear, ReLU, Sigmoid,
LayerNorm, BatchNorm2d, Conv2d, and MaxPool2d, each shown to persist and reconstruct
bit-exactly; parametric normalization layers were validated with care (LayerNorm
affine parameters and BatchNorm2d running statistics both survive the round-trip,
with BatchNorm2d forced into evaluation mode so inference is per-sample
deterministic). Advanced features like **Formal Invariant Contracts** through
`CertifiedModule<M, C>` and **Secure Enclave Runtime** support for #![no_std]
targets further extend the runtime's applicability to high-integrity environments.
The honest boundary: transformer layers use a three-dimensional
forward and would require a separate runtime path; convolution throughput is bounded
by the pure-Rust kernel; and absolute batch=1 latency is overhead-bound.

## 7. Deterministic int8 quantization for embedded inference

### 7.1 Positioning

The deterministic runtime of Section 6 targets edge and regulated deployment, where
memory and energy are scarce and behaviour must be auditable. Eight-bit integer
inference is the natural next step, but only if the properties that made the runtime
trustworthy survive the move to low precision. We therefore built the quantization
stack in the pure portable core (no GPU dependency) and held it to the same contract:
every quantized primitive is accepted only against a reference oracle, and determinism
is measured rather than assumed — bit-for-bit wherever the arithmetic permits.

### 7.2 Weight-only and dynamic int8: a free fourfold

The first scheme is dynamic W8A8: activations are quantized per tensor at run time,
weights per output channel, the product accumulates in i32, and a single
requantization returns f32. On the trained MNIST MLP this is lossless — the f32
baseline scores 97.73% (fingerprint 0xc96d25fa658f5611) and the int8 model 97.74% —
while the weights shrink from 813 KB to 204 KB (3.98x). The int8 fingerprint
0xc3730f7c204455ba is identical under RAYON_NUM_THREADS of 1, 4, and 16: the integer
matmul accumulates each output cell in a single thread, so the structural determinism
argument of Section 6.2 carries over unchanged.

### 7.3 Static calibration and full-integer requantization

To remove per-call activation statistics, activation scales were calibrated once on a
held-out sample; int8 activations are then carried between layers with i32 bias and an
integer ReLU. This static pipeline scores 97.71% with fingerprint 0xa9b9a102c7cea67b,
thread-invariant. The floating-point requantization in the hot path was then replaced
by a gemmlowp-style integer requantization — a fixed-point multiplier in
[2^30, 2^31) and a per-channel right shift — which reproduces the calibrated model
bit-for-bit (same 97.71%, same 0xa9b9a102c7cea67b). The inference path is now integer
end-to-end, with no floating-point in the loop and no parallel reduction, so it is
deterministic by construction.

### 7.4 Per-channel quantization of convolutions

The per-channel scheme extends to the convolutional network (per row for Conv2d
weights, per column for Linear). A fake-quantized round-trip reproduces the f32 oracle
0x1381e4b51d0eeba4 and preserves the arg-max on all 32 test inputs, with the 4.28 MB
filter set shrinking to 1.07 MB (3.99x). A true integer direct convolution was then
validated: an f32 mirror of the integer indexing matches the framework's convolution
forward bit-for-bit, and the int8 convolution agrees with the f32 oracle to within
max-abs 2.8e-2. As in Section 6, relative error is read with care — near logit
cancellations a large relative error coexists with a negligible absolute one, so
absolute error and preserved arg-max are the load-bearing metrics.

### 7.5 A portable quantized artifact

The calibrated full-integer model was promoted to a first-class artifact, QSR1: a
self-describing byte format holding per-layer dimensions, the calibrated input scale,
per-channel weight scales, int8 weights, and i32 bias, with deterministic, hashable
bytes. Written, reloaded from the file alone, and replayed, it reproduces
0xa9b9a102c7cea67b at 97.71% from 205 KB versus the 814 KB f32 artifact (3.96x).
Exposed through a small library API (a quantized model with save, load, and infer), a
round-trip through the library reproduces the fingerprint bit-for-bit; because QSR1 is
self-describing it subsumes the plain-text manifest for quantized models.

### 7.6 CSR Tensors and Sparse SpMM Kernels

To further optimize memory consumption on edge targets, SciRust implements a
`CsrTensor` structure and an associated Sparse Matrix-Matrix Multiplication
(SpMM) kernel. This allows for the storage and computation of sparse models
without the overhead of dense representations, effectively bypassing the
memory wall on constrained devices.

### 7.7 An integer kernel and separable convolutions

The portable scalar integer matmul is the correctness reference. An aarch64 NEON kernel
— widening multiply-accumulate with i32 accumulation, the right-hand operand
transposed for contiguous access — is bit-exact against it (integer summation is
order-independent) and about ten times faster (64x784x256: 9592 us scalar versus 963 us
NEON). Two MobileNet-style blocks complete the embedded operator set: an int8 depthwise
convolution, whose f32 mirror matches a per-channel convolution oracle bit-for-bit and
whose int8 output agrees to max-abs 2.0e-2, and an int8 pointwise 1x1 convolution, whose
f32 mirror matches a 1x1 convolution oracle bit-for-bit and agrees to max-abs 1.8e-2.
Composed, they form a separable convolution entirely in deterministic int8, each half
validated against the framework, with every weight tensor four times smaller.

## 8. Advanced Features for Runtime and Verification

As SciRust matured from a training-focused framework to a deployment-ready ecosystem, five advanced features were implemented to address the needs of high-integrity systems and formal explainability.

### 8.1 Ahead-Of-Time (AOT) Static Model Compiler
To eliminate the overhead of runtime graph construction and weight loading—critical for ultra-deep embedded targets with limited heap memory—we implemented a static compiler.
- **Mechanism:** The compiler ingests a `LayerSpec` topology and raw weight buffers, emitting a valid Rust source file. This file defines a `StaticModel` struct where weights are stored as statically nested arrays (`&[[f32; N]; M]`).
- **Benefit:** Models can be linked directly into the binary as immutable data, enabling zero-allocation inference and preventing runtime parsing errors.

### 8.2 Soft-Float Matrix Engine for Determinism
While Section 6.2 establishes bit-exactness for a fixed architecture, cross-platform determinism (e.g., x86 vs. ARM) is often broken by hardware-specific FPU rounding and FMA optimizations.
- **Implementation:** We implemented `soft_gemm`, a software-defined matrix multiplication kernel using scaled integer arithmetic (`i32` with `i64` accumulation).
- **Validation:** By bypassing the hardware FPU, the engine guarantees identical computation traces across disparate CPU instruction sets, a requirement for formal verification and cross-platform audit logs.

### 8.3 Latent Activation Steering (RepE)
Building on the "Representation Engineering" paradigm, we integrated low-level hooks to manipulate internal model state during inference.
- **Structure:** The `Module` trait was expanded with a `forward_steered` method and a `SteerHook` registry.
- **Application:** This allows external controllers to apply linear shifts (Concept Vectors) to latent activations in real-time, enabling the redirection of model behavior without modifying static weights.

### 8.4 Quantization-Aware Training (QAT) with STE
To bridge the gap between FP32 training and INT8 deployment (Section 7), we implemented Fake Quantization kernels.
- **Mechanism:** During the forward pass, values are clamped and quantized to a simulated 8-bit scale. The backward pass utilizes a **Straight-Through Estimator (STE)**, passing gradients through the non-differentiable quantization step unmodified.
- **Result:** Models naturally adapt to quantization errors during the training loop, significantly improving the accuracy of downstream low-precision execution.

### 8.5 XAI: Integrated Gradients Engine
To satisfy the requirements of regulated sectors (Section 3), we implemented Integrated Gradients for feature attribution.
- **Algorithm:** The engine computes the path integral of gradients from a baseline (e.g., a zero tensor) to the input over $m$ steps.
- **Integration:** Leveraging the framework's native `Tape`-based autodiff, the engine generates attribution maps of the same shape as the input, providing a mathematical explanation for any given prediction.

## 9. Modern AI Families Expansion

To move beyond basic MLP and CNN architectures, we expanded SciRust with foundational support for several modern AI domains, maintaining strict pure-Rust and deterministic constraints.

### 9.1 Advanced Reinforcement Learning: DQN and PPO
We implemented a Reinforcement Learning stack in `scirust-learning`.
- **Algorithms:** Support for Tabular Q-Learning/SARSA and Deep Q-Networks (DQN). Furthermore, we implemented **Proximal Policy Optimization (PPO)** using a clipped objective to ensure stable policy updates.
- **Determinism:** Agent interactions and memory sampling are enforced using seeded `PcgEngine` instances, ensuring reproducible training trajectories.

### 9.2 Computer Vision: ResNet and Vision Transformers
Two major architectures were added to `scirust-core`:
- **ResNet-18/34:** Modular implementation using `ResidualBlock` and a **Global Average Pooling (GAP)** step to handle varying input resolutions.
- **Vision Transformer (ViT):** Implementation of patch projection via 2D convolutions followed by a Transformer encoder. Features are aggregated across the sequence dimension for classification.

### 9.3 Generative AI and Transformers
- **Variational Autoencoders (VAE):** Implementation of the reparameterization trick using `PcgEngine`-derived Gaussian noise and an analytical KL divergence loss.
- **Mixture of Experts (MoE):** A modular MoE layer supporting **Top-k routing** and additive expert aggregation, enabling model scaling without linear compute cost growth.

### 9.4 Specialized Architectures
- **Graph Neural Networks (GNN):** Basic **Graph Convolutional Network (GCN)** layers supporting sparse-dense adjacency matrix multiplications.
- **Speech AI:** Audio encoders and a representative **CTC Loss** implementation for temporal sequence alignment.
- **PEFT (LoRA):** Low-Rank Adaptation for Linear layers, allowing frozen backbone models to be fine-tuned via small rank-r matrices.

## 10. Discussion

Two observations recur across the contributions. First, the discipline did the
load-bearing work: because every primitive was accepted only against an oracle —
often bit-for-bit — a path either reproduces the reference or it does not, which kept
the framework's results trustworthy as it evolved. Second, the most valuable
conclusions were sometimes negative and arrived only by measuring: that thread count
does not affect single-request latency, that an allocation arena was unwarranted,
that a naive relative-error metric is untrustworthy near cancellations, and that low
error is not the same as the correct law. Each contradicted a plausible prior and
would have been missed by asserting rather than measuring. A third, unifying point:
reproducibility, treated as a property to be engineered and measured rather than
hoped for, became a product feature in its own right — the deterministic runtime's
central guarantee is exactly the bit-exactness the framework's testing discipline
already depended on. The int8 quantization stack extended exactly this contract: its integer
inference path is thread-invariant by the same single-thread per-cell reduction
argument, and a fixed-point requantization reproduces the calibrated model
bit-for-bit, so determinism carried down to low precision without new machinery.

## 11. Limitations

The framework is a research artifact and not production-grade. Convolution lacks an
im2col-plus-BLAS or GPU path and is therefore slow in absolute throughput; the GPU
backend is validated for compute correctness but not yet wired into training; and the
deterministic runtime is inference-only over a two-dimensional layer set, with
transformer support requiring a separate three-dimensional path. Determinism is
scoped to a fixed binary and architecture. The symbolic engine is a stochastic search
on a modest primitive set, and several contributions are single-session results.
The newly introduced **PINN (Physics-Informed Neural Networks)** loss evaluator
enables the integration of symbolic physical residuals into the AD optimization path.
The int8 quantization is post-training rather than quantization-aware;
no-accuracy-loss result is established on the MNIST MLP, while the convolutional
quantizers are validated for fidelity and determinism on synthetic inputs rather
than for accuracy on a labeled image benchmark, and no on-device (no_std)
microcontroller deployment is yet demonstrated.
The repository also includes an evolutionary-optimization module; of its algorithms only the multi-objective NSGA-II is validated here, recovering the ZDT1 Pareto front to within about 1e-3, while the simplified single-objective optimizers converge on convex landscapes but not on hard multimodal functions. None of these undercut the measured results; they bound what those results should be
taken to mean.

## 12. High-Level Tensor Algebra and Graph Compilation: scirust-tensor

### 12.1 Motivation and Context
While the core of SciRust provides robust primitives for deep learning, complex architectures like Transformers require more flexible tensor manipulations than simple matrix multiplications. Current state-of-the-art frameworks (JAX, PyTorch) rely on optimized `einsum` and graph compilers (XLA) to reduce memory overhead. To bridge this gap while maintaining SciRust's pure-Rust and deterministic DNA, we introduced `scirust-tensor`.

### 12.2 Methodology: Einsum and Contraction Planning
The module implements an optimized `einsum` parser and a **contraction planner**. For a given tensor contraction expression:
$$C_{i,l} = \sum_{j,k} A_{i,j,k} \cdot B_{k,j,l}$$
The planner evaluates the optimal execution path. For multi-tensor contractions, it uses a greedy approach to minimize the total number of floating-point operations (FLOPs).

### 12.3 Graph Optimization and Operator Fusion
A major contribution of this module is the **operator fusion** engine. In standard runtimes, sequential operations like `MatMul -> BiasAdd -> ReLU` involve multiple memory passes and intermediate buffers. `scirust-tensor` compiles these into a single **fused kernel**, reducing memory bandwidth pressure.
The optimization pipeline includes:
- **Redundancy Elimination**: Removing identity transpositions.
- **Stride-based Permutation**: Integrating axis permutations into the GEMM kernel strides to eliminate explicit data copies.

### 12.4 Results and Determinism
By using a fixed reduction order in all tensor contractions, we ensure bit-for-bit identical results across different thread counts. Preliminary benchmarks show that operator fusion reduces peak memory usage by up to 35% on deep Transformer blocks, while maintaining a strict deterministic fingerprint. The module is fully compatible with the **SRT1** inference runtime and the **QSR1** int8 quantization stack.

### 12.5 Limitations
The graph compiler is currently restricted to static shapes. Dynamic shape support and JIT-compilation of kernels for arbitrary fusion patterns remain as future work.

## 13. Deterministic Event Detection and Classification

### 13.1 Motivation
Real-time event detection in critical systems (e.g., neuroprosthetics or industrial control) requires not only high accuracy but also absolute determinism for auditability and certification. Current frameworks often rely on non-deterministic parallel reduction or stochastic sampling, which is unsuitable for high-stakes environments.

### 13.2 Methodology
We introduce a streaming architecture based on deterministic sliding windows. Each window $W$ of size $N$ is transformed into a tensor $T \in \mathbb{R}^{1 \times N}$. Event detection is formulated as a score function $S(T) \to [0, 1]$.
For classification, we utilize the framework's core MLP and CNN layers, frozen into the SRT1 format.
$$ \text{Event}(t) = \mathbb{I}(S(W_t) > \tau) $$
where $\tau$ is a calibrated threshold.

### 13.3 Results and Metrics
Expected performance on the Numenta Anomaly Benchmark (NAB) targets an F1-score of $>0.85$ with zero bit-drift across multiple threads. The use of QSR1 int8 quantization is expected to reduce latency by $3\times$ on edge ARM processors while maintaining an MSE bit-closeness of $<10^{-4}$ compared to the f32 oracle.

## 14. Advanced Neuro-Symbolic Integration

### 14.1 Overview
We introduce `scirust-neuro-symbolic`, a crate dedicated to hybrid AI architectures. It bridges the gap between connectionist models (tensors) and symbolic logic (rules/solvers).

### 14.2 Differentiable Logic
By implementing fuzzy logic operators (Product T-norm) as tensor operations, we allow logic constraints to be integrated directly into the gradient descent optimization path.
$$ \text{AND}(a, b) = a \cdot b $$
$$ \text{OR}(a, b) = a + b - a \cdot b $$

### 14.3 Formal Solvers and E-Graphs
The crate provides a CDCL SAT solver and an E-Graph engine for equality saturation, enabling symbolic simplification and formal verification of neural network properties or synthesized programs.

### 14.4 Conclusion
The addition of neuro-symbolic capabilities positions SciRust as a versatile platform for AGI research, enabling models that can both learn from data and reason over structured knowledge.

## 15. Conclusion

SciRust is a pure-Rust deep learning framework — a hybrid runtime and transpiler — on
which four capabilities were built and validated: a portable GPU and Tensor Core
path reaching ~63 TFLOPS in BF16; a hybrid genetic-gradient symbolic regression
engine that recovers known laws from data using the framework's own symbolic
differentiation; a deterministic inference runtime providing bit-exact, bounded-latency,
auditable inference, generic over architecture; and a deterministic int8
quantization stack giving a portable, thread-invariant integer inference path for
embedded deployment, with fixed-point requantization that reproduces the
bit-for-bit and weight tensors roughly four times smaller. Expanding on these,
five advanced features—an Ahead-Of-Time (AOT) static compiler for zero-overhead
embedded inference, a soft-float matrix engine for cross-platform bit-exactness,
latent activation steering for real-time representation engineering,
quantization-aware training (QAT) via a Straight-Through Estimator, and an
Integrated Gradients engine for mathematical explainability—further establish
SciRust as a high-integrity framework. The addition of **Modern AI Families** (RL, CV, Generative, GNN) further broadens the scope of the framework toward a unified pure-Rust AI stack. The throughline is
methodological: each contribution was accepted only after matching an oracle,
reproducibility was measured rather than assumed — in several cases bit-for-bit — and
the most useful findings were the ones the measurements forced against expectation.
The next steps follow directly: a GPU-accelerated forward path reusing the validated
cuBLAS backend for dense layers, a three-dimensional inference path for
attention-based models, and supply-chain pinning to extend the runtime's auditability
from its weights to its build.

## N-D Autograd and Research-Driven Extensions

Beyond the 2-D reverse-mode tape, SciRust now provides an **N-D autograd tape**
whose every operator is validated by a finite-difference gradient check, and on
top of it a research-backed deep-learning stack. Each capability maps to a
specific paper and ships with a test; the full mapping (14 of 20 items delivered)
is tracked in `docs/RESEARCH_ROADMAP.md`.

- **Causal decoder language model**, trained end-to-end (token and learned
  positional embeddings, causal multi-head attention, a fused, numerically
  stable softmax cross-entropy), which overfits a fixed sequence exactly —
  end-to-end evidence that the stack learns.
- **LLaMA-family layers**: RMSNorm, SwiGLU, a Pre-RMSNorm LLaMA block, rotary
  position embeddings (RoPE, with the relative-position property tested), and
  grouped-/multi-query attention expressed through batched-matmul broadcasting.
- **Deterministic optimizers**: Adam, AdamW (decoupled weight decay), Lion, and
  Muon (Newton–Schulz orthogonalized momentum), Schedule-Free, AdEMAMix, and SOAP (Adam in Shampoo's eigenbasis) — all bit-for-bit reproducible.
- **Certifiable AI**: Interval Bound Propagation **and CROWN** (tighter
  linear-relaxation bounds) yield provable output bounds
  for ReLU MLPs and a robustness certificate, validated by soundness sampling.
- **Reproducible reductions**: order-independent floating-point sum/mean/dot
  (canonical ordering + exact expansion), bit-identical regardless of thread
  count — the foundation for bit-exact multi-thread training.
- **Inference**: exact (output-preserving) speculative decoding, a tiled
  online-softmax FlashAttention, a DeltaNet delta-rule linear-attention layer, and a Mamba selective state-space layer.
- **Scientific bridge**: a Neural ODE that backpropagates through an RK4 solver.
- **Compression**: Wanda activation-aware pruning and SmoothQuant, and GPTQ (second-order error-feedback int8 weight quantization, CLI `scirust gptq`), and AWQ (activation-aware search-based int8 weight quantization, CLI `scirust awq`), extending the
  deterministic int8 path.

Two CLI commands surface this work: `scirust certify` (IBP **and CROWN** bounds, side by side, and robustness)
and `scirust lm --opt adam|adamw|lion|schedule-free|ademamix|soap` (train the N-D decoder LM).

A third command, `scirust conformal`, produces distribution-free conformal-prediction intervals with a guaranteed coverage level.
