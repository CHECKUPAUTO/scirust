# SciRust — "Industrial Adoption" Roadmap

Implementation proposals ranked by value for an industrial adopter,
based on what SciRust already has that PyTorch/Burn/Candle do not:
**measured bit-exact determinism, total auditability (pure Rust,
zero FFI), embedded bit-exact int8 quantization, and an ownership
analyzer (SOM)**. The strategy is not to compete on TFLOPS:
it is to own the "certifiable and reproducible AI" niche.

Each proposal specifies: the target customer, the deliverable, and the
definition of done (always: green gates + oracle + docs).

---

## P0 — The trust foundation (unlocks everything else)

### P0.1 Proof-Carrying Inference
- **Customer**: finance (model auditing), medical/aerospace (traceability),
  insurance, AI compliance (EU AI Act art. 12 — logging).
- **What**: extend `scirust-runtime` so that every inference emits a
  **certificate**: hash of the architecture manifest, SRT1 hash of the
  weights, hash of the inputs, 64-bit fingerprint of the outputs (already
  existing), seed, versions. Independent verifier
  `scirust-verify <certificate> <artifacts>` that replays and compares
  bit for bit. The building blocks exist (`proof_bundle`, fingerprint) —
  what is missing is the stable format + the verifier + the contractual
  documentation.
- **Done when**: a third party reproduces an inference on another
  x86/aarch64 machine and the verifier says MATCH, in CI.

### P0.2 Release engineering: tooled v1.0
- **Customer**: every buyer — nobody embeds a repository without
  versions.
- **What**: semver tags + `cargo-dist` or GitHub releases with binaries
  (`som-analyze`, verifier), maintained CHANGELOG (created), stated MSRV
  policy, **SBOM** (CycloneDX via `cargo-sbom`/`cargo auditable`)
  attached to each release — the "100 % Rust supply chain,
  committed Cargo.lock, cargo-deny in CI" argument becomes verifiable at
  one click.
- **Done when**: `v0.14.0` tagged with binaries + SBOM + release notes.

### P0.3 "Stable" story: getting off nightly for consumers
- **Customer**: industrialization teams (internal policies often
  forbid nightly).
- **What**: `portable-simd` and the architecture extensions of
  `scirust-simd` (`nightly-simd`) are the only nightly dependencies of the
  core. Make them truly optional end to end and prove in
  CI a **stable** build of `scirust-core` (+ SOM, already stable-compatible
  via `syn`) with the existing runtime AVX2/NEON dispatch
  (stable intrinsics). Dedicated `stable-build` CI job.
- **Done when**: `cargo +stable test -p scirust-core -p scirust-som-*`
  is green in CI.

## P1 — The products that get signatures

### P1.1 "SciRust Edge Pack": industrialized deterministic int8
- **Customer**: embedded/IoT/automotive — this is the most
  differentiating capability already **validated** (bit-exact int8, NEON ×10, QSR1).
- **What**: turn the 19 audit binaries into a product: CLI
  `scirust-quantize <model.srt1>` → QSR1 artifact + deviation report
  (bit-exact or bounds), cross-compiled aarch64 example + measured binary
  size (`no_std`-friendly for `scirust-embedded` in the future), guide
  "from float to certified int8 in 30 minutes".
- **Done when**: a 1-page README reproduces the full chain on the
  repository's MNIST, with the expected hashes published.

### P1.2 SOM as a CI linter: `cargo som`
- **Customer**: any Rust team (beyond ML!) — the broadest commercial
  entry point of the repository.
- **What**: package `som-analyze` as a cargo subcommand
  (`cargo-som`) + GitHub Action (`som-action`): SARIF output for
  GitHub's Security tab, fault budget per PR, and the pedagogical
  per-token report (already done) as an artifact. Extend the frontend
  to `if/else` branches (conservative join: state = worst of the two
  branches) — that is the most visible limitation today.
- **Done when**: the action runs on this very repository and comments on a PR.

### P1.3 Public benchmarks against Burn/Candle/tch
- **Customer**: technical decision-makers in the evaluation phase.
- **What**: `examples/benchmarks` reintegrated into the workspace as an
  informative nightly CI job; matrix (matmul, conv, MNIST epoch, int8
  inference) × (SciRust, Burn, Candle) × (x86 AVX2, aarch64 NEON), published
  in `docs/BENCHMARKS.md` with the methodology. Own the defeats
  in raw speed; showcase the wins (determinism, zero variance,
  fingerprint, 100 % Rust build).
- **Done when**: figures reproducible via documented `cargo bench`.

## P2 — Technical depth (durable differentiation)

### P2.1 "Certified determinism" mode for training
- **DONE**: `DataParallelTrainer::train_batch_threaded(n_threads, ..)`
  runs the workers on N OS threads (work stealing via atomic
  counter) but writes each result into its own worker-indexed slot and
  always reduces in worker order `0,1,…,n-1`. Since floating-point
  addition is not associative, a "as-finished" reduction would depend on
  the scheduler; this one does not → result **bit-identical
  for 1/2/4/8 threads** and identical to the sequential one. CI tests:
  `train_batch_threaded_is_thread_count_invariant` (deliberately
  order-sensitive contributions, ±1e16) +
  `parallel_tape_training_is_deterministic_across_threads` (real backward
  autograd). To our knowledge, SciRust is the only **self-contained**
  DL framework (100 % auditable Rust stack, zero FFI in the compute
  path) delivering this CI-tested guarantee, with, in addition, embedded
  deterministic int8 and the audit pieces. Related work: RepDL
  (Microsoft, 2025, arXiv:2510.09180) provides bit-for-bit **cross-platform**
  reproducibility of a float32 subset of PyTorch via correct
  rounding — a stronger guarantee on that axis for f32, but as a layer on top of a
  C++/Python TCB, without low precision nor audit pieces.
- **DONE (full loop)**: `multi_step_training_is_thread_count_invariant`
  — a real multi-step SGD loop (shared linear model, shards per
  worker, MSE loss, real autograd) produces a weight trajectory
  **bit-identical for 1/2/4 threads**. Batch invariance therefore composes
  over the whole training run (the guarantee does not depend on the number of
  layers). The "scaling benchmark" is deliberately omitted (wall-clock
  time is not deterministic — not testable in CI).

### P2.2 GPU: cut cleanly and rewire properly
- **DONE (step 1 — cutting)**: removal of the dishonest GPU stubs
  (`gemm_f32` returned zeros); `scirust-gpu` now exposes a tested reference
  CPU backend + honest device paths (`Unavailable`).
- **DONE (step 2 — rewiring wgpu)**: real WGSL GEMM behind the `wgpu`
  feature, executed on a Vulkan adapter, **validated against the CPU oracle**
  (documented floating-point tolerance) and **tested in CI** on software
  Vulkan (Mesa lavapipe) — "no claim without test" respected. `cargo deny`
  passes on the wgpu dependency tree. Optional dependency (the 8 default
  gates do not compile it).
- **DONE (step 3 — autograd tape)**: `WgpuEngine` implements the `GpuEngine`
  hook of the `Tape`; `Var::matmul_gpu` executes **both forward AND backward**
  (`dA = g·Bᵀ`, `dB = Aᵀ·g`) on the GPU, device/pipeline cached.
  Validated end to end against the CPU tape (forward + 2 gradients) on
  lavapipe. Opt-in path (feature + `matmul_gpu`) → the bit-exact guarantee
  by default stays intact.
- **DONE (step 4 — Conv2d)**: the im2col GEMMs of Conv2d (forward `W·col`,
  backward `dW = dout·colᵀ`, `dInput = Wᵀ·dout`) go through the engine via
  `Tape::gemm_ab` (native transpose path), validated end to end against
  CPU Conv2d on lavapipe. im2col/col2im remain CPU for now.
- **Remaining**: keep activations in VRAM between layers (avoid the
  CPU round-trip per op) + im2col/col2im on GPU (reference-archived
  pipelines); more ops (elementwise, reductions). The bf16/cuBLASLt backend
  is now available behind the `cuda` feature with dynamic loading
  and `Unavailable` fallback without a runtime; what remains is to add a hardware CUDA runner
  for device parity and performance regressions.

### P2.3 SOM at rustc precision (HIR/MIR)
- The current `syn` oracle remains the conservative mode actually shipped.
- **Remaining**: design an ownership/borrow extraction MIR pass (NLL,
  resolved types) producing the **SOM report format**, with verified
  transformations and a blocking CI gate. The old analysis driver, which
  announced transformations without modifying the MIR, has been removed.

### P2.4 Unified N-D tensor
- Merge `tensor::TensorND` (already in core) with the 2D tape:
  prerequisite for the compiler ambitions (shape inference beyond
  rows/cols), to be done **before** any training IR.
- **DONE (foundation — shape inference primitives)**: `TensorND`
  exposes `broadcast_shape`, `matmul_shape`, `broadcast_to` (+ bridge
  `from/to_tensor_2d`). 12 tests.
- **DONE (N-D-capable autograd)**: `autodiff::nd` now expresses
  **full multi-head attention** `softmax(Q·Kᵀ/√d)·V` on `(heads, seq, d)`
  — via `bmm` (batched broadcast matmul), `transpose_last2`, `softmax`
  (last axis, Jacobian backward), `mul`/`add`/`sub`/`relu`/`sum` — all of it
  **gradient-checked** (finite differences). This is precisely what the 2D
  tape cannot do ⇒ N-D is the **capable superset**. 2D remains
  the production default **by architectural choice** (coexistence, cf.
  `GROWTH_PLAN.md`), not a TODO; rewriting `reverse.rs` wholesale is
  not desirable — we migrate in tested increments.

## What we do NOT propose (anti-goals)
- Chasing PyTorch/TensorRT TFLOPS: a lost niche from the start
  and against the philosophy.
- Multiplying crates: value comes from the depth of
  the guarantees, not the surface area. (`events-*`, `edge`, `bridge`
  remain frozen until a customer pulls them.)

## Recommended execution order
P0.2 (1 d) → P0.3 (2-3 d) → P0.1 (1 wk) → P1.2 (1 wk) →
P1.1 (1-2 wks) → P1.3 (ongoing) → P2.x depending on traction.
