# SciRust — Ambitious Growth Plan

> Strategic document. The detailed operational roadmap lives in
> [`INDUSTRIAL_ROADMAP.md`](INDUSTRIAL_ROADMAP.md); this document gives the
> **vision**, the **non-negotiable fundamentals**, and an **ambitious phasing**.
> The **"research → functions"** roadmap (real papers translated into
> concrete functions, with status) lives in
> [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md).

## 1. Vision: certifiable AI

SciRust is **not** trying to compete with PyTorch on TFLOPS. Its value
proposition, defensible and unique, is to **own the niche of certifiable,
reproducible and auditable AI**:

- **measured bit-exact determinism** (inference *and* multi-threaded training),
- **total auditability**: pure Rust, zero FFI, `Cargo.lock` + CycloneDX SBOM +
  `cargo deny`,
- **embedded / edge**: bit-exact int8 quantization, partial `no_std`,
- **forensic traceability**: inference proof certificates (SRT1).

Target customers: regulated sectors (finance, healthcare, safety, defense), edge/embedded,
reproducible research, AI audit/compliance.

## 2. Non-negotiable fundamentals

Any growth **must** respect these invariants (they are what make the
project's value):

1. **100 % of the code under `*/src/` is wired and tested.** Otherwise → `archive/`.
2. **Determinism.** Any randomness via a seeded `PcgEngine`; same seed ⇒
   bit-identical output. Parallel reductions use a fixed order.
3. **Pure Rust, zero FFI.** Auditable end to end.
4. **No overpromising.** Every claimed capability is backed by a test.
   *No claim without CI* (e.g.: no GPU claim without a GPU runner).
5. **8 green gates**: `fmt`, `clippy -D warnings`, build/test (nightly **and**
   stable), aarch64 cross-check, `cargo deny`, `rustdoc` 0 warnings.
6. **Documentary honesty.** The README/CHANGELOG reflect the measured state,
   never an aspiration.

## 3. Workstreams (goal · achieved · milestones)

### A. Unified N-D tensor core — *the architectural lock*
- **Goal**: an N-D autograd tape of which 2D becomes a special case
  (shape inference beyond `rows/cols`).
- **Achieved**: shape primitives (`broadcast_shape`/`matmul_shape`/
  `broadcast_to`); N-D autograd MVP (`autodiff::nd`: broadcasted add/sub/mul,
  2D matmul, **batched bmm**, relu, sum) — **gradient-checked**.
- **Milestones**: N-D softmax/layernorm/transpose-axes/reductions · `nd::Linear`
  then `nd::Attention` · progressive layer migration · eventually 2D = an
  alias.
- **Fundamental**: every op **gradient-checked**; determinism preserved.

### B. LLM stack — inference & serving
- **Goal**: a credible, deterministic, embeddable LLM path.
- **Achieved**: attention/flash/MoE/RoPE · **KV-cache O(n) proven equivalent** to
  full recomputation · **deterministic BPE** · `generate_ids` decoupled from the tokenizer.
- **Milestones**: **seeded sampling** (temperature/top-k/top-p → deterministic) ·
  production BPE-bytes tokenizer · KV-cache wired into a public `generate` ·
  batching · **int8 for LLM inference** · a small **trained** model (end-to-end
  proof).
- **Fundamental**: **seeded** sampling (reproducible); bit-exact int8.

### C. Mature GPU — wgpu, opt-in
- **Goal**: portable acceleration, without betraying determinism by default.
- **Achieved**: wgpu GEMM/Conv2d/elementwise · **VRAM residency** (whole layer)
  · CPU oracle · tested on software Vulkan (lavapipe).
- **Milestones**: complete GPU op set (softmax, layernorm, reductions) ·
  **transparent** residency in the tape (`DeviceTensor` materialized lazily) ·
  **hardware GPU runner in CI** (perf claim **only** then) · kernel
  fusion.
- **Fundamental**: bit-tolerant CPU oracle; **no perf claim without a runner**.

### D. Interoperability & ecosystem
- **Goal**: load/export models; a reproducible "model zoo".
- **Achieved**: ONNX export (template) + **weight import** (bit-exact round-trip).
- **Milestones**: **real ONNX protobuf** (external models) · faithful per-layer
  graph export · `safetensors` · model zoo (weights + manifest + **certificate**).
- **Fundamental**: import validated by **round-trip**; provenance via SBOM.

### E. Certified & distributed training
- **Goal**: extend the bit-exact guarantee to the distributed case.
- **Achieved**: data-parallel **certified determinism** (1/2/4/8 threads
  bit-identical); invariant multi-step SGD loop.
- **Milestones**: **multi-node** with fixed-tree reduction (inter-machine
  determinism) · deterministic checkpointing · **"training proof"**
  (reproducible certificate of a run).
- **Fundamental**: **fixed-order** reductions; certificates.

### F. Code analysis (SOM) — rustc precision
- **Goal**: move from the conservative `syn` oracle (fast mode) to
  NLL/resolved-types precision (precise mode).
- **Achieved**: conservative `syn` oracle and SARIF linter.
- **Milestones**: design a real **MIR** ownership/NLL pass with transformation
  oracles and a blocking CI gate before distributing a driver.

### G. Tooling & trust
- **Achieved**: CLI (53 commands) · CycloneDX SBOM · release automation ·
  proof certificates · `cargo deny`.
- **Milestones**: **branch protection** · parser fuzzing · measured
  coverage · **reproducible** benchmarks · exhaustive docs.

## 4. Phasing

| Horizon | Target deliverables |
|---|---|
| **Short term** (weeks) | seeded sampling + KV-cache in a public `generate` · N-D softmax/layernorm · broader ONNX import · branch protection + v0.14 release |
| **Medium term** (months) | `nd::Linear`/`nd::Attention` + migration · lavapipe → GPU runner + transparent residency · deterministic multi-node training · SOM MIR pass |
| **Long term** | unified N-D tape (2D = special case) · certified model zoo · "certifiable AI" as a product (audit/compliance) |

**Status (wave 29)**: short-term *seeded sampling + KV-cache* and *N-D softmax/
layernorm* = **done**; medium-term `nd::Linear`/`nd::Attention` = **done**
and already surpassed — the N-D tape now carries a **full causal decoder LM**
(`nn::nd_decoder::NdDecoderLM`: tok/pos embeddings, causal blocks, lm head,
cross-entropy) **trainable end to end** by a **deterministic N-D Adam
optimizer** (`nn::nd_optim::NdAdam` + `parameters()` on all layers).
All N-D ops (including `gather`, `cross_entropy`, causal attention) are
gradient-checked; the LM overfits and re-predicts a sequence exactly.

## 5. Success metric

Not TFLOPS — but the **number of testable certifiable properties**:
determinism (inference + training), end-to-end reproducibility,
auditability (SBOM, zero FFI), int8 bit-exactness, certificates. A **guarantees
dashboard** rather than a throughput benchmark.

## 6. How to contribute without breaking the fundamentals

- A new autograd op? → mandatory **gradient check**.
- A new "device" capability (GPU/CUDA)? → return `Unavailable` as long
  as there is no runner; **never** a fabricated result.
- A parser / format? → **round-trip** test.
- Parallelism? → **fixed-order** reduction + thread-count invariance test.
- Always: 8 green gates, and the README tells the **measured truth**.
