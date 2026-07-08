# Training SCIAGENT on NVIDIA Jetson Thor — feasibility & roadmap

Goal: train the **350M** config (and, it turns out, plausibly the **7B**) of
SCIAGENT on a single Jetson Thor. This document is the honest engineering path:
what already works, the memory reality (with an executable planner), the
critical-path work, and a phased plan.

## TL;DR

- The Thor's **128 GB unified memory** is the enabler — it fits models that a
  24 GB discrete GPU never could.
- SCIAGENT's forward/backward is currently **CPU tape-autodiff** (`scirust-core`)
  and does **not** use the GPU. Training 350M therefore needs a **GPU backend**
  — this is the critical path, and it must be built and verified on GPU
  hardware (the Thor itself), not on the CPU-only CI box.
- Two mandatory software prerequisites regardless of backend: **flash-attention**
  (never materialize the `S×S` score matrix) and **activation checkpointing**
  (recompute, don't store, per-layer activations). Without them the run does
  not fit; with them, even 7B fits.
- **Status (Route A, done):** a fully-resident wgpu/Vulkan backend now trains the
  real 350M model end-to-end on the Thor, every op gradient-checked, and runs the
  full on-device loop **train → fine-tune (LoRA/DoRA) → merge → generate**
  (KV-cached, sampled, batched prefill). But its **fp32 throughput is measured at
  ~34 tok/s training / ~3.5 tok/s single-stream 350M decode (<5% of peak)** —
  correct, not fast. Prompt **prefill** is far better (batched `m=P`: ~149 tok/s
  ingest at 350M, saturating). From-scratch 350M pretraining — and fast
  decode — needs FP16/Tensor cores (**Route B**); Route A's near-term value is
  fine-tuning, inference, and small-model training. See *Measured on the Thor* below.

## The memory reality (run it yourself)

`sciagent-plan` computes the first-order training-memory budget for any config.
Exact asymptotics, approximate linear constants:

```bash
cargo run --release -p scirust-sciagent --bin sciagent-plan -- \
  --model 350m --seq-len 8192 --batch 1 --precision bf16 --ceiling-gb 128
```

Measured (batch 1, seq 8192, ceiling 128 GB):

| Config | Precision | naive total | flash | flash + ckpt |
|--------|-----------|------------:|------:|-------------:|
| 350M | fp32 | **211 GB — NO** | 20 GB | 6 GB |
| 350M | bf16 | 107 GB (batch-1 only) | 11 GB | **4 GB** |
| 7B | bf16 | **436 GB — NO** | 118 GB | **72 GB** |

Reading it:
- **Naive fp32 350M @ 8k does not fit.** The `2·B·H·S²·L` attention score
  matrix alone is ~200 GB.
- **Flash-attention** removes the quadratic term → 350M drops to ~20 GB.
- **+ Activation checkpointing** → ~4–6 GB of activations, leaving the 128 GB
  almost entirely for batch size and throughput. At bf16, 350M @ 4096 runs at
  **batch 4** in ~5 GB; even **7B @ 8192 fits in 72 GB**.

Conclusion: the Thor can train these models, but only once flash-attention and
checkpointing exist. Those two are worth building first because they gate
everything.

## Critical path: a GPU backend for SCIAGENT

The transformer ops (matmul, GQA attention, RMSNorm, SwiGLU) and the reverse-mode
autodiff must run on GPU tensors. Two routes:

### Route A — Vulkan via `scirust-gpu` (wgpu) — recommended first
- `scirust-gpu` already exists in this workspace (its CI runs on wgpu/lavapipe).
- **Runs on the Thor today via Vulkan — no CUDA toolchain needed.**
- Work: implement the transformer's forward/backward as `scirust-gpu` kernels
  and route `SciAgentModel` through them behind a `gpu` feature; wire flash-
  attention and checkpointing into the training loop.
- Pro: portable, buildable incrementally, testable against the CPU path for
  numerical parity. Con: wgpu does not expose Blackwell Tensor cores, so it
  leaves throughput on the table.
- **✅ Wired (v1 — GEMM routing).** `scirust-sciagent`'s `gpu` feature attaches
  `scirust-gpu`'s validated `WgpuEngine` (the tape's GEMM hook) and flips the
  tape into GPU-matmul mode (`Tape::set_prefer_gpu_matmul`), so every projection,
  RoPE rotation, per-head `Q·Kᵀ`/`·V`, SwiGLU and the tied LM head run their
  **forward and backward** GEMMs on the device — no per-call-site changes, the
  autodiff graph unchanged, non-GEMM ops (softmax/RMSNorm/mask) on CPU. Parity
  vs the CPU reference (logits + every parameter gradient) is checked in
  `tests/gpu_parity.rs` and on-device by `examples/gpu_forward_parity.rs`. GEMMs
  are the dominant transformer FLOPs.
- **✅ Done (v2 — fully-resident all-op path).** `ResidentModel`
  (`scirust-sciagent/src/gpu.rs`) mirrors every `SciAgentModel` weight into VRAM
  and runs the whole decoder on `scirust-gpu`'s `GpuChain` — embed → N×GQA blocks
  → final RMSNorm → tied LM head → cross-entropy → **full backward → AdamW**,
  nothing leaving VRAM between ops (the path that beats the per-op tape ~4.15× on
  the Thor). **Every trainable weight** updates on-device: the tied embedding, all
  seven per-block projections, **and** all RMSNorm gains (`rms_norm_gain_backward`
  wired through the block and model backward). Each op is gradient-checked against
  a CPU oracle to f32 tolerance. A production run harness ships with it
  (`examples/resident_pretrain.rs`): real shard streaming, byte-level ingestion
  (no tokenizer needed), warmup+cosine LR, resumable safetensors checkpointing,
  throughput logging. Two hardware-only bugs the Thor caught and we fixed along
  the way — the swiglu 5-storage-buffer limit and the 65535-workgroup dispatch
  limit at 350M (grid-stride) — both invisible on lavapipe.

### Route B — native CUDA for Blackwell (sm_110) — later, for throughput
- CUDA 13, `cudarc` 0.19+, compute capability sm_110; Tensor cores with
  FP8/FP4 — the Thor's real horsepower.
- Work: a CUDA backend in `scirust-gpu` (or a sibling crate) with cuBLAS/
  cuDNN or hand-written WMMA kernels; a flash-attention kernel.
- Pro: maximum tokens/sec. Con: a substantial new backend; ties the build to
  the CUDA toolchain.
- **Design & feasibility: see [`ROUTE_B.md`](ROUTE_B.md)** — mixed-precision
  (bf16 compute + fp32 master) cuBLASLt approach, a `CudaChain` mirroring
  `GpuChain` so `ResidentModel` is a backend swap, the phased B0–B5 plan, and the
  **go/no-go probe (B0)**: a cuBLASLt bf16-vs-fp32 GEMM microbench on the Thor whose
  single number decides whether Route B is worth starting.

Recommended sequencing: **A to get correctness and a working run on the Thor,
then B to make it fast.**

## Measured on the Thor (resident path, Route A)

Route A is complete and validated on the Thor's Blackwell: the full **350M** step
(304.1M params) runs end-to-end, bit-tolerant to the CPU reference. Memory sat at
the planner's estimate (~1.2 GB weights + ~2.4 GB AdamW state in fp32, activations
extra) — comfortably within 128 GB. The open question was throughput, now measured
(`examples/resident_pretrain`, 350M config, single sequence, fp32):

| seq_len | tok/s |
|--------:|------:|
| 128 | ~40 (naive GEMM) |
| 128 | 25 (tiled GEMM) |
| 256 | 30 |
| 512 | 34 |
| 1024 | 33 |

Reading it:
- Throughput **rises modestly with `seq_len` then saturates at ~34 tok/s by
  seq 512** — beyond that the step is compute-bound and per-op submit overhead is
  already amortized. Larger `m` (seq/batch) fills the GPU only up to this ceiling.
- At seq 512 a 350M fwd+bwd over 512 tokens is ~0.93 TFLOP in ~7.7 s ≈
  **120 GFLOP/s — under 5% of the Thor's fp32 peak.** The wall is **kernel
  efficiency**, not memory or `m`.
- A **shared-memory tiled GEMM was tried and is a net regression here**
  (25 vs 40 tok/s at seq 128): Blackwell's L2 already absorbs the reuse tiling
  saves, so the `workgroupBarrier`/occupancy cost is pure overhead for these
  short-and-fat (`m=128`) matmuls. Kept the naive kernel. (PR closed with the
  measurement; output was bit-identical, so it was purely a perf call.)

**Conclusion.** Pure-fp32 WGSL compute tops out well below what a from-scratch
pretrain needs: at ~34 tok/s, a Chinchilla-optimal ~7B-token run for 350M would
take **6+ years** on one Thor. The only lever with an order-of-magnitude
multiplier left is **FP16/BF16 + Tensor cores (Route B)** — 10–30× on Blackwell —
which wgpu cannot reach; everything else (bigger tiles, kernel fusion, fewer
submits) is a fraction of that and does not change the order of magnitude.

Where Route A earns its keep, then, is **correctness (a gradient-checked on-device
reference), fine-tuning, inference, and small-model training** — not from-scratch
350M pretraining. That gates Route B as the real prerequisite for large-scale
training throughput.

### Inference throughput (measured)

The resident inference path (KV-cached generation with batched prefill) is now
measured on the Thor too (`examples/resident_infer_bench`, fp32, single sequence,
random weights — throughput is weight-independent). Two regimes, both timed through
the public `generate_cached` API:

| config | prefill ingest @ P=512 | decode (m=1) | ratio |
|--------|-----------------------:|-------------:|------:|
| d512 · 8L · ff1408   | **1078 tok/s** | 14.6 tok/s | ~74× |
| d1024 · 24L · ff4096 (~350M) | **149 tok/s** | 3.5 tok/s | ~43× |

Prefill ingestion vs prompt length (350M-class, `m = P` one forward):

| P | tok/s |
|--:|------:|
| 16 | 45 |
| 64 | 92 |
| 128 | 127 |
| 256 | 149 |
| 512 | 149 |

Reading it:
- **Prefill ingestion saturates** (256→512 is flat, 149→149) — the batched
  `m = P` forward becomes fully compute-bound past P≈256, so adding rows scales
  linearly (constant tok/s). At small P the per-forward fixed cost (dispatch,
  embedding gather, LM head) dominates, so tok/s is lower. This is the
  **batched-prefill win (infer-4) quantified**: ingesting a prompt at 149 tok/s
  vs decoding at 3.5 tok/s is the same GPU doing `m = P` vs `m = 1` work.
- **Decode is `m = 1`** — one row per forward, so it barely occupies the GPU:
  3.5 tok/s at 350M, ~43× below the saturated prefill rate. This — not memory —
  is the autoregressive-decode wall on the fp32 path, and it is the *same*
  kernel-efficiency ceiling the training numbers hit, seen from the worst-case
  `m = 1` end. It is exactly where **Route B (FP16/Tensor cores)** and/or
  batched-sequence or speculative decode (DeepSpec, below) would pay off.

**Conclusion.** On-device inference is *usable now* for short, latency-tolerant
generations and for prompt-heavy workloads (fast batched prefill), but single-stream
decode at a few tok/s for 350M confirms the same fp32 kernel-efficiency ceiling —
the order-of-magnitude lever remains Route B.

### Speculative decoding (measured cost model)

Speculative decoding attacks the `m = 1` decode wall **algorithmically**: a cheap
draft proposes `k` tokens, the target verifies all `k` in one wide `decode_batch`
(`m = k`), and accepts the longest prefix matching its own greedy pick plus one
correction. It is **exact** — the output equals plain greedy for *any* draft
(hardware-confirmed: `resident_speculative_matches_greedy` PASSes, and the bench's
`exact` column reads `yes` at every `k` even with a useless random draft). The
*speedup* is entirely a function of the draft's acceptance rate.

Measured on the Thor (`examples/resident_spec_bench`, fp32; target d512·8L, draft
d512·2L):

- greedy target **14.4 tok/s** (69.5 ms/tok); the 2-layer draft is **3.5× cheaper**
  (19.9 ms/tok).
- Cost model `speedup(a) = (a+1)·t_g / (k·t_d + 2·t_g)` (a = tokens accepted per
  round of `k`):

  | k | accept 50% | accept 75% | accept 100% | break-even accept |
  |--:|-----------:|-----------:|------------:|------------------:|
  | 2 | 0.78× | 0.97× | 1.17× | ~1.6 / 2 |
  | 4 | 0.95× | 1.27× | 1.59× | ~2.1 / 4 |
  | 8 | 1.17× | 1.63× | 2.10× | ~3.3 / 8 |

Reading it: on **random** weights the draft agrees ~by chance (≈0.4 %), so measured
speedup is `<1×` — the draft/verify overhead dominates. The value is the cost model:
with a 3.5×-cheaper draft, `k = 8` **breaks even at ~41 % acceptance** and a draft
**trained to track the target** (acceptance ~0.6–0.8, typical in practice) would
deliver **~1.6–2.1×**. So speculative decoding is a real, order-of-~2 lever for
single-stream decode *given a trained draft* — complementary to Route B (which lifts
the per-forward ceiling itself), not a substitute.

## What already works on the Thor today (verified)

- The whole workspace **cross-checks for `aarch64-unknown-linux-gnu`** in CI —
  the Thor's CPU architecture. The `sciagent*` binaries were confirmed to
  `cargo check --target aarch64-unknown-linux-gnu` cleanly.
- The BPE tokenizer is **embedded in the binary** (`include_bytes!`), and the
  `small` checkpoint ships in-repo, so `sciagent ask ...` runs on the Thor's
  CPU with no extra files and no GPU. Good for on-device **inference** now and
  as the numerical reference oracle for validating the GPU backend.

## Phased plan

1. **Foundations (backend-agnostic, CPU-testable — start here).**
   - `sciagent-plan` memory planner. ✅
   - Flash-attention **reference + numerical oracle** (`flash_attention`):
     block-streaming online-softmax, proven numerically identical to the dense
     path *and* to the model's own tape ops (`flash == dense == model`). ✅
     This is the correctness contract a GPU kernel must satisfy.
   - Gradient/activation **checkpointing technique** (`checkpointing`):
     segment recompute with an upstream-gradient surrogate, proven to yield
     gradients identical to a full end-to-end tape on `scirust-core`. ✅
     Remaining: wire it into the training loop's multi-segment param mapping.
2. **Route A GPU backend (on the Thor).**
   - `gpu` feature routing `SciAgentModel` ops through `scirust-gpu`. ✅ (GEMM
     forward+backward via the tape engine; see Route A above.)
   - Parity tests: GPU logits + all parameter grads vs CPU within tolerance. ✅
     (`tests/gpu_parity.rs`, `examples/gpu_forward_parity.rs`.)
   - Fully-resident all-op path (`GpuChain` / `ResidentModel`), every weight
     trainable, gradient-checked; production run harness. ✅ (see *Measured on the
     Thor* above). **Throughput characterized: ~34 tok/s, fp32 kernel-bound.**
   - Remaining (gated by Route B): mixed precision (bf16/fp16 compute, fp32
     optimizer master) — the order-of-magnitude throughput lever.
3. **Scale-up run — blocked on Route B for throughput, not correctness.**
   - The step *runs* at 350M today; a real from-scratch pretrain is not
     throughput-feasible in fp32 (see *Measured*). Use the resident path now for
     **fine-tuning / small-model training / inference**; defer from-scratch 350M
     pretraining until FP16/Tensor-core compute lands.
   - Checkpoint to the 128 GB host memory / NVMe; evaluate with `sciagent-eval`.
4. **Route B CUDA / FP16 Tensor cores — now the critical path for training
   throughput** (was "optional"). The measured fp32 ceiling makes this the only
   route to a practical large-scale run.

## Related: speculative decoding (DeepSpec)

[deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) is a
Python/PyTorch/CUDA framework (MIT) for training *draft models* for speculative
decoding (Eagle3/DFlash/DSpark). Its **code is not reusable** here — it would
violate SCIAGENT's pure-Rust, no-ML-runtime, deterministic invariants — but the
**pattern is a strong fit**: a small fast model (SCIAGENT `small`) drafts tokens
that a larger target (the `350m`, once trained) verifies, for exact
(verification-preserving) 2–4× inference speedups. Determinism is preserved
because verification makes the output identical to the target's. **Prerequisite
to flag now:** speculative decoding needs the draft and target to share a
tokenizer, but the configs currently mismatch (`small` vocab 8192 vs `350m`
vocab 32768). If drafting is a goal, align the `350m` tokenizer with `small`'s
(or plan a vocab bridge) before training it. Not a near-term task — it depends
on the 350M existing first.

## Risks / honesty

- A **correct** GPU backend was built and validated on the Thor (Route A). A
  **fast** one is a different, larger project: the measured fp32 ceiling
  (~34 tok/s, <5% of peak) confirms that practical training throughput needs
  FP16/Tensor cores (Route B), which wgpu cannot reach.
- The planner is first-order but held up: measured 350M memory matched the
  estimate. Still validate real peak with a profiler on longer runs.
- 7B "fits" at batch 1 / 72 GB, but even 350M is throughput-bound in fp32, so it
  remains the pragmatic target — and only once Route B exists.
- "The step runs" ≠ "the model can be trained from scratch here." At fp32
  throughput a Chinchilla-optimal 350M run is measured in years on one Thor; the
  resident path's near-term value is fine-tuning, inference, and small models.
