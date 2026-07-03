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
  are the dominant transformer FLOPs; the fully-resident all-op path
  (`GpuChain`, where the 8–60× `gpu_bench` speedups live) is the next lift.

### Route B — native CUDA for Blackwell (sm_110) — later, for throughput
- CUDA 13, `cudarc` 0.19+, compute capability sm_110; Tensor cores with
  FP8/FP4 — the Thor's real horsepower.
- Work: a CUDA backend in `scirust-gpu` (or a sibling crate) with cuBLAS/
  cuDNN or hand-written WMMA kernels; a flash-attention kernel.
- Pro: maximum tokens/sec. Con: a substantial new backend; ties the build to
  the CUDA toolchain.

Recommended sequencing: **A to get correctness and a working run on the Thor,
then B to make it fast.**

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
   - Remaining: fully-resident all-op path (`GpuChain`) to cut per-op host
     round-trips; mixed precision (bf16 compute, fp32 optimizer master).
3. **Scale-up run.**
   - `sciagent_350m` with flash + checkpointing, seq 2048→4096, bf16, on the
     Thor. Use `sciagent-plan` to pick seq/batch for the memory budget.
   - Checkpoint to the 128 GB host memory / NVMe; evaluate with `sciagent-eval`.
4. **Route B CUDA (optional, for throughput).**

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

- A correct, fast GPU backend is a multi-week systems project, not a
  single-session change; it must be developed where a GPU is present.
- The planner is first-order: validate real peak memory with a profiler on the
  first Thor runs and adjust the linear constants.
- 7B "fits" at batch 1 / 72 GB, but throughput on a single Thor makes 350M the
  pragmatic target first.
