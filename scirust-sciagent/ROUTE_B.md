# Route B — native CUDA + Tensor cores for the Thor (design & feasibility)

**Status: DONE — the whole Route-B stack (forward, backward, and a closed AdamW
training loop) is built, gradient-checked in bf16 on Tensor cores, and validated on
the Thor. Forward matches the CPU model to rel_err ~2.4 % and runs 6–8.3× Route A's
fp32 forward; the composed backward matches the CPU tape's tied-embedding grad
(rel_err 1.9e-1), the training loop reduces loss 4.81→0.04 over 41 steps, and a full
AdamW training step measures 3.1× (d512) / 4.7× (350M-class) Route A's fp32 training.**
This is the scoping companion to `JETSON_THOR.md`, which has the measured Route-A
ceiling this route exists to lift.

## TL;DR

- Route A (wgpu/Vulkan, fp32) is **done and validated** end-to-end on the Thor, but
  it is **kernel-efficiency bound**: ~34 tok/s training and ~3.5 tok/s single-stream
  350M decode — **<5 % of the Thor's fp32 peak** (measured, see `JETSON_THOR.md`).
  wgpu/WGSL cannot reach Blackwell **Tensor cores**, so this ceiling is structural,
  not a tuning problem.
- Route B = a **mixed-precision CUDA backend** (bf16 compute on Tensor cores, fp32
  accumulate, fp32 master weights). This is the **only** lever with an
  order-of-magnitude multiplier left.
- **B0 gate: PASSED.** On the Thor (CUDA 13.0, `compute_110`), a 350M-shaped GEMM
  `[512×4096]·[4096×4096]`: fp32 CUDA cores **3,038 GFLOP/s** vs bf16 Tensor cores
  **39,140 GFLOP/s = 12.9×** (measured under load; idle likely higher). The lever is
  real on this exact machine.
- It is a **second full backend** — bigger than the entire Route-A inference arc — but
  B0 justifies starting.

## Why (the measured ceiling)

From `JETSON_THOR.md`, all on the Thor's Blackwell, fp32, single sequence:

| regime | Route A (wgpu fp32) | why it's low |
|--------|--------------------:|--------------|
| training 350M | ~34 tok/s | ~120 GFLOP/s, <5 % of peak — kernel-bound |
| decode 350M (`m=1`) | ~3.5 tok/s | one row per forward, GPU idle |
| prefill 350M (`m=P`) | ~149 tok/s | compute-bound, but still CUDA-core fp32 |

The wall is **arithmetic throughput of fp32 CUDA-core GEMMs**. Blackwell's headline
FLOPs are in the **Tensor cores** at bf16/fp16/fp8/fp4 — unreachable from WGSL.
Nothing within Route A (bigger tiles, fusion, fewer submits — all tried or scoped)
changes the order of magnitude; a shared-memory tiled GEMM was even a **measured
regression** here (Blackwell's L2 already absorbs the reuse). Route B is the only
door to the other 95 %.

## The lever: mixed precision on Tensor cores

- **Compute** GEMMs in **bf16** (inputs) with **fp32 accumulation** — the standard
  Tensor-core mode. **bf16 over fp16** because its exponent range matches fp32, so
  **no loss scaling** is needed (fp16 would need it). Blackwell also offers fp8/fp4;
  those are a later, inference-only option, not the first target.
- **Master weights fp32**, **optimizer moments (AdamW m/v) fp32** — updates happen in
  fp32; a bf16 copy of each weight is produced for the forward/backward GEMMs. This
  is textbook mixed-precision training and preserves convergence.
- **Inference**: bf16 (or fp16) resident weights halve memory traffic — this is what
  helps the memory/latency-bound `m=1` **decode** path, on top of the FLOP boost that
  helps prefill/training.

Expected gains (**hypothesis, B0 confirms**): training 34 → few-hundred tok/s;
decode 3.5 → ~20–40 tok/s (decode gains less than the raw FLOP ratio because it stays
partly launch/memory bound). Even the low end changes 350M from-scratch pretrain from
"years" to "weeks", and makes speculative decoding (already built, exact) genuinely
worthwhile.

## How to reach Tensor cores from Rust — options

| option | what | pro | con | verdict |
|--------|------|-----|-----|---------|
| **B-cuBLASLt** | FFI to cuBLASLt for every GEMM (bf16 in, fp32 acc) | vendor-optimal GEMM for free; least kernel code | descriptor plumbing; still need CUDA kernels for the non-GEMM ops | **recommended** |
| B-CUTLASS | templated CUDA GEMM/epilogue | fuse epilogues (bias, act) into the GEMM | C++ templates, longer build, marginal vs cuBLASLt here | later, if fusion matters |
| B-WMMA | hand-written `mma.sync`/WMMA kernels | full control | will not beat cuBLASLt; large effort | no |

**Non-GEMM ops** (softmax, scale/mask, RMSNorm, RoPE, SwiGLU, slice/place,
concat/slice-rows, embed gather, cross-entropy, AdamW) still need **CUDA kernels** —
but they are a **1:1 port of the already-validated WGSL kernels** in `scirust-gpu`,
not new math. Attention can start as compose-from-primitives (as Route A does) and
later move to a **fused flash-attention** kernel for the real decode win.

**Rust↔CUDA plumbing:** `cudarc` (safe-ish driver API + cuBLAS bindings, already
named in `JETSON_THOR.md`) for device/stream/buffer management and cuBLASLt, plus a
`build.rs` that `nvcc`-compiles the custom `.cu` kernels to PTX/cubin and loads them
at runtime. Alternative: raw `cuda-sys`/`cust` FFI. **Decision needed** (see below).

## Integration architecture

A new **feature-gated `scirust-cuda` crate** (sibling to `scirust-gpu`), exposing a
`CudaChain` that **mirrors `GpuChain`'s API** (`matmul`, `matmul_t`, `rms_norm`,
`rope`, `attention`, `swiglu_mlp`, `slice_cols`, `concat_rows`, `adamw_step`, …).
Because `ResidentModel` is written against that resident-op surface, wiring it to a
`CudaChain` is then a **backend swap**, not a rewrite — the whole
train/fine-tune/generate/speculative stack rides on top unchanged.

- **Feature-gated & CI-safe:** `cuda` is off by default; the workspace must still
  `cargo build`/`clippy` with no CUDA present. Route B is **Thor-only** to build and
  test (CI has no GPU *and* no CUDA — worse than Route A, which at least builds and
  runs on lavapipe). Every Route-B test is `#[cfg(feature = "cuda")]` and skips (or
  is absent) elsewhere.
- **Validation unchanged in spirit:** each op **gradient-checked against the CPU
  oracle**, brick by brick — but at a **bf16-appropriate tolerance** (looser than
  fp32's `~2e-2`; expect `~2e-1` relative on some grads). The project's "bit-à-bit
  where possible" becomes "gradient-checked to a documented reduced-precision
  tolerance" — and that must be stated honestly wherever a number is quoted.

## Phased plan (each phase gated on the previous)

- **B0 — feasibility gate. ✅ PASSED.** CUDA 13.0 on the Thor lists `compute_110`;
  a bf16 Tensor-core GEMM measured **12.9×** the fp32 CUDA-core GEMM on a 350M shape
  (39.1 vs 3.0 TFLOP/s, under load). Well past the ~8–10× go threshold ⇒ **GO**.
- **B1 — plumbing + one GEMM. ✅ DONE.** `scirust-cuda` crate (cudarc, cuBLASLt,
  bf16); `CudaChain`/`CudaMatrix`; the bf16 Tensor-core GEMM gradient-checked vs CPU
  (rel_err 3.4e-3). Builds CUDA-free without the feature.
- **B2 — elementwise/attention kernels. ✅ DONE (forward).** NVRTC runtime-compiled,
  header-free bf16: `add`, `mul`, `swiglu`, `rms_norm`, `slice_cols`/`place_cols`,
  `softmax`, `scale_causal_mask`, `rope`, `embed`, plus cuBLASLt `matmul_bt` (A·Bᵀ).
  Each gradient-checked vs CPU at bf16 tolerance. (Backward adjoints are part of B4.)
- **B3 — resident forward. ✅ DONE.** `CudaModel` (`scirust-sciagent`, feature `cuda`)
  composes the full `embed → N×GQA → final RMSNorm → tied head` on `CudaChain` and
  matches the CPU `SciAgentModel` to **rel_err 2.37e-2** (`tests/cuda_parity.rs`).
- **B5 — measure (forward). ✅ DONE.** `examples/cuda_forward_bench` (fp32 wgpu vs
  bf16 Tensor cores, same forward): **6.0×** at d512·8L, **8.3×** at d1024·24L (350M:
  139 → 1,158 tok/s). Model-level realization of B0's 12.9× bare-GEMM (the gap is
  non-GEMM overhead + the host logits download; larger models close it).
- **B4 — backward + AdamW. ✅ DONE, validated on the Thor.**
  Built brick by brick, each gradient-checked against Route A's validated CPU oracle
  at bf16 tolerance (Thor numbers below):
  - **B4a** `matmul_at` (Aᵀ·B) — the second half of the matmul VJP (with `matmul_bt`).
    Thor: rel_err **2.37e-3**.
  - **B4b** the six backward adjoint kernels: `softmax_bwd`, `swiglu_bwd`,
    `rmsnorm_bwd`, `rmsnorm_gain_bwd`, `scale_mask_bwd`, `rope_bwd`. Thor: softmax
    6.7e-3, swiglu 2.5e-3/2.2e-3, rms 2.6e-3/3.4e-3, mask 1.2e-3, rope 2.1e-3.
  - **B4c** `embed_backward` (atomic-free scatter-add) + `cross_entropy_grad`. Thor:
    1.7e-3 / 1.6e-3.
  - **B4d** mixed-precision **AdamW**: `CudaF32` fp32 master weights + moments, bf16
    grad in, refreshed bf16 view out (checked over two steps vs `cpu_adamw_step`).
    Thor: fp32 master **~1e-8** (effectively exact), bf16 view 1.5e-3.
  - **B4e** the composed `CudaModel` backward (attention → block → model), validated by
    the **tied-embedding grad** vs the CPU tape — one number covering the whole chain.
    Thor: rel_err **1.95e-1** (bf16 backprop through a 2-layer decoder compounds; a
    wiring bug would be `O(1)`).
  - **B4f** `CudaTrainer`: the closed loop (forward → CE grad → backward → AdamW,
    refreshing bf16 views), proven to **reduce loss** by overfitting a fixed batch.
    Thor: loss **4.81 → 0.04** over 41 steps.
  - **B4g** `examples/cuda_train_bench` — the training-throughput bench (bf16 TC vs
    fp32 wgpu). Thor: **3.1×** at d512·8L (413 → 1,285 tok/s), **4.7×** at d1024·24L
    (45.7 → 213 tok/s).

  This turns Route B from inference-capable into **training-capable**: the full bf16
  forward+backward+optimizer runs on Blackwell Tensor cores. The training speedup
  (3.1–4.7×) is below the forward's 6–8.3× — expected and honest: the backward carries
  more non-GEMM adjoint kernels and the fp32-master AdamW step is memory-bound, both of
  which dilute the raw bf16-GEMM win.

- **B6 — production pretraining harness. ✅ DONE, validated on the Thor.**
  `CudaTrainer::pretrain` (warmup+cosine LR, wrapping token windows, throughput
  logging, resumable safetensors checkpointing) + `sync_to_model` (writes the fp32
  masters back, full-fidelity checkpoints), driven by `examples/cuda_pretrain` — the
  Route-B twin of `resident_pretrain`, but CUDA-only (no wgpu dependency), same
  `SCIAGENT_*` env interface. **Thor:** a real from-scratch byte-level pretrain on the
  scirust code tree (16 M bytes) drove loss **6.41 → 2.99 nats/byte** (53 %) over 500
  bf16 steps at ~2,300 tok/s, with checkpoint/resume across separate invocations. The
  350M-from-scratch pretrain — the goal that motivated the whole route — now runs in
  bf16 on the Thor, resumable.

- **B7/B8 — training stability at 270M (debugged on the Thor).** Scaling to a
  byte-level ~270M model (`code350m` preset) exposed a **deterministic collapse**: the
  loss fell to ~2, then jumped to the `ln(256) = 5.55` uniform floor and stuck. The
  debugging is worth recording because three plausible fixes each *disproved* a
  hypothesis:
  - **LR** (3e-3 → 3e-4): collapsed at the *same* step — not LR magnitude.
  - **B7 grad-norm clipping** (`sumsq` reduction + `global_grad_norm` + a `grad_scale`
    in `adamw_step`, gradient-checked): the `gnorm` it logged stayed *small* (~5) at the
    collapse — no spike — and Adam's `m/√v` is nearly scale-invariant to global clipping
    anyway, so it was a near-no-op. (Kept as hygiene + the `gnorm` diagnostic.)
  - **AdamW eps** (1e-8 → 1e-5, bf16-appropriate): no change.

  The **localizer** settled it: the collapse was at the *same corpus byte offset*
  (~166,400) at two seq lengths (step 325 @ 512, step 650 @ 256), i.e. a **pathological
  batch, not the optimizer**. The sorted file walk reads `.git` first, so byte 166 K was
  deep in **`.git`'s packed binary objects** — the model was training on compressed
  garbage. **B8** fixes ingestion to source text only (skip `.git`/`target`/… and
  non-UTF-8/NUL files). With clean data the same 270M run trains **stably**: loss
  **12.16 → 1.97 nats/byte** (84 %) over 500 steps, sailing straight through the old
  step-325 cliff. Lesson: a data-quality bug can masquerade as a numerics bug — the
  fix was hygiene, not the optimizer.

- **B9 — BPE-350M pipeline, validated end-to-end.** The *true* 350M (32768-vocab
  BPE, 304M params) runs the full path on the Thor: `train-tokenizer` (32574 merges)
  → `collect-data` (3.25M tokens → LE-u32 shards, multi-extension + dir hygiene) →
  `cuda_pretrain SCIAGENT_CONFIG=350m SCIAGENT_SHARDS=…`. 3000 bf16 steps at ~129
  tok/s: loss **10.45 → 8.94** — which *looks* weak but is the ln(32768)=10.4 vocab
  scale; **normalized per character it's ~1.79 nats/char** (8.94 ÷ ~5 chars/token),
  matching the byte model's 1.97 nats/char while processing 5× fewer tokens per
  document — the efficiency BPE buys. The plateau is under-training, not instability
  (3000 × 512 = 1.5 M tokens ≈ half an epoch of a 304M model; `gnorm` small, no
  collapse). Both a byte-level ~270M and a BPE ~304M from-scratch pretrain now run
  raw-source → trained checkpoint in bf16 on the Thor, resumable.

## Risks / honesty

- **Toolchain gate (highest risk):** if the Thor's installed CUDA can't emit sm_110,
  there is **no Tensor-core codegen** and Route B is blocked until JetPack/CUDA is
  upgraded. B0 catches this on day one.
- **CI can't cover it:** unlike Route A (lavapipe), nothing in CI builds or runs the
  CUDA path. Regressions are Thor-only to catch. Mitigate by keeping `CudaChain`'s CPU
  oracles and parity tests identical in structure to Route A's, run on the Thor.
- **Determinism erodes further:** bf16 + Tensor-core accumulation widens the gap to
  the CPU reference beyond fp32-wgpu's. Tolerances must be re-derived and **documented
  per op**; some finite-difference grad checks may need larger epsilons.
- **Effort vs payoff:** it's a second backend. Justified **only** by B0's number and
  by the intent to actually do large-scale training or fast serving on the Thor — for
  fine-tuning + short on-device generation, Route A already suffices.
- **Build coupling:** `nvcc` in the build, CUDA libs linked; keep it entirely behind
  the `cuda` feature so non-Thor builds are unaffected.

## B0 — the go/no-go probe (done)

Ran on the Thor: `nvcc` **13.0** lists `compute_110` (driver 580, CUDA 13.0), and a
`cublasGemmEx` bf16 GEMM vs `cublasSgemm` fp32 on `[512×4096]·[4096×4096]` gave
**39,140 vs 3,038 GFLOP/s = 12.9×** (under an active ollama load). **GO.**

## Open decisions

1. **Plumbing:** `cudarc` (safer, batteries-included) vs raw FFI (`cuda-sys`/`cust`,
   more control). *Recommend `cudarc`.*
2. **Precision:** bf16 (no loss scaling) vs fp16 (needs it). *Recommend bf16.*
3. **Scope:** full CUDA resident path (recommended — mixing wgpu+CUDA in one process
   is painful) vs a GEMM-only CUDA shim under the existing wgpu chain (not
   recommended).
