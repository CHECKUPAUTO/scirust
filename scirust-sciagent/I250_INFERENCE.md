# SCIAGENT I250 — batch-one inference target

## Objective

Reach **at least 250 generated tokens/s** for the 304,088,064-parameter SCIAGENT
`350m` configuration on Jetson AGX Thor, without modifying the Route-B production
training path or silently changing generation semantics.

The historical B49 cached CUDA decoder remains the correctness oracle until the fast
path is explicitly promoted.

## Isolation from the active training run

I250 lives in a dedicated inference implementation:

- `scirust-cuda/src/decode.rs`
- `scirust-sciagent/src/cuda_decode.rs`

It does not replace `CudaChain`, `CudaTrainer`, `cuda_pretrain`, checkpoint state, or
optimizer math. The semantics-v2 production training process can therefore continue
from its frozen source SHA while I250 evolves independently.

## I250-A — remove structural decode overhead

The first implementation attacks overhead that is unrelated to model quality:

1. **Fixed-capacity resident KV** — no per-token `concat_rows`, historical K/V copy,
   or cache reallocation.
2. **Fused single-query GQA kernel** — one launch per layer performs head-local
   semantics-v2 RoPE, current K/V insertion, QK scores, bf16 score boundary,
   256-way softmax, B49-style left-to-right context accumulation, and head assembly.
3. **Fused projection weights** — Q/K/V are stored as one `[Wq | Wk | Wv]` matrix;
   SwiGLU gate/up are stored as `[Wg | Wu]`.
4. **Persistent activation workspace** — token steps reuse preallocated device
   buffers instead of allocating temporary matrices for every operator.
5. **Resident prompt replay** — incremental prompt prefill performs no logit readback
   until the final prompt position.

The shared SCIAGENT host sampler is retained initially. Device-resident sampling is a
later optimization and can reuse the exact-sampler work being developed elsewhere in
SciRust rather than duplicating it here.

## Correctness gates

Before promotion, all of the following are mandatory:

- existing Route-B CUDA tests remain green;
- `cuda_decode_parity` matches B49 cached greedy generation token-for-token over the
  deterministic test prompts;
- the real 304M Thor benchmark reports `parity=true`;
- no training/checkpoint file is modified by the inference runtime.

Fused GEMM shapes and fused attention can choose a different floating-point reduction
order from the oracle even when every explicit bf16 boundary is preserved. Therefore
**token parity is measured, never assumed**.

## Performance gate

`cuda_decode_bench` emits one machine-readable record:

```text
SCIAGENT_I250_DECODE ... fast_tok_s=... b49_tok_s=... speedup=... target_tok_s=250 target_met=... parity=...
```

For the production gate:

```bash
SCIAGENT_DECODE_TARGET_TPS=250 \
SCIAGENT_DECODE_REQUIRE_TARGET=1 \
cargo +nightly-2026-07-02 run \
  -p scirust-sciagent \
  --features cuda \
  --release \
  --example cuda_decode_bench
```

Do **not** run this gate concurrently with the root production pretrainer: concurrent
GPU load makes the throughput result meaningless and needlessly steals device time
from training.

## If BF16 I250-A is below 250 tok/s

Optimization proceeds in measured order rather than by guesswork:

1. profile batch-one projection bandwidth and launch cost;
2. replace cuBLASLt `m=1` projections with a decode-specific transposed-weight GEMV
   only if the measured bandwidth is better;
3. fuse residual/bias-free epilogues and SwiGLU projection work where parity permits;
4. add CUDA Graph replay if launch latency is material;
5. introduce an explicitly quality-gated lower-precision inference weight format
   (FP8/INT8) only if BF16 memory bandwidth is the remaining wall;
6. move deterministic sampling/device feedback fully resident if host synchronization
   is still measurable.

No quantized path is promoted solely because it is faster: quality and deterministic
sampling behavior are separate gates.
