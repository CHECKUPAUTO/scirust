# SCIAGENT I250 — batch-one inference target

## Objective

Reach **at least 250 generated tokens/s** for the 304,088,064-parameter SCIAGENT
`350m` configuration on Jetson AGX Thor, without modifying the Route-B production
training path or silently changing generation semantics.

The historical B49 cached CUDA decoder remains the correctness oracle until a faster
path is explicitly promoted. The project target is not limited to 250 tok/s: I250 is
the first production gate, while the architecture must leave room for substantially
higher throughput through SciRust-native representations rather than only classic KV
optimisation.

## Isolation from the active training run

I250 lives in dedicated inference implementations. It does not replace `CudaChain`,
`CudaTrainer`, `cuda_pretrain`, checkpoint state, or optimizer math. The semantics-v2
production training process can therefore continue from its frozen source SHA while
I250 evolves independently.

Current implementation files:

- `scirust-cuda/src/decode.rs`
- `scirust-sciagent/src/cuda_decode.rs`
- `scirust-sciagent/src/elastic_decode_plan.rs`

## I250-A — remove structural dense-decode overhead

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

This remains useful as a dense correctness/performance baseline, but it is not the
intended endpoint.

## I250-B — ElasticKV absorbed decode

SciRust's Elastic Latent KV already provides reconstruction-free attention: key scores
are evaluated from latent coefficients, value aggregation stays latent, and only the
final value context is up-projected. I250-B moves the latent boundary into the model
weights themselves.

For one GQA KV head, with key basis `U_k` and value basis `U_v`, plan construction
precomputes:

```text
Wq_lat = Wq_head * U_k
Wk_lat = Wk_head * U_k
Wv_lat = Wv_head * U_v
Wo_lat = U_v^T * Wo_head
```

The intended CUDA token path can then:

1. project Q/K/V directly to latent coordinates;
2. keep historical K/V only in ElasticKV latent form;
3. score keys reconstruction-free;
4. accumulate values reconstruction-free;
5. map each latent head context directly to `d_model` through `Wo_lat`;
6. avoid both dense historical K/V and dense per-head value reconstruction.

At half key/value rank, the Q/K/V/O projection-weight traffic represented by this
plan is exactly half of the corresponding dense GQA projection-weight traffic. This is
an algebraic accounting result, not yet a full-model speed claim.

### RoPE rule

SCIAGENT semantics-v2 applies head-local RoPE after Q/K projection. Therefore a latent
basis is **not** allowed to assume that ordinary RoPE at latent width is equivalent.
`elastic_decode_plan` classifies the key basis explicitly:

- **FullIdentity** — full-rank identity coordinates; structural no-loss oracle.
- **NativePairPrefix** — reduced basis retains complete native RoPE pairs. The CUDA
  kernel can rotate the retained pairs directly, but the frequency exponent must keep
  the original `d_head` denominator, never the reduced rank.
- **ProjectedOperator** — a general learned basis. The runtime must use the
  basis-projected position operator; reduced-rank use is quality-gated and cannot be
  described as exact dense equivalence.

This distinction is permanent: a backend must never gain speed by silently changing
the model's positional geometry.

### Relationship to ElasticKV evolution

The first CUDA implementation will use a simple native-pair basis so its positional
math is independently auditable. Once the full-rank and reduced-prefix paths are
validated, the same boundary can consume committed learned ElasticKV bases, adaptive
ranks, sparse residuals, INT8/INT4 coefficient tiers, and HOT/WARM/COLD lifecycle
policies. Those mechanisms stay separate from the model-training trajectory and are
promoted only through measured quality gates.

## Sampling

The shared SCIAGENT host sampler is retained initially. Device-resident sampling is a
later optimization and can reuse SciRust's exact resident/parallel sampler work rather
than duplicating its probability or PCG semantics in a second implementation.

## Correctness gates

Before any fast path is promoted, all of the following are mandatory:

- existing Route-B CUDA tests remain green;
- dense `cuda_decode_parity` matches B49 cached greedy generation token-for-token;
- Elastic full-rank identity plan is bit-identical at the source-weight boundary;
- the CUDA full-rank Elastic path matches the dense oracle before any reduced rank is
  benchmarked as a production candidate;
- reduced-rank/quantized paths pass explicit model-quality gates; they are never
  relabelled as exact merely because their token sequence happens to match one prompt;
- the real 304M Thor benchmark reports its exact mode/rank/format and parity/quality
  status;
- no inference runtime modifies training/checkpoint files.

Fused GEMM shapes can choose a different floating-point reduction order from the
oracle even when every explicit bf16 boundary is preserved. Therefore generated-token
parity is measured, never assumed.

## Performance gate

`cuda_decode_bench` emits one machine-readable dense-I250 record. Elastic I250-B will
extend the same benchmark boundary with explicit representation/rank fields rather
than creating incomparable timing conventions.

```text
SCIAGENT_I250_DECODE ... fast_tok_s=... b49_tok_s=... speedup=... target_tok_s=250 target_met=... parity=...
```

For the dense production gate:

```bash
SCIAGENT_DECODE_TARGET_TPS=250 \
SCIAGENT_DECODE_REQUIRE_TARGET=1 \
cargo +nightly-2026-07-02 run \
  -p scirust-sciagent \
  --features cuda \
  --release \
  --example cuda_decode_bench
```

Do **not** run the Thor performance gate concurrently with the root production
pretrainer: concurrent GPU load makes the throughput result meaningless and steals
device time from training.

## Escalation beyond I250

If the first BF16 paths do not reach the target, optimisation proceeds from measured
costs rather than a standard-framework checklist:

1. compare dense-I250-A against Elastic-I250-B at identical benchmark boundaries;
2. profile projection-weight bandwidth, attention history bandwidth, kernel-launch
   cost, and host sampling synchronization independently;
3. specialise `m=1` projection kernels/layouts where cuBLASLt is not the best fit;
4. combine ElasticKV adaptive rank/format/lifecycle decisions with the deterministic
   compute capability planner rather than hard-coding one representation globally;
5. add CUDA Graph/persistent execution when launch latency is material;
6. add lower-precision model weights only behind an explicit quality gate;
7. move deterministic sampling and token feedback fully resident;
8. evaluate multi-token/speculative execution only as a separate exactness/quality
   contract, never by conflating aggregate throughput with single-token decode.

No path is promoted solely because it is faster. Reproducibility, declared numerical
semantics, quality, and hardware-measured throughput are independent gates.
