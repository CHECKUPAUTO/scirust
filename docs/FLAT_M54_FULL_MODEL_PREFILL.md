# FLAT M54 — full-model SCIAGENT prefill qualification

M54 answers the next product-level question left open by M53: does the asymmetric vec4 attention gain measured on physical NVIDIA Thor survive when attention is embedded inside the complete SCIAGENT prefill path?

## Baseline evidence

M53 qualified the opt-in asymmetric vec4 kernel on the physical Thor through SciRust's product attention bridge. For GQA 8/2, sequence lengths 128/512, head dimensions 64/128 and causal/non-causal masking, vec4 matched the portable attention output exactly in all eight rows and reduced median attention latency in all eight rows. The measured portable/vec4 ratio range was `1.098703..=1.188741`.

Those measurements are attention-kernel evidence only. They do not establish a model-level prefill speedup because projections, RMSNorm, cache seeding, MLP, final norm, tied LM head and product-visible readback can dominate the total wall time.

## M54 scope

The M54 harness uses the exact `SciAgentConfig::sciagent_350m()` geometry:

- 304,088,064 parameters;
- `d_model = 1024`;
- 24 transformer layers;
- 16 query heads / 4 KV heads;
- `head_dim = 64`;
- `d_ff = 2816`;
- vocabulary 32768;
- tied embeddings.

One `GpuChain` owns the WGPU device/queue and all resident model weights. Two FLAT bridges are compiled on that same context:

1. the current portable asymmetric pipeline;
2. the opt-in M53 vec4 pipeline.

No production route is changed.

## Timed product boundary

Each sample allocates fresh fixed-capacity K/V caches before the timer, matching `ResidentModel::generate_cached`, which allocates caches before entering `ResidentModel::prefill`.

The timed region mirrors the product prefill itself:

1. resident token embedding;
2. for every one of the 24 layers:
   - RMSNorm;
   - Q/K/V projections;
   - portable or M53 FLAT causal GQA attention;
   - head-local K RoPE and resident K/V cache seeding;
   - output projection and residual;
   - second RMSNorm;
   - gate/up projections, SwiGLU, down projection and residual;
3. final RMSNorm;
4. tied LM-head projection;
5. the same full-logit device-to-host readback used by the current product prefill, followed by selection of the last prompt row.

Weight upload, bridge compilation and K/V cache allocation are excluded from timing. The two routes share the same resident weights, prompt, adapter and WGPU context.

## Correctness gate

Before timing, portable and vec4 run once with fresh caches. M54 requires:

- bit-identical final vocabulary logits;
- identical greedy next-token argmax.

A parity failure aborts the benchmark and invalidates all timing claims.

## Physical Thor protocol

The GitHub workflow must:

- run only on a persistent `tarek-scirust-arm64-01..04` physical Thor runner;
- verify NVIDIA Thor and Vulkan visibility;
- lock `/dev/nvidia0` using the established cross-runner exclusion contract;
- require a continuous 300-second idle window before timing;
- fail if `cuda_pretrain` or any other compute process contaminates the timing window;
- build outside shared workspaces;
- run prompt lengths 128 and 512 independently;
- use order-rotated portable/vec4 measurements;
- record the exact SciRust source revision, FLAT pin, adapter and benchmark CSV.

The default protocol uses one warm-up and five measured repeats per route because this is a 304M full-model benchmark rather than a microkernel sweep.

## Accepted physical Thor evidence (2026-08-19)

GitHub Actions run `32308929421` on exact SciRust head `e7640173cf3c9680710683753f24a5d98753cdf4` (PR #1283, FLAT pinned to `43b4c0ba08e109ac7025a01a01837da6927d05d0`) executed successfully on persistent physical runner `tarek-scirust-arm64-01`, adapter `NVIDIA Tegra NVIDIA Thor`, backend Vulkan, driver 580.00. The full protocol was enforced: exact-head checkout, compile before GPU reservation, `/dev/nvidia0` lock, 300-second idle window, order-rotated portable/vec4 measurements, and empty post-run occupancy.

Recorded rows (304M SCIAGENT model, 1 warmup, 5 repeats, medians in milliseconds):

| prompt | portable median ms | portable p95 ms | vec4 median ms | vec4 p95 ms | portable / vec4 | different_logits | max abs | portable token | vec4 token |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 553.350 | 566.970 | 554.375 | 558.238 | 0.998150 | 0 | 0.0 | 1203 | 1203 |
| 512 | 2532.007 | 2536.655 | 2486.439 | 2495.846 | 1.018327 | 0 | 0.0 | 19704 | 19704 |

The full-model evidence: at prompt 128 the vec4 route is effectively neutral (0.998x, within measurement noise); at prompt 512 it is 1.8% faster (1.018x). Both lengths produce **bit-identical final vocabulary logits and identical greedy next-token argmax** (max abs 0.0).

Per the promotion rule, the model-level median does not materially improve at prompt 128 and improves only 1.8% at prompt 512. The M53 vec4 kernel therefore remains **opt-in, bounded attention evidence** and is not promoted to default routing on the basis of its microbenchmark gain. `performance_claim=none` remains in force.

## Promotion rule

M53 remains opt-in. M54 does not change routing.

A future routing change requires a real full-model improvement on physical Thor with the exact-head evidence preserved. If the model-level median does not improve, the M53 kernel remains useful only as bounded attention evidence and must not be promoted on the basis of its microbenchmark gain.

`performance_claim=none` remains in force until physical M54 evidence is inspected and accepted.
