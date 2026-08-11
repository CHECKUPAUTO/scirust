# FLAT M15 paired resident decode benchmark

## Status

This benchmark is the measurement gate after SciAgent's opt-in `ResidentModel::decode_step` routing to FLAT M15. It compares the qualified FLAT pre-rotated-K path with the previous SciRust resident incremental-attention composition on the same WGPU context.

It is a measurement harness, not a default-selection policy and not a performance claim by itself.

## Comparison boundary

Both paths receive the same resident inputs:

- raw Q for one decode query;
- K already RoPE-rotated exactly once, matching SciAgent's resident KV-cache representation;
- raw V;
- identical GQA/MQA head geometry and head dimension;
- the same absolute query position (`kv_len - 1`);
- the same WGPU adapter, device and queue ownership domain.

Q RoPE is included inside both timed paths:

- legacy SciRust calls `GpuChain::rope_heads` before reproducing `ResidentModel::incr_attention` head by head;
- FLAT M15 fuses Q RoPE inside `forward_pre_rotated_k`.

K RoPE, fixture upload, bridge construction and all input setup are outside the timed region.

## Legacy path

The legacy measurement deliberately reconstructs the exact resident decode composition rather than inventing a cheaper baseline:

1. head-local Q RoPE;
2. map each query head to its grouped KV head;
3. slice resident Q/K/V head columns;
4. call the existing single-head `GpuChain::attention` path;
5. place each context head into its output columns;
6. accumulate the placed heads.

The single-head attention call is non-causal because a one-row decode query at absolute position `kv_len - 1` may attend to every cached row. This matches the pre-FLAT `ResidentModel::incr_attention` contract.

## FLAT path

The FLAT measurement calls `WgpuFlatM11Bridge::forward_pre_rotated_k` with:

- `batch = 1`;
- `query_len = 1`;
- `kv_len = resident active length`;
- causal masking enabled;
- `query_position_offset = kv_len - 1`;
- `query_rope_position_offset = kv_len - 1`;
- `kv_rope_position_offset = 0` (K is already rotated).

Because the query is at the last active cache position, the causal visibility set is the same as the legacy non-causal one-row head call.

## Synchronization and timing

The benchmark uses `std::time::Instant` around each complete attention path and immediately downloads that path's final context matrix. The final context readback is the common synchronization barrier that ensures the queued device work has completed before the elapsed time is recorded.

Therefore the reported latency is a **synchronized public-boundary measurement**. It includes:

- host-side dispatch construction/submission for the measured path;
- all device work in that attention path;
- the same final context readback used as the completion fence.

It is not a kernel-only timer, and the context readback is not part of production resident decode. Its purpose is to make the old-vs-FLAT comparison observable and paired without introducing a benchmark-only synchronization API into the production runtime.

Warmup iterations are excluded. Timed iterations alternate legacy-first and FLAT-first ordering to reduce systematic order/thermal bias. The reported statistic is median latency.

## Correctness gate

Before timing each KV length, both paths execute once and their downloaded context vectors are compared element by element with the already-qualified bridge tolerance:

- absolute tolerance: `1.5e-4`;
- relative tolerance: `1.0e-3`.

Any parity failure aborts the benchmark. The CSV also reports the maximum absolute difference for provenance.

## Default real-device matrix

The default harness uses:

- Q heads: 8;
- KV heads: 2;
- head dimension: 64;
- KV lengths: 1, 17, 64, 256, 1024;
- warmups: 3 paired iterations;
- measured iterations: 11 per path and KV length;
- RoPE theta: 10,000.

Run it with:

```bash
cargo +1.89.0 run --release --locked -p scirust-gpu \
  --features flat-attention --example flat_decode_bench
```

Optional environment controls:

```bash
SCIRUST_FLAT_DECODE_BENCH_Q_HEADS=8 \
SCIRUST_FLAT_DECODE_BENCH_KV_HEADS=2 \
SCIRUST_FLAT_DECODE_BENCH_HEAD_DIM=64 \
SCIRUST_FLAT_DECODE_BENCH_KV_LENS=1,17,64,256,1024 \
SCIRUST_FLAT_DECODE_BENCH_WARMUPS=3 \
SCIRUST_FLAT_DECODE_BENCH_REPEATS=11 \
cargo +1.89.0 run --release --locked -p scirust-gpu \
  --features flat-attention --example flat_decode_bench
```

## CSV contract

One row is emitted for each KV length with:

- adapter name;
- query-head count;
- KV-head count;
- head dimension;
- active KV length;
- warmup count;
- timed repeat count;
- legacy median latency in microseconds;
- FLAT median latency in microseconds;
- measured `legacy / FLAT` latency ratio;
- maximum absolute parity difference.

The adapter field is CSV-escaped.

## CI smoke policy

Mesa lavapipe runs a deliberately small matrix to prove the harness itself remains executable. CI verifies:

- the benchmark compiles and is warning-clean with the FLAT feature;
- the expected KV-length rows are emitted;
- dimensions and repeat counts match the requested smoke configuration;
- both median latencies and their ratio are positive and finite;
- parity error is finite (the Rust harness has already enforced the tolerance).

**CI does not require FLAT to beat the legacy path.** Lavapipe is a software Vulkan implementation and its timing is not evidence of physical-GPU throughput or production speedup.

## Promotion rule

A selection-policy change requires paired measurements on the target real adapter(s), using the same commit, geometry and synchronization contract. Only those real-device results may support a claim that FLAT is faster for a workload class.

Until that evidence exists, SciRust's existing attention remains the fallback/oracle and FLAT remains opt-in.
