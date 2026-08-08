# Elastic Latent KV — Phase 23: deterministic WGPU sampler benchmark

## Status

Phase 23 characterizes the Phase 22 bounded top-k sampler with a reproducible benchmark. It separates deterministic algorithmic work from wall-clock timing so performance claims remain scoped to the adapter that actually executed the benchmark.

## Measurement boundary

The benchmark calls the public `WgpuDeterministicSampler::sample()` API. Each timed sample therefore includes:

1. one FP32 logits upload (`V * 4` bytes);
2. one deterministic WGPU sampler dispatch;
3. host synchronization on that dispatch;
4. one sampled-token readback (`4` bytes).

It is not a kernel-only timer and it is not the zero-per-token-transfer Phase 21 generation boundary. Phase 21 composes the same sampler without reading a token back after every generated step.

## Compared configurations

For every requested vocabulary size, the default benchmark runs:

- `top_k=0`: full-ranking fallback;
- `top_k=1`: greedy shortcut;
- `top_k=5`;
- `top_k=50`;
- `top_k=200`.

Temperature is fixed at `0.9` and top-p at `1.0` so the benchmark isolates the ranking-path choice rather than adding nucleus truncation work.

The canonical defaults are vocabulary sizes 1,024 and 4,096 with seven timed samples after two warmups. Larger production vocabularies can be requested explicitly, but the full O(V²) fallback may intentionally be expensive.

## Deterministic work metric

For vocabulary `V` and `P = ranking_passes_per_sample()`, selection sort performs exactly

`P * (2*V - P - 1) / 2`

candidate comparisons.

Phase 22 therefore reports:

- `P=0` for greedy;
- `P=K` when `0 < K < V` and `K != 1`;
- `P=V` for the full-ranking fallback.

The CSV includes both the exact comparison count and its fraction of a full `V`-pass ranking. This metric is deterministic and adapter-independent.

## Timing metric

Each timed call is measured with `std::time::Instant`. The benchmark reports the median sample latency and the reciprocal samples/second for that same public boundary.

Timing is environment-specific:

- a Jetson Thor or other real WGPU adapter measures that device and driver stack;
- Mesa lavapipe is a software Vulkan implementation and its numbers are not GPU-throughput claims;
- CI timing is used only to prove that the harness executes and emits positive finite measurements.

No fixed latency or speedup threshold is enforced in CI.

## Replay proof

After timed sampling, the sampler is reset to its exact seeded PCG state and the same number of outputs is generated again from identical logits. `deterministic=1` is emitted only if every sampled token matches the timed sequence exactly.

The output fingerprint is FNV-1a over the sampled token ids. It is a compact replay diagnostic, not a cryptographic digest.

## CSV contract

The benchmark emits:

- vocabulary size;
- configured top-k;
- ranking passes;
- exact selection comparison count;
- comparison fraction versus full ranking;
- bounded-fast-path flag;
- timed repeat count;
- median public-boundary sample latency;
- samples/second;
- deterministic replay flag;
- output fingerprint;
- resident bytes;
- upload bytes per sample;
- download bytes per sample.

## Running

```bash
cargo +1.89.0 run --quiet --release --locked -p scirust-gpu \
  --features wgpu --example deterministic_sampler_bench
```

Optional controls:

```bash
SCIRUST_SAMPLER_BENCH_VOCABS=1024,4096 \
SCIRUST_SAMPLER_BENCH_TOP_KS=0,1,5,50,200 \
SCIRUST_SAMPLER_BENCH_REPEATS=9 \
SCIRUST_SAMPLER_BENCH_WARMUP=3 \
cargo +1.89.0 run --quiet --release --locked -p scirust-gpu \
  --features wgpu --example deterministic_sampler_bench
```

## CI smoke contract

The permanent Phase 23 gate uses Mesa lavapipe with small vocabularies and verifies structure rather than timing magnitude. It requires:

- nightly-2026-07-02 rustfmt;
- strict Clippy with warnings denied;
- exact Rust 1.89.0 compatibility;
- successful release execution of the benchmark on lavapipe;
- the expected row matrix;
- exact ranking-pass and comparison-count formulas;
- correct fast-path classification;
- positive median timings and finite throughput values;
- deterministic replay for every row;
- non-zero fingerprints;
- exact transfer accounting (`V*4` upload bytes, `4` download bytes).

## Non-goals

Phase 23 does not:

- claim that lavapipe timing predicts real GPU timing;
- compare different GPUs as if their timings were interchangeable;
- hide upload/readback cost inside a kernel-only number;
- change the sampler algorithm introduced in Phase 22;
- parallelize ranking;
- change PCG, probability ordering, or sampling semantics;
- benchmark full end-to-end language-model generation.

The resulting real-device baseline is intended to guide the next optimization phase, where parallel ranking can be compared against the same public measurement contract.
