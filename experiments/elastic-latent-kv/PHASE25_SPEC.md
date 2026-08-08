# Elastic Latent KV — Phase 25: sequential vs parallel sampler benchmark

## Goal

Phase 25 measures the exact Phase 22 sequential bounded-top-k sampler against the exact Phase 24 64-lane parallel prototype through the same public `sample()` boundary. The result decides whether the parallel sampler deserves promotion into the Phase 21 device-feedback generation path.

## Measurement boundary

Both implementations are constructed before timing. Each timed sample includes:

1. one FP32 logits upload;
2. one WGPU dispatch;
3. synchronization;
4. one `u32` sampled-token readback.

The benchmark therefore compares equivalent end-to-end sampler calls rather than kernel-only timings.

## Default real-device matrix

The default comparison uses:

- vocabulary: 4096;
- `top_k`: 5, 50, 200;
- temperature: 0.9;
- top-p: 0.92;
- three warmups;
- seven timed samples;
- identical logits and seed for both implementations.

This matrix exercises small, medium and large bounded K while remaining inside the Phase 24 `K <= 256` contract.

## Exactness gate

A timing row is valid only when sequential and parallel samplers emit the exact same token sequence. The CSV includes both fingerprints and an `exact_stream_match` flag.

The benchmark does not change PCG state semantics, probability ordering, top-p accumulation order or categorical scanning order.

## Timing interpretation

The CSV reports:

- sequential median sample latency;
- parallel median sample latency;
- `sequential_median / parallel_median` as `speedup_parallel_vs_sequential`;
- exact-stream match;
- both output fingerprints;
- sequential and parallel ranking passes;
- parallel workgroup lane count.

A speedup greater than 1 means the parallel implementation was faster on that adapter for that row. No fixed pass/fail performance threshold is encoded before measurements are observed.

## CI strategy

Phase 25 has two benchmark environments:

### Mesa lavapipe

A small smoke matrix proves that the benchmark compiles, executes, emits structurally valid rows and preserves exact streams. Its timing is explicitly not treated as real-GPU performance.

### Jetson Thor

An internal self-hosted `jetson-thor` job runs the release benchmark with the default 4096-token matrix under WGPU/Vulkan. This is the decision-grade performance evidence for promotion because it executes on a real GPU rather than software Vulkan.

The Thor job is restricted to trusted same-repository pull requests using the same protection pattern as the existing Native ARM64 workflow.

## Promotion decision

Phase 25 itself does not modify Phase 21. After the real-device results are available:

- if the parallel prototype is materially beneficial and exact across the tested K values, the next phase may integrate it behind an explicit bounded-top-k selection policy;
- if it is neutral or slower, Phase 22 remains the production path and the next optimization should redesign the parallel algorithm rather than force promotion.

## Non-goals

Phase 25 does not:

- claim lavapipe timings as GPU performance;
- impose an arbitrary speedup threshold before data exists;
- benchmark full language-model generation;
- change sampling semantics;
- change Phase 21 device-feedback integration;
- compare CUDA and WGPU samplers.
