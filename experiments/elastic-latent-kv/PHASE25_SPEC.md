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

An internal self-hosted `jetson-thor` job runs the release benchmark with the default 4096-token matrix under WGPU/Vulkan. This is the decision-grade performance environment because it executes on a real GPU rather than software Vulkan.

The Thor job is restricted to trusted same-repository pull requests using the same protection pattern as the existing Native ARM64 workflow.

Because the Thor also hosts long-running SciAgent production training, the hardware comparison now participates in the host-local SciRust GPU lock `/tmp/scirust-thor-gpu.lock` and classifies accelerator occupancy before preparing or compiling the real-device benchmark. A recognized detached root `cuda_pretrain` SciAgent session causes the Phase 25 hardware comparison to be **deferred** successfully: no WGPU benchmark is launched, no new performance evidence is claimed, and the training process is left untouched. Unknown compute activity after acquiring the SciRust lock fails closed.

The managed-training classifier uses only process metadata readable by the self-hosted runner: the `nvidia-smi` process name, `/proc/<pid>/cmdline`, process user, parent PID and cgroup. The real-device build directory is created under `$RUNNER_TEMP` rather than inside the Git checkout, reducing its exposure to checkout/workspace cleanup.

A pre-protection run on 2026-08-08 executed the Phase 25 comparison while `cuda_pretrain` was already active and `nvidia-smi` reported 78% GPU utilization. Its sequential/parallel fingerprints matched exactly for `top_k` 5, 50 and 200, so it is useful as additional exactness evidence; its measured latencies and speedups are explicitly not treated as clean performance evidence because the accelerator was contended.

## Promotion decision

Phase 25 itself does not modify Phase 21. After clean real-device results are available:

- if the parallel prototype is materially beneficial and exact across the tested K values, the next phase may integrate it behind an explicit bounded-top-k selection policy;
- if it is neutral or slower, Phase 22 remains the production path and the next optimization should redesign the parallel algorithm rather than force promotion.

## Non-goals

Phase 25 does not:

- claim lavapipe timings as GPU performance;
- claim timings collected under another active GPU workload as decision-grade performance;
- terminate, restart or reconfigure a detached SciAgent production training process;
- impose an arbitrary speedup threshold before data exists;
- benchmark full language-model generation;
- change sampling semantics;
- change Phase 21 device-feedback integration;
- compare CUDA and WGPU samplers.
