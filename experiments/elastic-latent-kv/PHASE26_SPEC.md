# Phase 26 — Promote exact parallel sampling into resident generation

## Status

Implementation phase following the Phase 25 Jetson Thor benchmark.

## Evidence carried forward

Phase 25 compared the sequential bounded-top-k sampler with the Phase 24 64-lane sampler through the same public sampling boundary. On the internal NVIDIA Thor runner at vocabulary size 4096, top-k 5/50/200 produced exact stream matches and identical fingerprints. Measured median speedups were approximately 5.62x, 18.22x and 30.68x respectively.

Phase 26 therefore promotes the parallel implementation only inside the configuration region already covered by the exact algorithm:

- finite positive temperature;
- `2 <= top_k < vocab_size`;
- `top_k <= 256`;
- WGPU adapter supports a workgroup width of at least 64.

All other configurations retain `WgpuDeterministicSampler`.

## Runtime design

`WgpuResidentSampledMiniLlm` owns an internal sampler backend enum. Backend selection is deterministic and depends only on configuration and declared WGPU capability. There is no timing-dependent autotuning.

Both resident backends expose the same internal contract:

- resident logits buffer;
- resident sampler-state buffer;
- launch without host readback;
- enabled-state control;
- host draw-counter synchronization;
- resident sample readback when the host-facing API explicitly asks for one token;
- reset to the original seeded PCG stream.

This preserves the Phase 21 device-feedback path: generated logits remain on WGPU and a burst can schedule sampler, feedback, encoder and LM-head dispatches without per-token host-visible logits.

## Exactness invariants

The promoted parallel shader does not change the historical accumulation order for top-p normalization, final probability summation, categorical scanning or PCG state mutation. These operations remain serialized on lane 0. Parallel work is restricted to comparison/reduction and bounded top-k candidate selection.

The same seeded `SamplingConfig`, prompt and weights must therefore produce the same token sequence as the CPU cached sampled generator and the sequential WGPU oracle.

## Fallback policy

The sequential sampler remains mandatory for:

- greedy/zero-temperature sampling;
- `top_k` 0 or 1;
- unbounded/full-ranking sampling;
- `top_k > 256`;
- `top_k >= vocab_size`;
- adapters unable to execute a 64-lane workgroup.

A compile, allocation or dispatch failure after the parallel backend has been deterministically selected is an error. Phase 26 does not silently fall back after such a failure because that would hide backend regressions.

## Validation

Required gates:

1. `cargo fmt --all -- --check`.
2. `cargo clippy -p scirust-gpu --all-targets --features wgpu --locked -- -D warnings`.
3. `cargo test -p scirust-gpu --features wgpu --test deterministic_sampling_parallel --locked`.
4. `cargo test -p scirust-gpu --features wgpu --test elastic_latent_minillm_sampled_resident --locked`.
5. `cargo test -p scirust-gpu --features wgpu --test elastic_latent_minillm_device_feedback --locked`.
6. Existing Phase 25 comparison remains the performance/exactness evidence for the promoted bounded-top-k kernel.
7. Native ARM64/Thor CI must remain green.

## Non-goals

Phase 26 does not introduce a new sampling algorithm, change CPU sampling semantics, alter PCG seeding, widen the supported top-k range, or add timing-based runtime dispatch policy.
