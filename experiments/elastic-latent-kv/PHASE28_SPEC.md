# Phase 28 — End-to-end resident-generation baseline

## Status

Phase 28 follows the Phase 27 capability-planned sampler merge. It is a measurement phase, not an optimization phase. Its purpose is to establish a reproducible end-to-end baseline for the actual WGPU resident generation path before another performance change is proposed.

## What is measured

The benchmark exercises the public `WgpuResidentDeviceFeedbackMiniLlm::generate_ids_resident()` boundary used by the Phase 21 device-feedback path and the Phase 26/27 sampler integration.

Model/runtime construction and the CPU oracle are outside the timed region.

For each sampling profile and requested decode length, the benchmark records:

- prompt-only median latency (`generate_ids_resident(prompt, 0)`);
- prompt+decode median latency;
- the arithmetic difference between those medians as `incremental_decode_proxy_ns`;
- generated tokens per second using the full prompt+decode median;
- actual generated-token count, which may be shorter than the requested limit if EOS is reached;
- output fingerprint;
- resident bytes;
- prompt upload bytes per token;
- generated upload/download bytes per token;
- final burst readback bytes;
- sampler draw count.

`incremental_decode_proxy_ns` is deliberately named a proxy. It is a subtraction of independently measured medians and is not a direct GPU kernel-time, sampler-time or dispatch-time measurement. It may be noisy and is never used as a pass/fail performance threshold.

## Fixed synthetic model

The default real-device baseline uses one deterministic synthetic MiniLLM topology:

- vocabulary: 4096 tokens;
- `d_model = 64`;
- 4 attention heads;
- 2 transformer layers;
- `d_ff = 128`;
- prompt length: 16 tokens;
- requested decode lengths: 8 and 32 tokens;
- temperature: 0.9;
- top-p: 0.92;
- deterministic seed: `0x28000001`;
- 3 warmups;
- 7 measured repeats.

The tokenizer is generated deterministically from unique Unicode scalar values. `CharTokenizer` reserves id 0 for EOS and id 1 for unknown input; the synthetic corpus therefore contains exactly `vocab_size - 2` unique non-reserved characters.

The benchmark uses two sampling profiles over the same model and prompt:

- `top_k = 0`, retaining the historical unbounded/sequential sampling path;
- `top_k = 50`, exercising the bounded-top-k configuration promoted by Phase 26 and selected through the Phase 27 capability planner on compatible hardware.

This comparison is descriptive only. Phase 28 does not require one profile to beat the other.

## Exactness and transfer invariants

Every measured WGPU sequence must match `MiniLLM::generate_ids_cached_sampled()` exactly for the same model, prompt, sampling configuration and seed.

Repeated calls must replay the same sequence and fingerprint because `generate_ids_resident()` resets the resident runtime and seeded sampler at the start of a generation.

For generated tokens, the Phase 21 transfer contract remains mandatory:

- generated upload bytes per token = 0;
- generated download bytes per token = 0;
- prompt upload bytes per token = 4;
- one compact final readback of `(4 + requested_decode_tokens) * 4` bytes;
- sampler draws equal the number of actually generated tokens, including EOS when present.

No host-visible per-token logits are introduced by this benchmark.

## CI strategy

### Static gate

The permanent Phase 28 workflow checks:

1. repository rustfmt with nightly `2026-07-02`;
2. strict Clippy for `scirust-gpu` with WGPU;
3. Rust 1.89.0 `cargo check` for the same target.

### Lavapipe smoke

The software-Vulkan smoke uses a reduced deterministic configuration so CI validates the benchmark program itself without pretending to establish hardware performance:

- vocabulary 256;
- prompt length 8;
- decode limits 4 and 8;
- top-k 0 and 50;
- 1 warmup;
- 2 measured repeats.

The CSV validator checks exact CPU/WGPU parity, deterministic replay, positive raw timings and throughput, nonzero fingerprints/resident storage, zero generated per-token transfers and exact compact-readback accounting.

### Jetson Thor baseline

The real-device job requires an actually exposed NVIDIA Thor WGPU/Vulkan device. Because Phase 28 creates the first end-to-end resident-generation baseline, there is no previous hardware result that may be reused when the runner lacks the device.

The Thor job records the default 4096-vocabulary four-row matrix (`top_k` 0/50 × decode 8/32) and applies the same correctness/invariant checks as lavapipe. It intentionally contains no minimum throughput, maximum latency or speedup threshold.

## Interpretation

Phase 28 answers a narrower question than the older CPU Elastic-Latent-KV final benchmark: what does the current real resident WGPU generator cost end-to-end after Phases 20–27?

The existing final CPU benchmark remains useful for dense/latent memory variants but does not exercise `WgpuResidentDeviceFeedbackMiniLlm`, device-feedback generation or the Phase 26/27 sampler path.

Once the Thor baseline is recorded, the next optimization phase must be justified by the measured end-to-end profile or by a separately isolated benchmark. Phase 28 itself does not infer the next bottleneck from elapsed time alone.

## Non-goals

Phase 28 does not:

- change sampling mathematics;
- change the capability planner;
- change the resident MiniLLM implementation;
- introduce a performance threshold;
- claim production-model throughput from this synthetic MiniLLM;
- infer direct kernel timing from host wall-clock differences;
- replace the existing CPU Elastic-Latent-KV final benchmark.
