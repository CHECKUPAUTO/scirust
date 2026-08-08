# Phase 28 — End-to-end resident-generation baseline

## Status

Phase 28 follows the Phase 27 capability-planned sampler merge. It is a measurement phase, not an optimization phase. Its purpose is to establish a reproducible end-to-end baseline for the actual WGPU resident generation path before another performance change is proposed.

During Phase 28 validation, the 4096-vocabulary real-Thor probe isolated an important boundary: the bounded `top_k = 50` path is exact through 8- and 32-token device-feedback bursts, while the historical unbounded `top_k = 0` sequential oracle is exact through four generated tokens but fails at five on the tested Thor. The unbounded shader performs a full selection-sort ranking when `top_k = 0`; it is an exact oracle path, not the production baseline for a 4096-token vocabulary.

Phase 28 therefore keeps `top_k = 0` in reduced software validation, while the real-Thor performance baseline measures only the Phase 26/27 promoted bounded sampler.

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
- `top_k = 50`;
- deterministic seed: `0x28000001`;
- 3 warmups;
- 7 measured repeats.

The tokenizer is generated deterministically from unique Unicode scalar values. `CharTokenizer` reserves id 0 for EOS and id 1 for unknown input; the synthetic corpus therefore contains exactly `vocab_size - 2` unique non-reserved characters.

`top_k = 50` exercises the bounded-top-k configuration promoted by Phase 26 and selected through the Phase 27 capability planner on compatible hardware.

## Why the Thor baseline excludes `top_k = 0`

The sequential WGPU sampler intentionally preserves exact CPU ordering by selection-sorting probabilities. When `top_k` is unbounded (`top_k = 0`), its ranking limit is the full vocabulary, so the ranking work is quadratic in vocabulary size.

A fresh-process Thor diagnostic at vocabulary 4096 established:

- `top_k = 0`, burst 1: exact CPU/WGPU parity;
- `top_k = 0`, burst 2: exact parity;
- `top_k = 0`, burst 3: exact parity;
- `top_k = 0`, burst 4: exact parity;
- `top_k = 0`, burst 5: no generated token is committed (`generated_count = 0`);
- a previous 8-token run of the same oracle path also led to Vulkan/WGPU device loss on the Thor runner;
- `top_k = 50`, burst 8: exact parity, fingerprint `fb14a3ead98e6f97`;
- `top_k = 50`, burst 32: exact parity, fingerprint `c3cc52a850270b08`.

The host-stepped resident path produced the same first token as CPU in the failing oracle configurations. The evidence therefore does not justify adding host waits to Phase 21. The safe conclusion for Phase 28 is narrower: the exact full-ranking oracle is not a suitable 4096-vocabulary production throughput workload on this Thor/Vulkan stack, while the promoted bounded sampler is.

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

A separate 4096-vocabulary lavapipe probe validates the production bounded path at decode limits 8 and 32. The CSV validator checks exact CPU/WGPU parity, deterministic replay, positive raw timings and throughput, nonzero fingerprints/resident storage, zero generated per-token transfers and exact compact-readback accounting.

### Jetson Thor baseline

The real-device job requires an actually exposed NVIDIA Thor WGPU/Vulkan device. Because Phase 28 creates the first end-to-end resident-generation baseline, there is no previous hardware result that may be reused when the runner lacks the device.

The Thor job first requires the accelerator to be idle. If `nvidia-smi` reports another compute process, Phase 28 fails closed instead of publishing contaminated latency or throughput numbers. This guard was added after a diagnostic runner showed an unrelated `cuda_pretrain` process consuming approximately 86% GPU while the Phase 28 probe was running.

The Thor job then proves exact `top_k = 50` device-feedback bursts at 8 and 32 tokens and records the corresponding two-row performance baseline. It intentionally contains no minimum throughput, maximum latency or speedup threshold.

## Interpretation

Phase 28 answers a narrower question than the older CPU Elastic-Latent-KV final benchmark: what does the current real resident WGPU generator cost end-to-end after Phases 20–27 on the production bounded sampling path?

The existing final CPU benchmark remains useful for dense/latent memory variants but does not exercise `WgpuResidentDeviceFeedbackMiniLlm`, device-feedback generation or the Phase 26/27 sampler path.

Once the clean Thor baseline is recorded, the next optimization phase must be justified by the measured end-to-end profile or by a separately isolated benchmark. Phase 28 itself does not infer the next bottleneck from elapsed time alone.

## Non-goals

Phase 28 does not:

- change sampling mathematics;
- change the capability planner;
- change the resident MiniLLM implementation;
- add host waits to the Phase 21 burst;
- optimize the historical full-ranking sequential oracle;
- introduce a performance threshold;
- claim production-model throughput from this synthetic MiniLLM;
- infer direct kernel timing from host wall-clock differences;
- replace the existing CPU Elastic-Latent-KV final benchmark.
