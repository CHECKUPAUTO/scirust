# Elastic Latent KV — Phase 20: WGPU-resident sampled MiniLLM

## Status

Phase 20 composes the verified Phase 18 resident MiniLLM with the verified Phase 19 deterministic sampler on one WGPU device.

## Scope

`WgpuResidentSampledMiniLlm` keeps the complete seeded sampled cached-generation path on device:

- token embedding lookup;
- sinusoidal positional encoding;
- all resident Transformer encoder blocks and latent KV rings;
- MiniLLM post-encoder LayerNorm;
- LM-head logits;
- deterministic temperature/top-k/top-p sampling;
- PCG state.

The runtime shares a cloned `WgpuContext` handle, which refers to the same underlying WGPU device and queue. The LM head writes directly into the Phase 19 sampler's resident logits buffer.

## Prompt priming semantics

CPU `generate_ids_cached_sampled` processes every prompt token without consuming the RNG, then samples once from the last prompt token's logits. Phase 20 preserves that contract explicitly:

- `ingest_at(token, pos)` computes and stores resident logits without drawing;
- `sample_next()` consumes exactly one sampling decision from the pending logits;
- `step_sample_at(token, pos)` is the convenience composition of those two operations.

A pending logits row may be overwritten by the next prompt `ingest_at` without consuming RNG. After `sample_next`, another sample is rejected until a new token is ingested.

## Host/device boundary

For one generated token step:

- host → device: one `u32` input token id (4 bytes);
- device → host: one `u32` sampled token id (4 bytes).

The LM-head logits are written directly to the resident sampler buffer and never cross host memory.

Prompt priming requires only the 4-byte upload for each prompt token and no token-id download until sampling begins.

## Reproducibility

The Phase 19 PCG implementation and top-k/top-p sampler are reused, not reimplemented. `reset()` restores both the MiniLLM KV/position state and the exact seeded PCG state without reallocating persistent buffers.

## Validation

The Phase 20 integration suite requires exact generated-token sequence parity with `MiniLLM::generate_ids_cached_sampled` for:

- seeded temperature sampling;
- seeded simultaneous top-k and top-p sampling;
- prompt priming with zero RNG draws;
- reset/replay of the first sampled token from the same prompt and seed.

The permanent gate requires repository rustfmt, strict Clippy, exact Rust 1.89 and Mesa lavapipe execution.

## Non-goals

Phase 20 does not yet provide:

- device-side autoregressive loops that keep the sampled id entirely on WGPU between generation iterations;
- parallel vocabulary ranking / sampling kernels;
- CUDA-resident sampled MiniLLM parity;
- tokenizer execution on device;
- INT8/INT4 resident model or KV tiers;
- HOT/WARM/COLD device migration.

The next phase can eliminate the remaining per-token id round-trip by feeding the sampled token id directly into the next resident preprocess step for a bounded device-side generation loop.
