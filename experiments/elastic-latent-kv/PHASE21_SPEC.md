# Elastic Latent KV — Phase 21: WGPU device-feedback generation

## Status

Phase 21 removes the remaining generated-token host round-trip from the Phase 20 resident sampled MiniLLM path.

## Problem

Phase 20 keeps hidden states, logits and PCG state resident, but one generated token still crosses the host boundary twice between autoregressive steps:

1. `sample_next()` reads the sampled `u32` token id to the host;
2. the next `ingest_at()` writes the same `u32` token id back to WGPU.

That transfer is small, but it inserts a host-visible dependency into every generated-token step.

## Scope

Phase 21 adds a device-feedback burst path with these invariants:

- prompt tokens are primed with the existing Phase 20 path and consume zero RNG draws;
- sampled ids stay in the sampler state on WGPU;
- a bridge kernel copies each sampled id directly into the resident MiniLLM input state;
- the same bridge appends generated ids to one persistent device sequence buffer;
- the host submits the fixed dispatch sequence but does not inspect generated ids between steps;
- one compact readback returns the generated suffix after the burst;
- EOS token `0` is recorded exactly once and disables sampler + encoder state on device;
- post-EOS scheduled dispatches are no-ops, so PCG state and latent KV state do not advance after EOS;
- the sampler/encoder enable bits are restored after the final readback so the runtime remains reusable;
- host-side telemetry is synchronized from the final device control record rather than guessed per dispatch.

This is **data-resident autoregressive feedback**. Phase 21 does not claim GPU-side dynamic dispatch: the CPU still submits the known number of WGPU dispatches, but there is no per-token D2H/H2D dependency.

## Transfer boundary

For a prompt of `P` tokens and a generated suffix of at most `G` tokens:

- prompt priming: `4 * P` host-to-device bytes, unchanged from Phase 20;
- generated feedback: **0 host-to-device bytes per generated token**;
- generated feedback: **0 device-to-host bytes per generated token**;
- final result: one compact readback containing a small control header plus at most `4 * G` token bytes.

Hidden states, latent KV coefficients, logits, sampling scratch, PCG state and intermediate generated token ids remain device resident.

## Determinism and EOS

Phase 21 reuses the Phase 19 sampler kernel. Its state gains a device `enabled` word. The resident encoder gains the same concept. The feedback bridge sets both to disabled after recording EOS. Existing standalone and Phase 20 paths initialise/reset the words to enabled, so their behavior is unchanged.

The exact CPU oracle is `MiniLLM::generate_ids_cached_sampled` with the same model, prompt, `SamplingConfig` and seed.

## Validation

The permanent Phase 21 gate requires:

- nightly-2026-07-02 rustfmt;
- strict Clippy with `-D warnings`;
- Rust 1.89.0 MSRV check;
- Mesa lavapipe execution of the Phase 21 integration test.

The integration test must prove:

1. exact generated-id parity for seeded temperature sampling;
2. exact generated-id parity for combined top-k + top-p sampling;
3. prompt priming consumes zero RNG draws;
4. one final readback replaces per-token sampled-id readback/upload;
5. EOS disables further device state advancement;
6. reset/replay restores the seeded stream without reallocating persistent buffers.

## Non-goals

Phase 21 does not:

- move the dispatch loop itself into a GPU dynamic-dispatch mechanism;
- parallelize the Phase 19 O(V^2) ranking oracle;
- add CUDA device-feedback parity;
- add quantized model or KV tiers;
- change the MiniLLM training path.
