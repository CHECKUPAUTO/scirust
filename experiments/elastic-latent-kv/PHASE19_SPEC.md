# Elastic Latent KV — Phase 19: deterministic WGPU sampling substrate

## Status

Phase 19 establishes a portable deterministic sampling substrate before sampled MiniLLM generation is moved on device.

## Problem

SciRust's CPU `sample_token` is driven by `PcgEngine`, whose state transition is 64-bit. Portable WGSL does not expose a native `u64` suitable for this path. Replacing PCG with another RNG would break seeded reproducibility.

## Scope

Phase 19 adds `WgpuDeterministicSampler` with:

- persistent PCG state on WGPU;
- two-`u32` emulation of the 64-bit PCG state transition modulo 2^64;
- exact 32x32-to-64 multiplication through 16-bit limbs;
- CPU-compatible temperature scaling;
- deterministic probability ordering, descending with lower-token-id tie break;
- top-k masking;
- nucleus top-p masking;
- categorical sampling using the next PCG word;
- greedy shortcut semantics matching `scirust_core::nn::sampling::sample_token`;
- persistent logits/scratch/state allocations.

## Correctness baseline

The kernel deliberately runs in one invocation and uses a fixed-order selection sort. Complexity is O(vocab^2). This is an oracle/reproducibility baseline, not a throughput implementation.

Vocabulary size is restricted below 2^24 because scratch order indices are represented as exact FP32 integers in this baseline.

Non-finite logits are rejected at the host boundary.

## Validation

The Phase 19 integration tests compare WGPU-selected token ids exactly against the CPU `sample_token` stream for:

- seeded temperature sampling over a varying logit sequence;
- simultaneous top-k and top-p sampling;
- greedy equal-logit tie breaking;
- invalid shape and non-finite input rejection.

The permanent gate requires rustfmt, strict Clippy, Rust 1.89 and Mesa lavapipe execution.

## Non-goals

Phase 19 does not yet:

- integrate sampling into `WgpuResidentMiniLlm`;
- eliminate host logits upload for this standalone sampler oracle;
- claim PCG state compatibility for unsupported/non-finite sampling configurations;
- provide a parallel ranking kernel;
- provide a CUDA sampler.

The next phase should reuse this verified algorithm inside the resident MiniLLM postprocess so logits remain entirely on device.
