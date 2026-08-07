# Elastic Latent KV — Phase 15: WGPU-resident MHA projections

## Status

Phase 15 turns the Phase 14 resident latent KV substrate into a material resident self-attention layer for the legacy `MultiHeadAttention` weights.

The implementation is opt-in and inference-only. The historical CPU/tape paths are unchanged.

## Goal

Phase 14 kept latent KV coefficients and attention scratch resident on WGPU, but uploaded dense Q/K/V vectors from the host for every head and token. Phase 15 removes that boundary for one MHA layer:

1. upload one normalised `d_model` input row;
2. compute Q/K/V projections on device from persistent model weights;
3. project K/V into persistent latent rings;
4. compute fixed-order latent scores and stable softmax on device;
5. aggregate latent V and reconstruct all head contexts on device;
6. apply the persistent output projection on device;
7. download one final `d_model` attention row.

For a model width `d_model`, the per-token transfer boundary is therefore exactly:

- host → device: `d_model * sizeof(f32)`;
- device → host: `d_model * sizeof(f32)`.

No Q/K/V vector crosses the host boundary.

## Persistent state

`WgpuResidentLatentMha` owns fixed-size WGPU buffers for:

- packed Q/K/V/O weights and biases;
- committed per-head key/value latent bases;
- K/V latent coefficient rings;
- softmax and projection scratch;
- one dense concatenated head context;
- one input/output row.

Persistent storage is allocated at construction and does not grow when the sliding ring wraps.

## Determinism baseline

The initial fused WGSL kernel uses one invocation and fixed loop order for:

- dense projections;
- latent projection;
- score accumulation;
- maximum reduction;
- softmax denominator accumulation;
- weighted V accumulation;
- reconstruction;
- output projection.

This is a correctness and reproducibility baseline, not a throughput claim. A later parallel kernel must be differentially validated against this path before replacing it.

## Basis scope

Phase 15 accepts one uniform latent rank for all heads in the resident MHA instance. Each head still owns independent row-major `(d_head, rank)` key and value bases.

Per-head heterogeneous ranks, HOT/WARM/COLD storage formats, sparse residual payloads and live basis-version migration are intentionally outside this phase.

## Weight lifecycle

Weights are snapshotted from `MultiHeadAttention` at construction. `reload_weights` replaces Q/K/V/O data in the existing WGPU allocation and rejects topology changes. `reset` clears only logical ring state and reuses every persistent allocation.

## Validation requirements

The Phase 15 gate must prove:

1. workspace `rustfmt` passes;
2. strict Clippy passes with `-D warnings` for `scirust-gpu --features wgpu`;
3. Rust 1.89.0 builds the target;
4. the fused full-rank resident MHA matches the legacy dense incremental MHA within documented FP32 tolerance;
5. a lower-rank sliding ring matches an independent CPU latent-attention oracle after wrap;
6. out-of-order positions are rejected before logical state mutation;
7. reset preserves persistent allocation size;
8. Mesa lavapipe executes the real WGSL path in CI.

## Explicit non-goals

Phase 15 does **not** claim a fully device-resident Transformer encoder. LayerNorm, residual additions, FFN blocks and the legacy tape bridge still cross the host boundary outside this MHA primitive.

It also does not implement CUDA persistent latent KV, quantised HOT/WARM/COLD tiers, sparse residual migration, or throughput-optimised parallel attention. Those remain separate follow-up work.
