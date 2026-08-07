# Elastic Latent KV — Phase 14 WGPU Resident State

Phase 14 starts the transition from Phase 12's accelerator primitives to material device-resident decode state.

## Delivered substrate

`WgpuResidentLatentKvCache` lives in `scirust-gpu`, preserving the repository dependency direction `scirust-gpu -> scirust-core`.

For one attention head it owns persistent WGPU buffers containing:

- the committed key and value latent bases;
- a sliding ring of key latent coefficients;
- a sliding ring of value latent coefficients;
- projected-query scratch;
- score / softmax scratch;
- latent context scratch;
- a reusable dense IO buffer and parameter buffer.

The persistent buffers are allocated once at construction. Appending beyond capacity overwrites the oldest ring slot and does not grow persistent device storage.

## Per-token execution

`append` uploads one dense key and value vector. A WGSL kernel projects both vectors into their lower-rank bases and writes the coefficients directly into the resident ring.

`attention_into` uploads one dense query vector. A second WGSL kernel performs, in fixed logical oldest-to-newest order:

1. query projection into the key latent basis;
2. scaled query/key latent dot products;
3. numerically stable softmax;
4. weighted value aggregation in latent space;
5. reconstruction through the value basis.

Only the reconstructed dense head context is read back. The KV history, scores and latent context do not round-trip through host memory.

The current kernels use one invocation with fixed loop order. They are a deterministic correctness baseline, not a throughput claim. Future parallel kernels must be differentially validated against this baseline.

## Lower-rank compatibility

The cache consumes rectangular `[dimension, rank]` bases directly. It therefore accepts the committed lower-rank bases introduced by the Phase 10 -> Phase 13 handoff without padding them to a selectable full-rank basis.

## Scope boundary

This phase establishes real persistent WGPU residency for the latent KV state, but it does **not** yet claim complete Phase 13 parity on device:

- HOT/WARM/COLD tier migration is not device-resident yet;
- sparse residual payloads and their Int8/Int4 formats remain a CPU Phase 11 concern;
- Transformer Q/K/V/O linear projections are still outside this cache abstraction;
- CUDA persistent KV residency is not implemented by this phase;
- the baseline single-invocation kernels prioritise reproducible validation over throughput.

Those items must remain explicit follow-up work rather than being inferred from Phase 14.

## Validation

The Phase 14 gate requires:

- workspace rustfmt;
- strict Clippy for `scirust-gpu` with WGPU enabled;
- Rust 1.89 compilation;
- execution on Mesa lavapipe;
- differential lower-rank attention against a CPU oracle;
- sliding-window wrap validation;
- proof that the reported persistent device allocation does not grow as the ring wraps.
