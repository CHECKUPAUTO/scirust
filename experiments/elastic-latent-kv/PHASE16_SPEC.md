# Elastic Latent KV — Phase 16: WGPU-resident Transformer block

Phase 16 extends the Phase 15 resident latent MHA boundary across one complete legacy pre-LN `TransformerBlock` during incremental inference.

The implementation is opt-in and inference-only. Existing CPU/tape paths are unchanged.

## Per-token path

1. Upload one `d_model` token row.
2. Run pre-attention LayerNorm on WGPU.
3. Run resident Q/K/V projections and latent sliding-window attention.
4. Run output projection and the first residual.
5. Run pre-FFN LayerNorm.
6. Run FFN1 + ReLU, FFN2 and the second residual.
7. Download one final `d_model` block row.

No normalized row, Q/K/V vector, attention context, residual row or FFN activation crosses the host boundary.

## Persistent state

`WgpuResidentTransformerBlock` owns fixed WGPU allocations for LayerNorm parameters, attention and FFN weights, per-head latent bases, K/V latent rings, scratch, and one input/output row. Storage does not grow when the ring wraps.

## Determinism

The baseline kernel uses one invocation and fixed loop order for both LayerNorm reductions, dense projections, latent projection, score accumulation, stable softmax, value aggregation, reconstruction and residual additions. This is a correctness baseline rather than a throughput claim.

## Basis and lifecycle

One block uses a uniform latent rank across heads while each head keeps independent row-major `(d_head, rank)` key/value bases. Full-rank identity bases are validated against `TransformerBlock::infer_step`; lower-rank bases are validated against an independent CPU sliding-window oracle.

`reload_weights` refreshes LayerNorm, attention and FFN parameters without reallocating. `reset` clears logical KV state and token position while preserving allocation. The current implementation requires the legacy LayerNorm epsilon `1e-5`.

## Required validation

- workspace rustfmt;
- strict Clippy with `-D warnings`;
- Rust 1.89.0 build;
- full-rank multi-token parity with the legacy incremental block;
- lower-rank parity after ring wrap;
- position guard before state mutation;
- reset without allocation growth;
- real WGSL execution on Mesa lavapipe.

## Non-goals

This phase does not yet claim a fully device-resident multi-layer `TransformerEncoder`: separate resident blocks still have a host boundary between layers, and final encoder LayerNorm remains outside the primitive. CUDA resident blocks, heterogeneous head ranks, tier quantization and parallel throughput kernels remain follow-up work.
