# Elastic Latent KV — Phase 13 legacy incremental integration

## Status

This follow-up closes the legacy incremental-Transformer gap left after the initial Phase 13 runtime merge.

The original Phase 13 runtime already executes bounded numeric attention through `ElasticLatentDecodeRuntime`, but the historical `TransformerEncoder::infer_step` chain still reaches `MultiHeadAttention::infer_step`, whose internal `kv_cache` is dense. This integration adds an explicit opt-in path for callers that need the legacy Transformer block/encoder pipeline with Elastic Latent KV storage.

## API

`scirust_core::nn` now exports:

- `ElasticLatentLayerConfig` — one Phase 13 runtime configuration and one per-head calibration set for a Transformer layer;
- `ElasticLatentEncoderSession` — one bounded `ElasticLatentDecodeRuntime` per encoder block;
- `ElasticLatentInferStep` — method-style extension exposing `TransformerEncoder::infer_step_elastic`;
- `ElasticLatentTransformerError` — structured topology, shape, position and per-layer runtime failures.

The dense `TransformerEncoder::infer_step` API is intentionally unchanged for backwards compatibility.

## Decode semantics

For each incremental token the Elastic path performs, per encoder block:

1. pre-attention LayerNorm through the existing Transformer parameters;
2. Q/K/V projection, Elastic Latent KV append and attention through that layer's `ElasticLatentDecodeRuntime`;
3. attention output projection through the same runtime;
4. residual addition;
5. second LayerNorm;
6. the existing two-layer FFN and residual;
7. the encoder final LayerNorm after all blocks.

The session rejects a token before mutating KV state when:

- the encoder layer count no longer matches the frozen session topology;
- the encoder model width changed;
- the input is not exactly one token `(1, d_model)`;
- `pos` is skipped, repeated or otherwise differs from the admitted-token count of any layer.

## Dense-cache isolation

The Elastic path never writes `MultiHeadAttention::kv_cache`. Each attention layer stores K/V only inside its preallocated Phase 13 runtime backend. Unit tests assert that the legacy dense caches remain `None` after Elastic incremental decoding.

## Numerical differential oracle

A deterministic two-layer causal encoder test constructs full-rank F32 Phase 13 runtimes and compares every incremental output against the existing dense `TransformerEncoder::infer_step` oracle. The accepted tolerance covers only floating-point evaluation-order differences between the tape matmul path and the scalar Phase 12 kernel.

## Allocation and autograd boundary

This change removes legacy dense KV growth from the opted-in incremental attention path. It does **not** claim that the surrounding legacy `Tape`, LayerNorm, residual or FFN execution is allocation-free.

The bridge is inference-only: the numeric attention result is inserted back into the tape as an input. Gradients therefore do not propagate through the Elastic Latent attention step. Training and differentiable forward passes continue to use the existing `forward_3d`/legacy paths.

## Validation

The dedicated read-only CI gate runs:

```bash
cargo +nightly-2026-07-02 fmt --all -- --check
cargo +nightly-2026-07-02 clippy -p scirust-core --all-targets --locked -- -D warnings
cargo +1.89.0 check -p scirust-core --all-targets --locked
cargo +1.89.0 test -p scirust-core elastic_encoder_ --locked
cargo +1.89.0 run --quiet --locked -p scirust-core --example latent_kv_phase13
```

This integration does not yet turn the legacy tape-based Transformer into a device-resident WGPU/CUDA inference engine. Phase 12 kernels remain available to the per-layer runtime according to the selected `LatentKernelKind`.
