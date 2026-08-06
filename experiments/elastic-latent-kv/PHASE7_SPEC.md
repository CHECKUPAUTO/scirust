# Elastic Latent KV — Phase 7 Runtime Bridge

## Status

Phase 7 is the first production-facing bridge from the deterministic research
stack into `scirust-core`'s live numeric Transformer decode path.

It does not replace the existing dense, paged, or elastic backends. It adds a
fourth `AttentionBackend` implementation that can be selected independently
for each attention head.

## Objective

Evaluate attention directly from quantized latent coefficients:

\[
s_t = \frac{(U_K^T q)^T \widehat{c_t^K}}{\sqrt{d}},
\qquad
z = U_V \sum_t p_t \widehat{c_t^V}.
\]

The implementation must never reconstruct a dense key. It must accumulate
values in latent space and perform exactly one dense value up-projection per
query.

## Runtime representation

Each head owns:

- a fixed token capacity;
- a dense key basis `U_K` with shape `[d_head, r_key]`;
- a dense value basis `U_V` with shape `[d_head, r_value]`;
- independently selected coefficient formats for keys and values;
- a preallocated coefficient payload;
- one row scale per token for INT8 or INT4;
- fixed projection scratch used by append;
- fixed attention scratch used by the backend adapter.

Supported coefficient formats are:

- FP32, encoded as little-endian bytes;
- row-wise symmetric INT8 in `[-127, 127]`;
- packed row-wise symmetric INT4 in `[-7, 7]`.

## Allocation contract

`QuantizedLatentKvCache::new` allocates all persistent and append scratch
storage. `LatentAttentionScratch::new` allocates query-time scratch.

After construction:

- `append` must not grow any buffer;
- `attention_into` must not allocate or grow any buffer;
- capacity exhaustion must return a typed error;
- the `AttentionBackend` adapter may allocate only the owned dense output
  required by the existing trait signature.

## Determinism contract

The following operations use a fixed scalar order:

- dense-to-latent projection;
- quantization and nibble packing;
- query projection;
- dequantized coefficient dot products;
- stable softmax;
- latent value accumulation;
- final value up-projection.

Repeated executions with identical inputs must produce identical output bits.

## Validation

The Phase 7 test suite covers:

- full-rank FP32 agreement with contiguous attention;
- INT8 and INT4 output error bounds;
- fixed allocation across every append;
- strict capacity failure;
- deterministic repeated execution;
- invalid shape and non-finite input rejection;
- direct integration with `kv_backend::decode_step`;
- memory reduction at full cache capacity;
- a deterministic CSV harness across FP32, INT8, INT4, and reduced-rank INT4.

## Explicit limitations

Phase 7 deliberately does not yet implement:

- online basis learning or basis replacement;
- sparse residual channels in the production backend;
- per-token adaptive rank;
- token eviction or HOT/WARM/COLD transitions;
- basis sharing across attention heads;
- SIMD, WGPU, or CUDA kernels for latent dot products;
- changes to the legacy autodiff `infer_step` path.

Those are follow-on phases. Phase 7 establishes the safe scalar runtime and the
public backend contract they can optimize against.
