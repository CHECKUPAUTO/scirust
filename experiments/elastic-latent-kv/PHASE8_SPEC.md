# Elastic Latent KV — Phase 8 Sparse Residual Runtime

## Status

Phase 8 ports the deterministic sparse residual channel from the isolated
research harness into a production-facing `scirust-core` attention backend.
It is additive: the validated Phase 7 cache remains unchanged.

The implementation was merged by PR #946. A dedicated read-only workflow now
runs on `master` and on pull requests that touch the Phase 8 runtime,
specification, harness, or workflow. This status section is intentionally kept
inside the workflow path filter so post-merge validation can be evidenced by a
normal pull-request run rather than by the historical red checks from the
pre-formatting head of PR #946.

## Objective

For a dense key or value vector `x`, a caller-supplied orthonormal basis `U`,
latent coefficients `c = U^T x`, and a fixed-slot sparse residual `r`, Phase 8
stores

\[
x \approx Uc + r.
\]

Key attention scores are evaluated without dense reconstruction:

\[
s_t = \frac{(U_K^T q)^T \widehat{c_t^K} + q^T \widehat{r_t^K}}
{\sqrt{d}}.
\]

Values are accumulated through one latent path and one sparse path:

\[
z = U_V \sum_t p_t \widehat{c_t^V}
  + \sum_t p_t \widehat{r_t^V}.
\]

Dense keys are never reconstructed during attention. The value basis is
up-projected exactly once per query; sparse value corrections are applied as a
fixed-slot scatter afterwards.

## Residual selection

During append:

1. project the dense vector into the configured latent basis;
2. reconstruct the latent approximation in preallocated append scratch;
3. compute residual coordinates implicitly as `dense[i] - reconstruction[i]`;
4. select the largest absolute residual coordinate for each reserved slot;
5. break equal-magnitude ties by the lowest dense coordinate index;
6. encode the selected coordinate as `u16`;
7. encode selected residual values in the configured value format.

Unused slots carry the sentinel index `u16::MAX` and a zero value.

## Supported storage formats

Latent coefficients and sparse residual values are configured independently for
keys and values. Both support:

- FP32 encoded as little-endian bytes;
- row-wise symmetric INT8 in `[-127, 127]`;
- packed row-wise symmetric INT4 in `[-7, 7]`.

Each quantized row owns one FP32 scale. Residual indices always use `u16`.

## Allocation contract

`ResidualQuantizedLatentKvCache::new` allocates:

- latent key/value bases;
- fixed-capacity coefficient payloads and scales;
- fixed-capacity residual index payloads;
- fixed-capacity residual value payloads and scales;
- dense reconstruction scratch used only during append;
- latent projection scratch;
- residual selection values and coordinate markers.

`ResidualLatentAttentionScratch::new` allocates query-time score and latent
accumulator buffers.

After construction:

- `append` must not grow storage;
- `attention_into` must not allocate or grow storage;
- capacity exhaustion must return a typed error;
- invalid shapes, non-finite values, oversized residual slot counts, and
  insufficient scratch must return typed errors;
- the `AttentionBackend` adapter may allocate only the owned dense output
  required by the existing trait signature.

## Determinism contract

The implementation uses a fixed scalar order for:

- latent projection;
- latent reconstruction used during append;
- residual coordinate selection;
- quantization and nibble packing;
- query projection;
- latent and sparse key score accumulation;
- stable softmax numerator generation;
- latent value accumulation;
- one dense up-projection;
- sparse value scatter.

Repeated execution with identical inputs must produce identical output bits.

## Runtime integration

`ResidualLatentQuantizedBackend` implements `kv_backend::AttentionBackend` and
can be selected independently for each attention head by the live numeric
`decode_step` path.

The backend exposes the underlying cache for deterministic telemetry including:

- resident token count and fixed capacity;
- latent ranks and coefficient formats;
- key/value residual configurations;
- per-token selected residual indices;
- logical used bytes;
- actual fixed allocated bytes.

## Validation

The Phase 8 suite covers:

- full-rank FP32 agreement with contiguous dense attention;
- exact restoration of a structured reduced-rank tail with two FP32 residual
  slots;
- bounded INT8 and INT4 residual error against the FP32 residual oracle;
- deterministic equal-magnitude tie-breaking;
- fixed allocation across appends;
- strict residual slot validation;
- bit-identical repeated attention;
- direct `decode_step` integration;
- fixed-allocation reduction against a dense cache;
- a deterministic CSV harness comparing zero residuals with FP32, INT8 and
  INT4 residual channels.

## Explicit limitations

Phase 8 deliberately does not yet implement:

- online basis learning or basis replacement;
- per-token adaptive rank;
- adaptive residual slot counts;
- token eviction or HOT/WARM/COLD transitions;
- basis sharing across heads;
- SIMD, WGPU, or CUDA sparse-residual kernels;
- changes to the legacy autodiff `infer_step` path.

Those remain follow-on phases. Phase 8 establishes the safe scalar differential
contract for adaptive policies and accelerated kernels.
