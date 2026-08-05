# SciRust Elastic Latent KV

This directory contains the isolated experimental implementation of an
elastic, latent and quantized KV-cache architecture for SciRust.

It is deliberately maintained as an independent Cargo workspace until its
mathematical correctness, trajectory-level safety and performance have been
established.

## Phase 0

Phase 0 establishes deterministic reference oracles:

- a dense KV attention oracle;
- a fixed-rank latent KV representation;
- explicit latent reconstruction;
- reconstruction-free latent attention;
- preallocated scratch storage;
- differential tests between the implementations;
- a seeded deterministic sweep over dimensions, ranks, sequence lengths and amplitudes;
- no `unsafe` code.

For a latent key basis \(U_K\), latent value basis \(U_V\), key coefficients
\(c_t^K\) and value coefficients \(c_t^V\):

\[
K_t = U_K c_t^K,\qquad V_t = U_V c_t^V
\]

The reconstruction-free score is:

\[
q^\top K_t = (U_K^\top q)^\top c_t^K
\]

The reconstruction-free value aggregation is:

\[
\sum_t p_t V_t = U_V\left(\sum_t p_t c_t^V\right)
\]

The implementation must match the explicit reconstruction oracle within a
strict floating-point tolerance.

## Non-goals of Phase 0

Phase 0 does not yet implement:

- learned MHA-to-MLA conversion;
- adaptive token rank;
- product quantization;
- outlier residual channels;
- HOT/WARM/COLD transitions;
- GPU kernels;
- integration into `scirust-core`;
- claims of model-level quality preservation.

Those stages require the Phase 0 oracle as their differential reference.
