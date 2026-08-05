# Phase 0 specification

## Objective

Create a minimal deterministic reference implementation against which every
future compressed or hardware-specific implementation can be checked.

## Invariants

1. Keys and values use separate bases and may use separate ranks.
2. Basis matrices are row-major with shape `[head_dimension, rank]`.
3. Token coefficients are row-major with shape `[token_count, rank]`.
4. Attention uses stable softmax by subtracting the maximum score.
5. All hot computation receives preallocated scratch storage.
6. No `unsafe` code is permitted.
7. The cache cannot grow beyond its declared token capacity.
8. Reconstruction-free attention must numerically agree with explicit
   reconstruction.
9. Repeated execution over identical inputs must be bit-deterministic on the
   same target and build.

## Phase 0 acceptance gates

- `cargo fmt --check`
- all unit tests pass;
- identity-basis latent attention agrees with dense attention;
- arbitrary fixed-rank latent attention agrees with explicit reconstruction;
- seeded differential sweeps cover multiple dimensions, ranks, sequence lengths and amplitudes;
- capacity and shape violations fail explicitly;
- no modification is made to production SciRust crates.
