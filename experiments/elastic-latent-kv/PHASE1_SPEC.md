# Phase 1 specification

## Objective

Measure the fixed-rank latent runtime established in Phase 0 before any
adaptive rank, quantization, residual channel or hardware-specific kernel is
introduced.

Phase 1 remains an isolated experiment. It does not modify production SciRust
crates and does not claim model-level quality preservation.

## Deliverables

1. Exact deterministic byte accounting for dense and latent cache payloads.
2. Exact deterministic byte accounting for fixed bases and shared scratch.
3. Closed-form multiply-accumulate estimates for dense, explicit latent and
   reconstruction-free attention.
4. Differential error metrics over dense, explicit reconstruction and
   reconstruction-free outputs.
5. A stable output fingerprint for repeated execution on the same target and
   build.
6. A deterministic 24-scenario baseline suite.
7. Stable newline-terminated CSV export with no external dependency.

## Invariants

1. Phase 0 cache implementations remain the runtime under test.
2. Dense keys and values are reconstructed from the same latent bases and
   coefficients used by the latent cache.
3. Scenario generation is seeded and deterministic.
4. Every scenario validates capacity, dimensions, ranks, query count and
   amplitude before allocation.
5. Byte and operation-count arithmetic is checked for overflow.
6. Keys and values keep separate bases and ranks.
7. No `unsafe` code or FFI is introduced.
8. The measurement harness does not allocate inside the Phase 0 attention hot
   path; scratch and output vectors are allocated before each query loop.
9. CSV column order and numeric formatting are stable.

## Standard suite

The baseline suite covers:

- head dimensions: 8, 16, 32 and 64;
- key/value rank pairs: (2, 3), (4, 2) and (8, 6);
- token counts: 4 and 16;
- three deterministic amplitudes;
- three queries per scenario;
- 24 total scenarios.

## Acceptance gates

- `cargo +nightly-2026-07-02 fmt --all -- --check`
- `cargo +1.89.0 clippy --all-targets -- -D warnings`
- `cargo +1.89.0 test --all-targets`
- `cargo +1.89.0 test --all-targets --release`
- `cargo +1.89.0 run --release --bin phase1_harness > phase1.csv`
- repeated harness execution produces byte-identical CSV on the same target and
  build;
- all 24 scenarios complete;
- reconstruction-free output remains within the Phase 1 tolerance against both
  dense and explicit latent oracles.

## Deliberate non-goals

Phase 1 does not yet implement:

- adaptive per-token rank;
- learned bases;
- product quantization;
- outlier or residual channels;
- HOT/WARM/COLD cache tiers;
- GPU or SIMD kernels;
- wall-clock benchmarking claims;
- model conversion;
- model-level or trajectory-level acceptance thresholds.
