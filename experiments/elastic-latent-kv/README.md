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

## Phase 1

Phase 1 measures the fixed-rank runtime before any lossy mechanism is added:

- exact dense and latent byte accounting;
- basis and scratch accounting;
- dense, explicit-latent and reconstruction-free operation estimates;
- aggregate absolute, RMS and relative error metrics;
- deterministic output fingerprints;
- a deterministic 24-scenario baseline suite;
- stable CSV export through `phase1_harness`;
- no external dependency and no `unsafe` code.

Run the baseline harness with:

```bash
cargo +1.89.0 run --release --bin phase1_harness > phase1.csv
```

Phase 1 still does not introduce adaptive ranks, quantization, residual
channels, cache tiers or production integration.

## Phase 2

Phase 2 adds deterministic dense-to-latent conversion and fixed-rank selection:

- residual-pivoted modified Gram-Schmidt basis construction;
- deterministic tie-breaking and sign canonicalization;
- caller-buffer projection and reconstruction;
- key/value reconstruction metrics;
- smallest nested rank satisfying a relative-RMS target;
- dense-versus-projected reconstruction-free attention measurement;
- a stable 12-scenario CSV harness;
- no external dependency and no `unsafe` code.

Phase 2 still uses one fixed key rank and one fixed value rank per scenario. It
does not yet introduce per-token elasticity, quantization, residual channels or
production integration.

## Phase 3

Phase 3 adds deterministic strict-budget planning over separate key/value rank
pairs:

- exact persistent coefficient and basis byte accounting;
- exhaustive nested rank-pair evaluation;
- independent key and value ranks;
- reconstruction and attention quality guards;
- explicit quality failure when no pair meets every target;
- deterministic lexicographic selection;
- non-dominated Pareto frontier construction;
- a stable 12-scenario CSV harness;
- no quantization, per-token rank or production integration.

The persistent representation uses:

\[
B_{latent}=4(T+D)(R_K+R_V)
\]

while dense keys and values use:

\[
B_{dense}=8TD.
\]

Transient attention scratch is reported separately by Phase 1 and is not
charged to the persistent-cache budget in Phase 3.
