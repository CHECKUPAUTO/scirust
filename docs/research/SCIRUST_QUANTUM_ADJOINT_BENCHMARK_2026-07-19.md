# SciRust Exact Dense Adjoint-Jacobian Benchmark

**Date:** 2026-07-19  
**Base commit:** `dc193bcea91b528d587046b0a6c1eb056985cbcd`  
**Platform:** NVIDIA Jetson AGX Thor, AArch64, 14 CPU cores, 122 GiB RAM  
**Rust:** `rustc 1.98.0-nightly (4c9d2bfe4 2026-07-01)`

## Objective

Compare SciRust's exact dense adjoint Jacobian against exact parameter-shift
differentiation on identical deterministic quantum workloads.

## Protocol

- fixed seed: `6327026190105`;
- local SplitMix64 workload generator;
- symbolic `Rx`, `Ry`, and `Rz`;
- fixed `PhaseShift`, `H`, `S`, `T`, `CNOT`, and `CZ`;
- deterministic two-qubit Pauli observables;
- three warm-up runs;
- alternating timing order;
- Tukey outlier rejection;
- two-sided Welch unequal-variance test;
- sixteen configurations and 112 benchmark-schema records;
- controlled qubit, parameter, and observable sweeps validated as nested.

## Principal results

- adjoint faster in **16/16 configurations**;
- maximum median speedup: **14.783×**;
- maximum absolute gradient error: **3.576278687e-7**;
- all errors below the predefined `1e-4` threshold;
- all sixteen Welch comparisons significant at `p < 0.05`;
- numerical errors and Jacobian checksums reproduced exactly across two runs.

## Controlled scaling

| Sweep | Minimum speedup | Mean speedup | Maximum speedup |
|---|---:|---:|---:|
| Combined | 2.848× | 4.460× | 5.937× |
| Qubits | 3.900× | 4.017× | 4.275× |
| Parameters | 2.032× | 7.074× | 14.783× |
| Observables | 3.184× | 6.508× | 10.197× |

## Interpretation

The parameter sweep provides the clearest evidence for adjoint differentiation:
the speedup rises from approximately `2×` at four parameters to almost `15×`
at thirty-two parameters.

The qubit sweep keeps the speedup near `4×`. Both algorithms use the same dense
state-vector representation and therefore share its exponential dependence on
qubit count.

The observable sweep reduces the relative gain as observables increase because
the current adjoint implementation performs reverse propagation for each
observable.

## Reproducibility

The benchmark seed genuinely generates parameter values and fixed phase angles.
Every controlled family uses a fixed stream so smaller workloads are prefixes
of larger workloads.

Wall-clock timings are not bit-for-bit deterministic. Mathematical invariants
are deterministic and produced the same SHA-256 digest in both final runs:

`ba2dd6164ddcacde65e0532f8fa26f3a47a11eecbaeb6832d156f84159da8f36`

Raw data:

- [`SCIRUST_QUANTUM_ADJOINT_BENCHMARK_2026-07-19.jsonl`](data/SCIRUST_QUANTUM_ADJOINT_BENCHMARK_2026-07-19.jsonl)
- [`SCIRUST_QUANTUM_ADJOINT_INVARIANTS_2026-07-19.tsv`](data/SCIRUST_QUANTUM_ADJOINT_INVARIANTS_2026-07-19.tsv)

## Reproduction

```bash
cargo run -p scirust-core --release --example quantum_gradient_benchmark
```

## Scope

These results concern SciRust's dense CPU state-vector implementation on one
Jetson AGX Thor host, for deterministic circuits of four to ten qubits. They do
not establish performance for GPU, sparse, distributed, noisy, or hardware
quantum backends.
