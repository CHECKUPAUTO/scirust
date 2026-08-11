# SciRust rustc AutoDiff / Enzyme probe

This directory is an **experimental, standalone Cargo workspace**. It is not a
member of the root SciRust workspace and therefore cannot change the framework's
Rust 1.89 MSRV or normal stable CI graph.

Its purpose is narrow: compile one derivative with rustc's experimental AutoDiff
support and compare the result against `scirust-autodiff` as an independent
correctness/performance oracle.

## Requirements

Rust's AutoDiff support is nightly-only and requires a compiler/toolchain built
with the AutoDiff/Enzyme backend available. The local Cargo configuration passes
`-Zautodiff=Enable`, as required by rustc's unstable AutoDiff workflow.

Run from this directory with a compatible nightly toolchain:

```bash
cargo +nightly test
```

If the installed nightly reports that AutoDiff/Enzyme is unavailable, that is a
toolchain capability failure, not a failure of SciRust's production AutoDiff.
The production `scirust-autodiff` crate remains independent of this experiment.

## Current probe

The test differentiates the Rosenbrock function with respect to `x` using:

1. rustc AutoDiff / Enzyme forward mode;
2. SciRust's native forward-mode `Dual` implementation;
3. a known analytic derivative at `(x, y) = (3, 1)`.

The probe is intentionally a comparison harness, not a production backend.
