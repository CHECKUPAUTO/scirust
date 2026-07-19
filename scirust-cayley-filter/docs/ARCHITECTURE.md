# CayleyFilter architecture

## Responsibilities

- `scirust-simd` provides existing hypercomplex and SIMD primitives.
- `scirust-cayley-filter` provides operator construction, kernel analysis,
  filtering experiments, diagnostics, and falsification.
- `scirust-signal` will host integration benchmarks against established
  denoising methods.

## Validation rule

The scalar `f64` implementation is the initial numerical oracle.

An optimized implementation is accepted only after demonstrating bounded
numerical error against this oracle on deterministic test vectors.
