# scirust-cayley-filter

Experimental SciRust crate for investigating denoising operators derived from
Cayley–Dickson multiplication.

## Initial research question

For a fixed sedenion `a`, left multiplication defines a real-linear operator:

```text
L_a(x) = a · x```

When `a` is a zero divisor, `L_a` may have a non-trivial kernel. The project
tests whether an identified noise subspace can be mapped into or near this
kernel while preserving the useful signal.

## Scientific position

CayleyFilter does not assume superiority over established denoising methods.

Every claim must be supported by reproducible comparisons against suitable
baselines, including:

- Wiener filtering;
- LMS and RLS;
- singular-value decomposition;
- Kalman filtering;
- STFT-based denoising;
- spectral subtraction.

## Development order

1. Scalar `f64` mathematical oracle.
2. Exact Cayley-Dickson multiplication.
3. Construction of the real-linear operator `L_a`.
4. Kernel, rank, and singular-value diagnostics.
5. Synthetic falsification experiments.
6. Comparison with established denoising baselines.
7. Scalar `f32` numerical parity.
8. AArch64 and x86_64 SIMD kernels.

## Initial status

The crate currently contains only the deterministic scalar foundation.
