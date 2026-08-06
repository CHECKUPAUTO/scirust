# Phase 2 specification

## Objective

Convert dense key/value samples into deterministic fixed-rank latent bases and
coefficients, quantify reconstruction loss, and select the smallest nested rank
that satisfies an explicit relative-RMS target.

Phase 2 remains an isolated experiment. It does not modify production SciRust
crates and does not claim model-level quality preservation.

## Algorithm

For a dense sample matrix with vectors \(x_t \in \mathbb{R}^d\):

1. construct a maximum-rank orthonormal basis with residual-pivoted modified
   Gram-Schmidt;
2. select the lowest-index sample with maximum residual energy at every step;
3. canonicalize each selected basis-vector sign using its largest-magnitude
   component;
4. evaluate every nested prefix \(U_{1:r}\);
5. select the smallest rank whose reconstruction relative RMS is below the
   declared target;
6. project dense keys and values independently;
7. compare dense attention with reconstruction-free latent attention.

Projection and reconstruction are:

\[
c = U^\top x,\qquad \hat{x} = Uc
\]

Relative RMS is:

\[
\varepsilon_{\mathrm{rel}} =
\sqrt{\frac{\sum_i (\hat{x}_i-x_i)^2}{\sum_i x_i^2}}
\]

## Invariants

1. Key and value bases are learned and selected independently.
2. Basis matrices remain row-major with shape `[dimension, rank]`.
3. Basis prefixes are nested; rank selection never rebuilds a different basis.
4. Maximum-residual ties choose the lowest sample index.
5. Basis-vector signs are canonicalized deterministically.
6. Projection and reconstruction accept caller-owned output buffers.
7. Rank targets must be finite and non-negative.
8. No `unsafe` code or external dependency is introduced.
9. Repeated execution is bit-deterministic on the same target and build.
10. The CSV harness is stable and newline terminated.

## Acceptance gates

- `cargo +nightly-2026-07-02 fmt --all -- --check`
- `cargo +1.89.0 clippy --all-targets -- -D warnings`
- debug and release tests pass serially;
- doctests pass;
- exact low-rank datasets recover the intrinsic rank;
- exact low-rank projected attention remains within `2e-5` of dense attention;
- two release harness executions are byte-identical;
- the standard suite emits one header and 12 scenario rows;
- all modified files remain under `experiments/elastic-latent-kv/`.

## Deliberate non-goals

Phase 2 does not implement:

- per-token adaptive ranks;
- product or scalar quantization;
- residual/outlier side channels;
- online basis updates;
- HOT/WARM/COLD tier transitions;
- GPU or SIMD kernels;
- production cache integration;
- model-level perplexity or generation-trajectory validation.
