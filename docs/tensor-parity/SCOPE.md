# SciRust Tensor Parity Scope (Profile 1.0)

This document defines the *SciRust Tensor Parity Profile 1.0*: the exact
boundary of what "parity with PyTorch" means for this workspace, what is in
scope, what is out of scope, and the acceptance criteria attached to each
claim.

## Goal

Achieve **semantic parity** — not bitwise equality — between SciRust tensor
operations and the frozen PyTorch baseline
(`docs/tensor-parity/PYTORCH_BASELINE.md`, v2.13.0 / `cf30153c4c131c8164ee7798e5022d810682e2cb`)
on a documented profile of dtypes, layouts and devices. Then go **beyond**
PyTorch where the profile permits (deterministic reduction order, checked
arithmetic, structured errors, reproducible autograd).

## Definitions

- **Semantic parity**: for every pair of finite inputs in the domain defined by
  the operator's profile row, the SciRust output is within the operator's
  tolerance (absolute+relative) of the PyTorch output, and raises the
  structured equivalent of the PyTorch error class on the same shape/dtype
  error conditions.
- **Operator**: a named tensor kernel plus its autograd rule, documented in
  `tensor-operators.toml`.
- **Profile row**: `(operator, dtypes, layout, device, tolerance, autograd)`.

## Profile 1.0 — in scope

### Dtypes
- `f32` (primary), `f64` (where a SciRust path exists), `bf16` (storage +
  promotion behavior as documented), `i64` indices only (no integer
  arithmetic kernels in v1).
- Promotion: documented per operator; follows PyTorch promotion rules where
  the row says `promotion = "torch"`, otherwise fixed and listed.

### Layout
- Contiguous row-major storage; strided views (`view`/`reshape` semantics
  including non-contiguous strides) where the operator row says
  `layout = "strided"`.
- `permute`, `transpose`, `slice_axis`, `broadcast_to` are profile members.

### Device
- CPU single-threaded **and** multi-threaded (thread count pinned in tests).
- GPU (wgpu/CUDA) paths are tracked in the matrix but **not** required for a
  v1 parity claim; they are `experimental` until a differential run exists.

### Autograd
- Reverse-mode autodiff must match PyTorch gradients on all differentiable
  profile rows within the same tolerance (gradcheck-style, but with the
  frozen baseline's definition of "correct").
- Double-backward is out of scope for v1.

### Numerics
- Deterministic accumulation order on CPU for reductions where the row says
  `deterministic = true` (checked arithmetic: `checked_add`, `checked_mul`,
  `checked_sub` on shape/index arithmetic).
- No NaN-boxing, no FP contraction changes that break the documented
  tolerance: rows carry `tolerance = { atol, rtol }` and reference the test
  that enforces it.

### Errors
- All public API errors are structured (`SciRustError`), never `String`
  panics; each parity row maps PyTorch error classes
  (`RuntimeError: shape mismatch`, `IndexError`, `ValueError`) to SciRust
  error variants. `scirust-core/src/error.rs` is the single error source.

## Out of scope for Profile 1.0 (see EXCLUSIONS.md)

- Complex dtypes, quantized dtypes, integer arithmetic kernels.
- Sparse tensor *autograd* and sparse-sparse kernels (sparse storage exists,
  parity for sparse ops is a later profile).
- `torch.compile` graph export / AOTAutograd compatibility.
- Distributed / NCCL / gloo behaviors.
- RNG bit-exactness (seed/algorithm documented, output stream not tied to
  torch's MT19937 stream).
- Double backward, higher-order derivatives.
- In-place/`out=` variants: in-place forms of existing ops are in scope where
  already present (see registry), `out=` forms are deferred.

## Acceptance criteria (per claim)

`impls` records implementations that exist; it is not evidence by itself.
`verified_impls` records the exact production implementations exercised by a
direct differential proof. Implementations in `impls` but not in
`verified_impls` remain experimental. An operator row remains `experimental`
while only a subset of its implementation cells is verified; row-level
`parity` means every implementation listed in `impls` is directly verified.

`reference_impl` identifies the independent reference implementation used to
exercise the frozen witness, and `reference_verified` records that reference
proof's harness, fixtures, and date. These fields establish **reference parity**
only: they MUST NOT populate `verified_impls` or upgrade a production
implementation (`2d`, `nd`, `core`, `simd`, `gpu`, `cuda`, `sparse`) to
production `parity`.

A production coverage cell is only marked `parity` when, in the CI harness:

1. a differential test runs against a PyTorch 2.13.0 witness for that row
   (generated offline, committed as a fixture — CI does not run Python),
2. the numeric tolerance holds for the fixture set,
3. the autograd rule passes gradcheck against the same fixture set,
4. the error-path tests pass (shape/dtype/device misuse).

The coverage matrix (`artifacts/tensor-coverage.csv`) is **generated**, never
hand-edited; see `TOOLS.md` in this directory.
