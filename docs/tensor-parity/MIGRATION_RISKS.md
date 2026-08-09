# Migration Risks

Risks that a parity program PR would face, ordered by expected impact, each
with mitigation. Audit date 2026-08-04.

## R1 — Error-surface divergence in the prototype crates

The `scirust-tensor-*` crates return `Result<_, String>`. A parity claim
requires structured errors (`SciRustError`), but changing the prototype API is
a breaking change for `scirust-tensor-examples` and the fuzz targets
(`tensor_nd_ops_from_bytes.rs` parses bytes → String errors today).

- Mitigation: leave prototype crates untouched in parity PRs; parity surface
  is core `TensorND` + autodiff stacks. Register prototype ops as
  `experimental` with `errors = "string"` and no parity claim. Do the
  prototype-crate error migration as a separate, explicitly-labeled breaking
  PR if ever.

## R2 — Two autodiff stacks, two truth definitions

Ops added for parity must be added to both 2D and ND stacks (2D feeds
optimizers; ND is the N-D surface) or the registry row must say which stack
satisfies it. Duplication is currently *mapped* but not *merged*; a parity
test that passes on one stack gives no information about the other.

- Mitigation: registry rows carry `impl`; CI runs the same differential
  fixtures per `impl`. Do not merge stacks during the parity program (merge
  is a separate program with its own risk register).

## R3 — No dtype abstraction

Every parity row must record its dtype(s). Adding a `DType` to core `TensorND`
is invasive (storage is `Arc<[f32]>`; f64 paths exist only in `Dual`/sparse/
simd kernels).

- Mitigation: Profile 1.0 covers f32 (primary) and f64 (where paths exist);
  bf16 storage-only. No dtype layer in v1. Registry rows declare dtype
  explicitly; rows without an implementation dtype are `missing`.

## R4 — Fixture/oracle dependency on Python

CI cannot run Python. Differential fixtures must be generated offline
(against the frozen baseline), committed, and consumed by Rust-only tests.

- Mitigation: fixture generation script lives in `docs/tensor-parity/
  provenance/` (not run in CI); committed fixtures carry the baseline commit
  and a hash. Evidence from the independent reference kernels is recorded as
  `reference_verified`; production `verified` evidence is reserved for tests
  that call the claimed production implementation directly.

## R5 — Determinism vs PyTorch non-determinism

PyTorch's CPU reduction order is not guaranteed stable; SciRust promises
determinism (`checked` arithmetic, pinned thread counts). A differential test
that compares against a *single* PyTorch run can flake on summation order.

- Mitigation: fixtures store the PyTorch result as a *range* only where
  reduction order matters, or the row records
  `deterministic = "stricter-than-torch"` and the test compares with the
  tolerance widened accordingly. Multi-threaded CPU tests pin thread count.

## R6 — GPU paths are unverified against CPU semantics

GPU/CUDA kernels duplicate CPU semantics with no cross-device tests today.

- Mitigation: no device row is `parity` in v1; device rows are
  `experimental`. Cross-device differential tests are added only after CPU
  parity exists for the row (cheap: reuse fixtures, upload to device).

## R7 — Fuzz targets assume the old surfaces

`fuzz/fuzz_targets/` exercises prototype + core paths. Parity changes must
not break them (they gate CI).

- Mitigation: parity PRs do not alter prototype signatures (R1) and add
  fuzz targets for new structured-error paths *in the same PR*.

## R8 — `String` panics in autodiff kernels

Some 2D/ND kernels panic on bad shapes (asserts, index) instead of returning
errors. A parity error-path test would hit a panic, not a `SciRustError`.

- Mitigation: error-path parity rows are only claimed for ops whose
  error behavior is structured; a sweep PR converts kernel asserts to
  `SciRustError` returns **per op family**, each with its own tests, before
  that family's error-path rows are claimed. Panics become `internal` rows in
  the registry until converted.

## R9 — Registry/metadata drift

`tensor-operators.toml` is hand-edited and will drift from code.

- Mitigation: the coverage matrix is generated. `impls` is inventory only;
  production `parity` requires direct `verified` evidence, while reference
  evidence is stored separately in `reference_verified`. CI validates the
  committed fixture manifest and its hashes. A per-stack symbol/direct-call
  gate remains required before production parity can be promoted
  automatically.

## R10 — Scope creep toward torch.compile / in-place / out=

The program's guardrail is that parity claims cover the documented profile
only; `torch.compile`, `out=` variants and in-place forms beyond what exists
are out of scope (EXCLUSIONS.md) and attract review rejection.

- Mitigation: PR template for the parity program includes a checklist that
  forces the author to name the registry rows touched; the reviewer rejects
  rows outside Profile 1.0 unless the profile change PR landed first.
