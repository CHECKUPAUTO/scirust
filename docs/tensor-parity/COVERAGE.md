# Coverage Matrix

`artifacts/tensor-coverage.json` and `artifacts/tensor-coverage.csv` are the
single source of truth for SciRust tensor parity status. They are **generated**
from `tensor-operators.toml` by `tools/tensor-parity` — never hand-edited
(Risk R9, MIGRATION_RISKS.md).

## Semantics of a row

- `status`:
  - `missing` — no SciRust implementation exists for this operator in the
    profile domain.
  - `experimental` — implemented somewhere in the workspace (`impls` lists
    which stacks), but **not** yet verified against the frozen PyTorch
    baseline (no committed fixture, no harness run, or both).
  - `parity` — a Rust-only differential harness ran against committed
    fixtures generated from PyTorch 2.13.0 (`cf30153c...`), the numeric
    tolerances held, and the autograd rule passed gradcheck for the same
    fixtures.
- `impls` — which SciRust stacks satisfy the row (`2d`, `nd`, `tnd`, `core`,
  `simd`, `gpu`, `cuda`, `sparse`). See `DUPLICATION_MAP.md` for why multiple
  impls are allowed and how a row names its satisfying one.
- `dtypes` — dtypes covered by the row (Profile 1.0: f32 primary, f64 where
  the path exists, storage-only bf16).
- `tolerance` — `(atol, rtol)` used by the differential check for the row.
- `autograd` — whether the autograd rule is part of the claim.

## Current numbers

Generated output summary (regenerate with `cargo run` in
`tools/tensor-parity`):

- total rows: 74
- parity: 0
- experimental: 68
- missing: 6 (fft, svd, qr, lstsq, eig, sparse_autograd)

Zero parity rows is the honest and intended state of Phase 0: the matrix is
the *map*, not the territory. Rows move to `parity` only through the harness
pipeline described in `TOOLS.md`.

## Governance

- A PR that claims parity **must** ship: the row edit, the committed fixtures,
  the harness test, and the regenerated matrix — all in the same PR.
- The CI coverage gate (when added) fails when a `parity` row lacks fixtures
  or when `impls` names a symbol that does not exist (compile-checked).
- `missing` rows are not failures; they are the backlog. The program ships
  PRs per family (see `MIGRATION_RISKS.md` R10), each of which moves a family
  from `missing`/`experimental` toward `parity`.
