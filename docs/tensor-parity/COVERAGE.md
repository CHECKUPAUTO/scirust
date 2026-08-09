# Coverage Matrix

`artifacts/tensor-coverage.json` and `artifacts/tensor-coverage.csv` are the
single source of truth for SciRust tensor parity status. They are **generated**
from `tensor-operators.toml` by `tools/tensor-parity` — never hand-edited
(Risk R9, MIGRATION_RISKS.md).

## Semantics of a row

- `status` describes **production implementations named by `impls`**:
  - `missing` — no SciRust production implementation exists for this operator
    in the profile domain.
  - `experimental` — one or more implementations exist, and at least one
    implementation cell is not yet directly verified against the frozen
    PyTorch baseline by this campaign. A row may therefore contain a verified
    subset in `verified_impls`.
  - `parity` — every implementation named by `impls` is called directly by
    committed differential CI evidence and satisfies the row.
- `impls` — implementation inventory (`2d`, `nd`, `tnd`, `core`, `simd`,
  `gpu`, `cuda`, `sparse`). Presence in this list is **not** proof of parity.
- `verified_impls` — exact subset of `impls` exercised directly by the
  production differential proof recorded in `verified`. Other implementations
  on the same row remain experimental. A row reaches `parity` only when
  `verified_impls` covers every implementation in `impls`; partial proof keeps
  the row `experimental`.
- `reference_parity` / `reference_impl` — evidence obtained from the
  independent `scirust_core::tensor::parity` reference kernels. This validates
  the committed PyTorch witnesses and the reference semantics, but does not
  validate any production stack listed in `impls`.
- `dtypes` — dtypes covered by the row (Profile 1.0: f32 primary, f64 where
  the path exists, storage-only bf16).
- `tolerance` — `(atol, rtol)` used by the differential check for the row.
- `autograd` — whether the autograd rule is part of the claim.

## Current numbers

Generated output summary (regenerate with `cargo run` in
`tools/tensor-parity`):

- total rows: 74
- production parity rows: 15
- reference parity: 68
- experimental production rows: 53
- missing: 6 (`fft`, `svd`, `qr`, `lstsq`, `eig`, `sparse_autograd`)
- directly verified production implementation cells: 23 (`2d`)

The 68 reference-parity rows are verified by the Rust-only reference harness
`scirust-core/tests/parity_differential.rs` against committed fixtures
(`tests/parity/fixtures/`, generated offline from the frozen baseline —
see `provenance/generate_fixtures.py`). They carry
`reference_impl = "scirust_core::tensor::parity"` and
`reference_verified = { harness, fixtures, on }`.

Production evidence now exists for 23 `2d` implementation cells through
`scirust-core/tests/production_parity_2d.rs`. Fifteen operator rows have every
implementation listed in `impls` directly verified and therefore reach
row-level `parity`.

Eight rows contain a directly verified `2d` cell while remaining
`experimental`: `exp`, `sqrt`, `sigmoid`, `relu`, `add`, `sub`, `mul`, and
`div`. Their additional ND/SIMD/GPU/CUDA implementations listed in `impls`
remain outside the current production proof.

For `add`, `sub`, `mul`, `div`, and `atan2`, the production harness now consumes
25 committed frozen-baseline witnesses from
`tests/parity/fixtures/elementwise_broadcast/`: five deterministic PyTorch
broadcast cases per operator, each comparing forward output plus the VJP
gradients for both operands (`gx` and `gb`). Because `atan2` inventories only
the `2d` implementation, this closes its row at production `parity`. The four
basic arithmetic rows remain `experimental` because they inventory additional
implementations that have not yet been directly verified.

## Governance

- A PR that claims **production parity** must ship: the row edit, committed
  fixtures, a differential test that calls the claimed production
  implementation directly, and the regenerated matrix — all in the same PR.
- Reference fixtures may establish `reference_parity`, but cannot by themselves
  promote a production implementation from `experimental` to `parity`.
- The coverage tooling rejects a production `parity` row without direct
  `verified` evidence.
- `missing` rows are not failures; they are the backlog. The program ships
  PRs per family (see `MIGRATION_RISKS.md` R10), each of which moves a family
  from `missing`/`experimental` toward `parity`.
