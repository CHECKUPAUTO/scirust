# Tooling (Phase 0 deliverables)

## Registry: `tensor-operators.toml` (repo root)

Machine-readable operator registry. One `[[operator]]` entry per row with:
`name`, `torch` (reference symbol), `family`, `impls`, `dtypes`, `autograd`,
`tolerance { atol, rtol }`, `status`, optional `notes`, and separate proof
metadata:
- `verified_impls` + `verified` for direct production-implementation evidence;
- `reference_impl` + `reference_verified` for the independent
  `scirust_core::tensor::parity` reference kernels.

`impls` is an implementation inventory and never constitutes proof by itself.
The `baseline` header pins the frozen PyTorch identity
(2.13.0 / cf30153c4c131c8164ee7798e5022d810682e2cb).

## Generator: `tools/tensor-parity`

Standalone Rust binary (NOT part of the scirust workspace — it has its own
`[workspace]` table and builds depend on nothing else with `cargo run`):

- reads `tensor-operators.toml`,
- sorts rows by name (deterministic output; no timestamps),
- writes `artifacts/tensor-coverage.json` + `artifacts/tensor-coverage.csv`,
- prints a one-line summary (totals per status).

Regenerate:

```bash
cargo run --manifest-path tools/tensor-parity/Cargo.toml
```

The outputs must be committed with any registry change (CI, when added,
fails on drift).

## Fixture pipeline (next phase, after Phase 0 ships)

Fixtures are generated **offline** (never in CI) by executing the frozen
baseline and are *data*:

1. Script under `docs/tensor-parity/provenance/` runs PyTorch 2.13.0
   fixtures per row (inputs + outputs + grads) and records the baseline
   commit and inputs' PRNG seeds.
2. Fixtures are committed under `tests/parity/fixtures/<family>/<op>/` with a
   manifest (`fixtures.toml`) + content hash.
3. The current Rust-only **reference harness**
   (`scirust-core/tests/parity_differential.rs`, no Python dependency) loads
   fixtures and exercises `scirust_core::tensor::parity`. Passing this step
   establishes `reference_parity`.
4. A production row becomes `status = "parity"` only after a differential
   test calls the claimed 2D/ND/core/SIMD/GPU/CUDA/sparse implementation
   directly against the same witness and satisfies the row's autograd/error
   requirements.

## Integrity

- `artifacts/` files are generated; the JSON output carries `generated: true`
  and the frozen baseline identity.
- The coverage generator rejects a production `parity` row without direct
  `verified` metadata.
- `reference_verified` records witness/reference-kernel evidence only and
  cannot substitute for a production `verified` proof.
- The committed fixture manifest is independently validated for provenance,
  exhaustive file coverage and SHA-256 integrity in CI.
- Historical fixture families are immutable. The offline generator verifies
  their committed SHA-256 entries before any write and cannot dispatch or write
  those families.
- New fixture families are append-only and each owns an independent frozen PRNG
  seed. Adding a family therefore cannot shift the inputs or hashes of any
  earlier witness.
