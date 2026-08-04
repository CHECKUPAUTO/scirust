# Tooling (Phase 0 deliverables)

## Registry: `tensor-operators.toml` (repo root)

Machine-readable operator registry. One `[[operator]]` entry per row with:
`name`, `torch` (reference symbol), `family`, `impls`, `dtypes`, `autograd`,
`tolerance { atol, rtol }`, `status`, optional `notes`.
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
3. A Rust-only harness (`tests/parity/`, no Python dependency) loads
   fixtures, runs the SciRust row impl, compares against recorded outputs as
   `(atol, rtol)`, and runs gradcheck for `autograd` rows.
4. Only when step 3 passes in CI does the row's `status` become `parity`, and
   the generator is re-run.

## Integrity

- `artifacts/` files are generated; both carry `generated: true` in JSON and
  the baseline identity.
- Rows claiming `parity` without a committed fixture manifest will fail the
  future CI gate; until that gate exists, reviewers check the PR checklist
  (see MIGRATION_RISKS.md R10).