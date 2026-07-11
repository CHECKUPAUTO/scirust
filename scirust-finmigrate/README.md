# scirust-finmigrate

An **audit-gated COBOL→Rust financial migration harness**, demonstrated end to
end on one representative unit: `INTACCR`, a monthly interest-accrual routine.

It is the reference template for the migration protocol:

> *No migration without audit, no audit without a sandbox, no code without
> verified equivalence.*

## Layout

| Path | Phase | What it is |
|------|-------|------------|
| `cobol/INTACCR.cbl` | — | The legacy source of truth (real COBOL artifact). |
| `cobol/SEMANTICS.md` | 1 | Exact arithmetic contract: PIC, COMP-3, ROUNDED, intermediates. |
| `audit_report.md` | 1 | Pre-migration gap analysis (hidden dependencies + production gate). |
| `tests/sandbox/gen_baseline.py` | 1 | Deterministic Golden Baseline generator (Python `decimal`). |
| `tests/sandbox/golden_baseline.csv` | 1 | The oracle (+ `.sha256` tamper evidence). |
| `tests/equivalence.rs` | 2 | Exact decimal equivalence proof against the baseline. |
| `src/lib.rs` | 2–3 | The decimal-exact port, reversibility shim, replay tracing. |
| `audit_trail.log` | 3 | Append-only decision log incl. the recorded red→green. |

## The four hidden dependencies this unit exercises

1. **COMP-3** packed decimal is *decimal-exact* — the port uses `rust_decimal`,
   never `f64` (base-2 error would drift at the cent level).
2. **Fixed implied scale** (`V99`, `V9(5)`) enforced on store.
3. **ROUNDED default = NEAREST-AWAY-FROM-ZERO** (`0.005→0.01`, `−0.005→−0.01`),
   *not* banker's rounding — verified against IBM Enterprise COBOL docs.
4. **One rounding event** at the store; the `P*R/12` chain is carried at full
   precision, not rounded per operation.

## Run it

```sh
# (Phase 1) regenerate the Golden Baseline — must be byte-stable
python3 tests/sandbox/gen_baseline.py

# (Phase 2) prove equivalence + run unit tests
cargo test -p scirust-finmigrate
```

## Status & production gate

Phases 1–2 are complete and Phase 3 scaffolding is in place — **against a
model-derived baseline**. No `cobc` was available in this sandbox, so the
baseline is generated from the documented semantic model, not a live compiler.

**Before production**, the baseline MUST be regenerated from the target COBOL
compiler (`cobc -x -free cobol/INTACCR.cbl`, or the z/OS Enterprise COBOL build
under the production `ARITH` option) and re-diffed at 100% parity. See
`audit_report.md` §5. Until then, equivalence means "equivalent to the
documented semantic model" — and every artifact says so.
