# scirust-finmigrate

An **audit-gated COBOL→Rust financial migration harness**, demonstrated end to
end on two representative units:
* `INTACCR` — a monthly interest-accrual routine (single arithmetic store).
* `AMORTSCH` — a fixed-payment loan amortization schedule (running balance,
  accumulated rounding drift, final-payment reconciliation to exactly `0.00`).
* `PAYCALC` — the level (annuity) payment that AMORTSCH consumes, computed with
  a positive-integer exponent so it stays fixed-point decimal instead of
  triggering COBOL's floating-point `**` (proven equal to the float path at the
  cent).
* `DAYCOUNT` — accrued interest on the 30/360 US (NASD bond-basis) day count,
  implementing the February end-of-month rules that Excel `DAYS360` omits (the
  divergence is recorded per-scenario as audit evidence).
* `BRKTCALC` — progressive (marginal) bracketed tax over a COBOL `OCCURS` table;
  each rate applies only to its slice of the base, with a single rounding event
  (the wrong flat-top-rate figure is recorded as audit evidence).
* `CURRCVT` — national↔national euro conversion by triangulation (EC 1103/97):
  route through the euro, round the intermediate to ≥3 dp, then to the target
  currency's minor unit (2 dp, or 0 dp for ITL/ESP); the unlawful direct
  cross-rate is recorded to show the legally-material divergence.

It is the reference template for the migration protocol:

> *No migration without audit, no audit without a sandbox, no code without
> verified equivalence.*

> ⚠️ **Not production-ready.** The units are validated under **GnuCOBOL 3.1.2**,
> not the production IBM Enterprise COBOL / z/OS toolchain, and not against real
> data with business sign-off. The blocking exit criteria are consolidated in
> [`PRODUCTION_GATES.md`](PRODUCTION_GATES.md) — read it before deploying
> anything here against real money.

## Layout

| Path | Phase | What it is |
|------|-------|------------|
| `cobol/INTACCR.cbl` | — | The legacy source of truth (real COBOL artifact). |
| `cobol/SEMANTICS.md` | 1 | Exact arithmetic contract: PIC, COMP-3, ROUNDED, intermediates. |
| `audit_report.md` | 1 | Pre-migration gap analysis (hidden dependencies + production gate). |
| `tests/sandbox/gen_baseline.py` | 1 | Deterministic Golden Baseline generator (Python `decimal`). |
| `tests/sandbox/golden_baseline.csv` | 1 | The oracle (+ `.sha256` tamper evidence). |
| `tests/equivalence.rs` | 2 | Exact decimal equivalence proof against the baseline. |
| `src/lib.rs` | 2–3 | INTACCR port, reversibility shim, replay tracing. |
| `cobol/AMORTSCH.cbl` · `cobol/SEMANTICS_AMORT.md` | 1 | Unit 2: amortization source + contract. |
| `tests/sandbox/gen_amort_baseline.py` · `amort_baseline.csv` | 1 | Unit 2 baseline generator + oracle. |
| `src/amort.rs` · `tests/amort_equivalence.rs` | 2 | Unit 2 port + per-period equivalence proof. |
| `tests/no_float_guard.rs` | — | Fails if `f32`/`f64` appears in the money path. |
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
# (Phase 1) regenerate a Golden Baseline — must be byte-stable
python3 tests/sandbox/gen_baseline.py

# (Phase 2) prove equivalence + run unit tests
cargo test -p scirust-finmigrate

# (Phase 3) consolidated audit: verify every baseline's SHA-256 and print a
# one-shot audit report across all units. Exits non-zero on any tamper.
cargo run -p scirust-finmigrate --bin finaudit
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
