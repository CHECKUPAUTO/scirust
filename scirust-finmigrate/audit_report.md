# Pre-Migration Audit — INTACCR (Monthly Interest Accrual)

**Unit:** `cobol/INTACCR.cbl` → `scirust-finmigrate::accrue`
**Auditor:** Scirust Architect & Auditor
**Date:** 2026-07-11
**Gate status:** Phase 1 complete. Sandbox established, Golden Baseline
generated (`tests/sandbox/golden_baseline.csv`, 75 rows, sha256 recorded).

---

## 1. Scope

A single batch arithmetic core: compute one month's interest on a principal
and post it. Small on the surface, but it concentrates every failure mode that
sinks COBOL→Rust financial migrations: packed-decimal storage, fixed implied
scale, a specific tie-breaking rounding rule, and single-vs-per-operation
rounding. Getting this unit provably right is the template for the rest.

## 2. Hidden dependencies (the reason this needs an audit, not a rewrite)

Each item below was checked against authoritative COBOL references (§6). Each
is a place where a naive `f64` port silently produces wrong money.

### Gap-1 — Packed decimal is *decimal*, not binary. **(mitigated)**
`COMP-3` stores Binary-Coded-Decimal: two digits per byte, a trailing sign
nibble (`C`/`F` = positive, `D` = negative), and it is **exact** for every
value inside its digit budget. `principal * rate / 12` in IEEE-754 `f64`
would introduce base-2 representation error (e.g. `0.1` is inexact) and drift
at the cent level over a portfolio.
**Mitigation:** the port uses `rust_decimal::Decimal` exclusively. The crate
denies floating point at the API boundary; there is no `f32`/`f64` in the
money path. Enforced by test `no_float_in_money_path` and by the
`MANDATORY CONSTRAINTS` in the audit trail.

### Gap-2 — Implied scale is enforced on STORE, not at print. **(mitigated)**
`PIC S9(9)V99` always carries exactly 2 fractional digits; `SV9(5)` exactly 5.
The scale coercion (round/truncate) happens *at the assignment*, so the
result depends on **when** you round, not merely on the final displayed value.
**Mitigation:** the port models each field's scale explicitly and rounds only
at the field store, exactly as the COMPUTE does (Gap-4).

### Gap-3 — ROUNDED default = NEAREST-AWAY-FROM-ZERO, **not** banker's. **(mitigated)**
IBM Enterprise COBOL: with a bare `ROUNDED`, "the least significant digit of
the resultant identifier is increased by 1 whenever the most significant digit
of the excess is greater than or equal to **5**" — i.e. a tie (`…5`) rounds the
*magnitude* up, independent of sign (`+0.005→+0.01`, `−0.005→−0.01`). This is
**not** the round-half-to-even that many languages/libraries default to, and it
is **not** Rust's `f64` rounding. A banker's-rounding port fails ~1 row in 2 on
half-cent ties and biases a portfolio's interest downward.
**Mitigation:** `RoundingStrategy::MidpointAwayFromZero`. Baseline rows
`half_cent_positive` (`0.005→0.01`) and `half_cent_up_from_even`
(`0.025→0.03`, where banker's would give `0.02`) are the discriminators.

### Gap-4 — One rounding event, high-precision intermediate. **(mitigated, with residual risk → Gap-6)**
`COMPUTE WS-MONTHLY-INT ROUNDED = P * R / 12` rounds **once**, at the final
store. The `P*R` product is exact; the `/12` quotient is carried at high
intermediate precision. Rounding each sub-operation to 2 dp instead would
change results (e.g. it drops the fractional pennies that decide the tie).
**Mitigation:** the port computes the whole expression at full `Decimal`
precision and rounds only at the end. The un-ROUNDED companion
`WS-MONTHLY-TRUNC` (truncate toward zero, `RoundingStrategy::ToZero`) is
reproduced too, so both disciplines are proven.

### Gap-5 — Silent size-error truncation. **(documented divergence)**
`INTACCR` codes no `ON SIZE ERROR`. On the legacy platform a result exceeding
`9(9)` integer digits silently drops high-order digits — a catastrophic,
invisible corruption. The port refuses to reproduce silent corruption: it
returns `AccrualError::SizeError` instead. Sandbox inputs are bounded so the
branch is never exercised in equivalence; the divergence is deliberate and is
logged in the audit trail. **This must be signed off by the business before
production**, because it changes a silent-wrong into a loud-stop.

### Gap-6 — Intermediate precision is compiler-specific. **(residual risk — GATE)**
IBM fixed-point intermediates cap at **30 digits (compatibility mode) / 31
digits (extended mode)**; GnuCOBOL uses arbitrary-precision GMP. The Golden
Baseline models the intermediate at **38 digits**. For this unit's magnitudes
(product ≤ ~16 digits, then `/12`) 38 vs 31 cannot change the 2-dp result, so
the residual risk is *nil for the committed dataset*. But it is **not nil in
general**, and division digit-allocation is compiler- and option-dependent
(`ARITH(COMPAT|EXTEND)`). IBM even shipped a correctness APAR for exactly this
class of bug (PH64936, "incorrect rounding and truncation in COBOL arithmetic
expressions"). **Therefore the baseline is model-derived, not compiler-derived,
and the production gate (§5) is mandatory.**

## 3. Sandbox

`tests/sandbox/` contains:
* `gen_baseline.py` — deterministic generator (fixed seed) encoding the §6
  semantics in Python `decimal`. Emits inputs + expected outputs + a SHA-256.
* `dataset.csv` — 11 hand-picked edge cases (documented by `case_id`) + 64
  deterministic pseudo-random rows spanning the PIC range and sign.
* `golden_baseline.csv` — the oracle. Immutable; regeneration must reproduce it
  byte-for-byte or the change is flagged.
* `golden_baseline.sha256` — tamper evidence.

## 4. Signed-zero note
Truncating `−0.005` toward zero yields `−0.00`; COMP-3 can carry a `D` (negative)
sign nibble on a zero magnitude. Numerically this is zero. The equivalence test
therefore compares **parsed `Decimal` values**, not formatted strings, so a
`-0.00` vs `0.00` display difference is not a false failure. Flagged for the
downstream ledger team in case any consumer is sign-nibble sensitive.

## 5. Production gate (blocking, before this unit ships)
1. Compile `INTACCR.cbl` with the **target** compiler (`cobc -x -free`, or the
   z/OS Enterprise COBOL build) under the production `ARITH` option.
2. Drive it with `dataset.csv`; capture DISPLAY output in the baseline column
   layout.
3. Re-run the Rust equivalence test against the compiler-derived baseline.
4. Only on 100% parity (deviation `0`, tolerance `1e-10`) is the unit promoted.
Until step 4, every equivalence claim here is "equivalent to the documented
semantic model", explicitly labelled as such.

## 6. References (authoritative COBOL sources consulted 2026-07-11)
* IBM Enterprise COBOL for z/OS 6.4 — ROUNDED phrase (default: increment when
  excess digit ≥ 5 ⇒ NEAREST-AWAY-FROM-ZERO):
  https://www.ibm.com/docs/en/cobol-zos/6.4.0?topic=operations-rounded-phrase
* IBM Enterprise COBOL — intermediate results & arithmetic precision (30/31
  fixed-point digit cap):
  http://www.cadcobol.com.br/cobol_appendixes_intermediate_results_and_arithmetic_precision.htm
* IBM APAR PH64936 — "Incorrect rounding and truncation in COBOL arithmetic
  expressions for AMODE64" (compiler-version-dependent arithmetic ⇒ live-baseline
  gate): https://www.ibm.com/support/pages/apar/PH64936
* COMP-3 packed-decimal layout (BCD, 2 digits/byte, sign nibble C/F/D):
  https://www.mainframestechhelp.com/tutorials/cobol/comp-3.htm ·
  http://www.3480-3590-data-conversion.com/article-packed-fields.html
