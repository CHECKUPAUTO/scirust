# INTACCR — Legacy Semantic Specification

This document pins down the *exact* arithmetic contract of `INTACCR.cbl`.
It is the reference against which both the Golden Baseline generator
(`tests/sandbox/gen_baseline.py`) and the Rust port (`src/lib.rs`) are
written. If the target production compiler disagrees with any clause here,
that clause — not the Rust code — is the defect, and the baseline must be
regenerated from a live `cobc` run (see the Gate note at the bottom).

## Field model (PICTURE → value domain)

| Field              | PICTURE            | Storage | Digits | Scale (dp) | Signed | Domain                         |
|--------------------|--------------------|---------|--------|------------|--------|--------------------------------|
| `WS-PRINCIPAL`     | `S9(9)V99 COMP-3`  | packed  | 11     | 2          | yes    | −999 999 999.99 … 999 999 999.99 |
| `WS-ANNUAL-RATE`   | `SV9(5) COMP-3`    | packed  | 5      | 5          | yes    | −0.99999 … 0.99999             |
| `WS-MONTHLY-INT`   | `S9(9)V99 COMP-3`  | packed  | 11     | 2          | yes    | as principal                   |
| `WS-MONTHLY-TRUNC` | `S9(9)V99 COMP-3`  | packed  | 11     | 2          | yes    | as principal                   |
| `WS-NEW-BALANCE`   | `S9(9)V99 COMP-3`  | packed  | 11     | 2          | yes    | as principal                   |

* **COMP-3 (packed decimal).** Two decimal digits per byte plus a trailing
  sign nibble. It is a *decimal* representation: there is no binary-floating
  approximation and no base-2 rounding. Every value that fits the digit/scale
  budget is represented exactly. The Rust port therefore MUST use a decimal
  type (`rust_decimal::Decimal`), never `f32`/`f64`.
* **Implied scale is enforced on STORE.** `V` is an *implied* decimal point:
  the field always carries exactly its declared number of fractional digits.
  Assigning a value with more fractional digits than the field allows forces
  a scale adjustment (rounding or truncation, see below) at the moment of the
  store — not lazily at display time.

## The COMPUTE contract

```cobol
COMPUTE WS-MONTHLY-INT ROUNDED = WS-PRINCIPAL * WS-ANNUAL-RATE / 12.
```

1. **Intermediate precision.** The expression `principal * rate / 12` is
   evaluated as a single arithmetic expression. The `principal * rate`
   product is *exact* (2 dp × 5 dp ⇒ 7 dp). The `/ 12` division is carried
   at high intermediate precision — well beyond the 2 dp of the target field.
   We model the intermediate at **38 significant decimal digits**, which is
   more than the destination scale can ever observe and matches the
   standard-conforming behaviour of GnuCOBOL's GMP-backed decimal
   intermediates. **There is no per-operation rounding.**
2. **Single rounding event.** Rounding happens exactly once, when the fully
   evaluated intermediate is stored into the 2 dp field `WS-MONTHLY-INT`.
3. **Rounding mode = NEAREST-AWAY-FROM-ZERO.** The bare `ROUNDED` keyword
   selects COBOL's default mode: round half away from zero, sign-aware.
   * `+0.005 → +0.01`, `−0.005 → −0.01` (magnitude ties round *up*).
   * This is **not** banker's rounding (NEAREST-EVEN). `0.005` must become
     `0.01`, never `0.00`.
   * Maps to `rust_decimal::RoundingStrategy::MidpointAwayFromZero`
     and Python `decimal.ROUND_HALF_UP`.

```cobol
COMPUTE WS-MONTHLY-TRUNC = WS-PRINCIPAL * WS-ANNUAL-RATE / 12.
```

4. **Default = truncation toward zero.** Without `ROUNDED`, the surplus
   fractional digits are dropped toward zero (`+2.999 → +2.99`,
   `−2.999 → −2.99`). Maps to `RoundingStrategy::ToZero` /
   Python `decimal.ROUND_DOWN`.

```cobol
ADD WS-MONTHLY-INT TO WS-PRINCIPAL GIVING WS-NEW-BALANCE.
```

5. **Posting.** Both operands are already 2 dp; the sum is exact at 2 dp and
   stored without rounding. `new_balance = principal + monthly_int_rounded`.

## Overflow / size-error

`ON SIZE ERROR` is **not** coded. A result exceeding `9(9)` integer digits
would, on the legacy platform, silently truncate the high-order digits. The
sandbox inputs are bounded so this branch is never reached; the Rust port
treats a size-error as an explicit error (see `AccrualError::SizeError`)
rather than reproducing silent high-order truncation. This is a deliberate,
documented divergence — see `audit_report.md` §Gap-5.

## Gate note (authoritative baseline)

No `cobc` is available in this sandbox, so the committed Golden Baseline is
**model-derived**: produced by `gen_baseline.py`, which encodes the five
clauses above using Python's `decimal` module. Before this unit is promoted
to production, the baseline MUST be regenerated from a live GnuCOBOL
(`cobc -x INTACCR.cbl`) or the target mainframe compiler, and re-diffed. The
generator prints the exact command and expected column layout to make that a
mechanical swap. Until then, every equivalence claim in this crate is
"equivalent to the documented semantic model", not "equivalent to a live
compiler run" — and it is labelled as such in the audit trail.
