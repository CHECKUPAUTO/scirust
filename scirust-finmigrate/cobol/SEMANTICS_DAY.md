# DAYCOUNT — Legacy Semantic Specification

Exact contract of `DAYCOUNT.cbl`. Reference for
`tests/sandbox/gen_day_baseline.py` and `src/daycount.rs`.

## Field model

| Field           | PICTURE           | Notes |
|-----------------|-------------------|-------|
| `WS-PRINCIPAL`  | `S9(9)V99 COMP-3` | 2 dp |
| `WS-ANNUAL-RATE`| `SV9(7) COMP-3`   | 7 dp annual rate |
| `WS-Y/M/D 1,2`  | `9(4)/9(2)/9(2)`  | two calendar dates |
| `WS-DAYS`       | `S9(6) COMP-3`    | 30/360 day count (integer) |
| `WS-INTEREST`   | `S9(9)V99 COMP-3` | accrued interest, ROUNDED |

## The day count: 30/360 US (NASD bond basis)

`WS-DAYS = 360·(Y2−Y1) + 30·(M2−M1) + (D2adj − D1adj)`, where `D1adj`/`D2adj`
are the day-of-month after applying, **in this exact order**, with the
February flags read from the **original** dates:

1. If `D1` **and** `D2` are the last day of February → `D2adj = 30`.
2. If `D1` is the last day of February → `D1adj = 30`.
3. If `D2adj = 31` **and** (`D1adj = 30` or `31`) → `D2adj = 30`.
4. If `D1adj = 31` → `D1adj = 30`.

* **Ordering is load-bearing.** Rule 3 tests `D1adj = 30 or 31`, so it catches a
  `D1 = 31` that rule 4 has not yet reduced (e.g. `31-Jan → 31-Mar` = 60, not
  61). Reordering rule 4 before rule 3 without that `or 31` clause changes the
  answer.
* **"Last day of February"** = 29 in a leap year, else 28. Gregorian leap:
  `Y mod 4 = 0 and (Y mod 100 ≠ 0 or Y mod 400 = 0)`. Getting the century rule
  wrong (1900 is **not** leap; 2000 **is**) mis-flags Feb-EOM.

### This is NOT Excel `DAYS360`
Excel's US `DAYS360` omits rules 1–2 (the February EOM rules). The two agree
except when a date is the last day of February: e.g. `28-Feb-2023 → 31-Aug-2023`
is **180** days on the NASD bond basis but **183** in Excel. Choosing the wrong
variant silently mis-accrues interest. This unit implements the **NASD bond
basis**; the sandbox records the Excel day count alongside to make every
divergence visible (audit evidence, not shipped).

## The interest COMPUTE

`COMPUTE WS-INTEREST ROUNDED = principal · rate · days / 360` — a single
rounding event (NEAREST-AWAY-FROM-ZERO) into the 2 dp field; the `days/360`
factor is **not** pre-rounded (same discipline as INTACCR Gap-4).

## Contract / divergences
* Dates are assumed valid calendar dates with `Date2 ≥ Date1` (forward accrual).
  The adjustment rules are asymmetric in D1/D2, so a reversed pair is out of
  contract; the port computes the formula as coded (it does not attempt to
  re-derive a "negative" convention).
* Size error handled as elsewhere (loud `SizeError`, bounded out of the
  equivalence set).

## Gate
Baseline is **model-derived** (no `cobc`); regenerate from a live
`cobc -x -free cobol/DAYCOUNT.cbl` before shipping.

## References
* 30/360 US (NASD) ordered rules and formula — Wikipedia, *Day count convention*:
  https://en.wikipedia.org/wiki/Day_count_convention
* 30/360 US definition (SIFMA bond basis): https://cbonds.com/glossary/30-360-us/
