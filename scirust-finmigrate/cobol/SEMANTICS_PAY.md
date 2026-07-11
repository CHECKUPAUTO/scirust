# PAYCALC — Legacy Semantic Specification

Exact arithmetic contract of `PAYCALC.cbl`. Reference for
`tests/sandbox/gen_pay_baseline.py` and `src/paycalc.rs`. Builds on
`SEMANTICS.md`; only the exponentiation-specific semantics are detailed here.

## Field model

| Field            | PICTURE            | Digits | Scale (dp) | Notes |
|------------------|--------------------|--------|------------|-------|
| `WS-PRINCIPAL`   | `S9(9)V99 COMP-3`  | 11     | 2          | loan amount |
| `WS-RATE`        | `SV9(7) COMP-3`    | 7      | 7          | monthly rate, e.g. 0.0041667 |
| `WS-NUM-PERIODS` | `9(3) COMP-3`      | 3      | 0          | integer term, 1 … 999 |
| `WS-FACTOR`      | `S9(5)V9(9) COMP-3`| 14     | 9          | `(1+i)^n`, stored at 9 dp |
| `WS-PAYMENT`     | `S9(7)V99 COMP-3`  | 9      | 2          | level payment (ROUNDED) |

## Why the exponent is an integer (the crux)

Authoritative COBOL rule (IBM Enterprise COBOL, §References):

* **"The compiler performs an exponentiation to a nonzero integer power as a
  succession of multiplications"** — an **integer** exponent is evaluated in
  **fixed-point decimal**, not floating point.
* **"Expressions that contain … fractional exponents … are evaluated as if all
  operands … converted to long-precision floating point"** — a **fractional or
  negative** exponent forces the whole expression into IEEE-754 double.

The classic annuity formula `P·i / (1 − (1+i)^−n)` has a **negative** exponent
⇒ it would evaluate in float, importing base-2 error into the money path.
PAYCALC uses the algebraically-identical positive-integer form:

```
f = (1 + i) ** n           (n a positive integer ⇒ fixed-point multiplies)
payment = P · i · f / (f − 1)
```

Both forms are exactly equal in real arithmetic; only the *implementation
arithmetic* differs, and the integer form keeps everything decimal.

## The contract

1. **Zero-rate special case.** If `i = 0` then `f = 1` and `f − 1 = 0`; the
   annuity divide is undefined. The program computes the straight-line payment
   `principal / num_periods`, ROUNDED to cents. (This special-case is coded in
   the legacy program; it is not an inferred convenience.)
2. **Factor.** `f = (1+i)^n` as a succession of fixed-point multiplications,
   then **ROUNDED once** into the 9-dp `WS-FACTOR` (NEAREST-AWAY-FROM-ZERO).
   The store at 9 dp is deliberately far coarser than any compiler intermediate
   cap (IBM 30/31 fixed-point digits; `rust_decimal` 28 significant digits), so
   the 9-dp result is **insensitive** to which cap applies — the difference
   between the two lives far below the 10th decimal place for the bounded
   inputs (see §Bounds). This neutralises the Gap-6 residual risk *for this
   field*.
3. **Payment.** `payment = (P · i · f) / (f − 1)`, evaluated at full
   intermediate precision, ROUNDED once into the 2-dp `WS-PAYMENT`.
4. **Rounding mode.** Every `ROUNDED` is NEAREST-AWAY-FROM-ZERO, as in INTACCR.

## Bounds (keep the factor field and intermediates well-conditioned)

* `1 ≤ num_periods ≤ 120` and `0 ≤ i ≤ 0.05` per month. Then `f = (1+i)^n ≤
  (1.05)^120 ≈ 372`, comfortably inside `S9(5)V9(9)` and inside 28/30
  significant digits at 9 dp. Under these bounds the 28-vs-30/31-digit
  intermediate difference cannot change the 9-dp factor or the 2-dp payment.
* Inputs are further bounded so `WS-PAYMENT` stays inside `S9(7)V99`.

## Cross-check against the float path (audit evidence, not shipped)

The baseline generator ALSO computes the payment the legacy-float way (IEEE-754
double, the arithmetic the negative-exponent form would have used) and asserts
it equals the decimal-native payment **at the cent**. This proves the
arithmetic change (float → decimal) does **not** change the customer's payment.
The float path lives only in the oracle; the shipped Rust port is decimal-only.

## Divergences / gate
* Size error handled as in the other units (loud `SizeError`, bounded out of the
  equivalence set).
* Baseline is **model-derived** (no `cobc`); regenerate from a live compiler run
  of `PAYCALC.cbl` under the production `ARITH` option before shipping.

## References
* IBM Enterprise COBOL — integer power is a succession of multiplications;
  fractional-exponent expressions evaluate in long floating point:
  http://www.cadcobol.com.br/cobol_appendixes_intermediate_results_and_arithmetic_precision.htm ·
  https://www.ibm.com/docs/en/cobol-zos/6.4.0?topic=results-example-exponentiation-in-fixed-point-arithmetic
* IBM Enterprise COBOL — making exponentiations efficient (float for large
  exponents): https://www.ibm.com/docs/en/cobol-zos/6.4.0?topic=types-making-exponentiations-efficient
