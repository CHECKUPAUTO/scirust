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
money path. Enforced by construction — every signature and intermediate is
`Decimal`, and the `no_float_in_money_path` guard test greps the crate sources
(`src/lib.rs`, `src/amort.rs`, `src/paycalc.rs`, `src/daycount.rs`,
`src/brktcalc.rs`, `src/currcvt.rs`) for `f32`/`f64` and fails if any appears. The
`MANDATORY CONSTRAINTS` in the audit trail record the same rule.

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

---

# Pre-Migration Audit — AMORTSCH (Loan Amortization Schedule)

**Unit:** `cobol/AMORTSCH.cbl` → `scirust-finmigrate::amort::amortize`
**Date:** 2026-07-11 · **Gate status:** Phase 1 & 2 complete vs the model baseline
(`tests/sandbox/amort_baseline.csv`). Contract: `cobol/SEMANTICS_AMORT.md`.

## Why a second unit
INTACCR was a single arithmetic store. AMORTSCH carries a **running balance
across periods**, which surfaces two failure modes a one-shot routine cannot,
plus it re-exercises every INTACCR gap (COMP-3, implied scale, NEAREST-AWAY-FROM-ZERO,
one-rounding-event) on each iteration.

### Gap-A — Accumulated rounding drift. **(mitigated)**
Interest is independently `ROUNDED` to the cent every period, on the current
balance. Over N periods these half-cent roundings accumulate; a port that
rounds even slightly differently (banker's, or rounding the product before the
store) drifts by whole pennies by the final row. **Mitigation:** the same
`store_money_rounded` (MidpointAwayFromZero, single event) is applied per period;
the equivalence test checks every period-row of every scenario, so drift is
caught at the first divergent cent. Scenario `rounding_drift` targets ties.

### Gap-B — Final-payment reconciliation. **(mitigated)**
Because of Gap-A the scheduled payment cannot close the balance to zero. The
legacy rule — at the last period, or as soon as `princ >= balance`, set
`princ := balance` and let the actual payment absorb the difference — is what
makes the ending balance **exactly 0.00**. The `>=` boundary and the last-period
test are load-bearing: `>` instead of `>=`, or testing only the last period,
leaves a residual penny (an audit finding) or overshoots below zero.
**Mitigation:** reproduced verbatim; the equivalence test asserts the closing
balance is exactly `0.00` and matches the baseline row count (early-payoff /
reconciliation timing). Scenarios `neg_amortization`, `early_payoff`,
`uneven_tail`, `single_period` pin the boundary cases.

### Gap-C — Negative amortization & row count. **(mitigated)**
When `payment < interest` the principal portion is negative and the balance
**grows**; the loop still terminates via the last-period payoff. Early payoff
(large payment) makes the schedule **shorter** than `num_periods`. Both change
the emitted row set, so the test compares row COUNT, not just values.

### Gap-D — Size error under runaway neg-am. **(documented divergence)**
A balance growing past `9(9)V99` would silently truncate on the legacy platform;
the port returns `AccrualError::SizeError` (loud stop). Sandbox scenarios are
bounded (generation asserts no field overflows), so the equivalence set never
fires it; proven separately in `amort::tests::size_error_on_runaway_negative_amortization`.

### Deliberate scope choice — no annuity formula
The scheduled payment is an **input**, not derived from
`P·i / (1 − (1+i)^−n)`. COBOL's `**` operator evaluates with **floating-point**
intermediates even inside an otherwise fixed-point `COMPUTE` — importing float
into the money path. Keeping the payment an input holds the no-float mandate;
migrating the annuity-payment derivation is a separate unit with its own audit
of that float dependency.

## Production gate
Identical to §5 above: regenerate `amort_baseline.csv` from a live `cobc -x -free
cobol/AMORTSCH.cbl` (or the z/OS build under the production `ARITH` option),
drive it with `amort_scenarios.csv`, and re-diff at exact parity before shipping.

---

# Pre-Migration Audit — PAYCALC (Annuity Payment)

**Unit:** `cobol/PAYCALC.cbl` → `scirust-finmigrate::paycalc::payment`
**Date:** 2026-07-11 · **Gate status:** Phase 1 & 2 complete vs the model baseline
(`tests/sandbox/pay_baseline.csv`). Contract: `cobol/SEMANTICS_PAY.md`.

## Why a third unit
PAYCALC computes the fixed payment AMORTSCH consumes — closing the loop — and is
the unit that collides with the project's central tension: the **no-floating-point
mandate** versus a formula the legacy would normally evaluate in float.

### Gap-E — Exponentiation dispatches to float on a fractional/negative exponent. **(mitigated by rewrite)**
Authoritative COBOL rule: an expression with a **fractional or negative
exponent** is evaluated "as if all operands … converted to long-precision
floating point"; but an exponentiation "to a nonzero **integer** power" is "a
succession of multiplications" in **fixed-point**. The textbook annuity formula
`P·i / (1 − (1+i)^−n)` uses a negative exponent ⇒ float ⇒ base-2 error in the
money path. **Mitigation:** the port uses the algebraically-identical
positive-integer form `P·i·f/(f−1)`, `f=(1+i)^n`, so every operation is
fixed-point decimal. This is a genuine migration *decision*, not a mechanical
port — documented in the audit trail.

### Gap-F — Intermediate precision of the multiply chain. **(mitigated by staged scale)**
`(1+i)^n` as repeated multiplication grows fractional digits without bound;
COBOL caps fixed-point intermediates at 30/31 digits and `rust_decimal` at 28
significant digits (the Gap-6 tension, now on the critical path). **Mitigation:**
`WS-FACTOR` is stored at a fixed **9 dp** — a single rounding event at a scale
far coarser than either cap. Under the documented bounds (`n ≤ 120`, `i ≤ 0.05`,
so `f ≤ ~372`) the 28-vs-30/31-digit difference lives far below the 10th decimal
and cannot change the 9-dp factor or the 2-dp payment. Proven empirically:
`rust_decimal` (28-digit) and the Python model (38-digit) agree exactly on the
9-dp factor for every scenario.

### Gap-G — Zero-rate division by zero. **(mitigated)**
At `i = 0`, `f = 1` and `f − 1 = 0`; the annuity divide is undefined. The legacy
program special-cases it to the straight-line payment `principal / num_periods`.
The port reproduces the branch; a zero *term* (`num_periods = 0`, outside the
`9(3)` domain of ≥ 1) is rejected as a `SizeError` rather than dividing by zero.

## Cross-check: the arithmetic change does not move the money
The baseline generator computes the payment **both** ways — decimal-native (the
shipped path) and legacy-float (IEEE-754 double, the arithmetic the
negative-exponent form would have used) — and asserts they are equal **to the
cent** for every scenario. All 8 scenarios agree. This is the evidence that
choosing decimal over float (Gap-E) is safe: it changes the arithmetic, not the
customer's payment. The float path exists only in the oracle; the shipped port
is decimal-only (enforced by `tests/no_float_guard.rs`, now covering
`src/paycalc.rs`).

## Composition proof
`paycalc::tests::payment_amortizes_to_zero_in_amort` feeds PAYCALC's payment into
AMORTSCH and checks the schedule closes to exactly `0.00` — the two units agree
at their shared boundary.

## Production gate
Regenerate `pay_baseline.csv` from a live `cobc -x -free cobol/PAYCALC.cbl` under
the production `ARITH` option and re-diff at exact parity before shipping.

---

# Pre-Migration Audit — DAYCOUNT (30/360 US Accrued Interest)

**Unit:** `cobol/DAYCOUNT.cbl` → `scirust-finmigrate::daycount`
**Date:** 2026-07-11 · **Gate status:** Phase 1 & 2 complete vs the model baseline
(`tests/sandbox/day_baseline.csv`). Contract: `cobol/SEMANTICS_DAY.md`.

## Why a fourth unit
The arithmetic is a single `principal·rate·days/360`; the entire risk lives in
`days`. "30/360 US" is a name shared by two *different* conventions, and the
legacy answer depends on which one the shop coded.

### Gap-H — "30/360 US" is ambiguous: NASD bond basis vs Excel DAYS360. **(decided + evidenced)**
The SIFMA/NASD **bond basis** applies February end-of-month rules; Excel
`DAYS360` (US) does **not**. They disagree whenever a date is the last day of
February — materially: `28-Feb-2023 → 31-Aug-2023` is **180** days on the NASD
basis but **183** in Excel; on a $100,000 position at 5% that is a **$41.67**
difference for one period. **Decision:** implement the NASD bond basis (the
standard for US corporate bonds), cited in `SEMANTICS_DAY.md`. **Evidence:** the
baseline records the Excel count alongside the NASD count for every row; 3 of 10
scenarios diverge, and the equivalence test pins the NASD value.

### Gap-I — Rule ordering is load-bearing. **(mitigated)**
The four adjustment rules must run in order, with the February flags read from
the **original** dates, and rule 3 must test `D1 = 30 or 31` so it catches a
`D1 = 31` that rule 4 has not yet reduced. Otherwise `31-Jan → 31-Mar` yields 61
instead of 60. **Mitigation:** the port reproduces the exact order and the
`30 or 31` clause; `thirty_first_rules` pins the boundary cases.

### Gap-J — Leap-year definition of "last day of February". **(mitigated)**
"Last day of February" is 29 in a leap year, else 28. The Gregorian century rule
is a classic trap: **1900 is not leap, 2000 is**. A wrong rule mis-flags Feb-EOM
and shifts the day count. **Mitigation:** `is_leap` = `y%4==0 && (y%100!=0 ||
y%400==0)`; `leap_year_rule` pins 1900/2000.

### Gap-K — Single rounding event on the interest. **(mitigated)**
`principal·rate·days/360` rounds **once** into the 2 dp field; `days/360` is not
pre-rounded (same discipline as INTACCR Gap-4). `interest_single_rounding` pins
`1055.5555… → 1055.56`.

## Contract note
Dates are assumed valid with `Date2 ≥ Date1` (forward accrual). The rules are
asymmetric in D1/D2, so a reversed pair is out of contract — documented, not
silently "handled".

## Production gate
Regenerate `day_baseline.csv` from a live `cobc -x -free cobol/DAYCOUNT.cbl` and
re-diff before shipping.

---

# Pre-Migration Audit — BRKTCALC (Progressive Bracketed Tax)

**Unit:** `cobol/BRKTCALC.cbl` → `scirust-finmigrate::brktcalc`
**Date:** 2026-07-11 · **Gate status:** Phase 1 & 2 complete vs the model baseline
(`tests/sandbox/brkt_baseline.csv`). Contract: `cobol/SEMANTICS_BRKT.md`.

## Why a fifth unit
The first four units were single-`COMPUTE` arithmetic. BRKTCALC introduces the
COBOL **table (`OCCURS`) + `PERFORM VARYING`** idiom with **marginal** bracket
arithmetic — a different shape of code with its own failure modes.

### Gap-L — Marginal, not flat. **(mitigated + evidenced)**
Each bracket's rate applies only to the slice of the base within that bracket.
The classic porting error — `base × top_rate` — over-taxes wildly (base 100 000:
flat 32 % = 32 000 vs correct marginal 16 500 = 3 000 + 9 900 + 3 600).
**Mitigation:** the port sums per-bracket slices. **Evidence:** the baseline
records the flat figure per row and the equivalence test asserts `marginal < flat`
on every scenario past the first bracket; `flat_would_be_wrong` pins it.

### Gap-M — Boundary inclusivity. **(mitigated)**
Bracket `i` covers `(lower(i) .. lower(i+1)]` via `min(base, upper)` and
`max(0, upper − lower)`. A base exactly on a threshold fills the lower bracket
and leaves the next empty (base 40 000 ⇒ 3 000, bracket 3 contributes 0). An
off-by-one on the clamp double-counts or drops the boundary cent. Pinned by
`boundary_fills_lower_bracket_only`.

### Gap-N — Single rounding event. **(mitigated)**
Marginal amounts accumulate at full precision (COBOL `WS-ACCUM` at 7 dp; the
Rust port at full `Decimal`) and the total rounds **once**, NEAREST-AWAY-FROM-ZERO.
Rounding each bracket then summing can drift a cent. Pinned by
`single_rounding_event` (`234.567 → 234.57`).

### Gap-O — Empty / partial brackets. **(mitigated)**
A base below a bracket's floor contributes 0 (the `portion = 0` branch); a base
inside a bracket contributes a partial slice. Both are exercised by the scenario
set (`in_zero_bracket`, `mid_third_bracket`).

## Contract note
A negative base is out of contract (a tax base is ≥ 0); the port returns
`SizeError` rather than a nonsensical negative tax (`negative_base_is_size_error`).

## Production gate
Regenerate `brkt_baseline.csv` from a live `cobc -x -free cobol/BRKTCALC.cbl` and
re-diff before shipping.

---

# Pre-Migration Audit — CURRCVT (Euro Conversion by Triangulation)

**Unit:** `cobol/CURRCVT.cbl` → `scirust-finmigrate::currcvt`
**Date:** 2026-07-11 · **Gate status:** Phase 1 & 2 complete vs the model baseline
(`tests/sandbox/curr_baseline.csv`). Contract: `cobol/SEMANTICS_CURR.md`.

## Why a sixth unit
A LEGALLY DEFINED conversion algorithm — Council Regulation (EC) No 1103/97 —
whose rounding steps are prescribed by law. Getting them wrong is not just a
rounding bug, it is non-compliance. It also introduces two things the prior
units did not: a per-currency **rate table** and a **variable output scale**.

### Gap-P — Triangulation is mandatory; a direct cross-rate is not equivalent. **(mitigated + evidenced)**
National→national conversion must route through the euro with an intermediate
rounding. The "obvious" `amount × rate_to / rate_from` skips the euro rounding
and is unlawful — and it really differs: 100 DEM → FRF is **335.38** lawfully but
**335.39** direct (a one-centime, legally-material divergence); 100 000 ITL → FRF
diverges too. **Mitigation:** the port triangulates; the baseline records the
direct figure per row and the equivalence test asserts ≥ 2 lawful≠direct rows.

### Gap-Q — Intermediate euro rounded to ≥ 3 dp, once. **(mitigated — GATE)**
The euro amount is rounded to not fewer than 3 decimals (this unit uses exactly
3). Rounding it to 2 dp, or leaving it unrounded, changes the result. Using
**more** than 3 dp is permitted by the regulation and can change the final minor
unit in edge cases — so the chosen intermediate precision is part of the contract
and **must be confirmed against the target system** (a live-compiler GATE item,
mirroring INTACCR Gap-6).

### Gap-R — Variable target minor unit. **(mitigated)**
The final rounding scale is the target currency's minor unit — 2 dp for
DEM/FRF/IEP, **0 dp for ITL/ESP**. A port that hard-codes 2 dp mis-rounds every
lira/peseta amount. `minor(to)` is looked up per target; `target_minor_unit_zero_for_lira`
pins the 0-dp case.

### Gap-S — Six-significant-figure rates, never truncated. **(mitigated)**
Rates are fixed/irrevocable at six significant figures (`1936.27`, `0.787564` —
both 6 sig figs) and must not be rounded or truncated further. The port stores
them as exact decimals; a rate < 1 (IEP) and a large rate (ITL) are both
exercised.

## Contract note
Both endpoints are national currencies (neither is the euro); euro-endpoint
conversion (which rounds directly to cents without the 3-dp intermediate) is a
separate case, out of scope. An unknown currency code returns
`AccrualError::UnknownCurrency` rather than a silent zero.

## Production gate
Regenerate `curr_baseline.csv` from a live `cobc -x -free cobol/CURRCVT.cbl` and
**confirm the target's intermediate-euro precision** (Gap-Q) before shipping.

## References
* Council Regulation (EC) No 1103/97 (rates, triangulation, ≥3-dp intermediate):
  https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:31997R1103
* European Commission — converting to the euro (rounding rules):
  https://economy-finance.ec.europa.eu/euro/enlargement-euro-area/adoption-fixed-euro-conversion-rate/converting-euro_en
