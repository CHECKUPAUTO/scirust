# AMORTSCH — Legacy Semantic Specification

Exact arithmetic contract of `AMORTSCH.cbl`. Reference for both the baseline
generator (`tests/sandbox/gen_amort_baseline.py`) and the Rust port
(`src/amort.rs`). Builds on `SEMANTICS.md` (INTACCR); only the *new* semantics
are spelled out in full here.

## Field model

| Field               | PICTURE           | Digits | Scale (dp) | Signed | Notes |
|---------------------|-------------------|--------|------------|--------|-------|
| `WS-ORIG-PRINCIPAL` | `S9(9)V99 COMP-3` | 11     | 2          | yes    | loan amount / starting balance |
| `WS-MONTHLY-RATE`   | `SV9(7) COMP-3`   | 7      | 7          | yes    | monthly rate, e.g. 0.0041667 |
| `WS-PAYMENT`        | `S9(7)V99 COMP-3` | 9      | 2          | yes    | fixed scheduled payment (INPUT) |
| `WS-NUM-PERIODS`    | `9(3) COMP-3`     | 3      | 0          | no     | 1 … 999 |
| `WS-INTEREST`       | `S9(9)V99 COMP-3` | 11     | 2          | yes    | per-period, ROUNDED |
| `WS-PRINC-PORTION`  | `S9(9)V99 COMP-3` | 11     | 2          | yes    | per-period principal |
| `WS-ACTUAL-PAYMENT` | `S9(9)V99 COMP-3` | 11     | 2          | yes    | payment actually taken |
| `WS-BALANCE`        | `S9(9)V99 COMP-3` | 11     | 2          | yes    | running balance |

Same COMP-3 / implied-scale / rounding rules as `SEMANTICS.md`: decimal-exact,
scale enforced on store, `ROUNDED` = NEAREST-AWAY-FROM-ZERO.

## Per-period contract (the loop body)

For `period = 1 … num_periods`, stopping early if `balance = 0`:

1. **Interest.** `COMPUTE WS-INTEREST ROUNDED = balance * monthly_rate`.
   The product is exact; a SINGLE rounding event stores it into a 2 dp field,
   NEAREST-AWAY-FROM-ZERO. This is the drift source: it happens every period on
   the *current* balance.
2. **Principal portion.** `princ = payment − interest` (exact at 2 dp). May be
   **negative** if `payment < interest` (negative amortization → balance grows).
3. **Final-payment reconciliation.** If `period = num_periods` **OR**
   `princ >= balance` (note: **greater-or-equal**, sign-aware decimal compare):
   * `princ := balance` (pay off exactly the remaining balance),
   * `actual_payment := interest + princ`.
   Otherwise `actual_payment := payment` (the scheduled payment).
4. **Post.** `balance := balance − princ`.

### Consequences that the port MUST reproduce
* **Exact payoff.** For any schedule that reaches its last period with a
  non-growing balance, the final `balance` is **exactly 0.00** — the reconciliation
  in step 3 guarantees it despite the accumulated rounding of step 1.
* **Balance never goes negative.** `princ >= balance` caps the principal at the
  balance, so a normal period cannot overshoot below zero. Only negative
  amortization (step 2) moves the balance up.
* **Early payoff.** A large payment drives `balance` to 0 before `num_periods`;
  the loop stops and the schedule has fewer rows than `num_periods`.
* **Row count** = number of periods actually executed (≤ `num_periods`).

## Divergences from legacy (documented, as in INTACCR)
* **Size error.** No `ON SIZE ERROR` is coded. A balance that grows past
  `9(9)V99` under negative amortization would silently truncate on the legacy
  platform. The port returns `AccrualError::SizeError` instead (loud stop, not
  silent corruption). Sandbox inputs are bounded so this never fires in the
  equivalence set; it is proven separately in a unit test.

## Gate note
As with INTACCR, no `cobc` is available here, so the committed baseline is
**model-derived** (Python `decimal`, encoding the contract above). Before
production, regenerate from a live target-compiler run of `AMORTSCH.cbl` and
re-diff at exact parity. See `audit_report.md` §5.
