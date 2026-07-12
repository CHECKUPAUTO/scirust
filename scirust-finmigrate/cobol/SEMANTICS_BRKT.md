# BRKTCALC — Legacy Semantic Specification

Exact contract of `BRKTCALC.cbl`. Reference for
`tests/sandbox/gen_brkt_baseline.py` and `src/brktcalc.rs`.

## Field model

| Field       | PICTURE              | Notes |
|-------------|----------------------|-------|
| `WS-BASE`   | `S9(9)V99 COMP-3`    | taxable base, 2 dp |
| `WS-TAX`    | `S9(9)V99 COMP-3`    | computed tax, ROUNDED to 2 dp |
| `WS-LOWER`  | `S9(9)V99 COMP-3`    | bracket inclusive floor |
| `WS-RATE`   | `SV9(5) COMP-3`      | bracket marginal rate, 5 dp |
| `WS-ACCUM`  | `S9(13)V9(7) COMP-3` | high-precision accumulator (no rounding) |

## The bracket table (the contract)

| # | Lower (inclusive) | Marginal rate |
|---|-------------------|---------------|
| 1 | 0.00              | 0 %           |
| 2 | 10 000.00         | 10 %          |
| 3 | 40 000.00         | 22 %          |
| 4 | 85 000.00         | 24 %          |
| 5 | 165 000.00        | 32 % (open top) |

## The computation

For each bracket `i = 1 … 5`:

* `upper = LOWER(i+1)` for `i < 5`, else `upper = base` (the last bracket is
  unbounded); then clamp `upper = min(upper, base)`.
* `portion = max(0, upper − LOWER(i))`.
* `accum += portion × RATE(i)` — accumulated at **full precision**.

Finally `tax = ROUND(accum)` — a **single** NEAREST-AWAY-FROM-ZERO rounding into
the 2 dp field.

`tax = Σ_i  max(0, min(base, LOWER(i+1)) − LOWER(i)) × RATE(i)`  (LOWER(6) ≜ ∞).

### Load-bearing properties (the migration risks)
* **Marginal, not flat.** Each rate hits only its bracket's slice. A flat
  computation `base × RATE(top)` over-taxes wildly (e.g. base 100 000 → flat 32%
  = 32 000; correct marginal = 16 500 = 3 000 + 9 900 + 3 600). Pinned by
  `flat_would_be_wrong`.
* **Boundary inclusivity.** Bracket `i` covers `(LOWER(i) … LOWER(i+1)]` via the
  `min/max` clamps; a base exactly on a threshold fills the lower bracket and
  leaves the next empty. Pinned by an `exactly_on_boundary` case.
* **One rounding event.** Marginals accumulate at full precision (`WS-ACCUM`,
  7 dp) and the total rounds once. Rounding per bracket then summing can drift a
  cent. Rates here are ≤ 5 dp and portions are 2 dp, so each product is ≤ 7 dp
  exact; the accumulator holds them exactly and only the final store rounds.
* **Empty / partial brackets.** A base below a bracket's floor contributes 0
  (the `portion = 0` branch); a base inside a bracket contributes a partial slice.

## Divergences / gate
* Size error handled as elsewhere (loud `SizeError`, bounded out of the set).
* Negative base is out of contract (a tax base is ≥ 0); the port rejects it as a
  `SizeError` rather than producing a nonsensical negative tax.
* Baseline is **model-derived** (no `cobc`); regenerate from a live
  `cobc -x -free cobol/BRKTCALC.cbl` before shipping.

## References
* Marginal (graduated) taxation — bracket slices taxed at their own rate:
  https://en.wikipedia.org/wiki/Progressive_tax
