# CURRCVT — Legacy Semantic Specification

Exact contract of `CURRCVT.cbl`. Reference for
`tests/sandbox/gen_curr_baseline.py` and `src/currcvt.rs`. Implements
national→national euro conversion by triangulation, per Council Regulation
(EC) No 1103/97.

## Conversion rates (1 euro = national currency, six significant figures)

| Currency | Code | Rate (1 EUR =) | Minor unit (dp) |
|----------|------|----------------|-----------------|
| Deutsche Mark   | DEM | 1.95583   | 2 |
| French franc    | FRF | 6.55957   | 2 |
| Italian lira    | ITL | 1936.27   | 0 |
| Spanish peseta  | ESP | 166.386   | 0 |
| Irish punt      | IEP | 0.787564  | 2 |

* Rates are **fixed and irrevocable**, quoted to **six significant figures**
  (note: significant figures, not decimal places — `1936.27` and `0.787564`
  are both 6 sig figs). They MUST NOT be rounded or truncated further when
  converting.

## The algorithm (national A → national B)

```
euro   = ROUND( amount_A / rate(A), 3 )        # intermediate, >= 3 dp
result = ROUND( euro * rate(B), minor(B) )      # target minor unit
```

Both `ROUND`s are NEAREST-AWAY-FROM-ZERO. `minor(B)` is 2 for most currencies,
**0 for ITL and ESP**.

## Load-bearing properties (the migration risks)

### Gap-P — Triangulation is mandatory; a direct cross-rate is not equivalent.
The lawful result routes through the euro with an intermediate rounding. A naive
`ROUND(amount * rate(B) / rate(A), minor(B))` skips the 3-dp euro rounding and
can differ by a minor unit. The sandbox computes both and records the divergence;
the port implements only the triangulated form.

### Gap-Q — Intermediate euro rounded to ≥ 3 dp, exactly once.
Rounding the euro to 2 dp, or not rounding it at all, changes the final result.
The unit rounds to exactly 3 dp (the legal minimum). *Using more than 3 dp is
permitted by the regulation and can yield a different final minor unit in edge
cases — so the chosen intermediate precision is part of the contract and must be
confirmed against the target system (a GATE item, like INTACCR Gap-6).*

### Gap-R — Variable target minor unit.
The final rounding scale depends on the target currency: 2 dp for DEM/FRF/IEP,
**0 dp for ITL/ESP**. A port that hard-codes 2 dp mis-rounds every lira/peseta
amount. `minor(B)` is looked up per target.

### Gap-S — Rates below 1 and large integer rates.
`IEP = 0.787564` (< 1) and `ITL = 1936.27` (large) both stress the division and
multiplication; the rate is an exact decimal and is never truncated.

## Divergences / gate
* Both endpoints are national currencies (neither is the euro); euro-endpoint
  conversion (which rounds directly to cents without the 3-dp intermediate) is a
  separate case, out of this unit's scope.
* Size error handled as elsewhere (loud `SizeError`, bounded out of the set).
* Baseline is **model-derived** (no `cobc`); regenerate from a live
  `cobc -x -free cobol/CURRCVT.cbl` before shipping — and confirm the target's
  intermediate-euro precision (Gap-Q).

## References
* Council Regulation (EC) No 1103/97 — conversion rates (6 significant figures),
  no rounding/truncation of the rate, triangulation, and the ≥3-decimal
  intermediate: https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:31997R1103
* European Commission — converting to the euro (rounding rules):
  https://economy-finance.ec.europa.eu/euro/enlargement-euro-area/adoption-fixed-euro-conversion-rate/converting-euro_en
