# TDI-2 — Results of the preregistered continuous evaluation

Execution date: 12 July 2026<br>
Branch: `tdi-2-branching`<br>
Preregistration: `3c8f54d`<br>
Frozen implementation: `ccbfc31`

## Prior technical correction

The first execution attempt failed before producing any results due to an
intermediate overflow when comparing two `u128` rationals on the width-4
holdout.

The cross-product comparison was replaced by an exact comparison based on
the successive quotients of the Euclidean algorithm.

This correction:

- changes no population;
- changes no horizon;
- changes no feature;
- changes no model;
- changes no metric;
- consults no holdout result;
- adds a test with rationals close to `u128::MAX`.

## Software validation

- formatting: passed;
- workspace tests: passed;
- `tdi-core` tests: 38 passed;
- Clippy with `-D warnings`: passed;
- Git verification: passed;
- benchmark execution: passed.

## Population actually analyzed

| Set | Width | Accepted systems | Excluded at horizon 2 |
|---|---:|---:|---:|
| Training | 3 | 12,000 | 55 |
| Main holdout | 3 | 4,000 | 24 |
| Out-of-distribution holdout | 4 | 4,000 | 0 |

The exclusions correspond only to systems whose distributions were already
exactly identical at observation horizon 2.

## Main holdout — width 3

### Paired baseline

- MSE: `0.001816873`
- MAE: `0.018304843`
- R²: `0.393556075`
- Spearman: `0.411338780`

### Baseline + TDI-2

- MSE: `0.001579275`
- MAE: `0.017156635`
- R²: `0.472862476`
- Spearman: `0.557910624`

### Observed gain

- MSE improvement: `0.000237598`
- relative MSE reduction: `13.077285 %`
- MAE improvement: `0.001148208`

### Deterministic paired bootstrap — 2,000 replications

- 95 % CI of MSE improvement:
  `[0.000162544, 0.000315743]`
- median MSE improvement:
  `0.000236356`
- 95 % CI of MAE improvement:
  `[0.000788787, 0.001495130]`
- median MAE improvement:
  `0.001151804`

## Primary preregistered verdict

The primary criterion is **passed**:

1. relative MSE reduction greater than 5 %;
2. lower bound of the 95 % MSE CI strictly positive;
3. lower bound of the 95 % MAE CI strictly positive.

The gain of TDI-2 over the paired baseline is therefore statistically robust
on the main width-3 holdout.

## Out-of-distribution holdout — width 4

### Paired baseline

- MSE: `0.157274396`
- MAE: `0.372795169`
- R²: `-109506.813463054`
- Spearman: `-0.507018418`

### Baseline + TDI-2

- MSE: `0.147526127`
- MAE: `0.366151366`
- R²: `-102719.239290951`
- Spearman: `-0.489819920`

### Observed gain

- MSE improvement: `0.009748269`
- relative MSE reduction: `6.198256 %`
- MAE improvement: `0.006643803`

The preregistered minimal criterion of the out-of-distribution holdout only
required a positive MSE improvement. It is therefore formally passed.

## Major scientific caveat

The out-of-distribution confirmation must not be presented as good absolute
generalization.

Despite a relative improvement over the baseline:

- both models have extremely negative R²;
- both Spearman correlations are negative;
- the absolute errors are very high;
- a model trained at width 3 is manifestly poorly calibrated at width 4.

The honest conclusion is therefore:

- **confirmatory success on the main width-3 holdout**;
- **relative TDI-2 gain also observed at width 4**;
- **absence of absolute predictive validity out of distribution**;
- **need for a separate cross-width transfer study**.

## Conclusion

TDI-2 provides incremental predictive information beyond entropy and
topology matched at the same observation horizon.

The main result is positive and supported by the bootstrap intervals.
It establishes an incremental prospective signal in the width-3 population.

It does not yet establish a universal law independent of system size.
