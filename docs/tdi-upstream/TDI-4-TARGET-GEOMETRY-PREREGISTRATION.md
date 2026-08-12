# TDI-4 — Preregistration of the Target Geometry evaluation

## 1. Status

This document is established before:

- the implementation of the TDI-4 evaluator;
- the generation of the TDI-4 populations;
- the observation of the TDI-4 metrics;
- any adaptation of the success thresholds.

The TDI-3 results are known and motivate the formulation of TDI-4, but no
TDI-3 population will be reused to train or evaluate TDI-4.

## 2. Motivation

TDI-3 showed:

- a favorable predictive signal at width 3;
- a deterioration of errors at width 4;
- a large reduction of bias and MSE at width 5;
- negative R² and Spearman values at width 5;
- a growing concentration of the target `O₆` near `1`.

The observed mean of `O₆` was notably close to:

- `0.9836` at width 3;
- `0.9990` at width 4;
- `0.99988` at width 5.

A direct regression of `O₆` therefore confounds two phenomena:

1. the exact attainment of `O₆ = 1`;
2. the size of the residual deficit when `O₆ < 1`.

TDI-4 tests a two-part target geometry in order to separate these
phenomena.

## 3. Hypotheses

### H4.1 — Target geometry

A two-part representation of recovery at horizon 6 is more stable across
widths than a direct regression of `O₆`.

### H4.2 — Added value of the TDI variables

With identical model architecture and target geometry, the TDI variables
observed at horizon 2 improve prediction relative to the baseline variables
alone.

### H4.3 — Out-of-distribution transfer

The gain of the challenger trained on widths 3 and 4 remains positive on an
unseen width-5 population.

## 4. Systems and horizons

For each system:

- reference state: zero state of the considered width;
- perturbation: flip of the last node;
- future action: `Noop`;
- observation horizon: `2`;
- outcome horizon: `6`.

The distributions are propagated with the exact rational arithmetic based
on `BigUint`.

## 5. Exclusion

A system is excluded if the distributions are already exactly identical at
the observation horizon:

`O₂ = 1`.

This rule is identical to that of TDI-3 and avoids a direct logical leak of
the target.

Systems such that `O₆ = 1` are retained. They constitute precisely the
binary component of the new problem.

No other exclusion depending on the target is allowed.

## 6. Populations and seeds

The seed ranges are entirely new.

| Population | Width | Retained size | First seed |
|---|---:|---:|---:|
| training | 3 | 10,000 | 30,000,000 |
| holdout | 3 | 5,000 | 31,000,000 |
| training | 4 | 10,000 | 40,000,000 |
| holdout | 4 | 5,000 | 41,000,000 |
| OOD holdout | 5 | 5,000 | 50,000,000 |

Generation continues beyond the first seed until the preregistered number
of retained systems is obtained.

The training populations of widths 3 and 4 are combined.

The holdouts are never used to:

- choose the variables;
- choose the transformations;
- tune the thresholds;
- select the model;
- adjust `lambda`.

## 7. Explanatory variables

### 7.1 Matched baseline

The baseline keeps exactly the 13 variables used in TDI-3:

- normalized entropies;
- normalized numbers of reachable states;
- logarithms of the numbers of paths;
- width features.

The width is included as an explanatory variable in the same way as in
TDI-3.

### 7.2 TDI-4 challenger

The challenger contains the 13 baseline variables and the three already
defined TDI variables:

- `O₁`;
- `O₂`;
- `O₂ - O₁`.

No additional variable may be added after the freeze.

## 8. Two-part target geometry

Let:

`O₆ ∈ [0, 1]`

be the exact recovery at horizon 6.

### 8.1 Exact-recovery component

The binary target is:

`Z = 1` if `O₆ = 1`, otherwise `Z = 0`.

A first model predicts:

`p = P(Z = 1 | X)`.

The linear prediction is bounded in `[0, 1]`.

### 8.2 Conditional component

For the only systems such that `O₆ < 1`, the deficit is:

`D = 1 - O₆`.

The conditional target is:

`U = -log₂(D)`.

A large value of `U` indicates a very small deficit.

`U` is standardized from the mean and standard deviation computed
exclusively on the training systems that are not fully recovered.

### 8.3 Reconstruction

The final recovery prediction is:

`Ô₆ = 1 - (1 - p̂) × 2^(-Û)`.

The reconstructed value is bounded in `[0, 1]`.

The same procedure is applied to the baseline and the challenger.

## 9. Models

Both components use a deterministic ridge regression.

Fixed parameter:

`lambda = 1`.

Two heads are trained for each family of variables:

1. binary head on `Z`;
2. conditional head on `U`, only for the systems where `Z = 0`.

No hyperparameter search is allowed.

The floating-point accumulation order remains deterministic.

## 10. Main composite loss

The main loss is:

`L = 0.5 × Brier(Z, p̂) + 0.5 × MSE(U_std, Û_std | Z = 0)`.

The second component is computed only on the not fully recovered
observations.

If an evaluation population contains no not fully recovered system, the
evaluation must stop with an explicit error. No artificial zero term will
be substituted.

An improvement is defined by:

`ΔL = L_baseline - L_challenger`.

A positive value favors TDI-4.

## 11. Reported metrics

For each width and for the combined holdout, the following will be reported:

### Binary head

- Brier score;
- observed rate of exact recovery;
- mean of `p̂`;
- calibration intercept;
- calibration slope;
- fraction of predictions bounded at `0`;
- fraction of predictions bounded at `1`.

### Conditional head

- MSE on standardized `U`;
- MAE on standardized `U`;
- conditional R²;
- conditional Spearman.

### Reconstruction of `O₆`

- MSE;
- MAE;
- R²;
- Spearman;
- mean bias;
- observed mean;
- predicted mean;
- calibration intercept;
- calibration slope.

### Main loss

- composite loss `L`;
- absolute improvement `ΔL`;
- relative reduction of `L`.

## 12. Secondary comparator

A secondary comparator reproduces, on the new populations, the direct ridge
regression of `O₆` used in TDI-3.

This comparator serves only to determine whether the two-part geometry
improves stability.

It cannot replace the main baseline and does not participate in the TDI-4A
or TDI-4B verdict.

## 13. Bootstrap

The bootstrap is:

- paired;
- deterministic;
- performed separately for each population;
- composed of 2,000 replications.

Fixed seed:

`0x5444_4934_4745_4F4D`.

95 % percentile intervals are computed for:

- the composite loss improvement;
- the Brier score improvement;
- the conditional MSE improvement;
- the reconstructed MSE improvement;
- the reconstructed MAE improvement.

## 14. Primary criterion TDI-4A

TDI-4A is declared **PASSED** only if all the following conditions are
satisfied on the width-3 and width-4 holdouts:

1. relative reduction of the combined composite loss greater than or equal
   to `5 %`;
2. lower bound of the 95 % CI of the combined `ΔL` strictly positive;
3. lower bound of the 95 % CI of the combined Brier improvement strictly
   positive;
4. lower bound of the 95 % CI of `ΔL` at width 3 strictly positive;
5. lower bound of the 95 % CI of `ΔL` at width 4 strictly positive;
6. positive point improvement of the combined reconstructed MSE;
7. positive point improvement of the combined reconstructed MAE;
8. positive conditional Spearman for the challenger in both widths.

The failure of a single condition leads to the verdict **FAILED**.

## 15. Transfer criterion TDI-4B

TDI-4B is declared **PASSED** only if all the following conditions are
satisfied on the unseen width-5 holdout:

1. lower bound of the 95 % CI of `ΔL` strictly positive;
2. relative reduction of the reconstructed MSE greater than or equal to
   `5 %`;
3. lower bound of the 95 % CI of the reconstructed MSE improvement strictly
   positive;
4. lower bound of the 95 % CI of the Brier score improvement strictly
   positive;
5. strictly positive conditional Spearman of the challenger;
6. conditional Spearman of the challenger greater than or equal to that of
   the baseline;
7. absolute reconstructed bias of the challenger lower than that of the
   baseline.

The failure of a single condition leads to the verdict **FAILED**.

## 16. Secondary analyses

The following analyses are reported but do not modify the verdicts:

- results separated for each head;
- comparison with the direct TDI-3 regression;
- model coefficients;
- exact recovery rate by width;
- distribution of `U`;
- saturation of the predictions;
- results by deciles of observed deficit.

No secondary analysis may be requalified as a primary criterion after
observation of the results.

## 17. Error policy

An implementation error or a numerical limit discovered before the
production of metrics may be corrected if:

- the initial failure is preserved;
- the cause is documented;
- the criteria and populations are not modified;
- the corrected code is refrozen before a new execution.

Once the metrics are produced, no modification of the protocol, the
evaluator or the criteria is allowed for this experiment.

## 18. Expected verdicts in the output

The evaluator must produce exactly two final lines:

`CRITÈRE PRINCIPAL TDI-4A : RÉUSSI|ÉCHOUÉ`

`CRITÈRE TRANSFERT TDI-4B : RÉUSSI|ÉCHOUÉ`
