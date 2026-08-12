# TDI-3 — Preregistration of the cross-width transfer

Freeze date: 12 July 2026.

## 1. Motivation

TDI-2 established, on width-3 systems, that a distributional recovery
profile observed up to horizon 2 provides additional predictive information
relative to a matched entropic and topological baseline.

On the width-4 holdout, TDI-2 retained a relative MSE gain, but the absolute
predictions were very poorly calibrated:

- strongly negative R²;
- negative rank correlation;
- high absolute errors.

TDI-3 explicitly tests whether multi-width training and normalized features
enable cross-width generalization without removing the incremental TDI
signal.

## 2. Main questions

### TDI-3A — Known multi-width generalization

On unseen systems of widths 3 and 4, does the TDI challenger improve the
prediction of recovery at horizon 6 relative to a matched baseline using the
same observation horizon?

### TDI-3B — Transfer to a never-observed width

Does a model trained only on widths 3 and 4 retain:

- a relative predictive gain;
- a positive rank correlation;
- absolute predictive validity;

on width-5 systems never used during training?

## 3. System generation

The systems follow the TDI-2 protocol:

- finite systems with branching transitions;
- non-empty set of successors for each state;
- uniform choice among the distinct successors;
- deterministic generation by `SplitMix64`;
- zero reference state;
- perturbation by flipping bit `width - 1`;
- dynamical action `Noop`.

Distributional propagations use exact rationals.
Conversion to `f64` occurs only for training and metrics.

## 4. Frozen populations

### Training

- width 3: 8,000 accepted systems;
- width 4: 8,000 accepted systems;
- total: 16,000 systems;
- width 3: seeds starting from `0`;
- width 4: seeds starting from `10_000_000`.

### Width-3 holdout

- 4,000 accepted systems;
- seeds starting from `1_000_000`.

### Width-4 holdout

- 4,000 accepted systems;
- seeds starting from `11_000_000`.

### Out-of-distribution width-5 holdout

- 4,000 accepted systems;
- seeds starting from `20_000_000`;
- no width-5 data may intervene in the design, training, standardization or
  selection of the model.

Seeds are consumed sequentially until the planned number of accepted systems
is obtained.

## 5. Horizons and exclusion

- observation horizon: 2;
- target horizon: 6;
- target: exact recovery `O6` between the reference and perturbed
  distributions.

Systems for which `O2 = 1` are excluded, because their future recovery is
already determined by the Markov property.

No other exclusion is allowed.

The number of examined, accepted and excluded systems must be published for
each width.

## 6. Target

The continuous target is:

O6 = sum over x of min(P6(x), Q6(x)).

It belongs to the interval `[0, 1]` and corresponds to `1 - TV(P6,Q6)`.

## 7. Matched TDI-3 baseline

The baseline may only use information available up to horizon 2.

It comprises, for the reference and perturbed trajectories:

- entropies at depths 1 and 2;
- fractions of reachable states at depths 1 and 2;
- logarithms of the numbers of paths at depths 1 and 2;
- system width as a shared feature.

Frozen normalizations:

- entropy divided by `ln(2^width)` when the denominator is non-zero;
- number of reachable states divided by `2^width`;
- number of paths transformed by `ln(1 + count)`;
- width represented by its integer value converted to `f64`.

The baseline may not use:

- any overlap between distributions;
- any data beyond horizon 2;
- any orbital feature computed on the full future;
- any information derived from the target.

## 8. TDI-3 features

The challenger uses all the baseline features and adds:

- exact recovery `O1`;
- exact recovery `O2`;
- variation `O2 - O1`.

These three features are already dimensionless and do not depend directly on
the number of states.

The width is available identically to the baseline and the challenger.

## 9. Predictive model

Both models use exactly the same algorithm:

- deterministic ridge regression;
- standardization computed exclusively on the training set;
- unpenalized constant term;
- regularization coefficient fixed at `lambda = 1`;
- normal equations solved deterministically;
- predictions bounded in `[0, 1]`.

No hyperparameter may be tuned after computing the holdouts.

No post-hoc calibration will be fitted on the holdouts.

## 10. Metrics

For each width and for the combined holdout of widths 3 and 4:

- MSE;
- MAE;
- R²;
- Spearman rank correlation;
- mean bias: mean prediction minus mean target;
- calibration in the large: mean predicted and mean observed;
- calibration slope and intercept obtained by regressing the target on the
  prediction.

The metrics are computed separately for the baseline and the challenger.

## 11. Uncertainty

A deterministic paired bootstrap of 2,000 replications is applied:

- to the width-3 holdout;
- to the width-4 holdout;
- to the combined width-3 and width-4 holdout;
- to the out-of-distribution width-5 holdout.

It measures:

- `MSE_baseline - MSE_TDI`;
- `MAE_baseline - MAE_TDI`.

Bootstrap seed:

`0x5444_4933_494E_5445`.

The intervals use the empirical 2.5 % and 97.5 % quantiles.

## 12. Primary criterion TDI-3A

TDI-3A is declared successful only if the following conditions are all
satisfied on the combined width-3 and width-4 holdout:

1. relative MSE reduction greater than or equal to 5 %;
2. lower bound of the 95 % CI of the MSE improvement strictly positive;
3. lower bound of the 95 % CI of the MAE improvement strictly positive;
4. observed MSE improvement strictly positive separately at width 3;
5. observed MSE improvement strictly positive separately at width 4;
6. challenger Spearman strictly positive separately at widths 3 and 4;
7. challenger R² strictly positive separately at widths 3 and 4.

## 13. Transfer criterion TDI-3B

TDI-3B is declared successful at width 5 only if:

1. the observed MSE improvement is strictly positive;
2. the lower bound of the 95 % CI of the MSE improvement is strictly
   positive;
3. the lower bound of the 95 % CI of the MAE improvement is strictly
   positive;
4. the challenger R² is strictly positive;
5. the challenger Spearman is strictly positive;
6. the absolute value of the challenger mean bias is lower than that of
   the baseline.

A positive relative reduction accompanied by a negative R² or Spearman will
not be presented as valid generalization.

## 14. Secondary analyses

The following will be published without modifying the primary verdict:

- results separated by width;
- target distribution;
- prediction distribution;
- proportion of predictions bounded at 0 or 1;
- standardized coefficients of both models;
- results by target deciles;
- results by `O2` deciles;
- feature correlation matrices;
- execution times and number of rejected systems.

## 15. Mandatory software controls

Before the final evaluation:

- deterministic generation verified by tests;
- exact lengths of the feature vectors;
- absence of holdout data in the standardization;
- absence of overlap of the seed ranges;
- finite normalized values;
- predictions within `[0, 1]`;
- reproducible bootstrap;
- verification of the hash of this preregistration in the CI.

## 16. Prohibitions after freeze

After creation of the hash:

- no change of populations;
- no change of seeds;
- no change of horizons;
- no change of features;
- no change of lambda;
- no change of success thresholds;
- no examination of holdout results before complete freeze of the
  evaluation code.

Any bug correction discovered after the first computation must be
documented, tested and accompanied by a new complete execution.

## 17. Negative interpretation

The result is negative for TDI-3A if its primary criterion fails.

The result is negative for TDI-3B if at least one condition of the width-5
transfer fails.

The TDI-3A and TDI-3B verdicts are independent and must be published
separately.

A TDI-3A success combined with a TDI-3B failure would mean that the TDI
signal is usable in a known multi-width population, without evidence of
transfer to a new size.

## 18. Planned deliverables

- binary `tdi-interwidth-continuous`;
- script `scripts/reproduce-tdi3.sh`;
- deterministic log in `results/`;
- report `docs/TDI-3-INTERWIDTH-RESULTS.md`;
- SHA-256 hash of the preregistration;
- SHA-256 hash of the reference result;
- unit tests and CI controls;
- dedicated Git release if the evaluation is completed.
