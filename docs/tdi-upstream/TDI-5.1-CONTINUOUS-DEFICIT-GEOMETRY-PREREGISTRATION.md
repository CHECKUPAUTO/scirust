# TDI-5.1 Continuous Deficit Geometry

## Preregistration

### Status

TDI-5.1 is a new experiment derived from the frozen TDI-5 design. It is not a
repair, continuation, completion, or reinterpretation of TDI-5.

No full TDI-5.1 experiment may start before all of the following have been
committed:

1. this preregistration;
2. the TDI-5.1 evaluator SHA-256 manifest;
3. the TDI-5.1 scientific-code SHA-256 manifest.

Short deterministic unit tests and bounded smoke tests are permitted before
that commit. The full scientific campaign is not.

## 1. TDI-5 Failure Disposition

The frozen TDI-5 evaluator became non-terminating at width 6. The mathematical
successor-set space cardinality at width 6 is:

```text
2^(2^6) = 2^64 = 18_446_744_073_709_551_616
```

This value cannot be represented as `u64` because `u64::MAX` is `2^64 - 1`.
TDI-5 attempted the equivalent of `1_u64.checked_shl(64)`, which returns
`None`. The failure was masked because exact-analysis errors were converted
into absence of a candidate record, and generation then retried without a
deterministic termination budget.

Any partial TDI-5 output is non-final and excluded from TDI-5.1 analysis.

## 2. Cardinality Model

TDI-5.1 represents successor-set-space cardinalities with `u128` behind an
explicit cardinality status:

```text
Exact(value)
TooLarge { width, exponent }
Invalid { width, reason }
```

The exact successor-set-space cardinalities for widths 0 through 6 are:

| Width | State count `2^w` | Successor-set space `2^(2^w)` |
|---:|---:|---:|
| 0 | 1 | 2 |
| 1 | 2 | 4 |
| 2 | 4 | 16 |
| 3 | 8 | 256 |
| 4 | 16 | 65 536 |
| 5 | 32 | 4 294 967 296 |
| 6 | 64 | 18 446 744 073 709 551 616 |

The width-6 full space is represented exactly as `u128`. The non-empty
successor-mask count is computed by checked subtraction in `u128` and then
converted to `u64`, yielding exactly `u64::MAX`.

All shifts and exponentiation-like operations must be checked. Unsupported
runtime widths must produce typed errors and must not be sampled.

## 3. Populations and Seeds

TDI-5.1 preserves the TDI-5 scientific population plan but assigns it to the
new experiment only.

| Population | Width | Accepted records | Initial seed | Max attempts | No-progress threshold |
|---|---:|---:|---:|---:|---:|
| training | 3 | 15 000 | 60 000 000 | 960 000 | 25 000 |
| holdout | 3 | 5 000 | 61 000 000 | 320 000 | 25 000 |
| training | 4 | 15 000 | 70 000 000 | 1 440 000 | 50 000 |
| holdout | 4 | 5 000 | 71 000 000 | 480 000 | 50 000 |
| OOD principal | 5 | 10 000 | 80 000 000 | 1 280 000 | 75 000 |
| OOD extreme | 6 | 5 000 | 90 000 000 | 1 280 000 | 100 000 |

The maximum-attempt budgets are deterministic per width and target count:

| Width | Attempt multiplier | No-progress threshold |
|---:|---:|---:|
| 3 | 64 | 25 000 |
| 4 | 96 | 50 000 |
| 5 | 128 | 75 000 |
| 6 | 256 | 100 000 |

A seed denotes one candidate. A valid rejected candidate consumes its seed.
Accepted records, rejected candidates, total attempts, final exclusive seeds,
maximum-attempt budgets, and no-progress thresholds must be printed in the raw
output metadata.

## 4. Termination Rules

Generation for each population stops at the first of these deterministic
conditions:

1. the requested accepted record count is reached;
2. the maximum-attempt budget for that population is exhausted;
3. the no-progress threshold is reached without any accepted record;
4. a structural, arithmetic, cardinality, unsupported-width, seed-range, or
   dynamic-analysis evaluator error occurs.

Conditions 2 through 4 are explicit failures. They must include width, seed,
attempt index, failure category, accepted count, excluded count, target count,
maximum attempts, and no-progress threshold. They are not scientific
exclusions.

TDI-5.1 must never use an unbounded retry loop.

## 5. Error Semantics

TDI-5.1 distinguishes:

- `Accepted(record)`;
- `Rejected(reason)`, for valid preregistered sample exclusions;
- `EvaluationError`, for arithmetic, structural, cardinality,
  unsupported-width, seed-range, or dynamic-analysis failures;
- `GenerationError`, for evaluator failures, attempt-budget exhaustion, and
  no-progress termination.

Structural and arithmetic failures must never be transformed into
`Ok(None)`, absence of a record, or a valid rejection.

## 6. Horizons and Targets

Observation horizon:

```text
h_obs = 2
```

Target horizons:

```text
H = {3, 4, 5, 6, 8}
```

The primary confirmatory target remains `U_6`, where:

```text
U_h = -log2(1 - O_h)
```

The secondary trajectory targets are `U_3`, `U_4`, `U_5`, and `U_8`.

## 7. Exclusions

A candidate is validly rejected only if one of the preregistered scientific
exclusion conditions is met:

1. exact recovery at observation horizon `O_2 = 1`;
2. exact recovery at any target horizon;
3. non-finite or out-of-range observation geometry;
4. non-finite, negative, or out-of-range transformed target geometry;
5. non-finite feature geometry after otherwise valid exact analysis.

Evaluator failures are not exclusions.

## 8. Predictors and Models

The baseline contains the same 13 structural and entropic variables used by
TDI-5. The TDI predictors are:

```text
O_1, O_2, O_2 - O_1
```

The four model layouts are:

| Layout | Variables |
|---|---|
| M0 | baseline |
| M1 | baseline + `O_1` |
| M2 | baseline + `O_1` + `O_2` |
| M3 | baseline + `O_1` + `O_2` + `O_2 - O_1` |

Ridge regularization is fixed at `lambda = 1.0`.

## 9. Statistics and Success Criteria

The primary analysis fits models on the combined width-3 and width-4 training
population and evaluates `U_6` on untouched width-3 and width-4 holdouts.

The same model family is evaluated on OOD width 5 and extreme OOD width 6.

Metrics are:

- mean squared error;
- mean absolute error;
- `R^2`;
- Spearman correlation;
- prediction bias;
- calibration intercept and slope;
- lower-bound and upper-bound clipping fractions for reconstructed overlaps.

Bootstrap intervals use 2 000 deterministic paired bootstrap replicates with
seed `0x5444_4935_4344_4745`.

The confirmatory criteria are the same as TDI-5 but are evaluated only on
TDI-5.1 populations:

- TDI-5.1A: M3 improves standardized `U_6` MSE on combined width-3/4 holdout
  by at least 10%, with positive bootstrap lower bounds for combined and each
  in-distribution width and non-worse bias within the preregistered tolerance.
- TDI-5.1B: M3 improves width-5 OOD standardized `U_6` MSE by at least 20%,
  has positive bootstrap lower bound, positive Spearman, non-worse Spearman,
  better `R^2`, lower absolute bias, and improves reconstructed-overlap MSE and
  MAE.
- TDI-5.1C: M3 improves width-6 standardized `U_6` MSE, has positive bootstrap
  lower bound, positive and non-worse Spearman, non-worse absolute bias, and
  improves reconstructed-overlap MSE.
- TDI-5.1D: at least three of the four secondary horizons improve on combined
  width-3/4 holdout, `U_8` improves, no secondary relative reduction is below
  -5%, and the average secondary relative reduction is positive.

## 10. Determinism

All candidate generation, model fitting, bootstrap sampling, iteration order,
and diagnostics are deterministic functions of the preregistered constants.

Wall-clock timestamps may be recorded as reproduction metadata. They must not
affect sampling, model fitting, stopping, exclusions, or success criteria.

The mathematical acceptance criterion must not be modified to produce enough
records.
