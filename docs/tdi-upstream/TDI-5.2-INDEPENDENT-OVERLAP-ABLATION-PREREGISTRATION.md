# TDI-5.2 — Independent Early-Overlap Ablation and Seed-Block Replication

## Preregistration

## 1. Experimental status

TDI-5.2 is a new confirmatory experiment derived from the completed
TDI-5.1 analysis.

TDI-5.1 remains frozen. It must not be modified, continued under the same
identifier, or reinterpreted as TDI-5.2.

Frozen TDI-5.1 identity:

    commit:
    d9286eee25dd5b0735fa79fd64395e10ea4c93ac

    result SHA-256:
    f66666dba8a0a82a13aaf62e018a5500e57452db50eeff526b68b98f797a5895

The post-hoc TDI-5.1 analysis identified an exact linear redundancy:

    delta_O = O_2 - O_1

A linear model already containing O_1 and O_2 receives no new independent
information dimension from delta_O.

TDI-5.2 therefore separates the independent predictive contributions of
O_1 and O_2.

No full TDI-5.2 scientific execution may begin before all of the following
are committed and frozen:

1. this preregistration;
2. the final evaluator;
3. the evaluator SHA-256 manifest;
4. the scientific-code SHA-256 manifest;
5. the deterministic reproduction script;
6. the dedicated CI workflow;
7. bounded unit and termination tests.

## 2. Research questions

TDI-5.2 evaluates:

1. whether O_1 and O_2 jointly improve future-deficit prediction beyond
   the structural and entropic baseline;
2. whether O_2 contributes predictive information conditional on the
   baseline and O_1;
3. whether O_1 contributes predictive information conditional on the
   baseline and O_2;
4. whether these conclusions replicate across independent seed blocks;
5. whether the predictive signal transfers to widths 5 and 6;
6. whether the result persists across secondary target horizons.

## 3. Frozen dynamical construction

Unless explicitly changed in this document, TDI-5.2 preserves the frozen
TDI-5.1 definitions for:

- system generation;
- exact branching analysis;
- reference state;
- perturbation;
- observation geometry;
- target geometry;
- scientific exclusions;
- width-6 exact cardinality;
- deterministic iteration and accumulation order.

Observation horizon:

    h_obs = 2

Target horizons:

    H = {3, 4, 5, 6, 8}

Primary target:

    U_6

Transformed target:

    U_h = -log2(1 - O_h)

Exact width-6 successor-set-space cardinality:

    2^(2^6) = 2^64
            = 18_446_744_073_709_551_616

Exact non-empty width-6 successor-mask count:

    u64::MAX

Unsupported widths and arithmetic, structural or dynamic-analysis errors
must produce typed failures.

They must never be converted into scientific exclusions.

## 4. Predictors

The baseline contains the same 13 structural and entropic variables used
by TDI-5.1.

The confirmatory early-overlap predictors are:

    O_1
    O_2

Confirmatory model layouts:

| Layout | Variables |
|---|---|
| B0 | baseline |
| B1 | baseline plus O_1 |
| B2 | baseline plus O_2 |
| B12 | baseline plus O_1 and O_2 |

Exploratory layout:

| Layout | Variables |
|---|---|
| BD | baseline plus O_2 minus O_1 |

No confirmatory model may simultaneously contain O_1, O_2 and O_2 minus
O_1.

BD cannot determine any confirmatory success criterion.

## 5. Model family and preprocessing

The confirmatory model family is ridge regression with:

    lambda = 1.0

For each independent seed block:

1. width-3 and width-4 training populations are combined;
2. preprocessing is fitted only on that block's training population;
3. target standardization is fitted only on training targets;
4. holdout and OOD records never affect model fitting;
5. one model is fitted for every target horizon and feature layout;
6. iteration and floating-point accumulation order are deterministic;
7. all layouts use the same target scaler for a given block and horizon.

Standardized U space is the primary confirmatory domain.

Reconstructed O-space metrics are secondary diagnostics because O values
may saturate close to one.

## 6. Independent seed blocks

Three deterministic non-overlapping seed blocks are used.

Each block contains:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |
| OOD principal | 5 | 10,000 |
| OOD extreme | 6 | 5,000 |

Accepted records per block:

    55,000

Total accepted records:

    165,000

### Block A

| Population | Initial seed |
|---|---:|
| training w3 | 160,000,000 |
| holdout w3 | 170,000,000 |
| training w4 | 180,000,000 |
| holdout w4 | 190,000,000 |
| OOD w5 | 200,000,000 |
| OOD w6 | 210,000,000 |

### Block B

| Population | Initial seed |
|---|---:|
| training w3 | 260,000,000 |
| holdout w3 | 270,000,000 |
| training w4 | 280,000,000 |
| holdout w4 | 290,000,000 |
| OOD w5 | 300,000,000 |
| OOD w6 | 310,000,000 |

### Block C

| Population | Initial seed |
|---|---:|
| training w3 | 360,000,000 |
| holdout w3 | 370,000,000 |
| training w4 | 380,000,000 |
| holdout w4 | 390,000,000 |
| OOD w5 | 400,000,000 |
| OOD w6 | 410,000,000 |

The evaluator must verify at runtime that all consumed seed ranges are
pairwise disjoint.

## 7. Generation budgets

Deterministic generation limits:

| Width | Attempt multiplier | No-progress threshold |
|---:|---:|---:|
| 3 | 64 | 25,000 |
| 4 | 96 | 50,000 |
| 5 | 128 | 75,000 |
| 6 | 256 | 100,000 |

Generation terminates at the first of:

1. requested accepted count reached;
2. attempt budget exhausted;
3. no-progress threshold reached;
4. typed evaluator error.

Conditions 2 through 4 are explicit failures.

Every failure must print:

- seed block;
- population;
- width;
- seed;
- attempt index;
- accepted count;
- excluded count;
- target count;
- maximum-attempt budget;
- no-progress threshold;
- typed failure category;
- diagnostic message.

## 8. Scientific exclusions

A candidate may be rejected only for:

1. exact recovery at observation horizon;
2. exact recovery at a target horizon;
3. invalid observation-overlap geometry;
4. invalid target-overlap geometry;
5. invalid transformed-target geometry;
6. non-finite predictor geometry after otherwise valid analysis.

The evaluator must print rejection counts by:

- seed block;
- population;
- rejection reason.

Evaluator failures are not scientific exclusions.

## 9. Metrics

For every block, population, horizon and model layout, print:

- MSE;
- MAE;
- R-squared;
- Spearman correlation;
- prediction bias;
- observed mean;
- predicted mean;
- calibration intercept;
- calibration slope;
- lower clipping fraction;
- upper clipping fraction.

For every confirmatory comparison, print:

- absolute MSE difference;
- relative MSE reduction;
- absolute MAE difference;
- Spearman difference;
- R-squared difference;
- absolute-bias difference.

## 10. Deterministic bootstrap

Each block uses 4,000 paired bootstrap replicates.

Bootstrap seeds:

    block A: 0x5444493532410001
    block B: 0x5444493532420002
    block C: 0x5444493532430003

The stratified aggregate bootstrap seed is:

    0x5444493532414747

Aggregate bootstrap resampling must:

1. preserve seed-block membership;
2. resample records within each block;
3. retain deterministic block order;
4. use identical resampled indices for compared models.

For every confirmatory MSE comparison, report the two-sided 95 percent
interval of the baseline-minus-challenger MSE difference.

For equivalence classification, report a two-sided 95 percent interval of
the relative MSE difference.

## 11. Criterion TDI-5.2A — joint signal

Compare B12 against B0 on combined width-3 and width-4 holdout at U_6.

TDI-5.2A succeeds only if:

1. B12 has lower MSE in all three seed blocks;
2. each block bootstrap lower bound is above zero;
3. median relative MSE reduction across blocks is at least 15 percent;
4. stratified aggregate relative MSE reduction is at least 15 percent;
5. aggregate bootstrap lower bound is above zero;
6. B12 Spearman is greater than B0 Spearman in every block;
7. aggregate absolute bias is not worse by more than 0.02.

## 12. Criterion TDI-5.2B — independent O2 signal

Compare B12 against B1 on combined width-3 and width-4 holdout at U_6.

TDI-5.2B succeeds only if:

1. B12 has lower MSE in all three seed blocks;
2. each block bootstrap lower bound is above zero;
3. median relative MSE reduction across blocks is at least 10 percent;
4. aggregate relative MSE reduction is at least 10 percent;
5. B12 Spearman is not lower than B1 Spearman in any block;
6. aggregate absolute bias is not worse by more than 0.02.

This criterion tests O_2 conditional on the baseline and O_1.

## 13. Classification TDI-5.2C — independent O1 contribution

Compare B12 against B2 on combined width-3 and width-4 holdout at U_6.

Use a symmetric relative-MSE margin of 2 percent.

### Beneficial

Classify O_1 as beneficial if:

1. B12 improves MSE over B2 by at least 2 percent in at least two blocks;
2. the corresponding bootstrap lower bounds are above zero;
3. aggregate relative improvement is at least 2 percent;
4. aggregate bootstrap lower bound is above zero.

### Equivalent

Classify O_1 as equivalent conditional on O_2 if:

1. all three block point estimates lie between minus 2 and plus 2 percent;
2. at least two block confidence intervals lie wholly within that margin;
3. the aggregate confidence interval lies wholly within that margin.

### Harmful

Classify O_1 as harmful under the symmetric reverse version of the
beneficial conditions.

### Inconclusive

Any other outcome is classified as inconclusive.

TDI-5.2C is a preregistered classification and is not forced to produce a
positive result.

## 14. Criterion TDI-5.2D — OOD transfer

The challenger is B12 and the baseline is B0.

### Width 5 requirements

1. positive standardized-MSE improvement in every block;
2. positive bootstrap lower bound in every block;
3. median relative MSE reduction of at least 20 percent;
4. positive B12 Spearman in every block;
5. non-worse Spearman in every block;
6. lower aggregate absolute bias;
7. positive reconstructed-O MSE improvement;
8. positive reconstructed-O MAE improvement.

### Width 6 requirements

1. positive standardized-MSE improvement in every block;
2. positive bootstrap lower bound in at least two blocks;
3. positive aggregate bootstrap lower bound;
4. positive B12 Spearman in every block;
5. non-worse aggregate Spearman;
6. non-worse aggregate absolute bias;
7. positive aggregate reconstructed-O MSE improvement.

TDI-5.2D succeeds only if both width-5 and width-6 requirements succeed.

## 15. Criterion TDI-5.2E — multi-horizon trajectory

Compare B12 against B0 on combined width-3 and width-4 holdout at:

    U_3
    U_4
    U_5
    U_8

TDI-5.2E succeeds only if:

1. at least three secondary horizons improve in every block;
2. U_8 improves in every block;
3. no block-horizon relative reduction is below minus 5 percent;
4. average secondary reduction is positive in every block;
5. aggregate reduction is positive at all four secondary horizons.

## 16. Exploratory analyses

The following analyses are exploratory only:

1. BD against B0, B1, B2 and B12;
2. direct prediction in reconstructed O space;
3. ridge sensitivity at lambda 0.1, 1.0 and 10.0;
4. coefficient inspection;
5. feature-correlation inspection;
6. residualized O_1 and O_2;
7. width-specific model fitting;
8. pooled model fitting across seed blocks.

Exploratory findings cannot modify confirmatory criteria.

## 17. Required raw output

The evaluator must print:

1. Git commit;
2. compiler and Cargo versions;
3. evaluator SHA-256;
4. preregistration SHA-256;
5. scientific-manifest SHA-256;
6. all frozen constants;
7. all seed-block definitions;
8. requested, accepted, rejected and attempted counts;
9. rejection counts by reason;
10. final exclusive seeds;
11. generation budgets;
12. target scalers;
13. model coefficients;
14. all metrics;
15. all bootstrap intervals;
16. block-level criteria;
17. aggregate criteria;
18. final TDI-5.2A through TDI-5.2E verdicts;
19. TDI-5.2C classification;
20. deterministic termination diagnostics.

## 18. Determinism

The following must be deterministic functions of committed constants:

- candidate generation;
- seed consumption;
- exclusions;
- preprocessing;
- model fitting;
- bootstrap sampling;
- aggregation;
- metric calculation;
- iteration order;
- scientific-value formatting;
- final criteria.

Wall-clock timestamps may be recorded only as reproduction metadata.

They must not affect any scientific result.

## 19. Reproduction requirements

The reproduction script must:

1. refuse a dirty repository;
2. verify all frozen hashes;
3. refuse an existing partial or complete output;
4. acquire an exclusive lock;
5. compile offline in release mode;
6. execute the evaluator exactly once;
7. capture complete standard output and error output;
8. verify all final criterion lines;
9. write metadata and a completion marker;
10. hash all result artifacts;
11. make final artifacts read-only.

## 20. Interpretation boundaries

A successful TDI-5.2 would establish replicated predictive information
from early overlap observations within the preregistered generator and
linear ridge model family.

It would not establish:

- universal validity across all dynamical systems;
- causal intervention effects;
- nonlinear sufficiency;
- arbitrary-width calibration;
- implementation-independent replication;
- external empirical validity.

The independent contribution of O_1 may be classified as beneficial,
equivalent, harmful or inconclusive.

No classification may be rewritten after observing the full result.

## 21. Freeze rule

After the preregistration, evaluator, manifests, reproduction script and CI
workflow are frozen:

1. scientific code must not change;
2. constants must not change;
3. seed blocks must not change;
4. criteria must not change;
5. no full run may begin before all frozen hashes pass;
6. any scientific-code defect requiring modification creates a new
   experiment identifier.
