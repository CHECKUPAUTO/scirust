# TDI-5 — Continuous Deficit Geometry

## Experimental preregistration

### Status

This document defines the TDI-5 experiment before any implementation of
the evaluator and before any generation of the new populations.

No TDI-5 result must be produced before:

1. the commit of this preregistration;
2. its push to GitHub;
3. the separate implementation of the evaluator;
4. the SHA-256 freeze of the evaluator and of the scientific code;
5. the complete validation of the CI.

---

## 1. Motivation

The previous experiments studied the ability of early TDI observables to
predict the future recovery between a reference dynamics and a perturbed
dynamics.

The recovery at horizon \(h\) is:

\[
O_h
=
\sum_{x \in \mathcal X}
\min\!\left(P_h(x),Q_h(x)\right).
\]

It satisfies:

\[
O_h
=
1-\operatorname{TV}(P_h,Q_h).
\]

TDI-4 used a two-head architecture:

\[
Z=\mathbf 1[O_6=1]
\]

and, conditionally on \(O_6<1\),

\[
U_6=-\log_2(1-O_6).
\]

No exact recovery \(O_6=1\) was observed in the 35,000 TDI-4 systems. The
binary head was therefore degenerate.

In contrast, the continuous component showed:

- a combined loss reduction of 25.305777 % at widths 3 and 4;
- an OOD loss reduction of 50.033403 % at width 5;
- an OOD Spearman going from 0.467776309 to 0.615705212;
- an OOD \(R^2\) going from -0.762078825 to 0.119549183.

TDI-5 therefore directly studies the continuous geometry of the deficit.

---

## 2. Definitions

The recovery deficit is:

\[
D_h=1-O_h.
\]

The logarithmic coordinate of the deficit is:

\[
U_h=-\log_2(D_h)
=
-\log_2(1-O_h).
\]

The experiment retains only the systems satisfying:

\[
D_h>0
\]

at the target horizons concerned.

The early TDI observables are:

\[
\operatorname{TDI}_2
=
\left(
O_1,\,
O_2,\,
O_2-O_1
\right).
\]

---

## 3. Main scientific question

Do the TDI observables measured at horizons 1 and 2 provide additional and
transferable predictive information about the future trajectory of the
deficit, beyond the structural and entropic variables of the baseline?

The studied relationship is:

\[
(O_1,O_2,O_2-O_1)
\longrightarrow
(U_3,U_4,U_5,U_6,U_8).
\]

---

## 4. Hypotheses

### Main hypothesis

The addition of the three variables:

\[
O_1,\qquad O_2,\qquad O_2-O_1
\]

significantly reduces the prediction error of \(U_6\) on systems of widths
3 and 4 never seen during training.

### Transfer hypothesis

The gain transfers to widths not observed during training:

\[
w=5
\]

and:

\[
w=6.
\]

### Trajectory hypothesis

The gain is not limited to \(U_6\), but remains observable on several
future horizons:

\[
U_3,\ U_4,\ U_5,\ U_6,\ U_8.
\]

---

## 5. Population generation

All TDI-5 populations must be new and disjoint from the TDI-3 and TDI-4
populations.

| Population | Width | Accepted size | Initial seed |
|---|---:|---:|---:|
| training | 3 | 15,000 | 60,000,000 |
| training | 4 | 15,000 | 70,000,000 |
| holdout | 3 | 5,000 | 61,000,000 |
| holdout | 4 | 5,000 | 71,000,000 |
| main OOD | 5 | 10,000 | 80,000,000 |
| extreme OOD | 6 | 5,000 | 90,000,000 |

A seed denotes a generated candidate.

A rejected candidate definitively consumes its seed.

Generation continues until the exact number of accepted systems is reached
for each population.

The final exclusive seeds and the number of rejected candidates must be
recorded in the raw result.

---

## 6. Horizons

Observation horizon:

\[
h_{\mathrm{obs}}=2.
\]

Target horizons:

\[
\mathcal H=\{3,4,5,6,8\}.
\]

The main confirmatory target is:

\[
U_6.
\]

The targets \(U_3\), \(U_4\), \(U_5\) and \(U_8\) are preregistered
secondary analyses.

---

## 7. Exclusion criteria

A system is excluded if one of the following conditions is satisfied:

1. the exact recovery \(O_2\) equals 1;
2. the exact deficit is zero at one of the target horizons;
3. a required variable is not finite after transformation;
4. an exact operation of the dynamics engine fails;
5. the generation violates the structural invariants already imposed by
   TDI-4.

No exclusion may depend:

- on the predictions;
- on the model errors;
- on the value of the TDI variables relative to the baseline;
- on the observed improvement.

---

## 8. Explanatory variables

### Baseline

The baseline keeps exactly the 13 structural variables used by TDI-4.

No new structural variable must be added.

### Main challenger

The challenger contains the same 13 variables plus:

\[
O_1,
\]

\[
O_2,
\]

\[
O_2-O_1.
\]

The challenger therefore has 16 variables.

---

## 9. Secondary ablations

The secondary models are:

\[
M_0
=
\text{baseline},
\]

\[
M_1
=
\text{baseline}+O_1,
\]

\[
M_2
=
\text{baseline}+O_1+O_2,
\]

\[
M_3
=
\text{baseline}+O_1+O_2+(O_2-O_1).
\]

The main confirmatory model is \(M_3\).

The models \(M_1\) and \(M_2\) cannot replace \(M_3\) for the main verdicts.

---

## 10. Models

For each target horizon, the baseline and the challenger use a separate
ridge regression.

The objective function is:

\[
\min_{\beta_0,\beta}
\sum_{i=1}^{N}
\left(
\widetilde U_{h,i}
-
\beta_0
-
\widetilde X_i^\top\beta
\right)^2
+
\lambda\|\beta\|_2^2.
\]

The penalty is fixed at:

\[
\lambda=1.
\]

The intercept is not penalized.

The baseline and the challenger must use:

- the same algorithm;
- the same accumulation order;
- the same numerical precision;
- the same penalty;
- the same populations;
- the same target transformations.

The only difference is the addition of the three TDI variables to the
challenger.

---

## 11. Normalization

For each explanatory variable:

\[
\widetilde X_j
=
\frac{X_j-\mu_j}{s_j}.
\]

For each target horizon:

\[
\widetilde U_h
=
\frac{U_h-\mu_{U_h}}{s_{U_h}}.
\]

All means and scales are learned only on the combined training set of
widths 3 and 4.

They are then frozen for:

- the width-3 holdout;
- the width-4 holdout;
- the OOD width 5;
- the OOD width 6.

A zero scale is replaced by 1.

---

## 12. Main metric

For \(U_6\), the main metric is the MSE in the standardized space:

\[
\operatorname{MSE}_{U_6}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\widetilde U_{6,i}
-
\widehat{\widetilde U}_{6,i}
\right)^2.
\]

The absolute improvement is:

\[
\Delta_{U_6}
=
\operatorname{MSE}_{U_6}^{\mathrm{baseline}}
-
\operatorname{MSE}_{U_6}^{\mathrm{TDI}}.
\]

The relative reduction is:

\[
G_{U_6}
=
\frac{
\operatorname{MSE}_{U_6}^{\mathrm{baseline}}
-
\operatorname{MSE}_{U_6}^{\mathrm{TDI}}
}{
\operatorname{MSE}_{U_6}^{\mathrm{baseline}}
}.
\]

---

## 13. Secondary metrics

For each horizon and each population, the evaluator reports:

- standardized MSE;
- standardized MAE;
- \(R^2\);
- Spearman correlation;
- mean bias;
- observed mean;
- predicted mean;
- calibration slope;
- calibration intercept.

---

## 14. Recovery reconstruction

The reconstruction is:

\[
\widehat O_h
=
1-2^{-\widehat U_h}.
\]

The evaluator also reports, in the space of \(O_h\):

- MSE;
- MAE;
- \(R^2\);
- Spearman;
- mean bias;
- calibration;
- proportion of predictions brought back to the numerical bounds.

These metrics are secondary.

---

## 15. Paired bootstrap

The loss differences are evaluated by paired bootstrap.

Number of replications:

\[
B=2000.
\]

Seed:

\[
\texttt{0x5444\_4935\_4344\_4745}.
\]

Each replication resamples the example indices with replacement.

The baseline and challenger predictions remain paired for each index.

The 95 % interval uses the empirical 2.5 % and 97.5 % quantiles.

No retraining is performed inside the bootstrap.

---

## 16. Primary criterion TDI-5A

The primary criterion is evaluated on the combined width-3 and width-4
holdout for the target \(U_6\).

TDI-5A is declared **PASSED** if all the following conditions are
satisfied:

1. combined relative reduction:

\[
G_{U_6}^{(3+4)}\geq 10\%;
\]

2. lower bound of the 95 % bootstrap CI of
   \(\Delta_{U_6}^{(3+4)}\) strictly positive;

3. strictly positive point improvement at width 3;

4. strictly positive point improvement at width 4;

5. lower bound of the width-3 bootstrap CI strictly positive;

6. lower bound of the width-4 bootstrap CI strictly positive;

7. challenger Spearman strictly greater than that of the baseline on the
   combined holdout;

8. challenger Spearman strictly positive separately at widths 3 and 4;

9. challenger standardized absolute bias not exceeding that of the
   baseline by more than 0.02 on the combined holdout.

Otherwise:

`CRITÈRE PRINCIPAL TDI-5A : ÉCHOUÉ`

In case of success:

`CRITÈRE PRINCIPAL TDI-5A : RÉUSSI`

---

## 17. Transfer criterion TDI-5B

TDI-5B concerns width 5 for the target \(U_6\).

It is declared **PASSED** if all the following conditions are satisfied:

1. relative reduction:

\[
G_{U_6}^{(5)}\geq 20\%;
\]

2. lower bound of the 95 % bootstrap CI strictly positive;

3. challenger Spearman strictly positive;

4. challenger Spearman greater than or equal to that of the baseline;

5. challenger \(R^2\) strictly greater than the baseline \(R^2\);

6. challenger absolute bias strictly lower than the baseline absolute bias;

7. strictly positive point improvement of the reconstructed MSE;

8. strictly positive point improvement of the reconstructed MAE.

Otherwise:

`CRITÈRE TRANSFERT TDI-5B : ÉCHOUÉ`

In case of success:

`CRITÈRE TRANSFERT TDI-5B : RÉUSSI`

---

## 18. Extreme transfer criterion TDI-5C

TDI-5C concerns width 6 for the target \(U_6\).

It is declared **PASSED** if all the following conditions are satisfied:

1. point improvement:

\[
\Delta_{U_6}^{(6)}>0;
\]

2. lower bound of the 95 % bootstrap CI strictly positive;

3. challenger Spearman strictly positive;

4. challenger Spearman greater than or equal to that of the baseline;

5. challenger absolute bias less than or equal to the baseline absolute
   bias;

6. strictly positive point improvement of the reconstructed MSE.

Otherwise:

`CRITÈRE TRANSFERT EXTRÊME TDI-5C : ÉCHOUÉ`

In case of success:

`CRITÈRE TRANSFERT EXTRÊME TDI-5C : RÉUSSI`

---

## 19. Trajectory criterion TDI-5D

The secondary horizons are:

\[
\{3,4,5,8\}.
\]

For each horizon, the combined width-3 and width-4 improvement is computed.

TDI-5D is declared **PASSED** if:

1. at least three of the four horizons show a strictly positive point
   improvement;

2. \(U_8\) shows a strictly positive point improvement;

3. no target shows a relative degradation greater than 5 %;

4. the arithmetic mean of the four relative reductions is strictly
   positive.

Otherwise:

`CRITÈRE TRAJECTOIRE TDI-5D : ÉCHOUÉ`

In case of success:

`CRITÈRE TRAJECTOIRE TDI-5D : RÉUSSI`

This criterion is secondary and does not replace TDI-5A.

---

## 20. Secondary direct comparator

A ridge comparator directly predicts \(O_6\) with:

- the 13 baseline variables;
- then the 13 variables plus the three TDI variables.

This comparator is secondary.

Its results cannot modify the TDI-5A, TDI-5B, TDI-5C or TDI-5D verdicts.

---

## 21. Determinism

The evaluator must guarantee:

- deterministic generation by seed;
- fixed iteration order;
- fixed floating-point accumulation order;
- absence of parallelism in the computations that modify the order of the
  sums;
- deterministic bootstrap;
- deterministic textual results;
- offline execution;
- absence of network dependency;
- absence of selection after observation.

---

## 22. Required output

The raw result must include:

1. Git identity;
2. Rust and Cargo versions;
3. population sizes;
4. numbers of exclusions;
5. final exclusive seeds;
6. statistics of \(U_h\) for each horizon and population;
7. learned normalizations;
8. coefficients of all the confirmatory models;
9. results of \(M_0\), \(M_1\), \(M_2\) and \(M_3\);
10. metrics in the space of \(U_h\);
11. reconstructed metrics in the space of \(O_h\);
12. bootstrap intervals;
13. secondary direct comparator;
14. the four exact final lines:

`CRITÈRE PRINCIPAL TDI-5A : RÉUSSI|ÉCHOUÉ`

`CRITÈRE TRANSFERT TDI-5B : RÉUSSI|ÉCHOUÉ`

`CRITÈRE TRANSFERT EXTRÊME TDI-5C : RÉUSSI|ÉCHOUÉ`

`CRITÈRE TRAJECTOIRE TDI-5D : RÉUSSI|ÉCHOUÉ`

---

## 23. Prohibitions before freeze

Before the freeze of the evaluator, it is forbidden to:

- generate the complete TDI-5 populations;
- consult TDI-5 metrics;
- modify the criteria according to an intermediate result;
- test several values of \(\lambda\);
- select the seeds;
- change the baseline;
- remove an unfavorable width;
- remove an unfavorable horizon;
- modify the success thresholds.

Synthetic unit tests and determinism tests are allowed, provided that they
do not generate the preregistered populations.

---

## 24. Execution order

The immutable order is:

1. commit of the preregistration;
2. push of the branch;
3. implementation of the evaluator;
4. unit tests;
5. SHA-256 freeze of the evaluator;
6. freeze of the scientific code;
7. commit and push of the freeze;
8. CI validation;
9. first unique execution;
10. archiving of the results;
11. immutable tag;
12. PR to `main`.
