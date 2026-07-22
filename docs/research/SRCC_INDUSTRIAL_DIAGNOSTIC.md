# Phase-728 Diagnostic — was contamination robustness even the bottleneck?

**Question.** Phase 728 returned honest nulls on its two central
contamination questions: robust regression was inconclusive (Huber beat OLS
on FD003 but not FD001, so replication failed) and every unsupervised
anomaly detector landed at or below chance on the frozen SECOM test. Before
proposing a *more powerful* contamination-robust algorithm, this diagnostic
asks whether contamination robustness is the bottleneck at all — by
decomposing each null into its causal parts.

**Method.** Exploratory re-analysis, **not** a new preregistered test. The
binary `scirust-srcc-bench/src/bin/industrial-diagnostic.rs` reuses the
frozen phase-728 configuration (`configs/phase728.json`) verbatim — same
splits, decimation, imputation, contamination — on the same checksum-verified
real data. Deterministic; run twice, byte-identical.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
49a48165bb4d77fbe763164e04e8bd41c12b5d798a1a5d6e59520d117d869ae0
records: results/diagnostic_728.jsonl
```

## Regression (C-MAPSS FD001, FD003): the contamination is not an attack

For each method, the contaminated-fit test RMSE is decomposed against the
method's **clean-fit RMSE** (the best a perfectly robust method could
recover), the **predict-the-mean RMSE** (task ceiling), and the
**prediction shift** `RMS(pred_contaminated − pred_clean)` (how far the
injected contamination actually moved the fitted function, in RUL units).

| Subset | Method | clean-fit RMSE | ceiling ratio | @20 %: contam RMSE | recovery residual | prediction shift |
|--------|--------|---------------:|--------------:|-------------------:|------------------:|-----------------:|
| FD001 | OLS | 41.99 | 0.666 | 41.56 | **−0.43** | 3.64 |
| FD001 | Huber | 41.09 | 0.652 | 40.88 | −0.21 | 2.18 |
| FD001 | trimmed | 42.40 | 0.673 | 42.55 | +0.15 | 2.28 |
| FD003 | OLS | 62.75 | 0.607 | 62.85 | +0.10 | 1.40 |
| FD003 | Huber | 63.54 | 0.615 | 63.44 | −0.10 | 1.07 |
| FD003 | trimmed | 70.23 | 0.680 | 69.54 | −0.69 | 3.77 |

Three facts, each fatal to the "we need a stronger robust estimator" reading:

1. **The contamination barely moves the fit.** At 20 % coherent
   contamination the prediction shift is 1–4 RUL against clean-fit RMSEs of
   42–70 — a 2–9 % perturbation. The coherent cluster (features `+50`,
   target `+80`) lands as a low-leverage block on this pooled, decimated,
   underfit design and the least-squares fit hardly notices it.
2. **Contamination does not degrade test RMSE.** The recovery residual
   (`contaminated − clean_fit`) hovers at zero and is frequently *negative* —
   contamination marginally *helps* the underfit model. When the baseline is
   not hurt, robustness has nothing to repair and cannot win.
3. **The task ceiling is low and the methods barely differ clean.** The
   pooled linear model cuts RMSE only ~33–39 % over predicting the mean
   (ceiling ratio 0.61–0.68), and OLS/Huber clean fits are within ~1 RUL on
   both subsets (trimmed is *worse* on FD003). The room for any regression
   estimator to separate is a fraction of an RUL — below the noise the paired
   bootstrap could resolve at 100 decimated engines.

Median-of-means' catastrophic breakdown (728) is orthogonal and already
explained: pooled RUL is piecewise, coherent contamination touches a
majority of seeded blocks, and the block-majority guarantee is void — a
documented assumption failure, not a power deficit.

**Verdict.** H2's null is a **framing** result, not a robustness-power
deficit. The levers are the task (a piecewise/monotone RUL model instead of a
pooled linear one; less aggressive decimation to recover statistical power;
higher-leverage-relevant channels) and a contamination model that actually
attacks the estimator (high-leverage design points, not a low-leverage
coherent block). A more powerful robust regressor changes none of these.

## SECOM: the failures are not geometric outliers

The below-chance frozen-test AUROC is decomposed into distribution shift
versus wrong-tool, via regularized-Mahalanobis separability measured
**in-sample** and on the **frozen test**, plus feature drift and base rates
(416 kept columns; 11 near-constant survivors of the exact-constant drop are
counted separately and excluded from the drift statistics).

| Quantity | Value | Reading |
|----------|------:|---------|
| Mahalanobis AUROC, in-sample (train) | **0.561** | barely above chance *fitting on the same data* |
| Mahalanobis AUROC, frozen test | 0.469 | below chance |
| fail / pass mean-distance ratio (test) | **0.012** | failures sit ~83× *closer* to the normal centre than passes |
| mean standardized drift (405 informative features) | 0.371 MAD | moderate |
| features shifted > 1 train-MAD | 27 / 405 | real but not dominant |
| base rate, train → test | 0.081 → 0.054 | a 33 % relative drop |

The decisive number is the **in-sample** AUROC of 0.561: even with the
covariance fitted on the very data it scores, Mahalanobis distance barely
separates yield failures — and the fail/pass distance ratio of 0.012 says
failures are *more central*, not more outlying, than normal wafers. SECOM
failures are **not low-density geometric outliers**, so unsupervised
density/distance anomaly detection is the wrong tool by construction; no
amount of estimator power fixes a wrong tool. Real-but-moderate temporal
drift (27/405 features, base-rate drop) compounds the validation-to-test
collapse but is not its root cause.

**Verdict.** SECOM's null is a **problem-formulation** result: the labels are
not an anomaly-detection target. The lever is supervised or physics-informed
features (which the phase-728 preregistration deliberately did not license
mid-stream), not a stronger unsupervised detector.

## Consequence for the program

Both central nulls of phase 728 are explained by framing — low contamination
leverage and a low task ceiling for regression; non-outlier failure geometry
for SECOM — **not** by any deficiency in the robustness of the estimators we
already have. This is direct evidence against the "add a more powerful
contamination-robust algorithm" reflex:

- a high-breakdown MM/S-estimator or a high-dimensional filtering/SDP robust
  mean estimator would still face a contamination that does not move the
  baseline fit and a task ceiling of a fraction of an RUL — it cannot win a
  contest the baseline is not losing;
- no unsupervised detector, however powerful, separates SECOM failures that
  are geometrically central.

The honest next investments are **framing**, not power: (1) a monotone /
piecewise RUL target and reduced decimation; (2) a high-leverage
contamination model that genuinely stresses the estimator (so robustness has
something to demonstrate); (3) for SECOM, a supervised or feature-engineered
formulation, or retiring it as an unsupervised-anomaly workload. Each is its
own small, reviewable change; none requires a new "powerful" algorithm on the
current evidence.

## Limitations

- The diagnostic inherits phase 728's decimation and pooled-linear RUL
  baseline; it explains *those* nulls, and a re-framed task could of course
  behave differently — which is exactly the point.
- Prediction shift and recovery are measured at the preregistered coherent
  contamination; a different (high-leverage) contamination model is expected
  to move the fit more, and testing that is item (2) above, not a claim made
  here.
- The near-constant SECOM columns are excluded from the drift mean/max only;
  they remain in the Mahalanobis fit (ridge-regularized), as in phase 728.
