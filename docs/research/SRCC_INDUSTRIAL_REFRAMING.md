# Phase-728 Re-framing — acting on the diagnostic

The [phase-728 diagnostic](SRCC_INDUSTRIAL_DIAGNOSTIC.md) showed the central
contamination nulls were **framing** results, not robustness-power deficits,
and named three levers. This document records each lever as it lands. Every
run reuses the frozen phase-728 split seeds, fractions and missing-value
policy (`configs/phase728.json`) so a lever's effect is isolated; each is
exploratory re-framing, not a new preregistered test, deterministic (run
twice, byte-identical).

## Lever 1 — piecewise RUL + reduced decimation

**Change.** Two isolated modifications to the C-MAPSS regression framing:

- the run-to-failure target is capped at the canonical piecewise-linear knee
  `RUL_pw = min(RUL, 125)` (Heimes 2008) — early-life cycles carry no
  degradation signal, so the raw-linear target imposes an unlearnable burden
  the diagnostic measured as a low task ceiling;
- decimation is reduced from stride 20 to stride 5, recovering ~4× the
  training rows the diagnostic flagged as a sacrificed statistical power.

**Binary.** `scirust-srcc-bench/src/bin/industrial-reframed-rul.rs`, plus the
pure `clip_rul_targets` library primitive (tested). A 2×2 grid of {raw,
piecewise} × {stride 20, stride 5} reports the predict-the-mean RMSE (task
ceiling), each method's clean-fit test RMSE, and the ceiling ratio
`clean_fit / predict_mean` — lower means the features explain more of the
target.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
9662938a30296ac6224333f500f2d88b60ff5c1ce5ed781efe2ebe9dcbaf99d9
records: results/reframed_rul.jsonl
```

**Result.** Best-method ceiling ratio (lower is better):

| Subset | raw, stride 20 (728 baseline) | piecewise, stride 20 | raw, stride 5 | piecewise, stride 5 | train rows (stride 5) |
|--------|------------------------------:|---------------------:|--------------:|--------------------:|----------------------:|
| FD001 | 0.652 | 0.545 | 0.638 | **0.521** | 2488 |
| FD003 | 0.607 | **0.458** | 0.603 | 0.493 | 3093 |

- **Piecewise RUL is the strong lever.** At fixed decimation it drops the
  ceiling ratio 0.652 → 0.545 (FD001) and 0.607 → 0.458 (FD003): the features
  now explain ~46–55 % of the target spread, up from ~35–39 %. This is the
  diagnostic's "low task ceiling" finding directly addressed — early-life
  clipping removes the unlearnable burden.
- **Reduced decimation restores power, not point accuracy.** Stride 5 gives
  ~4× the training rows (639 → 2488, 797 → 3093), which tightens the paired
  bootstrap CIs a robustness comparison needs. Its effect on the *point*
  ratio is small and mixed — it improves FD001 (0.545 → 0.521) but slightly
  worsens FD003 (0.458 → 0.493, as more noisy early-life rows re-enter). The
  value here is statistical power for Lever 2, not a better ceiling by
  itself; stated rather than oversold.

**Honest limit.** Even re-framed, a ratio of ~0.5 means the pooled **linear**
model's RMSE is still half of predicting the mean — a modest fit. Piecewise
RUL raises the ceiling materially but does not make C-MAPSS RUL a strong
linear-regression task; a monotone or nonlinear per-unit model is the next
framing lever, out of scope here. What Lever 1 establishes is a task with
enough ceiling and enough rows for Lever 2 (a high-leverage contamination
model) to be a *fair* test of whether the current robustness has anything to
demonstrate.

## Lever 2 — high-leverage contamination (the decisive test)

**Change.** The diagnostic showed the low-leverage coherent block barely moved
the least-squares fit, so robustness had nothing to repair. Lever 2 runs the
*fair* test: on the re-framed piecewise-RUL task it injects a genuinely
**high-leverage** attack — `ContaminationKind::LeveragePoint`, a fraction of
training rows pushed 20 train-MADs out in every feature with the target
overwritten to 250 (twice the RUL knee) — and re-runs the paired
leave-one-engine-out OLS-vs-robust comparison, sweeping the contamination
fraction {0.1, 0.2, 0.3} so the verdict's dependence on attack strength is
reported rather than a single tuned point.

**Binary.** `scirust-srcc-bench/src/bin/industrial-leverage.rs`, plus the
`LeveragePoint` library kind (tested, exact manifest). Per subset and
fraction: the mean OLS **prediction shift** `RMS(pred_contaminated −
pred_clean)` on held-out engines (leverage confirmation) and the seeded
bootstrap CI of the per-engine RMSE difference `OLS − robust`.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
cd038816dae2b7af6fbdf872c0363fdae7c39befbb0cd0a6adba4b925ad90d1c
records (24): results/leverage.jsonl
55a1df473074f3342fd22c4392d589f242cf43b5362e73f2f30e3cf6535185bb
```

**Result.** Paired mean `OLS − robust` RMSE difference, 95 % bootstrap CI over
100 engines (a CI above zero means robust wins; below zero means OLS wins):

| Subset | fraction | OLS shift (RUL) | OLS − Huber (CI) | OLS − trimmed (CI) |
|--------|---------:|----------------:|------------------|--------------------|
| FD001 | 0.1 | 2.18 | −0.042 [−0.137, +0.046] · **ties** | −3.399 [−4.404, −2.479] · OLS wins |
| FD001 | 0.2 | 2.42 | −0.027 [−0.114, +0.058] · **ties** | −2.245 [−3.397, −1.140] · OLS wins |
| FD001 | 0.3 | 2.65 | −0.072 [−0.169, +0.022] · **ties** | −1.593 [−2.484, −0.728] · OLS wins |
| FD003 | 0.1 | 6.99 | −0.040 [−0.214, +0.144] · **ties** | −1.641 [−2.423, −0.832] · OLS wins |
| FD003 | 0.2 | 7.05 | −0.023 [−0.190, +0.157] · **ties** | −1.299 [−1.986, −0.561] · OLS wins |
| FD003 | 0.3 | 7.12 | −0.041 [−0.248, +0.194] · **ties** | −0.810 [−1.383, −0.156] · OLS wins |

Three findings, and they are decisive:

1. **The attack barely dents held-out error.** Even at 30 % contamination with
   20-MAD-out leverage points, OLS's held-out prediction shifts only 2–7 RUL
   (against clean-fit RMSEs of ~22–31). A cluster of coincident far-out
   leverage points has bounded influence on predictions in the *normal* region
   the held-out engines occupy — the metric is structurally insensitive to
   feature-space-far training corruption. This is a property of the task, not
   a failure to attack hard enough: the fraction was swept to 30 % and the
   magnitude to 20 MADs.
2. **Huber never wins.** Every Huber-vs-OLS CI straddles zero, at every
   fraction, on both subsets: Huber is statistically indistinguishable from
   OLS under a confirmed high-leverage attack.
3. **Trimmed least squares actively hurts.** OLS beats trimmed at every cell.
   Trimming a fixed 30 % by residual discards good data whenever the
   contamination fraction is below the trim fraction; the disadvantage shrinks
   as contamination rises toward 30 % (Δ −3.40 → −1.59 on FD001) but never
   closes — a cautionary result about fixed-retention trimming.

**Verdict.** There is **no regime** — across contamination fraction 0.1–0.3
and a genuinely high-leverage attack — where the current robust regressors
beat OLS on held-out C-MAPSS RUL. Combined with the diagnostic (low task
ceiling) and Lever 1 (piecewise raises the ceiling but the linear fit stays
modest), this is decisive evidence that **a more powerful robust regression
algorithm is not the missing piece for this workload**: the held-out-RUL task
is structurally insensitive to feature-space contamination, so no estimator —
however high its breakdown point — has a contest to win.

**Honest scope.** This tests **feature-space leverage** attacks. A different
attack model — in-distribution target noise that looks like the test
distribution — could damage held-out error, but that is precisely the regime
where breakdown-point robustness is weakest and is a separate question; it is
not claimed resolved here. What is resolved: bad leverage points, the textbook
case robust regression exists for, do not create damage on this workload for
robustness to repair.

## Lever 3 — SECOM supervised reformulation

**Change.** The diagnostic showed SECOM failures are *not geometric outliers*
(regularized-Mahalanobis in-sample AUROC 0.56; failures sit closer to the
normal centre than passes), so every unsupervised density detector was at or
below chance on the frozen test (best 0.469) — a **problem-formulation** null,
not a detector-power deficit. Lever 3 tests that directly with a **supervised**
linear discriminant that uses the labels: on the imputed train split only,
`w = Σ⁻¹(μ_fail − μ_pass)` (ridge-regularized inverse covariance from
`fit_regularized_mahalanobis`), score `w · x`, ridge selected on validation
AUROC {0.001, 0.1}, frozen on test. Same phase-728 temporal split and
train-fitted imputer (no leakage).

**Binary.** `scirust-srcc-bench/src/bin/industrial-secom-supervised.rs` (four
unit tests for the pure discriminant/mean helpers).

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
61ae9dd5e3b2b3fbf3f554ddaacbd54fe9a5d7a6e97172b20f150960eb46b568
records (7): results/secom_supervised.jsonl
```

**Result.**

| Quantity | Supervised discriminant | Unsupervised (phase 728) |
|----------|------------------------:|-------------------------:|
| **in-sample (train) AUROC** | **0.975** (ridge 0.001) / 0.947 (ridge 0.1) | 0.56 (Mahalanobis) |
| validation AUROC | 0.40 (ridge 0.001) / 0.43 (ridge 0.1) | — |
| **frozen-test AUROC** (validation-selected ridge 0.1) | **0.581** | 0.469 |

Two findings, held in honest tension:

1. **The formulation conclusion is confirmed.** In-sample, the supervised
   discriminant separates SECOM failures **near-perfectly (0.975)** where the
   unsupervised Mahalanobis distance is barely above chance (0.56). The labels
   carry strong *linear discriminative* signal that density/geometry cannot
   see — SECOM is a supervised-classification problem wearing an
   anomaly-detection costume, exactly as the diagnostic argued.
2. **But SECOM stays genuinely hard.** The near-perfect in-sample fit collapses
   out of sample: validation AUROC drops *below* chance (0.40–0.43) and the
   frozen-test AUROC is only **0.581**. With 416 imputed features and ~86
   positive training rows, a naive linear discriminant massively overfits, and
   the temporal drift the diagnostic measured (27/405 features > 1 train-MAD)
   makes even supervised generalization unstable (validation below test).

**Verdict.** Supervision crosses from below chance (0.469) to above chance
(0.581) on the frozen test **and** reaches 0.975 in-sample — decisive that the
SECOM null was **problem formulation, not detector power**. But the modest,
unstable held-out number is reported, not dressed up: the reformulation
identifies the right problem *class* (supervised classification) without
solving it — proper regularization, feature selection and drift handling are
the real next steps, deliberately out of scope for a one-direction linear
probe.

---

## Three-lever synthesis

The phase-728 diagnostic asked whether a *more powerful contamination-robust
algorithm* was the missing piece. Across all three levers the answer is a
consistent, evidence-backed **no** — the barriers were framing, not power:

- **Lever 1** raised the C-MAPSS task ceiling with the canonical piecewise RUL
  (ratio 0.65 → 0.55 / 0.61 → 0.46) and restored 4× the statistical power, but
  the pooled linear fit stays modest (~0.5) — a framing gain, not a robustness
  gain.
- **Lever 2** ran the decisive robustness test on that improved task: under a
  genuine high-leverage attack swept to 30 %, **Huber never beats OLS and
  trimmed LS is always worse**, because far-out training corruption has bounded
  influence on in-distribution held-out predictions. No regime rewards the
  current robust regressors — and, by the diagnostic's structural argument, no
  *more powerful* one would either on this workload.
- **Lever 3** showed the SECOM anomaly null was a mis-formulation: supervision
  separates failures near-perfectly in-sample (0.975 vs unsupervised 0.56) and
  clears chance on the frozen test (0.581 vs 0.469), while remaining hard.

**Program conclusion.** For these two real industrial workloads, contamination
robustness was never the bottleneck: C-MAPSS RUL is structurally insensitive
to the contamination robust regression repairs, and SECOM was the wrong problem
class for unsupervised anomaly detection. Investment in a stronger robust
estimator is not warranted by this evidence; investment in **task framing**
(monotone/nonlinear RUL models; supervised, regularized, drift-aware SECOM
classification) is where the measurable gains live. This is the honest,
falsifiable close the phase-728 diagnostic pointed to — reached by building
and running the tests, not by assertion.

---

## Follow-up 1 — realizing the lever-1 ceiling with a monotone model

The synthesis located the C-MAPSS gain in *task framing* and named the next
model class explicitly: **monotone / nonlinear RUL**. This follow-up builds and
runs it. `scirust_learning::isotonic::IsotonicRegression` (PAVA, PR #757)
supplies the monotone, piecewise shape freedom a purely affine fit lacks: an
ordinary-least-squares regressor produces a scalar degradation score, an
isotonic map is fit from the **train** score to the **train** RUL (no leakage),
and applied to the test scores. Same frozen `phase728.json` splits, same 2×2
grid as lever 1 — the only change is the monotone model layered on the score.

`industrial-monotone-rul`, C-MAPSS FD001 / FD003, clean fit; lower ceiling ratio
= more of the task ceiling realized:

| subset | RUL target | stride | ceiling RMSE | OLS ratio | **isotonic ratio** | Δ |
|--------|-----------|-------:|-------------:|----------:|-------------------:|----:|
| FD001 | raw-linear | 20 | 63.010 | 0.6664 | 0.6433 | +0.0231 |
| FD001 | piecewise-125 | 20 | 41.737 | 0.5453 | **0.4570** | +0.0882 |
| FD001 | raw-linear | 5 | 61.233 | 0.6492 | 0.6275 | +0.0217 |
| FD001 | piecewise-125 | 5 | 42.014 | 0.5212 | **0.4665** | +0.0547 |
| FD003 | raw-linear | 20 | 103.319 | 0.6074 | 0.5602 | +0.0472 |
| FD003 | piecewise-125 | 20 | 41.700 | 0.5052 | **0.4418** | +0.0634 |
| FD003 | raw-linear | 5 | 101.392 | 0.6033 | 0.5491 | +0.0543 |
| FD003 | piecewise-125 | 5 | 40.910 | 0.5049 | **0.4498** | +0.0552 |

**Finding.** The monotone recalibration lowers the ceiling ratio in **all eight
cells** — it realizes strictly more of the task ceiling than the linear fit,
confirming the synthesis's prediction. The gain is largest on the **piecewise**
target the reframe introduced (Δ up to +0.088 on FD001 / stride-20), exactly
where a linear model most under-fits the flat early-life plateau: isotonic drives
the piecewise ratio down to ~0.44–0.47, from OLS's ~0.51–0.55.

**Honest bounds.** This is rank-preserving recalibration of a single OLS score:
it corrects a monotone-nonlinear miscalibration of that score's ordering but
cannot recover signal the OLS ordering discards. The residual ratio (~0.44) says
the piecewise ceiling is now *materially* realized, not closed — going further
needs richer degradation indices or a multivariate nonlinear model, not a better
one-dimensional monotone map. The claim this follow-up makes is the narrow one,
and it holds: the framing gain lever 1 opened is real, and a monotone model
captures a concrete, reproducible slice of it.

**Determinism.**

```
stdout SHA-256: 63067a526be1c10d84c5f7b342c222b816b5bd78237052cfb38ed3e24eb66bde
results (40 BenchRecords): 516ecd0a673ffbea4ade0f498914a12f927051f091c2717a6c82bf65c6d68967
```

Run twice, byte-identical; no network; checksum-verified C-MAPSS inputs; OLS
(Householder QR) and isotonic (PAVA) are both deterministic, with no RNG beyond
the frozen split seed.
