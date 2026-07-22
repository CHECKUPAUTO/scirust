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

---

## Follow-up 2 — stabilizing the supervised SECOM discriminant

Lever 3 confirmed SECOM is a supervised-classification problem, but its
discriminant over-fit hard: **all ~416 imputed features**, a **total-scatter**
covariance, ridge only in {0.001, 0.1}. Its in-sample AUROC was 0.975 while its
**validation AUROC sat below chance (0.40–0.43)** — so its frozen-test 0.581 was
*not a validation-selectable result*, it was the less-bad of two broken options.
This follow-up builds the model lever 3 named but did not fit — a **regularized
LDA that tries to generalize** — with three additive stabilizers, each aimed at
the p ≫ n overfit (`industrial-secom-stable`):

1. **Standardization** — train-fitted per-feature z-scoring (scale-fair ridge and
   feature scores).
2. **Univariate feature selection** — keep the top-`k` features by the absolute
   pooled-standardized (Cohen's-d) class-mean difference on **train** only,
   cutting ~416 → `k` so the covariance is estimable (`k` ≪ the ~90 minority
   rows).
3. **Within-class (Fisher) covariance** — `w = S_W⁻¹ (μ_fail − μ_pass)` using the
   pooled *within-class* scatter (each row centered by its own class mean before
   the regularized-Mahalanobis fit), not lever 3's total scatter.

`(k, ridge)` selected on validation AUROC over fixed a-priori grids
(`k ∈ {5,10,20,40,80}`, `ridge ∈ {0.01,0.1,1,10}`), frozen on the same phase-728
temporal test split:

| model | validation AUROC | frozen-test AUROC | validation→test |
|-------|----------------:|------------------:|----------------:|
| unsupervised (phase 728, best density) | — | 0.469 | — |
| lever-3 supervised (416 feat, total scatter) | 0.40–0.43 | 0.581 | +0.15 (anti-predictive) |
| **follow-up 2** (k=20, within-class LDA, standardized) | **0.587** | **0.567** | **−0.020** |

**Finding — reliability, not a higher ceiling.** The stabilized model does **not**
raise the absolute test AUROC (0.567 vs 0.581 is within noise for ~1500 rows).
What it changes is *trustworthiness*: its validation AUROC (0.587) is now **above
chance and tightly predicts the frozen test** (0.567, drift −0.020), whereas
lever 3's below-chance validation could not select its own test number at all.
Stabilization converts an unselectable, lucky-looking 0.581 into a **validation-
selectable, drift-stable ~0.57** — the number you would actually have picked in
advance. It also does so with **20 features instead of 416**.

**Honest bound.** SECOM's linear-supervised ceiling is ~0.57 on this frozen
temporal split, and no combination in the grid clears it: standardization,
selection and within-class covariance curb variance but cannot manufacture the
signal a linear model in these features does not have. Going higher needs a
*nonlinear* classifier or explicit drift correction, not a better-conditioned
linear discriminant — the same "framing, then model class" ladder lever 1 / and
follow-up 1 climbed on the RUL side. The narrow, defensible claim: the supervised
SECOM result is now **reproducible and selectable**, which lever 3's was not.

**Determinism.**

```
stdout  SHA-256: 719d2f746856793494b5a98018674dd8552abed19df94fd4024a4fbf50c4d4a2
results SHA-256: 000d876b2d24d5e179c46801318db34c96786c45c6779c50aaaf58a3ab466336  (26 BenchRecords)
```

Run twice, byte-identical; no network; checksum-verified SECOM inputs; the
standardizer, univariate selection (`total_cmp` with index tie-break), and the
regularized within-class LDA are all deterministic, with no RNG beyond the frozen
temporal split.

---

## Follow-up 3 — the regime that reopens the question: native heavy-tailed noise

The whole program's opening question was whether the robust estimators the SRCC
work built are ever the missing piece. Lever 2 answered "not under *injected
high-leverage* contamination" — but that is one contamination geometry. This
follow-up tests the other canonical one, the regime robust M-/trimmed estimators
are actually *designed* for: **pervasive heavy-tailed vertical outliers** (a
native error distribution with heavy tails, not a handful of adversarial leverage
points).

Controlled and honest: the **real** C-MAPSS FD001 design matrix (imputed,
standardized — real feature geometry and collinearity, the same X as lever 2), a
fixed planted linear signal `Xβ`, and native errors drawn from a **Student-t**
whose degrees of freedom `ν` sweep the tail heaviness (`ν = 1` Cauchy … `ν = 30`
≈ Gaussian). Recovery is measured against the *noiseless* signal on a held-out
grouped split, so the metric is how well each estimator recovered the truth
despite heavy-tailed *training* noise; OLS / Huber-IRLS / trimmed LS are compared
with the same seeded paired bootstrap (of per-row signal squared error) as
lever 2 (`industrial-heavy-tailed`).

| `ν` (tail) | OLS signal-RMSE | Huber − OLS (paired Δ, 95% CI) | Trimmed − OLS |
|-----------:|----------------:|-------------------------------|---------------|
| **1** (Cauchy) | **39.63** | **+1568.9 [1307.9, 1867.7] robust wins** | **+1566.4 robust wins** |
| **2** | 1.681 | **+1.79 [1.34, 2.30] robust wins** | **+1.04 [0.50, 1.58] robust wins** |
| 3 | 1.286 | **+0.68 [0.32, 1.12] robust wins** | −1.46 ols wins |
| 5 | 1.106 | +0.13 [−0.08, 0.44] tie | −1.54 ols wins |
| 30 (~Gaussian) | 0.818 | −0.13 [−0.45, 0.05] tie | −1.65 ols wins |

**Finding — yes, a regime rewards the robust estimators, and it is the mirror
image of lever 2.** Under genuinely heavy tails (`ν ≤ 2–3`) **Huber-IRLS
decisively beats OLS** — at `ν = 1` OLS is wrecked (signal-RMSE 39.6, the Cauchy
tail dominates the L2 loss) while Huber and trimmed recover the signal orders of
magnitude better. As the tails lighten toward Gaussian (`ν = 30`) the advantage
vanishes: Huber converges to OLS (a tie, exactly as theory says) and OLS is
optimal. Lever 2 and follow-up 3 use the **same** C-MAPSS design and the **same**
paired test; only the contamination *geometry* differs — injected high-leverage
points (lever 2) → OLS wins; pervasive heavy-tailed vertical outliers
(follow-up 3) → robust wins. **The pivot is geometry, not estimator power.**

**Honest bounds.** (1) This is semi-synthetic: real `X`, but a planted signal and
native `t`-errors — the way to *isolate* the noise geometry a fixed real target
cannot vary. (2) The result is about which regime rewards robustness, not a claim
that C-MAPSS lives in it: the real RUL residuals are not this heavy, which is
precisely why lever 2 was null. (3) **Huber, not trimmed, is the robust estimator
that pays** — trimmed LS's fixed 0.7 retention helps only at the most extreme
tails (`ν ≤ 2`) and *hurts* from `ν = 3` up by discarding good rows; a fixed
trimming fraction is the wrong knob unless the contamination fraction is known.

**What it means for the program.** The contamination-robust tools are not dead
weight — they are the right instrument for **heavy-tailed vertical-outlier
noise**, a real and common sensor-failure mode. They are simply not what the two
industrial workloads studied here needed. "Do we need a more powerful robust
algorithm?" resolves to: *not for these workloads, but the regime where the
existing ones already win is well-defined and now demonstrated* — and Huber, the
bounded-influence M-estimator, is the one to reach for there.

**Determinism.**

```
stdout  SHA-256: ac16966be618033828fb86b868d8efa6bcf180b2b70dccc1d460107ad0e12387
results SHA-256: 6f366e1a0cf2aba36c6d09b06c0972e65f20dedb97976a5bf8cfc2ab227dca8f  (25 BenchRecords)
```

Run twice, byte-identical; no network; checksum-verified C-MAPSS inputs; the
Student-t draws use a seeded `SplitMix64` through the distribution's inverse CDF
(`ν`-combined seed), and OLS / Huber / trimmed are deterministic — no other RNG.

---

## Follow-up 4 (phase 729) — shadow deployment & promotion gates

The program produced *evidence*; a deployment needs a reproducible *decision*.
This is the industrialization layer: a deterministic
`scirust_srcc_bench::promotion::PromotionGate` that turns a **preregistered** rule
into a promote/hold verdict on a **shadow comparison** — a candidate model scored
alongside the incumbent on the same units. The rule has a **primary criterion**
(the candidate must improve the primary metric, with the *lower* bound of the
improvement's bootstrap CI clearing `min_improvement`) and any number of
**guardrails** (the candidate must not regress, with the *upper* bound of each
regression's CI staying below `max_regression`); promotion needs the primary to
pass **and** every guardrail to hold. Every decision is the seeded paired
bootstrap from `crate::paired` — never a point estimate — so the verdict is
bit-reproducible. The module ships with a manual typed `PromotionError` and six
unit tests (clear-win promote, straddle hold, guardrail-regression hold,
higher-is-better orientation, missing-metric error, determinism).

`industrial-promotion-gate` drives it on a real shadow comparison that
operationalizes follow-up 3: **candidate Huber-IRLS vs incumbent OLS** on the same
held-out C-MAPSS rows under native Student-t training noise, with a preregistered
gate (primary = per-row signal squared error, `min_improvement = 0`; guardrail =
per-row signal absolute error, `max_regression = 0.5`):

| `ν` (tail) | decision | primary Δ (95% CI) | guardrail regression (CI upper) |
|-----------:|:--------:|--------------------|---------------------------------|
| **2** (heavy) | **PROMOTE** | +2.645 [2.084, 3.241] ✓ | −0.753 (−0.619) ✓ |
| 5 (moderate) | **HOLD** | +0.014 [−0.415, 0.288] ✗ | −0.046 (0.013) ✓ |
| 30 (~Gaussian) | **HOLD** | +0.044 [−0.132, 0.345] ✗ | +0.026 (0.075) ✓ |

**The gate deploys Huber precisely when the data is heavy-tailed and holds the
incumbent otherwise** — the automated MLOps decision that the whole program's
finding implies, made mechanically rather than by eyeballing a mean. At `ν = 2`
the squared-error improvement CI clears zero and the absolute-error guardrail is
also favorable, so Huber is promoted; at `ν ≥ 5` the improvement CI straddles zero
(no statistically defensible gain) and the incumbent stays. This is exactly the
discipline the phase-728 preregistration argued for, now applied to the
promotion boundary itself: a candidate is adopted only on evidence fixed in
advance, not on a hopeful point estimate.

**Determinism.**

```
stdout  SHA-256: a555aab5303f6225be7ca27348d1588078123c671c0da334805860fc77eb3a42
results SHA-256: e39c82d1ac7b39a29cae66c1a17188e7e11dbbe5dacc36552c0ec6777b098397  (9 BenchRecords)
```

Run twice, byte-identical; no network; checksum-verified C-MAPSS inputs; the whole
decision is the seeded paired bootstrap — no RNG beyond its recorded seed.

---

## Follow-up 5 (axis 1) — multivariate nonlinear RUL: does it close the ceiling further?

Follow-up 1 realized part of the lever-1 ceiling with a *one-dimensional*
monotone recalibration (isotonic on the OLS score) — rank-preserving, so it
cannot recover signal the OLS ordering discards. This tests the model class
beyond it: a genuinely **multivariate nonlinear** regressor,
`scirust_learning::KernelRidgeRegression` (RBF, PR #763), with `(γ, λ)` selected
on validation and frozen on test, head-to-head with OLS and isotonic-OLS on the
**same** rows (`industrial-nonlinear-rul`). Decimation is stride 40 (~320 pooled
train rows) for all three models: kernel ridge is a dense `O(n³)` Cholesky and
the shared solver's per-iteration constant is large, so the finer strides balloon
the grid-search runtime — the absolute ratios thus differ from follow-up 1's
stride-20 numbers, and what is tested is the *ordering* on identical data.

Ceiling ratio (`clean_fit_rmse / predict_mean_rmse`, lower = more realized):

| subset | RUL target | OLS | isotonic-OLS | **kernel ridge** (γ, λ) | kernel − isotonic |
|--------|-----------|----:|-------------:|------------------------:|------------------:|
| FD001 | raw-linear | 0.679 | 0.667 | **0.608** (0.01, 1.0) | **+0.059 (kernel wins)** |
| FD001 | piecewise-125 | 0.573 | **0.445** | 0.448 (0.01, 0.1) | −0.004 (tie) |
| FD003 | raw-linear | 0.601 | **0.574** | 0.627 (0.10, 1.0) | −0.053 (isotonic wins) |
| FD003 | piecewise-125 | 0.507 | **0.438** | 0.485 (0.05, 1.0) | −0.048 (isotonic wins) |

**Finding — mostly no; nonlinearity is an inconsistent lever here.** The
multivariate nonlinear model closes the ceiling further than the 1-D isotonic
recalibration in **only one of four cells** (FD001 raw-linear, where the most
ceiling is unrealized and there is room for curvature to help); it **ties** on
the FD001 piecewise target and is **beaten on both FD003 cells**, where the
validation-selected kernel over-fits and generalizes worse than the simple
monotone recalibration. The 1-D isotonic recalibration is a surprisingly strong
baseline: on the reframed piecewise target it is at least as good as kernel ridge
everywhere.

**Interpretation.** For C-MAPSS RUL, most of the learnable signal is captured by
the *monotone ordering* of the linear degradation score; the extra capacity of a
multivariate nonlinear model mostly buys over-fitting rather than new signal,
especially on the harder FD003. This reinforces the program's through-line: the
big lever was **framing** (the piecewise target), not model complexity — richer
models give diminishing, dataset-dependent returns. Honest bound: this is at
stride 40 with a modest a-priori `(γ, λ)` grid; a larger sweep or a faster SPD
solver might narrow the FD003 gap, but the qualitative result — nonlinearity does
not *reliably* beat the 1-D recalibration — is unlikely to flip.

**Determinism.**

```
stdout  SHA-256: 9c529e9d89153944c4a5432cc0402ca4851ceab6c6f84bae634c81e0b3c75dbe
results SHA-256: 71788a2019cc32efd5e6bade26eb2abd2aa35df53d2a1a5ea23d9cfa9e47cc93  (24 BenchRecords)
```

Run twice, byte-identical (built `--release` for the `O(n³)` solves; release is
equally deterministic); no network; checksum-verified C-MAPSS inputs; OLS, isotonic
(PAVA) and kernel ridge (Cholesky) are all RNG-free.

## Follow-up 6 (axis 2) — SECOM past the linear ceiling: nonlinearity vs drift

Item 2 stabilized the supervised SECOM discriminant (standardize → univariate
top-`k` → within-class LDA) into a *selectable* ~0.567 frozen-test AUROC, but
could not raise the ceiling, and named two suspects for the residual barrier:
**temporal drift** (the SECOM process moves, so the earliest wafers least
resemble the test period) and **functional form** (a linear discriminant may be
too rigid). This axis tests each with its own binary, both under item 2's exact
protocol — temporal split, train-fit imputation, `(·)` selected on validation,
frozen on test — so the only thing that changes is the hypothesis.

### Route A — explicit drift correction (`industrial-secom-drift`)

Fit the *same* within-class LDA on a **recency window**: the most-recent fraction
`w` of the temporally-ordered train split, closest to the held-out test regime.
Sweep `w ∈ {0.4, 0.6, 0.8, 1.0}` (with `w = 1.0` recovering item 2's all-train
model), `k ∈ {5, 10, 20, 40}`, `ridge ∈ {0.1, 1, 10}`; skip any window with fewer
than five of either class; select `(w, k, ridge)` on validation AUROC.

**Finding — null; drift correction *hurts* here.** Validation selects
`w = 0.8, k = 40, ridge = 0.1` at validation AUROC **0.630**, but the frozen test
falls to **0.530** — a validation→test drop of **−0.099**, and *below* item 2's
0.567. Trimming stale training rows does not help; it overfits the small
validation set and throws away data the rare-failure class cannot spare. The
recency window is the wrong lever.

### Route B — low-order nonlinearity (`industrial-secom-nonlinear`)

Keep all training rows but let the discriminant be **nonlinear** in the top-`k`
features: expand them with a degree-2 polynomial map (squares and pairwise
products, width `k(k+3)/2`) before the within-class LDA. `degree = 1` recovers
item 2's linear model, so the sweep carries its own linear baseline. The honest
constraint is small samples — the train split holds only **76 failures** — so
every configuration whose expanded width exceeds that failure count is **skipped
a priori** (a discriminant with more parameters than the rarer class is
rank-starved); the complexity budget is tied to the data, not chosen to flatter
the result. That caps degree-2 at `k ≤ 8` (width 44); `k = 12, 20` (widths 90,
230) are skipped. Sweep `degree ∈ {1, 2}`, `k ∈ {4, 6, 8, 12, 20}`,
`ridge ∈ {0.1, 1, 10}`.

| model | selected config | validation AUROC | frozen-test AUROC |
|-------|-----------------|-----------------:|------------------:|
| linear (degree 1), best on validation | `k = 6, ridge = 1` | 0.615 | — (not selected) |
| **degree-2 polynomial LDA** | `k = 8, ridge = 10`, width 44 | **0.662** | **0.6177** |
| item-2 baseline (linear, all features via total scatter → within-class) | — | — | 0.567 |
| lever-3 baseline (all features, total scatter) | — | — | 0.581 |
| unsupervised baseline (best density detector) | — | — | 0.469 |

**Finding — yes; nonlinearity clears the ceiling.** Degree-2 dominates degree-1
on validation (0.662 vs 0.615, i.e. the sweep *chooses* to be nonlinear), and the
selected quadratic generalizes: frozen-test AUROC **0.6177**, above both item 2
(**+0.051**) and lever 3 (**+0.037**), with a modest validation→test drift
(**−0.044**, comparable to item 2's own −0.02). Curvature in the eight strongest
features carries real, transferable failure signal that the linear discriminant
discards.

### Axis-2 synthesis

The ~0.57 SECOM ceiling **is** breakable — but by the functional lever, not the
temporal one. Adding low-order curvature *within the failure-count budget* lifts
the frozen-test AUROC to 0.618; trimming stale rows to chase drift *drops* it to
0.530. The two routes fail and succeed for the same underlying reason: SECOM's
failures are rare, so the binding constraint is **model capacity per failure**.
The drift route spends that scarce budget by shrinking the training set; the
nonlinear route spends it by adding just enough parameters to stay under the
budget while capturing curvature. Same constraint, opposite sign. This refines —
does not overturn — the program's through-line: framing and capacity, matched to
how little labelled failure data exists, beat both raw robustness and naive
"use-more-recent-data" heuristics. Honest bound: SECOM's validation and test
folds are small (76 train / a few dozen test failures), so the point estimates
carry wide intervals; the deterministic winner clears both baselines but the
margin is not large.

**Determinism.**

```
drift      stdout  SHA-256: 23f55815bbfffcdb80ab6a0f50a758597697465a8dc2c1377230aea551c10b01
drift      results SHA-256: 226b01142e32de66617f2940ec9a55068455f7236d8ccd97edff570962409a40  (56 BenchRecords)
nonlinear  stdout  SHA-256: 34489263a2d788e8293cb60188038455dad0b5b7c01853a35a23b39cedb61f6e
nonlinear  results SHA-256: 1a6f012a0b75b2b112ab2098b04962f7776b45e7ddb5f82589435e0425eae20b  (33 BenchRecords)
```

Run twice, byte-identical; no network; checksum-verified SECOM inputs; both
binaries are all-linear-algebra (LDA covariance is `d × d`, no `O(n³)` kernel
solve) and RNG-free.

## Follow-up 7 (axis 3) — real native heavy tails: does robustness finally win on real data?

Follow-up 3 identified *where* robust regression wins — pervasive heavy-tailed
vertical noise — but it had to **manufacture** that regime: the real C-MAPSS
design matrix carried a *planted* linear signal plus *injected* Student-t errors,
because C-MAPSS's own residuals are not that heavy. This axis removes the
injection and anchors item 3 in real measurements: the in-repo Opel-Corsa **OBD2
telemetry** (43 139 rows, seven driving segments), predicting one real engine
channel from the other eleven sensors (`industrial-obd2-native`). No planted
signal, no injected noise — whatever heaviness the residuals carry is the car's.

Three targets are used, each with natively heavy-tailed OLS residuals; the binary
**re-derives** the tail statistics from its own pooled OLS residuals, so
"native" is checked, not asserted:

| target | OLS-residual excess kurtosis | beyond 3·MAD | beyond 5·MAD |
|--------|-----------------------------:|-------------:|------------:|
| `ENGINE_LOAD` | 45.2 | 3.87 % | 1.395 % |
| `THROTTLE_POS` | 19.0 | 5.05 % | 2.612 % |
| `MAF` | 10.3 | 2.36 % | 0.614 % |
| *normal reference* | 0 | 0.27 % | 0.00006 % |

A residual mass beyond 5·MAD of 0.6–2.6 % is three-to-four orders of magnitude
above the Gaussian rate: these are genuinely heavy tails, from a real sensor
stream. Evaluation is honest **leave-one-segment-out** cross-validation (each of
the seven segments is the held-out test set exactly once, so no within-segment
autocorrelation leaks train→test), estimators fit on the other six segments with
features standardized on the training rows, fixed a-priori hyperparameters (Huber
`δ = 1.345`, trimmed keep `0.9`) — nothing selected on any outcome.

Because a real held-out target is itself noisy (no noiseless truth to recover),
robustness is judged on **bulk** prediction. Which estimator wins each pooled
held-out metric (values: OLS vs the robust estimator it loses/wins against):

| target | RMSE (L2) | MAE | median-AE | 10 %-trim RMSE |
|--------|-----------|-----|-----------|----------------|
| `ENGINE_LOAD` | OLS 1.833 < 1.903 | **Huber 1.065** < 1.140 | **trim 0.63** < 0.81 | **trim 0.84** < 1.00 |
| `THROTTLE_POS` | OLS 1.674 < 1.784 | **Huber 0.892** < 0.964 | **trim 0.50** < 0.60 | **trim 0.62** < 0.80 |
| `MAF` | OLS 1.280 < 1.358 | **Huber 0.844** < 0.912 | **trim 0.50** < 0.71 | **trim 0.71** < 0.83 |

The split is perfectly consistent across all three real targets: **OLS wins RMSE**
(the L2 metric it alone minimizes, bought by chasing the tail spikes), while
**Huber and trimmed win MAE, median absolute error and trimmed RMSE** — every
bulk metric — because they refuse to let the heavy-tailed rows distort the fit.
The median-absolute-error gains are large (0.81→0.63, 0.60→0.50, 0.71→0.50): on
the typical held-out row a robust fit is materially closer.

The signed verdict, from the seeded paired bootstrap of per-row absolute-error
reduction (OLS − robust; positive ⇒ robust better; 95 % CI, 2000 resamples):

| target | Huber − OLS (abs-error) | Trimmed − OLS (abs-error) |
|--------|-------------------------|---------------------------|
| `ENGINE_LOAD` | +0.0752 [+0.0705, +0.0796] **wins** | +0.0260 [+0.0158, +0.0357] **wins** |
| `THROTTLE_POS` | +0.0721 [+0.0667, +0.0775] **wins** | +0.0571 [+0.0492, +0.0647] **wins** |
| `MAF` | +0.0679 [+0.0637, +0.0717] **wins** | +0.0468 [+0.0395, +0.0539] **wins** |

Every interval is strictly above zero: on this real workload robust regression
beats OLS on mean absolute error too — not merely on the tail-insensitive
summaries — with a margin that clears its own confidence interval in all six
target×method cells.

**Finding — yes; on real native heavy tails robustness wins, and the win is
metric-dependent exactly as theory predicts.** This is the program's first *real*
(non-semi-synthetic, no injection) workload where the robust regressors
decisively beat OLS, and it wins for the right reason: the OBD2 residuals are
natively heavy-tailed (kurtosis 10–45), so OLS's tail-chasing degrades its bulk
predictions, and Huber/trimmed — by bounding tail influence — predict the typical
held-out row markedly better. OLS retains only its home metric, RMSE.

**Interpretation.** Follow-up 3 predicted that the robust regressors reward one
specific regime — pervasive heavy-tailed vertical noise — and that the real
industrial RUL workload (lever 2's null) simply does not live in it. Axis 3
completes that argument from the other side: a *different* real workload that
*does* live in the heavy-tailed regime, and there the same estimators win, on
real data, under a preregistered protocol. The lesson is not "robustness is
better" or "worse" but **regime-matching** — the estimator has to fit the
contamination the data actually has. Automotive telemetry has heavy vertical
tails; turbofan degradation does not; the robust regressors help exactly where
the tails are, and the metric they help on (bulk vs L2) is itself diagnostic.

**Honest bounds.** (1) The win is on *bulk* metrics; if a downstream user's loss
is genuinely squared-error, OLS remains the RMSE-optimal choice — the honest
recommendation is "match the estimator to the loss". (2) Hyperparameters are
fixed a priori, not tuned; a tuned Huber/trim could widen or narrow the gap, but
the sign is robust across three targets and two estimators. (3) LOSO segments are
whole driving sessions, so segment-level regime shift (city vs motorway) is part
of the held-out difficulty — a feature, not a bug, of the real workload.

**Determinism.**

```
stdout  SHA-256: 0535a67bd4165b3f7fbb35f8c125da9e419d68f39d3398e1f1039179b5da7d73
results SHA-256: 32ad439fb4c536760b1f298e051b46a41c0c30469138a3b4206bd3e064bf9a93  (51 BenchRecords)
```

Run twice, byte-identical (built `--release`; release is equally deterministic);
no network; checksum-verified in-repo OBD2 CSV; OLS (QR), Huber-IRLS and trimmed
LS are RNG-free and the paired bootstrap is explicitly seeded.
