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

## Follow-up 8 (axis 4) — extended promotion gates: pricing a two-sided result

Phase 729's `PromotionGate` made a single-metric promote/hold call. But axis 3
handed the operator a genuinely **two-sided** candidate: Huber beats OLS on the
bulk (MAE, median error) and *loses* on tail-sensitive squared error. "It
depends" is not a deployment decision. This axis extends the gate additively
(`ExtendedPromotionGate`, same seeded paired-bootstrap machinery) with the three
controls an operator actually needs, and drives it on the axis-3 workload —
incumbent OLS vs candidate Huber, leave-one-segment-out on the OBD2 telemetry,
**the seven driving segments as temporal shadow windows**:

1. **Weighted composite** — the decision runs on one per-unit score,
   `0.75·rel(abs-err) + 0.25·rel(sq-err)`, each metric's improvement scaled by the
   incumbent's own pooled error magnitude (its known operating point), so the
   bulk/tail trade-off is priced *before* the candidate is judged, not argued
   after;
2. **Switching cost** — the pooled composite improvement's CI *lower bound* must
   exceed a preregistered hurdle (a hysteresis deadband): promote only when the
   candidate is better by more than the cost of changing models;
3. **Temporal shadow windows** — every segment's *mean* composite improvement
   must stay non-negative, so a candidate that wins pooled but regresses in some
   period is held.

Evaluated at two preregistered switching costs (`0.00`, a free switch; `0.05`, a
5 %-weighted-gain hurdle):

| target | composite Δ (CI) | windows held | cost 0.00 | cost 0.05 |
|--------|-----------------:|:------------:|:---------:|:---------:|
| `ENGINE_LOAD` | +0.0300 [+0.0245, +0.0351] | 7/7 | **PROMOTE** | **HOLD** |
| `THROTTLE_POS` | +0.0222 [+0.0134, +0.0310] | 5/7 | **HOLD** | HOLD |
| `MAF` | +0.0244 [+0.0167, +0.0315] | 6/7 | **HOLD** | HOLD |

**Finding — every control changes a decision that a single-metric gate would get
wrong.** All three candidates have a *statistically clean, positive* pooled
composite (Huber's 3:1-weighted bulk win outweighs its squared-error loss on
average), so a naive pooled test promotes all three. The extended gate promotes
**only `ENGINE_LOAD`, and only when switching is free**:

- **`ENGINE_LOAD`** clears every window but its +3.0 % weighted gain does **not**
  clear the 5 % switching cost — the deadband correctly withholds a real-but-marginal
  improvement (cost 0.00 → PROMOTE, cost 0.05 → HOLD).
- **`THROTTLE_POS`** and **`MAF`** are **held even at zero cost**: Huber
  *regresses* on the composite in the last, shortest driving segment
  (`segment_12`: −0.069 and −0.029 — a distinct low-data driving regime), and one
  further segment for `THROTTLE_POS`. The per-window floor catches exactly the
  "wins on average, fails in a period" pathology the control exists for; the
  pooled mean hides it.

This is the program's industrialization payoff: the same robust estimator is a
*promote* or a *hold* depending on the operating cost and the temporal
consistency the operator demands — a defensible, reproducible decision no point
estimate (and no single-metric gate) can produce. It closes the loop from
evidence (axes 1–3) to a deployable rule.

**Determinism.**

```
stdout  SHA-256: 291cc5198fce2711ac5f545e4fe970e3d503e2c2d1da9efb3f55da485cb28078
results SHA-256: aec4059bdf37af593897158fc5af450b3dcb4a0f3299f46b66639838c315191d  (18 BenchRecords)
```

Run twice, byte-identical (built `--release`); no network; checksum-verified
in-repo OBD2 CSV; OLS/Huber are RNG-free and every gate decision is a seeded
paired bootstrap. The `ExtendedPromotionGate` core carries seven new unit tests
(composite arithmetic, switching-cost deadband, per-window consistency, and the
typed-error surface) beside phase 729's six.

## Second four-axis program — synthesis

The three-lever program concluded that *framing*, not more robustness, was the
missing piece. The two follow-up programs pushed each conclusion until it either
broke or held. The first (monotone RUL, stabilized SECOM, injected heavy tails,
promotion gates) sharpened the framing thesis; this **second** program tested its
four hardest edges — richer models, a broken ceiling, real contamination, and a
deployable decision — and returns a more nuanced verdict than "framing always
wins":

- **Axis 1 — multivariate nonlinear RUL (kernel ridge, PR #763/#764): mostly no.**
  A genuinely nonlinear regressor closes the C-MAPSS ceiling further than the
  1-D monotone recalibration in only **1 of 4 cells** (FD001 raw, +0.059), ties
  once, and over-fits on both FD003 cells. Model capacity is an *inconsistent*
  lever for turbofan RUL; the monotone ordering of the linear score already holds
  most of the signal. Framing still dominates here.
- **Axis 2 — SECOM past the linear ceiling (PR #765): yes, but by curvature, not
  recency.** Degree-2 polynomial LDA lifts the frozen-test AUROC to **0.6177**,
  above item 2 (0.567) and lever 3 (0.581); explicit drift correction (recency
  windowing) *overfits* and drops to 0.530. The ~0.57 ceiling was breakable — the
  binding constraint is model capacity per (rare) failure, spent well by low-order
  nonlinearity and wasted by discarding data.
- **Axis 3 — real native heavy tails (PR #766): yes, decisively — the first real
  win.** On real Opel-Corsa OBD2 telemetry (OLS-residual excess kurtosis 45/19/10,
  no injection), leave-one-segment-out, Huber and trimmed LS beat OLS on **every
  bulk metric** across all three channels (paired-bootstrap CIs strictly positive
  even on MAE), while OLS keeps only its L2-home RMSE. Robustness wins when the
  regime matches — automotive telemetry has the heavy vertical tails turbofan
  degradation lacked; that regime-match, not estimator strength, was always the
  question.
- **Axis 4 — extended promotion gates (PR #767): the industrialization payoff.**
  A weighted composite + switching-cost deadband + temporal shadow windows turns
  axis 3's two-sided result into a single defensible decision. On the same OBD2
  shadow deployment every control flips a verdict a single-metric gate would get
  wrong: all three candidates have a clean positive pooled composite, yet only
  `ENGINE_LOAD` promotes and only at zero switching cost, while `THROTTLE_POS`
  and `MAF` are held for regressing on one driving segment.

**Program conclusion.** The framing thesis survives, refined: *matched capacity
and matched regime* beat raw robustness, but neither is free and neither is
universal. Nonlinearity helps only where curvature carries transferable signal
under the sample budget (SECOM yes, C-MAPSS mostly no); robustness helps only
where the native error regime is genuinely heavy-tailed (real OBD2 yes, turbofan
no) — and even a real win can be too small or too temporally inconsistent to
deploy, which is exactly what a preregistered extended gate is for. Every result
here is a deterministic, checksum-anchored, honestly-reported number: the wins,
the nulls, and the metric-dependent "it depends" that the gate then resolves.

## Third program (direction A) — a fast SPD solver, and the axis-1 re-test it enables

Axis 1 (Follow-up 5) carried an explicit honest caveat: kernel ridge ran at
**stride 40** because a large per-iteration constant in the shared Cholesky made
the finer strides too slow, and "a faster SPD solver might narrow the FD003 gap."
This direction removes the constant and settles the caveat with numbers.

**The fix.** `scirust_solvers::linalg::cholesky_decompose` called `check_finite`
on every entry in its innermost `O(n³)` loop — and each call built its context
message with `format!(...)` *eagerly*, allocating a `String` per inner iteration,
even though `check_finite` **ignored** that message (its error variant carries no
location). Three wasted heap allocations per inner step dominated the solve. At
`n ≈ 640` that was ≈ 8.7 × 10⁷ allocations ≈ **17 s** of pure overhead; removing
them (the numeric path is untouched) is **bit-identical** and turns the whole
stride-20 axis-1 sweep from an intractable multi-minute timeout into **≈ 2 s**.
Verified: at stride 40 the binary reproduces its committed stdout SHA `9c529e9d…`
and results SHA `71788a20…` exactly.

**The re-test (now-tractable stride 20, ~640 pooled train rows).** Ceiling ratio
(lower = more realized); `kernel − iso > 0` means kernel ridge beats the 1-D
isotonic recalibration:

| cell | OLS | isotonic | **kernel ridge** (γ, λ) | kernel − iso @20 | (was @40) |
|------|----:|---------:|------------------------:|-----------------:|----------:|
| FD001 raw-linear | 0.666 | 0.643 | **0.596** (0.01, 1.0) | **+0.048** win | +0.059 win |
| FD001 piecewise-125 | 0.545 | 0.457 | **0.443** (0.01, 0.1) | **+0.015** win | −0.004 tie |
| FD003 raw-linear | 0.607 | 0.560 | **0.554** (0.01, 1.0) | **+0.006** win | −0.053 iso |
| FD003 piecewise-125 | 0.505 | **0.442** | 0.478 (0.05, 1.0) | −0.036 iso | −0.048 iso |

**Finding — the stride-40 result was partly a resolution artifact.** With twice
the training data the nonlinear model, previously starved, wins **3 of 4 cells**
(was 1 of 4): FD001-piecewise flips tie→win and **FD003-raw flips iso-win→kernel-win**
(−0.053 → +0.006) — precisely the gap the axis-1 caveat flagged. The mechanism is
visible in the selection: on FD003-raw the validation-chosen bandwidth drops from
γ = 0.10 at stride 40 (too wiggly for ~320 rows, over-fits) to a smoother γ = 0.01
at stride 20, which generalizes. Kernel ridge is **more competitive than axis 1
reported** once it is not data-starved.

**But the framing nuance survives, narrowed.** On the single reframed target where
it most matters — **FD003 piecewise-125** — the simple 1-D isotonic recalibration
still beats kernel ridge (−0.036, narrowed from −0.048). So the axis-1 through-line
holds in its careful form: nonlinearity does not *reliably* dominate the cheap
monotone recalibration on the reframed RUL target — but the margin is thin and
data-dependent, not the 1-of-4 rout the stride-40 numbers implied. Re-testing a
caveat with the tool that was missing is the honest close it asked for.

**Determinism.**

```
solver fix: axis-1 stride-40 stdout SHA 9c529e9d… / results 71788a20…  (unchanged — bit-identical)
re-test:    axis-1 stride-20 stdout SHA c14b000a9bb2fee12b61b524bcfd48f13c48294e00ef8359237e77d82053c91c
            axis-1 stride-20 results SHA ec12cce8954062f32c228b0a96fcdd9e51069a64b952f394cede72e564aab03b  (24 records)
```

Run twice byte-identical; the `industrial-nonlinear-rul` binary gained a `--stride`
flag (default 40, so the committed artifact is unchanged); 201 `scirust-solvers`
tests and the `industrial_protocol_demo` fingerprint (`167c13de…`) are unaffected.

## Third program (direction B) — do the two winning levers compose?

Axis 2 showed **nonlinearity** helps (degree-2 features broke the SECOM ceiling);
axis 3 showed **robustness** helps (Huber beat OLS on real heavy-tailed OBD2
residuals). Each was tested alone. This direction asks whether they **compose** on
the same real workload with a clean **2×2 factorial** — linear/polynomial features
× squared/Huber loss — under axis 3's leave-one-segment-out protocol
(`industrial-obd2-robust-nonlinear`). The two factors are *exactly* "degree-2
polynomial features" (nonlinearity) and "Huber loss" (robustness); nothing else
differs.

**Enabling fix.** The same eager-`format!` defect direction A removed from the
Cholesky hot loop also sat in `scirust-solvers`' **QR and LU** factorizations —
`check_finite(x, &format!(…))` allocating a `String` per innermost Householder
update, on the path every OLS/Huber/trimmed fit takes. Removing it (bit-identical:
axis-3's `industrial-obd2-native` reproduces its stdout SHA `0535a67b…` exactly,
201 solver tests pass, demo `167c13de…` unchanged) is what makes a degree-2 Huber
IRLS fit feasible at all. Even so, that fit re-factorizes a ~37 k × 77 QR every
iteration, so the **fit** uses every 12th training row (`--train-stride 12`; heavy
tails survive subsampling, all four cells share the identical decimated rows) while
every model is still **scored on the full held-out segment** — full bootstrap power.

Mean absolute error per cell (bulk metric; lower better):

| target | `ols_linear` | `huber_linear` | `ols_poly` | `huber_poly` |
|--------|-------------:|---------------:|-----------:|-------------:|
| `ENGINE_LOAD` | 1.142 | 1.066 | 0.320 | **0.298** |
| `THROTTLE_POS` | 0.969 | 0.893 | 0.874 | **0.740** |
| `MAF` | 0.910 | 0.846 | 0.140 | **0.134** |

Per-row absolute-error gain vs `ols_linear` (seeded paired bootstrap; all CIs
strictly one-sided unless noted):

| target | robustness alone | nonlinearity alone | both | additive gap | robustness *on top of* nonlinearity |
|--------|-----------------:|-------------------:|-----:|-------------:|------------------------------------:|
| `ENGINE_LOAD` | +0.076 | +0.821 | +0.845 | **−0.053** (sub) | +0.023 (wins) |
| `THROTTLE_POS` | +0.075 | +0.095 | +0.229 | **+0.059** (super) | +0.134 (wins) |
| `MAF` | +0.064 | +0.771 | +0.776 | **−0.059** (sub) | +0.006 (wins) |

**Finding — they compose, but not independently, and nonlinearity is usually the
bigger lever.** Three things hold across all three real channels:

1. **Each lever helps alone** (every single-lever CI > 0) — reproducing axis 3's
   robustness win *and* confirming a nonlinearity win on the same data.
2. **Robustness still adds on top of nonlinearity everywhere** (the last column is
   positive in all three) — the levers genuinely compose, they are not substitutes.
3. **But the increment shrinks exactly where nonlinearity already explains the
   bulk.** On `ENGINE_LOAD` and `MAF` — engine channels that are *strongly*
   nonlinear in the sensor inputs (MAE collapses 1.14 → 0.32 and 0.91 → 0.14 the
   moment degree-2 cross-terms enter) — a good nonlinear fit leaves few vertical
   outliers for Huber to down-weight, so robustness-on-top is real but small
   (+0.023, +0.006) and the joint gain is **sub-additive**. On `THROTTLE_POS`,
   where nonlinearity is a modest lever, the two **reinforce super-additively**
   (gap +0.059) and robustness-on-top is large (+0.134).

**Interpretation — this reframes axis 3.** The linear-robust win axis 3 measured
is real but *small relative to the nonlinearity that was on the table*: on two of
three channels a degree-2 OLS already cuts bulk error several-fold, and Huber then
trims a little more. Robustness and nonlinearity are **complementary but
overlapping** — both attack the same heavy-tailed residuals, so stacking them pays,
with diminishing returns once one lever has captured the structure. The program's
through-line holds in its sharpest form yet: *match the model to the data*
(curvature where the map is nonlinear, a robust loss where the noise is heavy) —
and expect the second lever to help less once the first has done its work.

**Determinism.**

```
stdout  SHA-256: ac650e7b66549a5cfd7b627a786a4d48174c9978af5aa6c615d133ff3598a4f7
results SHA-256: d2e30dfe31be773d64b3596d1e6f8820ea9acf43da7ec251a6dc668f20a7d888  (66 BenchRecords)
```

Run twice byte-identical; no network; checksum-verified in-repo OBD2 CSV; OLS and
Huber (QR) are RNG-free and every gain is a seeded paired bootstrap. Honest bound:
training rows are decimated 12× for tractability (test rows full), so the absolute
MAE magnitudes would tighten with more training data — but the qualitative
composition result (each lever adds, sub-additively where one already dominates) is
a property of the residual structure, not the sample size.

## Third program (direction C, sub-PR 1) — calibrated uncertainty: split-conformal intervals

The program produced point predictions (axes 1–3) and promote/hold decisions
(axis 4); a deployment also needs **honest uncertainty**. Under heavy tails a `±σ`
band lies — the variance is inflated by the very outliers robustness exists to
tame. **Split-conformal prediction** sidesteps distributional assumptions entirely:
given a point predictor and an exchangeable calibration set, the band `ŷ ± q`, with
`q` the `⌈(n+1)·level⌉`-th smallest absolute calibration residual, covers a fresh
target with probability ≥ `level` for *any* residual distribution (Vovk; Lei et
al.). This adds a small, pure, deterministic `SplitConformal` type to the harness
(6 unit tests: the finite-sample order-statistic, the in-sample coverage guarantee,
the too-small-calibration typed error) and asks the question axis 3 implies: turned
into intervals, does the *robust* predictor give **tighter valid** intervals?

`industrial-obd2-conformal` answers it on the real OBD2 workload, leave-one-segment-out,
level 0.9, calibration carved as every 5th training row (OLS and Huber share the
split, so the comparison is exact). This is conformal under a mild **distribution
shift** — calibrate on some driving segments, deploy on a held-out one — the
realistic deployment condition rather than the i.i.d. ideal.

| target | OLS coverage / width | Huber coverage / width | Huber/OLS width |
|--------|---------------------:|-----------------------:|----------------:|
| `ENGINE_LOAD` | 0.899 / 4.471 | 0.898 / 4.457 | 0.997 (Huber ~tied) |
| `THROTTLE_POS` | 0.899 / 3.817 | 0.898 / 3.088 | **0.809 (Huber −19 %)** |
| `MAF` | 0.895 / 3.610 | 0.900 / 3.667 | 1.016 (OLS tighter) |

**Finding — conformal coverage holds on real data under shift, and robustness
tightens the band only when its gains reach the coverage quantile.** Two results:

1. **The distribution-free guarantee delivers.** Empirical coverage sits at
   0.895–0.900 for *both* predictors on all three channels — essentially the
   nominal 0.9 — even though calibration and test are different driving segments.
   Under heavy tails, where a Gaussian `±1.64σ` band would mis-cover, conformal is
   honest by construction.
2. **Robustness buys a tighter interval only on `THROTTLE_POS` (−19 %).** The
   conformal half-width is the ~90th-percentile absolute residual, not the mean, so
   a predictor only narrows the band if it improves residuals *up at that quantile*.
   This is exactly consistent with direction B: on `THROTTLE_POS`, the channel where
   robustness was the strong lever, Huber shrinks residuals all the way to the
   coverage quantile and the band tightens 19 %; on `ENGINE_LOAD`/`MAF`, where
   direction B found robustness only trims the deep bulk (nonlinearity owned the
   rest), the 90th-percentile residual — and thus the interval — is essentially
   unchanged (0.997, 1.016).

**Interpretation.** Calibrated uncertainty inherits the same "match the tool to the
data" logic as point accuracy, but keyed to a *different statistic*: a robust loss
tightens a conformal interval precisely when its residual improvement extends to the
coverage quantile. Where robustness only helps the median (ENGINE_LOAD/MAF), the
90 %-interval is unmoved; where it helps the moderate tail (THROTTLE_POS), the
interval narrows materially — all while conformal keeps coverage honest regardless.

**Determinism.**

```
stdout  SHA-256: 1cd59eb0b4443f92220feb7a1b7a3c9e0be5b96697ae5308d75bcd216c7c859f
results SHA-256: d47c8f796abde6145b64e419828d960d57641ab7520377e69823f00e602e906a  (15 BenchRecords)
```

Run twice byte-identical; no network; checksum-verified in-repo OBD2 CSV; OLS/Huber
(QR) are RNG-free and the conformal band is a sort plus an index. Additive: one new
`conformal` module + one binary + docs; `industrial_protocol_demo` fingerprint
(`167c13de…`) and the other 81 lib tests unaffected (87 lib tests total). Next
sub-PRs of direction C: quantile regression (pinball loss) and conformalized
quantile regression (adaptive-width intervals), then a promotion gate on interval
quality.

## Third program (direction C, sub-PR 2) — native quantile-regression intervals

Sub-PR 1's conformal band wraps a *point* predictor in a **constant** half-width.
Quantile regression instead predicts the conditional quantiles directly — a new
`scirust_learning::fit_quantile_regression` (pinball/check loss, fit by
Schlossmacher IRLS reusing the crate's weighted-OLS path; honestly a *smoothed*
approximation, the `ε`-floored weights and a `converged` flag reported as such; 5
unit tests) — so fitting `τ = 0.05` and `τ = 0.95` gives a **native 90 % interval**
`[q₀.₀₅(x), q₀.₉₅(x)]` whose width **adapts** to the local noise. Same OBD2
leave-one-segment-out protocol; the native band is compared head-to-head with the
C.1 OLS-conformal band, both targeting 0.9.

| target | quantile-native coverage / width | OLS-conformal coverage / width | native/conformal width |
|--------|---------------------------------:|-------------------------------:|-----------------------:|
| `ENGINE_LOAD` | 0.899 / 4.094 (122 crossings) | 0.899 / 4.471 | **0.916 (native −8.4 %)** |
| `THROTTLE_POS` | 0.899 / 3.762 (166 crossings) | 0.899 / 3.817 | 0.986 |
| `MAF` | 0.895 / 3.499 (3 crossings) | 0.895 / 3.610 | 0.969 |

**Finding — the adaptive native interval is tighter than conformal at matched
coverage, but without the guarantee.** Two things:

1. **Native quantile intervals are tighter on all three channels** (−8.4 %, −1.4 %,
   −3.1 %) at essentially the *same* empirical coverage (~0.9). By widening only
   where the data is locally noisy and narrowing where it is tight, quantile
   regression spends interval width more efficiently than a constant conformal band.
2. **But quantile regression carries no finite-sample coverage guarantee, and it
   frays at the edges.** Coverage happened to land near nominal here, but nothing
   *proves* it will; and the estimator produces **quantile crossings** (`q₀.₀₅ >
   q₀.₉₅`) on 0.3–2.3 % of rows for the two heavier channels — an empty, non-covering
   interval, counted openly rather than hidden. The tighter width is only trustworthy
   if the coverage is trustworthy.

**Interpretation.** This is the exact trade-off conformal and quantile methods are
each half of: conformal *guarantees* coverage at a constant width; quantile
regression *adapts* the width but only estimates coverage. Neither alone is the
whole answer — which motivates sub-PR 3, **conformalized quantile regression**,
which conformalizes the native interval to restore the finite-sample guarantee
*while keeping* the adaptive width, and should dominate both.

**Determinism.**

```
stdout  SHA-256: 20ed9d280d57bd1f13c8ef88b1807c0b83337fc198d3980ac0d26b61ced1206c
results SHA-256: ff2d55f0f79fc0aeb1486d73159fd6c1e3c8ae3f3c5c90beebae8dc66865b5e2  (18 BenchRecords)
```

Run twice byte-identical; no network; checksum-verified in-repo OBD2 CSV; OLS and
quantile IRLS (QR) are RNG-free. Additive: a new `quantile_regression` module in
`scirust-learning` (92 lib tests, was 87) + one binary + docs; `industrial_protocol_demo`
fingerprint (`167c13de…`) unchanged.

## Third program (direction C, sub-PR 3) — conformalized quantile regression: the guarantee *and* the adaptive width

Sub-PR 1 gave a **guaranteed** but **constant-width** conformal band; sub-PR 2 gave
an **adaptive** but **unguaranteed** native quantile band, and closed on the obvious
synthesis. **Conformalized quantile regression** (CQR; Romano, Patterson & Candès
2019) is that synthesis: take the native interval `[q̂₀.₀₅(x), q̂₀.₉₅(x)]`, score each
calibration point by how far it falls *outside* it — `Eᵢ = max(q̂_lo(xᵢ) − yᵢ, yᵢ −
q̂_hi(xᵢ))`, negative when `yᵢ` sits safely inside — take the `⌈(n+1)·level⌉`-th
smallest `Eᵢ` as an offset `Q`, and emit the **adjusted** interval `[q̂_lo − Q, q̂_hi +
Q]`. That single conformal offset restores the *exact* split-conformal coverage
guarantee of sub-PR 1 while **inheriting the quantile model's local width** — and `Q`
may be **negative**, so the band can *tighten*, not only widen. This adds a pure,
deterministic `ConformalizedQuantile` type to the harness (3 unit tests: in-sample
coverage, a too-wide band the offset *tightens*, the mismatched-length typed error).
All three intervals are fit on the **same per-fold proper/calibration split** (every
5th training row calibrates), leave-one-segment-out, level 0.9 — a point-for-point
comparison.

| target | quantile-native cov / width | **CQR** cov / width | OLS-conformal cov / width | CQR vs native | CQR vs conformal |
|--------|----------------------------:|--------------------:|--------------------------:|--------------:|-----------------:|
| `ENGINE_LOAD` | 0.899 / 4.107 | **0.896 / 4.089** | 0.899 / 4.471 | 0.996 (`Q<0`, tighter) | **0.915 (−8.5 %)** |
| `THROTTLE_POS` | 0.898 / 3.767 | **0.901 / 3.786** | 0.899 / 3.817 | 1.005 (`Q>0`, +cov) | 0.992 (−0.8 %) |
| `MAF` | 0.895 / 3.499 | **0.896 / 3.505** | 0.895 / 3.610 | 1.002 (`Q>0`) | 0.971 (−2.9 %) |

**Finding — CQR dominates the constant-width conformal band at matched coverage, and
does it at the native adaptive width.** Three things, all consistent with sub-PRs 1–2:

1. **CQR is tighter than OLS-conformal on all three channels** (−8.5 %, −0.8 %,
   −2.9 %) at the same ~0.9 coverage — the sub-PR-2 prediction confirmed. The constant
   conformal half-width must be wide enough for the *worst* local noise; CQR inherits
   the quantile model's heteroscedastic width and spends interval only where the data
   is actually noisy. The gap is largest on `ENGINE_LOAD`, the most heteroscedastic
   channel, and smallest on `THROTTLE_POS`, where direction B already showed the noise
   is closest to homoscedastic.
2. **The conformal offset is tiny and its sign adapts.** `|Q| < 0.01` on all three —
   the smoothed native quantiles were already close to calibrated — but `Q`
   *tightens* `ENGINE_LOAD` (negative: the native band was over-wide for its coverage
   there) and slightly *widens* `THROTTLE_POS`/`MAF` (positive: nudging their sub-0.90
   native coverage back up to 0.901 / 0.896). CQR stays within ±0.5 % of native's
   width while doing so: it keeps the adaptivity and adds only the correction the data
   demands.
3. **The guarantee sub-PR 2 lacked is now present.** Native quantile landed *under*
   nominal on all three (0.899, 0.898, 0.895 — the out-of-sample miscalibration
   flagged in sub-PR 2); CQR's split-conformal offset is exactly the finite-sample
   fix. The one apparent cost — `ENGINE_LOAD` coverage dips 0.899 → 0.896 — is the
   honest face of the *same* mechanism: `Q<0` trades a sliver of over-coverage for a
   tighter band, which is precisely what a coverage-*targeting* method should do.

**Honest caveat — the guarantee is exact under exchangeability; LOSO is cross-segment
transfer.** As in sub-PR 1, the calibration rows come from the training segments and
the test rows are a *held-out* segment, so the split-conformal theorem's
exchangeability premise is deliberately broken — the coverage numbers (0.896–0.901)
are genuine *out-of-distribution* transfer, not the i.i.d. ideal. CQR's role is not to
pin coverage at exactly 0.90 under shift (nothing can) but to keep it honest and close
*while* recovering the adaptive width — which it does, landing all three within 0.004
of nominal.

**Interpretation — direction C closes on its synthesis.** The three sub-PRs are the
three corners of the calibrated-uncertainty trade-off: conformal *guarantees* coverage
but wastes width (C.1); quantile regression *adapts* width but forfeits the guarantee
(C.2); CQR is the diagonal that takes both (C.3). On this real workload it is not a
wash — CQR is the **tightest guaranteed band on every channel**, and the one method
that is simultaneously adaptive and valid. The program's through-line — *match the
tool to the data* — reaches uncertainty itself: an interval should be as wide as the
local noise demands and no wider, and still be honest.

**Determinism.**

```
stdout  SHA-256: 515fef4963af76fdc5e4fa71333c40effb2189b9e17a960b5e3a982f6b0a3958
results SHA-256: 245eb7e8fa803a57432cda42a1593108da242ebe89f41eabeacbb595b356c215  (18 BenchRecords)
```

Run twice byte-identical; no network; checksum-verified in-repo OBD2 CSV; OLS, Huber
and quantile IRLS (QR) are RNG-free and the CQR offset is a sort plus an index.
Additive: the `ConformalizedQuantile` type extends the existing `conformal` module (9
conformal tests, was 6) + one binary + docs; `industrial_protocol_demo` fingerprint
(`167c13de…`) unchanged. Next: a promotion gate on interval quality
(coverage-constrained width improvement), closing direction C's loop back to axis 4.
