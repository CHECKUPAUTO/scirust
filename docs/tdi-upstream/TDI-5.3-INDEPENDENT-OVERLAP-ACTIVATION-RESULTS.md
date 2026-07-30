# TDI-5.3 — Independent Overlap Activation: Confirmatory Results

## Status

**Confirmatory. Real preregistered run. Executed exactly once, in full.**

This document reports the outcome of the one and only real execution of the
TDI-5.3 experiment defined in
[`TDI-5.3-INDEPENDENT-OVERLAP-ACTIVATION-PREREGISTRATION.md`](./TDI-5.3-INDEPENDENT-OVERLAP-ACTIVATION-PREREGISTRATION.md).
That preregistration is the sole normative definition of the experiment; this
document does not restate the design, it reports outcomes against it.

Per the preregistration's Section 9 (interpretation boundaries) and Section 10
(freeze rule), the classifications reported below are final and must not be
rewritten. Any transcription correction must take the form of a visible
erratum citing exact raw-output line numbers, never a silent edit of a
reported number or verdict. Any further scientific-code change or re-run
requires a new experiment identifier (TDI-5.4 or later); TDI-5.3 itself may
not be silently patched or repeated.

## Abstract

TDI-5.3 mechanically inherits the entire frozen scientific design of TDI-5.2
(deterministic seed-block generation, ridge regression over five feature
layouts, exact-rational candidate construction, paired and stratified-
aggregate bootstrap resampling) and adds only a human-gated activation
mechanism. Its first and only real run generated all 165,000 preregistered
records (18 populations across 3 independent seed blocks) with zero anomalous
rejections — every rejection that occurred was the single preregistered
exclusion reason (`observation-fully-recovered`), confined almost entirely to
width-3 populations. Of the five preregistered confirmatory criteria, four
succeeded outright: TDI-5.3A (joint signal of O_1+O_2 together), TDI-5.3B
(O_2 carries signal independent of O_1), TDI-5.3D (out-of-distribution
transfer to successor-set widths 5 and 6), and TDI-5.3E (the joint signal
holds at every secondary horizon U_3, U_4, U_5 and U_8, not only at the
primary horizon U_6). TDI-5.3C — whether O_1 carries signal independent of
O_2 — was classified **Equivalent**, not Beneficial: adding O_1 on top of O_2
changed aggregate MSE by an estimated 0.05%, far inside the preregistered 2%
equivalence margin. The result is an internally consistent, asymmetric
picture: O_2 alone recovers almost all of the joint predictive benefit; O_1
adds essentially nothing once O_2 is present, but is not interchangeable with
O_2 when O_2 is absent.

## Results at a glance

| Criterion | Question | Verdict |
|---|---|---|
| TDI-5.3A | Do O_1 and O_2 together carry joint signal beyond baseline (B0)? | **RÉUSSI (succeeded)** |
| TDI-5.3B | Does O_2 carry signal independent of O_1 (B1 → B12)? | **RÉUSSI (succeeded)** |
| TDI-5.3C | Does O_1 carry signal independent of O_2 (B2 → B12)? | **Equivalent** |
| TDI-5.3D | Does the joint signal transfer out-of-distribution (widths 5, 6)? | **RÉUSSI (succeeded)**, both widths |
| TDI-5.3E | Does the joint signal hold at every secondary horizon? | **RÉUSSI (succeeded)**, 4/4 horizons |

## 1. Provenance and integrity

| Field | Value |
|---|---|
| Git commit | `989a4c772024aa9ee1219d771d477357c42484f6` |
| Repository (execution host) | `/root/tdi-real-run` (dedicated clone, not a CI runner work directory) |
| Host | `tarek`, Linux 6.8.12-tegra aarch64 (NVIDIA Jetson) |
| rustc / cargo | 1.97.0 |
| Command | `target/release/tdi-independent-overlap-ablation-v53 --full` (invoked via `scripts/reproduce-tdi5.3.sh`) |
| Run start (UTC) | 2026-07-22T06:19:00Z |
| Run end (UTC) | 2026-07-22T09:43:33Z |
| Wall-clock duration | ≈ 3h 24m |
| v53 evaluator SHA-256 | `93181fb75d4882be2ca0b26c1babe9db747583d6e08e6992617ec12b7f65460f` |
| TDI-5.3 preregistration SHA-256 | `7223128dcfd751ebeb6488c01c3512d0a10b35937ec170504984295eb421682e` |
| TDI-5.3 scientific-manifest SHA-256 | `2659e0afae239074262b1900ff2d5f6754df5247a31a7e0c729fc3fda759e7c6` |
| Frozen TDI-5.2 evaluator SHA-256 | `2308607729659c7546a17530e69773f982d9a1cf41656ea7898e0123ca469ef7` |
| Frozen TDI-5.2 preregistration SHA-256 | `f57a054bc95eb2e041434d6e2049509b0dce1a5397f9666d274b1bbac332be35` |
| Result log SHA-256 | `fbe3feb2883d52de018e61e2ee5771a808fa334d756bd5de87c91d772be034ef` |

All hashes above were printed by the evaluator itself as part of the required
raw output and were independently reverified against the frozen manifests
already committed in this repository, and against the sidecar
`.sha256` file shipped with the result. The raw artifacts are committed at:

- `results/tdi5.3-independent-overlap-activation/tdi-independent-overlap-ablation-v53.log` (complete stdout, 3404 lines)
- `results/tdi5.3-independent-overlap-activation/tdi-independent-overlap-ablation-v53.log.sha256`
- `results/tdi5.3-independent-overlap-activation/tdi-independent-overlap-ablation-v53.metadata.txt`
- `results/tdi5.3-independent-overlap-activation/tdi-independent-overlap-ablation-v53.complete`

### Execution custody

This run was executed by the human project operator on their own hardware,
using the committed reproduction script and the exact preregistered
confirmation token (`TDI53_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI53_FREEZE_RULE`).
Per this project's standing governance constraint, the AI agent that
authored the TDI-5.3 code, tests, and reproduction tooling never invoked, and
must never invoke, the evaluator's `--full` mode with the real confirmation
value. This document was produced by reading the already-committed,
hash-verified result artifacts after the fact; it does not and cannot alter
them.

## 2. Execution integrity — population accounting

All 18 preregistered populations (3 seed blocks × 6 populations each) reached
their exact target accepted counts, for a total of **165,000 records**:

| Population | Requested | Accepted | Rejected | Rejection reason |
|---|---|---|---|---|
| training-w3 (×3 blocks) | 15,000 each | 15,000 each | 62 / 71 / 65 | `observation-fully-recovered` |
| holdout-w3 (×3 blocks) | 5,000 each | 5,000 each | 23 / 19 / 16 | `observation-fully-recovered` |
| training-w4 (×3 blocks) | 15,000 each | 15,000 each | 0 / 0 / 0 | — |
| holdout-w4 (×3 blocks) | 5,000 each | 5,000 each | 0 / 0 / 1 (block C only) | `observation-fully-recovered` |
| ood-w5 (×3 blocks) | 10,000 each | 10,000 each | 0 | — |
| ood-w6 (×3 blocks) | 5,000 each | 5,000 each | 0 | — |

Every rejection observed across all 165,000+ generation attempts was the
single preregistered exclusion reason, `observation-fully-recovered`; the two
other declared rejection categories (`InvalidTransformedTarget`,
`NonFiniteFeature`) never fired. Rejections were essentially confined to
width-3 populations (the narrowest, most constrained successor-set space) and
were a small fraction of each population (at most 71 out of 15,071
attempts). No generation budget or no-progress threshold was approached in
any of the 18 populations. This is a clean, anomaly-free generation record.

## 3. Results by criterion

Four feature layouts recur throughout: **B0** (13 baseline features only),
**B1** (B0 + O_1), **B2** (B0 + O_2), **B12** (B0 + O_1 + O_2). All MSE values
below are in standardized U_6 space (the confirmatory space defined by the
preregistration); reconstructed-O-space and Spearman/bias/calibration
quantities are recorded in the raw log but are exploratory only (Section 16
of the preregistration) and are not reproduced here.

### TDI-5.3A — joint signal (B12 vs. B0, U_6)

| Block | MSE (B0) | MSE (B12) | Relative reduction (median, 95% CI) |
|---|---|---|---|
| A | 0.336572 | 0.258431 | 23.2% [21.5%, 24.9%] |
| B | 0.353702 | 0.264096 | 25.4% [23.5%, 27.1%] |
| C | 0.336228 | 0.258465 | 23.1% [21.3%, 24.9%] |
| **Aggregate** | **0.342167** | **0.260330** | **23.9% [22.9%, 24.9%]** |

**Verdict: RÉUSSI.** All 7 preregistered sub-conditions were satisfied: lower
MSE in every block, positive block bootstrap lower bounds, median and
aggregate relative reduction both ≥ 15%, positive aggregate bootstrap lower
bound, Spearman improving in every block, aggregate bias not worse by more
than 0.02.

### TDI-5.3B — independent O_2 signal (B12 vs. B1, U_6)

| Block | MSE (B1) | MSE (B12) | Relative reduction (median, 95% CI) |
|---|---|---|---|
| A | 0.318849 | 0.258431 | 19.0% [17.3%, 20.6%] |
| B | 0.333168 | 0.264096 | 20.7% [19.0%, 22.4%] |
| C | 0.319013 | 0.258465 | 19.0% [17.2%, 20.7%] |
| **Aggregate** | **0.323676** | **0.260330** | **19.6% [18.6%, 20.5%]** |

**Verdict: RÉUSSI.** All 6 preregistered sub-conditions were satisfied: lower
MSE in every block, positive block bootstrap lower bounds, median and
aggregate relative reduction both ≥ 10%, Spearman not lower in any block,
aggregate bias not worse by more than 0.02.

### TDI-5.3C — independent O_1 contribution (B12 vs. B2, U_6)

| Block | MSE (B2) | MSE (B12) | Relative change (median, 95% CI) |
|---|---|---|---|
| A | 0.258553 | 0.258431 | 0.05% [0.0%, 0.09%] |
| B | 0.264246 | 0.264096 | 0.06% [−0.03%, 0.15%] |
| C | 0.258587 | 0.258465 | 0.05% [−0.06%, 0.15%] |
| **Aggregate** | **0.260462** | **0.260330** | **0.05% [0.004%, 0.099%]** |

**Verdict: Equivalent** (not Beneficial). B2 alone — O_2 without O_1 — is
already within a few hundredths of a percent of B12's MSE. `blocks_confirming_benefit: 0`
and `aggregate_relative_improvement_at_least_2_percent: false`, so the
preregistered Beneficial threshold was not reached; but
`all_block_point_estimates_within_equivalence_margin: true`,
`block_intervals_within_equivalence_margin: 3` (all three blocks), and
`aggregate_interval_within_equivalence_margin: true`, so every preregistered
condition for Equivalent was satisfied. Symmetrically,
`blocks_confirming_harm: 0` and the aggregate upper bound was not negative,
ruling out Harmful. Per Section 13 of the preregistration, Equivalent is a
fully legitimate preregistered outcome, not a failure of the experiment: TDI-5.3C
was preregistered precisely to test whether O_1 adds anything on top of O_2,
without forcing a positive result.

### TDI-5.3D — out-of-distribution transfer (B12 vs. B0)

Both widths test transfer of the same U_6 predictive task to inputs drawn
from wider, never-fitted successor-set spaces (the model is fit only on
combined width-3/width-4 holdout).

**Width 5:**

| Block | MSE (B0) | MSE (B12) | Relative reduction (median, 95% CI) |
|---|---|---|---|
| A | 0.130605 | 0.067204 | 48.5% [47.9%, 49.1%] |
| B | 0.136297 | 0.075352 | 44.7% [44.1%, 45.3%] |
| C | 0.136727 | 0.068315 | 50.0% [49.5%, 50.6%] |
| **Aggregate** | **0.134543** | **0.070290** | **47.8% [47.4%, 48.1%]** |

**Width 6:**

| Block | MSE (B0) | MSE (B12) | Relative reduction (median, 95% CI) |
|---|---|---|---|
| A | 0.390685 | 0.134370 | 65.6% [65.3%, 65.9%] |
| B | 0.421943 | 0.175011 | 58.5% [58.3%, 58.8%] |
| C | 0.429090 | 0.147057 | 65.7% [65.5%, 66.0%] |
| **Aggregate** | **0.413906** | **0.152146** | **63.2% [63.1%, 63.4%]** |

**Verdict: RÉUSSI** for both widths (all 8 width-5 sub-conditions and all 7
width-6 sub-conditions satisfied). The relative reductions are much larger
than in-distribution because the reference model B0 degrades catastrophically
out-of-distribution (aggregate standardized R² of −0.63 at width 5 and
**−11.9** at width 6 — worse than predicting the mean), whereas the challenger
B12, while also degraded relative to in-distribution performance, remains far
better calibrated (R² of +0.15 at width 5, −3.7 at width 6). These large
relative numbers should be read as "B12 degrades much less catastrophically
than B0 under distribution shift," not as evidence of strong absolute
out-of-distribution accuracy — see Interpretation Boundaries below.

### TDI-5.3E — multi-horizon trajectory (B12 vs. B0)

| Horizon | MSE (B0) | MSE (B12) | Relative reduction (median, 95% CI) |
|---|---|---|---|
| U_3 | 0.379084 | 0.196928 | 48.1% [47.1%, 48.9%] |
| U_4 | 0.352605 | 0.223267 | 36.7% [35.6%, 37.7%] |
| U_5 | 0.345165 | 0.245074 | 29.0% [27.9%, 30.0%] |
| U_6 (primary) | 0.342167 | 0.260330 | 23.9% [22.9%, 24.9%] |
| U_8 | 0.341052 | 0.281618 | 17.4% [16.4%, 18.4%] |

**Verdict: RÉUSSI.** `horizons_improving_in_every_block: 4` — all four
secondary horizons improved in every block, exceeding the minimum of 3
required. U_8 improved in every block, no block/horizon combination showed a
reduction below −5%, and both the average secondary reduction per block and
the aggregate reduction at every secondary horizon were positive.

The relative reduction decreases smoothly and **monotonically** from the
nearest secondary horizon (U_3, 48.1%) through the primary horizon (U_6,
23.9%) to the furthest secondary horizon (U_8, 17.4%). This is the shape one
would expect if early-overlap observations carry predictive information that
decays gracefully with prediction distance, rather than a discontinuous or
horizon-specific artifact.

## 4. Final verdicts (verbatim from required raw output)

```text
=== VERDICTS FINAUX (Section 17, items 18-19) ===
TDI-5.3A — signal joint                    : RÉUSSI
TDI-5.3B — signal O2 indépendant            : RÉUSSI
TDI-5.3C — classification O1 indépendante   : equivalent
TDI-5.3D — transfert OOD (largeurs 5 et 6)  : RÉUSSI
TDI-5.3E — trajectoire multi-horizon        : RÉUSSI
```

## 5. Discussion

**The O_1 / O_2 asymmetry.** The four single-layout aggregate MSE values,
read together, tell a consistent story:

| Layout | Features | Aggregate MSE |
|---|---|---|
| B0 | baseline only | 0.342167 |
| B1 | baseline + O_1 | 0.323676 |
| B2 | baseline + O_2 | 0.260462 |
| B12 | baseline + O_1 + O_2 | 0.260330 |

Adding O_1 alone to the baseline (B0 → B1) recovers only a small fraction of
the total improvement available; adding O_2 alone (B0 → B2) recovers almost
all of it. Adding O_1 on top of O_2 (B2 → B12) changes aggregate MSE by
about 0.05%, statistically indistinguishable from no effect at the
preregistered 2% margin. This is not a new observation in isolation: it is
consistent with, and now confirmatory of, the post-hoc linear-redundancy
pattern noted exploratorily in TDI-5.1 regarding `delta_O = O_2 − O_1`.
TDI-5.3 was preregistered specifically to test this asymmetry under a
confirmatory protocol, and the real result reproduces it: **O_2 is doing
essentially all of the work; O_1 is redundant given O_2, but O_1 is not a
substitute for O_2 when O_2 is absent** (B1 is much closer to B0 than to B12).

**Reconstructed-O-space intervals crossing zero.** Several of the
reconstructed-O_6-space bootstrap intervals for MSE/MAE improvement (recorded
in the raw log, not reproduced in the tables above) straddle zero in
individual blocks even where the corresponding standardized-U_6-space
criterion succeeds decisively. This is expected and is not a contradiction:
per Section 16 of the preregistration, reconstructed-space quantities are
exploratory only. The confirmatory criteria are defined exclusively in
standardized U-space, where all reductions were unambiguously positive with
bootstrap lower bounds above zero.

**Out-of-distribution effect sizes.** The width-5 and width-6 relative
reductions (47.8% and 63.2% aggregate) are the largest in the entire result,
but they are large because the reference model's absolute performance
collapses out-of-distribution (negative, and at width 6 strongly negative,
R²), not because the challenger generalizes well in an absolute sense. Both
models are extrapolating outside their fitting distribution; B12 simply
degrades less badly than B0. This distinction matters for how the result
should be read (see below).

## 6. Interpretation boundaries

Reproduced from Section 9 of the preregistration, and directly load-bearing
now that a real result exists:

A successful TDI-5.3 run establishes replicated predictive information from
early overlap observations, within the preregistered generator and linear
ridge model family. It does **not** establish: universal validity across all
dynamical systems; causal intervention effects; nonlinear sufficiency;
arbitrary-width calibration; implementation-independent replication; or
external empirical validity. The independent contribution of O_1 (TDI-5.3C)
may be classified as Beneficial, Equivalent, Harmful or Inconclusive — the
real result classified it Equivalent, and per the preregistration **that
classification may not be rewritten after observing the full result.**

Concretely, this result does not claim that O_1 carries no information in
any general sense — only that, within this preregistered generator, this
linear ridge model family, and this exact protocol, O_1 adds no detectable
independent value beyond what O_2 already provides at the U_6 primary
horizon (and, per Section 4, this finding is defined only at U_6; TDI-5.3C
was not repeated at the secondary horizons).

## 7. Reproducibility

The full command sequence, hash verification, and artifact layout are
specified in Section 8 of the preregistration and implemented in
`scripts/reproduce-tdi5.3.sh`. This document exists to interpret the result
already produced by that one real, human-confirmed execution
(`TDI53_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI53_FREEZE_RULE`); it is not an
invitation to re-run the experiment. Per the freeze rule (Section 10 of the
preregistration), TDI-5.3 — including this result — may not be silently
repeated or patched. Any future scientific-code change, correction, or
re-execution requires a new experiment identifier.
