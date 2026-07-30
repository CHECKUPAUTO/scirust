# TDI-5.4 — Nonlinear Sufficiency and Horizon-Invariance: Confirmatory Results

## Status

**Confirmatory. Real preregistered run. Executed exactly once, in full.**

This document reports the outcome of the one and only real execution of the
TDI-5.4 experiment defined in
[`TDI-5.4-NONLINEAR-OVERLAP-SUFFICIENCY-PREREGISTRATION.md`](./TDI-5.4-NONLINEAR-OVERLAP-SUFFICIENCY-PREREGISTRATION.md).
That preregistration is the sole normative definition of the experiment;
this document does not restate the design, it reports outcomes against it.

Per the preregistration's Section 18 (interpretation boundaries) and
Section 19 (freeze rule), the classifications reported below are final and
must not be rewritten. Any transcription correction must take the form of a
visible erratum citing exact raw-output line numbers, never a silent edit of
a reported number or verdict. Any further scientific-code change or re-run
requires a new experiment identifier (TDI-5.5 or later).

## Abstract

TDI-5.4 asks whether the redundancy of the early-overlap predictor O₁ given
O₂ — established linearly at the primary horizon U₆ by TDI-5.3C — survives
when the model family is enriched with a fixed nonlinear basis (quadratic
terms O₁², O₂² and the interaction O₁·O₂), and whether that answer is
invariant across target horizons. It compares two nonlinear ridge layouts on
fresh, independent seed blocks D/E/F: **N2** (baseline + O₂ + O₂²) versus
**N12** (baseline + O₁ + O₂ + O₁² + O₂² + O₁·O₂). Because the 13 baseline
features stay linear, any N12-minus-N2 improvement is attributable solely to
O₁, its curvature, and its interaction with O₂.

The real run generated all 120,000 preregistered records with no anomalous
rejections. The result is an **asymmetry that is not horizon-invariant**. At
the primary horizon U₆, O₁'s nonlinear marginal contribution is classified
**Equivalent** (aggregate MSE reduction ≈1.2%, inside the ±2% margin): O₁
stays redundant given O₂ even under nonlinearity, so the linear TDI-5.3C
finding was **not** an artifact of linearity. But the marginal contribution
**decays monotonically with the prediction horizon** — ≈2.5% at U₃ (crossing
the 2% threshold: **Beneficial**), ≈1.5% at U₄ (**Inconclusive**), and ≈1.2%
at U₅/U₆/U₈ (**Equivalent**). O₁ therefore carries a genuine independent
nonlinear contribution at the shortest horizon that fades to practical
redundancy by the middle horizons. The horizon-invariance summary is
accordingly **not invariant** (2 of 4 secondary horizons match the
U₆ class). A subtle but important point: at every horizon the aggregate
bootstrap lower bound on the improvement is strictly positive, so O₁'s
nonlinear contribution is *statistically detectable everywhere* — it is only
*practically negligible* (below the preregistered 2% margin) from U₄ onward.

## Results at a glance

| Criterion | Question | Verdict |
|---|---|---|
| **TDI-5.4A** | Does O₁ contribute beyond O₂ under nonlinearity at U₆? | **Equivalent** |
| **TDI-5.4B** | Is that answer invariant across U₃/U₄/U₅/U₈? | **Not invariant** (U₃ Beneficial, U₄ Inconclusive, U₅/U₈ Equivalent) |

## 1. Provenance and integrity

| Field | Value |
|---|---|
| Git commit | `7fa8b88061ff9a2d64aa14a8007b654b272c2465` (merge of the frozen TDI-5.4 scaffold, PR #12) |
| Repository (execution host) | `/root/tdi-real-run` (dedicated clone, not a CI runner work directory) |
| Host | `tarek`, Linux 6.8.12-tegra aarch64 (NVIDIA Jetson) |
| rustc / cargo | 1.97.0 |
| Command | `target/release/tdi-independent-overlap-ablation-v54 --full` (invoked via `scripts/reproduce-tdi5.4.sh`) |
| Run start (UTC) | 2026-07-22T15:11:15Z |
| Run end (UTC) | 2026-07-22T15:24:36Z |
| Wall-clock duration | ≈ 13.5 minutes |
| v54 evaluator SHA-256 | `dcf24d7eb1ccd938a81163738c38d31a693474c8a1d94046734bda243ca772bf` |
| TDI-5.4 preregistration SHA-256 | `229a0a8efa391c67c4dda1322b984109b142be3abf972d0a08f3c4ac742ec6ac` |
| TDI-5.4 scientific-manifest SHA-256 | `6a57ac4bb4763bf9fb9bc3ad12683bbc952c7729d7387202f58a85222a2ce33f` |
| Frozen TDI-5.3 evaluator SHA-256 | `93181fb75d4882be2ca0b26c1babe9db747583d6e08e6992617ec12b7f65460f` |
| Frozen TDI-5.2 evaluator SHA-256 | `2308607729659c7546a17530e69773f982d9a1cf41656ea7898e0123ca469ef7` |
| Result log SHA-256 | `a4e729bfdba90dc6c29ff0230b44c96721049675098cc101991e9b81d2a1816e` |

All hashes above were printed by the evaluator itself as part of the required
raw output and were independently reverified against the frozen manifests
committed in this repository and against the result's own `.sha256` sidecar.
The run used exactly the committed frozen v54 evaluator (its hash equals the
committed `docs/TDI-5.4-…-EVALUATOR.sha256`, which the reproduction script
verifies before building). Raw artifacts are committed at:

- `results/tdi5.4-nonlinear-overlap-sufficiency/tdi-independent-overlap-ablation-v54.log` (complete stdout)
- `results/tdi5.4-nonlinear-overlap-sufficiency/tdi-independent-overlap-ablation-v54.log.sha256`
- `results/tdi5.4-nonlinear-overlap-sufficiency/tdi-independent-overlap-ablation-v54.metadata.txt`
- `results/tdi5.4-nonlinear-overlap-sufficiency/tdi-independent-overlap-ablation-v54.complete`

### Execution custody

This run was executed by the human project operator on their own hardware,
using the committed reproduction script and the exact preregistered
confirmation token (`TDI54_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI54_FREEZE_RULE`).
Per this project's standing governance constraint, the AI agent that authored
the TDI-5.4 code, tests and reproduction tooling never invoked, and must
never invoke, the evaluator's `--full` mode with the real confirmation value.
This document was produced by reading the already-committed, hash-verified
result artifacts after the fact; it does not and cannot alter them.

## 2. Execution integrity — population accounting

All 12 preregistered populations (3 seed blocks × 4 populations each) reached
their exact target accepted counts, for a total of **120,000 records**:

| Population | Requested | Accepted | Rejected (D / E / F) | Rejection reason |
|---|---|---|---|---|
| training-w3 | 15,000 | 15,000 | 85 / 54 / 55 | `observation-fully-recovered` (+1 `target-fully-recovered-h3` in D) |
| holdout-w3 | 5,000 | 5,000 | 18 / 17 / 19 | `observation-fully-recovered` (+1 `target-fully-recovered-h3` in E) |
| training-w4 | 15,000 | 15,000 | 0 / 0 / 0 | — |
| holdout-w4 | 5,000 | 5,000 | 0 / 0 / 0 | — |

Every rejection was one of the preregistered exclusion reasons, essentially
confined to the width-3 populations (the narrowest successor-set space) and a
small fraction of each (at most 85 of 15,085 attempts). Both width-4
populations in every block had zero rejections. No generation budget or
no-progress threshold was approached. This is a clean, anomaly-free
generation record.

## 3. Results by criterion

All MSE values below are in standardized U-space (the confirmatory space
defined by the preregistration); reconstructed-O-space and Spearman/bias/
calibration quantities are recorded in the raw log but are exploratory only
and are not reproduced here. Every criterion applies the frozen four-way
Beneficial/Equivalent/Harmful/Inconclusive classifier (inherited from
TDI-5.2 Section 13) with a symmetric ±2% relative-MSE margin.

### TDI-5.4A — nonlinear O₁ sufficiency at U₆ (N12 vs N2)

| Block | MSE (N2) | MSE (N12) | Relative reduction (median, 95% CI) |
|---|---|---|---|
| D | 0.240977 | 0.238243 | 1.14% [0.61%, 1.64%] |
| E | 0.246124 | 0.242688 | 1.40% [0.93%, 1.89%] |
| F | 0.242424 | 0.239675 | 1.13% [0.68%, 1.60%] |
| **Aggregate** | **0.243175** | **0.240202** | **1.22% [0.93%, 1.50%]** |

**Verdict: Equivalent.** `blocks_confirming_benefit: 0` and
`aggregate_relative_improvement_at_least_2_percent: false`, so the Beneficial
threshold was not reached; but `all_block_point_estimates_within_equivalence_margin: true`,
`block_intervals_within_equivalence_margin: 3` (all three blocks), and
`aggregate_interval_within_equivalence_margin: true`, so every condition for
Equivalent held. `blocks_confirming_harm: 0` ruled out Harmful. Note
`aggregate_bootstrap_lower_bound_positive: true`: N12's ≈1.2% improvement
over N2 is statistically distinguishable from zero, but bounded well inside
the ±2% margin — hence Equivalent, i.e. *detectably nonzero yet practically
negligible*.

**Adding O₁ and its nonlinear terms on top of O₂ does not materially improve
U₆ prediction. The redundancy of O₁ given O₂ is therefore not an artifact of
the linear model family** — it persists when the quadratic terms O₁², O₂² and
the interaction O₁·O₂ are all available.

### TDI-5.4B — horizon-invariance of the nonlinear O₁ contribution

The identical N12-vs-N2 classification, at each horizon:

| Horizon | Aggregate MSE reduction (median, 95% CI) | Classification |
|---|---|---|
| U₃ | 2.51% [2.12%, 2.90%] | **Beneficial** |
| U₄ | 1.49% [1.17%, 1.80%] | **Inconclusive** |
| U₅ | 1.22% [0.93%, 1.50%] | **Equivalent** |
| U₆ (primary, 5.4A) | 1.22% [0.93%, 1.50%] | **Equivalent** |
| U₈ | 1.19% [0.91%, 1.48%] | **Equivalent** |

**Horizon-invariance summary:** `primary_classification: Equivalent`,
`horizons_matching_primary_class: 2` (U₅ and U₈), `secondary_horizon_count: 4`,
`invariant: false`.

The per-horizon classifications reflect a clean, **monotone decay of O₁'s
nonlinear marginal value with prediction distance**:

- **U₃ — Beneficial.** At the nearest secondary horizon, all three blocks
  confirm a ≥2% improvement and the aggregate reduction (2.51%) exceeds the
  margin. O₁ carries a real independent nonlinear contribution here.
- **U₄ — Inconclusive.** The aggregate reduction (1.49%) sits inside the ±2%
  band by point estimate, but only one of three block confidence intervals
  lies wholly within the margin (`block_intervals_within_equivalence_margin: 1`),
  so it qualifies neither as Beneficial nor as Equivalent. This is the genuine
  transition horizon.
- **U₅, U₆, U₈ — Equivalent.** The reduction has settled to ≈1.2% with all
  three block intervals inside the margin: O₁ is practically redundant given
  O₂ at the primary and longer horizons.

## 4. Final verdicts (verbatim from required raw output)

```text
=== VERDICTS FINAUX (Section 15, items 18-19) ===
TDI-5.4A — contribution O1 nonlinéaire (N12 vs N2, U6) : equivalent
TDI-5.4B — classification U_3                          : beneficial
TDI-5.4B — classification U_4                          : inconclusive
TDI-5.4B — classification U_5                          : equivalent
TDI-5.4B — classification U_8                          : equivalent
TDI-5.4B — 2 horizon(s) secondaire(s) sur 4 concordent avec U6
TDI-5.4B — classification invariante à travers les horizons : non
```

## 5. Discussion

**The asymmetry is real but horizon-dependent.** TDI-5.3 established, under a
linear ridge model at U₆ only, that O₁ adds essentially nothing beyond O₂
(Equivalent). TDI-5.4 sharpens this in two ways. First, it confirms the
finding is **not a linear artifact**: at U₆, even with O₁², O₂² and O₁·O₂
available, O₁'s marginal contribution stays inside the ±2% margin (Equivalent).
Second, it shows the finding is **not universal across horizons**: at the
shortest horizon U₃, O₁'s nonlinear contribution rises to ≈2.5% and is
classified Beneficial. The full horizon profile is monotone —
2.51% (U₃) → 1.49% (U₄) → 1.22% (U₅) → 1.22% (U₆) → 1.19% (U₈) — a large drop
across the near horizons that plateaus around 1.2% from U₅ onward.

**Interpretation.** The nonlinear terms that distinguish N12 from N2 are the
O₁ main effect, its curvature O₁², and the interaction O₁·O₂. Their combined
predictive value is greatest when the prediction target is closest to the
observation window (U₃, two steps beyond the observation horizon h_obs = 2)
and decays as the target recedes. By the primary horizon and beyond, whatever
short-range structure O₁ adds beyond O₂ has washed out below the 2%
practical-significance threshold. This is consistent with, and refines, the
`delta_O = O₂ − O₁` linear-redundancy picture inherited from TDI-5.1/5.2: the
redundancy is close but not exact, its residual is nonlinear, and that
residual is only material at short range.

**"Equivalent" means bounded, not zero.** At every horizon the aggregate
improvement interval excludes zero (positive lower bound), so O₁ is never
*informationally* inert here — it is *practically* redundant from U₄ onward
under the preregistered 2% margin. Readers should not read "Equivalent" as
"O₁ carries no information"; it means "O₁'s independent nonlinear contribution
is confined within ±2% relative MSE".

**Scope.** This concerns the specific preregistered nonlinear basis (a single
quadratic-plus-interaction expansion of the two overlap predictors), the
frozen generator, and the ridge family. It does not speak to richer nonlinear
families (deep networks, trees, kernels).

## 6. Interpretation boundaries

Reproduced from Section 18 of the preregistration, and directly load-bearing
now that a real result exists:

A TDI-5.4 result establishes the (non)contribution of O₁ conditional on O₂
**within the frozen generator and the specific preregistered nonlinear basis
of Section 5** (quadratic + single pairwise interaction), replicated across
three seed blocks. It does **not** establish: sufficiency under arbitrary
nonlinear families; causal effects; universal validity across dynamical
systems; arbitrary-width calibration; implementation-independent replication;
or external empirical validity. The TDI-5.4A classification and the TDI-5.4B
per-horizon classifications and invariance summary **may not be rewritten
after observing the full result.**

## 7. Reproducibility

The full command sequence, hash verification, and artifact layout are
specified in Section 17 of the preregistration and implemented in
`scripts/reproduce-tdi5.4.sh`. This document interprets the result already
produced by that one real, human-confirmed execution
(`TDI54_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI54_FREEZE_RULE`); it is not an
invitation to re-run the experiment. Per the freeze rule (Section 19),
TDI-5.4 — including this result — may not be silently repeated or patched.
Any future scientific-code change, correction, or re-execution requires a new
experiment identifier.
