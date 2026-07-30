# TDI-6.6 — Re-standardized Cross-Generator Transfer

## Preregistration

This document is frozen before the evaluator is written and before any
confirmatory data is generated. Every constant, layout, criterion and
classification rule below is fixed in advance. Nothing here may be rewritten
after observing a result.

**Freeze rule.** The real confirmatory run is a single, deliberate, human-only
action behind the exact confirmation token of Section 20. No commit, test, or
CI run may supply the token. The authoring agent must never invoke `--full`
with the real token.

## 1. Experimental status, provenance, and the single changed factor

TDI-6.6 derives from **TDI-6.5** (`tdi-independent-overlap-ablation-v65.rs`,
SHA-256 `75bd5198486e7e3c6072deebbdebd256aa3152a7b43b60054349f8e181c200f0`;
preregistration SHA-256
`f44eb21446ffdc6897c76818f4d4b22ecf266cf4f2707a4a8d995b0479acd589`) by exactly
**one changed factor**:

> **The feature standardization applied at transfer time.**

TDI-6.5C fitted GK and GKT models on family F0-base and evaluated them on
F1-sparse's holdout, applying the **source domain's** feature means and scales
at prediction time. TDI-6.6 keeps that pipeline verbatim and adds two further
*arms* that differ only in which domain's statistics are used to standardize
before the frozen coefficients are applied.

Everything else is inherited unchanged: the four generator families, the
population contract, the feature layouts (CK/SK/GK/GKT), the horizons, the
linear ridge (λ = 1), the deterministic bootstrap, the four-way classifier and
its symmetric 2 % relative-MSE margin, and the non-exact determinism discipline
of TDI-6.1 Section 12.

### 1.1 Why this experiment, and why now

Two independent experiments have now measured the same failure:

- **TDI-5.8B** — a model fitted at width 3 and applied at width 5 is badly
  miscalibrated at every setting (`R²` far below zero for both layouts), while
  rank ordering survives through the overlaps (Spearman 0.695) and not through
  the exact descriptors (−0.040);
- **TDI-6.5C** — a model fitted on F0-base and applied to F1-sparse is
  miscalibrated in the same way (`R²` −2.174 and −1.265 at U₃), while ordering
  again survives through the overlaps (Spearman 0.694 versus 0.404).

Both result documents name the **same untested explanation**: the descriptors
drift between domains, so a model fitted in one is extrapolating outside its
feature distribution in the other. TDI-5.8 Section 6 states the candidate fix
explicitly — *"re-standardising with width-5 statistics before transfer"* — and
records that the experiment does not test it. TDI-6.6 tests it.

The result is informative in both directions. If re-standardization repairs
calibration, the failure was in the feature scale and the fitted relationship
transports after all. If it does not, the drift explanation that both prior
documents advance as "consistent but not demonstrated" is eliminated, and the
failure is located elsewhere.

### 1.2 Why the cross-generator axis, and not cross-width

The candidate fix is named in TDI-5.8 (widths) but the failure is more extreme
in TDI-6.5 (generators): descriptor drift across families spans 0.895 in δ and
2.632 in `s₂`, against 0.075 and 0.136 across widths. If re-standardization
helps anywhere, it should help most where the extrapolation is largest.
Deriving from TDI-6.5 also keeps the changed factor singular — the transfer
axis is already fixed by the ancestor.

The cross-width axis is therefore **out of scope** and TDI-5.8B's finding is
untouched by any TDI-6.6 result (Section 23).

## 2. Research questions

1. Does replacing the source domain's feature standardization with the **target
   domain's own**, without using any target label, reduce cross-generator
   transfer error? (TDI-6.6A)
2. Does it repair **calibration** — that is, does the transferred model become
   better than predicting the target's mean? (TDI-6.6B)
3. If not, is the residual failure in the **feature** scale or in the **target**
   scale? (TDI-6.6C, the oracle arm)
4. Does any of this hold across generator pairs other than F0→F1? (TDI-6.6D)
5. Does re-standardization change the **ordering** the transferred model
   produces, or only its scale?

## 3. Relationship to the frozen ancestors

TDI-6.6 modifies no frozen file. It derives a fresh evaluator, adds fresh
disjoint seed blocks and bootstrap seeds, and verifies the full frozen ancestor
chain (TDI-5.1 → 5.9, TDI-6.1 → 6.5: evaluator, preregistration and, where
present, scientific-manifest hashes) before any generation. The scientific
ancestor is TDI-6.5; TDI-5.8 is the source of the hypothesis but is not a code
ancestor.

## 4. Design notes and confirmatory integrity

### 4.1 The three arms

A fitted model is a triple: feature means `μ`, feature scales `σ`, and
coefficients `β` (plus the `TargetScaler` `(m, s)` that maps `U_h` to and from
standardized space). Prediction on a record is

    ŷ_std = β₀ + Σⱼ βⱼ · (xⱼ − μⱼ) / σⱼ ,      ŷ = m + s · ŷ_std

TDI-6.6 evaluates the **same `β`** — never refitted — under three substitutions:

| Arm | Feature `μ, σ` | Target `(m, s)` | Uses target labels? |
|---|---|---|:--:|
| **A0 — source** (TDI-6.5C, unchanged) | source | source | no |
| **A1 — feature re-standardized** | **target** | source | **no** |
| **A2 — oracle** | **target** | **target** | **yes** |

**A1 is the experiment.** It is ordinary unsupervised domain adaptation:
computing a feature mean and standard deviation in the target domain requires
target *inputs* only. A1 is a deployable procedure.

**A2 is not a predictor and is never reported as one.** Fitting the target
scaler requires the target domain's `U_h` values — the very quantity being
predicted. A2 exists solely as an **upper bound**: it answers "if the target
scale were known, would the transferred relationship be good?", which localizes
the residual failure. Every table and every verdict line naming A2 carries the
`oracle` label, and Section 23 forbids reading it as achievable performance.

### 4.2 Where the target statistics come from (no transduction)

A1 and A2 compute the target domain's statistics on that family's **combined
training population** — never on the holdout being scored. The target family's
training and holdout populations are disjoint by construction (Section 8), so
no record contributes both to the standardization and to the evaluation. This
removes any transductive objection: the procedure requires a sample of target
inputs, not the specific inputs being predicted.

A transductive variant (standardizing on the scored holdout itself) is a
different design and is **not** part of TDI-6.6.

### 4.3 The label-use boundary is enforced structurally, not by convention

The distinction between A1 and A2 is the entire scientific content of this
experiment, and a naive implementation could silently leak target labels into
A1 and report the result as if it were label-free. Three mechanisms guard it:

1. the arm is a Rust enum, not a flag pair, so no combination outside the table
   in Section 4.1 is representable;
2. A1's re-standardization path derives its statistics only from
   `model_features(record, layout)` and never calls the target-value or
   target-scaler routines;
3. **a bounded test proves label-freeness behaviourally**: A1's predictions
   must be *bit-identical* when every target value in the target-domain records
   is arbitrarily perturbed. A test that fails this is a defect in the
   evaluator, not a result.

Point 3 is required. A code-reading argument is not sufficient evidence for the
claim A1 makes.

### 4.4 Re-standardization adds no new non-exactness

Feature means and scales are already computed, in the same IEEE-754 binary64,
single-threaded, fixed-operation-order regime, by the frozen ridge fit
(`fit_ridge`). A1 and A2 reuse that arithmetic on a different record set. The
two non-exact descriptors `g = 1 − |λ₂|` and `τ_ε / T_max` remain the only
non-exact quantities in the design, and reproduction stays tolerance-based
exactly as in TDI-6.1 and TDI-6.5. The degenerate-scale floor is inherited
unchanged: a feature scale that is non-finite or `≤ 1.0e-12` is replaced by
`1.0`, in every arm identically.

### 4.5 Why the coefficients are never refitted

Refitting on the target domain would answer a different and much easier
question ("can a model be fitted on family F1?" — TDI-6.5A already says yes).
TDI-6.6 asks whether a relationship *learned elsewhere* transports. `β` is
therefore frozen at its source-domain value in all three arms, and only the
standardization changes.

## 5. Generator families, kernel and descriptors (inherited)

The four families of TDI-6.5 Section 9 and the one-step `Noop` kernel of
Section 6, unchanged: **F0-base** (uniform over non-empty successor subsets),
**F1-sparse** (out-degree `d ∈ {1,2}`), **F2-dense** (all states minus
`e ∈ {0,1}` excluded bits), **F3-local** (Hamming ≤ 1 neighbourhood, self
forced if empty). Descriptors: exact δ, δ̄, `s₂`, `s₃`; non-exact `g`, `τ_ε`
with `ε = 1/4`, `T_max = 4096`, eigensolver tolerance `1.0e-12`.

## 6. Feature layouts (inherited)

| Layout | Features |
|---|---:|
| CK | baseline + δ + δ̄ | 15 |
| SK | CK + `s₂` + `s₃` | 17 |
| GK | SK + `g` + `τ_ε` | 19 |
| GKT | GK + `O₁` + `O₂` | 21 |

The transfer comparisons use **GK** as baseline and **GKT** as challenger,
matching TDI-6.5C.

## 7. Model (inherited)

Linear ridge, `lambda = 1.0`, on standardized features and standardized `U_h`,
solved by the frozen deterministic linear solve. Nine model layouts per
horizon, six target horizons `H = {3,4,5,6,7,8}`, observation horizon 2, focal
horizons **U₃** and **U₆**.

## 8. Populations

Inherited from TDI-6.5 Section 10, under **fresh** seeds. Per family, three
independent seed blocks; per block:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |

40,000 accepted per block, 120,000 per family, **480,000 total**. Width-3 and
width-4 populations of a block are combined for fitting and for evaluation, as
in every prior experiment. Generation budgets, attempt multipliers,
no-progress thresholds and the preregistered rejection categories are inherited
verbatim from TDI-5.2 Section 7.

## 9. Independent seed blocks (fresh)

Numeric seeds and bootstrap seeds are disjoint from every prior experiment. The
bootstrap seeds follow the series ASCII scheme `0x5444_4936_36…` ("TDI" + "6" +
"6").

| Family | Block | train w3 | holdout w3 | train w4 | holdout w4 | bootstrap |
|---|---|---:|---:|---:|---:|---|
| F0-base | g0 | 6200000000 | 6210000000 | 6220000000 | 6230000000 | `0x5444493636000001` |
| F0-base | g1 | 6300000000 | 6310000000 | 6320000000 | 6330000000 | `0x5444493636000002` |
| F0-base | g2 | 6400000000 | 6410000000 | 6420000000 | 6430000000 | `0x5444493636000003` |
| F1-sparse | g0 | 6500000000 | 6510000000 | 6520000000 | 6530000000 | `0x5444493636000004` |
| F1-sparse | g1 | 6600000000 | 6610000000 | 6620000000 | 6630000000 | `0x5444493636000005` |
| F1-sparse | g2 | 6700000000 | 6710000000 | 6720000000 | 6730000000 | `0x5444493636000006` |
| F2-dense | g0 | 6800000000 | 6810000000 | 6820000000 | 6830000000 | `0x5444493636000007` |
| F2-dense | g1 | 6900000000 | 6910000000 | 6920000000 | 6930000000 | `0x5444493636000008` |
| F2-dense | g2 | 7000000000 | 7010000000 | 7020000000 | 7030000000 | `0x5444493636000009` |
| F3-local | g0 | 7100000000 | 7110000000 | 7120000000 | 7130000000 | `0x544449363600000A` |
| F3-local | g1 | 7200000000 | 7210000000 | 7220000000 | 7230000000 | `0x544449363600000B` |
| F3-local | g2 | 7300000000 | 7310000000 | 7320000000 | 7330000000 | `0x544449363600000C` |

Stratified aggregate bootstrap seeds, one per family:
`0x5444493636004700` (F0-base), `…4701` (F1-sparse), `…4702` (F2-dense),
`…4703` (F3-local). Seed disjointness is checked in code and in a bounded test.

**Because the seeds are fresh, TDI-6.6's A0 arm will not reproduce TDI-6.5C's
numbers exactly.** A0 is included so that the A1-vs-A0 comparison is *paired*
on identical populations; any comparison to TDI-6.5C's published values is
across independent samples and must be reported as such.

## 10. Deterministic bootstrap (inherited)

4,000 replicates, stratified by seed block, frozen seeds as in Section 9. The
same paired-resampling discipline as TDI-6.5.

## 11. Metrics and standardized-U primacy (inherited)

Primary space is **standardized U**; reconstructed-O metrics are reported
alongside. For each arm, layout and horizon the evaluator reports MSE, MAE,
`R²`, Spearman, bias, observed and predicted means, calibration intercept and
slope, and the bound fractions.

## 12. Criterion TDI-6.6A — does label-free re-standardization reduce transfer error? (primary)

On the **F0-base → F1-sparse** transfer, at focal horizons **U₃** and **U₆**,
compare arm **A1 against arm A0** under the **GKT** layout, using the frozen
four-way classifier with its symmetric 2 % relative-MSE margin and its full
three-condition rule (blocks confirming, aggregate margin, aggregate bootstrap
bound).

- *Beneficial* — label-free re-standardization reduces transfer error;
- *Equivalent* — it changes nothing beyond the margin;
- *Harmful* — it makes transfer worse.

The same comparison is reported for the **GK** layout, so that any effect can
be attributed to the re-standardization rather than to the overlaps.
Preregistered classification, forced to no result. **No outcome is a success or
a failure**: *Harmful* would be as informative as *Beneficial*, since it would
rule the drift explanation out.

## 13. Criterion TDI-6.6B — is calibration repaired? (primary)

For each arm, layout and focal horizon, report the aggregate standardized-U
`R²` and the preregistered boolean

    calibration_repaired  ≡  (aggregate standardized-U R² > 0)

together with its 95 % bootstrap interval. TDI-6.6B is the conjunction

- **repaired** iff `calibration_repaired` holds under **A1** for the **GKT**
  layout at **both** focal horizons;
- otherwise a **located non-repair**, naming each (layout, horizon) whose `R²`
  remains ≤ 0.

This is the criterion that decides whether the transfer failure of TDI-5.8B and
TDI-6.5C is fixable by feature alignment alone. Preregistered classification,
forced to no result.

The threshold `R² > 0` is chosen because it is the exact statement that failed
in both prior experiments ("worse than predicting the target's mean") and
because it is scale-free and needs no arbitrary margin. It is reported with its
bootstrap interval so that a marginal case is visible as marginal.

## 14. Criterion TDI-6.6C — the oracle arm (descriptive)

### 14.1 Why A2 may not be compared to A0 or A1 by relative MSE

The evaluator standardizes the **ground truth** with the same target scaler it
uses for predictions. A0 and A1 both carry the *source* scaler, so their
standardized-U errors live in one space and their relative-MSE comparison
(Section 12) is well posed. A2 carries the *target* scaler, so its errors live
in a **different** space: with `y_std = (y − m) / s`, changing `s` rescales every
squared error by `s²`. A relative-MSE reduction between A2 and either other arm
would therefore measure the ratio of two scaler variances, not any property of
the models. **No such comparison is computed or reported**, and the evaluator has
no code path that can construct one.

Scale-free quantities are unaffected and are the only ones reported across the
A2 boundary:

- **`R²`**, because `R² = 1 − MSE / Var(y_std)` and numerator and denominator
  both scale by `s²`. "Beats predicting the target's mean" is invariant to the
  affine rescaling, which is precisely why Section 13's criterion is stated as a
  sign test on `R²` rather than as an MSE margin;
- **Spearman**, invariant to any strictly monotone transformation of the target;
- the **calibration slope**, invariant by construction.

### 14.2 What is reported

For arm **A2**, always labelled `oracle`: the standardized-U `R²` with its
bootstrap interval, the `calibration_repaired` flag of Section 13, Spearman, and
the calibration slope — per layout and focal horizon. Raw MSE and MAE are
printed for completeness but explicitly marked as **not comparable across
arms**. A2 bounds what target-scale knowledge could buy:

- if A2 repairs calibration and A1 does not, the residual failure is in the
  **target** scale, and no label-free procedure can fix it;
- if neither repairs it, the failure is in the fitted relationship itself, not
  in either standardization;
- if both repair it, A2 quantifies how much of the repair is unavailable in
  practice.

Descriptive; no success/failure claim. A2 is never described as a prediction
method.

## 15. Criterion TDI-6.6D — all ordered generator pairs (descriptive)

Repeat Sections 12–13's quantities for **all 12 ordered pairs** of distinct
families (each family's aggregate fit evaluated on each other family's
holdouts), at both focal horizons, GK and GKT. Report per pair: the A1-vs-A0
relative-MSE classification (both arms share the source scaler, so this is well
posed), `calibration_repaired` per arm, and Spearman per arm. Arm A2
contributes only the scale-free quantities of Section 14.1, under the `oracle`
label.

Report also the **direction consistency**: whether the A1-vs-A0 classification
is identical across all 12 pairs, and if not, which pairs differ. Descriptive —
this is the dense-grid analogue of the series' focal/grid split, and it exists
so that a result on F0→F1 alone is not over-read.

## 16. Ordering versus scale (reported under every criterion)

Re-standardization changes each feature's contribution by a *different* affine
map, so the transferred prediction is **not** an affine function of its A0
counterpart and the induced ordering can change. Spearman is therefore reported
for every arm, layout, horizon and pair. A finding of "scale repaired, ordering
unchanged" and one of "ordering degraded to buy scale" are different results
and must be distinguishable in the output.

## 17. Descriptor drift (context only, no criterion)

Per-family holdout means of δ, δ̄, `s₂`, `s₃`, `g`, `τ_ε` and their
across-family ranges, as in TDI-6.5D, so that each transfer pair's difficulty is
visible next to its result. By construction the re-standardized features have
mean 0 and unit scale in the target domain, so this table describes the *raw*
descriptors only.

## 18. Non-exact determinism discipline (inherited)

IEEE-754 binary64, single-threaded, fixed operation order, no FMA or parallel
reduction. Declared tolerances: eigensolver convergence `1.0e-12`, cross-method
agreement `1.0e-9`, degenerate-scale floor `1.0e-12`. The three-method spectral
cross-validation table of TDI-6.1/6.5 is reproduced, sampling candidates in each
of the four families.

**Reading the cross-validation table.** The rigorous correctness witness for the
frozen eigenvalue path is the **trace residual** `Σλᵏ = trace(Pᵏ)`. The
method-1 ↔ method-2 disagreement is a **diagnostic** and is expected to be large
when `λ₂` is complex; TDI-6.5 established that where the two disagree it is
method 2 that is wrong, since it returns `|λ₂| > 1` values that are impossible
for a row-stochastic matrix.

## 19. Required raw output

The evaluator prints, in a fixed order: provenance and the full frozen ancestor
chain; the frozen constants; the family rules; the seed blocks; the spectral
cross-validation table; per-family population counts, rejection reasons, final
exclusive seeds and budgets; per-arm normalization summaries; for every
(pair, arm, layout, focal horizon) the full metric block in both spaces with
stratified bootstrap intervals; the per-criterion block-level and aggregate
conditions; and the final verdict lines for TDI-6.6A, 6.6B, 6.6C and 6.6D.

Every A2 line is prefixed `oracle`. Every `calibration_repaired` line prints the
`R²` and its interval, never the boolean alone.

## 20. Operational activation and full-run entrypoint contract

The evaluator exposes exactly `--termination-smoke`, `--preflight` and
`--full`; a bare invocation refuses. `--full` requires the exact confirmation
variable

    TDI66_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI66_FREEZE_RULE

and refuses on any other value, including an empty one. The reproduction script
additionally refuses a dirty repository, verifies the full frozen hash chain
before any generation, verifies the final criterion lines afterwards, and
writes read-only artifacts under
`results/tdi6.6-restandardized-transfer/`.

## 21. Determinism

Given the frozen seeds and the declared floating-point regime, a faithful
re-run on the reference toolchain and architecture reproduces the result log
byte-for-byte. Across architectures the last binary64 digits may differ; the
±2 % classifier margin and the `R² > 0` threshold are robust to that by many
orders of magnitude, so every reported classification and flag reproduces
exactly. Generation is deterministic and independent of arm: all three arms
score the *same* records.

## 22. Reproduction requirements

    TDI66_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI66_FREEZE_RULE \
      bash scripts/reproduce-tdi6.6.sh

## 23. Interpretation boundaries

A TDI-6.6 result characterizes cross-**generator** transfer of a **linear ridge**
model between the four hand-specified families of TDI-6.5, at widths 3–4, under
the one-step `Noop` kernel. It does **not** establish:

- anything about cross-**width** transfer (TDI-5.8B stands as reported, and no
  TDI-6.6 outcome revises it);
- that arm **A2** is achievable. A2 uses the target domain's labels and is an
  upper bound only; reporting it as attainable performance is forbidden;
- that a *refitted* model would or would not transport — the coefficients are
  frozen by design (Section 4.5);
- transfer under any model family other than linear ridge (the nonlinear
  control is TDI-6.2, on F0-base only), or under any standardization scheme
  other than mean/standard-deviation alignment — whitening, CORAL, and
  distribution matching are untested;
- generality beyond four hand-specified synthetic families, or anything outside
  small synthetic finite-state families.

If TDI-6.6A returns *Harmful* or *Equivalent* and TDI-6.6B returns a
non-repair, the correct conclusion is that feature-scale drift does **not**
explain the transfer failure of TDI-5.8B and TDI-6.5C — which retires an
explanation both documents explicitly flagged as unproven, and is a result, not
a null.

The TDI-6.6A / 6.6B / 6.6C / 6.6D summaries may not be rewritten after
observing the result.

## 24. Freeze rule

This preregistration is frozen at its SHA-256. The evaluator, its reproduction
script, its CI workflow and its bounded tests are frozen before the run. The
confirmatory run happens once, as a deliberate human action under the token of
Section 20; no commit, test or CI run may supply that token, and the authoring
agent must never invoke `--full` with it.
