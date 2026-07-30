# TDI-6.7 — Observable-Offset Cross-Generator Transfer

## Preregistration

This document is frozen before the evaluator is written and before any
confirmatory data is generated. Every constant, layout, criterion and
classification rule below is fixed in advance. Nothing here may be rewritten
after observing a result.

**Freeze rule.** The real confirmatory run is a single, deliberate, human-only
action behind the exact confirmation token of Section 18. No commit, test, or
CI run may supply the token. The authoring agent must never invoke `--full`
with the real token.

## 1. Experimental status, provenance, and the single changed factor

TDI-6.7 derives from **TDI-6.6** (`tdi-independent-overlap-ablation-v66.rs`,
SHA-256 `b563a712b80bd983bfc3d6c7d4ffecc7b7fec35e6f3be8fea539ecb984346e7b`;
preregistration SHA-256
`5216be5d8103505d63a0c9fd78d57f668d7005afd525a8481e259736df7276a9`) by exactly
**one changed factor**:

> **What the label-free transfer correction is.** TDI-6.6 aligned the feature
> *scale*; TDI-6.7 shifts the predicted *level* by an offset estimated from
> observable early-horizon deficits.

Everything else is inherited verbatim: the four generator families, the
population contract, the CK/SK/GK/GKT layouts, the linear ridge (λ = 1), the
horizons and focal horizons, the never-refitted coefficients, the deterministic
bootstrap, the four-way classifier with its symmetric 2 % relative-MSE margin,
the `R² > 0` sign test, the oracle arm, the twelve ordered pairs, and the
non-exact determinism discipline.

### 1.1 Why this experiment

TDI-6.6 established three things that together specify this design:

1. **Feature-scale alignment is not the fix** — criterion 6.6A returned
   *Harmful* in all four confirmatory cells, and 6.6B returned *not repaired*;
2. **the mechanism** (6.6 §5) — re-standardizing features on the target
   domain's own mean centres them at zero *by construction*, which annihilates
   the domain displacement that was carrying the level information. On
   F0→F1/GK/U₃ the observed mean is −3.354 in source-standardized units while
   arm A1 predicts +0.002;
3. **the failure is located in the target scale** (6.6C) — the oracle arm, whose
   only difference is the target scaler, repairs calibration everywhere.

The obvious reading of (3) is pessimistic: the missing quantity is the target
domain's deficit *level*, which is what one is trying to predict. TDI-6.7 tests
whether that pessimism is complete, because **a proxy for the level is
observable without any label.**

### 1.2 The observable proxy, and why it is genuinely label-free

The target geometry of the whole series is

    U_h = −log₂(1 − O_h)

and the **observation horizon is 2**. The early overlaps `O₁, O₂` are *features*
— they are on every record and are used as predictors throughout TDI-5.2 → 6.6.
Therefore `U₁ = −log₂(1 − O₁)` and `U₂ = −log₂(1 − O₂)` are computable from the
feature vector alone, for any record, in any domain, **without reading a single
target value**.

`U₂` is the last fully-observed deficit. Its domain-level mean is an observable
statistic of an unlabelled sample. If the between-domain shift at the observed
horizon predicts the shift at the predicted horizons, a label-free level
correction exists. If it does not, the pessimistic reading of 6.6C is complete
and the transfer failure is irreducible within this family of corrections.

The result is informative either way, and neither outcome is a success or a
failure (Section 12).

### 1.3 What TDI-6.7 must not be

TDI-6.6's data has been observed. Its Section 8 reported, descriptively, that
feature alignment substantially improves *rank* transfer (Spearman −0.600 →
0.879 on one pair). **Designing a criterion around that observation would be
post-hoc criterion selection on seen data**, which is exactly what this
programme's preregistration discipline exists to prevent. TDI-6.7 therefore
does **not** adopt a rank-based criterion, does **not** re-analyse TDI-6.6's
frozen log, and generates **fresh populations under fresh seeds** (Section 7).

The only quantities TDI-6.7 inherits from TDI-6.6 are its *design* decisions —
already frozen before 6.6's data existed — and the mechanism of §5, which is an
algebraic property of the standardization, not an empirical fit.

## 2. Research questions

1. Does shifting the transferred prediction by an offset estimated from the
   **observable** early-horizon deficit reduce cross-generator transfer error?
   (TDI-6.7A)
2. Does it repair **calibration** — does the transferred model beat predicting
   the target's mean? (TDI-6.7B)
3. How much of the oracle's advantage does it recover? (TDI-6.7C)
4. Does it hold across generator pairs other than the confirmatory one?
   (TDI-6.7D)

## 3. The three arms

Prediction in standardized-U space is `ŷ_std = β₀ + Σⱼ βⱼ·(xⱼ − μⱼ)/σⱼ`, then
`ŷ = m + s·ŷ_std`. Coefficients `β` are **never refitted** in any arm.

| Arm | feature `μ, σ` | target `(m, s)` | prediction shift | reads a target label? |
|---|---|---|---|:--:|
| **B0** — source | source | source | none | no |
| **B1** — observable offset | source | source | **`+ Δ̂_std`** | **no** |
| **B2** — oracle | source | **target** | none | **yes** |

**B0 is byte-for-byte TDI-6.6's A0** — the TDI-6.5C behaviour, carried forward
unchanged so that the B1-vs-B0 comparison is paired on identical populations.

**B1 is the experiment.** It adds a single scalar to every standardized
prediction of a given (pair, horizon). It changes no feature statistic, so the
mechanism that destroyed A1 (6.6 §5) cannot arise: the feature displacement that
carries level information is left intact and the correction is *added* to it.

**B2 is the same oracle as TDI-6.6's A2 in role but not in construction**: it
keeps the **source** feature statistics (unlike A2, which also re-standardized
features) and replaces only the target scaler. This isolates the target-scale
contribution alone, which is what B1 is trying to approximate. It is **not a
prediction method** and every output line naming it is labelled `oracle`.

### 3.1 The offset estimator, fully specified

For an ordered pair (source `S`, target `T`) and horizon index `h`:

1. compute `u₂(r) = −log₂(1 − O₂(r))` for every record `r`, from the record's
   feature vector only;
2. `μ₂ˢ` = mean of `u₂` over `S`'s combined **training** populations;
   `μ₂ᵀ` = mean of `u₂` over `T`'s combined **training** populations;
3. the raw observable shift is `Δ = μ₂ᵀ − μ₂ˢ`;
4. the applied shift is `Δ̂_std = Δ / sˢ_h`, where `sˢ_h` is the **source**
   target-scaler scale at horizon `h` — the same scaler B0 and B1 both carry,
   so the shift is expressed in the units B0's predictions already live in.

`Δ̂_std` is a **single scalar per (pair, horizon)**, added to every prediction.
No per-record quantity, no fitting, no target label. It is computed on training
populations only (never the scored holdout), so the design is not transductive.

**Degenerate guard.** `O₂ = 1` would make `u₂` infinite. The frozen population
contract already excludes fully-recovered observations, so this cannot occur;
the evaluator nevertheless refuses with a declared error rather than
propagating a non-finite value.

### 3.2 Why `U₂` and not `U₁`, and why an additive shift

The observation horizon is 2, so `U₂` is the **last** fully observed deficit and
the closest observable neighbour of the first predicted horizon `U₃`. `U₁` is
reported alongside as a descriptive companion (Section 15) but is not used by
any criterion.

The correction is **additive in standardized-U space** because that is exactly
where the failure was measured: 6.6 §5 showed a *displacement* between predicted
and observed means, not a scale error. A multiplicative or affine two-parameter
correction would require estimating a target *spread* as well, which needs a
second observable statistic and a second justification; it is deferred
(Section 19).

### 3.3 The label-use boundary is enforced structurally

As in TDI-6.6 §4.3, the distinction between B1 and B2 is the scientific content
of the design, and a naive implementation could leak target labels into B1:

1. the arm is a Rust enum, so no combination outside Section 3's table is
   representable;
2. B1's offset path derives `u₂` only from `Record::early_overlap` and never
   reads `targets_u` or calls the target-scaler fit;
3. **a bounded test proves label-freeness behaviourally**: B1's predictions must
   be *bit-identical* when every target value in both domains' records is
   arbitrarily perturbed. A mirror test asserts B2's scaler *does* move under the
   same perturbation, so the first test cannot be satisfied vacuously.

Point 3 is required. A code-reading argument is not sufficient evidence.

## 4. Generator families, kernel, descriptors, layouts, model

Inherited unchanged from TDI-6.6 §§5–7: families F0-base / F1-sparse / F2-dense
/ F3-local; the one-step `Noop` kernel; exact δ, δ̄, `s₂`, `s₃`; non-exact `g`,
`τ_ε` with `ε = 1/4`, `T_max = 4096`, eigensolver tolerance `1.0e-12`; layouts
CK 15 / SK 17 / GK 19 / GKT 21; linear ridge `λ = 1`; observation horizon 2;
horizons `H = {3,4,5,6,7,8}`; focal horizons **U₃** and **U₆**. Transfer
comparisons use **GK** and **GKT**.

## 5. Populations

Inherited from TDI-6.6 §8 under **fresh** seeds: per family three independent
seed blocks; per block 15,000 training and 5,000 holdout at width 3 and the same
at width 4. 40,000 accepted per block, 120,000 per family, **480,000 total**.
Generation budgets, attempt multipliers, no-progress thresholds and the
preregistered rejection categories are inherited verbatim from TDI-5.2 §7.

## 6. Metrics and standardized-U primacy

Inherited unchanged. Primary space is standardized U; reconstructed-O metrics
are reported alongside. Per arm, layout and horizon the evaluator reports MSE,
MAE, `R²`, Spearman, bias, observed and predicted means, calibration intercept
and slope, and the bound fractions.

**Cross-arm comparability, inherited from TDI-6.6 §14.1.** B0 and B1 carry the
**same** target scaler, so their ground truth is standardized identically and
their relative-MSE comparison is well posed. B2 carries a different scaler, so
across the B2 boundary only **scale-free** quantities may be compared — `R²`
(numerator and denominator both scale by `s²`), Spearman, and the calibration
slope. Raw MSE and MAE are printed for completeness and explicitly marked as not
comparable across arms. The evaluator must **refuse** to construct a
relative-MSE comparison between arms whose target scalers differ.

**Pooled-Spearman note, inherited from TDI-6.6 §8.1.** Aggregate metrics
concatenate three seed blocks each carrying its own target scaler, so the map
between two arms' pooled ground truth is piecewise affine and does not preserve
global rank order. B0 and B1 share scalers and are therefore exempt; only the
B2 boundary is affected, and no criterion depends on Spearman.

## 7. Independent seed blocks (fresh)

Numeric seeds continue the series arithmetic with a fresh origin, and bootstrap
seeds follow the ASCII scheme `0x5444_4936_37…` ("TDI" + "6" + "7").

- population base seed: `base(f, b) = 7.4e9 + f·300e6 + b·100e6`, with the four
  populations at `base + {0, 10, 20, 30}·1e6`. The 7.4e9 origin clears TDI-6.6's
  last reservation (7.33e9 + 5,030), so every TDI-6.7 seed is disjoint from every
  prior experiment's;
- per-block bootstrap seed: `0x5444_4936_3700_0000 + (3·f + b) + 1`;
- per-family stratified aggregate seed: `0x5444_4936_3700_4700 + f`;
- per-ordered-pair bootstrap stream:
  `0x5444_4936_3700_4700 + 0x10·(1 + f_source) + f_target`, disjoint from the
  per-family seeds because pair offsets start at `0x10`.

Seed disjointness is checked in code and in a bounded test.

Because the seeds are fresh, TDI-6.7's B0 arm will not reproduce TDI-6.6's A0
numbers exactly. B0 exists so the B1-vs-B0 comparison is *paired* on identical
populations; any comparison to TDI-6.6's published values is across independent
samples and must be reported as such.

## 8. Deterministic bootstrap

4,000 replicates, stratified by seed block, frozen seeds as in Section 7,
inherited paired-resampling discipline. Single-arm `R²` intervals are
recomputed inside each replicate with the total sum of squares taken about that
replicate's own resampled mean, as in TDI-6.6.

## 9. Non-exact determinism discipline

Inherited: IEEE-754 binary64, single-threaded, fixed operation order, no FMA or
parallel reduction; eigensolver tolerance `1.0e-12`, cross-method agreement
`1.0e-9`, degenerate-scale floor `1.0e-12`. `g` and `τ_ε` remain the only
non-exact quantities; the offset estimator adds none — it is a mean of `log₂`
values in the same regime. The three-method spectral cross-validation table is
reproduced, sampling candidates in each family.

**Reading that table.** The rigorous witness is the **trace residual**; the
method-1 ↔ method-2 disagreement is a diagnostic expected to be large when `λ₂`
is complex, and TDI-6.5 established that where they disagree it is method 2 that
is wrong, since it returns `|λ₂| > 1` values impossible for a row-stochastic
matrix.

## 10. Criterion TDI-6.7A — does the observable offset reduce transfer error? (primary)

On the **F0-base → F1-sparse** transfer, at focal horizons **U₃** and **U₆**,
compare arm **B1 against arm B0** under **GKT**, using the frozen four-way
classifier with its symmetric 2 % relative-MSE margin and full three-condition
rule. The same comparison is reported for **GK**, so any effect can be
attributed to the correction rather than to the overlaps.

*Beneficial* = the observable offset reduces transfer error; *Equivalent* = it
changes nothing beyond the margin; *Harmful* = it makes transfer worse.

Preregistered classification, forced to no result. **No outcome is a success or
a failure**: *Harmful* or *Equivalent* would establish that the observable early
shift does not predict the late shift, completing the pessimistic reading of
TDI-6.6C.

## 11. Criterion TDI-6.7B — is calibration repaired? (primary)

For each arm, layout and focal horizon report the aggregate standardized-U `R²`,
its 95 % bootstrap interval, and

    calibration_repaired  ≡  (aggregate standardized-U R² > 0)

TDI-6.7B is the conjunction: **repaired** iff `calibration_repaired` holds under
**B1** for **GKT** at **both** focal horizons; otherwise a **located
non-repair** naming each (layout, horizon) still at or below zero. Preregistered
classification, forced to no result.

## 12. Criterion TDI-6.7C — recovered fraction of the oracle (descriptive)

Report, per layout and focal horizon, the scale-free quantities of Section 6 for
**B2** (`oracle` label), and the preregistered summary

    recovered_fraction = (R²_B1 − R²_B0) / (R²_B2 − R²_B0)

when `R²_B2 > R²_B0`, and `not-applicable` otherwise. This places B1 on the
segment between doing nothing and knowing the answer. It is descriptive: no
threshold is preregistered and no value is a success.

## 13. Criterion TDI-6.7D — all twelve ordered pairs (descriptive)

Repeat Sections 10–12 for **all 12 ordered pairs** of distinct families at both
focal horizons, GK and GKT. Report per pair: the B1-vs-B0 classification,
`calibration_repaired` per arm, `recovered_fraction`, and the raw observable
shift `Δ`. Report **direction consistency** — whether the B1-vs-B0
classification is identical across all pairs at GKT and both focal horizons —
and name every divergent pair.

**Reading rule, preregistered.** A *Beneficial* classification accompanied by a
still-negative `R²` means the arm is less bad, not good. Every *Beneficial* cell
whose `R²` remains ≤ 0 must be reported with that qualification attached, in the
same line.

## 14. Relationship between `Δ` and the true level shift (descriptive)

For each ordered pair and focal horizon, report the observable shift `Δ`
alongside the **true** shift `μ_hᵀ − μ_hˢ` computed from the holdout labels, and
their ratio. This is a *diagnostic that uses labels* and is therefore reported
under the `oracle` label; it never feeds B1, whose construction sees only `Δ`.
It answers directly why B1 succeeded or failed.

## 15. Companion statistics (context only, no criterion)

The same table for `U₁` in place of `U₂`, and the per-family means of `u₁`, `u₂`
with their across-family ranges, so that each pair's difficulty is visible
beside its result. Per-family descriptor drift (δ, δ̄, `s₂`, `s₃`, `g`, `τ_ε`) is
reported as in TDI-6.6 §17.

## 16. Required raw output

Provenance and the full frozen ancestor chain; frozen constants; family rules;
seed blocks; the spectral cross-validation table; per-family population counts,
rejection reasons, final exclusive seeds and budgets; per-arm normalization
summaries; for every (pair, arm, layout, focal horizon) the full metric block in
both spaces with stratified bootstrap intervals; the per-criterion block-level
and aggregate conditions; and the final verdict lines for 6.7A, 6.7B, 6.7C and
6.7D. Every B2 line is prefixed `oracle`. Every `calibration_repaired` line
prints the `R²` and its interval, never the boolean alone. Every offset line
prints `Δ`, `sˢ_h` and `Δ̂_std` separately.

## 17. Determinism

Given the frozen seeds and the declared floating-point regime, a faithful re-run
on the reference toolchain and architecture reproduces the log byte-for-byte.
Across architectures the last binary64 digits may differ; the ±2 % margin and
the `R² > 0` threshold are robust to that by many orders of magnitude.
Generation is arm-independent: all three arms score the same records.

## 18. Operational activation and full-run entrypoint contract

The evaluator exposes exactly `--termination-smoke`, `--preflight` and
`--full`; a bare invocation refuses. `--full` requires

    TDI67_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI67_FREEZE_RULE

and refuses on any other value, including an empty one. The reproduction script
additionally refuses a dirty repository, verifies the full frozen hash chain
before any generation, verifies the final criterion lines afterwards, and writes
read-only artifacts under `results/tdi6.7-observable-offset-transfer/`.

    TDI67_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI67_FREEZE_RULE \
      bash scripts/reproduce-tdi6.7.sh

## 19. Interpretation boundaries

A TDI-6.7 result characterizes cross-**generator** transfer of a **linear ridge**
model between the four hand-specified families of TDI-6.5, at widths 3–4, under
the one-step `Noop` kernel, corrected by a **single additive scalar** estimated
from the observed-horizon deficit. It does **not** establish:

- that **B2** is attainable — it reads target labels and is a bound only;
- anything about **cross-width** transfer; TDI-5.8B stands as reported;
- that a **two-parameter** (offset and scale) or per-record correction would
  behave the same way — only the additive one-parameter form is tested
  (Section 3.2);
- that a **refitted** model would fail — coefficients are frozen by design;
- generality beyond four synthetic families, or anything outside small synthetic
  finite-state families.

If TDI-6.7A returns *Harmful* or *Equivalent* and TDI-6.7B returns a non-repair,
the correct conclusion is that the observed-horizon shift does not predict the
predicted-horizon shift, so the pessimistic reading of TDI-6.6C is complete
within this family of corrections — which is a result, not a null.

The TDI-6.7A / 6.7B / 6.7C / 6.7D summaries may not be rewritten after observing
the result.

## 20. Freeze rule

This preregistration is frozen at its SHA-256. The evaluator, its reproduction
script, its CI workflow and its bounded tests are frozen before the run. The
confirmatory run happens once, as a deliberate human action under the token of
Section 18; no commit, test or CI run may supply that token, and the authoring
agent must never invoke `--full` with it.
