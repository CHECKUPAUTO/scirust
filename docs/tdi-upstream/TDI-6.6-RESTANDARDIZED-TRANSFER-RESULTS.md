# TDI-6.6 — Re-standardized Cross-Generator Transfer: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-6.6 run. The design
was frozen before the evaluator was written and before any data was generated
(`docs/TDI-6.6-RESTANDARDIZED-TRANSFER-PREREGISTRATION.md`, SHA-256
`5216be5d8103505d63a0c9fd78d57f668d7005afd525a8481e259736df7276a9`) and the run
was a deliberate one-time human action under the exact confirmation token. No
classification below may be rewritten.

**Headline.** The candidate fix fails, and it fails informatively.
TDI-5.8 §6 and TDI-6.5 §5 both propose the same untested explanation for the
cross-domain transfer failure — descriptor drift — and TDI-5.8 names the
candidate remedy: re-standardize with the target domain's statistics before
transferring. **Label-free feature alignment does not repair calibration. It
makes it dramatically worse**: criterion TDI-6.6A is ***Harmful*** in all four
confirmatory cells, and under arm A1 the standardized-U `R²` collapses from
−1.23 … −2.73 to −20.7 … −29.1. Criterion TDI-6.6B returns **not repaired**.

The oracle arm locates the failure exactly. A2 differs from A1 by **one thing**
— it also replaces the target scaler — and A2 repairs calibration in **all four**
cells (`R²` +0.18 … +0.68). What cross-generator transfer is missing is not the
feature scale but knowledge of the target domain's **deficit level** — which is
the very quantity being predicted. **No label-free procedure of this family can
supply it.**

One genuine consolation: A1 degrades the level while *improving the ordering*
(Spearman 0.717 → 0.792 on the confirmatory cell; 0.105 → 0.723 on F1→F0/GK/U₃).
The "calibration dies, ordering survives" pattern of TDI-5.8B and TDI-6.5C now
has a lever attached to it.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `657dc397be011ec8bc2e6e77685caebcd5d3bc58` |
| Evaluator | `tdi-bench/src/bin/tdi-independent-overlap-ablation-v66.rs` |
| Evaluator SHA-256 | `b563a712b80bd983bfc3d6c7d4ffecc7b7fec35e6f3be8fea539ecb984346e7b` |
| Preregistration SHA-256 | `5216be5d8103505d63a0c9fd78d57f668d7005afd525a8481e259736df7276a9` |
| Scientific-manifest SHA-256 | `31979a44973e484fc4baccfd03e8c37a978f7443b8df86110ec8919596291bb6` |
| Result log SHA-256 | `c9cfa9d0c8cd2b1ac9cd31a6a059ae6c1d7269f77538790aa91208efa5e29c0a` |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 (c980f4866 2026-06-30) |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-30T09:40:10Z / 2026-07-30T10:48:39Z (1 h 08 min 29) |

The committed log has been independently rehashed in this working tree to
`c9cfa9d0c8cd2b1ac9cd31a6a059ae6c1d7269f77538790aa91208efa5e29c0a`, matching the
value the run recorded for itself. The full frozen ancestor chain, including
TDI-6.5 as the scientific ancestor, was verified before any generation.
Reproduction is tolerance-based; `g` and `τ_ε` remain the only non-exact
quantities and re-standardization adds none.

> **Cosmetic defect in the artifact.** The metadata header carries
> `experiment=TDI-6.6 generator-family robustness of the literal-spectral
> control` — a description string left over from deriving the reproduction
> script from TDI-6.5's. It mislabels the experiment's *subject*. No number,
> hash, seed or criterion is affected, and the artifact is left as produced
> rather than retouched after the fact.

## 2. Design recap

A fitted model is `β` plus the statistics that standardize it. **`β` is never
refitted**; only the standardization changes.

| Arm | feature `μ, σ` | target `(m, s)` | reads a target label? |
|---|---|---|:--:|
| **A0** — source (TDI-6.5C unchanged) | source | source | no |
| **A1** — feature re-standardized | **target** | source | **no** |
| **A2** — oracle | **target** | **target** | **yes** |

Target statistics come from the target family's combined **training**
populations, never the scored holdout, so the design is not transductive. A2 is
an upper bound, never a prediction method.

**Populations.** Four families, three fresh blocks each, 40,000 accepted per
block — **480,000 accepted** for 548,231 attempted, i.e. **68,231 preregistered
exclusions**, distributed as in TDI-6.5: F0-base 262, F1-sparse 8,153,
F2-dense 58,002, F3-local 1,814.

## 3. Criterion TDI-6.6A — label-free alignment is *Harmful*, at every cell

F0-base → F1-sparse, A1 against A0, aggregate relative MSE reduction:

| Layout | Horizon | Relative MSE reduction | Blocks confirming harm | Classification |
|---|---|---:|:--:|:--:|
| GK | U₃ | **−4.957** | 3 / 3 | ***Harmful*** |
| GK | U₆ | **−9.296** | 3 / 3 | ***Harmful*** |
| GKT | U₃ | **−8.079** | 3 / 3 | ***Harmful*** |
| GKT | U₆ | **−12.365** | 3 / 3 | ***Harmful*** |

A relative reduction of −8.079 means the error grew by a factor of about nine.
Every cell satisfies the classifier's full harm rule: 3/3 blocks confirming,
aggregate worsening beyond the 2 % margin, aggregate bootstrap upper bound
strictly negative.

**This is a result, not a null.** The preregistration (§12) fixed in advance
that *Harmful* would be as informative as *Beneficial*, because it retires the
drift explanation. It does more than retire it: the proposed remedy is
counterproductive.

## 4. Criterion TDI-6.6B — calibration is not repaired

Aggregate standardized-U `R²` per arm, on the confirmatory pair:

| Layout | Horizon | A0 | **A1** | A2 (`oracle`) |
|---|---|---:|---:|---:|
| GK | U₃ | −2.731 | **−21.225** | **+0.179** |
| GK | U₆ | −1.921 | **−29.075** | **+0.406** |
| GKT | U₃ | −1.387 | **−20.672** | **+0.678** |
| GKT | U₆ | −1.233 | **−28.848** | **+0.595** |

**`repaired = no`**, with all four cells reported as located non-repairs. The
95 % bootstrap intervals are far from zero and do not overlap between arms
(e.g. GKT/U₃: A0 [−1.445, −1.387], A1 [−21.116, −20.665], A2 [0.671, 0.678]),
so nothing here is marginal.

Recall why this criterion is a sign test on `R²` rather than an MSE margin: A2
standardizes its ground truth with a different scaler, and `R² = 1 − MSE/Var`
is invariant to that affine rescaling while MSE is not (preregistration §14.1).
The A0/A1/A2 column comparison above is therefore legitimate; **the MSE columns
are not comparable across the A2 boundary and are never compared.**

## 5. Why A1 fails — the mechanism is visible in one line

On F0-base → F1-sparse, GK, U₃, block b0, in source-standardized U space:

| Quantity | Value |
|---|---:|
| observed mean | **−3.354** |
| A0 predicted mean | −4.527 |
| **A1 predicted mean** | **+0.002** |

The target domain sits 3.35 source-standard-deviations *below* the source mean.
A0 partially tracks that offset because it applies the source's feature means
and scales to the target's raw features — the resulting standardized features
retain their displacement relative to the source distribution, and that
displacement is what carries the level information.

A1 centres each feature on the **target domain's own** mean. By construction the
standardized features are then centred at zero in the target domain, so
`β₀ + Σβⱼ·(xⱼ−μⱼ)/σⱼ` collapses to approximately the intercept — and the
intercept was fitted on source-standardized data whose mean was zero. A1
therefore predicts ≈ 0 where the truth is ≈ −3.35.

**Feature alignment destroys exactly the signal it was supposed to fix.** The
domain shift was not noise corrupting the model; in A0 it was doing useful work.

This is a mechanism read off the reported means, and it is consistent across
every high-shift pair (e.g. F3-local → F2-dense at U₆: observed mean 16.497,
A1 predicted mean −0.0003). It is not a separately tested claim.

## 6. Criterion TDI-6.6C — the failure lives in the target scale

| Layout | Horizon | A1 repaired | A2 repaired |
|---|---|:--:|:--:|
| GK | U₃ | no | **yes** |
| GK | U₆ | no | **yes** |
| GKT | U₃ | no | **yes** |
| GKT | U₆ | no | **yes** |

**`residual_failure_in_target_scale = yes`** in all four cells. A2 and A1 share
identical feature statistics and identical, never-refitted coefficients; they
differ *only* in the target scaler. So the entire gap between "useless" and
"works" is the target domain's `(m, s)`.

That is the sharpest localization this design could produce, and its practical
consequence is negative and specific: **the missing quantity is the target
domain's deficit level, which is what the model is asked to predict.** Fitting
it requires target labels. No amount of unsupervised feature alignment reaches
it. Section 23 forbids reading A2 as attainable performance, and this document
does not.

## 7. Criterion TDI-6.6D — twelve ordered pairs, and a trap

**`direction_consistent = no`.** Across the 12 ordered pairs at GKT and both
focal horizons — 24 cells in all: **17 *Harmful***, 1 *Inconclusive*, and
**6 *Beneficial*, every one of them with F2-dense as the source.**

| Source | Target | U₃ reduction | `R²` under A1 | repaired |
|---|---|---:|---:|:--:|
| F2-dense | F0-base | **+0.835** | −72.20 | no |
| F2-dense | F1-sparse | **+0.973** | −311.91 | no |
| F2-dense | F3-local | **+0.981** | −262.00 | no |

**Those *Beneficial* labels must not be read as success.** A1 improves on A0
there only because A0 is even more catastrophic: F2-dense → F1-sparse at U₃ has
A0 `R² = −11746` against A1's −312. Both are unusable; A1 is merely less
absurd. This is the TDI-5.8B trap in a more extreme form, and the direction flag
exists precisely so that a reader cannot take the confirmatory pair's result as
universal without seeing that the sign flips for one source family.

**Two pairs actually transfer.** F1-sparse ↔ F3-local have `R² > 0` under *both*
A0 and A1 at both focal horizons (F3→F1 GKT: A0 0.650 at U₃ and 0.449 at U₆,
A1 0.496 and 0.440) — the only cells in the experiment where a transferred model beats
predicting the target's mean without an oracle. These are exactly the two
families TDI-6.5 identified as **censored** (`|λ₂| = 1` for a large fraction of
candidates, `τ_ε` pinned at the iteration ceiling). Their mutual transferability
was not predicted and is offered as an observation, not an explanation.

Descriptor drift is unchanged from TDI-6.5 within sampling noise:

| Family | δ | δ̄ | s₂ | s₃ | `g` | `τ_ε` |
|---|---:|---:|---:|---:|---:|---:|
| F0-base | 0.949736 | 0.582776 | 1.121779 | 1.017344 | 0.640872 | 0.001330 |
| F1-sparse | 1.000000 | 0.884369 | 1.651425 | 1.434200 | 0.132936 | 0.373461 |
| F2-dense | 0.104564 | 0.073621 | 1.005999 | 0.999402 | 0.925309 | 0.000244 |
| F3-local | 1.000000 | 0.863860 | 3.641395 | 2.078867 | 0.090606 | 0.404303 |
| **range** | 0.895436 | 0.810748 | **2.635396** | 1.079464 | 0.834703 | 0.404058 |

## 8. Ordering versus scale — and a numerical caveat

A1 damages the level and *helps* the ordering:

| Pair / cell | Spearman A0 | Spearman A1 |
|---|---:|---:|
| F0→F1, GKT, U₃ | 0.717 | **0.792** |
| F0→F1, GKT, U₆ | 0.588 | **0.783** |
| F1→F0, GK, U₃ | 0.105 | **0.723** |
| F1→F2, GK, U₃ | −0.600 | **0.879** |

The last row is striking: A0's ranking is *anti-correlated* with the truth
(−0.600) and feature alignment turns it into 0.879. So the finding of TDI-5.8B
and TDI-6.5C — calibration dies, ordering survives — gains an actionable
refinement: **if only the ranking is wanted, aligning the features is a real
improvement; if the level is wanted, it is actively harmful.**

### 8.1 Why A1 and A2 report slightly different Spearman values

A1 and A2 share identical feature statistics, and the standardized prediction
does not depend on the target scaler, so their standardized predictions are
**bit-identical**. A single target scaler is a monotone affine map and Spearman
is rank-based, so at first sight the two arms' Spearman values should coincide
exactly. They do not: the reported values differ by ~1 × 10⁻⁴ typically,
sign-varying, and by 5 × 10⁻³ at most (F0→F1).

**The cause is pooling, not arithmetic.** Aggregate metrics concatenate the
three seed blocks, and **each block carries its own target scaler**. Under A1
the three scalers are the source family's three per-block scalers; under A2 they
are the target family's. The map taking A1's pooled ground truth to A2's is
therefore not a single affine map but a **piecewise** one, with a different
piece per block — and a piecewise-affine map does **not** preserve the global
rank order of the pooled vector. Two records in different blocks can swap
relative rank between arms, which moves the pooled Spearman.

This is demonstrated, not conjectured. Simulating the pooled computation with
three blocks and per-block scalers reproduces a discrepancy of the observed
order (≈ 6 × 10⁻⁵); forcing the scalers to be **identical across blocks**, with
everything else unchanged, drives the discrepancy to **exactly zero**. The
control isolates the cause.

> **Correction.** An earlier revision of this section attributed the discrepancy
> to floating-point tie formation in the average-rank assignment. That
> explanation is **wrong** and has been tested: an affine rescaling does not
> differentially merge `f64` values, and a direct simulation produced a Spearman
> difference of exactly zero. It was reported at the time as an undemonstrated
> conjecture; it is now replaced by the demonstrated cause above.

**Nothing else is affected.** No criterion depends on Spearman. No prior
experiment is touched either: every earlier comparison contrasts two *layouts*
evaluated under the **same** per-block scalers, so their pooled ground truth is
identical and the effect cannot arise. It appears in TDI-6.6 only because this
is the first design that compares predictors carrying **different** target
scalers. The per-block Spearman values printed in the raw output are unaffected
by construction, and the reported A0→A1 ordering gains of Section 8 are three
orders of magnitude larger than this effect.

## 9. Interpretation and boundaries

TDI-6.6 closes the question TDI-5.8 and TDI-6.5 both left open, in the negative
and with a mechanism: the cross-domain transfer failure is **not** a feature-scale
artifact, and the remedy those documents floated makes matters worse. The
failure is located in the target scale, and reaching it requires the labels one
is trying to predict.

What this does **not** establish:

- **that A2's performance is attainable.** A2 reads the target domain's `U_h`
  values. It is a bound, never a method;
- **anything about cross-width transfer.** TDI-5.8B stands exactly as reported;
  TDI-6.6 tested the generator axis only;
- **that no domain-adaptation scheme can work.** Only mean/standard-deviation
  alignment was tested. Whitening, CORAL, distribution matching, importance
  weighting and any scheme that estimates a target *offset* from unlabelled
  inputs are all untested — and §5's mechanism suggests the last of those is
  where to look next;
- **that a refitted model would fail.** Coefficients are frozen by design (§4.5);
  TDI-6.5A already showed models fit *within* each family work well;
- generality beyond four hand-specified synthetic families, widths 3–4, the
  linear ridge, and the one-step `Noop` kernel; nothing outside small synthetic
  finite-state families.

The TDI-6.6A / 6.6B / 6.6C / 6.6D summaries are frozen as reported.

## 10. Reproduction

    TDI66_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI66_FREEZE_RULE \
      bash scripts/reproduce-tdi6.6.sh

The script refuses without the exact token, refuses a dirty repository, verifies
the full frozen hash chain before any generation, executes the evaluator once
with `--full`, verifies the final criterion lines, and writes read-only
artifacts under `results/tdi6.6-restandardized-transfer/`. Budget roughly
1 h 15 on the reference host.
