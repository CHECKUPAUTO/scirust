# TDI-6.7 — Observable-Offset Cross-Generator Transfer: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-6.7 run. The design
was frozen before the evaluator was written and before any data was generated
(`docs/TDI-6.7-OBSERVABLE-OFFSET-TRANSFER-PREREGISTRATION.md`, SHA-256
`46e92e9f8abf1bab828c45b18128e3b8affe6a6546f30800015e169c6082993b`, merged alone
in PR #47 before a single line of `v67` existed) and the run was a deliberate
one-time human action under the exact confirmation token. Per §19 of that
preregistration, no classification below may be rewritten.

**Headline.** TDI-6.6C left one escape route open: if the residual transfer
failure is a *level* failure, and if the level shift is visible at the
**observation horizon**, then a label-free additive correction could reach it.
`O₂` is a feature on every record, so `Δ = μ₂ᵀ − μ₂ˢ` is computable without
target labels. TDI-6.7 tests exactly that, and **closes the route**.

Criterion **TDI-6.7A is *Harmful* in all four confirmatory cells**
(relative MSE reduction −2.13 … −5.38). Criterion **TDI-6.7B returns
not repaired**, with all four cells located as non-repairs.

But the interesting result is *why*, and it is not the reason the
preregistration anticipated. §14's diagnostic shows `Δ` under-estimates the true
level shift (ratio 0.57 … 0.85 at U₃, 0.25 … 0.94 at U₆), which invites the
reading "the estimator is too weak". **That reading is wrong.** The frozen
model, applied to a foreign target with *no* correction at all, has already
moved its prediction past the target's mean: on the confirmatory cell the target
mean is −3.365 in source-standardized units and arm B0 already predicts −4.002.
There is no missing level to add. A post-hoc counterfactual (§8) settles it: a
**perfect** `Δ` would improve the aggregate bias in **9 of 24** cells — *fewer*
than the imperfect one's 12. The additive-offset model of cross-generator
transfer is not badly estimated; it is the wrong model.

The experiment also yields an exact structural fact (§5): because B1 differs
from B0 by an additive constant, it moves **only** the bias and leaves the
residual spread untouched — verified across all 144 blocks to a maximum relative
deviation of `3.044e-12`. The preregistered four-way classification therefore
reduces, empirically, to a single question — does `|bias|` shrink? — and that
rule reproduces **47 of 47** decided cells with zero counter-examples.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `26e4a73e6a0df01c67fd45e18b45113d87640188` |
| Evaluator | `tdi-bench/src/bin/tdi-independent-overlap-ablation-v67.rs` |
| Evaluator SHA-256 | `f47b4b295431009ae78ebe292467411a172203d5ca84c1f69bec56edb629577d` |
| Preregistration SHA-256 | `46e92e9f8abf1bab828c45b18128e3b8affe6a6546f30800015e169c6082993b` |
| Scientific-manifest SHA-256 | `f7e55d17cc009e485e5b3c7043db00d9b7d908899f8912c5f4768db3d1aa6e2d` |
| Result log SHA-256 | `825306be6c70e9e664e21790d31d3f476d91a1132c9c49980c1aa8538508826d` |
| Result log size | 24 886 lines |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 (c980f4866 2026-06-30) |
| Host | Linux `tarek` 6.8.12-tegra, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-30T16:19:25Z / 2026-07-30T17:27:22Z (1 h 07 min 57) |

The committed log has been independently rehashed in this working tree, from the
repository root, to
`825306be6c70e9e664e21790d31d3f476d91a1132c9c49980c1aa8538508826d`, matching the
value the run recorded for itself. The evaluator and preregistration hashes in
the metadata match the frozen manifests. The full frozen ancestor chain —
TDI-5.1 … 5.9, 6.1, 6.2, 6.5 and **6.6 as the scientific ancestor** — was
verified before any generation. Reproduction is tolerance-based (§17); `g` and
`τ_ε` remain the only non-exact quantities and the additive offset introduces
none.

## 2. Design recap

One factor changes from TDI-6.6: **what the transfer arm corrects**.

| Arm | Construction | Uses target labels? |
|---|---|---|
| **B0** `SourceStandardized` | source feature statistics, source target scaler | no |
| **B1** `ObservableOffset` | B0 **plus** `Δ̂_std = Δ / sˢ_h` added to the intercept | no |
| **B2** `OracleTargetScaler` | B0 with the target domain's own target scaler | **yes — bound only** |

`Δ = μ₂ᵀ − μ₂ˢ` is the difference in mean **observed-horizon** deficit between
the two domains' **training** populations. It never touches a holdout and never
touches a label. Everything else — the four frozen generator families F0–F3, the
one-step `Noop` kernel, widths 3–4, the GK/GKT layouts, the linear ridge with
λ = 1, three independent seed blocks per family, the 480 000 accepted records —
is inherited unchanged.

Criteria: **6.7A** B1-vs-B0 on the confirmatory pair F0-base → F1-sparse at the
focal horizons U₃ and U₆ (primary); **6.7B** `calibration_repaired ≡ R² > 0`
(primary); **6.7C** recovered fraction of the oracle (descriptive); **6.7D** all
12 ordered pairs (descriptive), with the preregistered reading rule that any
*Beneficial* cell whose `R²` stays ≤ 0 must carry the qualification on the same
line.

## 3. Criterion TDI-6.7A — *Harmful* in all four confirmatory cells

F0-base → F1-sparse, B1 against B0, frozen four-way classifier, symmetric 2 %
margin:

| Layout | Horizon | Relative MSE reduction | Classification |
|---|---|---|---|
| GK | U₃ | **−4.578844** | *Harmful* |
| GK | U₆ | **−2.125168** | *Harmful* |
| GKT | U₃ | **−5.380269** | *Harmful* |
| GKT | U₆ | **−2.186298** | *Harmful* |

All four are unanimous on the full three-condition rule: 3 of 3 blocks
confirming harm, aggregate relative worsening beyond 2 %, and the aggregate
bootstrap upper bound strictly negative. Nothing is marginal.

## 4. Criterion TDI-6.7B — calibration is not repaired

**TDI-6.7B: not repaired.** All four cells are located non-repairs.

| Layout | Horizon | `R²` B0 | `R²` B1 | `R²` B2 (oracle) |
|---|---|---|---|---|
| GK | U₃ | −2.574860 | **−18.943586** | −20.442876 |
| GK | U₆ | −1.781292 | **−7.692007** | −17.770325 |
| GKT | U₃ | −1.316672 | **−13.780990** | −15.882831 |
| GKT | U₆ | −1.145860 | **−5.837350** | −15.518498 |

Every 95 % bootstrap interval under B1 sits far below zero (e.g. GKT U₃:
[−14.098818, −13.778966]). This is the same shape as TDI-6.6A/B — a candidate
label-free repair that makes the collapse worse — but through a completely
different mechanism, which §5 makes exact.

One contrast with TDI-6.6 deserves recording. There, arm A1 degraded the level
while **improving the ordering** (Spearman 0.717 → 0.792). Here it does not, and
cannot: B1 adds a constant, and a constant preserves ranks. Per seed block the
Spearman changes by at most `1.34e-6` across all 144 blocks. The consolation
TDI-6.6 offered is structurally unavailable to this family of corrections.

## 5. Why B1 fails — an additive constant moves the bias and nothing else

### 5.1 The exact identity

Within a seed block, B1's prediction is B0's plus a constant `b = Δ̂_std`.
Therefore

    MSE₁ − MSE₀  =  b² + 2·b·bias₀  =  bias₁² − bias₀²

exactly, and the residual spread `MSE − bias²` is **invariant**. This is not a
modelling assumption; it is arithmetic, and the log confirms it: over all
**144** blocks (12 pairs × 2 layouts × 2 horizons × 3 blocks) the maximum
relative deviation of `MSE − bias²` between B0 and B1 is

    3.044e-12          (worst: F0-base → F3-local, GK, U₆, block F0-base/b2)

i.e. machine noise. **B1 cannot touch the part of the error that is not a level
offset.** Since 6.7A/6.7D classify on relative MSE, the preregistered classifier
reduces empirically to "does `|bias|` shrink?" — a rule that reproduces
**47 of 47** decided cells with **zero counter-examples**, the 48th being the
single *Equivalent* cell (F2-dense → F1-sparse, GK, U₆), which a binary rule
cannot express by construction.

### 5.2 The mechanism is visible in one line

Block `F0-base/b0`, GKT at U₃ — the confirmatory cell:

| Arm | mean observed | mean predicted | bias |
|---|---|---|---|
| B0 | −3.373824479841 | **−4.018583934274** | −0.644759454432 |
| B1 | −3.373824479841 | **−6.022429923646** | −2.648605443805 |

`bias(B1) − bias(B0) = −2.003845989`, exactly the block's `Δ̂_std`. The
correction lands precisely where it was designed to land.

The problem is the starting point. In source-standardized units the target's
mean is **−3.374** — and B0, with *no* correction whatsoever, already predicts
**−4.019**. The frozen model has already carried the level shift across the
domain boundary through the features (δ, δ̄, s₂, s₃, `g`, `τ_ε`, and under GKT
also `O₁`, `O₂`), and then some. Adding a further −2.004 drives the prediction to
−6.022.

Aggregated, at GKT, the sign of `bias₀` is **negative in 19 of 24 cells** — the
frozen model systematically under-predicts the foreign target's deficit — while
the sign of `Δ` is positive only when the target family is *harder* than the
source. The correction can therefore cancel the bias only on transfers toward a
harder target, and reinforces it on every transfer toward an easier one:

| Pair (GKT, U₃) | mean observed | pred. B0 | pred. B1 | bias B0 | bias B1 | verdict |
|---|---|---|---|---|---|---|
| F0→F1 | −3.3654 | −4.0016 | −6.0019 | −0.6362 | −2.6365 | *Harmful* |
| F0→F2 | 7.5301 | 3.4222 | 8.4566 | −4.1078 | +0.9265 | *Beneficial* |
| F0→F3 | −3.0663 | −3.3753 | −5.1219 | −0.3090 | −2.0557 | *Harmful* |
| F1→F2 | 14.9601 | 4.6219 | 14.2804 | −10.3382 | −0.6797 | *Beneficial* |
| F2→F1 | −5.3130 | −29.3066 | −32.7375 | −23.9936 | −27.4245 | *Harmful* |
| F3→F2 | 14.5167 | 3.9608 | 13.2547 | −10.5559 | −1.2620 | *Beneficial* |

## 6. Criterion TDI-6.7C — the oracle is not an upper bound

On the confirmatory pair, `recovered_fraction` is **not-applicable in all four
cells**, because the condition `R²_B2 > R²_B0` fails everywhere: the oracle arm
is *worse than doing nothing* (−15.88 against −1.32 at GKT U₃).

This is not a degenerate accident. Extending the computation to all 12 ordered
pairs at GKT — see §10 on why this had to be computed rather than read —
`R²_B2 > R²_B0` holds in only **8 of 24** cells, and where the fraction is
defined it is:

| Pair (GKT) | U₃ | U₆ |
|---|---|---|
| F2-dense → F0-base | **+0.227892** | −0.078823 |
| F2-dense → F1-sparse | −0.142135 | −0.055093 |
| F2-dense → F3-local | −0.156477 | −0.066247 |
| F3-local → F1-sparse | *n/a* | +1.097163 |
| F3-local → F2-dense | *n/a* | +1.973366 |

§12 framed this quantity as placing B1 "on the segment between doing nothing and
knowing the answer". **B1 lies on that segment in exactly 1 of 24 cells.** Six of
the eight defined fractions are negative — B1 moves *away* from the oracle — and
two exceed 1.

The reason the oracle fails is worth stating plainly, because it refines
TDI-6.6C rather than contradicting it. TDI-6.6's oracle arm A2 replaced the
target scaler **and** the feature statistics, and it repaired calibration in all
four confirmatory cells (`R²` +0.18 … +0.68). TDI-6.7's B2 replaces **only** the
target scaler, and repairs nothing. Substituting the true target scale into a
model whose coefficients were fitted in the *source's* standardized geometry is
not a partial repair — it is a mismatch. The two arms live in different
experiments on independent populations (§9), so this is a directional
observation, not a controlled contrast; but the effect sizes are three orders of
magnitude larger than the population-to-population variation, and the direction
is unambiguous.

### 6.1 The §14 diagnostic — `Δ` against the true level shift

The observable shift is a *biased* estimator of the quantity B1 needs, and the
bias grows with the horizon, exactly as the target geometry `U_h = −log₂(1 − O_h)`
predicts: the deficit accumulates, `Δ` is measured at the observation horizon 2,
and the focal horizons are 3 and 6.

| Unordered pair | `Δ` (observable, U₂) | ratio `Δ` / true at U₃ | at U₆ |
|---|---|---|---|
| F0 ↔ F1 | ∓2.025794748 | 0.594158 | 0.274725 |
| F0 ↔ F2 | ±5.098389206 | 0.668686 | 0.339772 |
| F0 ↔ F3 | ∓1.768854691 | 0.569388 | 0.249085 |
| F1 ↔ F2 | ±7.124183954 | 0.645657 | 0.318339 |
| F1 ↔ F3 | ±0.256940057 | 0.848179 | 0.942950 |
| F2 ↔ F3 | ∓6.867243897 | 0.639940 | 0.310640 |

The ratio is symmetric under reversal (both `Δ` and the true shift change sign),
below 1 everywhere, and roughly halves from U₃ to U₆ — except for F1 ↔ F3, the
two easiest families (mean `u₂` = 0.423 and 0.680), where the deficits barely
move and the ratio approaches 1.

The natural inference — "the estimator is too weak, a better one would work" —
is false. §8 shows why.

## 7. Criterion TDI-6.7D — twelve ordered pairs, split exactly down the middle

**Direction consistency: no.** Twelve divergent pairs are named at GKT.

| Layout | *Beneficial* | *Harmful* | *Equivalent* |
|---|---|---|---|
| GK | 14 | 9 | 1 |
| GKT | 12 | 12 | 0 |
| **Total (48 cells)** | **26** | **21** | **1** |

At GKT the split is exactly 12/12. Applying the preregistered reading rule of
§13, of the 12 *Beneficial* GKT cells only **4** have `R² > 0`:

| *Beneficial* with repaired calibration | reduction | `R²` under B1 |
|---|---|---|
| F0-base → F2-dense, U₃ | +0.823218 | +0.183506 |
| F1-sparse → F0-base, U₆ | +0.710184 | +0.588507 |
| F3-local → F0-base, U₃ | +0.758450 | +0.416553 |
| F3-local → F1-sparse, U₆ | +0.094428 | +0.485908 |

The other **8** carry the qualification the preregistration demands — *less bad,
not good* — including F1-sparse → F2-dense at U₆, where a 69 % MSE reduction
still leaves `R² = −2.596821`, and F2-dense → F0-base at U₃, where an 18.7 %
reduction leaves `R² = −369.412512`.

Symmetrically, three cells are *Harmful* where B0 already worked, and B1 spoils
it:

| *Harmful* despite a working baseline | `R²` B0 | `R²` B1 |
|---|---|---|
| F1-sparse → F3-local, U₃ | +0.788077 | +0.626334 |
| F1-sparse → F3-local, U₆ | +0.537190 | +0.420887 |
| F3-local → F1-sparse, U₃ | +0.635914 | +0.604795 |

The classification is almost entirely determined by the **target** family: all 6
cells with target F2-dense are *Beneficial*, 5 of 6 with target F0-base are, 1 of
6 with target F1-sparse, and **0 of 6** with target F3-local. Ordered by mean
`u₂` — F1-sparse 0.423 < F3-local 0.680 < F0-base 2.449 < F2-dense 7.547 —
transfers *toward difficulty* are helped and transfers *toward ease* are hurt,
which is precisely the sign rule of §5.2 restated in family terms.

### 7.1 Per-family context (§15, no criterion)

| Family | mean `u₁` | mean `u₂` | δ | δ̄ | s₂ | s₃ | `g` | `τ_ε` |
|---|---|---|---|---|---|---|---|---|
| F0-base | 0.855963 | 2.448795 | 0.948403 | 0.582231 | 1.122019 | 1.016617 | 0.640981 | 0.001163 |
| F1-sparse | 0.177222 | 0.423000 | 1.000000 | 0.883375 | 1.658150 | 1.430117 | 0.134111 | 0.372022 |
| F2-dense | 3.463053 | 7.547184 | 0.104554 | 0.073514 | 1.005987 | 0.999413 | 0.925757 | 0.000244 |
| F3-local | 0.331496 | 0.679940 | 1.000000 | 0.864069 | 3.641795 | 2.072226 | 0.090333 | 0.402366 |

Inter-family ranges: δ = 0.895446, δ̄ = 0.809861, s₂ = 2.635808, s₃ = 1.072813,
`g` = 0.835424, `τ_ε` = 0.402122.

## 8. A perfect `Δ` would help *less* — post-hoc counterfactual

This section is **not preregistered**. It is arithmetic performed on published
quantities after the fact, and it changes no criterion; §19 forbids rewriting the
6.7A–D summaries and nothing here does.

Because B1 shifts predictions by exactly `Δ̂_std`, the bias an *oracle* additive
offset would produce is computable in closed form from the logged values:
`bias₀ + Δ̂_std / (Δ/true)`. Substituting the true shift for the observable one:

| | improves the aggregate bias |
|---|---|
| B1 as run (observable `Δ`) | **12 of 24** GKT cells |
| B1 with a **perfect** `Δ` | **9 of 24** GKT cells |

On the confirmatory pair the perfect offset is strictly worse: at GKT U₃ the bias
would go to **−4.0029** against B1's actual −2.6365, from a baseline of −0.6362.

The under-estimation documented in §6.1 is therefore, in aggregate, an
*accidental mitigation*: `Δ` is too small, and too small is closer to the
correct answer — which is *no additive correction at all* — than the true shift
would be. This is the decisive evidence that the failure is not estimation
error. **The additive-offset model of cross-generator transfer is wrong at the
level of its form, not its calibration.**

## 9. Relation to TDI-6.6 — independent populations

TDI-6.7 draws **fresh, disjoint seed blocks** as §7 requires: generation seeds
begin at `7 400 000 000` where TDI-6.6's began at `6 200 000 000`. Arms B0 and
A0 are the same construction, but on independent populations, and they differ
accordingly — at GKT U₃ on the confirmatory pair, `R²` −1.316672 here against
−1.387165 there. **No number in this document should be compared bit-for-bit
with TDI-6.6's.**

What *does* replicate is the family characterization. F0-base's descriptors
across two independent 120 000-record draws:

| | δ | δ̄ | s₂ | s₃ | `g` | `τ_ε` |
|---|---|---|---|---|---|---|
| TDI-6.6 | 0.949736 | 0.582776 | 1.121779 | 1.017344 | 0.640872 | 0.001330 |
| TDI-6.7 | 0.948403 | 0.582231 | 1.122019 | 1.016617 | 0.640981 | 0.001163 |

Agreement to ~10⁻³ on all six descriptors, on populations sharing no seed. The
families are stable objects, not artefacts of one draw.

### 9.1 The pooled-Spearman effect, reproduced at 1700× the amplitude

TDI-6.6 §8.1 documented — after correcting a wrong first hypothesis — that
pooling three seed blocks, each carrying its own scaler, makes the map between
two arms' pooled ground truth **piecewise affine**, so pooled rank statistics are
not preserved even when per-block ones are. There the effect measured
`6.364e-05`. Here the same mechanism is visible with a clean control and a far
larger amplitude, because each block carries its own `sˢ_h` and therefore its own
`Δ̂_std`:

    |ΔSpearman| max, per block   B0 → B1 :  0.000001340197
    |ΔSpearman| max, pooled      B0 → B1 :  0.108039099981   (F3-local → F2-dense, GKT, U₃)

An additive constant preserves ranks *within* a block — the per-block figure is
f64 tie-breaking noise — and demonstrably does not preserve them across the
pooled concatenation. This is an independent confirmation of the corrected
TDI-6.6 §8.1, obtained without seeking it. No criterion in TDI-6.7 reads a
pooled rank statistic, so nothing here is affected; the figure is recorded so
that no reader infers "additive shift ⇒ invariant Spearman" from the tables.

## 10. Erratum — stale labels in the printed artifact

The evaluator is frozen and the run has been executed against it, so the log is
left exactly as produced. Four labelling defects, inherited by copy-derivation
from TDI-6.6, are recorded here instead:

1. The raw metric-block headers begin `=== TDI-6.6 — …` and read
   `A1 contre A0`, in the TDI-6.7 log.
2. The TDI-6.7A criterion and verdict lines read `A1 contre A0` where the arms
   are named `B0-source` and `B1-observable-offset` everywhere else. The
   TDI-6.7D lines correctly read `B1 contre B0`.
3. Section references carry TDI-6.6's numbering: `(Section 19)` and
   `(Section 17, items 14-15)` where TDI-6.7's raw-output section is §16, and
   `(Sections 12-14)` where the confirmatory criteria are §10–12. The
   `[NON comparables entre bras — Section 6]` annotation is correct.
4. §13 requires `recovered_fraction` and per-arm `calibration_repaired` for all
   12 ordered pairs; the verdict lines print them only for the confirmatory
   pair. The underlying per-arm `R²` values *are* logged for every pair, layout
   and focal horizon, which is why §6 could compute the missing fractions —
   they are derived in this document, not read from the log.

The computation is unaffected. The comparison is wired to the correct arms —
`baseline_fit` resolves to `TransferArm::SourceStandardized` and
`challenger_fit` to `TransferArm::ObservableOffset` — and only the identifier
`a1_vs_a0` and its printed captions are stale. Item 4 is a reporting gap against
the preregistration, not a data gap.

This is the **fourth generation** of the same defect class in this campaign
(after v57's TDI-5.6 leftovers, v66's `evaluator_source()` reading its ancestor,
and v67's inherited script description, both caught pre-run). Copy-derivation
reliably carries stale strings forward, and reliably fails to carry stale
*section numbers* into anyone's attention. Any future derivation must diff every
literal string against the new preregistration before freezing, not only the
logic.

## 11. Interpretation and boundaries

What TDI-6.7 establishes:

- A label-free **additive** correction estimated from the observed horizon does
  not repair cross-generator transfer. It is *Harmful* in all four confirmatory
  cells and does not repair calibration in any of them.
- It cannot, structurally, repair anything but a level offset: it leaves the
  residual spread invariant to `3.044e-12` across 144 blocks.
- The failure is **not** estimation error in `Δ`. A perfect `Δ` would help in
  fewer cells (9/24 against 12/24), and would be strictly worse on the
  confirmatory pair.
- Replacing only the target scaler, even with oracle knowledge, is not a partial
  repair: B2 is worse than B0 in 16 of 24 GKT cells.

Combined with TDI-6.6, the pessimistic reading of TDI-6.6C is now **complete for
this family of corrections**: neither label-free feature re-standardization
(6.6, A1) nor a label-free additive level correction (6.7, B1) repairs
cross-generator transfer, and the two failures have different, fully identified
mechanisms. Per §19, this is a result, not a null.

What it does **not** establish:

- that **B2** is attainable — it reads target labels and was a bound only, and
  in this experiment it is not even that;
- anything about **cross-width** transfer; TDI-5.8B stands as reported;
- that a **two-parameter** (offset *and* scale) or per-record correction would
  behave the same way — only the additive one-parameter form was tested;
- that a **refitted** model would fail — coefficients are frozen by design, and
  §5's identity applies only to frozen coefficients;
- generality beyond four synthetic families of small finite-state systems.

The 12/12 split at GKT is *not* evidence that the correction works half the
time. It is evidence that its sign is decided by the target family, which §5.2
makes mechanical and which no label-free procedure can know in advance without
already knowing the answer.

## 12. Reproduction

    TDI67_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI67_FREEZE_RULE \
      bash scripts/reproduce-tdi6.7.sh

The script refuses a dirty repository, refuses to overwrite an existing or
incomplete output, verifies the full frozen hash chain before any generation,
verifies the presence of the final criterion lines afterwards, and writes
read-only artifacts under `results/tdi6.7-observable-offset-transfer/`. No CI
workflow sets the confirmation variable; the full run is a deliberate one-time
human action.

Artifacts: `results/tdi6.7-observable-offset-transfer/` — the log, its SHA-256,
the run metadata and the completion marker.
