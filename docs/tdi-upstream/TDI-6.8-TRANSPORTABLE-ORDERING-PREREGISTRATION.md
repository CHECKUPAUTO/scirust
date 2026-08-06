# TDI-6.8 — Transportable Ordering Across Generator Families

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

TDI-6.8 derives from **TDI-6.7** (`tdi-independent-overlap-ablation-v67.rs`,
SHA-256 `f47b4b295431009ae78ebe292467411a172203d5ca84c1f69bec56edb629577d`;
preregistration SHA-256
`46e92e9f8abf1bab828c45b18128e3b8affe6a6546f30800015e169c6082993b`) by exactly
**one changed factor**:

> **What the criteria measure.** TDI-6.7 measured the transferred *level* —
> relative MSE and `R²`. TDI-6.8 measures the transferred **ordering** —
> per-seed-block rank correlation between prediction and truth on the target
> domain's holdout.

Everything else is inherited verbatim: the four generator families, the
population contract, the one-step `Noop` kernel, the exact and non-exact
descriptors, the CK/SK/GK/GKT layouts, the linear ridge (λ = 1), the horizons
and focal horizons, the never-refitted coefficients, the twelve ordered pairs,
the deterministic bootstrap, the three-condition classifier *structure*, and the
non-exact determinism discipline.

### 1.1 Why the comparison arms change, and why that is not a second factor

TDI-6.7 §5 established, as an exact algebraic property rather than an empirical
finding, that its arm B1 is arm B0 plus an additive constant **within each seed
block**. TDI-6.6's arms A1 and A2 differ from A0 by affine maps of the target.
A strictly increasing map preserves rank correlation exactly. Therefore, under a
rank criterion, **every correction arm of TDI-6.6 and TDI-6.7 is identical to
doing nothing, by construction** — a table of structural zeros, of the same kind
as `Unique(O₁) = 0` under MMI in TDI-6.3, and worth exactly as little.

The comparison dimension is therefore *forced*, not chosen: the only frozen
dimension of this design that can move a rank is the **feature layout**. TDI-6.8
compares the four inherited layouts under plain transfer with **no correction of
any kind**. This is a consequence of the single changed factor, not an
independent second change.

### 1.2 Why this experiment, stated plainly

The hypothesis under test is **not new to this document**. Four preregistered
experiments have now reported, each in a discussion section and never as a
criterion, the same asymmetry:

1. **TDI-5.8B** — cross-width transfer: every `R²` far below zero while rank
   ordering survives through the overlaps (Spearman 0.695) and not through the
   exact descriptors (−0.040);
2. **TDI-6.5C** — cross-generator transfer: calibration fails, ordering
   survives;
3. **TDI-6.6** — feature re-standardization is *Harmful* to the level while
   *improving* the ordering (Spearman 0.717 → 0.792 on the confirmatory cell);
4. **TDI-6.7** — an additive observable offset is *Harmful* to the level and
   provably cannot touch the ordering at all.

This document therefore states, without hedging, that its hypothesis is
**suggested by observations already made**. That is a real threat to the
preregistration discipline, and Section 1.3 states precisely what makes the test
legitimate anyway and what it costs.

### 1.3 What makes a rank criterion admissible now, and what it does not excuse

TDI-6.7 §1.3 **refused** to adopt a rank-based criterion. That refusal was
correct and is not being reversed on a whim. Three things have changed:

1. **The data those observations came from stays untouched.** TDI-6.8
   re-analyses no frozen log. It generates **fresh populations under fresh
   seeds** (Section 7), disjoint from every prior experiment. The hypothesis is
   old; the evidence that will decide it does not yet exist.
2. **A sound rank criterion has only just become constructible.** TDI-6.7 §9.1
   demonstrated, with a clean control, that the *pooled* rank statistic in this
   design is not trustworthy: pooling three seed blocks that each carry their
   own target scaler makes the map between two arms' pooled ground truth
   piecewise affine, and the pooled Spearman drifted by **0.108** where the
   per-block statistic was invariant to `1.34e-6`. Every rank number quoted in
   the four discussions above is a *pooled* number. A criterion built on them
   before TDI-6.7 would have been measuring a pooling artifact. Section 6
   therefore forbids the pooled statistic from entering any criterion.
3. **The criterion is frozen before the data exists**, at the SHA-256 of this
   document, with its margin, its three conditions and its reading rule fixed.

What this does **not** excuse: the effect size cannot be treated as an
independent replication of the four observations, because the hypothesis was
drawn from them. TDI-6.8 establishes whether the effect exists under a
preregistered criterion on fresh data. It does not establish that the effect was
discovered blind, and no result section may claim otherwise.

## 2. Research questions

1. Does adding the overlaps `O₁, O₂` to the strongest baseline improve the
   **transferred ordering** across generator families? (TDI-6.8A)
2. Does the ordering transfer **at all** — is the transferred rank correlation
   distinguishable from zero? (TDI-6.8B)
3. How much of the within-domain ordering survives the domain change?
   (TDI-6.8C)
4. Does the answer hold across all twelve ordered pairs and along the whole
   layout ladder? (TDI-6.8D)

## 3. The four layout arms

No correction of any kind is applied. For every ordered pair, the model fitted
on the **source**'s training populations is applied to the **target**'s
holdouts, carrying the source's feature statistics and the source's target
scaler — the plain-transfer configuration that TDI-6.7 called B0 and TDI-6.6
called A0. Coefficients are **never refitted**.

| Arm | Features | Count |
|---|---|---|
| **CK** | BASELINE + δ + δ̄ | 15 |
| **SK** | CK + `s₂` + `s₃` | 17 |
| **GK** | SK + `g` + `τ_ε` | 19 |
| **GKT** | GK + `O₁` + `O₂` | 21 |

The primary comparison is **GKT against GK** — the overlaps against the
strongest descriptor baseline the campaign has built, including the literal
spectral gap and the ε-mixing time. CK and SK are reported so the ladder is
visible; no criterion is defined on them.

## 4. Generator families, kernel, descriptors, layouts, model

Inherited unchanged from TDI-6.7 §4: families F0-base / F1-sparse / F2-dense /
F3-local; the one-step `Noop` kernel; exact δ, δ̄, `s₂`, `s₃`; non-exact `g`,
`τ_ε` with `ε = 1/4`, `T_max = 4096`, eigensolver tolerance `1.0e-12`; layouts
CK 15 / SK 17 / GK 19 / GKT 21; linear ridge `λ = 1`; observation horizon 2;
horizons `H = {3,4,5,6,7,8}`; focal horizons **U₃** and **U₆**.

## 5. Populations

Inherited from TDI-6.7 §5 under **fresh** seeds: per family three independent
seed blocks; per block 15,000 training and 5,000 holdout at width 3 and the same
at width 4 — so **10,000 holdout records per block**. 40,000 accepted per block,
120,000 per family, **480,000 total**. Generation budgets, attempt multipliers,
no-progress thresholds and the preregistered rejection categories are inherited
verbatim from TDI-5.2 §7.

## 6. The rank statistic, and the per-block rule

The primary statistic is **Spearman's ρ** between the predicted and the true
standardized `U_h`, computed **within a single seed block's holdout**, with
**average ranks for ties**.

**Pooling is forbidden as a criterion input.** No criterion in this document may
read a rank statistic computed across a concatenation of seed blocks. The reason
is TDI-6.7 §9.1: blocks carry different target scalers, the pooled map is
piecewise affine, and the pooled statistic drifted by 0.108 in that experiment
where the per-block statistic was invariant to `1.34e-6`. The pooled value is
still **printed**, for continuity with TDI-5.2 → 6.7, and every printed pooled
rank line must carry the marker `[POOLÉ — NON RECEVABLE COMME ENTRÉE DE
CRITÈRE — Section 6]`.

Per (pair, layout, horizon) the aggregate rank statistic is the **unweighted
mean of the three per-block ρ**, written `ρ̄`. Blocks are equal-sized by
construction (Section 5), so the mean needs no weighting.

**The criterion statistic lives in standardized-U space only.** Rank correlation
is invariant under a *strictly* increasing transformation, and `O_h = 1 − 2^(−U_h)`
is strictly increasing — but the reconstruction to O space **saturates** at the
representable bounds, and saturation is monotone without being strictly
increasing: it manufactures ties. The effect is not hypothetical. In TDI-6.7's
own frozen log, 202 reported blocks carry a non-zero bound fraction, and on
block `F0-base/b2` the same predictions give `ρ = 0.485180585367` in
standardized-U space and **exactly `0.0`** in reconstructed-O space.

This also settles the reading of an inherited line. TDI-5.8's results state that
"in reconstructed-O space the baseline collapses entirely (`R²` −35031, Spearman
exactly 0.000000000)". Its log shows `fraction borne basse = 1.0` on that block:
**every** prediction was clamped to the floor, so the predictions are a constant
and `ρ = 0` follows by definition. That zero measures saturation, not lost
ordering.

The reconstructed-O `ρ` is therefore reported as a **companion only**, always
beside its bound fractions, and may not enter any criterion.

**Kendall's τ-b** is computed for every cell as a companion (Section 15). It
enters **no** criterion. It exists so that the choice of Spearman cannot be
mistaken for a lever: if the two rank statistics disagree in direction anywhere,
that disagreement is reported.

## 7. Independent seed blocks (fresh)

Numeric seeds continue the series arithmetic with a fresh origin, and bootstrap
seeds follow the ASCII scheme `0x5444_4936_38…` ("TDI" + "6" + "8").

- population base seed: `base(f, b) = 8.6e9 + f·300e6 + b·100e6`, with the four
  populations at `base + {0, 10, 20, 30}·1e6`. The 8.6e9 origin clears TDI-6.7's
  last reservation (8,530,005,038), so every TDI-6.8 seed is disjoint from every
  prior experiment's;
- per-block bootstrap seed: `0x5444_4936_3800_0000 + (3·f + b) + 1`;
- per-family stratified aggregate seed: `0x5444_4936_3800_4800 + f`;
- per-ordered-pair bootstrap stream:
  `0x5444_4936_3800_4800 + 0x10·(1 + f_source) + f_target`, disjoint from the
  per-family seeds because pair offsets start at `0x10`.

Seed disjointness is checked in code and in a bounded test.

## 8. Deterministic bootstrap

4,000 replicates, frozen seeds as in Section 7. Resampling is **within a seed
block**, over that block's holdout records, and the two layouts of a comparison
are resampled with the **same indices** so the increment is paired. The
aggregate interval for `ρ̄` is the mean of the three per-block replicate values,
replicate by replicate — never a resampling of a pooled sample.

A replicate whose resampled predictions or truths are constant yields an
undefined `ρ`; such replicates are **counted and reported**, and the interval is
taken over the defined replicates. If more than 1 % of replicates are undefined
in any cell, that cell's interval is reported as **not-available** and its
criterion result as *Indeterminate*.

## 9. Non-exact determinism discipline

Inherited from TDI-6.7 §9: IEEE-754 binary64, single-threaded, fixed operation
order, no FMA or parallel reduction; eigensolver tolerance `1.0e-12`,
cross-method agreement `1.0e-9`, degenerate-scale floor `1.0e-12`. `g` and `τ_ε`
remain the only non-exact quantities; rank statistics add none — they are
comparisons and averaged integer ranks. The three-method spectral
cross-validation table is reproduced, sampling candidates in each family, and is
read as in TDI-6.7 §9: the rigorous witness is the trace residual.

Rank statistics are **more** robust to last-digit drift than MSE, with one
exception: exact ties broken differently could move a rank. The evaluator must
therefore use average ranks for tied values (Section 6), which is
tie-breaking-free, and must report the tie count per cell.

## 10. Criterion TDI-6.8A — does the overlap improve transferred ordering? (primary)

On the **F0-base → F1-sparse** transfer, at focal horizons **U₃** and **U₆**,
compare **GKT against GK**. Let

    Δρ = ρ̄(GKT) − ρ̄(GK)

with the frozen symmetric margin **`m = 0.02`, absolute**, on the bounded
[−1, 1] rank scale.

*Beneficial* iff **all three** hold:

1. `ρ(GKT) > ρ(GK)` in **all three** seed blocks;
2. `Δρ ≥ +0.02`;
3. the 95 % bootstrap lower bound of `Δρ` is strictly positive.

*Harmful* iff the mirror image holds in all three.

*Equivalent* iff **both**: all three per-block increments lie within ±0.02, and
the aggregate 95 % interval of `Δρ` lies entirely within ±0.02.

*Indeterminate* otherwise, including the undefined-replicate case of Section 8.

**Why `m = 0.02`.** With 10,000 holdout records per block, the standard error of
a single Spearman ρ is approximately `1/√(n−1) ≈ 0.010`. The margin is two such
standard errors, so an increment inside it cannot be distinguished from sampling
noise by the statistic itself. It is also the direct transposition of the
campaign's frozen 2 % relative-MSE margin onto a bounded scale. This value is
fixed here and may not be revisited after seeing a result.

The same comparison is reported for **GK against SK** and **SK against CK**, so
that any GKT effect can be read against what the descriptor ladder already
contributes. No criterion is defined on those two.

Preregistered classification, forced to no result. **No outcome is a success or
a failure.** *Harmful* or *Equivalent* would establish that the four prior
discussions were reading a pooling artifact or a within-domain effect, which is
a result of the same weight as confirmation.

## 11. Criterion TDI-6.8B — does the ordering transfer at all? (primary)

For each layout and focal horizon define

    rank_transfers ≡ ρ(GKT) > 0 in all three seed blocks
                     AND the 95 % bootstrap lower bound of ρ̄ is > 0

TDI-6.8B is the conjunction: **transfers** iff `rank_transfers` holds under
**GKT** at **both** focal horizons on the confirmatory pair; otherwise a
**located failure** naming each (layout, horizon) at or below zero.

This criterion exists because "ordering survives" has been *asserted* four times
and *tested* zero times. A transferred `ρ` indistinguishable from zero would
mean the surviving-ordering reading is unsupported, independently of TDI-6.8A's
outcome — and the two criteria can disagree: an increment can be *Beneficial*
while both correlations sit at zero.

Preregistered classification, forced to no result.

## 12. Criterion TDI-6.8C — how much ordering survives the domain change? (descriptive)

The same fitted model is scored on the **source's own** holdout, giving
`ρ̄_within`. Report, per layout and focal horizon,

    retention = ρ̄_transfer / ρ̄_within        when ρ̄_within > 0
                not-applicable                otherwise

This reads **source** holdout labels, which are available in the source domain
by construction and are never fitted on. It is **not** an oracle: no target
label is read anywhere in TDI-6.8, in any arm, for any criterion. The
experiment has no oracle arm, because there is no scale to supply.

Descriptive: no threshold is preregistered and no value is a success.

## 13. Criterion TDI-6.8D — all twelve ordered pairs and the whole ladder (descriptive)

Repeat Sections 10–12 for **all 12 ordered pairs** at both focal horizons and
all four layouts. Report per pair: the GKT-vs-GK classification, `rank_transfers`
per layout, `retention`, and the per-block ρ values. Report **direction
consistency** — whether the GKT-vs-GK classification is identical across all
pairs at both focal horizons — and name every divergent pair.

**Reading rule, preregistered.** A *Beneficial* increment accompanied by
`ρ̄(GKT) ≤ 0` means **better-ordered noise, not transfer**. Every such cell must
be reported with that qualification attached, in the same line. Symmetrically, a
*Harmful* increment where `ρ̄(GK) ≤ 0` means the baseline was not ordering
anything either, and must be marked likewise.

## 14. Ordering against domain distance (descriptive)

For each ordered pair and focal horizon, report `ρ̄(GKT)` alongside the
**label-free** domain distance `|ū₂ᵀ − ū₂ˢ|` — the gap in mean observed-horizon
deficit between the two domains' training populations, the same observable
statistic TDI-6.7 used as `Δ`. This asks whether transferred ordering degrades
with domain distance, and it is answerable without labels, which the level
question was not.

Descriptive: no threshold, no criterion.

## 15. Companion statistics (context only, no criterion)

Kendall's τ-b for every (pair, layout, horizon, block) beside the Spearman ρ,
with any direction disagreement between the two named explicitly. The pooled
Spearman, carrying its Section 6 marker. The reconstructed-O ρ, always printed
on the same line as that cell's lower and upper bound fractions, so a saturated
zero can never be read as a measured collapse. The tie counts per cell. The
per-family means of `u₁`, `u₂` with their across-family ranges, and per-family
descriptor drift (δ, δ̄, `s₂`, `s₃`, `g`, `τ_ε`) as in TDI-6.7 §15.

## 16. Required raw output

Provenance and the full frozen ancestor chain; frozen constants; family rules;
seed blocks; the spectral cross-validation table; per-family population counts,
rejection reasons, final exclusive seeds and budgets; per-layout normalization
summaries; for every (pair, layout, focal horizon) the three per-block ρ, their
mean, the paired bootstrap interval, the τ-b companion, the tie count and the
undefined-replicate count; the within-domain reference of Section 12; the
per-criterion block-level and aggregate conditions; and the final verdict lines
for 6.8A, 6.8B, 6.8C and 6.8D.

Every pooled rank line carries the Section 6 marker. Every reconstructed-O ρ
line carries its two bound fractions. Every `rank_transfers` line prints the
three per-block ρ and the interval, never the boolean alone. Every `retention`
line prints `ρ̄_transfer` and `ρ̄_within` separately.

## 17. Determinism

Given the frozen seeds and the declared floating-point regime, a faithful re-run
on the reference toolchain and architecture reproduces the log byte-for-byte.
Across architectures the last binary64 digits may differ; the ±0.02 margin and
the `ρ > 0` threshold are robust to that by many orders of magnitude, and
average-rank ties remove the one path by which last-digit drift could move a
rank. Generation is layout-independent: all four layouts score the same records.

## 18. Operational activation and full-run entrypoint contract

The evaluator exposes exactly `--termination-smoke`, `--preflight` and
`--full`; a bare invocation refuses. `--full` requires

    TDI68_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI68_FREEZE_RULE

and refuses on any other value, including an empty one. The reproduction script
additionally refuses a dirty repository, verifies the full frozen hash chain
before any generation, verifies the final criterion lines afterwards, and writes
read-only artifacts under `results/tdi6.8-transportable-ordering/`.

    TDI68_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI68_FREEZE_RULE \
      bash scripts/reproduce-tdi6.8.sh

## 19. Interpretation boundaries

A TDI-6.8 result characterizes the **rank** transfer of a **linear ridge** model
between the four hand-specified families of TDI-6.5, at widths 3–4, under the
one-step `Noop` kernel, with coefficients frozen and no correction applied. It
does **not** establish:

- anything about **calibration**, which TDI-5.8B, 6.5C, 6.6 and 6.7 have already
  measured and found to fail; a positive ordering result does not soften that;
- that a **usable** predictor follows. Rank transfer without level transfer
  supports ranking candidates within a new domain, not predicting their
  deficits;
- that the effect was found blind — Section 1.2 states it was not;
- anything about **cross-width** rank transfer; TDI-5.8B's rank observation
  remains a pooled, uncriterioned number;
- that Spearman is the right rank statistic; τ-b is reported precisely so this
  can be checked and not assumed;
- generality beyond four synthetic families of small finite-state systems.

If TDI-6.8A returns *Beneficial* and TDI-6.8B returns **transfers**, the correct
conclusion is that within this scope the overlaps carry transportable *ordering*
information beyond the full descriptor ladder, while carrying no transportable
*level* information — a sharper and more limited claim than the campaign's
headline, and the one the evidence supports.

The TDI-6.8A / 6.8B / 6.8C / 6.8D summaries may not be rewritten after observing
the result.

## 20. Freeze rule

This preregistration is frozen at its SHA-256. The evaluator, its reproduction
script and its manifests are frozen with it. The confirmatory run is a single,
deliberate, human-only action under the exact token of Section 18. No commit,
test or CI run may supply that token, and the authoring agent must never invoke
`--full` with it.
