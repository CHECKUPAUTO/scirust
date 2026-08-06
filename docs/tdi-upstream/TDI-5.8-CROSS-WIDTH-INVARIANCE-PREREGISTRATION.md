# TDI-5.8 — Cross-Width Invariance: Does the Overlap Signal Survive Across Kernel Sizes?

## Preregistration

This document is the frozen preregistration for TDI-5.8. Once its SHA-256
manifest, the v58 evaluator, the reproduction script, the CI workflow and the
bounded tests are committed, this design is frozen under the Section 21 freeze
rule: no scientific constant, seed block, feature definition, width, baseline
or criterion may change without a new experiment identifier. Freezing the
design does not authorize a run; the real experiment begins only as the
deliberate one-time human action of Section 16. The authoring agent never
invokes `--full`.

## 1. Experimental status, provenance, and the single changed factor

TDI-5.8 is a new confirmatory experiment derived from the completed and merged
**TDI-5.6** result. It is **not** a continuation, patch, or reinterpretation of
TDI-5.1 … 5.7, TDI-6.1, TDI-6.2 or TDI-6.5, each of which remains frozen under
its own identifier.

Every prior TDI-5.x / TDI-6.x result computes its descriptors on candidate
kernels of **width 3 and width 4 only** (`N = 2^w ∈ {8, 16}` states). Whether
the early overlaps' marginal value beyond the exact descriptor set is a
property of *those two kernel sizes* or of the branching dynamics *at any size*
is the single most-repeated open limitation ("robustness to … cross-width
invariance", named in every recent Section 21). A skeptic can still argue the
overlap signal is an artifact of small kernels (8 or 16 states), where
finite-size effects dominate.

**TDI-5.8 confronts this directly.** The single factor changed from TDI-5.6 is
the **width**: the base composition (widths {3, 4} pooled into one model) is
replaced by a **cross-width design** in which each width is analysed
independently — its own populations, its own fitted model, and its own
SKT-vs-SK classification — across a width set that extends beyond the base. The
confirmatory question of TDI-5.6 (does `{O_1, O_2}` reduce error beyond the
exact contraction descriptors δ, δ̄ **and** the exact spectral moments s₂, s₃?)
is asked **once per width**, and the criteria test whether the answer is
invariant across widths and whether a model transports between widths.

TDI-5.8 stays entirely **bit-exact** — it uses only the inherited exact
descriptors and layouts; it introduces no non-exact quantity (the literal
spectral gap and mixing time are the TDI-6 track, already exercised by 6.1/6.5
and out of scope here).

Frozen ancestor identities (verified at runtime and in CI): the v56 evaluator,
the TDI-5.6 preregistration, and the full frozen chain **TDI-5.1 → TDI-6.5**
(every ancestor evaluator and preregistration hash) are verified before any
generation.

No full TDI-5.8 run may begin before all of the following are committed and
frozen: this preregistration; the final evaluator; the evaluator SHA-256
manifest; the scientific-code SHA-256 manifest; the deterministic reproduction
script; the dedicated CI workflow; bounded unit and termination tests.

## 2. Research questions

Within the frozen candidate machinery and the frozen exact descriptor set:

1. does the SKT-minus-SK marginal value of `{O_1, O_2}` — the overlaps' value
   beyond the exact contraction descriptors **and** the exact spectral moments
   — classify **Beneficial at every width** at the focal horizons U₃ and U₆
   (**cross-width replication**, criterion **TDI-5.8A**, primary)?
2. does a model fitted at the smallest width **transfer** to the largest width
   (**cross-width transfer**, criterion **TDI-5.8B**, descriptive)?
3. how **stable** is the effect size across widths — is the aggregate
   relative-MSE reduction tightly clustered above the 2% margin, or does it
   drift with kernel size (**effect-size stability**, criterion **TDI-5.8C**)?
4. how do the four exact descriptors δ, δ̄, s₂, s₃ **themselves move** with
   width — in particular the moments' range `[0, N]` grows with `N = 2^w`
   (**descriptor drift across widths**, criterion **TDI-5.8D**, descriptive)?

TDI-5.8 does **not** re-open the exact-descriptor questions (5.5/5.6), the
generator-family question (5.7), the literal spectral controls (6.1/6.5), or
nonlinear models (6.2). It changes **only** the width, holding every feature,
layout, model, generator and criterion machine of 5.6 fixed.

## 3. Relationship to the frozen ancestors

**Inherited unchanged** from TDI-5.2 … 5.6 (frozen; still bit-exact; not
re-derived): the exact candidate analysis and per-candidate exclusion criteria;
the single base generator (`build_system` over uniform non-empty successor
masks); observation geometry and target geometry `U_h = -log2(1 - O_h)`; the 13
structural/entropic baseline variables and the overlaps O₁, O₂; the two exact
contraction descriptors δ, δ̄; the two exact spectral moments s₂ = trace(P²),
s₃ = trace(P³); the layouts **CK / SK / SKT** (Section 6); ridge `lambda = 1.0`
with training-only preprocessing and per-scaler target standardization; the
paired + stratified-aggregate bootstrap (4,000 replicates); the four-way ±2%
classifier; the dense grid `H = {3, 4, 5, 6, 7, 8}` and focal horizons U₃, U₆;
and the `tdi-core` exact primitives.

**New in TDI-5.8** (the only substantive change): the **width** becomes the
analysis-grouping dimension. In place of TDI-5.6's single model pooling widths
{3, 4}, TDI-5.8 fits and classifies **one model per width** over a width set
that extends beyond the base (Section 7); fresh, pairwise-disjoint per-(width,
block) seed reservations (Section 9); fresh `TDI5`/`38`-marked bootstrap seeds
(Section 10); the confirmation guard `TDI58_CONFIRM_FULL_RUN`; and the criteria
5.8A / 5.8B / 5.8C / 5.8D (Sections 13–15) applied to the per-width SKT-vs-SK
comparison.

## 4. Design notes and confirmatory integrity

### 4.1 Why per-width, single generator

Holding the entire 5.6 measurement apparatus and its single generator fixed and
varying only the kernel size isolates one factor — the width — so any change in
the SKT-vs-SK classification is attributable to kernel size and nothing else.
Analysing each width independently (its own model, standardization and
classification) is what turns "we ran at more widths" into a genuine invariance
test: it asks whether the *same conclusion* holds at each size, not whether a
size-pooled model happens to work.

### 4.2 The exact descriptors remain exact at every width

The exact contraction descriptors and the exact spectral moments s₂ = trace(P²),
s₃ = trace(P³) are finite sums of unit-fraction products over the `N = 2^w`
states, computed with the inherited arbitrary-precision `ExactRatio` arithmetic
and a single final `f64` rounding — at width `w` the moments are exact rationals
in `[0, N]`. No eigenvalue or floating-point iteration is introduced; TDI-5.8
stays bit-exact at every width, so its reproduction is **byte-exact** (unlike
the tolerance-based TDI-6 track).

### 4.3 Width set and generation feasibility (confirmed by a bounded probe)

The base widths {3, 4} are proven reachable at scale by every prior `--full`
run. TDI-5.8 adds width **5** (`N = 32`), the natural next kernel size, giving a
three-point invariance test across `N ∈ {8, 16, 32}` (a 4× size range). The
inherited generation budget for width 5 (attempt multiplier 128, no-progress
limit 75,000) governs acceptance. A bounded generation probe (a few hundred
accepted records, starting from the frozen width-5 seed blocks) observed
**zero rejections** — acceptance is effectively total under the base
generator's uniform non-empty successor-mask rule at width 5 — so the
preregistered 15,000-record per-block target is comfortably reachable, far
inside the 75,000 no-progress cap and the 128×-multiplier attempt ceiling. The
width set `W = {3, 4, 5}` and the per-width counts of Sections 7–9 are
confirmed and frozen. The dominant real-run cost at width 5 is **wall-clock**,
not acceptance: the exact `ExactRatio` descriptor arithmetic runs
substantially slower per candidate at width 5 than at widths 3–4, so the
width-5 populations are the majority of the run's duration; this is a
scheduling property of the human `--full` action (Section 16), not a
scientific-design constraint. Widths `≥ 6` (`N ≥ 64`) are a **documented
out-of-scope boundary** (generation cost and acceptance at `N ≥ 64` are
deferred to a future identifier), recorded honestly in Section 19.

### 4.4 Feature scale and the transfer question

The overlaps O₁, O₂ and the contraction descriptors δ, δ̄ live in `[0, 1]`
independent of width; the exact moments s₂, s₃ live in `[0, N]` and therefore
grow with kernel size. Within a width, training-only standardization removes
this scale, so the per-width fit (5.8A) is unaffected. **Across** widths,
however, a model fitted at one width carries that width's standardization, so
5.8B (transfer) genuinely tests whether the learned relationship transports
despite the size-dependent moment scale — a scientifically meaningful stress,
reported descriptively.

### 4.5 The width-indicator baseline variable is structurally inert within each per-width fit

The 13 inherited structural/entropic baseline variables (Section 3) include
the candidate's own width as their 13th entry. Every prior TDI-5.x experiment
pools widths 3 and 4 into one population per model, so this variable carries
real within-population variance there. TDI-5.8 instead groups by width
(Section 7): within any single width's own fitted model, this variable is
therefore a **constant** equal to that width's `w`, so its raw variance is
exactly zero and the ridge fit's zero-variance standardization guard assigns
it a fixed fallback scale rather than dividing by zero; the resulting
standardized column is identically `0` for every record, so the closed-form
ridge solve necessarily assigns it coefficient `0` in every layout, at every
horizon, in every block, at every width. This is an **expected, harmless**
structural consequence of the per-width design, not a defect: the variable is
identically inert in both the SK baseline and the SKT challenger of every
TDI-5.8A/B/C confirmatory comparison, so it cannot bias any of them. It
reduces the 13 baseline variables to 12 informative ones per width — recorded
here so the observation is documented rather than a silent side effect of the
relabelling from TDI-5.6's pooled-width design.

## 5. The exact descriptors

Inherited unchanged from TDI-5.5 (δ, δ̄) and TDI-5.6 Section 5 (s₂, s₃). For a
candidate system of width `w`, `P` is the one-step `Noop` kernel on the
`N = 2^w` states (`P(s, ·)` uniform over `s`'s successor set). The Dobrushin
coefficient δ and mean pairwise total variation δ̄ are the exact contraction
descriptors; s₂ = trace(P²) = Σ_{i,j} P_{ij}P_{ji} and s₃ = trace(P³) =
Σ_{i,j,k} P_{ij}P_{jk}P_{ki} are the exact spectral moments (power sums
Σλᵢ², Σλᵢ³). All four are computed per candidate as exact rationals and rounded
once to `f64`, unchanged from the frozen ancestors, at every width in the set.

## 6. Feature layouts

Inherited unchanged from TDI-5.6 Section 6; all layouts linear.

| Layout | Variables | Count | Role |
|---|---|---:|---|
| **CK** | baseline + δ + δ̄ | 15 | contraction baseline |
| **SK** | CK + s₂ + s₃ | 17 | exact spectral baseline (5.8A/B/C baseline) |
| **SKT** | SK + O₁ + O₂ | 19 | full model |

`SKT − SK` isolates the overlaps' marginal value after the exact contraction
descriptors and the exact spectral moments — the confirmatory comparison,
computed **independently within each width**. Ridge `lambda = 1.0`; per width,
block and horizon one target scaler is shared across the three layouts.

## 7. Widths and populations

The analysis-grouping dimension is the **width**, over the set `W = {3, 4, 5}`
(confirmed reachable; Section 4.3). For **each** width `w ∈ W`, generate
in-distribution populations across **three** fresh, pairwise-disjoint seed
blocks:

| Population | Accepted records |
|---|---:|
| training | 15,000 |
| holdout | 5,000 |

Accepted records per (width, block): **20,000**; per width (3 blocks):
**60,000**; total (3 widths): **180,000**. **No OOD populations.** For each
width, the per-width model is fitted on that width's combined three-block
training population and every criterion is evaluated on that width's combined
three-block holdout population. Holdout records never affect fitting. The
per-width generation budgets are inherited unchanged from TDI-5.2 Section 7
(width 3: multiplier 64 / no-progress 25,000; width 4: multiplier 96 /
no-progress 50,000; width 5: multiplier 128 / no-progress 75,000).

## 8. Standardized-U primacy and metrics

Standardized-U space is the primary confirmatory domain (TDI-5.2 Section 5);
reconstructed-O quantities are secondary diagnostics and determine no criterion.
For every width, block, population, horizon and layout, print the full inherited
metric set of TDI-5.2 Section 9, plus, for every confirmatory comparison, the
absolute MSE difference, relative MSE reduction, absolute MAE difference,
Spearman difference, R² difference and absolute-bias difference.

## 9. Independent seed blocks (fresh)

Eighteen deterministic, pairwise-disjoint reservations (3 widths × 3 blocks × 2
populations), **disjoint from every prior block** (TDI-5.7 up to ≈ 2.53×10⁹;
TDI-6.1 3.0–3.23×10⁹; TDI-6.2 4.0–4.23×10⁹; TDI-6.5 5.0–6.13×10⁹; TDI-5.8
starts at 7.0×10⁹). For width index `wi ∈ {0,1,2}` (widths 3/4/5) and block
index `b ∈ {0,1,2}`:

    base(wi, b) = 7_000_000_000 + wi · 300_000_000 + b · 100_000_000

with the two populations at `base + {0, 10} · 1_000_000` (training, holdout).
Explicitly the training bases:

| Width | Block bases (training seed) |
|---|---|
| 3 | 7,000,000,000 · 7,100,000,000 · 7,200,000,000 |
| 4 | 7,300,000,000 · 7,400,000,000 · 7,500,000,000 |
| 5 | 7,600,000,000 · 7,700,000,000 · 7,800,000,000 |

Blocks are spaced 100,000,000 apart and populations 10,000,000 apart; each
population consumes at most a few million seeds, so all 18 reservations are
pairwise disjoint and disjoint from every earlier block. The evaluator verifies
disjointness of all consumed ranges at runtime.

## 10. Deterministic bootstrap

The bootstrap engine, replicate count (**4,000**) and resampling discipline are
inherited unchanged (bit-exact, integer seeds). TDI-5.8 uses fresh bootstrap
seeds in the `0x5444_4935_3800_…` (`TDI5`/`38` = ".8") range, disjoint from
every prior bootstrap seed:

    block seed (width wi, block b)   : 0x5444_4935_3800_0000 + (3·wi + b) + 1
                                       (w3: …0001/0002/0003 … w5: …0007/0008/0009)
    per-width aggregate seed (wi)    : 0x5444_4935_3800_4700 + wi
                                       (w3: …4700 · w4: …4701 · w5: …4702)
    TDI-5.8B transfer aggregate seed : 0x5444_4935_3800_4703

Each width's stratified-aggregate bootstrap runs over its own three blocks with
its own aggregate seed. The TDI-5.8B transfer comparison (Section 14) resamples
the transfer-target width's holdout under the transfer-source width's fitted
model, so it draws from its own dedicated aggregate seed — one past the last
per-width aggregate seed — rather than the transfer-source width's own
per-width seed, keeping the transfer comparison's resampling independent of
that width's own grid-comparison resampling. For each confirmatory comparison,
report the two-sided 95% interval of the baseline-minus-challenger MSE
difference and, for equivalence classification, the two-sided 95% interval of
the relative MSE difference.

## 11. Focal horizons and grid

Inherited: the dense grid `H = {3, 4, 5, 6, 7, 8}` and the focal horizons **U₃**
and **U₆**. The confirmatory criteria classify at the focal horizons; per-width
per-horizon SKT-vs-SK reductions across the grid are reported.

## 12. Determinism

Inherited from TDI-5.2 Section 18. Candidate generation at every width, seed
consumption, exclusions, preprocessing, exact contraction-descriptor and
spectral-moment construction, model fitting, bootstrap sampling, aggregation,
metric calculation, iteration order, scientific-value formatting and final
criteria are deterministic functions of committed constants; reproduction is
byte-exact on any conforming toolchain. Wall-clock timestamps are reproduction
metadata only.

## 13. Criterion TDI-5.8A — cross-width replication (primary)

For **each** width `w ∈ W`, compute the **SKT-vs-SK** four-way classification
(exact 4-way logic of TDI-5.2 Section 13, symmetric 2% relative-MSE margin) at
the focal horizons **U₃** and **U₆** on that width's combined holdout. TDI-5.8A
is the preregistered conjunction:

- **replicated** iff the classification is **Beneficial at both U₃ and U₆ for
  every width in `W`**;
- otherwise a **located non-replication** — the evaluator names each (width,
  horizon) whose classification is not Beneficial.

TDI-5.8A is a preregistered classification, forced to no result. Full
replication would show the overlaps' value beyond the exact descriptors is a
property of the branching dynamics across kernel sizes, not of the base widths;
a located non-replication would bound TDI's generality to the widths where it
holds.

## 14. Criterion TDI-5.8B — cross-width transfer (descriptive)

Fit the SK and SKT models on the **smallest** width's (`w = 3`) combined
training population; evaluate the SKT-vs-SK comparison on the **largest**
width's (`w = 5`) combined holdout, standardizing the width-5 features with the
width-3 training statistics (so the transfer is genuine — Section 4.4). Report
the standardized-U R² of each layout and the four-way classification at U₃ and
U₆. This distinguishes "the signal exists within each width" (5.8A) from "the
*same fitted* model transports across kernel sizes despite the size-dependent
moment scale." Descriptive; makes no pass/fail claim.

## 15. Criteria TDI-5.8C and TDI-5.8D — stability and descriptor drift (descriptive)

**TDI-5.8C — effect-size stability.** Across the widths in `W`, report at each
focal horizon the aggregate relative-MSE reduction of SKT over SK: its
**minimum, maximum and range**, and whether **all** widths exceed the 2%
margin. A tight cluster well above 2% is strong cross-width generality of effect
size; a systematic drift with kernel size (even if all Beneficial) is a
preregistered caveat.

**TDI-5.8D — descriptor drift across widths.** For each width, report the
holdout means of the four exact descriptors δ, δ̄, s₂, s₃ (and their
across-width range). Because s₂, s₃ ∈ `[0, N]` with `N = 2^w`, their raw means
are expected to grow with width; the table documents this size scaling (immaterial
to the per-width fit after standardization, but central to the 5.8B transfer
stress) and shows whether the contraction descriptors δ, δ̄ ∈ `[0, 1]` also
drift with size.

Both TDI-5.8C and TDI-5.8D are preregistered **descriptive** summaries; neither
makes a success/failure claim.

## 16. Operational activation and full-run entrypoint contract

The v58 evaluator exposes exactly three modes: `--termination-smoke`,
`--preflight`, `--full`. A bare invocation refuses to run. `--termination-smoke`
uses only bounded tiny data and produces no result artifacts. `--preflight`
performs no scientific generation: it verifies the full frozen configuration
(all 18 seed reservations, all expected counts, all bootstrap constants, the
width set and per-width budgets), verifies that the full pipeline is wired to
`--full`, prints all TDI-5.8 and ancestor identities and the exact real-run
command, and exits without a result.

`--full` requires the exact confirmation environment variable:

    TDI58_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI58_FREEZE_RULE

Without that exact value, `--full` fails before any generation, fitting or
bootstrap. The confirmation check is a pure function of the environment value,
unit-testable without starting the experiment. No TDI-5.8 commit, test or CI run
supplies the token. The full run is a deliberate, one-time human action; the
authoring agent never invokes `--full` with the real token.

## 17. Required raw output

Inherited from TDI-5.2 Section 17 with TDI-5.8 identities and **per-width**
reporting: git commit; compiler/Cargo versions; the v58 evaluator SHA-256; the
TDI-5.8 preregistration and scientific-manifest SHA-256; the full frozen
ancestor chain (TDI-5.1 → 6.5); all frozen constants; the width set and
per-width generation budgets; the seed-block definitions; per-width
requested/accepted/rejected/attempted counts; rejection counts by reason; final
exclusive seeds; target scalers; the CK, SK and SKT model coefficients for every
width and block; all metrics; all bootstrap intervals; the per-width per-horizon
SKT-vs-SK comparisons across the grid U₃…U₈; the TDI-5.8A focal classifications
per width and the replication verdict; the TDI-5.8B transfer classification; the
TDI-5.8C stability summary; the TDI-5.8D descriptor-drift table; deterministic
termination diagnostics.

## 18. Reproduction requirements

The TDI-5.8 reproduction script must satisfy every requirement of TDI-5.2
Section 19 / TDI-5.6 Section 19 (refuse a dirty repository; verify all frozen
hashes including TDI-5.1 … 5.7, 6.1, 6.2, 6.5 and 5.8; refuse an existing
partial or complete result; acquire an exclusive lock; compile offline in
release mode; execute the evaluator exactly once with `--full`; capture complete
output; verify all final criterion lines; write metadata and a completion
marker; hash all artifacts; make final artifacts read-only), plus: it must
require the exact confirmation variable before invoking the evaluator, and must
refuse to run over an existing TDI-5.8 result. Reproduction is **byte-exact**
(the completion check verifies the result-log SHA-256 and the presence of the
5.8A/5.8B/5.8C/5.8D lines).

## 19. Interpretation boundaries

A TDI-5.8 result establishes the (non)replication and effect-size stability of
the `{O_1,O_2}`-beyond-exact-descriptors signal **across the specific
preregistered width set `W`**, on the single base generator, within the frozen
exact machinery. It does **not** establish: invariance at widths outside `W`
(in particular `N ≥ 64`, deferred; Section 4.3); robustness across the TDI-5.7
generator families (settled separately, on widths {3,4}); control against the
literal spectral gap / mixing time (6.1/6.5) or nonlinear models (6.2); causal
effects; or external validity. The TDI-5.8A / B / C / D summaries may not be
rewritten after observing the result.

## 20. Deferred tracks

The literal spectral controls and nonlinear/PID/causal questions are the TDI-6
track (6.1/6.2/6.5 merged; 6.3/6.4 reserved) and are out of scope for TDI-5.8,
which stays bit-exact and single-generator. Widths `N ≥ 64` and generator-family
× width interactions are documented boundaries, deferred to future identifiers.

## 21. Freeze rule

Once the SHA-256 manifests, the v58 evaluator, the reproduction script, the CI
workflow and the bounded tests are committed, this design is frozen: scientific
code must not change; constants must not change; the width set `W`, the seed
blocks, the layouts and the criteria must not change; no full run may begin
before all frozen hashes pass (TDI-5.1 … 5.7, 6.1, 6.2, 6.5 and 5.8); any
scientific-code defect discovered after freezing requires a new experiment
identifier — TDI-5.8 may not be silently patched. The result classifications,
once produced, are frozen as reported.
