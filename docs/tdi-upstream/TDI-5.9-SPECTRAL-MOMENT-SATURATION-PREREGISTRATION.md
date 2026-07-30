# TDI-5.9 — Higher Exact Moments: The Fourth Spectral Moment and Descriptor Saturation

## Preregistration

This document is the frozen preregistration for TDI-5.9. Once its SHA-256
manifest, the v59 evaluator, the reproduction script, the CI workflow and
the bounded tests are committed, this design is frozen under the Section 23
freeze rule: no scientific constant, seed block, feature definition,
baseline or criterion may change without a new experiment identifier.
Freezing the design does not authorize a run; the real experiment may begin
only as the deliberate one-time human action described in Section 17.

## 1. Experimental status and provenance

TDI-5.9 is a new confirmatory experiment derived from the completed and
merged TDI-5.6 result. It is **not** a continuation, patch, or
reinterpretation of TDI-5.1 through TDI-5.8, TDI-6.1, TDI-6.2, TDI-6.3,
TDI-6.4 or TDI-6.5, each of which remains frozen under its own identifier.
Its single scientific ancestor is TDI-5.6; the other ten experiments are
verified only as members of the frozen ancestor chain (Section 3), exactly
as TDI-6.4 treated TDI-6.1/6.2/6.3/6.5 (its preregistration Section 1).

TDI-5.6 established that `{O_1, O_2}` carry predictive information beyond
the exact Dobrushin contraction descriptors (δ, δ̄) **and** the exact
spectral moments `s_2 = trace(P^2)`, `s_3 = trace(P^3)` of the one-step
kernel — the strongest confound the bit-exact program could express at the
time. TDI-5.6 Section 4.2 explicitly considered and deferred the question
this experiment now takes up: does the exact-spectral confound keep growing
stronger as higher moments are added, or does its marginal value per moment
shrink? TDI-5.9 answers this by adding the **fourth spectral moment**
`s_4 = trace(P^4)` to the confound set and measuring, **on the same fresh
data**, whether its own marginal contribution is smaller than the combined
contribution `{s_2, s_3}` already made in TDI-5.6.

### 1.1 Why now, and why last

The forward-program roadmap (`docs/TDI-FORWARD-PROGRAM-ROADMAP.md` Section
2, Track A) named three exact-track continuations: TDI-5.7 (generator
robustness), TDI-5.8 (cross-width invariance) and TDI-5.9 (higher exact
moments / descriptor saturation), describing 5.9 explicitly as "the
exact-side ceiling of the 'is TDI redundant?' question." TDI-5.7 and 5.8 are
now built and merged; TDI-5.9 is the one Track-A roadmap slot that had not
yet been started when this document was drafted (verified by the absence of
any `v59` evaluator, any `TDI-5.9` document, or any matching commit anywhere
in the repository's history before this one). It does not depend on the
result of TDI-5.7, TDI-5.8, or any TDI-6.x experiment — its only scientific
prerequisite is TDI-5.6, frozen since before TDI-5.7 began — and building it
now, after the TDI-6.4 causal probe, is a scheduling choice (the last
unclaimed roadmap slot), not a scientific dependency. Because the roadmap
frames 5.9 as the exact track's *ceiling* test, and no further Track-A
experiment is named beyond it, TDI-5.9 is expected to be the final new build
of the current forward-program roadmap; anything past it would need a new
roadmap revision informed by results not yet in hand (mirroring exactly how
TDI-6.5 itself was added only after TDI-6.1's real result, per that
preregistration's own Section 1).

### 1.2 Single changed factor: the fourth moment, not a new kind of feature

The roadmap's one-line description reads "Add `s_4 = trace(P^4)` (and the
exact return-probability profile) to SK." Read literally, "the exact
return-probability profile" could suggest a separate vector-valued feature
distinct from `s_4` itself. It is not: `s_k = trace(P^k) = sum_i (P^k)_{ii}`
**is**, term for term, the sum of the exact `k`-step return probabilities
`(P^k)_{ii}` over every state `i` (TDI-5.6 Section 5 already establishes
this reading for `s_2, s_3`). The "return-probability profile" is the
sequence `(s_2, s_3, s_4, ...)` itself; `s_4` is simply its next entry, and
introducing any further quantity beyond `s_4` in this experiment would add a
second changed factor and break the single-changed-factor discipline every
TDI-5.x derivation has followed since TDI-5.2. TDI-5.9 therefore adds
**exactly one** new descriptor, `s_4`, extending the profile `{s_2, s_3}`
established by TDI-5.6 to `{s_2, s_3, s_4}` — nothing else about the
candidate machinery, the observation geometry, the ridge model, the
bootstrap engine or the classifier changes.

Frozen ancestor identities (to be verified at runtime and in CI):

| Artifact | SHA-256 |
|---|---|
| TDI-5.6 evaluator (v56) | `0820274b3edb58a6e123c612dbed8dd8a1725221240365f142d9510404e1d1b2` |
| TDI-5.6 preregistration | `59e3375b82d0bb7aad7be0591b9d1eac074d4b194678dfe0e06e73c8aac89807` |

The v59 evaluator and the CI workflow additionally verify the **full frozen
chain** — every prior TDI-5.x and TDI-6.x evaluator and preregistration hash
(Section 3) — before any generation.

No full TDI-5.9 run may begin before all of the following are committed and
frozen: this preregistration; the final evaluator; the evaluator SHA-256
manifest; the scientific-code SHA-256 manifest; the deterministic
reproduction script; the dedicated CI workflow; bounded unit and termination
tests.

## 2. Research questions

TDI-5.9 evaluates, within the frozen candidate machinery:

1. whether `{O_1, O_2}` contribute predictive information about `U_h`
   **beyond a structural/entropic baseline augmented with the exact
   Dobrushin descriptors *and* the exact spectral moments `s_2, s_3, s_4`**
   (the **richer spectral confound**, criterion TDI-5.9A), at the focal
   horizons U₃ and U₆ — the same question TDI-5.6A asked, now under a
   strictly richer exact control;
2. whether the **fourth spectral moment alone** contributes predictive
   information beyond `{δ, δ̄, s_2, s_3}` (the **marginal fourth-moment
   value**, criterion TDI-5.9B), at U₃ and U₆;
3. what **functional form** the overlaps' marginal value beyond the full
   `{δ, δ̄, s_2, s_3, s_4}` descriptor set follows across the dense horizon
   grid U₃…U₈ (the **decay law**, criterion TDI-5.9C, descriptive);
4. whether the **per-moment marginal value is shrinking**: is the fourth
   moment's own marginal contribution (question 2) smaller than the
   combined second-and-third-moment contribution already measured, **on the
   same fresh data**, by the intermediate SK-vs-CK comparison this
   experiment recomputes (the **descriptor saturation** question, criterion
   TDI-5.9D, descriptive) — the direct test of the roadmap's "exact-side
   ceiling" framing;
5. whether all conclusions replicate across three independent seed blocks
   Y/Z/AA.

TDI-5.9 does **not** re-test the joint signal, independent activation, OOD
transfer, nonlinear-basis findings, the persistence confound, the
second-and-third-moment confound in isolation (TDI-5.6A/B/C, settled under
their own identifier), generator robustness (5.7), cross-width invariance
(5.8), or any TDI-6.x question (literal spectral gap, nonlinear model
families, information decomposition, causal effects, generator-family
robustness); those are settled, deferred, or out of scope under their own
identifiers. It does **not** use non-exact spectral descriptors (the literal
eigenvalues, the spectral gap, or the ε-threshold mixing time) or
non-parametric model families; those remain out of scope for the exact
track (Section 22).

## 3. Relationship to the frozen ancestors

**Inherited unchanged** from TDI-5.2 through TDI-5.6 (frozen; not re-derived
here):

- the entire dynamical construction, exact candidate analysis, observation
  geometry, target geometry (`U_h = -log2(1 - O_h)`), width-6 exact
  cardinality, and the preregistered per-candidate exclusion criteria
  (TDI-5.2 Sections 3, 8);
- observation horizon `h_obs = 2`; primary target `U_6`;
- the 13 structural/entropic baseline variables and the two early-overlap
  predictors O₁, O₂ (TDI-5.2 Section 4);
- the two exact contraction descriptors δ, δ̄ (TDI-5.5 Section 5);
- the two exact spectral moments `s_2 = trace(P^2)`, `s_3 = trace(P^3)`
  (TDI-5.6 Section 5), computed identically, unchanged;
- ridge regression with `lambda = 1.0`, training-only preprocessing and
  target standardization, deterministic accumulation order, closed-form
  linear-in-parameters solution (TDI-5.2 Section 5);
- the deterministic width-3 and width-4 generation budgets (TDI-5.2
  Section 7);
- the dense target-horizon grid `H = {3, 4, 5, 6, 7, 8}` and the focal
  horizons U₃, U₆ (TDI-5.5 Section 3);
- the paired + stratified-aggregate bootstrap engine and its resampling
  discipline (TDI-5.2 Section 10);
- the 4-way Beneficial / Equivalent / Harmful / Inconclusive classification
  logic and the symmetric 2% relative-MSE margin (TDI-5.2 Section 13);
- the exact overlap / total-variation primitives of `tdi-core`
  (`uniform_branching_state_distribution`, `distribution_overlap`,
  `ExactRatio`), used **unchanged** to compute the new fourth moment —
  `tdi-core` itself is not modified.

**New in TDI-5.9** (the only substantive scientific additions):

- one **exact fourth spectral moment** `s_4 = trace(P^4)` of the one-step
  kernel, computed per candidate system from the inherited exact primitives
  (Section 5), and two new layouts **SK4** and **SKT4** (Section 6);
- **fresh, independent seed blocks Y/Z/AA** (Section 9), disjoint from every
  prior block, with fresh bootstrap seeds (Section 10);
- four criteria: **TDI-5.9A** (signal beyond the richer spectral profile),
  **TDI-5.9B** (marginal fourth-moment value), **TDI-5.9C** (decay law) and
  **TDI-5.9D** (descriptor saturation), Sections 13–16.

**Dropped relative to TDI-5.6.** None: TDI-5.9 keeps every TDI-5.6 design
choice (no persistence competitor, no OOD populations, linear-only layouts)
unchanged; it strictly adds one descriptor, two layouts, and one criterion.

## 4. Design notes and confirmatory integrity

### 4.1 Why the fourth moment stays exact

`trace(P^4)` is, like `trace(P^2)` and `trace(P^3)`, a finite sum of
products of the kernel's rational entries (Section 5): a closed 4-walk
`i -> j -> k -> l -> i` contributes the unit fraction `1/(d_i d_j d_k d_l)`,
accumulated with the same arbitrary-precision `ExactRatio` addition used for
every other exact quantity in the program, with a single final rounding to
`f64`. No eigenvalue, characteristic polynomial, or floating-point iteration
is involved. TDI-5.9 therefore stays entirely within the bit-exact,
closed-form, deterministic invariant that defines the TDI-5.x program
(TDI-5.6 Section 4.1); it does not touch the deferred non-exact track.

### 4.2 Why the fourth moment, and why it is (for now) the last one

TDI-5.6 Section 4.2 omitted the first moment (`trace(P) = sum_i P_{ii}`,
redundant with the baseline's local branching information) and stopped at
the third, noting the pair `{s_2, s_3}` already gives the baseline "a
symmetric and an asymmetric summary of the subdominant spectrum." The fourth
moment `s_4 = sum_i lambda_i^4` is the next even power sum: it is dominated
by the same leading subdominant eigenvalues `s_2` already sees, so it is
expected, if anything, to be **more** correlated with `{δ, δ̄, s_2, s_3}`
than `s_3` was with `{δ, δ̄}` — making TDI-5.9B a genuine test of whether the
exact descriptor set has begun to saturate, not merely a mechanical repeat
of TDI-5.6B. Moments of order five and above are explicitly out of scope
(Section 22): each additional even or odd moment adds combinatorially more
closed-walk terms for a diminishing return that this experiment's own
Section 16 (TDI-5.9D) is designed to measure honestly, rather than assumed.
Should TDI-5.9D find the fourth moment's marginal value has *not* shrunk,
that would itself be informative and would motivate a further experiment
under a new identifier — TDI-5.9 does not preclude one.

### 4.3 Range and standardization

For an `N`-state kernel (`N = 2^w`), `s_4` satisfies `0 <= s_4 <= N` by the
same argument as `s_2, s_3` (TDI-5.6 Section 4.3): the diagonal of a
non-negative matrix power is non-negative, and every eigenvalue of a
row-stochastic matrix satisfies `|lambda_i| <= 1`, so
`s_4 = sum_i lambda_i^4 <= sum_i |lambda_i|^4 <= N`. It is standardized
downstream with training-only statistics exactly like every other feature.
A non-finite moment (impossible for the exact computation, but guarded)
triggers the same graceful per-candidate exclusion as any non-finite
feature.

### 4.4 The saturation comparison is computed fresh, not against TDI-5.6's frozen numbers

TDI-5.9D (Section 16) compares the marginal value of `s_4` alone against the
combined marginal value of `{s_2, s_3}`. That combined value was already
measured once, as TDI-5.6B, on the TDI-5.6 seed blocks J/K/L. TDI-5.9 does
**not** reuse TDI-5.6B's frozen point estimate for this comparison: doing so
would confound the saturation question with ordinary sampling variation
across two different populations drawn from two different seed blocks.
Instead, TDI-5.9 **recomputes** the `SK`-vs-`CK` comparison (`{s_2, s_3}`'s
combined marginal value) on its own fresh Y/Z/AA populations, immediately
alongside the new `SK4`-vs-`SK` comparison (`s_4`'s marginal value alone),
so both halves of the saturation comparison are measured on identical data,
under identical seed blocks, with identical bootstrap resampling. This is a
deliberate internal replicate of TDI-5.6B's design (not a new criterion in
its own right — its layouts and comparison are unchanged from TDI-5.6
Sections 6 and 14) that exists solely to give TDI-5.9D a controlled
baseline; it is reported in Section 16 alongside TDI-5.9D and is not itself
pass/fail.

### 4.5 Single generator; independence from observed data

TDI-5.9 uses a **single** generator — the base width-3 + width-4
in-distribution composition inherited from TDI-5.4/5.5/5.6 (generator
robustness is a separate, already-settled question, TDI-5.7). It uses
**fresh seed blocks Y/Z/AA**, disjoint from every earlier block, so every
confirmatory quantity is produced from data never used in any observed
result.

## 5. Exact spectral moments

For a candidate system of width `w`, let `P` be the one-step `Noop` kernel on
the `N = 2^w` states, exactly as defined in TDI-5.6 Section 5:
`P_{s,t} = 1/d_s` when `t` is one of the `d_s` distinct `Noop` successors of
`s`, and `0` otherwise. `P` is row-stochastic and total. The second and
third moments are inherited unchanged:

    s_2 = trace(P^2) = sum over ordered pairs (i, j) with j a successor of i
          and i a successor of j, of 1/(d_i d_j);

    s_3 = trace(P^3) = sum over ordered triples (i, j, k) with j a successor
          of i, k a successor of j and i a successor of k, of
          1/(d_i d_j d_k).

TDI-5.9 adds the fourth moment, the trace of the fourth kernel power, as a
sum over closed 4-walks:

    s_4 = trace(P^4) = sum_{i,j,k,l} P_ij P_jk P_kl P_li
        = sum over ordered quadruples (i, j, k, l) with j a successor of i,
          k a successor of j, l a successor of k, and i a successor of l, of
          1/(d_i d_j d_k d_l).

Each summand is a unit fraction whose denominator is a product of at most
four branching factors, each `<= 2^width <= 16`, so the denominator is at
most `16^4 = 65536` and fits in `u128` alongside `s_2` and `s_3`'s
denominators. The summands are accumulated into a running exact rational
with the inherited arbitrary-precision `ExactRatio` addition
(`checked_add`), and only the final total is converted to `f64` with a
single `as_f64()` rounding — the same exactness discipline used for δ, δ̄,
`s_2`, `s_3`, O₁ and O₂. No eigenvalue, characteristic polynomial, or
floating-point iteration is involved; the result is a deterministic exact
rational in `[0, N]`. All three moments are computed per candidate system
during generation (extending the existing `spectral_moments` computation
from two closed-walk lengths to three within the same function), stored on
the record alongside δ and δ̄, and standardized downstream like every other
feature.

## 6. Feature layouts

The 13 baseline variables stay linear and unchanged. All TDI-5.9 layouts are
linear in their features:

| Layout | Variables | Count | Role |
|---|---|---:|---|
| **B0** (C₀) | 13 baseline | 13 | baseline; exploratory reference |
| **B1** | baseline + O₁ | 14 | exploratory (inherited) |
| **B2** | baseline + O₂ | 14 | exploratory (inherited) |
| **B12** | baseline + O₁ + O₂ | 15 | exploratory: TDI beyond C₀ |
| **BD** | baseline + (O₂ − O₁) | 14 | exploratory (inherited) |
| **CK** | baseline + δ + δ̄ | 15 | contraction baseline (recomputed fresh; TDI-5.9D control) |
| **SK** | baseline + δ + δ̄ + s₂ + s₃ | 17 | second/third-moment baseline (recomputed fresh; TDI-5.9B/D control) |
| **SK4** | baseline + δ + δ̄ + s₂ + s₃ + s₄ | 18 | **full exact spectral baseline** (confirmatory) |
| **SKT4** | baseline + δ + δ̄ + s₂ + s₃ + s₄ + O₁ + O₂ | 20 | **full model** (confirmatory) |

`SKT4` minus `SK4` isolates the marginal contribution of `{O_1, O_2}`
**after** the contraction descriptors and all three exact spectral moments
are already present (criteria 5.9A, 5.9C). `SK4` minus `SK` isolates the
marginal contribution of `s_4` alone **after** `{δ, δ̄, s_2, s_3}` are
present (criterion 5.9B). `SK` minus `CK`, recomputed fresh on the TDI-5.9
populations, isolates the combined marginal contribution of `{s_2, s_3}`
under the same data as the 5.9B comparison, enabling the controlled
saturation comparison of criterion 5.9D (Section 4.4, Section 16). Ridge
`lambda = 1.0` is unchanged; all layouts for a block and horizon share one
target scaler.

## 7. No persistence competitor

Unchanged from TDI-5.6: TDI-5.9 introduces **no fixed non-fitted
competitor**. The persistence confound is settled by TDI-5.5B. Every
TDI-5.9 criterion compares two fitted ridge layouts through the identical
paired / stratified-aggregate bootstrap and four-way classifier.

## 8. Populations

TDI-5.9 uses a single generator and generates only in-distribution
populations. **No OOD populations are generated.** For each of the three
seed blocks:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |

Accepted records per block: **40,000**. Total: **120,000**. Models are
fitted on each block's combined width-3 + width-4 training population; every
criterion is evaluated on that block's combined width-3 + width-4 holdout
population. Holdout records never affect fitting.

## 9. Independent seed blocks (fresh)

Three deterministic, pairwise-disjoint seed blocks, **disjoint from every
prior block** (TDI-6.4, the most recently allocated block, consumes seeds up
to ≈ 9.23×10⁹; TDI-5.9 starts at 1.0×10¹⁰). The single-capital-letter
block-naming sequence begun at TDI-5.2 (A, B, C) has, by TDI-6.4, used every
letter through X; TDI-5.9's first two blocks take the sequence's last two
letters, **Y** and **Z**, and its third continues with the spreadsheet-style
extension **AA** rather than reusing any earlier letter.

### Block Y

| Population | Initial seed |
|---|---:|
| training w3 | 10,000,000,000 |
| holdout w3 | 10,010,000,000 |
| training w4 | 10,020,000,000 |
| holdout w4 | 10,030,000,000 |

### Block Z

| Population | Initial seed |
|---|---:|
| training w3 | 10,100,000,000 |
| holdout w3 | 10,110,000,000 |
| training w4 | 10,120,000,000 |
| holdout w4 | 10,130,000,000 |

### Block AA

| Population | Initial seed |
|---|---:|
| training w3 | 10,200,000,000 |
| holdout w3 | 10,210,000,000 |
| training w4 | 10,220,000,000 |
| holdout w4 | 10,230,000,000 |

Generation budgets are inherited unchanged from TDI-5.2 Section 7 (width-3
multiplier 64 / no-progress 25,000; width-4 multiplier 96 / no-progress
50,000). Each population consumes at most a few million seeds; the initial
seeds are spaced 10,000,000 apart within a block and 100,000,000 apart
across blocks, so the reservations are pairwise disjoint and disjoint from
every earlier block. Total seed reservations: **12**.

## 10. Deterministic bootstrap

The bootstrap engine, replicate count (4,000) and resampling discipline are
inherited unchanged from TDI-5.2 Section 10. TDI-5.9 uses fresh bootstrap
seeds, disjoint from every earlier bootstrap seed, following the `TDI5`
ASCII-prefix scheme established from TDI-5.7 onward (`0x5444_4935` = `"TDI5"`,
followed by the ASCII digit of the sub-experiment, `0x39` = `"9"`):

    block Y / Z / AA      : 0x5444493539000001 / …000002 / …000003
    stratified aggregate  : 0x5444493539004700

For each confirmatory comparison, report the two-sided 95% interval of the
baseline-minus-challenger MSE difference and, for equivalence
classification, the two-sided 95% interval of the relative MSE difference.

## 11. Metrics

For every block, population, horizon and layout, print the full metric set
of TDI-5.2 Section 9, plus, for every confirmatory comparison, the absolute
MSE difference, relative MSE reduction, absolute MAE difference, Spearman
difference, R² difference and absolute-bias difference.

## 12. Standardized-U primacy

Standardized U space is the primary confirmatory domain (TDI-5.2 Section 5).
Reconstructed-O-space quantities are secondary diagnostics only and cannot
determine any TDI-5.9 criterion.

## 13. Criterion TDI-5.9A — signal beyond the full exact spectral profile

Compare **SKT4 against SK4** on combined width-3 + width-4 holdout at the
focal horizons **U₃** and **U₆**, using the symmetric relative-MSE margin of
2 percent and the exact 4-way classification logic of TDI-5.2 Section 13.
This yields one 4-way classification at U₃ and one at U₆.

TDI-5.9A is the **primary** preregistered classification, not forced to any
result. *Beneficial* would show `{O_1, O_2}` carry predictive information
none of the exact contraction descriptors nor any of the three exact
spectral moments express — the strongest exact-track evidence yet for a
genuinely independent informational dimension; *Equivalent* would show the
overlaps are, within the exact scope, fully redundant with the richer
spectral structure.

## 14. Criterion TDI-5.9B — marginal fourth-moment value

Compare **SK4 against SK** on the same holdout at the focal horizons **U₃**
and **U₆**, using the same margin and four-way classifier. This yields one
4-way classification at each focal horizon.

TDI-5.9B is a preregistered classification, not forced to any result. It
measures whether the fourth exact spectral moment adds predictive value
**beyond `{δ, δ̄, s_2, s_3}`** in this system, calibrating the strength of
the control applied in TDI-5.9A exactly as TDI-5.6B calibrated TDI-5.6A.
Either outcome is reported honestly and neither weakens TDI-5.9A, which is
classified on its own terms.

## 15. Criterion TDI-5.9C — decay law and redundancy horizon

Evaluate the **SKT4-vs-SK4** comparison at **every** horizon of the dense
grid `H = {3, 4, 5, 6, 7, 8}`, and characterize the overlaps' marginal value
beyond the full exact descriptor set across horizons, exactly as TDI-5.6C.
For each horizon `h`, the marginal value is the aggregate relative-MSE
reduction of SKT4 over SK4 in standardized-U space; its 4-way classification
is as in Section 13.

TDI-5.9C reports, as its preregistered descriptive summary:

1. the six aggregate relative-MSE reductions, one per horizon;
2. the six 4-way classifications, one per horizon;
3. `monotone_non_increasing` — whether the six reductions are non-increasing
   in horizon;
4. `first_equivalent_horizon` — the redundancy horizon `h★`, the smallest
   horizon whose classification is Equivalent, or none;
5. `successive_ratios` — the five ratios `r_{h+1} / r_h` of the reductions.

TDI-5.9C makes **no** success/failure claim: it is a preregistered
descriptive criterion.

## 16. Criterion TDI-5.9D — descriptor saturation

At each focal horizon (**U₃**, **U₆**), report side by side, both computed
on the identical TDI-5.9 holdout populations under the identical bootstrap
resampling (Section 4.4):

1. the aggregate relative-MSE reduction of **SK over CK** — the combined
   marginal value of `{s_2, s_3}` (a fresh replicate of TDI-5.6B's
   comparison, not itself a new pass/fail criterion);
2. the aggregate relative-MSE reduction of **SK4 over SK** — the marginal
   value of `s_4` alone (TDI-5.9B's own point estimate);
3. `saturating` — the preregistered boolean `(2) < (1)`: whether the single
   fourth moment's marginal reduction is smaller than the combined
   second-and-third-moment reduction;
4. whether the two reductions' 95% bootstrap intervals overlap, so
   `saturating` is never read as a confident inequality when the two
   intervals are indistinguishable from noise.

TDI-5.9D makes **no** success/failure claim and does not gate any other
criterion. It is an explicitly asymmetric, non-matched comparison (a
two-feature increment against a one-feature increment) and is reported and
interpreted as such — a genuinely informative descriptive answer to the
roadmap's "exact-side ceiling" question either way: `saturating = true`
would show each additional exact spectral moment is worth less than the
last, consistent with an approaching ceiling; `saturating = false` would
show the fourth moment is at least as informative as the second and third
combined, undercutting the ceiling framing and motivating a further exact
moment under a new identifier (Section 4.2).

## 17. Operational activation and full-run entrypoint contract

The v59 evaluator exposes exactly three modes:

    --termination-smoke
    --preflight
    --full

A bare, no-argument invocation must refuse to run. `--termination-smoke`
uses only bounded tiny data and produces no result artifacts. `--preflight`
performs no scientific generation: it verifies the full frozen configuration
(all 12 seed reservations, all expected counts, all bootstrap constants),
verifies that the full pipeline is wired to `--full`, prints all TDI-5.9 and
ancestor identities and the exact real-run command, and exits without a
result.

`--full` requires the exact confirmation environment variable:

    TDI59_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI59_FREEZE_RULE

Without that exact value, `--full` must fail before any generation, fitting
or bootstrap. The confirmation check is a pure function of the environment
value, unit-testable without starting the experiment. No TDI-5.9 commit,
test, or CI run may supply the token. The full run is a deliberate, one-time
human action; the authoring agent must never invoke `--full` with the real
token.

## 18. Required raw output

Inherited from TDI-5.2 Section 17 with TDI-5.9 identities: git commit;
compiler/Cargo versions; v59 evaluator SHA-256; TDI-5.9 preregistration
SHA-256; TDI-5.9 scientific-manifest SHA-256; the frozen-ancestor hashes; all
frozen constants; the seed-block definitions; requested/accepted/rejected/
attempted counts; rejection counts by reason; final exclusive seeds;
generation budgets; target scalers; **CK, SK, SK4 and SKT4 model
coefficients** (including the contraction-descriptor and all three
spectral-moment coefficients) for every block; all metrics; all bootstrap
intervals; the per-horizon SKT4-vs-SK4 and the focal SK4-vs-SK and
SK-vs-CK comparisons; the TDI-5.9A and TDI-5.9B focal classifications; the
TDI-5.9C decay-law summary; the TDI-5.9D saturation summary; deterministic
termination diagnostics.

## 19. Determinism

Inherited from TDI-5.2 Section 18. Candidate generation, seed consumption,
exclusions, preprocessing, contraction-descriptor and all-three-spectral-
moment construction, model fitting, bootstrap sampling, aggregation, metric
calculation, iteration order, scientific-value formatting and final criteria
are deterministic functions of committed constants. Wall-clock timestamps
are reproduction metadata only.

## 20. Reproduction requirements

The TDI-5.9 reproduction script must satisfy every requirement of TDI-5.2
Section 19 / TDI-5.6 Section 19 (refuse a dirty repository; verify all
frozen hashes including the full ancestor chain through TDI-6.4 and TDI-5.9
itself; refuse an existing partial or complete result; acquire an exclusive
lock; compile offline in release mode; execute the evaluator exactly once
with `--full`; capture complete output; verify all final criterion lines;
write metadata and a completion marker; hash all artifacts; make final
artifacts read-only), plus: it must require the exact confirmation variable
before invoking the evaluator, and must refuse to run over an existing
TDI-5.9 result.

## 21. Interpretation boundaries

A TDI-5.9 result establishes the (non)contribution of `{O_1, O_2}` beyond
the exact Dobrushin contraction descriptors and the exact spectral moments
`s_2, s_3, s_4` of Section 5, and the marginal value of `s_4` and its
saturation profile relative to `{s_2, s_3}`, within the frozen candidate
machinery and the single base generator, replicated across three seed
blocks. It does **not** establish: control against the literal second
eigenvalue / spectral gap or the mixing time (TDI-6.1); control against
spectral moments of order five or above (Section 4.2); sufficiency under
nonlinear or non-parametric model families (TDI-6.2); a formal information
decomposition (TDI-6.3); causal effects (TDI-6.4); robustness to generator
changes (TDI-5.7, TDI-6.5) or to width beyond 3–4 (TDI-5.8); universal
validity across dynamical systems; or external empirical validity. The
TDI-5.9A, TDI-5.9B, TDI-5.9C and TDI-5.9D summaries may not be rewritten
after observing the full result.

## 22. Out of scope

Spectral moments of order five and above (Section 4.2); the literal
eigenvalues, spectral gap and mixing time (TDI-6.1, non-exact); nonlinear or
non-parametric model families (TDI-6.2); a formal information decomposition
of `{O_1, O_2}` (TDI-6.3); causal/interventional effects (TDI-6.4); generator
robustness, both single-family (TDI-5.7) and cross-family under the
non-exact discipline (TDI-6.5); cross-width invariance (TDI-5.8). None of
these is presupposed, constrained, or authorized by TDI-5.9.

## 23. Freeze rule

After the TDI-5.9 preregistration, v59 evaluator, manifests, reproduction
script and CI workflow are frozen: scientific code must not change;
constants must not change; seed blocks must not change; the spectral
moments, the layouts and the criteria must not change; no full run may
begin before all frozen hashes pass (the full ancestor chain through
TDI-6.4, and TDI-5.9 itself); any scientific-code defect discovered after
freezing requires a new experiment identifier — TDI-5.9 may not be silently
patched.
