# TDI-5.5 — The Baseline Challenge: Contraction and Persistence Confounds

## Preregistration

This document is the frozen preregistration for TDI-5.5. Once its SHA-256
manifest, the v55 evaluator, the reproduction script, the CI workflow and
the bounded tests are committed, this design is frozen under the Section 21
freeze rule: no scientific constant, seed block, feature definition,
baseline, competitor or criterion may change without a new experiment
identifier. Freezing the design does not authorize a run; the real
experiment may begin only as the deliberate one-time human action described
in Section 16.

## 1. Experimental status and provenance

TDI-5.5 is a new confirmatory experiment derived from the completed and
merged TDI-5.4 result. It is **not** a continuation, patch, or
reinterpretation of TDI-5.1, TDI-5.2, TDI-5.3 or TDI-5.4, each of which
remains frozen under its own identifier.

Every TDI-5.x result to date compares an overlap-augmented model against a
**structural/entropic** baseline. That baseline contains no descriptor of
**contraction** (mixing), even though the TDI target itself,
`U_h = -log2(1 - O_h)` with `O_h = 1 - TV_h`, is a total-variation
contraction ratio and is therefore mathematically adjacent to the classical
theory of Markov-chain contraction (Dobrushin coefficients, coupling,
mixing). The baseline also does not control for **temporal persistence**:
`O_2` is observed at horizon `h_obs = 2`, close to the near targets, so part
of its predictive power could be trivial continuation of the recent
trajectory rather than deep dynamical structure.

TDI-5.5 confronts both confounds head-on. It asks whether the early overlaps
`O_1, O_2` still carry predictive information about the future deficit
**after** the model is given (a) the best *exact* contraction descriptor of
the system, and (b) a naive temporal-persistence competitor. If the overlap
signal survives both, it is a candidate for a genuinely independent
informational dimension (within the exact scope of Section 4). If it does
not, TDI is better understood as a compact estimator of already-known
contraction / persistence structure.

Frozen ancestor identities (to be verified at runtime and in CI):

| Artifact | SHA-256 |
|---|---|
| TDI-5.4 evaluator (v54) | `dcf24d7eb1ccd938a81163738c38d31a693474c8a1d94046734bda243ca772bf` |
| TDI-5.4 preregistration | `229a0a8efa391c67c4dda1322b984109b142be3abf972d0a08f3c4ac742ec6ac` |
| TDI-5.3 evaluator (v53) | `93181fb75d4882be2ca0b26c1babe9db747583d6e08e6992617ec12b7f65460f` |
| TDI-5.3 preregistration | `7223128dcfd751ebeb6488c01c3512d0a10b35937ec170504984295eb421682e` |

No full TDI-5.5 run may begin before all of the following are committed and
frozen: this preregistration; the final evaluator; the evaluator SHA-256
manifest; the scientific-code SHA-256 manifest; the deterministic
reproduction script; the dedicated CI workflow; bounded unit and
termination tests.

## 2. Research questions

TDI-5.5 evaluates, within the frozen candidate machinery:

1. whether `{O_1, O_2}` contribute predictive information about `U_h`
   **beyond a structural/entropic baseline augmented with an exact
   contraction descriptor** (the **contraction confound**, criterion
   TDI-5.5A), at the focal horizons U₃ and U₆;
2. whether the best overlap-augmented model beats a **naive temporal
   persistence competitor** — a fixed, zero-parameter linear extrapolation
   of the recent deficit trajectory (the **persistence confound**, criterion
   TDI-5.5B), at U₃ and U₆;
3. what **functional form** the overlaps' marginal value beyond contraction
   follows across the dense horizon grid U₃…U₈, and at which horizon it
   first becomes practically negligible — a **redundancy horizon** `h★`
   (the **decay law**, criterion TDI-5.5C, descriptive);
4. whether all conclusions replicate across three independent seed blocks
   G/H/I.

TDI-5.5 does **not** re-test the joint signal (5.2A/5.3A), the independent
O₂ signal (5.2B/5.3B), OOD transfer (5.2D/5.3D), or the nonlinear-basis
findings (5.4A/5.4B); those are settled under their own identifiers. It does
**not** use non-exact contraction descriptors (spectral gap, second
eigenvalue, mixing time) or non-parametric model families (trees, kernels,
networks); those are deferred to a separate non-exact track (Section 21).

## 3. Relationship to the frozen ancestors

**Inherited unchanged** from TDI-5.2/5.3/5.4 (frozen; not re-derived here):

- the entire dynamical construction, exact candidate analysis, observation
  geometry, target geometry (`U_h = -log2(1 - O_h)`), width-6 exact
  cardinality, and the preregistered per-candidate exclusion criteria
  (TDI-5.2 Sections 3, 8);
- observation horizon `h_obs = 2`; primary target `U_6`;
- the 13 structural/entropic baseline variables and the two early-overlap
  predictors O₁, O₂ (TDI-5.2 Section 4);
- ridge regression with `lambda = 1.0`, training-only preprocessing and
  target standardization, deterministic accumulation order, closed-form
  linear-in-parameters solution (TDI-5.2 Section 5);
- the deterministic width-3 and width-4 generation budgets (TDI-5.2
  Section 7);
- the paired + stratified-aggregate bootstrap engine and its resampling
  discipline (TDI-5.2 Section 10);
- the 4-way Beneficial / Equivalent / Harmful / Inconclusive classification
  logic and the symmetric 2% relative-MSE margin (TDI-5.2 Section 13);
- the **exact overlap / total-variation primitives** of `tdi-core`
  (`uniform_branching_state_distribution`, `distribution_overlap`), used
  **unchanged** to compute the new contraction descriptors — `tdi-core`
  itself is not modified.

**New in TDI-5.5** (the only substantive scientific additions):

- two **exact contraction descriptors** (the Dobrushin coefficient and the
  mean pairwise total variation of the one-step kernel), computed per
  candidate system from the inherited exact primitives (Section 5), and two
  new layouts **CK** and **CKT** (Section 6);
- a **naive temporal-persistence competitor** (Section 7);
- **fresh, independent seed blocks G/H/I** (Section 9), disjoint from the
  TDI-5.4 blocks D/E/F, with fresh bootstrap seeds (Section 10);
- a **denser target-horizon grid** `H = {3, 4, 5, 6, 7, 8}` (adds U₇);
- three new criteria: **TDI-5.5A** (signal beyond contraction), **TDI-5.5B**
  (signal beyond persistence) and **TDI-5.5C** (decay law), Sections 13–15.

The nonlinear layouts N2/N12/R2/R12 are **not** used: TDI-5.5 is a
linear-in-features confound-control experiment, not a basis-richness study.

## 4. Design notes and confirmatory integrity

### 4.1 Why the contraction descriptors are the *exact* ones only

The TDI-5.x program rests on **bit-exact, deterministic, closed-form**
evaluation. The Dobrushin coefficient and the mean pairwise total variation
of the one-step kernel are **exact rationals** (maxima and means of exact
sum-of-minima overlaps), so they preserve that invariant. The **spectral
gap / second eigenvalue** and the **ε-threshold mixing time** are
transcendental or iterative and would break bit-exactness; they are the
strongest classical contraction descriptors but belong to the deferred
non-exact TDI-6 track (Section 21), **not** TDI-5.5. A TDI-5.5 result
therefore establishes contraction-confound control **only for the exact
Dobrushin-based descriptors**, a boundary recorded honestly in Section 20.

### 4.2 Which contraction descriptors, and why two

Contraction theory bounds trajectory divergence by
`TV(P^h μ, P^h ν) ≤ δ^h · TV(μ, ν)`, where `δ` is the Dobrushin coefficient
of the one-step kernel `P` — precisely the worst-case one-step contraction
rate that *should* predict how fast the deficit `D_h = 1 - O_h` shrinks. The
Dobrushin coefficient (a maximum over state pairs) can saturate near 1, so
TDI-5.5 pairs it with the **mean pairwise total variation** — a
non-saturating aggregate contraction summary. Both are exact, cheap and
distinct from the instance-specific `O_1, O_2`, giving the baseline both a
worst-case and a typical contraction descriptor.

### 4.3 Why a naive persistence competitor, and which one

Because `O_2` is observed just two steps before the near targets, a linear
continuation of the recent deficit trajectory is a strong, zero-parameter
null. The primary competitor extrapolates in **deficit (U) space**, the
primary confirmatory domain:

    U_hat_h = U_2 + (h - 2)(U_2 - U_1),    U_k = -log2(1 - O_k),

equivalently a linear extrapolation of `log2 D_h`. It is a fixed formula
with no learned parameters, so a model that only narrowly beats it is
capturing little beyond trajectory continuation. The O-space linear
extrapolator `O_hat_h = O_2 + (h - 2)(O_2 - O_1)` is retained as an
exploratory alternative only.

### 4.4 Single generator; independence from observed data

TDI-5.5 uses a **single** generator — the base width-3 + width-4
in-distribution composition inherited from TDI-5.4 (generator perturbation
is a separate question, out of scope here). It uses **fresh seed blocks
G/H/I**, disjoint from the TDI-5.4 blocks D/E/F and all earlier blocks, so
every confirmatory quantity is produced from data never used in an observed
result.

## 5. Exact contraction descriptors

For a candidate system of width `w`, every one of the `2^w` states has a
defined one-step `Noop` transition. Let `P(s, ·)` be the exact uniform
distribution over the successor set of state `s` (i.e.
`uniform_branching_state_distribution(system, s, Noop, 1)`). For a pair of
states `(i, j)`, the exact total variation is

    TV(P_i, P_j) = 1 - overlap(P_i, P_j),

with `overlap` the inherited exact sum-of-minima (`distribution_overlap`).
TDI-5.5 computes two descriptors over all unordered state pairs:

- **Dobrushin coefficient** — `delta = max_{i<j} TV(P_i, P_j)`;
- **mean pairwise total variation** — `delta_bar = mean_{i<j} TV(P_i, P_j)`.

Both are exact rationals in `[0, 1]`, converted to `f64` exactly like O₁ and
O₂ (`ratio.as_f64()`), and standardized downstream with training-only
statistics like every other feature. A non-finite descriptor triggers the
same graceful per-candidate exclusion as any non-finite feature.

## 6. Feature layouts

The 13 baseline variables stay linear and unchanged. All TDI-5.5 layouts are
linear in their features:

| Layout | Variables | Count | Role |
|---|---|---:|---|
| **B0** (C₀) | 13 baseline | 13 | baseline; exploratory reference |
| **B1** | baseline + O₁ | 14 | exploratory (inherited) |
| **B2** | baseline + O₂ | 14 | exploratory (inherited) |
| **B12** | baseline + O₁ + O₂ | 15 | exploratory: TDI beyond C₀ |
| **BD** | baseline + (O₂ − O₁) | 14 | exploratory (inherited) |
| **CK** | baseline + δ + δ̄ | 15 | **contraction baseline** (confirmatory) |
| **CKT** | baseline + δ + δ̄ + O₁ + O₂ | 17 | **full model** (confirmatory) |

`CKT` minus `CK` isolates the marginal contribution of `{O_1, O_2}` **after**
the contraction descriptors are already present. Ridge `lambda = 1.0` is
unchanged; all layouts for a block and horizon share one target scaler.

## 7. Persistence competitor

The naive persistence competitor is a fixed, zero-parameter predictor
(Section 4.3), not a fitted layout. For target horizon `h` and each holdout
record it predicts the standardized deficit from
`U_hat_h = U_2 + (h - 2)(U_2 - U_1)` (with `U_k = -log2(1 - O_k)` and the
overlap clamped into `[0, 1)` so full early recovery maps to a large finite
deficit), standardized with the block's horizon-`h` target scaler; its
reconstructed-overlap image is `1 - 2^{-U_hat_h}`, clamped to `[0, 1]`. It is
compared to a fitted layout through exactly the same paired /
stratified-aggregate bootstrap and four-way classifier as any layout-vs-
layout comparison.

## 8. Populations

TDI-5.5 uses a single generator and generates only in-distribution
populations. **No OOD populations are generated.** For each of the three
seed blocks:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |

Accepted records per block: **40,000**. Total: **120,000**. Models are fitted
on each block's combined width-3 + width-4 training population; every
criterion is evaluated on that block's combined width-3 + width-4 holdout
population. Holdout records never affect fitting.

## 9. Independent seed blocks (fresh)

Three deterministic, pairwise-disjoint seed blocks G/H/I, **disjoint from the
TDI-5.4 blocks D/E/F and all earlier blocks**. The evaluator verifies at
runtime that all consumed seed ranges are pairwise disjoint. Total seed
reservations: **12**.

### Block G

| Population | Initial seed |
|---|---:|
| training w3 | 760,000,000 |
| holdout w3 | 770,000,000 |
| training w4 | 780,000,000 |
| holdout w4 | 790,000,000 |

### Block H

| Population | Initial seed |
|---|---:|
| training w3 | 860,000,000 |
| holdout w3 | 870,000,000 |
| training w4 | 880,000,000 |
| holdout w4 | 890,000,000 |

### Block I

| Population | Initial seed |
|---|---:|
| training w3 | 960,000,000 |
| holdout w3 | 970,000,000 |
| training w4 | 980,000,000 |
| holdout w4 | 990,000,000 |

Generation budgets are inherited unchanged from TDI-5.2 Section 7 (width-3
multiplier 64 / no-progress 25,000; width-4 multiplier 96 / no-progress
50,000).

## 10. Deterministic bootstrap

The bootstrap engine, replicate count (4,000) and resampling discipline are
inherited unchanged from TDI-5.2 Section 10. TDI-5.5 uses fresh bootstrap
seeds, disjoint from every TDI-5.2/5.3/5.4 bootstrap seed:

    block G / H / I       : 0x5444493535000001 / …000002 / …000003
    stratified aggregate  : 0x5444493535004747

For each confirmatory comparison, report the two-sided 95% interval of the
baseline-minus-challenger MSE difference and, for equivalence
classification, the two-sided 95% interval of the relative MSE difference.

## 11. Metrics

For every block, population, horizon and layout, print the full metric set of
TDI-5.2 Section 9, plus, for every confirmatory comparison (including the
persistence comparison), the absolute MSE difference, relative MSE reduction,
absolute MAE difference, Spearman difference, R² difference and absolute-bias
difference.

## 12. Standardized-U primacy

Standardized U space is the primary confirmatory domain (TDI-5.2 Section 5).
Reconstructed-O-space quantities are secondary diagnostics only and cannot
determine any TDI-5.5 criterion.

## 13. Criterion TDI-5.5A — signal beyond contraction

Compare **CKT against CK** on combined width-3 + width-4 holdout at the focal
horizons **U₃** and **U₆**, using the symmetric relative-MSE margin of 2
percent and the exact 4-way classification logic of TDI-5.2 Section 13. This
yields one 4-way classification at U₃ and one at U₆.

TDI-5.5A is a preregistered classification, not forced to any result. The
informative outcomes are symmetric: *Beneficial* would show `{O_1, O_2}`
carry predictive information the exact contraction descriptors do not — a
candidate independent informational dimension; *Equivalent* would show the
overlaps are, within the exact scope, redundant with contraction and TDI is
better read as a compact contraction estimator.

## 14. Criterion TDI-5.5B — signal beyond persistence

Compare **CKT against the naive persistence competitor** (Section 7) on the
same holdout at the focal horizons **U₃** and **U₆**, using the same margin
and four-way classifier. This yields one 4-way classification at each focal
horizon.

TDI-5.5B is a preregistered classification, not forced to any result.
*Beneficial* would show the model captures dynamical structure beyond
trajectory continuation; *Equivalent or Harmful* (especially at the near
horizon U₃) would show its predictive power there is largely temporal
persistence.

## 15. Criterion TDI-5.5C — decay law and redundancy horizon

Evaluate the **CKT-vs-CK** comparison at **every** horizon of the dense grid
`H = {3, 4, 5, 6, 7, 8}`, and characterize the overlaps' marginal value
beyond contraction across horizons. For each horizon `h`, the marginal value
is the aggregate relative-MSE reduction of CKT over CK in standardized-U
space; its 4-way classification is as in Section 13.

TDI-5.5C reports, as its preregistered descriptive summary:

1. the six aggregate relative-MSE reductions, one per horizon;
2. the six 4-way classifications, one per horizon;
3. `monotone_non_increasing` — whether the six reductions are non-increasing
   in horizon;
4. `first_equivalent_horizon` — the redundancy horizon `h★`, the smallest
   horizon whose classification is Equivalent, or none;
5. `successive_ratios` — the five ratios `r_{h+1} / r_h` of the reductions,
   reported so the decay can be inspected for a geometric shape.

TDI-5.5C makes **no** success/failure claim: it is a preregistered
descriptive criterion. A monotone decay to negligibility, a non-monotone
profile, or a sharp threshold are all legitimate outcomes.

## 16. Operational activation and full-run entrypoint contract

The v55 evaluator exposes exactly three modes:

    --termination-smoke
    --preflight
    --full

A bare, no-argument invocation must refuse to run. `--termination-smoke` uses
only bounded tiny data and produces no result artifacts. `--preflight`
performs no scientific generation: it verifies the full frozen configuration
(all 12 seed reservations, all expected counts, all bootstrap constants),
verifies that the full pipeline is wired to `--full`, prints all TDI-5.5 and
ancestor identities and the exact real-run command, and exits without a
result.

`--full` requires the exact confirmation environment variable:

    TDI55_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI55_FREEZE_RULE

Without that exact value, `--full` must fail before any generation, fitting
or bootstrap. The confirmation check is a pure function of the environment
value, unit-testable without starting the experiment. No TDI-5.5 commit,
test, or CI run may supply the token. The full run is a deliberate, one-time
human action; the authoring agent must never invoke `--full` with the real
token.

## 17. Required raw output

Inherited from TDI-5.2 Section 17 with TDI-5.5 identities: git commit;
compiler/Cargo versions; v55 evaluator SHA-256; TDI-5.5 preregistration
SHA-256; TDI-5.5 scientific-manifest SHA-256; the frozen-ancestor hashes;
all frozen constants; the seed-block definitions; requested/accepted/
rejected/attempted counts; rejection counts by reason; final exclusive seeds;
generation budgets; target scalers; **CK and CKT model coefficients**
(including the contraction-descriptor coefficients) for every block; all
metrics; all bootstrap intervals; the per-horizon CKT-vs-CK and the focal
CKT-vs-persistence comparisons; the TDI-5.5A and TDI-5.5B focal
classifications; the TDI-5.5C decay-law summary; deterministic termination
diagnostics.

## 18. Determinism

Inherited from TDI-5.2 Section 18. Candidate generation, seed consumption,
exclusions, preprocessing, **contraction-descriptor and persistence-competitor
construction**, model fitting, bootstrap sampling, aggregation, metric
calculation, iteration order, scientific-value formatting and final criteria
are deterministic functions of committed constants. Wall-clock timestamps are
reproduction metadata only.

## 19. Reproduction requirements

The TDI-5.5 reproduction script must satisfy every requirement of TDI-5.2
Section 19 / TDI-5.3 Section 8 / TDI-5.4 Section 17 (refuse a dirty
repository; verify all frozen hashes including TDI-5.1/5.2/5.3/5.4 and
TDI-5.5; refuse an existing partial or complete result; acquire an exclusive
lock; compile offline in release mode; execute the evaluator exactly once
with `--full`; capture complete output; verify all final criterion lines;
write metadata and a completion marker; hash all artifacts; make final
artifacts read-only), plus: it must require the exact confirmation variable
before invoking the evaluator, and must refuse to run over an existing
TDI-5.5 result.

## 20. Interpretation boundaries

A TDI-5.5 result establishes the (non)contribution of `{O_1, O_2}` beyond the
**exact Dobrushin-based contraction descriptors of Section 5** and beyond the
**specific naive persistence competitor of Section 7**, within the frozen
candidate machinery and the single base generator, replicated across three
seed blocks. It does **not** establish: control against non-exact contraction
descriptors (spectral gap, mixing time — Section 21); control against
arbitrary persistence models; sufficiency under nonlinear or non-parametric
model families; robustness to generator changes; causal effects; universal
validity across dynamical systems; or external empirical validity. The
TDI-5.5A, TDI-5.5B and TDI-5.5C summaries may not be rewritten after
observing the full result.

## 21. Deferred non-exact track (TDI-6)

The strongest classical contraction descriptors (spectral gap / second
eigenvalue, ε-threshold mixing time) and genuinely non-parametric model
families (trees, kernels, networks) are **incompatible with the bit-exact,
closed-form, deterministic invariant** that defines the TDI-5.x program
(Section 4.1). Together with a formal information-decomposition treatment of
the unique/redundant/synergistic split between O₁ and O₂, they are **formally
deferred to a separate future experiment identifier, TDI-6**, which would
carry its own preregistration and its own explicitly non-exact determinism
discipline (fixed training seeds, a declared floating-point / threading
regime, tolerance-based reproduction). TDI-6 is **out of scope for TDI-5.5**
and is recorded here only so the omission is a deliberate, documented
boundary. Nothing in TDI-5.5 presupposes, constrains, or authorizes a TDI-6
run.

## 22. Freeze rule

After the TDI-5.5 preregistration, v55 evaluator, manifests, reproduction
script and CI workflow are frozen: scientific code must not change; constants
must not change; seed blocks must not change; the contraction descriptors,
the persistence competitor, the layouts and the criteria must not change; no
full run may begin before all frozen hashes pass (TDI-5.1, 5.2, 5.3, 5.4 and
5.5); any scientific-code defect discovered after freezing requires a new
experiment identifier — TDI-5.5 may not be silently patched.
