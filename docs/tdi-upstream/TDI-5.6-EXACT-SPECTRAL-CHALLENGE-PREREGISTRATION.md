# TDI-5.6 — The Spectral Challenge: Exact Spectral Moments of the One-Step Kernel

## Preregistration

This document is the frozen preregistration for TDI-5.6. Once its SHA-256
manifest, the v56 evaluator, the reproduction script, the CI workflow and
the bounded tests are committed, this design is frozen under the Section 22
freeze rule: no scientific constant, seed block, feature definition,
baseline or criterion may change without a new experiment identifier.
Freezing the design does not authorize a run; the real experiment may begin
only as the deliberate one-time human action described in Section 16.

## 1. Experimental status and provenance

TDI-5.6 is a new confirmatory experiment derived from the completed and
merged TDI-5.5 result. It is **not** a continuation, patch, or
reinterpretation of TDI-5.1, TDI-5.2, TDI-5.3, TDI-5.4 or TDI-5.5, each of
which remains frozen under its own identifier.

TDI-5.5 established that the early overlaps `{O_1, O_2}` carry predictive
information about the future deficit **beyond an exact contraction
descriptor** (the Dobrushin coefficient `delta` and the mean pairwise total
variation `delta_bar` of the one-step kernel) and **beyond a naive temporal
persistence competitor**, at every horizon U₃…U₈. The Dobrushin coefficient
is, however, only the *worst-case one-step* contraction rate: a single
maximum over state pairs. It does not see the **subdominant spectrum** of the
kernel — the decay rates that classical mixing theory ties to the
**second eigenvalue** and the **spectral gap**. A skeptic can therefore still
argue that TDI's overlap signal is a repackaging of spectral structure the
Dobrushin descriptor cannot express.

TDI-5.6 confronts that objection with the strongest contraction descriptors
that remain **bit-exact**: the **exact spectral moments** of the one-step
kernel, `s_2 = trace(P^2)` and `s_3 = trace(P^3)`. These are exact rational
functions of the kernel (sums of closed-walk products of unit fractions),
they equal the power sums `sum_i lambda_i^2` and `sum_i lambda_i^3` of the
eigenvalue spectrum, and they encode subdominant-eigenvalue information the
Dobrushin coefficient does not. TDI-5.6 asks whether `{O_1, O_2}` still carry
predictive information **after** the model is given both the exact Dobrushin
descriptors **and** the exact spectral moments. If the overlap signal
survives this strictly richer exact spectral control, it is a stronger
candidate for a genuinely independent informational dimension (within the
exact scope of Section 4). If it does not, TDI is better understood as a
compact estimator of already-known spectral structure.

Frozen ancestor identities (to be verified at runtime and in CI):

| Artifact | SHA-256 |
|---|---|
| TDI-5.5 evaluator (v55) | `10df698d10f010b9f6c18e2a4d78042eb399d3812b8d69c2b4bb799de828b835` |
| TDI-5.5 preregistration | `37260b3349107659487e42e66c269ecad44efaf6131f8206bb28dfbcf83f9da1` |
| TDI-5.4 evaluator (v54) | `dcf24d7eb1ccd938a81163738c38d31a693474c8a1d94046734bda243ca772bf` |
| TDI-5.4 preregistration | `229a0a8efa391c67c4dda1322b984109b142be3abf972d0a08f3c4ac742ec6ac` |

The v56 evaluator and the CI workflow additionally verify the **full frozen
chain** TDI-5.1 → TDI-5.5 (every ancestor evaluator and preregistration hash)
before any generation.

No full TDI-5.6 run may begin before all of the following are committed and
frozen: this preregistration; the final evaluator; the evaluator SHA-256
manifest; the scientific-code SHA-256 manifest; the deterministic
reproduction script; the dedicated CI workflow; bounded unit and termination
tests.

## 2. Research questions

TDI-5.6 evaluates, within the frozen candidate machinery:

1. whether `{O_1, O_2}` contribute predictive information about `U_h`
   **beyond a structural/entropic baseline augmented with the exact Dobrushin
   contraction descriptors *and* the exact spectral moments** of the one-step
   kernel (the **spectral confound**, criterion TDI-5.6A), at the focal
   horizons U₃ and U₆;
2. whether the **exact spectral moments themselves** contribute predictive
   information beyond the exact Dobrushin descriptors (the **marginal
   spectral value**, criterion TDI-5.6B), at U₃ and U₆ — a measurement that
   calibrates how strong the spectral control in TDI-5.6A actually is in this
   system;
3. what **functional form** the overlaps' marginal value beyond the full
   exact descriptor set follows across the dense horizon grid U₃…U₈, and at
   which horizon it first becomes practically negligible — a **redundancy
   horizon** `h★` (the **decay law**, criterion TDI-5.6C, descriptive);
4. whether all conclusions replicate across three independent seed blocks
   J/K/L.

TDI-5.6 does **not** re-test the joint signal (5.2A/5.3A), the independent
O₂ signal (5.2B/5.3B), OOD transfer (5.2D/5.3D), the nonlinear-basis
findings (5.4A/5.4B), or the persistence confound (5.5B); those are settled
under their own identifiers. It does **not** use non-exact spectral
descriptors (the literal second eigenvalue `|lambda_2|`, the spectral gap
`1 - |lambda_2|`, or the ε-threshold mixing time) or non-parametric model
families (trees, kernels, networks); those are deferred to the separate
non-exact track (Section 21).

## 3. Relationship to the frozen ancestors

**Inherited unchanged** from TDI-5.2/5.3/5.4/5.5 (frozen; not re-derived
here):

- the entire dynamical construction, exact candidate analysis, observation
  geometry, target geometry (`U_h = -log2(1 - O_h)`), width-6 exact
  cardinality, and the preregistered per-candidate exclusion criteria
  (TDI-5.2 Sections 3, 8);
- observation horizon `h_obs = 2`; primary target `U_6`;
- the 13 structural/entropic baseline variables and the two early-overlap
  predictors O₁, O₂ (TDI-5.2 Section 4);
- the two **exact contraction descriptors** δ (Dobrushin coefficient) and
  δ̄ (mean pairwise total variation) of the one-step `Noop` kernel, computed
  per candidate system exactly as in TDI-5.5 Section 5 (unchanged);
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
- the **exact overlap / total-variation primitives** of `tdi-core`
  (`uniform_branching_state_distribution`, `distribution_overlap`,
  `ExactRatio`), used **unchanged** to compute the new spectral moments —
  `tdi-core` itself is not modified.

**New in TDI-5.6** (the only substantive scientific additions):

- two **exact spectral moments** of the one-step kernel, `s_2 = trace(P^2)`
  and `s_3 = trace(P^3)`, computed per candidate system from the inherited
  exact primitives (Section 5), and two new layouts **SK** and **SKT**
  (Section 6);
- **fresh, independent seed blocks J/K/L** (Section 9), disjoint from the
  TDI-5.5 blocks G/H/I and all earlier blocks, with fresh bootstrap seeds
  (Section 10);
- three new criteria: **TDI-5.6A** (signal beyond spectral moments),
  **TDI-5.6B** (marginal spectral value) and **TDI-5.6C** (decay law),
  Sections 13–15.

**Dropped relative to TDI-5.5.** The naive temporal-persistence competitor
(TDI-5.5 Section 7) is **removed**: the persistence confound is settled by
TDI-5.5B, and every TDI-5.6 criterion is a fitted-layout-versus-fitted-layout
comparison. The nonlinear layouts N2/N12/R2/R12 are not used: TDI-5.6 is a
linear-in-features confound-control experiment, not a basis-richness study.

## 4. Design notes and confirmatory integrity

### 4.1 Why the spectral descriptors are the *exact moments* only

The TDI-5.x program rests on **bit-exact, deterministic, closed-form**
evaluation. The literal second eigenvalue `|lambda_2|`, the spectral gap
`1 - |lambda_2|`, and the ε-threshold mixing time are **transcendental or
iterative**: they are roots of a characteristic polynomial or fixed points of
an iteration, and computing them introduces floating-point tolerances that
would break the bit-exact invariant. They belong to the deferred non-exact
track (Section 21), **not** TDI-5.6.

The **spectral moments** `s_2 = trace(P^2)` and `s_3 = trace(P^3)` are, by
contrast, **exact rationals**: `trace(P^k)` is a finite sum of products of the
kernel's rational entries (Section 5), so it is computed with the same
arbitrary-precision `ExactRatio` arithmetic as every other exact quantity in
the program, with a single final rounding to `f64`. They are also the power
sums of the eigenvalue spectrum,

    s_k = trace(P^k) = sum_i lambda_i^k,

so they carry genuine information about the subdominant eigenvalues (and hence
about the spectral gap) **without ever computing an eigenvalue**. A TDI-5.6
result therefore establishes spectral-confound control **for the exact
spectral moments of orders 2 and 3**, a boundary recorded honestly in
Section 20; it does not, and does not claim to, control for the literal
spectral gap.

### 4.2 Which spectral moments, and why two

The first moment `trace(P) = sum_i P_{ii}` counts one-step self-loops and adds
nothing beyond the local branching already visible to the baseline, so it is
omitted. The **second moment** `s_2 = sum_i lambda_i^2 = 1 + sum_{i>=2}
lambda_i^2` is the first moment that sees the subdominant spectrum: a kernel
with a small spectral gap (a subdominant eigenvalue near ±1) has a larger
`s_2`. The **third moment** `s_3 = sum_i lambda_i^3` adds the sign/asymmetry
information the even moment cannot express (it distinguishes a subdominant
`lambda ≈ +r` from `lambda ≈ -r`, and is sensitive to short directed cycles).
Together the two exact moments give the baseline a symmetric and an
asymmetric summary of the subdominant spectrum, both exact, both cheap, and
both distinct from the instance-specific `O_1, O_2`.

### 4.3 Range and standardization

For an `N`-state kernel (`N = 2^w`), each moment satisfies `0 <= s_k <= N`
(the diagonal of a non-negative matrix power is non-negative, and
`|sum_i lambda_i^k| <= sum_i |lambda_i|^k <= N` because every eigenvalue of a
stochastic matrix satisfies `|lambda_i| <= 1`). Unlike δ, δ̄ ∈ [0, 1], the
moments live in `[0, N]`; they are standardized downstream with training-only
statistics exactly like every other feature, so their differing scale is
immaterial to the ridge fit. A non-finite moment (impossible for the exact
computation, but guarded) triggers the same graceful per-candidate exclusion
as any non-finite feature.

### 4.4 Single generator; independence from observed data

TDI-5.6 uses a **single** generator — the base width-3 + width-4
in-distribution composition inherited from TDI-5.4/5.5 (generator
perturbation is a separate question, out of scope here). It uses **fresh seed
blocks J/K/L**, disjoint from the TDI-5.5 blocks G/H/I and all earlier
blocks, so every confirmatory quantity is produced from data never used in an
observed result.

## 5. Exact spectral moments

For a candidate system of width `w`, let `P` be the one-step `Noop` kernel on
the `N = 2^w` states: every state `s` has a defined `Noop` transition, and
`P(s, .)` is the exact uniform distribution over the successor set of `s`
(i.e. `uniform_branching_state_distribution(system, s, Noop, 1)`), so
`P_{s,t} = 1/d_s` when `t` is one of the `d_s` distinct successors of `s`, and
`0` otherwise. `P` is row-stochastic and total.

The spectral moments are the traces of the matrix powers, written as sums over
**closed walks**:

    s_2 = trace(P^2) = sum_{i,j} P_{ij} P_{ji}
        = sum over ordered pairs (i, j) with j a successor of i and i a
          successor of j, of 1/(d_i d_j);

    s_3 = trace(P^3) = sum_{i,j,k} P_{ij} P_{jk} P_{ki}
        = sum over ordered triples (i, j, k) with j a successor of i, k a
          successor of j and i a successor of k, of 1/(d_i d_j d_k).

Each summand is a **unit fraction** whose denominator is a product of at most
three branching factors, each `d <= N <= 16`, so the denominator is at most
`16^3 = 4096` and fits in `u128`. The summands are accumulated into a running
**exact rational** with the inherited arbitrary-precision `ExactRatio`
addition (`checked_add`), and only the final total is converted to `f64` with
a single `as_f64()` rounding — the same exactness discipline used for δ, δ̄,
O₁ and O₂. No eigenvalue, characteristic polynomial, or floating-point
iteration is involved; the result is a deterministic exact rational in
`[0, N]`.

Both moments are computed per candidate system during generation, stored on
the record alongside δ and δ̄, and standardized downstream like every other
feature.

## 6. Feature layouts

The 13 baseline variables stay linear and unchanged. All TDI-5.6 layouts are
linear in their features:

| Layout | Variables | Count | Role |
|---|---|---:|---|
| **B0** (C₀) | 13 baseline | 13 | baseline; exploratory reference |
| **B1** | baseline + O₁ | 14 | exploratory (inherited) |
| **B2** | baseline + O₂ | 14 | exploratory (inherited) |
| **B12** | baseline + O₁ + O₂ | 15 | exploratory: TDI beyond C₀ |
| **BD** | baseline + (O₂ − O₁) | 14 | exploratory (inherited) |
| **CK** | baseline + δ + δ̄ | 15 | **contraction baseline** (confirmatory; 5.6B baseline) |
| **SK** | baseline + δ + δ̄ + s₂ + s₃ | 17 | **spectral baseline** (confirmatory) |
| **SKT** | baseline + δ + δ̄ + s₂ + s₃ + O₁ + O₂ | 19 | **full model** (confirmatory) |

`SKT` minus `SK` isolates the marginal contribution of `{O_1, O_2}` **after**
both the contraction descriptors and the spectral moments are already present
(criteria 5.6A, 5.6C). `SK` minus `CK` isolates the marginal contribution of
the spectral moments `{s_2, s_3}` **after** the contraction descriptors are
present (criterion 5.6B). Ridge `lambda = 1.0` is unchanged; all layouts for a
block and horizon share one target scaler.

## 7. No persistence competitor

Unlike TDI-5.5, TDI-5.6 introduces **no fixed non-fitted competitor**. The
persistence confound is settled by TDI-5.5B. Every TDI-5.6 criterion compares
two fitted ridge layouts through the identical paired / stratified-aggregate
bootstrap and four-way classifier.

## 8. Populations

TDI-5.6 uses a single generator and generates only in-distribution
populations. **No OOD populations are generated.** For each of the three seed
blocks:

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

Three deterministic, pairwise-disjoint seed blocks J/K/L, **disjoint from the
TDI-5.5 blocks G/H/I and all earlier blocks**. The evaluator verifies at
runtime that all consumed seed ranges are pairwise disjoint. Total seed
reservations: **12**.

### Block J

| Population | Initial seed |
|---|---:|
| training w3 | 1,060,000,000 |
| holdout w3 | 1,070,000,000 |
| training w4 | 1,080,000,000 |
| holdout w4 | 1,090,000,000 |

### Block K

| Population | Initial seed |
|---|---:|
| training w3 | 1,160,000,000 |
| holdout w3 | 1,170,000,000 |
| training w4 | 1,180,000,000 |
| holdout w4 | 1,190,000,000 |

### Block L

| Population | Initial seed |
|---|---:|
| training w3 | 1,260,000,000 |
| holdout w3 | 1,270,000,000 |
| training w4 | 1,280,000,000 |
| holdout w4 | 1,290,000,000 |

Generation budgets are inherited unchanged from TDI-5.2 Section 7 (width-3
multiplier 64 / no-progress 25,000; width-4 multiplier 96 / no-progress
50,000). Each population consumes at most a few million seeds; the initial
seeds are spaced 10,000,000 apart within a block and 100,000,000 apart across
blocks, so the reservations are pairwise disjoint and disjoint from every
earlier block.

## 10. Deterministic bootstrap

The bootstrap engine, replicate count (4,000) and resampling discipline are
inherited unchanged from TDI-5.2 Section 10. TDI-5.6 uses fresh bootstrap
seeds, disjoint from every TDI-5.2/5.3/5.4/5.5 bootstrap seed:

    block J / K / L       : 0x5444493536000001 / …000002 / …000003
    stratified aggregate  : 0x5444493536004747

For each confirmatory comparison, report the two-sided 95% interval of the
baseline-minus-challenger MSE difference and, for equivalence classification,
the two-sided 95% interval of the relative MSE difference.

## 11. Metrics

For every block, population, horizon and layout, print the full metric set of
TDI-5.2 Section 9, plus, for every confirmatory comparison, the absolute MSE
difference, relative MSE reduction, absolute MAE difference, Spearman
difference, R² difference and absolute-bias difference.

## 12. Standardized-U primacy

Standardized U space is the primary confirmatory domain (TDI-5.2 Section 5).
Reconstructed-O-space quantities are secondary diagnostics only and cannot
determine any TDI-5.6 criterion.

## 13. Criterion TDI-5.6A — signal beyond spectral moments

Compare **SKT against SK** on combined width-3 + width-4 holdout at the focal
horizons **U₃** and **U₆**, using the symmetric relative-MSE margin of 2
percent and the exact 4-way classification logic of TDI-5.2 Section 13. This
yields one 4-way classification at U₃ and one at U₆.

TDI-5.6A is the **primary** preregistered classification, not forced to any
result. The informative outcomes are symmetric: *Beneficial* would show
`{O_1, O_2}` carry predictive information neither the exact contraction
descriptors nor the exact spectral moments express — a stronger candidate
independent informational dimension than TDI-5.5 established; *Equivalent*
would show the overlaps are, within the exact scope, redundant with the exact
spectral structure and TDI is better read as a compact spectral estimator.

## 14. Criterion TDI-5.6B — marginal spectral value

Compare **SK against CK** on the same holdout at the focal horizons **U₃** and
**U₆**, using the same margin and four-way classifier. This yields one 4-way
classification at each focal horizon.

TDI-5.6B is a preregistered classification, not forced to any result. It
measures whether the exact spectral moments add predictive value **beyond the
Dobrushin descriptors** in this system, and so calibrates the strength of the
control applied in TDI-5.6A. *Beneficial* would show the spectral moments are
a genuinely informative addition (making TDI-5.6A a demanding test);
*Equivalent* would show that, in this generator, the moments add little beyond
the worst-case contraction rate (so TDI-5.6A is close to a re-run of TDI-5.5A
against a slightly richer but weak spectral baseline). Either outcome is
reported honestly and neither weakens TDI-5.6A, which is classified on its own
terms.

## 15. Criterion TDI-5.6C — decay law and redundancy horizon

Evaluate the **SKT-vs-SK** comparison at **every** horizon of the dense grid
`H = {3, 4, 5, 6, 7, 8}`, and characterize the overlaps' marginal value beyond
the full exact descriptor set across horizons. For each horizon `h`, the
marginal value is the aggregate relative-MSE reduction of SKT over SK in
standardized-U space; its 4-way classification is as in Section 13.

TDI-5.6C reports, as its preregistered descriptive summary:

1. the six aggregate relative-MSE reductions, one per horizon;
2. the six 4-way classifications, one per horizon;
3. `monotone_non_increasing` — whether the six reductions are non-increasing
   in horizon;
4. `first_equivalent_horizon` — the redundancy horizon `h★`, the smallest
   horizon whose classification is Equivalent, or none;
5. `successive_ratios` — the five ratios `r_{h+1} / r_h` of the reductions,
   reported so the decay can be inspected for a geometric shape.

TDI-5.6C makes **no** success/failure claim: it is a preregistered
descriptive criterion. A monotone decay to negligibility, a non-monotone
profile, or a sharp threshold are all legitimate outcomes.

## 16. Operational activation and full-run entrypoint contract

The v56 evaluator exposes exactly three modes:

    --termination-smoke
    --preflight
    --full

A bare, no-argument invocation must refuse to run. `--termination-smoke` uses
only bounded tiny data and produces no result artifacts. `--preflight`
performs no scientific generation: it verifies the full frozen configuration
(all 12 seed reservations, all expected counts, all bootstrap constants),
verifies that the full pipeline is wired to `--full`, prints all TDI-5.6 and
ancestor identities and the exact real-run command, and exits without a
result.

`--full` requires the exact confirmation environment variable:

    TDI56_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI56_FREEZE_RULE

Without that exact value, `--full` must fail before any generation, fitting or
bootstrap. The confirmation check is a pure function of the environment value,
unit-testable without starting the experiment. No TDI-5.6 commit, test, or CI
run may supply the token. The full run is a deliberate, one-time human action;
the authoring agent must never invoke `--full` with the real token.

## 17. Required raw output

Inherited from TDI-5.2 Section 17 with TDI-5.6 identities: git commit;
compiler/Cargo versions; v56 evaluator SHA-256; TDI-5.6 preregistration
SHA-256; TDI-5.6 scientific-manifest SHA-256; the frozen-ancestor hashes; all
frozen constants; the seed-block definitions; requested/accepted/rejected/
attempted counts; rejection counts by reason; final exclusive seeds;
generation budgets; target scalers; **CK, SK and SKT model coefficients**
(including the contraction-descriptor and spectral-moment coefficients) for
every block; all metrics; all bootstrap intervals; the per-horizon SKT-vs-SK
and the focal SK-vs-CK comparisons; the TDI-5.6A and TDI-5.6B focal
classifications; the TDI-5.6C decay-law summary; deterministic termination
diagnostics.

## 18. Determinism

Inherited from TDI-5.2 Section 18. Candidate generation, seed consumption,
exclusions, preprocessing, **contraction-descriptor and spectral-moment
construction**, model fitting, bootstrap sampling, aggregation, metric
calculation, iteration order, scientific-value formatting and final criteria
are deterministic functions of committed constants. Wall-clock timestamps are
reproduction metadata only.

## 19. Reproduction requirements

The TDI-5.6 reproduction script must satisfy every requirement of TDI-5.2
Section 19 / TDI-5.3 Section 8 / TDI-5.4 Section 17 / TDI-5.5 Section 19
(refuse a dirty repository; verify all frozen hashes including
TDI-5.1/5.2/5.3/5.4/5.5 and TDI-5.6; refuse an existing partial or complete
result; acquire an exclusive lock; compile offline in release mode; execute
the evaluator exactly once with `--full`; capture complete output; verify all
final criterion lines; write metadata and a completion marker; hash all
artifacts; make final artifacts read-only), plus: it must require the exact
confirmation variable before invoking the evaluator, and must refuse to run
over an existing TDI-5.6 result.

## 20. Interpretation boundaries

A TDI-5.6 result establishes the (non)contribution of `{O_1, O_2}` beyond the
**exact Dobrushin contraction descriptors of TDI-5.5 Section 5** and the
**exact spectral moments `s_2 = trace(P^2)`, `s_3 = trace(P^3)` of Section 5**,
within the frozen candidate machinery and the single base generator,
replicated across three seed blocks. It does **not** establish: control
against the **literal second eigenvalue / spectral gap or the mixing time**
(transcendental / iterative — Section 21), which the exact moments constrain
but do not pin down; control against spectral moments of order `>= 4`;
sufficiency under nonlinear or non-parametric model families; robustness to
generator changes; causal effects; universal validity across dynamical
systems; or external empirical validity. The TDI-5.6A, TDI-5.6B and TDI-5.6C
summaries may not be rewritten after observing the full result.

## 21. Deferred non-exact track (TDI-6)

The **literal** spectral descriptors (the second eigenvalue `|lambda_2|`, the
spectral gap `1 - |lambda_2|`, the ε-threshold mixing time) and genuinely
non-parametric model families (trees, kernels, networks) are **incompatible
with the bit-exact, closed-form, deterministic invariant** that defines the
TDI-5.x program (Section 4.1). Together with a formal
information-decomposition treatment of the unique/redundant/synergistic split
between O₁ and O₂, they are **formally deferred to a separate future
experiment identifier, TDI-6**, which would carry its own preregistration and
its own explicitly non-exact determinism discipline (fixed training seeds, a
declared floating-point / threading regime, tolerance-based reproduction).
TDI-6 is **out of scope for TDI-5.6** and is recorded here only so the
omission is a deliberate, documented boundary. Nothing in TDI-5.6 presupposes,
constrains, or authorizes a TDI-6 run.

## 22. Freeze rule

After the TDI-5.6 preregistration, v56 evaluator, manifests, reproduction
script and CI workflow are frozen: scientific code must not change; constants
must not change; seed blocks must not change; the spectral moments, the
layouts and the criteria must not change; no full run may begin before all
frozen hashes pass (TDI-5.1, 5.2, 5.3, 5.4, 5.5 and 5.6); any scientific-code
defect discovered after freezing requires a new experiment identifier —
TDI-5.6 may not be silently patched.
