# TDI-5.4 — Nonlinear Sufficiency and Horizon-Invariance of the O₁/O₂ Asymmetry

## Preregistration

This document is the frozen preregistration for TDI-5.4. Once its SHA-256
manifest, the v54 evaluator, the reproduction script, the CI workflow and
the bounded tests are committed, this design is frozen under the Section 19
freeze rule: no scientific constant, seed block, nonlinear basis, or
criterion may change without a new experiment identifier. Freezing the
design does not authorize a run; the real experiment may begin only as the
deliberate one-time human action described in Section 14.

## 1. Experimental status and provenance

TDI-5.4 is a new confirmatory experiment derived from the completed and
merged TDI-5.3 result. It is **not** a continuation, patch, or
reinterpretation of TDI-5.1, TDI-5.2 or TDI-5.3, each of which remains
frozen under its own identifier.

Motivating result (TDI-5.3, real confirmatory run):

- TDI-5.3B **succeeded**: O₂ carries predictive signal independent of O₁.
- TDI-5.3C classified O₁ as **Equivalent** given O₂: within the frozen
  generator and the **linear** ridge family, adding O₁ on top of O₂
  changed aggregate U₆ MSE by ≈0.05%, well inside the 2% margin.

Frozen ancestor identities (to be verified at runtime and in CI):

| Artifact | SHA-256 |
|---|---|
| TDI-5.3 evaluator (v53) | `93181fb75d4882be2ca0b26c1babe9db747583d6e08e6992617ec12b7f65460f` |
| TDI-5.3 preregistration | `7223128dcfd751ebeb6488c01c3512d0a10b35937ec170504984295eb421682e` |
| TDI-5.2 evaluator (v52) | `2308607729659c7546a17530e69773f982d9a1cf41656ea7898e0123ca469ef7` |
| TDI-5.2 preregistration | `f57a054bc95eb2e041434d6e2049509b0dce1a5397f9666d274b1bbac332be35` |

The TDI-5.3 finding raises exactly one scientifically sharp open question
that TDI-5.3's own interpretation boundaries (Section 9) explicitly
excluded: **nonlinear sufficiency**. The Equivalent classification was
established only for the linear ridge family. TDI-5.4 asks whether the
redundancy of O₁ given O₂ survives when the model family is enriched with
a preregistered nonlinear basis — and whether that answer is invariant
across target horizons.

No full TDI-5.4 run may begin before all of the following are committed
and frozen: this preregistration; the final evaluator; the evaluator
SHA-256 manifest; the scientific-code SHA-256 manifest; the deterministic
reproduction script; the dedicated CI workflow; bounded unit and
termination tests.

## 2. Research questions

TDI-5.4 evaluates, within the frozen generator:

1. whether O₁ contributes predictive information conditional on O₂ **when
   the model family is enriched with a preregistered nonlinear basis**
   (nonlinear sufficiency) at the primary horizon U₆;
2. whether that nonlinear O₁/O₂ classification is **invariant across the
   secondary horizons** U₃, U₄, U₅, U₈;
3. whether both conclusions replicate across three independent seed blocks.

TDI-5.4 does **not** re-test the joint signal (TDI-5.2A/5.3A), the
independent O₂ signal (TDI-5.2B/5.3B), or OOD transfer (TDI-5.2D/5.3D);
those are settled under their own identifiers and are out of scope here.

## 3. Relationship to the frozen ancestors

**Inherited unchanged** from TDI-5.2/5.3 (frozen; not re-derived here):

- the entire dynamical construction, observation geometry, target geometry
  (`U_h = -log2(1 - O_h)`), width-6 exact cardinality, and scientific
  exclusions (TDI-5.2 Sections 3, 8);
- observation horizon `h_obs = 2`; target horizons `H = {3, 4, 5, 6, 8}`;
  primary target `U_6`;
- the 13 structural/entropic baseline variables and the two early-overlap
  predictors O₁, O₂ (TDI-5.2 Section 4);
- ridge regression with `lambda = 1.0`, training-only preprocessing and
  target standardization, deterministic accumulation order (TDI-5.2
  Section 5);
- the deterministic generation budgets (TDI-5.2 Section 7);
- the paired + stratified-aggregate bootstrap engine and its resampling
  discipline (TDI-5.2 Section 10);
- the 4-way Beneficial / Equivalent / Harmful / Inconclusive classification
  logic and the symmetric 2% relative-MSE margin (TDI-5.2 Section 13).

**New in TDI-5.4** (the only substantive scientific additions):

- a preregistered **nonlinear basis expansion** over the overlap
  predictors, and two new model layouts **N2** and **N12** (Section 5);
- **fresh, independent seed blocks D/E/F** (Section 7);
- two new criteria, **TDI-5.4A** (nonlinear O₁ sufficiency at U₆) and
  **TDI-5.4B** (horizon-invariance), Sections 11–12;
- a leaner population set: **no OOD populations** (Section 6).

## 4. Design notes and confirmatory integrity

This section records the design reasoning so the confirmatory status of
each criterion is auditable.

### 4.1 Why there is no standalone "linear redundancy" criterion

TDI-5.1 (post-hoc) and TDI-5.2 (Section 4) already establish that
`delta_O = O_2 - O_1` is an **exact linear redundancy**: a linear model
containing O₁ and O₂ spans the same column space as one containing O₁, O₂
and `O_2 - O_1`. TDI-5.2 accordingly declared the `BD` layout
exploratory-only and barred it from any confirmatory criterion. A
confirmatory "does delta_O add signal to a linear model" test would
therefore be **true by construction** — not a hypothesis. The scientifically
meaningful version of the delta_O question exists **only under
nonlinearity** (does an explicit `O_1·O_2` interaction, equivalently a
curvature term distinguishing O₁ from O₂, carry signal?). TDI-5.4 folds
that question into the nonlinear basis (Section 5): the interaction term
`O_1·O_2` is included precisely so that, if the linear redundancy were
masking a nonlinear O₁ contribution, TDI-5.4A/B would detect it.

### 4.2 Why horizons are tested under the nonlinear basis, not the linear one

A "does O₁ stay Equivalent at U₃…U₈ under the **linear** model" criterion
would analyse quantities that are largely **already observable** in the
merged TDI-5.3 result (the linear B2 and B12 models at every horizon were
fitted and their metrics printed). Preregistering an analysis of
already-observed data is not confirmatory. TDI-5.4B therefore evaluates
horizon-invariance of the **nonlinear** comparison (N12 vs N2), whose
model fits have **never been computed**, preserving genuine confirmatory
status.

### 4.3 Independence from the observed TDI-5.3 data

TDI-5.4 uses **fresh seed blocks D/E/F** (Section 7), disjoint from the
TDI-5.3 blocks A/B/C, so the confirmatory quantities are produced from
data never used in an observed result. The N2/N12 nonlinear model fits are
unobserved regardless of seeds; fresh seeds additionally make TDI-5.4 an
independent replication rather than a re-analysis.

## 5. Nonlinear basis expansion and new layouts

The confirmatory novelty of TDI-5.4 is a single, fixed, deterministic
nonlinear basis over the two early-overlap predictors. The 13 baseline
variables remain **linear and unchanged**, so that any difference between
the two new layouts is attributable solely to O₁, O₂ and their curvature
and interaction.

Nonlinear terms are constructed deterministically from the **raw** O₁, O₂
values and then z-standardized using **training-only** statistics, exactly
like every other feature (TDI-5.2 Section 5). No baseline variable receives
a nonlinear term.

Fixed nonlinear terms:

    O_1^2 , O_2^2 , O_1 * O_2

New confirmatory layouts:

| Layout | Variables |
|---|---|
| **N2** (nonlinear, O₂ only) | 13 baseline + O₂ + O₂² |
| **N12** (nonlinear, O₁+O₂) | 13 baseline + O₁ + O₂ + O₁² + O₂² + O₁·O₂ |

N12 minus N2 isolates the full nonlinear marginal contribution of O₁: the
linear term O₁, its curvature O₁², and its interaction with O₂ (`O_1·O_2`).
If all three add nothing beyond N2, O₁ is redundant even under this
nonlinear enrichment. If they add signal, the linear Equivalent finding of
TDI-5.3C was masking a nonlinear O₁ contribution.

No confirmatory layout may contain `O_2 - O_1` (the exploratory BD
direction); the nonlinear O₁ contribution is expressed only through the
terms above.

Ridge `lambda = 1.0` is unchanged and applied to the standardized expanded
design matrix. All layouts for a given block and horizon share one target
scaler (TDI-5.2 Section 5).

## 6. Populations

TDI-5.4 generates only the in-distribution populations needed by its
criteria. **No OOD (width-5, width-6) populations are generated** — OOD
transfer is settled by TDI-5.2D/5.3D and is out of scope.

Each of the three seed blocks contains:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |

Accepted records per block: **40,000**. Total accepted records:
**120,000**.

Models are fitted on each block's combined width-3 + width-4 **training**
population; all criteria are evaluated on that block's combined width-3 +
width-4 **holdout** population. Holdout records never affect fitting.

## 7. Independent seed blocks (fresh)

Three deterministic, pairwise-disjoint seed blocks, **disjoint from the
TDI-5.3 blocks A/B/C**. The evaluator must verify at runtime that all
consumed seed ranges are pairwise disjoint.

### Block D

| Population | Initial seed |
|---|---:|
| training w3 | 460,000,000 |
| holdout w3 | 470,000,000 |
| training w4 | 480,000,000 |
| holdout w4 | 490,000,000 |

### Block E

| Population | Initial seed |
|---|---:|
| training w3 | 560,000,000 |
| holdout w3 | 570,000,000 |
| training w4 | 580,000,000 |
| holdout w4 | 590,000,000 |

### Block F

| Population | Initial seed |
|---|---:|
| training w3 | 660,000,000 |
| holdout w3 | 670,000,000 |
| training w4 | 680,000,000 |
| holdout w4 | 690,000,000 |

Generation budgets are inherited unchanged from TDI-5.2 Section 7 (width-3
multiplier 64 / no-progress 25,000; width-4 multiplier 96 / no-progress
50,000).

## 8. Deterministic bootstrap

The bootstrap engine, replicate count (4,000 per block) and resampling
discipline are inherited unchanged from TDI-5.2 Section 10. Because
TDI-5.4 introduces new comparisons (N12 vs N2), it uses **new, distinct**
bootstrap seeds, disjoint from every TDI-5.2/5.3 bootstrap seed:

    block D:              0x5444493534440001
    block E:              0x5444493534450002
    block F:              0x5444493534460003
    stratified aggregate: 0x5444493534444747

For each confirmatory comparison, report the two-sided 95% interval of the
baseline-minus-challenger MSE difference and, for equivalence
classification, the two-sided 95% interval of the relative MSE difference.

## 9. Metrics

For every block, population, horizon and layout (N2, N12), print the full
metric set of TDI-5.2 Section 9 (MSE, MAE, R², Spearman, bias, observed
mean, predicted mean, calibration intercept, calibration slope, lower and
upper clipping fractions), plus, for every confirmatory comparison, the
absolute MSE difference, relative MSE reduction, absolute MAE difference,
Spearman difference, R² difference and absolute-bias difference.

## 10. Standardized-U primacy

Standardized U space is the primary confirmatory domain (TDI-5.2 Section 5).
Reconstructed-O-space quantities are secondary diagnostics only and cannot
determine any TDI-5.4 criterion.

## 11. Criterion TDI-5.4A — nonlinear O₁ sufficiency at U₆

Compare **N12 against N2** on combined width-3 + width-4 holdout at **U₆**,
using the symmetric relative-MSE margin of 2 percent and the exact 4-way
classification logic of TDI-5.2 Section 13:

- **Beneficial** — N12 improves MSE over N2 by ≥2% in ≥2 blocks with the
  corresponding bootstrap lower bounds above zero, aggregate relative
  improvement ≥2%, and aggregate bootstrap lower bound above zero.
- **Equivalent** — all three block point estimates lie in [−2%, +2%]; ≥2
  block confidence intervals lie wholly within that margin; the aggregate
  confidence interval lies wholly within that margin.
- **Harmful** — the symmetric reverse of Beneficial.
- **Inconclusive** — any other outcome.

TDI-5.4A is a preregistered classification and is **not** forced to produce
any particular result. The scientifically informative outcomes are
symmetric: *Equivalent* would show the O₁ redundancy is not an artifact of
linearity; *Beneficial* would show the linear Equivalent finding of
TDI-5.3C masked a nonlinear O₁ contribution.

## 12. Criterion TDI-5.4B — horizon-invariance

Repeat the N12-vs-N2 classification of Section 11, unchanged, at each
secondary horizon **U₃, U₄, U₅, U₈**, yielding one 4-way classification per
horizon.

Define the primary-horizon class `c₆` from TDI-5.4A. TDI-5.4B reports, as
its preregistered summary:

1. the four secondary-horizon classifications;
2. `horizons_matching_primary_class` — the count of secondary horizons
   whose classification equals `c₆`;
3. `invariant` — true iff all four secondary horizons share the same
   classification as `c₆`.

TDI-5.4B makes **no** success/failure claim beyond reporting these
pre-declared quantities: it is a preregistered descriptive criterion, not
a pass/fail gate, because either invariance or a horizon-dependent
transition is a legitimate, interpretable scientific outcome.

## 13. Exploratory analyses

The following are exploratory only and cannot modify any TDI-5.4 criterion:

1. the linear layouts B0/B1/B2/B12 and BD on the TDI-5.4 populations;
2. per-term coefficient inspection of N2 and N12;
3. ridge sensitivity at lambda 0.1, 1.0, 10.0 on N2/N12;
4. reconstructed-O-space prediction;
5. an OOD (width-5/6) nonlinear probe, if generated separately in future.

## 14. Operational activation and full-run entrypoint contract

Identical in spirit to TDI-5.3 Section 5. The v54 evaluator exposes exactly
three modes:

    --termination-smoke
    --preflight
    --full

A bare, no-argument invocation must refuse to run. `--termination-smoke`
uses only bounded tiny data and produces no result artifacts. `--preflight`
performs no scientific generation: it verifies the full frozen
configuration (all 12 seed reservations, all expected counts, all bootstrap
constants), verifies that the full pipeline is wired to `--full`, prints all
TDI-5.4 and ancestor identities and the exact real-run command, and exits
without producing a result.

`--full` requires the exact confirmation environment variable:

    TDI54_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI54_FREEZE_RULE

Without that exact value, `--full` must fail before any generation, fitting
or bootstrap. The confirmation check is a pure function of the environment
value, unit-testable without starting the experiment. No TDI-5.4 commit,
test, or CI run may supply the token. The full run is a deliberate, one-time
human action; the authoring agent must never invoke `--full` with the real
token.

## 15. Required raw output

Inherited from TDI-5.2 Section 17 with TDI-5.4 identities: git commit;
compiler/Cargo versions; v54 evaluator SHA-256; TDI-5.4 preregistration
SHA-256; TDI-5.4 scientific-manifest SHA-256; the frozen-ancestor hashes;
all frozen constants; the seed-block definitions; requested/accepted/
rejected/attempted counts; rejection counts by reason; final exclusive
seeds; generation budgets; target scalers; **N2 and N12 model
coefficients** (including nonlinear-term coefficients); all metrics; all
bootstrap intervals; block-level and aggregate criteria; the TDI-5.4A
classification; the TDI-5.4B per-horizon classifications and invariance
summary; deterministic termination diagnostics.

## 16. Determinism

Inherited from TDI-5.2 Section 18. Candidate generation, seed consumption,
exclusions, preprocessing, **nonlinear-term construction**, model fitting,
bootstrap sampling, aggregation, metric calculation, iteration order,
scientific-value formatting and final criteria are deterministic functions
of committed constants. Wall-clock timestamps are reproduction metadata
only and must not affect any scientific result.

## 17. Reproduction requirements

The TDI-5.4 reproduction script must satisfy every requirement of TDI-5.2
Section 19 / TDI-5.3 Section 8 (refuse a dirty repository; verify all frozen
hashes including TDI-5.1/5.2/5.3 and TDI-5.4; refuse an existing partial or
complete result; acquire an exclusive lock; compile offline in release mode;
execute the evaluator exactly once with `--full`; capture complete output;
verify all final criterion lines; write metadata and a completion marker;
hash all artifacts; make final artifacts read-only), plus: it must require
the exact confirmation variable before invoking the evaluator, and must
refuse to run over an existing TDI-5.4 result.

## 18. Interpretation boundaries

A TDI-5.4 result establishes the (non)contribution of O₁ conditional on O₂
**within the frozen generator and the specific preregistered nonlinear
basis of Section 5** (quadratic + single pairwise interaction), replicated
across three seed blocks. It does **not** establish: sufficiency under
arbitrary nonlinear families (deep networks, trees, kernels); causal
effects; universal validity across dynamical systems; arbitrary-width
calibration; implementation-independent replication; or external empirical
validity. The TDI-5.4A classification and the TDI-5.4B horizon summary may
not be rewritten after observing the full result.

## 19. Freeze rule

After the TDI-5.4 preregistration, v54 evaluator, manifests, reproduction
script and CI workflow are frozen: scientific code must not change;
constants must not change; seed blocks must not change; criteria and the
nonlinear basis must not change; no full run may begin before all frozen
hashes pass (TDI-5.1, 5.2, 5.3 and 5.4); any scientific-code defect
discovered after freezing requires a new experiment identifier — TDI-5.4
may not be silently patched.
