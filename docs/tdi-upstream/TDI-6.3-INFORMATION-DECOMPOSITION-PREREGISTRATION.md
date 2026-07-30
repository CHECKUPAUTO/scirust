# TDI-6.3 — Information Decomposition: How Do O₁ and O₂ Jointly Inform U_h?

## Preregistration

This document is the frozen preregistration for TDI-6.3. Once its SHA-256
manifest, the v63 evaluator, the reproduction script, the CI workflow and the
bounded tests are committed, this design is frozen under the Section 22 freeze
rule: no scientific constant, tolerance, FP-regime declaration, seed block,
information-decomposition definition, or criterion may change without a new
experiment identifier. Freezing the design does not authorize a run; the real
experiment begins only as the deliberate one-time human action of Section 17.
The authoring agent never invokes `--full`.

## 1. Experimental status, provenance, and the single changed factor

TDI-6.3 is a new confirmatory experiment derived from the completed and merged
**TDI-5.6** result, and is the third experiment under the **TDI-6** identifier
(after TDI-6.1's literal spectral gap and TDI-6.2's nonlinear model), reserved
for this purpose since TDI-6.1 Section 21 ("an information-decomposition (PID)
account (TDI-6.3)") and named in the working roadmap as turning "the 'which
overlap carries what' question from ridge coefficients into an
information-theoretic decomposition."

Every TDI-5.x/6.x result to date answers the **same shape of question**: does
`{O_1, O_2}` reduce a fitted ridge model's error **beyond** some baseline
feature set (contraction, exact spectral moments, literal spectral gap,
nonlinear basis, generator family, width)? TDI-6.3 asks a **structurally
different** question, not another ablation: **how is the predictive
information that `{O_1, O_2}` jointly carry about `U_h` distributed between
them** — is it redundant (both convey the same information), unique to one
overlap alone, or synergistic (visible only when both are known together)?
This reframes "which overlap carries what" as a formal **two-source partial
information decomposition (PID)**, computed directly from the empirical joint
distribution of `(O_1, O_2, U_h)` — not from a fitted model's coefficients.

**The single changed factor from TDI-5.6 is the confirmatory analysis
machinery itself**: TDI-6.3 replaces the ridge-model / four-way-classifier
ablation machinery entirely with a closed-form information decomposition. It
inherits TDI-5.6's candidate generation, target construction, and exact
descriptor computation **verbatim** — including the exact contraction
descriptors δ, δ̄ and the exact spectral moments s₂, s₃, which are generated
for structural fidelity to the frozen ancestor and reported as descriptive
context (Section 4.5) but consumed by **no** TDI-6.3 criterion, which uses
only the early overlaps `O_1, O_2` and the target `U_h`.

Because the decomposition itself is computed from `f64` covariances via
logarithms and matrix determinants — transcendental / non-rational operations
— TDI-6.3 is a **non-exact** TDI-6-track experiment, under the same kind of
explicit non-exact determinism discipline TDI-6.1 introduced (Section 8):
declared FP regime, declared tolerances, two independent computation methods
cross-validated against each other, and tolerance-based (not byte-exact)
reproduction. The relaxation is localized: only the decomposition's own
computation is non-exact; candidate generation, target construction and the
exact descriptors remain bit-exact, inherited unchanged.

Frozen ancestor identities (verified at runtime and in CI): the v56 evaluator,
the TDI-5.6 preregistration, and the full frozen chain **TDI-5.1 → TDI-5.8,
TDI-6.1, TDI-6.2, TDI-6.5** (every ancestor evaluator and preregistration hash)
are verified before any generation. Section 1.3 distinguishes this full
verified chain from TDI-6.3's actual scientific/code ancestor.

No full TDI-6.3 run may begin before all of the following are committed and
frozen: this preregistration; the final evaluator; the evaluator SHA-256
manifest; the scientific-code SHA-256 manifest; the deterministic reproduction
script; the dedicated CI workflow; bounded unit and termination tests.

### 1.1 Why TDI-6.3 is being built now, after TDI-6.1, TDI-6.2, TDI-6.5 and TDI-5.8

TDI-6.3's identifier was **reserved at design time, not assigned at build
time**. `docs/TDI-FORWARD-PROGRAM-ROADMAP.md` (an unfrozen design document, not
a preregistration) named four planned TDI-6 sub-experiments in its Section 2
("Track B"): 6.1 (literal spectral gap), 6.2 (nonlinear model families), 6.3
(information decomposition / PID — "turns the 'which overlap carries what'
question from ridge coefficients into an information-theoretic
decomposition"), 6.4 (a causal probe). Its Section 3 ("Recommended sequence")
explicitly left the order among them open: *"Then TDI-5.8 / TDI-6.2 / TDI-6.3 /
TDI-6.4 as the questions each prior result sharpens."* The roadmap was a slot
reservation, not a build queue — TDI-6.3 has been waiting on that reserved slot
since TDI-6.1 was designed, not invented or renumbered now for convenience.

TDI-6.5 is not in that original roadmap at all — it did not exist as a concept
until TDI-6.1's own real confirmatory result came back. TDI-6.5's Section 1
states its provenance directly: it is "derived from the completed and merged
TDI-6.1 result," running exactly the generator-family-robustness control that
"TDI-6.1 Section 21 named ... explicitly" as an open limitation. Because that
question only became askable once TDI-6.1's result existed, and because
answering it was a mechanical recombination of two *already-frozen* designs
(TDI-5.7's generator-family machinery and TDI-6.1's literal-spectral
descriptors — Section 1 of TDI-6.5 is explicit that it "combines two frozen
designs without introducing any new scientific mechanism"), it required no new
mathematical apparatus and was fast to design, build and freeze.

TDI-6.3 depends on none of TDI-6.1, TDI-6.2 or TDI-6.5's results — its sole
scientific ancestor is TDI-5.6 (Section 1.3), frozen since 2026-07-22
(commit `6a9aaa3`, merged `b650d51`), before TDI-6.1 was even drafted. What
TDI-6.3 needed instead was a genuinely new closed-form apparatus (a two-source
partial information decomposition, Sections 5-6) with its own numerically
verified derivation — design and verification work orthogonal to, and slower
than, the mechanical ablation-machinery derivations behind 6.1/6.2/6.5/5.8.
Building the faster, already-well-understood derivations first and the
novel-apparatus experiment once it was fully designed and independently
verified is an efficiency ordering, not a scheduling accident, an
abandonment, or a re-numbering of a preregistered experiment.

### 1.2 Relationship to TDI-6.5: an independent branch, not a sequential revision

TDI-6.5 (frozen; PR #31, merge commit `9700c43`) and TDI-6.3 derive from
different, disjoint points in the experiment tree — TDI-6.5 from TDI-6.1 (plus
TDI-5.7), TDI-6.3 directly from TDI-5.6 — and neither reads, extends, patches,
or reinterprets the other:

- TDI-6.3's evaluator (`tdi-independent-overlap-ablation-v63.rs`) is
  transplanted from TDI-5.6's evaluator (`v56.rs`); it shares no code, seed
  range, criterion, or scientific claim with TDI-6.5's evaluator (`v65.rs`).
- TDI-6.3 makes no statement about, and does not test, TDI-6.5's subject
  (whether the literal-spectral control replicates across generator
  families). TDI-6.3 uses the single base generator and TDI-5.6's original
  population structure, not TDI-5.7/6.5's generator families.
- The only place TDI-6.5 appears anywhere in the TDI-6.3 artifacts is as one
  of two hash entries in TDI-6.3's own rolling `SCIENTIFIC-CODE.sha256`
  manifest (Section 1.3) — a repository chain-of-custody entry every
  experiment in this program includes for whichever two experiments were
  most recently merged at its own build time, independent of scientific
  relationship. (TDI-6.5's own manifest, symmetrically, cites TDI-6.1 and
  TDI-6.2 for the same bookkeeping reason, despite deriving its actual design
  from TDI-6.1 alone.) This citation is chain-of-custody bookkeeping, not a
  scientific dependency claim, and must not be read as one.
- TDI-6.5's preregistration, evaluator, manifests and (once its own `--full`
  run is performed) results are untouched by the TDI-6.3 cycle. Every one of
  TDI-6.5's three frozen hashes was independently re-verified immediately
  before TDI-6.3 work resumed this session, with no drift.

Building TDI-6.3 now therefore does not rewrite, reinterpret, or in any way
revise TDI-6.5: the two stand as independent, permanently frozen siblings
under the shared TDI-6 identifier, exactly as TDI-6.1 and TDI-6.2 do.

### 1.3 The reproducibility chain: scientific ancestor vs. rolling-manifest ancestors vs. full verified chain

Three distinct notions of "ancestor" apply to TDI-6.3, and this preregistration
keeps them separate rather than collapsing them into one list:

| Role | Experiment(s) | Why |
|---|---|---|
| **Direct scientific/code ancestor** | TDI-5.6 (`v56.rs`, frozen `6a9aaa3`, merged `b650d51`, PR #17) | TDI-6.3's population generation, target construction, and exact descriptors are transplanted verbatim from v56; the preregistration's single changed factor (Section 1) is defined relative to TDI-5.6 specifically. |
| **Rolling scientific-code manifest ancestors** | TDI-5.8 (`v58.rs`, frozen `18671ad`, merged `0a3cf12`, PR #32); TDI-6.5 (`v65.rs`, frozen `f896ea2`, merged `9700c43`, PR #31) | The two most-recently-merged experiments at TDI-6.3's build time, included in `docs/TDI-6.3-SCIENTIFIC-CODE.sha256` per this program's established rolling-manifest convention (every experiment's manifest chains to the two immediately before it, regardless of derivation relationship), so the manifest also attests to the integrity of the whole chain-so-far. Not a claim that TDI-6.3 derives code or design from either. |
| **Full verified ancestor chain** | TDI-5.1 → TDI-5.8, TDI-6.1, TDI-6.2, TDI-6.5 (eleven experiments; every evaluator + preregistration + scientific-code hash) | Re-verified by the TDI-6.3 reproduction script (Section 20) and CI before any generation, exactly as every prior experiment's script re-verifies its own complete ancestor set. This establishes non-regression of the whole frozen program, not a code-derivation claim. |

TDI-5.6 is itself the frozen terminus of the TDI-5.1 → TDI-5.6 mechanical
derivation chain (each a "single changed factor" step from the previous); that
chain is transitively part of TDI-6.3's scientific lineage through TDI-5.6, and
is covered by the full verified chain above.

## 2. Research questions

Within the frozen candidate machinery, the frozen exact descriptors, and the
new information-decomposition machinery:

1. at the focal horizons U₃ and U₆, how is the joint mutual information
   `I(U_h; {O_1,O_2})` partitioned into **Redundancy**, **Unique(O_1)**,
   **Unique(O_2)** and **Synergy** (criterion **TDI-6.3A**, primary,
   descriptive)?
2. how does this four-way decomposition **evolve across the dense horizon
   grid** U₃…U₈ (criterion **TDI-6.3B**, descriptive)?
3. is the **dominant** component of the decomposition **consistent across
   three independent seed blocks** at each focal horizon (criterion
   **TDI-6.3C**, descriptive replication check)?

TDI-6.3 does **not** re-test whether `{O_1,O_2}` improves on any baseline
(settled by 5.2 … 6.5); it does not use ridge regression, feature layouts, or
the four-way Beneficial/Equivalent/Harmful/Inconclusive classifier at all.
None of its criteria are pass/fail classifications — a partial information
decomposition has no natural "success" or "failure" outcome, and none is
forced here.

## 3. Relationship to the frozen ancestors

**Inherited unchanged** from TDI-5.2 … 5.6 (frozen; bit-exact; not re-derived):
the exact candidate analysis and per-candidate exclusion criteria; the single
base generator (`build_system` over uniform non-empty successor masks, widths
3 and 4); observation geometry and target geometry `U_h = -log2(1 - O_h)`; the
13 structural/entropic baseline variables; the two early overlaps `O_1, O_2`;
the two exact contraction descriptors δ, δ̄; the two exact spectral moments
s₂ = trace(P²), s₃ = trace(P³); the dense horizon grid `H = {3,4,5,6,7,8}` and
focal horizons U₃, U₆; the deterministic per-width generation budgets and the
four-population-per-block seed-reservation structure (training-w3, holdout-w3,
training-w4, holdout-w4); and the `tdi-core` exact primitives.

**New in TDI-6.3** (the only substantive additions, none touching generation):
the two-source partial information decomposition of `U_h` with respect to
`(O_1, O_2)` (Sections 5–7); its non-exact determinism discipline (Section 8);
fresh, independent seed blocks (Section 10); a covariance-resampling bootstrap
for the four PID components (Section 11); and the criteria TDI-6.3A / B / C
(Sections 14–16).

**Dropped relative to TDI-5.6.** No feature layouts (CK/SK/SKT), no ridge
fitting, no target scalers used for prediction, no MSE-based four-way
classifier, no baseline-vs-challenger comparison of any kind. TDI-6.3 computes
one thing: the information decomposition of `(O_1, O_2)` about `U_h`.

## 4. Design notes and confirmatory integrity

### 4.1 Why a two-source PID, and why now

The exact series (5.2 → 6.5) repeatedly asks "does `{O_1,O_2}` help beyond
baseline X?" and repeatedly finds yes. It never asks how the *two* overlaps
relate to *each other* in what they convey about `U_h`: do they duplicate one
another (redundancy), each carry an independent slice (uniqueness), or only
become informative in combination (synergy)? Ridge coefficients cannot answer
this — a large coefficient on `O_1` and a small one on `O_2` do not distinguish
"O_1 is more informative" from "O_1 and O_2 are highly redundant and the fit
arbitrarily favors one." A formal PID answers this directly and is the
question TDI-6.1's Section 21 and the working roadmap named for TDI-6.3.

### 4.2 Choice of PID definition: Gaussian / Minimum Mutual Information (MMI)

Partial information decomposition is an active research area with several
competing, non-equivalent definitions (Williams & Beer's original `I_min`;
Bertschinger et al.'s `I_BROJA`; Ince's `I_ccs`; Barrett's MMI). TDI-6.3
commits to a single, fully specified measure and does not claim it is
canonical:

- **Redundancy is the Minimum Mutual Information (MMI) measure** (Barrett,
  *Phys. Rev. E* 91, 052802, 2015): `Red(U; O_1, O_2) = min(I(U;O_1),
  I(U;O_2))`.
- **The joint distribution of `(O_1, O_2, U_h)` is treated under a Gaussian
  (second-moment-only) working model**: every mutual information in this
  design is the closed-form differential mutual information of a
  multivariate-normal model fitted to the sample covariance matrix, **not** a
  nonparametric or discretized estimate of the true information-theoretic
  quantities. This is a stated modeling choice, not an empirical claim that
  `(O_1, O_2, U_h)` are jointly Gaussian; if the true joint distribution has
  higher-order structure (skew, heavy tails, nonlinear dependence) beyond its
  covariance, the reported decomposition characterizes that covariance
  structure, not necessarily the "true" information content (Section 18).

This combination is chosen, not merely convenient: **Barrett (2015) proves
that for jointly Gaussian systems, MMI redundancy yields a mathematically
consistent decomposition in which Redundancy, Unique(O_1), Unique(O_2) and
Synergy are all guaranteed non-negative** — a property that does not hold in
general for other redundancy measures, or for MMI itself under non-Gaussian /
naively-discretized empirical distributions. Combined with the fact that the
whole decomposition is closed-form (Section 6) and needs no discretization
(bin-count) parameter — an arbitrary, un-preregistered-feeling choice a
binned/discrete PID would require — the Gaussian/MMI combination is the most
tractable, deterministic, and defensible choice available for a frozen,
reproducible preregistration.

### 4.3 Unconditional scope (no baseline conditioning)

The decomposition is computed on the **unconditional** joint distribution of
`(O_1, O_2, U_h)` — it does not condition on, or control for, the exact
contraction/spectral descriptors the ablation experiments used as baselines.
Conditional / interaction-information generalizations of PID exist but are
substantially more complex and contested in the literature; TDI-6.3 keeps to
the well-established two-source, unconditional case. This is a genuine scope
boundary (Section 18): TDI-6.3 characterizes how `{O_1,O_2}`'s *own* joint
information about `U_h` is structured; it does not ask whether that
information is *redundant with the exact baseline descriptors* (that is what
the ablation experiments 5.5 → 6.5 already established).

### 4.4 Invariance to standardization (a design simplification, not an assumption)

Differential mutual information between two Gaussian-model variables is a
function of their **correlation only** and is invariant to any invertible
affine (location–scale) rescaling applied to either variable individually
(mean-centering and positive rescaling do not change a Pearson correlation
coefficient). Consequently, computing the decomposition in standardized-U
space versus raw reconstructed-O space, or on raw `O_1, O_2` versus any
rescaling of them, yields **identical** PID values. TDI-6.3 computes on the
same raw, unstandardized `O_1, O_2` and `U_h` values recorded during
generation; no target scaler or feature standardization is needed or used.

### 4.5 Population re-use without a train/holdout split

TDI-5.6's population structure (three blocks, each with training-w3/holdout-w3
/training-w4/holdout-w4 populations) is reused **verbatim** for its seed
reservations and generation budgets, but the training/holdout split served
TDI-5.6's out-of-sample predictive-error testing, which TDI-6.3 does not do:
TDI-6.3 estimates a joint covariance structure directly, and using all
available records maximizes the precision of that estimate. **All records
generated for a block (training and holdout, widths 3 and 4 combined) are
pooled** before computing that block's decomposition; there is no fitting/
prediction split. This is a re-use-for-consistency choice, not a scientific
requirement of the analysis.

### 4.6 The exact descriptors are generated but not consumed

Because candidate generation is inherited verbatim from TDI-5.6, every record
still carries the 13 baseline variables and the exact contraction/spectral
descriptors δ, δ̄, s₂, s₃. TDI-6.3 reports their pooled means (Section 4.5:
all four populations of a block, not a holdout-only subset) as plain
descriptive context (Section 12) — exactly as TDI-6.5's Section 19 (criterion
TDI-6.5D) diagnostic did for its own unused-by-criteria descriptors — but
**no** TDI-6.3 criterion reads them; only `O_1, O_2` and `U_h` feed the
decomposition.

## 5. The two-source partial information decomposition (definitions)

Let `T = U_h` (the target at a given horizon) and `S_1 = O_1`, `S_2 = O_2` (the
two early overlaps), all real-valued. Under the Gaussian working model
(Section 4.2), for any two jointly-Gaussian-modeled vectors `X` (dimension
`p`) and `Y` (dimension `q`) with covariance blocks `Σ_X`, `Σ_Y` and joint
covariance `Σ`:

    I(X; Y) = 0.5 · log2( det(Σ_X) · det(Σ_Y) / det(Σ) )

Instantiated for the scalar target against each source alone, and against the
pair:

    I(T; S_1)      = -0.5 · log2(1 - ρ_{T,S1}²)
    I(T; S_2)      = -0.5 · log2(1 - ρ_{T,S2}²)
    I(T; {S_1,S_2}) = 0.5 · log2( var(T) · det(Σ_{S1,S2}) / det(Σ_{T,S1,S2}) )

where `ρ_{T,Si}` is the Pearson correlation of `T` and `S_i`, `Σ_{S1,S2}` is
the 2×2 covariance of `(S_1,S_2)`, and `Σ_{T,S1,S2}` is the full 3×3 covariance
of `(T,S_1,S_2)`. All three mutual informations are non-negative by
construction (each is `-0.5·log2(1 - R²)` for the relevant squared
correlation / multiple-correlation coefficient, and `R² ∈ [0,1)` whenever the
covariance matrix is non-degenerate).

The two-source PID lattice (Redundancy, Unique(S_1), Unique(S_2), Synergy)
under MMI redundancy:

    Red   = min( I(T;S_1), I(T;S_2) )
    Un_1  = I(T;S_1) - Red
    Un_2  = I(T;S_2) - Red
    Syn   = I(T;{S_1,S_2}) - I(T;S_1) - I(T;S_2) + Red

satisfying the PID identity `Red + Un_1 + Un_2 + Syn = I(T; {S_1,S_2})` exactly
(the defining partition of the joint mutual information), and, by Barrett's
result (Section 4.2), all four terms are non-negative whenever the
Gaussian-model mutual informations above are well-defined (non-degenerate
covariance). All quantities are reported in **bits** (`log2`).

## 6. Computation method and the declared tolerance

**Canonical method (method 1, the frozen feature path).** Compute the sample
covariance matrix of `(T, S_1, S_2)` (and its 2×2 and 1×1 sub-blocks) from
centered sums in `f64`, single-threaded, fixed accumulation order. Compute
each required `log2(det(·))` via a **Cholesky decomposition** `Σ = L Lᵀ`
(`log(det(Σ)) = 2·Σᵢ log(L_ii)`), the standard numerically stable method for
the log-determinant of a symmetric positive-definite matrix; a failed Cholesky
factorization (non-positive-definite covariance — degenerate/collinear data)
is a declared, deterministic failure signaled by `NaN`, not silently
substituted. `I(T;S_1)` and `I(T;S_2)` use the 1-D specialization
`-0.5·log2(1-ρ²)` directly (no Cholesky needed for a 2×2 case; both are
algebraically identical to the general log-det formula).

**Cross-check method (method 2, computed on both the real `--full` path and in
tests).** Compute the same three mutual informations via the classical
**multiple-correlation-coefficient** identity, using a genuinely different
arithmetic path from pairwise Pearson correlations alone:

    R²_{T|S1,S2} = (ρ_{T,S1}² + ρ_{T,S2}² - 2·ρ_{T,S1}·ρ_{T,S2}·ρ_{S1,S2})
                   / (1 - ρ_{S1,S2}²)
    I(T; {S_1,S_2}) [method 2] = -0.5 · log2(1 - R²_{T|S1,S2})

This is mathematically equivalent to the method-1 log-det formula but computed
through an entirely different sequence of arithmetic operations (a rational
expression in three pairwise correlations, rather than a matrix
factorization), so agreement between the two methods is a genuine
implementation cross-check, not a tautology — it catches coding errors
(indexing, sign, formula transcription) the way TDI-6.1's eigensolver
cross-checks catch eigensolver bugs.

**Declared tolerances (frozen constants).** Cross-method agreement:
`PID_CROSS_METHOD_TOLERANCE = 1e-9` bits, on each of `I(T;S_1)`, `I(T;S_2)`,
`I(T;{S_1,S_2})`. Cholesky degeneracy floor: a covariance matrix is treated as
degenerate (computation excluded, `NaN`-propagated) if any pivot in the
factorization is `<= 1e-12`; this is expected never to trigger on the real
populations (`O_1, O_2, U_h` all have genuine variance throughout the frozen
generator), and the bounded tests exercise the degenerate path directly with a
synthetic collinear input.

**Floating-point regime**, declared as in TDI-6.1 Section 12: IEEE-754
binary64 (`f64`), single-threaded, fixed operation order (no parallel
reduction, no FMA reordering). Reproduction is tolerance-based: a faithful
re-run on the reference toolchain/architecture reproduces the result log
byte-for-byte; across architectures the raw bit values may differ in the last
digits, but because the descriptive quantities are reported to a fixed
precision and the qualitative flags (dominant component, cross-block
consistency) are robust to last-digit drift by a wide margin (typical PID
components differ by far more than `1e-9` bits in practice), the reported
decomposition and flags reproduce exactly.

## 7. Populations

Reused verbatim from TDI-5.6 Section 8: single generator, three fresh seed
blocks, no OOD populations. For each of the three seed blocks:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |

Accepted records per block: **40,000**; total: **120,000**. All four
populations of a block are **pooled** (Section 4.5) before computing that
block's decomposition; there is no fitting/prediction split. Generation
budgets are inherited unchanged from TDI-5.2 Section 7.

## 8. Independent seed blocks (fresh)

Three deterministic, pairwise-disjoint seed blocks, **disjoint from every
prior block** (TDI-5.7 ≤ 2.53×10⁹; TDI-6.1 3.0–3.23×10⁹; TDI-6.2 4.0–4.23×10⁹;
TDI-6.5 5.0–6.13×10⁹; TDI-5.8 7.0–7.81×10⁹; TDI-6.3 starts at 8.0×10⁹):

    base(b) = 8_000_000_000 + b · 100_000_000   for block index b ∈ {0,1,2}

and the four populations start at `base + {0, 10, 20, 30} · 1_000_000`
(training-w3, holdout-w3, training-w4, holdout-w4). Explicitly the training-w3
bases: block 0 → 8,000,000,000; block 1 → 8,100,000,000; block 2 →
8,200,000,000. Twelve total reservations. The evaluator verifies disjointness
of all consumed ranges at runtime.

## 9. Metrics and standardized-U primacy

TDI-6.3 does not use the standardized-U / reconstructed-O metric machinery of
prior experiments (no MSE, MAE, R², Spearman, bias, calibration — those are
predictive-accuracy metrics for a fitted model, and TDI-6.3 fits no model). Its
metrics are the covariances, correlations, mutual informations and PID
components of Sections 5–6, computed directly on the raw `(O_1, O_2, U_h)`
values (Section 4.4 establishes this is invariant to any standardization
convention, so none is needed).

## 10. Focal horizons and grid

Inherited: the dense grid `H = {3,4,5,6,7,8}` and the focal horizons **U₃**
and **U₆**. TDI-6.3A classifies at the focal horizons; TDI-6.3B reports the
decomposition across the full grid.

## 11. Deterministic bootstrap

Because TDI-6.3 has no baseline-vs-challenger comparison, its bootstrap
resamples **records** (not paired predictions) to build a resampling
distribution of the four PID components. For each block, resample that
block's own pooled records (Section 4.5) with replacement; for the
aggregate, resample uniformly over the full pooled records of all three
blocks combined — a **plain** (non-stratified) resample, not the
block-stratified aggregate bootstrap of the ablation experiments (5.2 …
6.5): those experiments' aggregate stratified by block because their
aggregate metric was a paired-prediction comparison in which each block
contributed a fixed-size fold; TDI-6.3's aggregate is a single covariance
estimated directly from the pooled records (Section 4.5), which a plain
uniform resample over that same pooled set estimates correctly, with no
per-block fold to preserve. Resample with replacement (**4,000
replicates**, inherited replicate count), recompute the covariance matrix
and the full decomposition (method 1 only; the bootstrap does not re-run
the cross-check) on each replicate, and report the two-sided 95%
percentile interval for each of Red, Un_1, Un_2, Syn, and for the joint
mutual information `I(T;{S_1,S_2})`. The point-estimate proportions of
`I(T;{S_1,S_2})` (Section 5) are reported alongside these intervals, but
are not themselves bootstrapped: a ratio-of-two-resampled-quantities
interval is a materially different (and more involved) construction than
the frozen `PidBootstrapIntervals` core module computes, so extending to
proportion-level intervals is out of scope for this preregistration.
Bootstrap seeds are fresh, in the
`0x5444_4936_3300_…` (`TDI6`/`33` = ".3") range, disjoint from every prior
bootstrap seed:

    block seed (block b)  : 0x5444_4936_3300_0000 + b + 1   (…0001/0002/0003)
    aggregate seed         : 0x5444_4936_3300_4700

## 12. Descriptor diagnostic (context only, no criterion)

For each block, report the all-pooled means (Section 4.5) of the four exact
descriptors δ, δ̄, s₂, s₃ (Section 4.6) — plain descriptive context, consistent
with every prior experiment's practice of reporting the full frozen descriptor
set, but consumed by **no** TDI-6.3 criterion.

## 13. Criterion TDI-6.3A — the decomposition at the focal horizons (primary, descriptive)

At each focal horizon **U₃** and **U₆**, for each of the three blocks and for
the pooled aggregate, report: `I(T;S_1)`, `I(T;S_2)`, `I(T;{S_1,S_2})`;
Redundancy, Unique(O₁), Unique(O₂), Synergy (bits); each component's proportion
of `I(T;{S_1,S_2})`; the 95% bootstrap interval for each component (Section
11 — proportions are reported as point estimates only, not bootstrapped);
and the method-1/method-2 cross-check agreement. TDI-6.3A further
reports, per block and for the aggregate, the **dominant component** — whichever
of {Redundancy, Unique(O₁), Unique(O₂), Synergy} has the largest aggregate
point estimate.

TDI-6.3A is a preregistered **descriptive** summary; it is not a pass/fail
classification, and no outcome (redundancy-dominant, synergy-dominant, or any
other pattern) is treated as a success or failure.

## 14. Criterion TDI-6.3B — decomposition across the dense grid (descriptive)

Evaluate the full decomposition (Section 13's quantities, aggregate only) at
every horizon of the dense grid `H = {3,4,5,6,7,8}`. Report, as a compact
per-horizon table: the four components and their proportions, and the dominant
component per horizon. Report whether the dominant component is **stable**
(identical at every horizon) or **shifts**, and if it shifts, at which
horizon(s). Purely descriptive; no success/failure claim.

## 15. Criterion TDI-6.3C — cross-block consistency (descriptive replication check)

At each focal horizon, compare the **per-block** dominant component (computed
independently on each of the three blocks, Section 13) across all three
blocks. Report `cross_block_dominant_component_consistent`: true iff all three
blocks agree on the dominant component at that horizon; if not, name the
blocks that disagree and their respective dominant components. This is the
replication-style check every prior experiment performs (its own analogue of
"3/3 blocks confirm"), adapted to a descriptive decomposition rather than a
classification; it is not itself a pass/fail criterion.

## 16. Operational activation and full-run entrypoint contract

The v63 evaluator exposes exactly three modes: `--termination-smoke`,
`--preflight`, `--full`. A bare invocation refuses to run. `--termination-smoke`
uses only bounded tiny data and produces no result artifacts. `--preflight`
performs no scientific generation: it verifies the full frozen configuration
(all 12 seed reservations, all expected counts, all bootstrap constants, the
declared tolerances and FP regime), verifies that the full pipeline is wired
to `--full`, prints all TDI-6.3 and ancestor identities and the exact real-run
command, and exits without a result.

`--full` requires the exact confirmation environment variable:

    TDI63_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI63_FREEZE_RULE

Without that exact value, `--full` fails before any generation or computation.
The confirmation check is a pure function of the environment value,
unit-testable without starting the experiment. No TDI-6.3 commit, test or CI
run supplies the token. The full run is a deliberate, one-time human action;
the authoring agent never invokes `--full` with the real token.

## 17. Required raw output

git commit; compiler/Cargo versions; the declared FP regime and all
tolerances; the v63 evaluator SHA-256; the TDI-6.3 preregistration and
scientific-manifest SHA-256; the full frozen ancestor chain (TDI-5.1 → 5.8,
TDI-6.1, TDI-6.2, TDI-6.5 — Section 1.3); all frozen constants; the seed-block definitions; per-block
requested/accepted/rejected/attempted counts; rejection counts by reason;
final exclusive seeds; the per-block and aggregate covariance matrices and
correlations; the method-1/method-2 cross-check table; the TDI-6.3A focal
decomposition (per block and aggregate, both focal horizons); the TDI-6.3B
dense-grid decomposition table; the TDI-6.3C cross-block consistency report;
the Section 12 descriptor-diagnostic table; deterministic termination
diagnostics.

## 18. Interpretation boundaries

A TDI-6.3 result characterizes the two-source partial information
decomposition of `(O_1, O_2)` about `U_h`, **under an explicit Gaussian
(second-moment-only) working model** and the **MMI redundancy measure**, on
the single base generator, within the frozen exact machinery. It does **not**
establish: that `(O_1, O_2, U_h)` are actually jointly Gaussian (a modeling
choice, not an empirical finding); a decomposition under any other PID
definition (`I_BROJA`, `I_ccs`, discretized `I_min`, or others), which could
disagree with the MMI values reported here; a decomposition conditioned on, or
controlling for, the exact contraction/spectral baseline descriptors (Section
4.3) — that is the separate, already-settled ablation question of 5.5 → 6.5;
robustness across generator families (5.7/6.5) or widths (5.8); a causal
account (TDI-6.4); or external empirical validity. The TDI-6.3A / B / C
summaries may not be rewritten after observing the result.

## 19. Determinism

Candidate generation, seed consumption, exclusions, preprocessing, and the
exact contraction/spectral descriptor construction are deterministic functions
of committed constants, inherited unchanged from TDI-5.6. Covariance
accumulation, the Cholesky factorization, both mutual-information computation
methods, the PID lattice assembly, bootstrap resampling, and the descriptive
summaries are deterministic functions of the generated records and the frozen
constants of Sections 6, 8 and 11, under the fixed single-threaded FP regime
of Section 6. Wall-clock timestamps are reproduction metadata only.

## 20. Reproduction requirements

The TDI-6.3 reproduction script must satisfy every requirement of TDI-5.2
Section 19 / TDI-6.1 Section 20 (refuse a dirty repository; verify all frozen
hashes including TDI-5.1 … 5.8, TDI-6.1, TDI-6.2, TDI-6.5 and TDI-6.3 (Section
1.3); refuse an existing partial or
complete result; acquire an exclusive lock; compile offline in release mode;
execute the evaluator exactly once with `--full`; capture complete output;
verify all final criterion lines; write metadata and a completion marker; hash
all artifacts; make final artifacts read-only), plus: it must require the
exact confirmation variable before invoking the evaluator. The completion
check verifies (a) the result-log SHA-256 (byte-exact on the reference
toolchain/architecture) and (b) that the printed TDI-6.3A/B/C lines are
present — the tolerance-robust invariant of Section 6.

## 21. Deferred tracks

Robustness across generator families (settled by 5.7/6.5) and widths (5.8),
control against the literal spectral gap / mixing time (6.1) or a nonlinear
model (6.2), a causal account (TDI-6.4), and alternative PID definitions are
all out of scope for TDI-6.3, which is deliberately narrow: one PID definition,
one generator, the base widths, computed unconditionally.

## 22. Freeze rule

Once the SHA-256 manifests, the v63 evaluator, the reproduction script, the CI
workflow and the bounded tests are committed, this design is frozen: scientific
code must not change; constants, tolerances and the FP-regime declaration must
not change; the PID definition (Gaussian working model + MMI redundancy), the
seed blocks and the criteria must not change; no full run may begin before all
frozen hashes pass (TDI-5.1 … 5.8, 6.1, 6.2, 6.5 and 6.3); any scientific-code
defect
discovered after freezing requires a new experiment identifier — TDI-6.3 may
not be silently patched. The result classifications, once produced, are frozen
as reported.
