# TDI-6.5 — Generator-Family Robustness of the Literal-Spectral Control: Does the Overlap Signal Survive `1 − |λ₂|` and `τ_ε` *Across a Family of Generators*?

## Preregistration

This document is the frozen preregistration for TDI-6.5. Once its SHA-256
manifest, the v65 evaluator, the reproduction script, the CI workflow and the
bounded tests are committed, this design is frozen under the Section 24 freeze
rule: no scientific constant, tolerance, FP-regime declaration, seed block,
generator-family rule, spectral-descriptor definition, feature definition,
baseline or criterion may change without a new experiment identifier. Freezing
the design does not authorize a run; the real experiment begins only as the
deliberate one-time human action of Section 20. The authoring agent never
invokes `--full`.

## 1. Experimental status, provenance, and the single changed factor

TDI-6.5 is a new confirmatory experiment derived from the completed and merged
**TDI-6.1** result. It is **not** a continuation, patch, or reinterpretation of
TDI-5.1 … 5.7, TDI-6.1 or TDI-6.2, each of which remains frozen under its own
identifier.

TDI-6.1 established, on the **single base generator**, that the early overlaps
`{O_1, O_2}` retain predictive value beyond a baseline that already contains the
exact contraction descriptors (δ, δ̄), the exact spectral moments (s₂, s₃),
**and** the two *literal* non-exact spectral descriptors — the spectral gap
`g = 1 − |λ₂|` and the ε-mixing time `τ_ε` of the one-step Noop kernel
(criterion 6.1A was *Beneficial* at U₃ and U₆). TDI-6.1 Section 21 named its
first open limitation explicitly: it does **not** establish *robustness of the
literal-spectral control across the TDI-5.7 generator families* — a single base
generator was used, isolating the literal-spectral factor.

**TDI-6.5 runs exactly that control.** The single factor changed from TDI-6.1 is
the **generator**: the single base generator is replaced by the frozen TDI-5.7
family of four structurally distinct exact generators **F0–F3** (Section 9).
Everything else is inherited unchanged — the two non-exact spectral descriptors
and their non-exact determinism discipline from TDI-6.1, and the exact
mask-based family machinery from TDI-5.7. Because the *baseline* against which
the overlaps are tested is now the literal-spectral **GK** layout (not the exact
**SK** layout of TDI-5.7), TDI-6.5 is precisely *"TDI-5.7 asked against the
TDI-6.1 baseline"*: the confirmatory comparison becomes **GKT vs GK** within
each family, in place of TDI-5.7's SKT vs SK.

TDI-6.5 therefore combines two frozen designs without introducing any new
scientific mechanism: it adopts TDI-5.7's replication / heterogeneity /
transfer / descriptor-drift criteria template (Sections 17–19), now applied to
the GKT-vs-GK comparison, and TDI-6.1's localized non-exactness (only `g` and
`τ_ε` are non-exact; Section 13).

Frozen ancestor identities (verified at runtime and in CI): the v57 and v61
evaluators, the TDI-5.7 and TDI-6.1 preregistrations, and the full frozen chain
**TDI-5.1 → TDI-6.2** (every ancestor evaluator and preregistration hash) are
verified before any generation.

No full TDI-6.5 run may begin before all of the following are committed and
frozen: this preregistration; the final evaluator; the evaluator SHA-256
manifest; the scientific-code SHA-256 manifest; the deterministic reproduction
script; the dedicated CI workflow; bounded unit and termination tests.

## 2. Research questions

Within the frozen candidate machinery, the frozen exact descriptor set, the two
inherited non-exact spectral descriptors, and the frozen TDI-5.7 generator
family:

1. does the GKT-minus-GK marginal value of `{O_1, O_2}` — the overlaps' value
   beyond contraction, exact moments **and** the literal spectral gap + mixing
   time — classify **Beneficial in every generator family** at the focal
   horizons U₃ and U₆ (**replication**, criterion **TDI-6.5A**, primary)?
2. how **heterogeneous** is that effect size across families — is the aggregate
   relative-MSE reduction tightly clustered above the 2% margin, or does it
   swing widely / cross into Equivalence for some family (**heterogeneity**,
   criterion **TDI-6.5B**)?
3. does a model fitted on one family **transfer** to another at the same widths
   (**cross-generator transfer**, criterion **TDI-6.5C**, descriptive)?
4. how much do the six descriptors δ, δ̄, s₂, s₃, **and the literal `g`, `τ_ε`**
   themselves move across families — i.e. how demanding is the GK baseline in
   each family, and does the literal spectral structure vary across the family's
   distinct mixing regimes (**descriptor drift**, criterion **TDI-6.5D**,
   descriptive)?

TDI-6.5 does **not** re-open the exact-descriptor questions (5.5/5.6), the
generator question in the exact regime (settled by 5.7), the single-generator
literal-spectral control (settled by 6.1), or nonlinear model families (6.2). It
changes **only** the generator relative to 6.1, holding every feature, layout,
model, tolerance and non-exact discipline of 6.1 fixed, and reuses the exact
family rules of 5.7 unchanged.

## 3. Relationship to the frozen ancestors

**Inherited unchanged from TDI-6.1** (frozen; non-exact only in `g`, `τ_ε`; not
re-derived): the one-step Noop kernel `P` (Section 6); the two non-exact
spectral descriptors `g = 1 − |λ₂|` and `τ_ε` and their three-method
cross-validation (Sections 7–8); the layouts **CK / SK / GK / GKT** (Section
11); ridge `lambda = 1.0`; the paired + stratified-aggregate bootstrap; the
four-way ±2% classifier; the exact descriptors δ, δ̄, s₂, s₃; the 13
structural/entropic baseline variables; the two early overlaps O₁, O₂; the
non-exact determinism discipline (Section 13); and the `tdi-core` exact
primitives.

**Inherited unchanged from TDI-5.7** (frozen; still bit-exact; not re-derived):
the four deterministic successor-mask generator rules **F0–F3** (Section 9); the
per-family three-block population structure (Section 10); the transfer pair
**F0 → F1**; and the replication / heterogeneity / transfer / descriptor-drift
criteria template (Sections 17–19). The frozen `build_system` assembles every
family's candidate exactly as before.

**New in TDI-6.5** (the only additions, none a new scientific mechanism): the
*combination* of the TDI-6.1 GK baseline with the TDI-5.7 family — i.e. the
GKT-vs-GK comparison evaluated **independently within each family**; fresh,
pairwise-disjoint seed blocks (Section 12); fresh `TDI6`/`35`-marked bootstrap
seeds (Section 12); the confirmation guard `TDI65_CONFIRM_FULL_RUN`
(Section 20); and the criteria 6.5A / 6.5B / 6.5C / 6.5D (Sections 17–19)
applied to GKT-vs-GK.

## 4. Design notes and confirmatory integrity

### 4.1 Why the family × literal-spectral control is the right next step

TDI-6.1 refuted "TDI is just the literal spectral gap" — but only on one
generator. TDI-5.7 refuted "TDI is an artifact of one generator distribution" —
but only against the *exact* baseline. Neither closes the conjoined objection:
that on *some other* generator, the literal spectral gap and mixing time *would*
subsume the overlaps. TDI-6.5 tests the literal-spectral control on four
structurally distinct generators at once, so a family-wide *Beneficial* 6.5A is
the strongest generality statement the series can make within its scope.

### 4.2 The families span distinct mixing regimes, making the control demanding

The four frozen rules produce qualitatively different one-step kernels, and
therefore qualitatively different literal spectra and mixing times: F1 (sparse,
out-degree 1–2) yields near-deterministic rows, `|λ₂|` close to 1, a small gap
`g` and slow mixing (frequently saturating the mixing cap); F2 (dense,
near-complete branching) yields near-rank-one kernels, `|λ₂|` close to 0, a
large gap and fast mixing; F3 (Hamming-≤1 local with self-loops) yields a
structured hypercube-local spectrum. The GK baseline is thus stringent in
different ways across families — a single family could plausibly let `g`/`τ_ε`
subsume the overlaps even though the base generator did not. TDI-6.5A therefore
subjects the overlap signal to the literal-spectral control across the full
regime span. (These are design observations, not predictions; no criterion is
forced to any outcome.)

### 4.3 The non-exactness stays localized and minimal (unchanged from 6.1)

Only `g` and `τ_ε` are non-exact, computed in IEEE-754 `f64` under the Section
13 discipline. Generation (**including every family's successor-mask rule and
its `splitmix64` draw sequence**), target construction, ridge fitting and the
bootstrap remain bit-exact. Because the four-way classifier's margin is ±2%
relative MSE — many orders of magnitude larger than the f64 eigensolver
tolerance — the 6.5A/6.5B classifications are robust to last-digit f64
variation even though the raw result log is reproducible only within tolerance
(Section 21).

### 4.4 Three independent methods, now across four families

The correctness guarantee for `|λ₂|`/`τ_ε` remains cross-method agreement within
a declared tolerance (Section 8), inherited verbatim from TDI-6.1. Because
TDI-6.5 evaluates the spectral descriptors on the more varied F1/F2/F3 kernels
(near-deterministic, near-rank-one, and hypercube-local), the Section 21
three-method spectral cross-validation table **samples candidate kernels from
all four families**, and the trace-consistency witness
`max_k |Σλᵢᵏ − trace(Pᵏ)|` is reported per family.

## 5. Confirmatory comparison

The confirmatory comparison is **GKT vs GK** — the overlaps' marginal value
beyond contraction, the exact moments, **and** the literal spectral gap +
mixing time — computed **independently within each generator family**. This is
the TDI-6.1A comparison, now replicated across the TDI-5.7 family in place of
TDI-5.7's exact SKT-vs-SK comparison.

## 6. The one-step Noop kernel

Inherited unchanged from TDI-6.1 Section 5. For a candidate system of width `w`,
`P` is the one-step `Noop` kernel on the `n = 2^w` states:
`P[i][j] = 1/deg(i)` if `j` is a successor of state `i` under `Noop`, else 0.
Every candidate's kernel is total (every state has ≥ 1 successor, guaranteed by
each family rule), so `P` is row-stochastic and admits a stationary
distribution `π` (`πP = π`, `Σπ = 1`). `P` is assembled directly from the
frozen `build_system` successor structure — no new generation.

## 7. The non-exact spectral descriptors

Inherited unchanged from TDI-6.1 Section 6.

- **Literal spectral gap** `g = 1 − |λ₂|`, where `|λ₂|` is the second-largest
  eigenvalue modulus (SLEM) of `P`: the largest `|λ|` over all eigenvalues
  `λ ≠ 1`. `g ∈ [0, 1]`; larger `g` = faster mixing.
- **ε-mixing time** `τ_ε = min { t ≥ 1 : max_i ‖P^t(i, ·) − π‖_TV ≤ ε }`, with
  the threshold frozen at **ε = 1/4** and an iteration cap `T_max` (Section 13);
  reported as `τ_ε / T_max` (bounded to `[0, 1]`). If convergence is not reached
  within `T_max`, `τ_ε = T_max` (a declared, deterministic saturation — expected
  to occur for the near-deterministic F1 kernels and never masked).

Both are computed per candidate system and standardized like every other
feature. Their exploratory relationship to the exact moments (do s₂, s₃ predict
`g`?) is printed per family but drives no criterion.

## 8. Spectral-descriptor computation and the declared tolerance

Inherited unchanged from TDI-6.1 Section 7. The canonical path (method 1)
computes the spectrum by Hessenberg reduction followed by shifted QR iteration
(complex arithmetic, Wilkinson shift) with convergence tolerance `η = 1e-12` and
an iteration cap; `|λ₂|` is the max modulus over the non-Perron eigenvalues. The
mixing time iterates `Pᵗ` in `f64`. The bounded tests assert that methods 1, 2
and 3 agree on `|λ₂|` to within `1e-9` on a battery of kernels with **known**
spectra (symmetric, permutation, reversible birth–death, randomly generated
stochastic matrices), and that `τ_ε` matches a direct brute-force iteration
exactly. On the more varied F1/F2/F3 kernels a scalar deflated power iteration
(method 2) is **not** a valid `|λ₂|` witness — `λ₂` there may be complex — so
the canonical method-1 path's correctness on the family kernels is instead
established by the rigorous **trace-consistency** residual
`max_k |Σλᵢᵏ − trace(Pᵏ)|`, reported per family in the Section 21
cross-validation table and enforced by the trace-invariant bounded test.
Cross-method agreement within tolerance on the known-spectra battery, together
with the per-family trace witness, **is** the correctness guarantee that
replaces bit-exact reproduction for these descriptors.

## 9. The generator family

Inherited **unchanged** from the frozen TDI-5.7 preregistration Section 5. Let
`states = 2^width`. For each source state index `s`, the rule advances the
`splitmix64` chain (seeded by the candidate seed, exactly as the inherited
generator) and produces `mask[s]`, a `u64` successor mask assembled into a
system by the unchanged frozen `build_system`. The four frozen rules:

- **F0 — base (inherited, unchanged).** `d0 = next(chain); mask[s] = d0 %
  (2^states − 1) + 1` — a uniform draw over all non-empty successor subsets;
  reproduces the TDI-6.1 base generator's distribution exactly (under fresh
  seed blocks).
- **F1 — sparse (low out-degree).** `d = 1 + (next(chain) % 2)` (`d ∈ {1,2}`);
  select `d` distinct successor indices by rejection from the chain. Low
  branching → weaker mixing, larger spectral moments, `|λ₂|` near 1.
- **F2 — dense (high out-degree).** `e = next(chain) % 2`; `mask[s] =
  2^states − 1`; if `e == 1`, clear bit `(next(chain) % states)`. Near-complete
  branching → fast mixing, `|λ₂|` near 0. `mask[s]` keeps ≥ `states − 1` bits,
  so it is non-empty for `states ≥ 2`.
- **F3 — local (Hamming ≤ 1 neighbourhood).** Neighbourhood `N(s) = {s} ∪
  {s XOR (1 << b) : b ∈ 0..width}`; `r = next(chain)`; include neighbour
  `N(s)[j]` iff bit `j` of `r` is set; if none included, include `N(s)[0] = s`.
  Neighbour order fixed as `[s, s XOR 1, s XOR 2, s XOR 4, …]`. Structured
  hypercube-local kernel with self-loops.

All four rules are exact, deterministic, width-parametric, and produce a valid
total Noop kernel, exactly as frozen in TDI-5.7. No floating-point candidate
construction is introduced; the *candidate* stays bit-exact — only the derived
descriptors `g`, `τ_ε` are non-exact (Section 13).

## 10. Populations

Inherited from the frozen TDI-5.7 Section 7. For **each** of the four generator
families, generate the same in-distribution populations as TDI-6.1, across
**three** fresh, pairwise-disjoint seed blocks:

| Population | Width | Accepted records |
|---|---:|---:|
| training | 3 | 15,000 |
| holdout | 3 | 5,000 |
| training | 4 | 15,000 |
| holdout | 4 | 5,000 |

Accepted records per block: **40,000**; per family (3 blocks): **120,000**;
total (4 families): **480,000**. **No OOD populations.** Per-family models are
fitted on that family's combined width-3 + width-4 training populations and
every criterion is evaluated on that family's combined holdout populations.
Holdout records never affect fitting. Generation budgets are inherited unchanged
from TDI-5.2 Section 7.

## 11. Feature layouts

Inherited unchanged from TDI-6.1 Section 8.

| Layout | Features | Count | Role |
|---|---|---:|---|
| **CK** | baseline + δ + δ̄ | 15 | contraction baseline |
| **SK** | CK + s₂ + s₃ | 17 | exact baseline |
| **GK** | SK + g + τ_ε | 19 | **exact + literal-spectral baseline** (6.5A/B baseline) |
| **GKT** | GK + O₁ + O₂ | 21 | full model |

`GKT − GK` isolates the overlaps' marginal value after contraction, the exact
moments, **and** the literal spectral gap + mixing time — the confirmatory
comparison, computed within each family (criteria 6.5A, 6.5B, 6.5C). `GK − SK`
(the marginal value of the literal spectral descriptors in each family) is
reported as a descriptive diagnostic within criterion 6.5D. Per family, block
and horizon one target scaler is shared across the four layouts.

## 12. Independent seed blocks and deterministic bootstrap (fresh)

Twelve deterministic, pairwise-disjoint population seed blocks (three per
family), **disjoint from every prior block** (TDI-5.7 up to ≈ 2.53×10⁹; TDI-6.1
M/N/O at 3.0–3.23×10⁹; TDI-6.2 P/Q/R at 4.0–4.23×10⁹; TDI-6.5 starts at
5.0×10⁹). For family index `f ∈ {0,1,2,3}` (F0…F3) and block index
`b ∈ {0,1,2}`:

    base(f, b) = 5_000_000_000 + f · 300_000_000 + b · 100_000_000

and the four populations start at `base + {0, 10, 20, 30} · 1_000_000`
(training-w3, holdout-w3, training-w4, holdout-w4). Explicitly the training-w3
bases:

| Family | Block bases (training-w3 seed) |
|---|---|
| F0 | 5,000,000,000 · 5,100,000,000 · 5,200,000,000 |
| F1 | 5,300,000,000 · 5,400,000,000 · 5,500,000,000 |
| F2 | 5,600,000,000 · 5,700,000,000 · 5,800,000,000 |
| F3 | 5,900,000,000 · 6,000,000,000 · 6,100,000,000 |

Blocks are spaced 100,000,000 apart and populations 10,000,000 apart; each
population consumes at most a few million seeds, so all **48** reservations are
pairwise disjoint and disjoint from every earlier block. The evaluator verifies
disjointness of all consumed ranges at runtime.

The bootstrap engine, replicate count (**4,000**) and resampling discipline are
inherited unchanged (bit-exact, integer seeds). TDI-6.5 uses fresh bootstrap
seeds in the `0x5444_4936_3500_…` (`TDI6`/`35` = ".5") range, disjoint from the
frozen `TDI5`- and `TDI6.1`/`6.2`-prefixed seeds:

    block seed (family f, block b) : 0x5444_4936_3500_0000 + (3·f + b) + 1
                                     (F0: …0001/0002/0003 … F3: …000A/000B/000C)
    family aggregate seed (family f): 0x5444_4936_3500_4700 + f
                                     (F0: …4700 … F3: …4703)

Each family's stratified-aggregate bootstrap runs over its own three blocks with
its own aggregate seed. For each confirmatory comparison, report the two-sided
95% interval of the baseline-minus-challenger MSE difference and, for
equivalence classification, the two-sided 95% interval of the relative MSE
difference.

## 13. Non-exact determinism discipline (the TDI-6 convention)

Inherited unchanged from TDI-6.1 Section 12.

- **Floating-point regime.** All spectral computation is IEEE-754 binary64
  (`f64`), **single-threaded**, with a declared, fixed operation order (no
  parallel reduction, no FMA reordering, no `-ffast-math` equivalent). The
  evaluator sets and prints this regime.
- **Tolerances (frozen constants).** eigensolver convergence `η = 1e-12`;
  cross-method agreement `1e-9`; mixing threshold `ε = 1/4`; iteration cap
  `T_max` (frozen).
- **Reproduction is tolerance-based, not byte-exact.** A faithful re-run on the
  same toolchain/architecture reproduces the result log byte-for-byte; across
  architectures the raw metrics may differ in the last f64 digits, but the
  criterion classifications and the ±2% margins reproduce exactly.
- **Everything outside the two spectral descriptors remains bit-exact**
  (generation including every family rule, targets, ridge, bootstrap seeds).

## 14. Metrics

For every family, block, population, horizon and layout, print the full
inherited metric set of TDI-5.2 Section 9 (standardized-U and reconstructed-O:
MSE, MAE, R², Spearman, bias, means, calibration, bound fractions), plus, for
every confirmatory comparison, the absolute MSE difference, relative MSE
reduction, absolute MAE difference, Spearman difference, R² difference and
absolute-bias difference.

## 15. Standardized-U primacy

Standardized-U space is the primary confirmatory domain (TDI-5.2 Section 5).
Reconstructed-O-space quantities are secondary diagnostics only and cannot
determine any TDI-6.5 criterion.

## 16. Focal horizons and grid

Inherited from TDI-6.1: the dense grid `H = {3, 4, 5, 6, 7, 8}` and the focal
horizons **U₃** and **U₆**. The confirmatory criteria classify at the focal
horizons; per-family per-horizon GKT-vs-GK reductions across the grid are
reported (Section 21).

## 17. Criterion TDI-6.5A — replication of the literal-spectral control across generators (primary)

For **each** family Fᵢ, compute the **GKT-vs-GK** four-way classification (the
exact 4-way logic of TDI-5.2 Section 13, symmetric 2% relative-MSE margin) at
the focal horizons **U₃** and **U₆** on that family's combined holdout. TDI-6.5A
is the preregistered conjunction:

- **replicated** iff the classification is **Beneficial at both U₃ and U₆ for
  all four families**;
- otherwise a **located non-replication** — the evaluator names each (family,
  horizon) whose classification is not Beneficial.

TDI-6.5A is a preregistered classification, forced to no result. Full
replication would show the overlaps' value beyond the literal spectral gap is a
property of exact branching dynamics broadly, not of the single base generator;
a located non-replication would bound the literal-spectral control's generality
to the families where it holds.

## 18. Criterion TDI-6.5B — effect-size heterogeneity

Across the four families, report at each focal horizon the aggregate
relative-MSE reduction of GKT over GK: its **minimum, maximum and range**, and
whether **all four** exceed the 2% margin. A tight cluster well above 2% is
strong generality of effect size; a wide swing (even if all Beneficial) is a
preregistered caveat on transportability. Descriptive: it makes no pass/fail
claim beyond the "all four exceed 2%" flag.

## 19. Criteria TDI-6.5C and TDI-6.5D — transfer and descriptor drift (descriptive)

**TDI-6.5C — cross-generator transfer.** Fit the GK and GKT models on family
**F0**'s combined training population; evaluate the GKT-vs-GK comparison on
family **F1**'s combined holdout. Report the standardized-U R² of each layout
and the four-way classification. This distinguishes "the literal-spectral
control holds within each generator" (6.5A) from "the *same fitted* model — and
in particular the fitted use of `g`, `τ_ε` — transports across generators."

**TDI-6.5D — descriptor drift.** For each family, report the holdout means of
the **six** descriptors δ, δ̄, s₂, s₃, **`g` and `τ_ε`** (and their across-family
range), together with the family's descriptive **GK-vs-SK** focal relative-MSE
reduction (the marginal value of the literal spectral descriptors in that
family — the per-family analogue of TDI-6.1B). This contextualizes how demanding
the GK baseline is in each family: a family whose kernels are near-uniform
(δ, δ̄, s₂ small, `g` large) offers the descriptors little to work with, making
its 6.5A test effectively GKT-vs-baseline; a family with strong contraction /
slow mixing (`g` small, `τ_ε` large) makes 6.5A a demanding test.

Both TDI-6.5C and TDI-6.5D are preregistered **descriptive** summaries; neither
makes a success/failure claim.

## 20. Operational activation and full-run entrypoint contract

The v65 evaluator exposes exactly three modes: `--termination-smoke`,
`--preflight`, `--full`. A bare invocation refuses to run. `--termination-smoke`
uses only bounded tiny data and produces no result artifacts. `--preflight`
performs no scientific generation: it verifies the full frozen configuration
(all 48 seed reservations, all expected counts, all bootstrap constants, all
four family rules present, all tolerances and the FP regime), verifies that the
full pipeline is wired to `--full`, prints all TDI-6.5 and ancestor identities
and the exact real-run command, and exits without a result.

`--full` requires the exact confirmation environment variable:

    TDI65_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI65_FREEZE_RULE

Without that exact value, `--full` fails before any generation, fitting or
bootstrap. The confirmation check is a pure function of the environment value,
unit-testable without starting the experiment. No TDI-6.5 commit, test or CI run
supplies the token. The full run is a deliberate, one-time human action; the
authoring agent never invokes `--full` with the real token.

## 21. Required raw output

Inherited from TDI-5.2 Section 17 / TDI-6.1 Section 19 with TDI-6.5 identities
and **per-family** reporting: git commit; compiler/Cargo versions; the declared
FP regime and all tolerances; the v65 evaluator SHA-256; the TDI-6.5
preregistration and scientific-manifest SHA-256; the full frozen ancestor chain
(TDI-5.1 → 6.2); all frozen constants; the four family rules; the seed-block
definitions; per-family requested/accepted/rejected/attempted counts; rejection
counts by reason; final exclusive seeds; generation budgets; target scalers; the
CK, SK, GK and GKT model coefficients for every family and block; the
**three-method spectral cross-validation table** (per-candidate `|λ₂|`, `g`,
`τ_ε` from methods 1/2/3 and their max disagreement, with candidates **sampled
from each family** and the per-family trace-consistency residual); all metrics;
all bootstrap intervals; the per-family per-horizon GKT-vs-GK comparisons across
the grid U₃…U₈; the per-family focal GK-vs-SK diagnostic; the TDI-6.5A focal
classifications per family and the replication verdict; the TDI-6.5B
heterogeneity summary; the TDI-6.5C transfer classification; the TDI-6.5D
descriptor-drift table; deterministic termination diagnostics.

## 22. Determinism

Inherited from TDI-5.2 Section 18 and TDI-6.1 Section 12. Candidate generation
(**including every family's successor-mask rule and its `splitmix64` draw
sequence**), seed consumption, exclusions, preprocessing, exact
contraction-descriptor and spectral-moment construction, the `f64` spectral
descriptors under the fixed operation order of Section 13, model fitting,
bootstrap sampling, aggregation, metric calculation, iteration order,
scientific-value formatting and final criteria are deterministic functions of
committed constants. Wall-clock timestamps are reproduction metadata only.

## 23. Reproduction requirements and interpretation boundaries

The TDI-6.5 reproduction script must satisfy every requirement of TDI-5.2
Section 19 / TDI-6.1 Section 20 (refuse a dirty repository; verify all frozen
hashes including TDI-5.1 … 5.7, TDI-6.1, TDI-6.2 and TDI-6.5; refuse an existing
partial or complete result; acquire an exclusive lock; compile offline in
release mode; execute the evaluator exactly once with `--full`; capture complete
output; verify all final criterion lines; write metadata and a completion
marker; hash all artifacts; make final artifacts read-only), plus: it must
require the exact confirmation variable before invoking the evaluator. The
completion check verifies (a) the result-log SHA-256 (byte-exact on the
reference toolchain/architecture) and (b) that the printed 6.5A/6.5B/6.5C/6.5D
lines are present — the tolerance-robust invariant of Section 13.

A TDI-6.5 result establishes the (non)replication and effect-size stability of
the `{O_1,O_2}`-beyond-literal-spectral signal **across the specific
preregistered generator family F0–F3**, at the fixed base widths, within the
frozen machinery and the localized non-exactness of `g`, `τ_ε`. It does **not**
establish: robustness to generators outside that family; cross-width invariance
(TDI-5.8); an information-decomposition (PID) account (TDI-6.3); causal effect
(TDI-6.4); sufficiency under nonlinear model families (settled separately by
TDI-6.2, on the base generator only); higher-order or multi-step spectral
structure; or external empirical validity. The mixing time uses a single frozen
threshold ε = 1/4, and `|λ₂|`/`τ_ε` are f64 quantities (tolerance-based
reproduction) rather than bit-exact. The TDI-6.5A / B / C / D summaries may not
be rewritten after observing the result.

## 24. Freeze rule

Once the SHA-256 manifests, the v65 evaluator, the reproduction script, the CI
workflow and the bounded tests are committed, this design is frozen: scientific
code must not change; constants, tolerances and the FP-regime declaration must
not change; the four generator-family rules, the seed blocks, the layouts, the
spectral-descriptor definitions and the criteria must not change; no full run
may begin before all frozen hashes pass (TDI-5.1 … 5.7, 6.1, 6.2 and 6.5); any
scientific-code defect discovered after freezing requires a new experiment
identifier — TDI-6.5 may not be silently patched. The result classifications,
once produced, are frozen as reported.
