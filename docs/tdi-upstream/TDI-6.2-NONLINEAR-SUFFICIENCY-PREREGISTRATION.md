# TDI-6.2 — Nonlinear Sufficiency: Does the Overlap Signal Survive the Literal Spectral Gap Under a *Nonlinear* Model?

## Preregistration

This document is the frozen preregistration for TDI-6.2. Once its SHA-256
manifest, the evaluator, the reproduction script, the CI workflow and the
bounded tests are committed, this design is frozen under the Section 20 rule: no
scientific constant, tolerance, FP-regime declaration, seed block, model-family
definition, feature definition, baseline or criterion may change without a new
experiment identifier. Freezing does not authorize a run; the real experiment
begins only as the deliberate one-time human action of Section 16. The authoring
agent never invokes `--full`.

## 1. Experimental status, provenance, and the single changed factor

TDI-6.2 is a new confirmatory experiment derived from the completed and merged
TDI-6.1 result. It is **not** a continuation, patch, or reinterpretation of
TDI-5.1 … 6.1, each of which remains frozen under its own identifier.

TDI-6.1 established (single base generator, linear ridge) that the early
overlaps `{O₁, O₂}` carry predictive value beyond an exact contraction baseline
(δ, δ̄), the exact spectral moments (s₂, s₃), **and** the *literal* spectral gap
`g = 1 − |λ₂|` plus the ε-mixing time `τ_ε` of the one-step Noop kernel
(criterion 6.1A *Beneficial* at U₃ and U₆). Its most important stated
limitation (Section 21) is that the fit used a **linear** ridge model: a skeptic
can object that a linear model cannot extract the *nonlinear* information latent
in the spectral gap and mixing time, so the overlaps win only because the linear
model happens to be able to use them.

**TDI-6.2 changes exactly one factor to close that objection: the model family.**
Everything else — the exact candidate generation, the base width-3 + width-4
generator, the exact descriptors δ, δ̄, s₂, s₃, the non-exact literal spectral
descriptors g, τ_ε (computed exactly as in the frozen v61 evaluator), the four
feature layouts, the paired + stratified bootstrap, the four-way ±2 % classifier
and the `tdi-core` primitives — is **inherited unchanged**. Only the linear ridge
fit is replaced by a **degree-2 interaction-expanded ridge** (Section 6): a
genuinely nonlinear model that can represent squares and pairwise interactions of
every feature, including nonlinear functions of the spectral gap and mixing time.

Frozen ancestor identities (verified at runtime and in CI): the full frozen chain
TDI-5.1 → TDI-6.1 (every ancestor evaluator and preregistration hash) is verified
before any generation.

## 2. Research questions

Within the frozen candidate machinery, the frozen descriptor set, and the new
nonlinear model family:

1. does the SKT-style marginal value of `{O₁, O₂}` **survive a nonlinear model
   whose baseline already contains the literal spectral gap and the ε-mixing
   time**, at the focal horizons U₃ and U₆ (criterion **TDI-6.2A**, primary)?
2. do the literal spectral descriptors themselves add predictive value **under
   the nonlinear model** beyond the exact moments (criterion **TDI-6.2B**) — i.e.
   is 6.2A a demanding control, with a nonlinear-capable spectral baseline, and
   not a baseline of features the nonlinear model cannot exploit?
3. how does the overlaps' nonlinear marginal value **decay** across the dense
   grid U₃…U₈ (criterion **TDI-6.2C**)?

A *Beneficial* 6.2A would refute the "the linear model artificially favored the
overlaps" objection within this scope: even a model that can extract nonlinear
functions of `g`, `τ_ε`, `s₂`, `s₃` does not subsume the overlap signal.
TDI-6.2A is a preregistered classification, forced to no result.

## 3. Relationship to the frozen ancestors

**Inherited unchanged** (frozen; not re-derived): the exact candidate analysis,
the base width-3 + width-4 generator, observation geometry, target geometry
`U_h = -log2(1 - O_h)`, the 13 structural/entropic baseline variables, the two
early overlaps O₁, O₂, the two exact contraction descriptors δ, δ̄, the two exact
spectral moments s₂, s₃, the two **non-exact** literal spectral descriptors
`g = 1 − |λ₂|` and `τ_ε / T_max` (computed by the v61 eigensolver + mixing-time
path, with the same NaN-rejection self-guard), the layouts **CK / SK / GK / GKT**
(feature *sets*), the paired + stratified-aggregate bootstrap, the four-way ±2 %
classifier, and the `tdi-core` exact primitives. The non-exact determinism
discipline (Section 12) is inherited **verbatim**.

**New in TDI-6.2** (the only substantive change): the model family — a
**degree-2 interaction-expanded ridge** replaces the linear ridge (Section 6);
fresh, independent seed blocks (Section 10); and the criteria 6.2A / 6.2B / 6.2C
(Sections 15–17).

## 4. Design notes and confirmatory integrity

### 4.1 Why a nonlinear model is the decisive control on 6.1

TDI-6.1 put the literal `|λ₂|` and `τ_ε` into a **linear** baseline. A linear
model can only use those descriptors linearly; it cannot represent, e.g., a
`g²` curvature, a `g·τ_ε` interaction, or a threshold-like saturation in the
mixing time. If the true relationship between recovery and the spectral gap is
nonlinear, a linear GK baseline understates what the spectral gap "knows", and
the overlaps could win by default. TDI-6.2 gives the GK baseline a model that
**can** represent those nonlinear functions. If the overlaps still help
(6.2A *Beneficial*), the signal is not a linear-modeling artifact.

### 4.2 Why degree-2 interaction ridge (and why it adds no new non-exactness)

The degree-2 interaction expansion maps a layout's `d` features
`x₁ … x_d` to `[1, x₁ … x_d, {xᵢ·xⱼ : 1 ≤ i ≤ j ≤ d}]` — the linear terms plus
all `d(d+1)/2` pairwise products (including squares). This is a genuinely
nonlinear model (polynomial regression: it can fit curvature and interactions),
yet it is fit by the **same deterministic ridge solve** already used in every
prior TDI-5.x/6.1 experiment — no iterative optimizer, no randomness, no new
dependency. Because every feature is already a finite `f64` on the record (the
exact descriptors converted once; the two non-exact spectral descriptors from
the v61 path), the expansion and the standardized ridge fit run in the **same
IEEE-754, single-threaded, fixed-operation-order regime as 6.1**. TDI-6.2
therefore introduces **no new non-exactness**: `g` and `τ_ε` remain the only
non-exact quantities, and reproduction is tolerance-based exactly as in 6.1
(Section 12). As in 6.1, the ±2 % relative-MSE classifier margin dwarfs the f64
tolerances, so the criterion classifications are robust to last-digit variation.

### 4.3 Why the marginal-value comparisons stay valid under expansion

Each layout is expanded **consistently**: `GK` is the degree-2 expansion of
`baseline + δ + δ̄ + s₂ + s₃ + g + τ_ε`, and `GKT` is the degree-2 expansion of
that same set **plus** `O₁, O₂`. `GKT − GK` therefore isolates the overlaps'
*and all their pairwise interactions'* marginal value (O₁², O₂², O₁·O₂, and each
overlap crossed with every baseline / descriptor term) — the full nonlinear
marginal value of the overlaps beyond a nonlinear spectral+contraction baseline.
`GK − SK` isolates the literal spectral descriptors' nonlinear marginal value
beyond the exact moments. The baseline block's expansion is identical across a
comparison's two layouts, so each contrast isolates exactly the added block and
its interactions, nothing else.

### 4.4 Degree 2, not higher

Degree 2 is the minimal nonlinear step and keeps the expanded design
well-conditioned (GKT: 21 linear + 231 interaction + 1 intercept = 253 columns
against 30,000 training rows per width). Higher polynomial degrees, kernel
methods, and tree/forest ensembles are deferred to later experiments (they would
require either infeasible memory or a new, carefully-controlled non-exact
training procedure). The regularization `lambda = 1.0` on standardized expanded
features is inherited unchanged.

## 5. The one-step Noop kernel and the descriptors (inherited)

Identical to TDI-6.1 Sections 5–7: for a candidate of width `w`, `P` is the
one-step `Noop` kernel on `n = 2^w` states (`w ∈ {3, 4}`); the exact contraction
descriptors δ, δ̄ and exact spectral moments s₂, s₃ are exact rationals; the
literal spectral gap `g = 1 − |λ₂|` and normalized ε-mixing time `τ_ε / T_max`
are the two non-exact `f64` descriptors, from the frozen v61 complex shifted-QR
eigensolver and direct `P^t` iteration (ε = 1/4, `T_max = 4096`), with the same
per-candidate NaN-rejection self-guard. No descriptor is redefined.

## 6. The nonlinear model family (the only substantive change)

For each feature layout with feature vector `x = (x₁, …, x_d)`, the model is a
ridge regression on the **degree-2 interaction expansion**

    φ(x) = ( x₁, …, x_d, x₁², x₁x₂, …, x₁x_d, x₂², x₂x₃, …, x_d² ),

i.e. the `d` linear terms followed by the `d(d+1)/2` pairwise products in a fixed
canonical order (`i` outer, `j ≥ i` inner). The expanded features are
standardized (mean/scale) and fit with ridge `lambda = 1.0` and an unpenalized
intercept — the inherited ridge machinery, applied to `φ(x)` instead of `x`. The
expansion is a pure, deterministic function of the record's features; fitting and
prediction remain single-threaded with fixed operation order (Section 12).
Per block and horizon, one target scaler is shared across the four layouts, as in
6.1.

## 7. Feature layouts

The feature **sets** are inherited from TDI-6.1 Section 8; the model applied to
each is now the degree-2 expansion of Section 6.

| Layout | Base features | Base count `d` | Expanded columns `d + d(d+1)/2` | Role |
|---|---|---:|---:|---|
| **CK** | baseline + δ + δ̄ | 15 | 135 | contraction baseline (reporting) |
| **SK** | CK + s₂ + s₃ | 17 | 170 | exact baseline (6.2B baseline) |
| **GK** | SK + g + τ_ε | 19 | 209 | exact + literal-spectral baseline (6.2A baseline) |
| **GKT** | GK + O₁ + O₂ | 21 | 252 | full model |

(Expanded *columns* exclude the intercept.) `GK − SK` isolates the literal
spectral descriptors' nonlinear marginal value; `GKT − GK` isolates the overlaps'
nonlinear marginal value after contraction, exact moments and the literal
spectral gap + mixing time.

## 8. Populations

**Single base generator** (the inherited uniform width-3 + width-4 composition,
i.e. TDI-6.1's base generator under fresh seeds) — the generator question is
settled by TDI-5.7 and the literal-spectral base result by TDI-6.1, so 6.2
isolates the *new* factor (the nonlinear model). Three fresh seed blocks
**P / Q / R**; **40,000 accepted records per block, 120,000 total**; no OOD
populations. Models fitted on each block's combined width-3 + width-4 training
population; every criterion evaluated on that block's combined holdout;
standardized-U space primary.

## 9. Independent seed blocks (fresh)

Three deterministic, pairwise-disjoint seed blocks P/Q/R, disjoint from every
prior block (TDI-6.1 consumes seeds up to ≈ 3.23×10⁹; 6.2 starts at 4.0×10⁹):

- population base seed `base(b) = 4_000_000_000 + b · 100_000_000` for block
  index `b ∈ {0,1,2}`; the four populations start at `base + {0, 10, 20, 30}·10⁶`.
- block bootstrap seed `0x5444_4936_3200_0000 + b + 1` (the `TDI6` prefix and the
  `32` = ".2" marker distinguish these from the frozen `31`/`TDI5`-prefixed
  seeds).
- stratified aggregate bootstrap seed `0x5444_4936_3200_4700`.

Generation budgets are inherited unchanged. The evaluator verifies disjointness
of the consumed ranges at runtime (all 12 reservations).

## 10. Deterministic bootstrap

Inherited engine and replicate count (4000); only the seeds are fresh (Section
9). Bootstrap resampling remains bit-exact (integer seeds); only the descriptor
values `g, τ_ε` carry non-exact f64 content, and the model fit is deterministic
f64.

## 11. Non-exact determinism discipline

Inherited **verbatim** from TDI-6.1 Section 12: IEEE-754 binary64,
single-threaded, fixed operation order; eigensolver convergence η = 1e-12;
cross-method agreement 1e-9; mixing threshold ε = 1/4; iteration cap
`T_max = 4096`. The degree-2 expansion and the ridge fit add no new non-exact
quantity — `g` and `τ_ε` remain the only non-exact descriptors. Reproduction is
tolerance-based; the criterion classifications and the ±2 % margins reproduce
exactly.

## 12. Metrics and standardized-U primacy

Inherited: for every block, population, horizon and layout, print the full metric
set (standardized-U and reconstructed-O: MSE, MAE, R², Spearman, bias, means,
calibration, bound fractions). Standardized-U space is the primary confirmatory
domain; reconstructed-O quantities are secondary and determine no criterion.

## 13. Criterion TDI-6.2A — signal beyond the literal spectral gap under a nonlinear model (primary)

Compare **GKT against GK** (both degree-2 models) on combined holdout at the
focal horizons **U₃** and **U₆**, four-way classification with the symmetric 2 %
relative-MSE margin. *Beneficial* = the overlaps reduce error beyond a nonlinear
baseline of contraction + exact moments + literal spectral gap + mixing time (the
decisive positive result — the overlap signal is not a linear-modeling artifact);
*Equivalent* = the nonlinear spectral baseline subsumes the overlap signal;
*Harmful* = it over-explains it. Preregistered classification, forced to no
result.

## 14. Criterion TDI-6.2B — nonlinear marginal value of the literal spectral descriptors

Compare **GK against SK** (both degree-2 models) at the focal horizons — the
control that makes 6.2A demanding. *Beneficial* would show the literal spectral
gap + mixing time carry nonlinear predictive value the exact moments do not, so
6.2A tests against a spectral baseline the nonlinear model can genuinely exploit;
an *Equivalent* 6.2B would caveat 6.2A as testing against added features the
nonlinear model cannot use beyond the moments.

## 15. Criterion TDI-6.2C — decay law and redundancy horizon

Evaluate GKT-vs-GK (degree-2) at every horizon of the dense grid U₃…U₈; report
the per-horizon aggregate relative-MSE reduction, its classification, whether the
sequence is monotone non-increasing, the first-*Equivalent* redundancy horizon
`h★` (if any), and the successive ratios. Descriptive.

## 16. Operational activation and full-run entrypoint contract

The evaluator exposes exactly `--termination-smoke`, `--preflight`, `--full`; a
bare invocation refuses. `--full` requires the exact confirmation variable

    TDI62_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI62_FREEZE_RULE

checked, as a pure function of the environment value, *before* any generation,
fitting or bootstrap. No commit, test or CI supplies the token; the full run is a
deliberate one-time human action; the authoring agent never invokes it.

## 17. Required raw output

Inherited from TDI-6.1 Section 19 with TDI-6.2 identities: git commit;
compiler/Cargo versions; the declared FP regime and all tolerances; the v62
evaluator SHA-256; the TDI-6.2 preregistration and scientific-manifest hashes;
the full frozen ancestor chain (TDI-5.1 → 6.1); all frozen constants; the
seed-block definitions; per-block counts, rejection reasons, final seeds,
budgets; target scalers; the model-family declaration and the expanded-column
counts; CK/SK/GK/GKT coefficients per block (over the expanded design); the
three-method spectral cross-validation table (inherited); all metrics; all
bootstrap intervals; the per-horizon GKT-vs-GK and focal GK-vs-SK comparisons;
the 6.2A/6.2B focal classifications; the 6.2C decay summary; deterministic
termination diagnostics.

## 18. Reproduction and tolerance-based verification

`scripts/reproduce-tdi6.2.sh` refuses without the exact token, refuses a dirty
repository, verifies the full frozen chain (TDI-5.1 → 6.1 + the TDI-6.2
manifests) before any generation, runs `--full` once, and writes read-only
result, metadata, hash and completion artifacts. The completion check verifies
(a) the result log SHA-256 (byte-exact on the reference toolchain/architecture)
and (b) that the printed 6.2A/6.2B/6.2C classification lines are present.

## 19. Limitations

This design establishes, if 6.2A is *Beneficial*, that the overlap signal
survives the literal spectral gap and ε-mixing time **under a degree-2
polynomial model** on the base generator. It does **not** establish: sufficiency
under arbitrarily flexible learners (kernel machines, tree/forest ensembles,
deep networks — deferred, as they need infeasible memory or a controlled
non-exact training procedure); robustness across the TDI-5.7 generator families
(single base generator here — deferred to the literal-spectral × families
experiment); a PID / information-decomposition account; causal effect; higher-
order or multi-step spectral structure; cross-width invariance (TDI-5.8); or
external empirical validity. The nonlinear family is fixed at degree 2 with
`lambda = 1.0`; the mixing time uses a single frozen threshold ε = 1/4.

## 20. Freeze rule

Once the SHA-256 manifests, the evaluator, the reproduction script, the CI
workflow and the bounded tests are committed, this design is frozen: no
scientific constant, tolerance, FP-regime declaration, seed block, model-family
definition, feature definition, baseline or criterion may change without a new
experiment identifier. The result classifications, once produced, are frozen as
reported.
