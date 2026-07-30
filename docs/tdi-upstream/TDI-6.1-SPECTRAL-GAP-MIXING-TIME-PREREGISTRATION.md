# TDI-6.1 — The Literal Spectral Gap and Mixing Time: Does the Overlap Signal Survive the *Actual* Second Eigenvalue?

## Preregistration

This document is the frozen preregistration for TDI-6.1 — the design of the
**first TDI-6 experiment** and the first to relax bit-exactness. Once its
SHA-256 manifest, the evaluator, the reproduction script, the CI workflow and
the bounded tests are committed, this design is frozen under the Section 22
freeze rule: no scientific constant, tolerance, FP-regime declaration, seed
block, spectral descriptor definition, feature definition, baseline or
criterion may change without a new experiment identifier. Freezing the design
does not authorize a run; the real experiment begins only as the deliberate
one-time human action of Section 18. The authoring agent never invokes
`--full`.

## 1. Experimental status, provenance, and the TDI-6 discipline shift

TDI-6.1 is a new confirmatory experiment derived from the completed and merged
TDI-5.7 result. It is **not** a continuation, patch, or reinterpretation of
TDI-5.1 … 5.7, each of which remains frozen under its own identifier.

The whole exact series (TDI-5.2 … 5.7) established that the early overlaps
`{O_1, O_2}` carry predictive signal beyond an **exact** contraction baseline
(δ, δ̄; 5.5), the **exact** spectral moments (s₂ = trace(P²), s₃ = trace(P³);
5.6), and a **family of exact generators** (5.7). Every one of those
experiments explicitly deferred the decisive skeptic control to TDI-6: the
signal was never tested against the *literal* second eigenvalue `|λ₂|` (the
spectral gap `1 − |λ₂|`) or the ε-threshold mixing time, because those are
**transcendental / iterative** and cannot be expressed in the frozen bit-exact
rational arithmetic.

**TDI-6.1 runs exactly that control.** It is therefore the first experiment to
abandon bit-exactness — and it does so **only** for the two new spectral
descriptors, under the explicit non-exact determinism discipline of Section 12.
Everything else (candidate generation, ridge `lambda = 1.0`, the paired +
stratified bootstrap, the four-way ±2% classifier, the exact descriptors δ, δ̄,
s₂, s₃) is inherited **unchanged and still bit-exact** from the frozen
ancestors.

Frozen ancestor identities (verified at runtime and in CI): the v57 evaluator,
the TDI-5.7 preregistration, and the full frozen chain TDI-5.1 → TDI-5.7
(every ancestor evaluator and preregistration hash) are verified before any
generation.

## 2. Research questions

Within the frozen candidate machinery, the frozen exact descriptor set, and the
new non-exact spectral descriptors:

1. does the SKT-style marginal value of `{O_1, O_2}` **survive a baseline that
   additionally contains the literal spectral gap `1 − |λ₂|` and the ε-mixing
   time** of the one-step Noop kernel, at the focal horizons U₃ and U₆
   (criterion **TDI-6.1A**, primary)?
2. do the literal spectral descriptors **themselves add predictive value**
   beyond the exact moments s₂, s₃ (criterion **TDI-6.1B**) — i.e. is 6.1A a
   demanding control rather than a baseline padded with inert features?
3. how does the overlaps' marginal value **decay** across the dense grid
   U₃…U₈ once the literal spectral descriptors are in the baseline (criterion
   **TDI-6.1C**)?

This is the control the series has pointed at since TDI-5.5. A *Beneficial*
6.1A would be the strongest evidence yet that the overlap signal is not a proxy
for classical mixing/spectral structure; an *Equivalent* or *Harmful* 6.1A
would bound TDI's contribution to what the literal spectral gap already
captures. TDI-6.1A is a preregistered classification, forced to no result.

## 3. Relationship to the frozen ancestors

**Inherited unchanged** (frozen; still bit-exact; not re-derived): the exact
candidate analysis, the base width-3 + width-4 generator, observation geometry,
target geometry `U_h = -log2(1 - O_h)`, the 13 structural/entropic baseline
variables, the two early overlaps O₁, O₂, the two exact contraction descriptors
δ, δ̄, the two exact spectral moments s₂, s₃, the layouts **CK / SK**, ridge
`lambda = 1.0`, `build_system`, the paired + stratified-aggregate bootstrap,
the four-way ±2% classifier, and the `tdi-core` exact primitives.

**New in TDI-6.1** (the only substantive additions): two **non-exact spectral
descriptors** of the one-step Noop kernel — the literal spectral gap
`g = 1 − |λ₂|` and the ε-mixing time `τ_ε` (Section 6) — computed in `f64`
under the non-exact determinism discipline (Section 12) and cross-validated by
three independent methods (Section 7); two new feature layouts **GK** and
**GKT** (Section 8); fresh, independent seed blocks (Section 10); and the
criteria 6.1A / 6.1B / 6.1C (Sections 15–17).

## 4. Design notes and confirmatory integrity

### 4.1 Why the literal spectral gap is the decisive control

The exact moments s₂ = Σλᵢ², s₃ = Σλᵢ³ are *power sums* of the spectrum: they
constrain, but do not determine, the second-eigenvalue modulus `|λ₂|` that
governs the asymptotic mixing rate, nor the ε-mixing time that governs
finite-horizon convergence. A skeptic can still argue the overlap signal is a
finite-sample estimator of the *literal* spectral gap. TDI-6.1 puts that exact
quantity (to f64 precision) directly into the baseline. If `{O_1, O_2}` still
help, the "TDI is just the spectral gap" hypothesis is refuted within this
scope.

### 4.2 Why non-exact, and why the relaxation is minimal

`|λ₂|` of a non-symmetric row-stochastic matrix is a root of the characteristic
polynomial — in general irrational or complex — and the ε-mixing time is
defined by an iterative convergence condition. Neither is expressible in the
frozen rational arithmetic. TDI-6.1 therefore computes them in IEEE-754 `f64`.
Crucially the relaxation is **localized**: only the two spectral descriptors are
non-exact; generation, target construction, ridge fitting and the bootstrap are
inherited unchanged. Because the four-way classifier's margin is ±2% relative
MSE — many orders of magnitude larger than the f64 eigensolver tolerance
(Section 7) — the **criterion classifications are robust** to the last-digit
variation of the descriptors, even though the raw result log is only
reproducible within tolerance (Section 20).

### 4.3 Three independent methods (the non-exact correctness guarantee)

Bit-exactness is replaced by **cross-method agreement within a declared
tolerance**. The literal `|λ₂|` and `τ_ε` are computed by three independent
implementations that must agree (Section 7):

1. **canonical (frozen feature path):** a pure-Rust standard dense eigenvalue
   algorithm — Hessenberg reduction followed by shifted QR iteration (in complex
   arithmetic, so real and complex-conjugate eigenvalues are handled uniformly)
   — on the ≤ 16×16 kernel, giving all eigenvalues → the literal `|λ₂|`. No new
   dependency; `unsafe`-free.
2. **cross-check A (tests):** power iteration on the kernel deflated against the
   Perron (stationary) direction, plus the geometric TV-distance decay rate, as
   an independent witness of `|λ₂|`.
3. **cross-check B (tests):** a battle-tested reference eigensolver crate,
   admitted **only as a `dev-dependency`** (test builds only) so the frozen
   scientific feature path stays dependency-free; it independently confirms the
   full spectrum.

The ε-mixing time is computed by direct iteration of `Pᵗ` (method for all
three: the mixing time is an observable, not an eigenvalue) and cross-checked
against the spectral-gap bound `τ_ε ≲ log(1/ε)/(1 − |λ₂|)`.

Cross-check B (Section 7, method 3) is included as a **test-only
`dev-dependency`** reference eigensolver: it enlarges `Cargo.lock` and
`vendor/` but never enters the frozen scientific feature path (method 1), which
stays dependency-free and `unsafe`-free. Where offline vendoring of the
reference crate is unavailable, the cross-validation test falls back to the
methods-1↔2 agreement together with the closed-form **known-spectra battery**
of Section 7 — which alone establish the canonical path's correctness — so the
frozen feature path and its guarantees never depend on the reference crate
being present.

## 5. The one-step Noop kernel

For a candidate system of width `w`, let `P` be the one-step `Noop` kernel on
the `n = 2^w` states: `P[i][j] = 1/deg(i)` if `j` is a successor of state `i`
under `Noop`, else 0. Every candidate's kernel is total (every state has ≥ 1
successor, by construction), so `P` is row-stochastic and admits a stationary
distribution `π` (`πP = π`, `Σπ = 1`). `P` is assembled directly from the
frozen `build_system` successor structure — no new generation.

## 6. The non-exact spectral descriptors

- **Literal spectral gap** `g = 1 − |λ₂|`, where `|λ₂|` is the second-largest
  eigenvalue modulus (SLEM) of `P`: the largest `|λ|` over all eigenvalues
  `λ ≠ 1` (the Perron eigenvalue is exactly 1). `g ∈ [0, 1]`; larger `g` = faster
  mixing.
- **ε-mixing time** `τ_ε = min { t ≥ 1 : max_i ‖P^t(i, ·) − π‖_TV ≤ ε }`, with
  the threshold frozen at **ε = 1/4** and an iteration cap `T_max` (Section 12);
  `τ_ε` is reported as `τ_ε / T_max` (bounded to `[0, 1]`) so it is on the same
  scale as the other features. If convergence is not reached within `T_max`,
  `τ_ε = T_max` (a declared, deterministic saturation).

Both are computed per candidate system, standardized like every other feature.
Their **exploratory diagnostic** relationship to the exact moments (do s₂, s₃
predict `g`?) is printed but drives no criterion.

## 7. Spectral-descriptor computation and the declared tolerance

The canonical path (method 1) computes the spectrum by Hessenberg reduction
followed by shifted QR iteration (in complex arithmetic, Wilkinson shift) with a
declared convergence tolerance `η = 1e-12` and iteration cap; `|λ₂|` is the max
modulus over the non-Perron eigenvalues. The mixing time
iterates `Pᵗ` in `f64`. The bounded tests assert that methods 1, 2 and 3 agree
on `|λ₂|` to within `1e-9` on a battery of kernels with **known** spectra
(symmetric, permutation, reversible birth–death, and randomly generated
stochastic matrices), and that `τ_ε` matches a direct brute-force iteration
exactly. Cross-method agreement within tolerance **is** the correctness
guarantee that replaces bit-exact reproduction for these descriptors.

## 8. Feature layouts

| Layout | Features | Count | Role |
|---|---|---:|---|
| **SK** | baseline + δ + δ̄ + s₂ + s₃ | 17 | exact baseline (inherited; 6.1B baseline) |
| **GK** | SK + g + τ_ε | 19 | **exact + literal-spectral baseline** (6.1A baseline) |
| **GKT** | GK + O₁ + O₂ | 21 | full model |

`GK − SK` isolates the marginal value of the literal spectral descriptors after
the exact moments (criterion 6.1B). `GKT − GK` isolates the overlaps' marginal
value **after** contraction, exact moments, **and** the literal spectral gap +
mixing time (criteria 6.1A, 6.1C). Per block and horizon one target scaler is
shared across the three layouts.

## 9. Populations

**Single base generator** (the inherited uniform width-3 + width-4 composition,
i.e. TDI-5.7's F0Base under fresh seeds) — the generator question is settled by
TDI-5.7, so 6.1 isolates the *new* factor (the literal spectral descriptors).
Three fresh seed blocks **M / N / O**; **40,000 accepted records per block,
120,000 total**; no OOD populations. Models fitted on each block's combined
width-3 + width-4 training population; every criterion evaluated on that block's
combined holdout; standardized-U space primary.

## 10. Independent seed blocks (fresh)

Three deterministic, pairwise-disjoint seed blocks M/N/O, **disjoint from every
prior block** (TDI-5.7 consumes seeds up to ≈ 2.53×10⁹; 6.1 starts at 3.0×10⁹):

- population base seed `base(b) = 3_000_000_000 + b · 100_000_000` for block
  index `b ∈ {0,1,2}`; the four populations start at `base + {0, 10, 20, 30}·10⁶`.
- block bootstrap seed `0x5444_4936_3100_0000 + b + 1` (the `TDI6` prefix and
  the `31` = ".1" marker distinguish these from the frozen `TDI5`-prefixed
  seeds).
- stratified aggregate bootstrap seed `0x5444_4936_3100_4700`.

Generation budgets are inherited unchanged from TDI-5.2. The evaluator verifies
disjointness of the consumed ranges at runtime (all 12 reservations).

## 11. Deterministic bootstrap

Inherited engine and replicate count (4000); only the seeds are fresh
(Section 10). Bootstrap resampling remains bit-exact (integer seeds); only the
feature *values* g, τ_ε carry non-exact f64 content.

## 12. Non-exact determinism discipline (the TDI-6 convention)

This section is the substantive new discipline TDI-6 introduces, and it applies
to every future TDI-6 experiment:

- **Floating-point regime.** All spectral computation is IEEE-754 binary64
  (`f64`), **single-threaded**, with a declared, fixed operation order (no
  parallel reduction, no fused-multiply-add reordering, no `-ffast-math`
  equivalent). The evaluator sets and prints this regime.
- **Tolerances (frozen constants).** eigensolver convergence `η = 1e-12`;
  cross-method agreement `1e-9`; mixing threshold `ε = 1/4`; iteration cap
  `T_max` (frozen).
- **Reproduction is tolerance-based, not byte-exact.** A faithful re-run on the
  same toolchain/architecture reproduces the result log byte-for-byte; across
  architectures the raw metrics may differ in the last f64 digits, but **the
  criterion classifications and the ±2% margins reproduce exactly** (Section
  4.2). Both guarantees are declared and tested.
- **Everything outside the two spectral descriptors remains bit-exact**
  (generation, targets, ridge, bootstrap seeds).

## 13. Metrics

For every block, population, horizon and layout, print the full inherited
metric set (standardized-U and reconstructed-O: MSE, MAE, R², Spearman, bias,
means, calibration, bound fractions).

## 14. Standardized-U primacy

Standardized-U space is the primary confirmatory domain (inherited);
reconstructed-O quantities are secondary diagnostics and determine no criterion.

## 15. Criterion TDI-6.1A — signal beyond the literal spectral gap (primary)

Compare **GKT against GK** on combined holdout at the focal horizons **U₃** and
**U₆**, four-way classification with the symmetric 2% relative-MSE margin.
*Beneficial* = the overlaps reduce error beyond contraction + exact moments +
literal spectral gap + mixing time (the decisive positive result);
*Equivalent* = the literal spectral descriptors subsume the overlap signal;
*Harmful* = they over-explain it. Preregistered classification, forced to no
result.

## 16. Criterion TDI-6.1B — marginal value of the literal spectral descriptors

Compare **GK against SK** at the focal horizons — the control that makes 6.1A
demanding. *Beneficial* would show the literal spectral gap + mixing time carry
predictive value the exact moments s₂, s₃ do not, so 6.1A is a genuine
strengthening; an *Equivalent* 6.1B would caveat 6.1A as testing against
near-inert added features.

## 17. Criterion TDI-6.1C — decay law and redundancy horizon

Evaluate GKT-vs-GK at every horizon of the dense grid U₃…U₈; report the
per-horizon aggregate relative-MSE reduction, its classification, whether the
sequence is monotone non-increasing, the first-Equivalent redundancy horizon
`h★` (if any), and the successive ratios. Descriptive.

## 18. Operational activation and full-run entrypoint contract

The evaluator exposes exactly `--termination-smoke`, `--preflight`, `--full`; a
bare invocation refuses. `--full` requires the exact confirmation variable

    TDI61_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI61_FREEZE_RULE

checked, as a pure function of the environment value, *before* any generation,
fitting or bootstrap. No commit, test or CI supplies the token; the full run is
a deliberate one-time human action; the authoring agent never invokes it.

## 19. Required raw output

Inherited from TDI-5.2 Section 17 with TDI-6.1 identities: git commit;
compiler/Cargo versions; the **declared FP regime and all tolerances**; the v61
evaluator SHA-256; the TDI-6.1 preregistration and scientific-manifest hashes;
the full frozen ancestor chain (TDI-5.1 → 5.7); all frozen constants; the
seed-block definitions; per-block counts, rejection reasons, final seeds,
budgets; target scalers; CK/SK/GK/GKT coefficients per block; the **three-method
spectral cross-validation table** (per-candidate `|λ₂|`, `g`, `τ_ε` from methods
1/2/3 and their max disagreement); all metrics; all bootstrap intervals; the
per-horizon GKT-vs-GK and focal GK-vs-SK comparisons; the 6.1A/6.1B focal
classifications; the 6.1C decay summary; deterministic termination diagnostics.

## 20. Reproduction and tolerance-based verification

`scripts/reproduce-tdi6.1.sh` refuses without the exact token, refuses a dirty
repository, verifies the full frozen chain (TDI-5.1 → 5.7 + the TDI-6.1
manifests) before any generation, runs `--full` once, and writes read-only
result, metadata, hash and completion artifacts. The completion check verifies
(a) the result log SHA-256 (byte-exact on the reference toolchain/architecture)
and (b) that the printed 6.1A/6.1B/6.1C classification lines are present —
the tolerance-robust invariant of Section 12.

## 21. Limitations

This design establishes, if 6.1A is Beneficial, that the overlap signal survives
the *literal* spectral gap and ε-mixing time of the one-step kernel on the base
generator. It does **not** establish: robustness of the *literal-spectral*
control across the TDI-5.7 generator families (single base generator here);
sufficiency under nonlinear / non-parametric model families (TDI-6.2); an
information-decomposition (PID) account (TDI-6.3); causal effect (TDI-6.4);
higher-order or multi-step spectral structure; cross-width invariance
(TDI-5.8); or external empirical validity. The mixing time uses a single frozen
threshold ε = 1/4.

## 22. Freeze rule

Once the SHA-256 manifests, the evaluator, the reproduction script, the CI
workflow and the bounded tests are committed, this design is frozen: no
scientific constant, tolerance, FP-regime declaration, seed block, spectral
descriptor definition, feature definition, baseline or criterion may change
without a new experiment identifier. The result classifications, once produced,
are frozen as reported.
