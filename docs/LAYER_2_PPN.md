# Layer 2 — PPN parameter extraction (γ, β): design & conventions

This is the architecture-decision record for the second Layer 2 increment:
oracle-backed extraction of the Eddington–Robertson PPN parameters **γ** and
**β** from static, spherically symmetric, asymptotically flat, weak-field metrics
given in a PPN-compatible **isotropic** radial coordinate. It is written before
the code, per the platform rule.

It extends the Layer 2 Covariant Gravity Workbench
([`LAYER_2_COVARIANT_GRAVITY.md`](LAYER_2_COVARIANT_GRAVITY.md)) and **reuses**
the Layer 1 geometry core (`Metric`, `IsotropicSchwarzschild`), the experiment
framework, the determinism guarantees, and the scientific-category discipline.
It does **not** restart or redesign any validated component, and it introduces
**no** symbolic-algebra system.

## 1. Scientific target and conventions (signs stated explicitly)

Signature is `(-,+,+,+)` throughout (the crate convention: `Minkowski` gives
`g_00 = -1`). For a static, spherically symmetric metric in a PPN-compatible
isotropic radial coordinate `ρ`, with Newtonian compactness

```text
U(ρ) = G M / ρ ,        0 < U ≪ 1  (weak field),
```

the Eddington–Robertson weak-field expansions are

```text
g_00(ρ) = -1 + 2U - 2β U^2 + O(U^3),
g_ij(ρ) = A(ρ) δ_ij ,   A(ρ) = 1 + 2γ U + O(U^2).
```

`A(ρ)` is the **spatial conformal factor** (the isotropic spatial metric is
conformally flat: `g_ρρ = g_θθ/ρ^2 = g_φφ/(ρ^2 sin^2θ) = A`). For **General
Relativity in isotropic Schwarzschild coordinates**,

```text
g_00 = -((1 - M/2ρ)/(1 + M/2ρ))^2 = -1 + 2U - 2U^2 + O(U^3)  ⇒  β = 1,
A    = (1 + M/2ρ)^4               = 1 + 2U + (3/2)U^2 + O(U^3)  ⇒  γ = 1,
```

(hand-derived series, documented alongside the oracle). The extractor must
asymptotically recover `γ = β = 1`. Signs are never inferred silently: the
estimators below are written for exactly this signature and expansion.

## 2. Finite-radius effective estimators

Solving the two expansions for the leading coefficients gives the per-radius
effective estimators

```text
γ_eff(ρ) = (A(ρ) - 1) / (2U)                    = γ + O(U),
β_eff(ρ) = -(g_00(ρ) + 1 - 2U) / (2U^2)         = β + O(U).
```

These are **diagnostic data**, not the answer: each carries an `O(U)` (γ) or
`O(U)` (β, after the `U^2` division) contamination from the higher-order terms.
They are exposed per radius (`FiniteRadiusEstimate { radius, compactness,
value }`) and never returned as the asymptotic result.

## 3. Asymptotic extraction (deterministic polynomial least-squares)

The asymptotic value is the `U → 0` intercept of the effective estimator. We fit

```text
γ_eff(U) ≈ γ + c_1 U + c_2 U^2 + … + c_n U^n ,
β_eff(U) ≈ β + d_1 U + d_2 U^2 + … + d_n U^n ,
```

by ordinary least squares and read the intercept. Implementation choices, all to
keep the fit deterministic and auditable:

- **Normalized fit variable** `s = U / U_max ∈ (0, 1]` (the intercept `s → 0` is
  the same `U → 0` limit). Normalization keeps the Vandermonde/normal-equations
  system well scaled; without it the moments span `1 … U_max^{2n}` and become
  hopelessly ill-conditioned for `U ≪ 1`.
- A **small self-contained dense solver** for the `(n+1)×(n+1)` normal equations
  `(VᵀV) c = Vᵀy`: Gaussian elimination with partial pivoting, deterministic row
  order and accumulation order, no allocation-order dependence, no randomness, no
  hidden regularization. (`invert_metric`/`determinant` are const-generic and
  cannot take a runtime degree, so a dedicated small solver is used — no new
  dependency, matching the crate's self-contained linear-algebra style.)
- **Configurable degree** within a small validated range `1 ≤ n ≤ 4`.
- **Rejections** (typed `PpnError`): fewer samples than `n + 1`
  (underdetermined), non-finite inputs, a (near-)singular pivot (`SingularFit` /
  `IllConditionedFit`), an unsupported degree.
- A reported **conditioning indicator** (the scaled minimum pivot magnitude);
  ill-conditioned systems are rejected rather than silently trusted.

## 4. Radial domain

`PpnDomain { mass_scale, radius_min, radius_max, sample_count, sampling }` with a
deterministic `sampling` policy — uniform in compactness `U`, logarithmic in `ρ`,
or explicit radii. Validated: `mass_scale` finite and positive; `radius_min <
radius_max`; the largest compactness `U_max = M/radius_min` is below a weak-field
cap; every radius outside the metric's validity domain (caught by the metric's
own finiteness/validity checks); deterministic sample ordering; at least `n + 1`
samples for the requested degree. No single radius interval is hardcoded as
universally valid.

## 5. Coordinate-validity contract (mandatory)

PPN coefficients are coordinate-sensitive, so the extractor never accepts a raw
`Metric` and silently reads its radial coordinate as isotropic. Extraction is
defined over a dedicated trait

```text
StaticIsotropicMetric {
    mass_scale(&self) -> f64,
    g_tt(&self, radius) -> Result<f64, PpnError>,
    spatial_conformal_factor(&self, radius) -> Result<f64, PpnError>,
}
```

Two ways to obtain one:

- **Directly** — the synthetic oracles implement it from their defining
  coefficients (isotropic by construction).
- **Via an explicit adapter** over a spherical `Metric<4>` that **checks
  conformal flatness** at each radius: `g_ρρ`, `g_θθ/ρ^2`, `g_φφ/(ρ^2 sin^2θ)`
  must agree, else `PpnError::NonIsotropicCoordinates`. `IsotropicSchwarzschild`
  passes this check; **areal Schwarzschild fails it** (`g_rr = 1/f` while
  `g_θθ/r^2 = 1`) and is rejected — the mandatory negative oracle. It is thus
  impossible to *silently* apply the isotropic PPN formulas to areal coordinates.

## 6. Error and contamination characterization

The returned uncertainty is not decorative. Several deterministic diagnostics are
computed and combined conservatively into an **estimated numerical uncertainty**
(explicitly *not* a rigorous bound):

- fit residual norm of the selected model;
- **window sensitivity** — re-extract on the weaker-field sub-window and compare
  intercepts;
- **order sensitivity** — compare adjacent valid extrapolation degrees;
- **resolution sensitivity** — re-extract on a deterministic sub-sample;
- **oracle error** — absolute/relative error when an exact answer is known;
- **conditioning** — the solver's conditioning indicator.

Names use "estimated uncertainty" / "sensitivity" / "fit diagnostic", never
"proof" or "certified bound".

## 7. Oracle hierarchy

1. **Exact synthetic PPN metric** — `g_00 = -1 + 2U - 2β⋆U^2`,
   `A = 1 + 2γ⋆U`, exactly. For these the effective estimators are already
   *constant* (`γ_eff ≡ γ⋆`, `β_eff ≡ β⋆`), so the extractor recovers the injected
   pair to machine precision. Pairs include `(1, 1)` and non-GR pairs. Isolates
   extraction correctness from contamination.
2. **Contaminated synthetic** — add known `a_3 U^3 + a_4 U^4` to `g_00` and
   `b_2 U^2 + b_3 U^3` to `A`. Finite-radius estimators are biased; the
   extrapolation converges to the injected values; a weaker-field window improves
   the result; the reported sensitivity tracks the actual error.
3. **Isotropic Schwarzschild** — the exact `IsotropicSchwarzschild` metric
   functions (not the truncated series the extractor uses) → `γ = β = 1`.
4. **Invalid coordinates** — areal-coordinate `Schwarzschild` through the adapter
   is rejected (`NonIsotropicCoordinates`), never silently trusted. Mandatory.

## 8. Scope and explicit exclusions

**In scope:** static · spherically symmetric · asymptotically flat · isotropic
(or explicitly PPN-compatible) coordinates · γ and β · finite-radius estimates ·
asymptotic extrapolation · convergence/contamination diagnostics · the synthetic
and isotropic-Schwarzschild oracles.

**Excluded for now** (documented so the boundary is explicit): preferred-frame
and nonconservative PPN parameters; the full ten-parameter formalism; rotating,
time-dependent, non-spherical/multipolar, or cosmological metrics; automated
arbitrary-coordinate conversion; observational likelihoods.

## 9. Category labels and honesty

- The estimators and the extrapolation are a **numerical approximation**; the
  extrapolation is never labelled exact.
- The synthetic-metric recoveries and the isotropic-Schwarzschild convergence are
  **validation oracles** for the *implementation* — passing the GR oracle
  validates the code, not an alternative theory.
- The exact synthetic coefficients and the isotropic-Schwarzschild series are
  **exact analytical** inputs.
- Estimated uncertainty is **not** automatically a rigorous bound; observational
  consistency would require a separate data-analysis layer (Layer 5), out of
  scope here.

## 10. Placement

The `ppn` module lives in `scirust-relativity` (as linearized gravity does; a
dedicated `scirust-gravity` crate remains an option if the Layer 2 surface
grows), with a dedicated `PpnError`, a `ppn`-scoped module tree, a
`linearized_gravity`-style experiment, and `criterion` benches — all reusing the
existing conventions.

## 11. Hardening addendum: decomposed diagnostics

This section records a follow-up hardening of the increment above (§1–10, all
unchanged): the single blended `estimated_uncertainty` is now backed by an
explicit, independently-inspectable diagnostics surface, so a caller can see
*which* sensitivity axis is driving a reported uncertainty rather than only its
conservative maximum.

**New types** (`ppn` module): `ParameterSensitivity { deviation, available }` — a
single probe's absolute intercept deviation, with `available: false` (never a
misleading `deviation: 0.0`) when the perturbed fit could not be attempted at
all (too few points after windowing/thinning, or an ill-conditioned sub-system).
`PpnParameterDiagnostics { radial_window_sensitivity, fit_order_sensitivity,
resolution_sensitivity, conditioning_class }` — the three probes broken out
individually (previously folded into one `worst`-case number inside
`ParameterEstimate.estimated_uncertainty`, which is retained unchanged as the
conservative summary: the maximum of whichever probes were available, or the
fit residual if none was). `ConditioningClass` (`WellConditioned` / `Marginal` /
`IllConditioned`) via `classify_conditioning(indicator)`, with the marginal
threshold `CONDITIONING_MARGINAL_THRESHOLD = 1e-6` four orders above the
existing hard-reject floor (`1e-10`). `ParameterEstimate` gained a `diagnostics:
PpnParameterDiagnostics` field; `PpnEstimate` gained `mass_scale: f64`,
completing the weak-field domain summary already carried by `sample_count`,
`compactness_min`, and `compactness_max`.

**Two deliberate non-changes, both licensed by "adapt naming and shape to
repository conventions":**

- **No fourth `Rejected` conditioning variant.** A fit below the hard-reject
  floor never reaches a `ParameterEstimate` — `fit_polynomial_intercept` already
  returns `Err(PpnError::IllConditionedFit)` before one is constructed. Adding a
  `Rejected` variant to a successful result's diagnostics would be a
  contradiction in terms; `IllConditioned` documents that boundary and is
  independently unit-tested via `classify_conditioning`, but is unreachable from
  a successful extraction.
- **No separate `SamplingSpacing` enum.** `PpnSampling` already has
  `UniformCompactness` (linear in the dimensionless weak-field variable `U`) and
  `LogarithmicRadius` as distinct variants, which states the sampled unit in the
  variant name rather than needing a second enum to disambiguate it. The
  `ppn_extraction` experiment sweeps both.
- **No `scientific_category` CSV column.** Every existing experiment in the
  suite carries its scientific-category label in the `#`-prefixed header
  (`print_experiment_header`'s `layer` argument) and in prose, never as a
  per-row data column; the `ppn_extraction` CSV's `metric` field (for example
  `synthetic_exact_gr` vs `isotropic_schwarzschild` vs `areal_schwarzschild`)
  already identifies which oracle class a row belongs to. Adding a redundant
  column would break with that established convention.

**Experiment.** `ppn_extraction` now also sweeps: a second exact-synthetic
oracle away from `(1, 1)` (`gamma_star = 0.8, beta_star = 1.2`, avoiding the
accidental symmetry a GR-only sweep could hide); the `LogarithmicRadius`
sampling spacing; and reports, per parameter, `*_relative_error` (against the
known oracle), the three sensitivity axes (`na` when unavailable, never a
misleading `0`), and `*_conditioning_class`, alongside the pre-existing
`*_estimated_uncertainty` and `*_fit_residual` columns.

**Nothing in §1–10 changed.** The physics, the coordinate contract, the solver,
and all four oracles are exactly as validated in the original increment; this
section only makes the numerical honesty already designed in §6 more
explicit and independently testable.
