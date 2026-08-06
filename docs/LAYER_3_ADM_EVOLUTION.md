# Layer 3.1 — ADM constraint and evolution core: design note

This note fixes the opening Layer 3 (Numerical Relativity) increment before any
code lands. Layer 3 was, until now, **absent** (`docs/PLATFORM_ARCHITECTURE.md`
§8: "no perturbation theory, self-force, or ADM/BSSN evolution"). This slice is
the minimal, deterministic, oracle-validated bridge from Layer 2's ADM
*kinematics* (`docs/LAYER_2_ADM.md`) to future time evolution.

## 1. Why this is not a duplicate of Layer 2's `adm` module

Layer 2's `adm_decomposition`/`adm_constraints` (`scirust-relativity/src/adm.rs`)
**extract** the lapse, shift, spatial metric, and extrinsic curvature *backward*
from an already-known analytic 4-metric, by finite-differencing that metric in
time and space. They validate that a *known solution* satisfies the constraints
— useful for checking an exact spacetime, but they need the whole 4D spacetime
already in hand and cannot be handed independently-evolving 3+1 data.

This increment is the opposite, *forward* direction: given the spatial metric,
the extrinsic curvature, the lapse, and the shift as **independently specified
data** (plus matter sources), evaluate the constraint residuals and the
evolution right-hand sides. This is what a time integrator will need to step
`(gamma_ij, K_ij)` forward — it takes no already-known 4-metric as input at all.
Nothing is duplicated: the *quantities* (`R^(3)`, the Gauss-Codazzi form) are
the same physics Layer 2 already validates, but the *data flow* is inverted,
which is precisely the missing capability for Layer 3.

## 2. Scope

**In scope**: the standard 3+1 ADM equations of established general relativity,
as pure local (single-point) evaluators:

- typed matter-source projections (`AdmSources`);
- the Hamiltonian constraint, decomposed into its contributing terms;
- the momentum constraint, decomposed into its contributing terms;
- the right-hand side of `partial_t gamma_ij`;
- the right-hand side of `partial_t K_ij`, decomposed into its contributing terms.

**Explicitly out of scope** (documented, not attempted, and not to be conflated
with what *is* delivered):

- **Time integration.** These are right-hand-side *evaluators* at an instant,
  not a stepper; no `scirust-sim`-style time loop is introduced.
- **A grid, mesh, or finite-difference stencil across a spatial domain.** Every
  function here takes one point; extending to a discretized 3D domain (needed
  for a real solver) is future work.
- **BSSN** (or any reformulation for numerical stability under long-time
  evolution) — ADM is well known to be only weakly hyperbolic and numerically
  unstable for binary-black-hole-scale evolutions; BSSN (or generalized
  harmonic) exists specifically to fix this. Introducing it now, before a grid
  or integrator exists to need it, would be premature.
- **Gauge/slicing conditions** (1+log lapse, Gamma-driver shift, ...). The
  lapse and shift are free data here, supplied by the caller.
- **Constraint damping** (e.g. the "Z4"-family constraint-violation-suppressing
  terms). Out of scope until a real evolution exists to damp.
- **Mesh refinement, wave extraction, black-hole horizons/mergers.**

## 3. Conventions (must match Layer 2's `adm` module exactly)

Signature `(-,+,+,+)`, geometric units `G = c = 1`, spatial indices
`i, j, k in {1,2,3}`. The extrinsic curvature sign is **fixed by Layer 2** and
not renegotiable here:

```text
K_ij = -1/(2N) ( partial_t gamma_ij - D_i N_j - D_j N_i )     (docs/LAYER_2_ADM.md)
```

equivalently `K_ij = -gamma_i^mu gamma_j^nu grad_mu n_nu` for the future-pointing
unit normal `n`. Inverting this definition gives the metric evolution equation
(oracle-free — it is an algebraic identity of the definition above):

```text
partial_t gamma_ij = -2 alpha K_ij + (Lie_beta gamma)_ij
```

which matches the mission specification exactly and requires no sign check.

### 3.1 Matter projections

```text
rho   = T_(mu nu) n^mu n^nu                        (energy density)
S_i   = -gamma_(i mu) n_nu T^(mu nu)                (momentum density)
S_ij  = gamma_(i mu) gamma_(j nu) T^(mu nu)          (spatial stress)
S     = gamma^(ij) S_ij                              (spatial trace)
```

`AdmSources` stores `rho`, `S_i`, `S_ij` as data (this increment does not derive
them from a general 4D `T_(mu nu)` field — see §6); `S` is a method, not a
stored field, since it depends on the inverse metric at the query point.

### 3.2 Hamiltonian and momentum constraints

```text
Hamiltonian:  R^(3) + K^2 - K_ij K^(ij) - 16 pi rho = 0
Momentum:     D_j ( K^(ij) - gamma^(ij) K ) - 8 pi S^i = 0
```

Both match the mission specification and Layer 2's vacuum form exactly
(Layer 2's `-2 Lambda` is the special case `Lambda = 8 pi rho_vacuum`, confirmed
in §4 below) — **no sign correction needed here**.

### 3.3 The extrinsic-curvature evolution equation — corrected sign

The mission's literal specification writes the matter term with a **`+`** sign:

```text
partial_t K_ij = ... + Lie_beta K_ij + 8 pi alpha ( S_ij - 1/2 gamma_ij (S - rho) )   [as literally given]
```

**This sign is wrong for this repository's `K_ij` convention** and has been
corrected after independent numerical verification (§4). The implemented
equation is:

```text
partial_t K_ij = -D_i D_j alpha
               + alpha ( R_ij + K K_ij - 2 K_ik K^k_j )
               + (Lie_beta K)_ij
               - 8 pi alpha ( S_ij - 1/2 gamma_ij (S - rho) )
```

with **`D_i D_j alpha = partial_i partial_j alpha - Gamma^(3)k_ij partial_k alpha`**
(the covariant Hessian, using the *spatial* Christoffel symbols) and
`(Lie_beta T)_ij = beta^k partial_k T_ij + T_kj partial_i beta^k + T_ik partial_j beta^k`
for any symmetric spatial 2-tensor `T` (used for both `gamma` and `K`).

## 4. Independent numerical verification of the sign (not taken on faith)

Given how easy extrinsic-curvature sign errors are to introduce, and per the
mission's explicit instruction not to copy the formula blindly, the sign above
was **verified numerically against an exact solution before being written into
any implementation**, using a throwaway probe (not part of the final diff):

Exponential FLRW (`Flrw` + `ExponentialScaleFactor`) is de Sitter in cosmological
slicing: `N = 1`, `N^i = 0`, `R^(3) = 0`, `K_ij = -H gamma_ij` (already validated
in Layer 2). Representing its cosmological constant as an equivalent perfect
fluid (`rho = Lambda / 8 pi`, `p = -rho`, `S_ij = p gamma_ij`, exact since a
`Lambda` term and vacuum energy are mathematically interchangeable in Einstein's
equations), the probe:

1. computed `partial_t K_ij` **directly** by central time-differencing Layer 2's
   already-validated `adm_decomposition` at `t +/- dt` (independent of any new
   code — pure ground truth from the existing, tested extraction), and
2. evaluated **both** candidate signs of the new closed-form RHS at `t`, using
   the `K_ij`, `gamma_ij`, `R^(3)` already read off the same decomposition.

Result (`H = 0.5`, `dt = 1e-4`):

| convention | max\|direct - candidate\| |
|---|---|
| `- 8 pi alpha (...)` (this note's choice) | `5.3e-9` (finite-difference floor) |
| `+ 8 pi alpha (...)` (mission's literal text) | `1.83` (wrong by a full sign flip, not a rounding issue) |

This is decisive: the `+` sign is not a rounding-level discrepancy but a
qualitatively wrong prediction (indeed the wrong-sign candidate is
`-2` times the correct answer here). The `-` sign is adopted and implemented.
A second, independent oracle (Oracle B, §5) checks the *other* half of the same
equation — the `-D_i D_j alpha + alpha R_ij` combination — using static
Schwarzschild, where a genuinely time-*independent* solution forces this
combination to vanish exactly; together the two oracles exercise every additive
term in the equation.

## 5. Oracles

- **Oracle A — Minkowski.** Flat spatial metric (reusing `Minkowski`'s spatial
  block), zero extrinsic curvature, unit lapse (`sqrt(-g_00)` of `Minkowski`,
  which is `1`), zero shift, vacuum sources. Expected: both constraints and both
  evolution right-hand sides are zero to the finite-difference floor.
- **Oracle B — static Schwarzschild slice (time-symmetric).** The spatial block
  and lapse `sqrt(-g_00)` are read directly from the existing, validated
  `Schwarzschild` background (not re-derived by hand); extrinsic curvature is
  declared zero (the time-symmetric slicing), shift is zero, sources are
  vacuum. Because this *is* an exact, genuinely time-*independent* vacuum
  solution, `partial_t K_ij` must vanish identically — which forces
  `-D_i D_j alpha + alpha R_ij = 0` (the quadratic-`K` and matter terms already
  vanish identically here). This is the oracle that pins the Hessian/Ricci half
  of the equation, complementing the FLRW check in §4, which pinned the
  quadratic-`K`/matter half. The Ricci **tensor** used is Schwarzschild's
  spatial slice's `R_ij` (new: `ricci_tensor_from_metric`, §6), which is
  nonzero and anisotropic even though its trace `R^(3)` is exactly zero
  (a scalar-flat, non-Ricci-flat 3-geometry) — a nontrivial, independent check.
- **Oracle C — flat FLRW.** `gamma_ij = a(t)^2 delta_ij`, `K_ij = -H(t) gamma_ij`
  (`H` from the existing `Flrw::hubble_parameter`, not re-derived), unit lapse,
  zero shift. The Hamiltonian constraint reduces to the (first) Friedmann
  equation `K^2 - K_ij K^(ij) = 6 H^2 = 16 pi rho` for a source with
  `rho = 3 H^2 / (8 pi)` (matching `Lambda = 3 H^2`, established in Layer 2).
- **Oracle D — deliberate constraint violation.** Flat space, hand-chosen
  extrinsic-curvature perturbations that are **not** consistent with any
  vacuum solution: (i) a *constant* `K_ij = diag(eps, eps, 0)` gives a clean,
  closed-form nonzero Hamiltonian residual `2 eps^2` (constant `K` makes the
  momentum residual identically zero, isolating the Hamiltonian check); (ii) a
  *position-dependent, traceless* `K_ij = diag(eps x, -eps x/2, -eps x/2)` gives
  a clean, closed-form, **exact** (linear-in-`x`, so the central difference is
  exact) nonzero momentum residual `M^1 = eps` (zero Hamiltonian contamination,
  since the trace is exactly zero). Both are checked at several `eps` and shown
  to scale monotonically, and the diagnostic decomposition is checked to
  attribute the violation to the correct term.
- **Gauge unit tests (isolated from the physics oracles).** To validate the
  Lie-derivative and lapse-Hessian machinery *itself*, independent of whether a
  configuration is a genuine GR solution: a quadratic lapse
  `alpha = 1 + a r^2` on flat space gives an exact (to floating point) Hessian
  `2 a delta_ij`; a linear shift `beta^i = M^i_j x^j` with a constant tensor
  field gives an exact (Lie derivative of a linear vector field under central
  differences has no truncation error) `(Lie_beta T)_ij = (M + M^T)_ij` (times
  the tensor's proportionality constant for `K`). These give zero-truncation,
  fully deterministic checks of the differentiation machinery, decoupled from
  needing a real curved+gauge exact solution (which is hard to construct by
  hand for a nontrivial gauge).

## 6. Representation and abstractions

Three small traits give the extension point the mission asks for, without
committing to any single field representation:

```rust
pub trait SpatialScalarField { fn value(&self, coordinates: &[f64; 3]) -> f64; }
pub trait SpatialVectorField { fn components(&self, coordinates: &[f64; 3]) -> [f64; 3]; }
pub trait SpatialTensorField { fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3]; }
```

`SpatialTensorField` is deliberately **not** `Metric<3>`: an extrinsic curvature
need not be positive-definite or invertible, so reusing `Metric<3>` for it would
be misleading (a reader could reasonably expect `invert_metric` to apply). Each
trait has one blanket closure implementation
(`impl<F: Fn(&[f64;3]) -> T> Trait for F`), so oracles and callers can pass
closures directly; each is a genuinely new, minimal abstraction (not a
duplicate of anything existing), and none is `dyn`-boxed or tied to a mesh.

The spatial metric itself reuses the **existing** `Metric<3>` trait directly
(exactly as Layer 2's `adm` module already does for its `SpatialSlice`
adapter) — no new metric abstraction.

**Derivative interface.** The mission suggests a `SpatialDerivativeProvider`
abstraction. This increment does **not** introduce one, deliberately: finite
differences are the only differentiation method anywhere in Layers 1–2 today,
so a provider trait with a single implementation would be premature abstraction
(this platform's own stated engineering value: no abstraction beyond what the
task requires). Instead:

- first/second derivatives of a scalar, first derivatives of a vector, and
  first derivatives of a rank-2 tensor are four small, private, deterministic
  central-difference helper functions (following the exact inline-FD idiom
  already used throughout `curvature.rs`, `action.rs`, and `adm.rs` — this
  repository does not have, and does not otherwise use, a shared generic FD
  utility module);
- the spatial Christoffel symbols reuse the existing, dimension-generic
  [`numerical_christoffel`] at `D = 3` (exactly as Layer 2's `adm` module does);
- the spatial Ricci **tensor** is new (`ricci_tensor_from_metric`, added to
  `curvature.rs` alongside the existing `ricci_scalar_from_metric`, which is
  refactored to a thin wrapper around it — zero duplication, and the existing
  function's signature and behavior are unchanged); the spatial Ricci scalar
  continues to reuse the existing `ricci_scalar_from_metric` directly.

If a second differentiation method (spectral, analytic, automatic
differentiation) is ever needed, it can be introduced without changing any of
the four public constraint/evolution function signatures, since they depend
only on the three field traits above, never on how those fields are
differentiated.

## 7. Diagnostics

Every result is a struct exposing the total **and** each additive contributing
term (never only a blended scalar), matching the mission's requirement:

- `HamiltonianConstraint`: `signed_residual`, `absolute_residual`,
  `normalized_residual: Option<f64>` (`None` when the normalization scale is
  too small to be meaningful — an honest "undefined", not a fabricated zero),
  `spatial_ricci_scalar`, `mean_curvature_squared`, `extrinsic_curvature_norm`,
  `matter_term`, `scale`.
- `MomentumConstraint`: `residual: [f64; 3]` (also the per-component view),
  `residual_norm`, `geometric_term`, `matter_term`.
- `MetricEvolutionRhs`: `total`, `extrinsic_curvature_term`, `lie_derivative`.
- `CurvatureEvolutionRhs`: `total`, `lapse_hessian` (the raw `D_i D_j alpha`,
  not pre-negated, so it can be checked against a hand-derived Hessian
  directly), `ricci_term` (`alpha * R_ij`), `quadratic_extrinsic_term`
  (`alpha * (K K_ij - 2 K_ik K^k_j)`), `lie_derivative`, `matter_term`. `total`
  is documented to assemble as
  `total = -lapse_hessian + ricci_term + quadratic_extrinsic_term + lie_derivative + matter_term`.

No conditioning *classification* (well-conditioned/marginal/ill-conditioned) is
introduced: unlike the PPN polynomial fit or the action-variation quadrature,
there is no fit or iterative solve here to classify — every quantity is a
direct, closed-form evaluation at a point. Where the mission asks for
"conditioning... or documented limitation if not meaningful yet," this is that
documented limitation: the raw finite-difference steps and the residual scale
are the only meaningful diagnostics available at this stage.

## 8. Errors

`AdmEvolutionError` is scoped to failure modes this concrete implementation can
actually produce (no dead variants for mission-suggested cases this design
makes structurally impossible):

- `NonFiniteCoordinate`, `InvalidStep` — request validation.
- `SingularSpatialMetric` — covers both "singular metric" and "failed index
  raising" (raising needs the same inverse, so they share one failure point).
- `NonFiniteField { field }` — a supplied lapse/shift/extrinsic-curvature field
  evaluated to a non-finite value.
- `NonFiniteResult { quantity }` — an assembled output quantity is non-finite.
- `Curvature(RelativityError)` — a propagated failure from
  `ricci_tensor_from_metric` / `ricci_scalar_from_metric` / `numerical_christoffel`.

"Unsupported convention or source representation" is not a variant: this
increment fixes one convention throughout (§3), with no runtime-selectable
alternative to reject. "Inconsistent tensor dimensions" cannot occur: every
tensor here is a fixed-size `[[f64; 3]; 3]`, so mismatched dimensions are a
compile-time impossibility, not a runtime error.

## 9. Deliverable shape and placement

New module `scirust-relativity/src/adm_evolution.rs` (`pub mod adm_evolution;`),
living directly in `scirust-relativity` beside `action` and `adm` — not a new
crate, following the same "reuse-first, no new crate until a slice needs one"
placement already established for Layer 2. Tests in
`scirust-relativity/tests/adm_evolution.rs`, benchmarks in
`scirust-relativity/benches/adm_evolution.rs`, and the deterministic
`adm_constraint_sweep` experiment.

The follow-on after this slice (not part of it) is a discretized spatial grid
and a time integrator to actually evolve `(gamma_ij, K_ij)` — at which point
ADM's weak hyperbolicity will make a BSSN-style reformulation the natural next
design note.
