# SciRust Nonlocal Relativity — Platform Roadmap

This document states the long-horizon vision for turning the current
experimental worldline simulator into a broad, rigorous, open-source platform
for relativistic geometry and gravitational physics, and — just as importantly
— it draws a hard line between **what is built and validated today** and **what
is planned**. It is a roadmap, not a status report; the authoritative
per-feature status lives in
[`NONLOCAL_RELATIVITY_V2_STATUS.md`](NONLOCAL_RELATIVITY_V2_STATUS.md),
[`EXPERIMENTAL_NONLOCAL_RELATIVITY.md`](EXPERIMENTAL_NONLOCAL_RELATIVITY.md),
and the crate `README`s.

The objective is not to advance a new theory of gravity. It is to build the
most rigorous, deterministic, auditable, and reusable open-source foundation
for relativistic geometry and gravitational-physics computation that we can,
and to grow it one *validated* increment at a time. Scientific integrity takes
precedence over novelty. Negative results are kept, not discarded.

## Non-negotiable engineering and scientific principles

These constraints govern every increment and are not traded away for scope:

- 100% Rust, zero FFI, zero `unsafe` unless mathematically unavoidable (and
  none is used today).
- No `TODO`, `FIXME`, placeholder, or stub implementations in the hardened
  crates; no panic-based control flow; typed errors everywhere.
- Deterministic execution — no RNG and no wall-clock in library or experiment
  code paths — and bit-for-bit reproducibility wherever mathematically
  possible.
- CI stays green: `cargo fmt --check`, `cargo clippy -D warnings`, tests,
  doctests, and a forbidden-marker scan over the hardened sources.
- Every algorithm is referenced to the literature; every numerical claim is
  backed by a reproducible experiment or test against a known oracle.
- No claim of new physics without both a mathematical derivation and
  observational validation. Established general relativity, phenomenological
  models, and speculative extensions are labelled as such and never blurred.

## The classification discipline

Every result the platform produces belongs to exactly one of the following
categories, and each is labelled explicitly in code, docs, and experiment
headers. Blurring these categories is treated as a defect.

1. **Exact mathematical result** — a closed-form identity (e.g. the
   Kretschmann scalar `48 M^2 / r^6` of Schwarzschild, or `R = 4 Lambda` for a
   4D maximally symmetric spacetime).
2. **Numerical implementation of established physics** — a deterministic
   computation whose target is textbook general relativity, validated against
   an exact oracle to a stated tolerance (e.g. the finite-difference curvature
   engine).
3. **Numerical approximation** — a scheme with a disclosed, quantified error
   (e.g. finite-difference truncation, bounded-memory truncation).
4. **Phenomenological model** — a deterministic but uncalibrated construction
   with free parameters and no claim of physical law (e.g. the curvature-memory
   modulation hooks in the experimental worldline layer).
5. **Speculative model** — an extension explored for its structure, explicitly
   not asserted to describe nature.
6. **Empirically validated physics** — a claim checked against observation.
   The platform currently makes **none** of these.

## The six-layer vision

| Layer | Scope | Status |
|------|-------|--------|
| 1 — Geometry Core | manifolds, metrics, tetrads, connections, curvature tensors, geodesics, parallel transport, bitensors, world function, geometry diagnostics | **partially delivered** (see below) |
| 2 — Covariant Gravity Workbench | symbolic actions, variational calculus, automatic field-equation derivation, weak-field / PPN / cosmological limits, stability & ghost checks | near-term slices delivered: linearized gravity, PPN `gamma`/`beta`, Einstein–Hilbert action variation, and 3+1 (ADM) kinematics ([`LAYER_2_COVARIANT_GRAVITY.md`](LAYER_2_COVARIANT_GRAVITY.md), [`LAYER_2_PPN.md`](LAYER_2_PPN.md), [`LAYER_2_ACTION_VARIATION.md`](LAYER_2_ACTION_VARIATION.md), [`LAYER_2_ADM.md`](LAYER_2_ADM.md)); bridges to Layer 3 |
| 3 — Numerical Relativity | linear perturbations, self-force, EMRI; then ADM/BSSN, constraint damping, AMR, wave extraction | **opening**: ADM constraint and evolution core ([`LAYER_3_ADM_EVOLUTION.md`](LAYER_3_ADM_EVOLUTION.md)), homogeneous ADM time evolution ([`LAYER_3_HOMOGENEOUS_EVOLUTION.md`](LAYER_3_HOMOGENEOUS_EVOLUTION.md)), the BSSN formulation core ([`LAYER_3_BSSN.md`](LAYER_3_BSSN.md)), and a periodic 1D BSSN grid ([`LAYER_3_BSSN_PERIODIC_1D.md`](LAYER_3_BSSN_PERIODIC_1D.md)) delivered — the last stable across every resolution tested once `Rtilde_ij` was written in genuine BSSN form, and now carrying the full moving-puncture gauge — 1+log slicing and the Gamma-driver — each validated against a closed form; two spatial dimensions delivered and validated ([`LAYER_3_BSSN_PERIODIC_2D.md`](LAYER_3_BSSN_PERIODIC_2D.md)); the **third dimension** now validated too ([`LAYER_3_BSSN_PERIODIC_3D.md`](LAYER_3_BSSN_PERIODIC_3D.md)), carrying a transverse-traceless gravitational wave along the body diagonal — the first configuration exercising all six metric components with every axis varying — at order 1.94, so the grid substrate is complete and what remains is physics rather than plumbing; strong fields, punctures, radiative boundaries, and constraint damping are next |
| 4 — Gravitational Memory Lab | standard / Christodoulou / fractional memory, observer and detector response | partially explored in the experimental worldline layer (phenomenological) |
| 5 — Astrophysical Inference | waveform generation, noise models, likelihood, MCMC / nested sampling, matched filtering | planned |
| 6 — Relativistic Navigation | proper time, Shapiro delay, redshift, GNSS corrections, filtering across Earth/Moon/Mars/Sun/deep space | planned |

"Planned" means exactly that: not started, or present only as an experimental,
clearly-labelled prototype. The table is revised only when a layer's status
genuinely changes, and only downward-conservatively.

## Current status — what is actually built and validated

The following exists today, is tested, and ships in a green CI:

**Layer 1 (Geometry Core), delivered subset:**

- Trait-based backgrounds: `Metric<D>` and `Connection<D>` over const-generic
  dimension, with deterministic metric inversion (`invert_metric`) and typed
  errors (`RelativityError`).
- Backgrounds with analytic connections: Minkowski, spherical-chart Minkowski,
  Schwarzschild, Reissner–Nordström, de Sitter, anti-de Sitter, and spatially
  flat FLRW (generic over a scale factor); Kerr and isotropic-coordinate
  Schwarzschild with a finite-difference connection whose truncation cost is
  measured and disclosed.
- A shared lapse-metric helper (`static_spherical`) for static, spherically
  symmetric `f(r)` spacetimes, reused by the de Sitter / anti-de Sitter and
  spherical-Minkowski backgrounds.
- **Curvature engine** (`CurvatureTensors`): Riemann, Ricci, Ricci scalar,
  Einstein, and Kretschmann tensors from any `Metric + Connection` background
  by central finite differences of the Christoffel symbols, with typed errors
  and no non-finite results.
- **Exact-oracle validation** of the curvature engine (category 2 against
  category 1):
  - Minkowski: every curvature component is *exactly* zero.
  - Schwarzschild: Ricci-flat, and `K = 48 M^2 / r^6` (e.g. exactly
    `1.8310546875e-4` at `M = 1, r = 8`) to finite-difference tolerance.
  - de Sitter / anti-de Sitter: `R_(mu nu) = Lambda g_(mu nu)`,
    `R = 4 Lambda`, `G_(mu nu) = -Lambda g_(mu nu)`, `K = 8 Lambda^2 / 3`,
    and the maximally-symmetric Riemann form
    `R_(abcd) = (Lambda / 3)(g_(ac) g_(bd) - g_(ad) g_(bc))`.
  - Riemann index symmetries and the first Bianchi identity.
  - A `curvature_invariants` experiment that reports invariants against these
    oracles and sweeps the finite-difference step to expose the `O(h^2)`
    truncation-vs-roundoff trade-off.
- **Coordinate-independence diagnostics**: scalar invariants (`R`, `K`) agree
  across two charts of the same geometry — Cartesian vs spherical Minkowski, and
  areal vs isotropic Schwarzschild (where `K = 48 M^2 / r^6` uses the *areal*
  radius `r = rho (1 + M/2rho)^2`). Spherical Minkowski is a strong flatness
  test: non-zero Christoffel symbols yielding numerically zero curvature.
  Validated by tests and a `coordinate_independence` experiment.
- **Parallel-transport engine**: transport of contravariant vectors, covectors,
  and rank-2 covariant tensors along a coordinate path (segment / polyline /
  closed-loop holonomy), reusing the `scirust-sim` RK4 integrator. Validated by
  flat-space exactness, metric compatibility (vector-norm preservation, metric
  self-transport, index-lowering commutation, contraction preservation), zero
  holonomy on flat closed loops, and the holonomy/curvature identity checked
  against the numerical Riemann tensor.
- **Geodesic-deviation (Jacobi) fields**: the covariant Jacobi system
  integrated with the Riemann source, validated against the actual separation
  of nearby geodesics (and exact flat linear growth), with a focusing
  experiment across de Sitter / anti-de Sitter / Schwarzschild.
- **Exponential / logarithm maps**: `exp_p(v)` (geodesic endpoint at unit
  affine parameter) and `log_p(q)` (its Newton-shooting inverse), validated by
  flat exactness and the curved round-trip identity.
- **Local orthonormal frames (tetrads)**: `orthonormal_tetrad` builds an
  observer frame `{e_a}` with `g(e_a, e_b) = eta_ab` by metric Gram-Schmidt,
  validated by orthonormality, completeness, and preservation under parallel
  transport; the experimental worldline observer tetrad delegates to it (one copy
  of the algorithm across both crates).
- **Synge's world function**: `world_function` / `world_function_with_gradients`
  give `sigma(x', x)` and its gradient bitensors `sigma^{mu'}`, `sigma^mu` from
  the geodesic logarithm map, validated by flat exactness, base/field symmetry,
  the fundamental identity `2 sigma = g sigma^mu sigma^mu`, and the gradient
  round trip.
- **Van Vleck–Morette determinant**: `van_vleck_determinant` (on a deterministic
  `determinant` primitive) from the exponential-map Jacobian, validated by flat /
  coincidence unity, the `Delta(x', x) = Delta(x, x')` symmetry, and the known
  maximally-symmetric leading expansion `(Delta - 1)/sigma -> Lambda/3`.
- Geodesic integration (`GeodesicSystem`) compatible with `scirust-sim`.

An engineering map of the whole relativistic-computation subgraph — capabilities,
validated mathematics, technical debt, duplication, and extension points — is
maintained in [`PLATFORM_ARCHITECTURE.md`](PLATFORM_ARCHITECTURE.md).

**Layer 4 (memory), experimental and phenomenological only:** the
`scirust-nonlocal-relativity` crate integrates fractional-memory *test-particle*
worldlines on a fixed background. It does **not** modify the field equations
and claims **no** physical validity; see its status document. It is listed here
as an existing prototype, not a delivered platform layer.

Everything not in this section is not yet built.

## Increment plan

The platform grows by small, individually-validated pull requests. Each lands
only with tests or experiments against an oracle, green CI, and honest labels.
The near-term ordering within Layer 1:

1. **Curvature core + de Sitter / anti-de Sitter** — *delivered.* Riemann
   through Kretschmann, validated against exact maximally-symmetric and
   Schwarzschild oracles.
2. **Coordinate-independence / invariant diagnostics** — *delivered.* Scalar
   invariants agree across charts of the same geometry (Cartesian/spherical
   Minkowski, areal/isotropic Schwarzschild), with `MinkowskiSpherical` and
   `IsotropicSchwarzschild` backgrounds and a `coordinate_independence`
   experiment.
3. **Consolidate the duplicated non-uniform Caputo memory helpers** — *delivered.*
   Both callers (the fixed-step `MemoryLaw` impls and the embedded adaptive
   controller) now delegate to one shared `nonuniform_kernel` module; the logic
   was moved verbatim, so the crate's bit-identity golden tests still pass.
4. **Reusable parallel-transport engine** — *delivered.* Transport of a vector
   along a coordinate path (reusing the `scirust-sim` RK4 integrator), with
   holonomy, validated by flat-space exactness, metric compatibility, zero
   holonomy on flat closed loops, and the holonomy/curvature identity
   `Delta V = -R V A B` checked against the numerical Riemann tensor
   (`parallel_transport` tests + experiment).
5. **FLRW cosmological background** — *delivered.* Spatially flat FLRW generic
   over a `ScaleFactor` (exponential = de Sitter; power-law = radiation/matter
   eras), validated against the exact Friedmann curvature formulas and against
   the static de Sitter chart (coordinate independence).
6. **Geodesic-deviation (Jacobi) fields** — *delivered.* The covariant Jacobi
   system `(x, u, xi, w)` integrated with the `scirust-sim` RK4 engine and the
   Riemann source, validated against the convention-free ground truth (the
   coordinate separation of two nearby geodesics) and exact flat linear growth.
7. **Geodesic exponential / logarithm maps** — *delivered.* `exp_p(v)` follows
   the geodesic for unit affine parameter; `log_p(q)` inverts it by Newton
   shooting (the exponential's finite-difference Jacobian inverted with the
   crate's Gauss-Jordan routine). Validated by flat exactness and the curved
   round-trip identity.
8. **Covector and rank-2 tensor parallel transport** — *delivered.* The
   transport engine extended to lower-index objects, validated by the three
   metric-compatibility signatures: the metric transports to itself, index
   lowering commutes with transport, and the covector-vector contraction is
   preserved (`covariant_transport` tests + experiment).
9. **Tetrads (orthonormal frames) in the geometry core** — *delivered.*
   `orthonormal_tetrad` builds a local orthonormal frame `{e_a}` for a timelike
   observer by metric Gram-Schmidt (`e_0` the normalized four-velocity), with a
   typed `RelativityError` for every failure. This *generalises* the
   experimental worldline crate's observer tetrad instead of duplicating it: the
   worldline's `build_orthonormal_tetrad` now delegates to this primitive and
   re-exports the shared `OrthonormalTetrad` type, keeping its public API and all
   four of its tetrad tests bit-for-bit unchanged (the Gram-Schmidt lives in
   exactly one place). Validated by orthonormality (`g(e_a,e_b) = eta_ab`),
   completeness (any vector reconstructs from its frame components), agreement of
   the frame temporal/spatial split with the closed-form metric split, and
   preservation of orthonormality under parallel transport of the frame legs
   (`tetrad` tests + `orthonormal_tetrad` experiment).
10. **Synge's world function and its gradient bitensors** — *delivered.*
    `world_function` and `world_function_with_gradients` give
    `sigma(x', x) = (1/2) g(x')(v, v)` with `v = log_{x'}(x)` (reusing the
    geodesic logarithm map), plus the gradient bitensors `sigma^{mu'} = -v` and
    `sigma^mu = -log_x(x')`. Validated by flat exactness (`sigma` and both
    gradients exact), base/field symmetry, the field-point fundamental identity
    `2 sigma = g(x) sigma^mu sigma^mu` (an independent shooting from `x`), the
    gradient round trip, and quadratic vanishing near coincidence (`synge` tests
    + `world_function` experiment).
11. **The van Vleck–Morette determinant** — *delivered.* `van_vleck_determinant`
    computes `Delta(x', x) = sqrt|g(x')| / (sqrt|g(x)| det J)` from the
    exponential-map Jacobian `J = d exp_{x'}(v)/dv` (the Lorentzian-correct form
    of `-det(sigma_{mu nu'}) / sqrt(g g')`), on a new deterministic `determinant`
    primitive. Validated by flat and coincidence unity, the symmetry
    `Delta(x', x) = Delta(x, x')` (an independent-Jacobian cross-check), and the
    known maximally-symmetric leading expansion `(Delta - 1)/sigma -> Lambda/3`
    for de Sitter / anti-de Sitter (`van_vleck` tests + `van_vleck_determinant`
    experiment).
12. **First performance benchmarks** — *delivered.* `criterion` wall-clock
    benches for the geometry-core hot paths (`scirust-relativity/benches/geometry_core.rs`:
    Christoffel, metric inversion, the finite-difference curvature engine, RK4
    transport, and the world-function / van Vleck shooting) and the `O(N^2)`
    Caputo history (`scirust-nonlocal-relativity/benches/caputo_history.rs`, whose
    time roughly quadruples per doubling of `N`). Benchmark **timing is
    machine-dependent and not bit-reproducible** — that is inherent; the library
    functions stay deterministic, and the reproducible companion is the
    operation-count proxy in `complexity_scaling`. This is the last near-term
    Layer 1 item; Layers 2–6 open next.

Layers 2–6 begin only after Layer 1 is broad and solid. Each will open with a
design note fixing its oracles and its category labels before any code lands.

With the near-term Layer 1 sequence (items 1–12) delivered, **Layer 2 (Covariant
Gravity Workbench) is now open**, on its design note,
[`LAYER_2_COVARIANT_GRAVITY.md`](LAYER_2_COVARIANT_GRAVITY.md). Its first
increment — **linearized gravity** — is **delivered**: [`LinearizedField`]
computes the linearized Riemann/Ricci/Einstein tensors and the trace-reversed
perturbation (whose Lorenz-gauge vacuum equation is the wave operator) to first
order in a metric perturbation about Minkowski, validated by weak-field-Schwarzschild
vacuum, the Newtonian Poisson limit (`G^(1)_00 = 2 nabla^2 Phi`, exact for a
polynomial potential), gauge invariance of the linearized Riemann, and an
`O(h^2)` cross-check against the Layer 1 nonlinear curvature (`linearized` tests +
`linearized_gravity` experiment).

Its second increment — **PPN parameter extraction** — is also **delivered**
(design & conventions: [`LAYER_2_PPN.md`](LAYER_2_PPN.md)). The `ppn` module
extracts the Eddington–Robertson `gamma` and `beta` from static, spherically
symmetric, asymptotically flat weak-field metrics in an isotropic radial
coordinate, by deterministic polynomial extrapolation of the finite-radius
effective estimators to zero compactness. It enforces an explicit
isotropic-coordinate contract (areal-coordinate charts are rejected, not silently
misused), reports the radial-window / fit-order / resolution sensitivities
*individually* (each an available-or-not `ParameterSensitivity`, never a false
zero) plus a well-conditioned/marginal/ill-conditioned classification, alongside
the conservative blended *estimated* numerical uncertainty (not a bound) —
hardened per `LAYER_2_PPN.md` §11 — and is validated against exact GR and
non-GR synthetic metrics (injected `gamma`, `beta` recovered to machine
precision), controlled higher-order contamination (finite-radius bias,
extrapolation recovery), and exact isotropic Schwarzschild (`gamma = beta = 1`)
— `ppn` tests + `ppn_extraction` experiment + `ppn` benches. Only `gamma` and
`beta` are implemented; the exclusions are listed in the design note.

Its third increment — the **Einstein–Hilbert action and its numerical
variation** — is also **delivered** (design & conventions:
[`LAYER_2_ACTION_VARIATION.md`](LAYER_2_ACTION_VARIATION.md)). The `action`
module numerically varies `S = integral (R - 2 Lambda) sqrt(-g) d^4x` for a
static, axisymmetric background against a compact test perturbation — a central
difference in the amplitude of a Simpson-quadratured action whose integrand uses
a new metric-only nested-difference Ricci scalar ([`ricci_scalar_from_metric`],
a Layer 1 generalization) — and compares it to the analytic-integrand prediction
`-integral sqrt(-g) E^{ab} h_{ab}` from the Einstein tensor. The static +
axisymmetric symmetry reduces the 4D variation to a 2D `(r, theta)` integral;
the compact bump makes the Gibbons–Hawking boundary term vanish. Validated by
metric-only curvature recovering `4 Lambda` / `0`, vacuum stationarity for
Schwarzschild and `Lambda`-matched de Sitter (`G_{mu nu} + Lambda g_{mu nu} = 0`,
residual ~`O(dx^4)`), a mismatched-`Lambda` nonzero cross-check against the
Einstein tensor, and grid convergence (`action` tests + `action_variation`
experiment + `action` benches). A numerical approximation, never an exact
variation; only vacuum stationarity is validated (no matter sources).

Its fourth increment — **3+1 (ADM) kinematics** — is also **delivered** (design
& conventions: [`LAYER_2_ADM.md`](LAYER_2_ADM.md)). The `adm` module decomposes a
4-metric on the foliation by constant-time slices into the lapse `N`, shift
`N^i`, spatial metric `gamma_ij`, and extrinsic curvature `K_ij`, and evaluates
the Gauss–Codazzi **Hamiltonian** (`R^(3) + K^2 - K_ij K^{ij} - 2 Lambda`) and
**momentum** (`D_j(K^{ij} - gamma^{ij} K)`) constraints, which vanish for exact
solutions. It reuses [`ricci_scalar_from_metric`] and `numerical_christoffel` at
`D = 3` for the spatial curvature and connection, and is validated on four exact
foliations — Schwarzschild (scalar-flat slice), static de Sitter
(`R^(3) = 2 Lambda`), FLRW (`K = -3H`), and the horizon-penetrating
[`PainleveGullstrand`] slicing (non-zero shift, spatially varying `K`, a
non-trivial momentum constraint) — plus the algebraic reconstruction identity
(`adm` tests + `adm_kinematics` experiment + `adm` benches). This is the natural
bridge to Layer 3; it evolves nothing in time.

With these four slices the near-term Layer 2 sequence is complete. The full
symbolic-algebra action machinery is deliberately deferred until a slice needs
it and it can be made deterministic.

**Layer 3 (Numerical Relativity) now opens**, on its design note
[`LAYER_3_ADM_EVOLUTION.md`](LAYER_3_ADM_EVOLUTION.md). Its first increment —
the **ADM constraint and evolution core** (Layer 3.1) — is **delivered**. Unlike
Layer 2's `adm` module, which *extracts* the lapse/shift/spatial-metric/
extrinsic-curvature *backward* from an already-known 4-metric, the `adm_evolution`
module evaluates the Gauss–Codazzi Hamiltonian and momentum constraints and the
right-hand sides of `partial_t gamma_ij` and `partial_t K_ij` *forward*, from
independently supplied 3+1 data (`AdmSources` for the matter projections, plus
lapse/shift/extrinsic-curvature fields) — the direction a future time integrator
actually needs. It reuses the dimension-generic `ricci_scalar_from_metric` and a
new sibling `ricci_tensor_from_metric` (added by refactoring the former into a
thin wrapper, zero duplication) at `D = 3`. Notably, deriving the extrinsic-
curvature evolution equation surfaced a **real sign error**: the naively-copied
textbook matter-term sign (`+ 8 pi alpha (...)`) is wrong for this repository's
established `K_ij` convention; the correct sign (`-`) was confirmed by directly
comparing the closed-form right-hand side against a time finite-difference of
Layer 2's already-validated FLRW extraction (`5.3e-9` agreement for the correct
sign vs. `1.83` — a full sign flip — for the naive one). Validated on Minkowski,
a static Schwarzschild slice (whose lapse-Hessian/Ricci-tensor combination must
cancel exactly for a time-independent solution), flat FLRW (reduces to the first
Friedmann equation), and deliberate constraint violations with closed-form,
monotonically scaling residuals (`adm_evolution` tests + `adm_constraint_sweep`
experiment + `adm_evolution` benches).

Its second increment — **homogeneous ADM time evolution** (Layer 3.2) — is also
**delivered** (design & conventions:
[`LAYER_3_HOMOGENEOUS_EVOLUTION.md`](LAYER_3_HOMOGENEOUS_EVOLUTION.md)). The
`adm_homogeneous` module integrates the Layer 3.1 right-hand sides *forward in
time* through the platform's existing `scirust_sim::simulate` RK4 (no new
integrator), for spatially homogeneous data with a barotropic perfect fluid
`p = w rho`, and monitors the Hamiltonian constraint as a **per-sample
diagnostic that is never enforced, projected, or damped** — the posture of a
real free-evolution code. The restriction to the homogeneous sector is
deliberate and scientific, not a shortcut: ADM is only *weakly hyperbolic*, so
free evolution of generic inhomogeneous data on a grid is unstable (which is
precisely why BSSN exists), whereas every field here is position-independent, so
all spatial derivatives vanish identically, the pathology cannot arise, and
exact closed-form solutions exist to check against. Validated against the three
exact Friedmann solutions — de Sitter (`a = exp(H t)`), dust (`a ~ t^(2/3)`),
and radiation (`a ~ t^(1/2)`), each read from the repository's
already-validated [`ExponentialScaleFactor`] / [`PowerLawScaleFactor`] rather
than retyped — recovering fourth-order convergence (the error falls by
`~2^4 = 16` per step halving, so RK4's order survives the finite-difference
right-hand sides), constraint preservation at the integration-error floor (down
to `1.1e-15` for de Sitter), and detection of initial data seeded deliberately
off the constraint surface (`adm_homogeneous` tests +
`adm_homogeneous_evolution` experiment + `adm_homogeneous` benches). This says
**nothing** about ADM's stability for inhomogeneous data, and it is not a
cosmological model.

Its third increment — the **BSSN formulation core** (Layer 3.3) — is also
**delivered** (design & conventions: [`LAYER_3_BSSN.md`](LAYER_3_BSSN.md)). The
`bssn` module implements the conformal-traceless variable transformation
(`phi` canonical, `chi` derived; `gammatilde_ij`, `K`, `Atilde_ij`,
`Gammatilde^i`) and its inverse, the three algebraic constraints reported
separately, explicit never-silent projections with pre/post residuals and
idempotence, the conformal Ricci decomposition `R_ij = Rtilde_ij + Rphi_ij`
(with `Rtilde` reusing the Layer 1 Ricci engine — only `Rphi` is new — and the
sum cross-checked against the independently computed physical Ricci to the
nested finite-difference floor, `~7e-7` on a component scale of `1.7e-1`), and
the local evolution right-hand sides with every additive term exposed.

Deriving it surfaced a second **real reconciliation issue**, found numerically
before any code was written: the standard BSSN `d_t K` is the
**constraint-substituted** form (it uses the Hamiltonian constraint to eliminate
`R`), so ADM/BSSN right-hand-side equivalence is exact only *on the constraint
surface*. Off it the two differ by **exactly** `alpha * H` — measured difference
`-0.8942257331` versus `alpha * H = -0.8942257331`, agreeing to `0.000e0`. The
API reports both the raw and the constraint-corrected difference (`3.87e-1` and
`5.55e-17` in the experiment), so the residual is visibly accounted for rather
than tuned away. `d_t phi`, `d_t gammatilde_ij`, and `d_t Atilde_ij` are
equivalent to ADM unconditionally.

**Implementing BSSN does not, by itself, demonstrate numerical stability** —
strong hyperbolicity is a property of the full system including gauge and the
principal part on a discretized domain, none of which exists here. A discretized
spatial grid, together with the live gauge conditions a practical BSSN evolution
requires, is the next Layer 3 frontier.

Its fourth increment — **BSSN on a periodic one-dimensional grid** (Layer 3.4) —
is **delivered**, and its headline result is **negative**
(design: [`LAYER_3_BSSN_PERIODIC_1D.md`](LAYER_3_BSSN_PERIODIC_1D.md)). The
`grid` and `bssn_grid` modules supply a half-open periodic grid, centred
finite-difference operators, a grid-backed derivative provider, and a
method-of-lines adapter that integrates through the existing
`scirust_sim::simulate` RK4. This is the first increment in which BSSN is a
partial differential equation rather than an algebraic identity. It is a **1D3V
reduction** — one spatially varying coordinate, full 3x3 tensors — not a
three-dimensional numerical-relativity solver.

The pipeline is second-order accurate and the measurements are clean: Minkowski
is *exactly* stationary (`0.000000e0` state change at every resolution); the
conformal Ricci reconstruction converges at observed order **1.97 / 1.99 /
2.00** — turning Layer 3.3's unexplained `~1e-6` "finite-difference floor" into
truncation error with an error model; short-time wave propagation converges at
**1.96 / 1.99 / 2.00**; and RK4 retains **4.00 / 4.00 / 3.99** against a fixed
spatial operator.

Getting there required a correction worth recording. The first revision reused
Layer 3.3's `conformal_ricci` unchanged — which computes the *generic* Ricci
tensor of `gammatilde_ij` and never reads `Gammatilde^i` — and inherited its
`d_t Gammatilde^i = 0`. That is BSSN *variables* carrying ADM's *principal
part*, and it behaved exactly like weakly hyperbolic ADM: `N >= 64` blew up
before `t = 1` at every Courant factor, worse under refinement, and dissipation
could not cure it. The failure was first attributed to stencil width; the
connection constraint's **resolution-independent** growth disproved that, and
supplying `d_t Gammatilde^i` alone did not help either — because the Ricci
tensor never read it.

The fix is to write `Rtilde_ij` in genuine BSSN form, where
`gammatilde_{k(i} d_{j)} Gammatilde^k` carries the evolved connection and
removes the mixed second derivatives from the principal part, *and* to supply
the deferred `d_t Gammatilde^i` equation. Both are needed. Afterwards every
resolution from `N = 32` to `N = 256` reaches `t = 4` with the error decreasing
under refinement, the Courant boundary is a genuine resolution-**independent**
`C <= 1`, and dissipation becomes a no-op. **Stability here is measured, not
proven** — strong hyperbolicity is an analytic property of the continuum system
and nothing here establishes it.

The increment closes with the first **live gauge**: 1+log slicing,
`d_t alpha = -2 alpha K`, with the lapse an evolved field and the three terms
that were structurally zero under a constant lapse (`-D^i D_i alpha` in
`d_t K`, the trace-free lapse Hessian in `d_t Atilde_ij`, and
`-2 Atilde^{ij} d_j alpha` in `d_t Gammatilde^i`) all supplied. It has a sharp
closed-form oracle: linearised about flat space the lapse obeys a wave equation
with characteristic speed exactly `sqrt(2)` — superluminal, and legitimately so,
because the lapse is gauge and carries no signal. Measured against
`alpha = 1 + A cos(sqrt(2) k t) sin(kx)`, the error converges at order **2.00 /
2.00** with the amplitude ratio going `0.995309 -> 0.998828 -> 0.999708`.
Prescribed slicing stays the **default**, so a live gauge is never enabled
implicitly, and Minkowski is bit-for-bit undisturbed by turning it on. **The
Gamma-driver shift is not implemented** — `beta^i = 0` throughout — and is the
next increment.

That increment followed immediately. The shift `beta^i` and the driver auxiliary
`B^i` are stored fields too, so every shift term in every equation is live, and
the **Gamma-driver** (`d_t beta^i = (3/4) B^i`,
`d_t B^i = d_t Gammatilde^i - eta B^i`) completes the moving-puncture gauge. Two
closed forms validate it. For a *spatially constant* shift every `d(beta)` term
vanishes, so the right-hand-side difference must equal `v d_x(field)` exactly for
all seventeen evolved components — measured relative residuals `8.80e-16`,
`1.47e-16`, `4.91e-19`, i.e. **rounding**. And on flat space the driver decouples
into `B = B0 exp(-eta t)`, `beta = beta0 + (3/4)(B0/eta)(1 - exp(-eta t))`,
matched to relative `4e-12` … `5e-10`. Prescribed slicing and a prescribed shift
remain the **defaults**, and Minkowski under the full moving-puncture gauge is
bit-for-bit stationary. The gauge is nonetheless only **weak-field tested** —
never on a puncture, a horizon, or a strong field, which is what it exists for.

## What this platform will not do

- It will not assert modified-gravity or nonlocal-gravity physics as
  established. Such models remain in clearly-labelled experimental layers.
- It will not present a numerical approximation as an exact result, or a
  phenomenological hook as a physical law.
- It will not hide a limitation. Disclosed error (finite-difference truncation,
  bounded memory) is part of the result, reported alongside it.
