# Layer 3.2 — Homogeneous ADM time evolution: design note

This note fixes the second Layer 3 (Numerical Relativity) increment before any
code lands. Layer 3.1 (`docs/LAYER_3_ADM_EVOLUTION.md`) delivered the ADM
constraint and evolution **right-hand sides** at a point but deliberately
evolved nothing in time. This slice closes that gap in the one sector where
time evolution can be validated against **exact closed-form solutions**: the
spatially homogeneous (cosmological) sector.

## 1. Why the homogeneous sector, and not a 3D grid

The natural-sounding next step is "a spatial grid plus a time integrator."
That step is deliberately **split**, and this increment takes only the second
half, for a scientific reason that must be stated plainly:

**The ADM system is only weakly hyperbolic.** Free evolution of generic
inhomogeneous ADM data on a grid is well known to be numerically unstable:
constraint-violating modes grow without bound, which is precisely why the field
moved to BSSN and generalized-harmonic formulations. Shipping a 3D ADM grid
evolution now would produce numbers that *look* like a simulation but could not
be honestly validated against anything, and whose failure mode (growing
constraint violation) is a property of the formulation, not of the code.

The homogeneous sector is the complement of that problem:

- **All spatial derivatives vanish identically** (every field is
  position-independent), so the weak-hyperbolicity pathology — which lives
  entirely in the spatial-derivative principal part — cannot arise. The
  evolution is a well-posed system of ODEs.
- **Exact closed-form solutions exist** and are *already in this repository*:
  `ExponentialScaleFactor` (de Sitter, `a = exp(H t)`) and
  `PowerLawScaleFactor` (`a = (t / t_ref)^p`: dust `p = 2/3`, radiation
  `p = 1/2`), both already validated in Layer 1's FLRW curvature oracles.
- It genuinely exercises the Layer 3.1 right-hand sides through a real time
  integrator, and it makes **constraint preservation over time** — the central
  diagnostic of numerical relativity — measurable against an exact answer.

So this increment answers a question the grid cannot yet answer honestly: *do
the Layer 3.1 evolution equations, integrated forward, actually reproduce known
solutions of general relativity, and does the Hamiltonian constraint stay
satisfied as they do?*

## 2. Scope

**In scope**: free evolution of homogeneous, spatially flat ADM data with a
barotropic perfect-fluid source.

- the homogeneous ADM state `(gamma_ij, K_ij, rho)`;
- a barotropic equation of state `p = w rho` (`w = -1` vacuum energy /
  cosmological constant, `w = 0` dust, `w = 1/3` radiation);
- the matter continuity equation `rho' = -3 H (rho + p)`, which is the `nabla_mu
  T^{mu nu} = 0` projection along the normal;
- **free evolution with constraint monitoring**: the Hamiltonian constraint is
  *evaluated as a diagnostic at every recorded step*, never enforced,
  re-projected, or damped — exactly the posture of a real numerical-relativity
  free-evolution code.

**Explicitly out of scope** (documented, not attempted):

- **Any spatial grid, mesh, or inhomogeneity.** Every field here is
  position-independent by construction. This increment therefore says
  **nothing** about ADM's stability for inhomogeneous data.
- **BSSN** and every other reformulation. Still deferred, and for the reason in
  §1: it becomes necessary exactly when a grid is introduced, which is not here.
- **Constraint damping / projection**, adaptive or symplectic time stepping,
  gauge (slicing/shift) conditions — the lapse is unit and the shift zero
  throughout, which is the standard synchronous/comoving gauge for this sector.
- **Spatially curved (open/closed) FLRW.** Only the spatially flat case
  (`R^(3) = 0`) is covered, matching the repository's existing `Flrw` background.
- Cosmological *modelling* of any kind: no parameter fitting, no observational
  comparison, no dark-energy phenomenology. The three equations of state are
  numerical validation oracles, not a cosmology.

## 3. Conventions

Signature `(-,+,+,+)`, `G = c = 1`, inherited unchanged from Layer 3.1 and
Layer 2 — in particular `K_ij = -1/(2N)(partial_t gamma_ij - D_i N_j - D_j N_i)`,
so a **expanding** universe has **negative** `K` (`K = -3H`).

With unit lapse, zero shift, and position-independent fields, the Layer 3.1
right-hand sides reduce to

```text
partial_t gamma_ij = -2 K_ij
partial_t K_ij     = K K_ij - 2 K_ik K^k_j - 8 pi ( S_ij - 1/2 gamma_ij (S - rho) )
partial_t rho      = -3 H ( rho + p ) ,        H = -K/3 ,   K = gamma^{ij} K_ij
```

(the lapse Hessian, the Lie derivatives, and `R_ij` all vanish identically),
and the Hamiltonian constraint reduces to the **first Friedmann equation**

```text
R^(3) + K^2 - K_ij K^{ij} - 16 pi rho = 0    <=>    H^2 = 8 pi rho / 3 .
```

**The reduction is not hard-coded.** The implementation calls the Layer 3.1
`metric_evolution_rhs` / `curvature_evolution_rhs` / `hamiltonian_constraint`
with genuinely constant-in-space field adapters, so the same finite-difference
code path a future inhomogeneous evolution would take is exercised here — the
spatial derivatives simply evaluate to exact zeros (a central difference of a
constant is exactly `0`, and the Christoffel symbols of a constant metric are
exactly `0`, so `R_ij = 0` to the last bit, not merely to truncation). This
costs some redundant arithmetic and buys real validation coverage: the tested
path is the production path.

## 4. Integrator — reused, not written

Time stepping reuses **`scirust_sim::simulate`**, the platform's existing
deterministic fixed-step classical RK4, via its existing `System` trait
(`dim` + `derivatives(t, y, dydt)` over a flat `&[f64]` state). It already
provides step validation, a non-finite-state guard, and a `Trajectory`. **No
new integrator is introduced**, consistent with the platform rule against
duplicating an algorithm that already exists (`GeodesicSystem` reuses the same
engine).

The state is flattened deterministically as
`[gamma_00..gamma_22 (9), K_00..K_22 (9), rho (1)]` — 19 components, fixed
order, no hidden state.

## 5. Oracles

Each is an **exact** solution, and each is checked against the repository's
*already-validated* `ScaleFactor` implementations rather than a hand-retyped
formula:

1. **O1 — de Sitter / vacuum energy (`w = -1`).** `rho` is constant and
   `a(t) = exp(H t)`, checked against `ExponentialScaleFactor::value`.
2. **O2 — dust (`w = 0`).** `a(t) = (t/t_ref)^{2/3}`, checked against
   `PowerLawScaleFactor::value`.
3. **O3 — radiation (`w = 1/3`).** `a(t) = (t/t_ref)^{1/2}`, checked against
   `PowerLawScaleFactor::value`.
4. **O4 — fourth-order convergence.** Halving the step must reduce the
   scale-factor error by `~2^4 = 16`, confirming the RK4 order is genuinely
   achieved through the full evaluator stack (not accidentally degraded to a
   lower order by the finite-difference machinery inside the right-hand sides).
5. **O5 — constraint preservation.** The Hamiltonian residual, evaluated as a
   diagnostic and never enforced, must stay at the integration-error floor
   along the whole trajectory and must shrink as the step shrinks. This is the
   property that a real evolution code lives or dies by.
6. **O6 — deliberate constraint violation.** Initial data seeded *off* the
   constraint surface (`rho` inconsistent with `H^2 = 8 pi rho / 3`) must
   produce a nonzero Hamiltonian residual that does **not** silently vanish —
   the monitor must actually detect bad initial data rather than reporting
   success.

## 6. Diagnostics and honesty

Each recorded sample carries the time, the scale factor `a = (det gamma)^{1/6}`
(the geometric mean of the spatial scale, exact for the isotropic case here),
the Hubble parameter `H = -K/3`, the energy density, and the Hamiltonian
constraint residual. The residual is a **diagnostic of the free evolution**,
not a bound and not an enforced condition; it is reported at every sample so
that constraint drift is visible rather than hidden.

`a` is derived from `det gamma` rather than from a single component so that it
remains meaningful if the state ever leaves exact isotropy through rounding.

## 7. Deliverable shape and placement

New module `scirust-relativity/src/adm_homogeneous.rs` (`pub mod
adm_homogeneous;`), beside `adm` and `adm_evolution`. It exposes the
homogeneous state, a `BarotropicFluid` equation of state, a `System`
implementation, an `evolve_homogeneous` driver returning sampled history with
per-sample constraint residuals, and a typed `HomogeneousEvolutionError`. Tests
in `scirust-relativity/tests/adm_homogeneous.rs`, benchmarks in
`scirust-relativity/benches/adm_homogeneous.rs`, and the deterministic
`adm_homogeneous_evolution` experiment.

The follow-on after this slice — still **not** authorized by this note — is the
spatial grid, at which point weak hyperbolicity becomes the governing concern
and a BSSN (or generalized-harmonic) reformulation becomes the natural next
design note.
