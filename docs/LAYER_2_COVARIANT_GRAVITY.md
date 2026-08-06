# Layer 2 — Covariant Gravity Workbench: design note

This note opens Layer 2. Per the platform rule that *each layer opens with a
design note fixing its oracles and category labels before any code lands*, it
defines the scope, the scientific-category discipline, the concrete first
increment with its exact oracles, and the follow-on slices. No Layer 2 code
exists yet; this is the contract the first increments must meet.

It supersedes nothing in Layer 1 and adds no dependency to it: Layer 2 is built
**on top of** the `scirust-relativity` geometry core (metrics, connections,
`CurvatureTensors`, the finite-difference machinery), reuse-first, never
duplicating it.

## 1. Scope

Layer 2 is the **Covariant Gravity Workbench**: tools that work with the field
content of gravity itself — the linearized field equations, weak-field and
post-Newtonian (PPN) observables, the gravitational action and the field
equations obtained by varying it, and 3+1 (ADM) kinematics. Where Layer 1 asks
"given a metric, what is its geometry?", Layer 2 asks "what equations does the
metric satisfy, and what are its weak-field and variational structures?".

The full symbolic-algebra vision (a computer-algebra action functional
differentiated in closed form) is explicitly **out of scope for the opening
increments**: it is hard to make deterministic and auditable in pure Rust, and
it is not needed for the physics below. Layer 2 begins with the numerically
tractable, exactly-checkable slices and only adds symbolic machinery later, if
and when a slice genuinely needs it and it can be made deterministic.

## 2. Category discipline (unchanged, restated)

Every Layer 2 result carries one of the platform's fixed category labels:

- **Established general relativity** — standard textbook GR (the linearized
  Einstein tensor, the Newtonian limit, the PPN values of GR). The opening
  increments are all in this category.
- **Numerical approximation** — a quantity produced by a disclosed numerical
  method (finite-difference derivatives of the perturbation, a numerical
  functional derivative of an action). Its truncation error is measured and
  reported, exactly as the curvature engine's is.
- **Speculative / phenomenological** — modified-gravity actions, extra fields.
  These, if ever added, live behind explicit opt-in and are never presented as
  established.

Layer 2 will not present a numerical functional derivative as an exact
variation, will not assert a modified-gravity field equation as established, and
will not blur the weak-field approximation into an exact statement.

## 3. First increment — linearized gravity

**Goal.** Given a metric perturbation `h_{mu nu}(x) = g_{mu nu}(x) - eta_{mu nu}`
about Minkowski (`|h| << 1`), compute the linearized curvature and field tensors
and validate them against exact, convention-free oracles.

**Quantities (all established GR, computed by central finite differences of `h`,
reusing the geometry core's difference machinery):**

- linearized Riemann
  `R^{(1)}_{mu nu rho sigma} = (1/2)(d_nu d_rho h_{mu sigma} + d_mu d_sigma h_{nu rho} - d_nu d_sigma h_{mu rho} - d_mu d_rho h_{nu sigma})`;
- linearized Ricci `R^{(1)}_{mu nu}` and scalar `R^{(1)}`;
- the trace-reversed perturbation `hbar_{mu nu} = h_{mu nu} - (1/2) eta_{mu nu} h`,
  `h = eta^{mu nu} h_{mu nu}`;
- the linearized Einstein tensor `G^{(1)}_{mu nu}`;
- the Lorenz-gauge wave operator: in Lorenz gauge `d^mu hbar_{mu nu} = 0`,
  `G^{(1)}_{mu nu} = -(1/2) box hbar_{mu nu}`, so the vacuum equation is
  `box hbar_{mu nu} = 0`.

**Oracles (what the increment's tests and experiment must show):**

1. **Weak-field Schwarzschild is linearized-vacuum.** The far-field Schwarzschild
   perturbation in isotropic-like form, `h_{00} = 2M/r`, `h_{ij} = (2M/r) delta_{ij}`
   (i.e. `Phi = -M/r`, `h_{00} = -2 Phi`, `h_{ij} = -2 Phi delta_{ij}`), satisfies
   `G^{(1)}_{mu nu} = 0` away from the source, to finite-difference tolerance.
   This is the linearized statement that Schwarzschild is a vacuum solution.
2. **Newtonian limit reproduces the Poisson equation.** For a static perturbation
   built from a potential `Phi` with `h_{00} = -2 Phi`, `h_{ij} = -2 Phi delta_{ij}`,
   the linearized `G_{00}` equals `2 nabla^2 Phi`, so `G_{00} = 8 pi rho` reduces
   to `nabla^2 Phi = 4 pi rho`. Checked exactly with a polynomial `Phi` of known
   Laplacian (e.g. `Phi = a(x^2 + y^2 + z^2)` gives `nabla^2 Phi = 6a`) — an
   analytic oracle with no source ambiguity.
3. **Gauge invariance of the linearized Riemann.** Under an infinitesimal gauge
   transformation `h_{mu nu} -> h_{mu nu} + d_mu xi_nu + d_nu xi_mu`, the
   linearized Riemann tensor is invariant. Verified numerically:
   `R^{(1)}(h)` equals `R^{(1)}(h + gauge)` to finite-difference tolerance, for a
   nontrivial `xi(x)`. This is convention-free and does not rely on any oracle
   metric — it tests the operator's tensorial correctness directly.
4. **Consistency with Layer 1.** For a genuinely weak background, the linearized
   Ricci scalar agrees with the geometry core's full nonlinear `CurvatureTensors`
   Ricci scalar to first order (the difference is `O(h^2)` and shrinks
   quadratically as the perturbation amplitude is scaled down) — a cross-check
   that ties Layer 2 back to Layer 1.

**Deliverable shape.** A `linearized` module in `scirust-relativity` (or a new
`scirust-gravity` crate if the surface grows) exposing a
`LinearizedField`-style struct built from a perturbation sampler, with typed
errors and a no-non-finite-output guarantee, mirroring `CurvatureTensors`. Tests
for oracles 1–4 and a `linearized_gravity` experiment reporting the residuals
and the `O(h^2)` scaling of oracle 4.

## 4. Follow-on slices (each its own later increment, each with oracles)

- **PPN parameters `gamma`, `beta`.** Extract the two Eddington–Robertson
  parameters from a static, spherically symmetric weak metric expansion. Oracle:
  Schwarzschild (in isotropic coordinates) gives `gamma = beta = 1` exactly;
  Reissner–Nordström and the existing backgrounds provide further checks.
- **Einstein–Hilbert action and its variation.** The action density
  `R sqrt(-g)` and the field equations from a *numerical* functional derivative
  `delta S / delta g^{mu nu}` (finite-difference on the discretized action).
  Oracle: the variation reproduces `G_{mu nu} = 0` in vacuum backgrounds
  (Schwarzschild, de Sitter with the cosmological-constant term) to the disclosed
  numerical tolerance. Labeled a *numerical approximation*, never an exact
  variation.
- **3+1 (ADM) kinematics.** Lapse, shift, spatial metric, and extrinsic
  curvature from a foliation, with the Hamiltonian and momentum constraints as
  oracles (they vanish for exact solutions). This is also the natural bridge to
  Layer 3 (numerical relativity).

## 5. What Layer 2 will not do

- It will not assert modified-gravity or extra-field actions as established
  physics.
- It will not present the numerical functional derivative of the action as a
  closed-form variation; its truncation error is part of the result.
- It will not extend the weak-field or PPN results beyond their regime of
  validity, or describe a first-order-in-`h` quantity as exact.
- It will not duplicate Layer 1: the linearized operators reuse the geometry
  core's difference machinery and are cross-checked against its nonlinear
  curvature.

## 6. First step

Implement the **linearized gravity** increment of §3 — linearized
Riemann/Ricci/Einstein plus the Lorenz-gauge wave operator, validated by the
four oracles — as the first Layer 2 pull request. PPN extraction (§4) follows
once linearized gravity is in and green.
