# Layer 2 — Einstein–Hilbert action and its numerical variation: design note

This note fixes the third Layer 2 (Covariant Gravity Workbench) increment before
any code lands, following the platform rule that each slice opens with its
oracles and category labels. It refines the follow-on named in
`docs/LAYER_2_COVARIANT_GRAVITY.md` §4:

> **Einstein–Hilbert action and its variation.** The action density `R sqrt(-g)`
> and the field equations from a *numerical* functional derivative
> `delta S / delta g^{mu nu}` ... Oracle: the variation reproduces
> `G_{mu nu} = 0` in vacuum backgrounds (Schwarzschild, de Sitter with the
> cosmological-constant term) to the disclosed numerical tolerance. Labeled a
> *numerical approximation*, never an exact variation.

It reuses the Layer 1 geometry core (metrics, `numerical_christoffel`,
`CurvatureTensors`) and adds one small Layer 1 primitive that the variation
needs; it duplicates nothing.

## 1. Scope and category

**In scope.** The gravitational action density and the vacuum field equations
obtained by *numerically* varying it, for **static, axisymmetric** backgrounds:

- the Einstein–Hilbert action with a cosmological term,
  `S[g] = integral_Omega (R - 2 Lambda) sqrt(-g) d^4x` (geometric units, coupling
  `1/2kappa` normalized to `1`; only the field equations, not absolute action
  values, are physical);
- its directional functional derivative `D S[g] . h = d/d eps S[g + eps h]` at
  `eps = 0`, evaluated numerically by a central difference in `eps`, against a
  test perturbation `h`;
- the identity `D S[g] . h = -integral_Omega sqrt(-g) E^{ab} h_{ab} d^4x` with the
  Euler–Lagrange tensor `E_{mu nu} = G_{mu nu} + Lambda g_{mu nu}`, so a solution
  of `G_{mu nu} + Lambda g_{mu nu} = 0` makes the action **stationary**.

**Category labels** (kept strictly separate, per the platform rule):

- **Established general relativity** — that the Einstein–Hilbert action's
  variation yields `G_{mu nu} + Lambda g_{mu nu} = 0` is textbook GR. No new
  physics; no modified-gravity action is asserted.
- **Numerical approximation** — every number here is produced by disclosed
  finite-difference and quadrature methods (a metric-only nested-difference Ricci
  scalar, a grid quadrature of the action, and a central difference in `eps`).
  Its truncation error is measured and reported, exactly as the curvature
  engine's is. **It is never presented as a closed-form (exact) variation.**

**Explicitly out of scope** (documented, not attempted): closed-form / symbolic
variation; non-vacuum stress-energy sources; dynamical or non-stationary or
non-axisymmetric backgrounds (the reduction in §3.3 relies on two Killing
vectors); modified-gravity or extra-field actions; boundary (Gibbons–Hawking–
York) terms as a deliverable (they are *arranged to vanish*, see §3.2); and any
claim about the action's absolute value.

## 2. Physics and conventions

Signature `(-,+,+,+)`, geometric units `G = c = 1`. Varying `S` with respect to
the inverse metric,

```text
delta S = integral sqrt(-g) (R_{mu nu} - 1/2 R g_{mu nu} + Lambda g_{mu nu}) delta g^{mu nu} d^4x
        + integral partial_mu( sqrt(-g) w^mu ) d^4x,
```

where the second (total-derivative) term is the boundary term with
`w^mu = g^{ab} delta Gamma^mu_{ab} - g^{mu b} delta Gamma^a_{ab}`. Writing the
Euler–Lagrange tensor `E_{mu nu} = G_{mu nu} + Lambda g_{mu nu}` and perturbing
the *covariant* metric `g_{ab} -> g_{ab} + eps h_{ab}` (so
`delta g^{mu nu} = -eps g^{mu a} g^{nu b} h_{ab}`),

```text
D S[g] . h = -integral_Omega sqrt(-g) E^{ab} h_{ab} d^4x     (boundary term vanishing),
```

with `E^{ab} = g^{a mu} g^{b nu} E_{mu nu}`. For a vacuum-with-Lambda solution
`E_{mu nu} = 0`, so `D S[g] . h = 0` for every admissible `h` — the action is
stationary. This is the statement the increment checks numerically.

## 3. Numerical method

### 3.1 Metric-only nonlinear Ricci scalar (the Layer 1 primitive)

The action integrand needs `R` of a *perturbed field* that has no analytic
connection, so we add
`ricci_scalar_from_metric(metric, coords, connection_step, metric_step)`: it
builds the Christoffel symbols from central differences of the metric components
(`numerical_christoffel`, step `metric_step`), differences *those* again (step
`connection_step`) for `partial Gamma`, and contracts the geometry core's
existing Riemann → Ricci → scalar assembly. It is a **nested** finite difference
(one layer more than `CurvatureTensors` on an analytic-connection background),
reusing the curvature engine's private assembly rather than re-deriving it, with
the same typed errors and no-non-finite guarantee. Validated by recovering
`R = 4 Lambda` (de Sitter) and `R = 0` (Schwarzschild) — oracle O1.

### 3.2 Compact test perturbation (why the boundary term vanishes)

The Einstein–Hilbert action is second-order in the metric, so its variation
carries the boundary term above. We make it vanish **exactly** by choosing a test
perturbation `h_{ab}(x) = phi(x) B_{ab}` whose profile `phi` is a
compactly-supported polynomial bump, a product of `(1 - u^2)^4` factors
(`u` the normalized offset), which vanishes together with its first three
derivatives at the support boundary. Then `h` and `partial h` — hence
`delta Gamma` and `w^mu` — are zero on `partial Omega`, so the surface flux is
zero. (A Gaussian leaks a boundary flux that grows as the grid resolves it; a
sharp mollifier `exp(-1/(1-u^2))` is boundary-clean but its derivatives spike
under the nested difference. The polynomial bump is both boundary-clean and
smooth — this choice is the crux of a clean result.)

### 3.3 Static + axisymmetric reduction to two dimensions

For a static, axisymmetric background (`partial_t g = partial_phi g = 0`) we take
`phi = phi(r, theta)` compact in `(r, theta)` and **constant** in `(t, phi)`.
Then the integrand `sqrt(-g) delta R` is independent of `t` and `phi`, so:

- the `t` and `phi` integrals factor out as common constants (they cancel between
  the numeric variation and the prediction), and
- the `t` and `phi` boundary fluxes **telescope to zero** (equal on opposite
  faces of the box, opposite outward normals), while the `r` and `theta` fluxes
  vanish by compact support (§3.2).

The 4D variation therefore reduces to a **2D `(r, theta)` integral**, which lets
us use a fine grid and reach a tight, cleanly-converging tolerance at negligible
cost. This reduction is the reason the increment restricts to static,
axisymmetric backgrounds; it is a stated scope choice, not a hidden assumption.

### 3.4 Quadrature and the eps-derivative

The 2D integral uses composite **Simpson** quadrature on an odd grid. The
directional derivative is a central difference
`(S[g + eps h] - S[g - eps h]) / (2 eps)`, so the `O(eps^2)` truncation is
disclosed and the background action cancels to leading order. The prediction
`-integral sqrt(-g) E^{ab} h_{ab}` is evaluated on the same grid from the
geometry core's analytic-connection Einstein tensor (`CurvatureTensors`), so the
two sides are compared like-for-like.

## 4. Oracles (what the tests and experiment must show)

1. **O1 — metric-only curvature.** `ricci_scalar_from_metric` recovers
   `R = 4 Lambda` for de Sitter and `R = 0` for Schwarzschild, to the nested-
   difference tolerance (~`1e-6`). Validates the §3.1 primitive.
2. **O2 — vacuum stationarity.** For Schwarzschild (`Lambda = 0`) and de Sitter
   (`Lambda` matched to the background), the numeric variation `D S[g] . h -> 0`
   to the disclosed tolerance, for perturbations in several components and at
   several support centers. This is "the variation reproduces
   `G_{mu nu} + Lambda g_{mu nu} = 0` in vacuum."
3. **O3 — the variation is the Einstein tensor, not merely zero.** With the
   action's `Lambda` deliberately *mismatched* to the background (e.g. `Lambda = 0`
   on de Sitter), the numeric variation reproduces the **known nonzero**
   prediction `-integral sqrt(-g) E^{ab} h_{ab}` computed independently from
   `CurvatureTensors`, to a small relative tolerance. This shows the machinery
   computes the correct functional derivative, not just that solutions are
   stationary.
4. **O4 — convergence.** The O2 residual and the O3 relative error fall under
   grid refinement (empirically ~`O(dx^4)` with Simpson), giving quantitative
   convergence evidence rather than a single tolerance.

## 5. Diagnostics and honesty

The result carries the residual `|numeric - predicted|`, the grid resolution, the
`eps` and finite-difference steps, and (for O2/O3) the convergence trend across
resolutions. All are numerical approximations to a disclosed tolerance; none is a
bound, and none is an exact variation. The reduction of §3.3 is valid only for
static, axisymmetric backgrounds with test perturbations constant in `(t, phi)` —
stated as a precondition and enforced by the API (a background supplying the
metric and connection, and a validated request).

## 6. Deliverable shape and placement

- **Layer 1 (`scirust-relativity` curvature core):** `ricci_scalar_from_metric`
  — the metric-only nested-difference Ricci scalar (§3.1), reusing the existing
  curvature assembly.
- **Layer 2 (`scirust-relativity` `action` module):** the Einstein–Hilbert
  action variation — a validated request (component, support, `(r, theta)`
  domain, grid, `eps`, steps, action `Lambda`), a result carrying the numeric
  variation, the prediction, and the residual, and a typed `ActionError`. Tests
  for O1–O4 and an `action_variation` experiment reporting the residuals and the
  convergence, under the established-GR / numerical-approximation labels.
- **Not** its own crate yet; it stays beside the geometry core it reuses.

The follow-on after this slice is **3+1 (ADM) kinematics** — the bridge to
Layer 3 — as named in the Layer 2 design note.
