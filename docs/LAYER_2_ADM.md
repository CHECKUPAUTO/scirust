# Layer 2 — 3+1 (ADM) kinematics: design note

This note fixes the fourth and final scoped Layer 2 (Covariant Gravity
Workbench) increment before any code lands, following the platform rule that
each slice opens with its oracles and category labels. It refines the follow-on
named in `docs/LAYER_2_COVARIANT_GRAVITY.md` §4:

> **3+1 (ADM) kinematics.** Lapse, shift, spatial metric, and extrinsic
> curvature from a foliation, with the Hamiltonian and momentum constraints as
> oracles (they vanish for exact solutions). This is also the natural bridge to
> Layer 3 (numerical relativity).

It reuses the Layer 1 geometry core — in particular the dimension-generic
`ricci_scalar_from_metric` (added for the action variation) at `D = 3` for the
spatial curvature, and `numerical_christoffel` at `D = 3` for the spatial
connection — and duplicates nothing.

## 1. Scope and category

**In scope.** The 3+1 decomposition of a 4-metric on the foliation by constant
time-coordinate slices, and the Gauss–Codazzi constraints as oracles:

- the **lapse** `N`, **shift** `N^i`, **spatial metric** `gamma_ij`, and
  **extrinsic curvature** `K_ij` of the slicing;
- the derived spatial Ricci scalar `R^(3)`, mean curvature `K = gamma^{ij} K_ij`,
  and `K_ij K^{ij}`;
- the **Hamiltonian constraint** `H = R^(3) + K^2 - K_ij K^{ij} - 2 Lambda` and
  the **momentum constraint** `M^i = D_j (K^{ij} - gamma^{ij} K)`, which both
  vanish for a vacuum-with-`Lambda` solution.

**Category labels:**

- **Established general relativity** — the ADM decomposition and the
  Gauss–Codazzi constraints are textbook GR. No new physics.
- **Numerical approximation** — `R^(3)`, `K_ij`, and the constraint divergences
  are finite-difference quantities (a metric-only nested difference for `R^(3)`,
  a time difference for `partial_0 gamma`, and spatial differences for the
  covariant derivatives). Their truncation error is measured and reported.

**Explicitly out of scope** (documented, not attempted): time evolution (the
ADM/BSSN *equations of motion* — this is Layer 3); matter sources beyond the
cosmological constant (the constraints are validated in vacuum-with-`Lambda`);
constraint *damping* and gauge/slicing conditions; and non-constant-`t`
foliations (the slicing is by the chart's time coordinate).

## 2. Physics and conventions

Signature `(-,+,+,+)`, geometric units `G = c = 1`, spatial indices `i, j in
{1,2,3}`. On the foliation by constant-`x^0` slices, with `g` the 4-metric:

```text
gamma_ij = g_ij                                  (spatial 3-metric)
N_i      = g_{0i} ,   N^i = gamma^{ij} N_j        (shift)
N        = sqrt( N_i N^i - g_{00} )               (lapse)
K_ij     = -1/(2N) ( partial_0 gamma_ij - D_i N_j - D_j N_i )   (extrinsic curvature)
```

where `D_i` is the covariant derivative of `gamma` (`D_i N_j = partial_i N_j -
Gamma^{(3)k}_{ij} N_k`) and `gamma^{ij}` is the inverse of the 3-metric. The
decomposition is exactly invertible: `g_{00} = -N^2 + N_i N^i`, `g_{0i} = N_i`,
`g_ij = gamma_ij` (oracle O0). The **Gauss–Codazzi constraints** for a
vacuum-with-`Lambda` solution are

```text
R^(3) + K^2 - K_ij K^{ij} = 2 Lambda ,     D_j ( K^{ij} - gamma^{ij} K ) = 0 .
```

## 3. Numerical method

- **Spatial curvature `R^(3)`** reuses `ricci_scalar_from_metric` at `D = 3` on
  the spatial slice (a `Metric<3>` presenting `gamma_ij` at fixed `x^0`) — the
  same metric-only nested finite difference built for the action variation, now
  in three dimensions.
- **Spatial connection `Gamma^{(3)}`** reuses `numerical_christoffel` at
  `D = 3`, used both for `D_i N_j` in `K_ij` and for the covariant divergence in
  the momentum constraint.
- **`partial_0 gamma_ij`** is a central time difference of the spatial block
  (nonzero only for a time-dependent slicing, e.g. FLRW).
- **The momentum constraint** `M^i = partial_j P^{ij} + Gamma^{(3)i}_{jk} P^{kj}
  + Gamma^{(3)j}_{jk} P^{ik}` with `P^{ij} = K^{ij} - gamma^{ij} K` uses a
  central spatial difference of `P^{ij}` (each neighbour recomputing `K`), so it
  is one finite-difference layer above `K_ij` and correspondingly the noisiest
  quantity — reported honestly.

## 4. Oracle backgrounds and checks

Four foliations, spanning the regimes (static/time-symmetric, spatially curved,
time-dependent, and non-zero shift), each an exact vacuum-with-`Lambda` solution:

1. **Schwarzschild** (standard coordinates) — a static, time-symmetric slicing:
   `N = sqrt(1 - 2M/r)`, `N^i = 0`, `K_ij = 0`, and `R^(3) = 0` (the slice is
   *scalar-flat* although it is curved — a non-trivial numerical cancellation).
   Hamiltonian and momentum constraints vanish.
2. **Static de Sitter** — a static slicing with `K_ij = 0` but curved space:
   `R^(3) = 2 Lambda` exactly (an analytic spatial-curvature oracle), so the
   Hamiltonian constraint `R^(3) - 2 Lambda = 0`.
3. **Exponential FLRW** (de Sitter in cosmological slicing) — time-dependent,
   spatially flat: `N = 1`, `N^i = 0`, `R^(3) = 0`, `K_ij = -H gamma_ij`, so
   `K = -3H` and `K_ij K^{ij} = 3 H^2`. The Hamiltonian constraint gives
   `K^2 - K_ij K^{ij} = 6 H^2 = 2 Lambda` (with `Lambda = 3 H^2`) — a non-zero
   extrinsic-curvature oracle.
4. **Painlevé–Gullstrand Schwarzschild** — a horizon-penetrating slicing with a
   **non-zero radial shift** `N^r = sqrt(2M/r)`, unit lapse, and a **flat**
   spatial slice (`R^(3) = 0`), whose extrinsic curvature is non-zero and
   spatially varying. This exercises the shift terms in `K_ij` and makes the
   momentum constraint non-trivial; both constraints vanish (vacuum).

Plus **O0 — algebraic reconstruction:** the extracted `(N, N^i, gamma_ij)`
rebuild the 4-metric to rounding, on every background (validates the split
itself, independent of any curvature computation).

## 5. Diagnostics and honesty

The decomposition result carries `N`, `N^i`, `gamma_ij`, `K_ij`, `R^(3)`, `K`,
and `K_ij K^{ij}`; the constraint result carries the Hamiltonian and momentum
residuals. All finite-difference quantities are numerical approximations to a
disclosed tolerance (the momentum constraint, a spatial difference of a
time-difference, is the coarsest). Nothing here evolves the data in time; the
ADM *evolution* equations are Layer 3.

## 6. Deliverable shape and placement

- **`scirust-relativity` `PainleveGullstrand`** — a horizon-penetrating
  Schwarzschild foliation (`Metric<4>`), the non-zero-shift oracle background.
- **`scirust-relativity` `adm` module** — `adm_decomposition` (lapse, shift,
  spatial metric, extrinsic curvature, `R^(3)`, `K`, `K_ij K^{ij}`) and
  `adm_constraints` (the Hamiltonian and momentum residuals), a validated
  request, a typed `AdmError`, and a no-non-finite guarantee. Tests for O0–O4
  and an `adm_kinematics` experiment reporting the constraint residuals and the
  extrinsic-curvature invariants, under the established-GR /
  numerical-approximation labels.

This slice completes the near-term Layer 2 follow-ons; it is the natural bridge
to **Layer 3 (Numerical Relativity)**, whose evolution equations advance exactly
this ADM data in time.
