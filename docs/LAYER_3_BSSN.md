# Layer 3.3 — BSSN formulation core: design note

This note fixes the third Layer 3 increment before any code lands: a **local,
deterministic, validated BSSN formulation core** — the variable transformation,
algebraic constraints, geometric reconstruction, and local evolution terms that
future grid-based work will need.

## 1. Motivation, and what implementing BSSN does *not* prove

The ADM evolution system delivered in Layer 3.1/3.2 is only **weakly
hyperbolic** under common gauge choices, so free evolution of generic
inhomogeneous data on a grid is numerically unstable. BSSN (Baumgarte–Shapiro–
Shibata–Nakamura) is the standard conformal–traceless reformulation adopted
precisely to repair that.

**Implementing BSSN does not, by itself, demonstrate numerical stability.**
Strong hyperbolicity is a property of the full evolution system *including its
gauge conditions and principal part on a discretized domain*; none of that
exists in this increment. What this increment establishes is that the BSSN
variables, constraints, reconstruction, and local right-hand sides are
implemented correctly and agree with the already-validated ADM system. Nothing
here is evidence of stability, and no such claim is made anywhere.

## 2. Scope

**In scope**: purely local (single-point) BSSN machinery —
the variable transformation and its inverse, the algebraic constraints,
explicit projections, the conformal Ricci decomposition, the local evolution
right-hand sides with decomposed diagnostics, and a homogeneous BSSN time
evolution reusing the existing integrator.

**Explicitly out of scope** (documented, not attempted): any multidimensional
**spatial grid**, finite-difference evolution on a domain, adaptive mesh
refinement, black-hole evolution, punctures, waveform extraction, distributed
execution, **live gauge evolution** (1+log lapse, Gamma-driver shift, gauge
damping), **constraint damping**, and any observational validation.

## 3. Conventions — inherited unchanged

All conventions are inherited verbatim from Layer 3.1
(`docs/LAYER_3_ADM_EVOLUTION.md`) and Layer 2 (`docs/LAYER_2_ADM.md`), and every
equation below was reconciled against them rather than copied from a reference:

- signature `(-,+,+,+)`, geometric units `G = c = 1`;
- future-pointing unit normal, with
  `K_ij = -1/(2N)(partial_t gamma_ij - D_i N_j - D_j N_i)` — so an **expanding**
  slice has **negative** `K` (`K = -3H` for FLRW);
- spatial curvature from the Layer 1 `ricci_tensor_from_metric` at `D = 3`;
- lapse `alpha` and shift `beta^i` are **prescribed** here (a gauge provider),
  not evolved;
- matter enters through the existing Layer 3.1 `AdmSources`
  (`rho`, `S_i`, `S_ij`, and the derived trace `S`) — **no second source
  convention is introduced**.

## 4. BSSN variables

The **canonical** conformal variable is

```text
phi = (1/12) ln gamma ,        gamma = det gamma_ij
```

so `e^{4 phi} = gamma^{1/3}`. `chi = e^{-4 phi} = gamma^{-1/3}` is exposed as a
**derived** accessor, never stored, so the two can never disagree. (`phi` is
chosen because this increment has no punctures, where `chi`'s better behaviour
at a coordinate singularity would matter; the choice is recorded here so a later
puncture increment can revisit it deliberately.)

```text
gammatilde_ij = e^{-4 phi} gamma_ij ,          det gammatilde = 1
K             = gamma^{ij} K_ij
Atilde_ij     = e^{-4 phi} ( K_ij - (1/3) gamma_ij K ) ,   gammatilde^{ij} Atilde_ij = 0
Gammatilde^i  = gammatilde^{jk} Gammatilde^i_jk
```

**Authoritative definition of `Gammatilde^i`:** the contracted conformal
Christoffel symbol above, evaluated from `gammatilde_ij` through the existing
`numerical_christoffel` at `D = 3`. The algebraically equivalent
`-partial_j gammatilde^{ij}` (valid under `det gammatilde = 1`) is **not** the
definition used; the difference between a *stored* `Gammatilde^i` and the one
reconstructed from the conformal metric is exposed as the **connection
constraint**.

## 5. Conformal Ricci decomposition

```text
R_ij = Rtilde_ij + Rphi_ij
Rphi_ij = -2 Dt_i Dt_j phi - 2 gammatilde_ij gammatilde^{kl} Dt_k Dt_l phi
          + 4 (d_i phi)(d_j phi) - 4 gammatilde_ij gammatilde^{kl} (d_k phi)(d_l phi)
```

where `Dt` is the covariant derivative of `gammatilde`. **`Rtilde_ij` is not
re-implemented**: it is the Ricci tensor of `gammatilde_ij`, obtained from the
existing Layer 1 `ricci_tensor_from_metric` at `D = 3`. Only `Rphi_ij` is new.
The **mandatory cross-check** is that `Rtilde_ij + Rphi_ij` reproduces the
physical spatial Ricci tensor computed independently by the same Layer 1
routine applied to `gamma_ij` — two genuinely independent paths, so the check
validates the new `Rphi` term rather than restating it. Measured agreement on a
Schwarzschild slice: `6.9e-7` against a component scale of `1.7e-1`, i.e. the
nested finite-difference floor.

## 6. Evolution equations

With `alpha` the lapse and `beta^i` the shift:

```text
d_t phi        = -(1/6) alpha K + beta^k d_k phi + (1/6) d_k beta^k

d_t gammatilde_ij = -2 alpha Atilde_ij + beta^k d_k gammatilde_ij
                  + gammatilde_ik d_j beta^k + gammatilde_jk d_i beta^k
                  - (2/3) gammatilde_ij d_k beta^k

d_t K          = -D^i D_i alpha + alpha ( Atilde_ij Atilde^{ij} + (1/3) K^2 )
                 + beta^i d_i K + 4 pi alpha ( rho + S )

d_t Atilde_ij  = e^{-4 phi} [ -D_i D_j alpha + alpha ( R_ij - 8 pi S_ij ) ]^TF
                 + alpha ( K Atilde_ij - 2 Atilde_ik Atilde^k_j )
                 + beta^k d_k Atilde_ij + Atilde_ik d_j beta^k + Atilde_jk d_i beta^k
                 - (2/3) Atilde_ij d_k beta^k
```

`[ ]^TF` is the trace-free part with respect to `gamma` (equivalently
`gammatilde` — the conformal factors cancel).

`d_t Gammatilde^i` is exposed with its standard terms; because this increment
has **no spatial grid**, the second-shift-derivative terms are supplied through
the derivative provider and vanish identically for the constant and homogeneous
fields the oracles use. Its role here is structural (so the state and its rates
are complete) rather than dynamically exercised.

## 7. The constraint-substituted `d_t K` — an exact, quantified caveat

This is the increment's central scientific finding, established **numerically
before the implementation was written**, not taken from a reference.

Deriving `d_t K` directly from the Layer 3.1 ADM equation gives

```text
d_t K = -D^i D_i alpha + alpha ( R + K^2 ) - 12 pi alpha rho + 4 pi alpha S .
```

The standard BSSN form above is obtained from it by **substituting the
Hamiltonian constraint** `R = 16 pi rho - K^2 + K_ij K^{ij}` to eliminate `R`.
The two therefore differ, **exactly**, by the Hamiltonian residual:

```text
( d_t K )_ADM  -  ( d_t K )_BSSN  =  alpha * H ,
H = R + K^2 - K_ij K^{ij} - 16 pi rho .
```

Verified on deliberately off-constraint data: the measured difference was
`-0.8942257331` and `alpha * H` was `-0.8942257331`, agreeing to **exactly
`0.000e0`**. On the constraint surface the two agree to `5.6e-17`.

Consequences, all of which the implementation honours:

- the constraint-substituted form is **canonical** (it is what BSSN codes use,
  and what makes the formulation work), and it is documented as such;
- **ADM/BSSN right-hand-side equivalence is exact only on the constraint
  surface**, and off it the discrepancy is not an error but precisely `alpha*H`;
- `d_t phi`, `d_t gammatilde_ij`, and `d_t Atilde_ij` are equivalent to ADM
  **unconditionally** — verified independently, and the `Atilde` equation
  matched the chain-rule truth to every printed digit.

The equivalence check therefore reports **both** the raw difference and the
constraint-corrected difference, so a reader can see that the residual is
accounted for rather than tuned away.

## 8. Algebraic constraints and explicit projection

Evaluated and reported **separately**, never blended into one scalar:

```text
D = det gammatilde - 1                          (unit determinant)
T = gammatilde^{ij} Atilde_ij                    (trace-free)
C^i = Gammatilde^i_stored - Gammatilde^i_from_metric   (connection consistency)
```

Each carries signed and absolute residuals, a scale, and a normalized residual
where the scale is meaningful (`None` otherwise — an honest "undefined", never a
fabricated zero).

**Projection is never silent.** Ordinary conversion does *not* project onto the
constraint surface. Two explicit operations are provided — unit-determinant
rescaling of `gammatilde_ij` and trace removal from `Atilde_ij` — each returning
the pre- and post-projection residuals and the correction magnitude, rejecting
singular or non-finite input, and idempotent within tolerance. **No constraint
damping** is added in this increment.

## 9. Oracles

- **A — Minkowski.** `gammatilde = delta`, `phi = 0` (`chi = 1`), `K = 0`,
  `Atilde = 0`, `Gammatilde^i = 0`, every right-hand side zero under unit lapse
  and zero shift, and an exact round trip.
- **B — homogeneous flat FLRW** (de Sitter, dust, radiation, reusing the Layer
  3.2 states): `gammatilde` stays the unit-determinant Euclidean metric,
  `Gammatilde^i = 0`, `Atilde = 0`, and the expansion is carried entirely by
  `phi` and `K`. The BSSN right-hand sides must reconstruct the Layer 3.1 ADM
  right-hand sides — an independent equivalence test, not a comparison against
  duplicated formulas.
- **C — anisotropic homogeneous state.** A manufactured diagonal anisotropic
  metric (clearly labelled as an algebraic oracle, **not** claimed to be a
  physical cosmological solution) exercising nonzero `Atilde_ij`, trace removal,
  determinant normalization, and the round trip.
- **D — curved time-symmetric slice.** The static Schwarzschild slice:
  the conformal Ricci decomposition against the physical Ricci, `K = 0`,
  `Atilde = 0`, and ADM/BSSN right-hand-side agreement.
- **E — deliberate algebraic violations.** Controlled perturbations of the
  determinant, the trace, and the stored connection functions: each residual
  responds to its own perturbation, scales with the amplitude, and is repaired
  by the corresponding explicit projection.

## 10. Deliverable shape

New module `scirust-relativity/src/bssn.rs`, beside `adm`, `adm_evolution`, and
`adm_homogeneous`. Tests in `scirust-relativity/tests/bssn.rs`, benchmarks in
`scirust-relativity/benches/bssn.rs`, and the deterministic
`bssn_homogeneous_evolution` experiment comparing the ADM and BSSN evolution
paths against the exact Friedmann solutions.

Time stepping continues to reuse `scirust_sim::simulate`; **no new integrator
and no duplicated ADM equations** are introduced.

## 11. Known limitations

No spatial grid; no proof of strong hyperbolicity from implementation alone; no
live gauge; no black-hole simulation; no constraint damping; no AMR; no
observational validation. The next increment (**not** authorized by this note)
is the discretized spatial grid, together with the live gauge conditions that a
practical BSSN evolution requires.
