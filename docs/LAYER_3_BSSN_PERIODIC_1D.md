# Layer 3.4 — BSSN on a periodic one-dimensional grid

Layer 3.3 delivered the BSSN algebra at a *point*. Every spatial-derivative term
in the evolution equations vanished identically there, because every field was
spatially constant. This increment supplies the missing ingredient: a spatial
grid. It is the first increment in which BSSN is a partial differential equation
rather than an algebraic identity.

It is also deliberately the smallest such increment that can be validated
honestly.

## 1. Why one spatially varying coordinate

The obvious next step after Layer 3.3 is "a 3D grid". That step is not taken
here, for the same reason Layer 3.2 declined to put ADM on a grid: a result that
cannot be checked against something independent is not a result.

A one-dimensional periodic domain buys three things a 3D domain does not:

- **Exact analytic derivative oracles.** On a periodic domain a smooth field's
  derivatives are known in closed form, so the finite-difference operators can be
  validated against `sin(kx)` rather than against themselves.
- **No boundary problem.** Periodicity is an exact boundary condition. An outer
  radiative boundary is an open research problem and a large source of error
  that would contaminate every convergence measurement made here.
- **An exact evolution oracle.** A linearized transverse-traceless wave
  propagating along `x` is an exact solution of *linearized* general relativity
  in this gauge, with a closed-form time dependence to compare against.

Fields vary only along `x`, but every grid point stores complete
three-dimensional tensors — `gammatilde_ij` and `Atilde_ij` are full symmetric
3x3 objects, `Gammatilde^i` is a full 3-vector. This is a **1D3V reduction**:
one spatial dimension, three-component vectors, three-by-three tensors. It is
**not** a general three-dimensional numerical-relativity solver and must not be
described as one.

## 2. How this reuses Layer 3.3 — and the measurement that decided it

Every Layer 3.1 and Layer 3.3 evaluator has the same shape:

```rust
pub fn bssn_evolution_rhs<G: Metric<3>>(
    spatial_metric: &G, extrinsic_curvature: &impl SpatialTensorField,
    coordinates: &[f64; 3], gauge: &BssnGauge, sources: &AdmSources,
    settings: &AdmEvolutionSettings,
) -> Result<(BssnState, BssnEvolutionRhs), BssnError>
```

The fields are **sampled by coordinate**, and all spatial derivatives are taken
internally by nested central differences at `settings.spatial_step` and
`settings.metric_step`. There is no seam through which precomputed grid
derivatives can be injected.

That left two options: refactor `bssn.rs` to accept injected derivatives, or
supply a field that *reads from the grid* and let the existing machinery
difference it. The second duplicates nothing, but risks a real pathology: the
nested difference samples `x ± 2 dx`, and a second difference of spacing `2 dx`
can decouple the even and odd grid points into independent sub-lattices — the
classic checkerboard mode.

This was measured before it was decided. A grid-lookup `Metric<3>` (nearest
periodic index) was fed to `conformal_ricci` with
`spatial_step = metric_step = dx`, against a converged analytic reference:

| N   | dx       | Linf(R_grid − R_exact) | ratio | even points | odd points |
| --- | -------- | ---------------------- | ----- | ----------- | ---------- |
| 32  | 0.031250 | `2.574330e-3`          | —     | `2.574330e-3` | `2.522839e-3` |
| 64  | 0.015625 | `6.462596e-4`          | 3.983 | `6.462596e-4` | `6.430096e-4` |
| 128 | 0.007812 | `1.617200e-4`          | 3.996 | `1.617200e-4` | `1.615198e-4` |
| 256 | 0.003906 | `4.042696e-5`          | 4.000 | `4.042696e-5` | `4.041133e-5` |

Two conclusions about **accuracy**, both load-bearing:

- The error ratio converges to **exactly 4.000** per halving. The grid path is
  second-order accurate.
- Even- and odd-index *errors* agree to **0.04%** at `N = 256`, so the
  truncation error shows no even/odd split on smooth data.

On that basis the grid-provider route was adopted, and `bssn.rs` is **not
modified at all**: no BSSN equation, no Ricci engine, no constraint evaluator,
and no integrator is duplicated by this increment.

**That probe measured accuracy, and accuracy was not the whole question.** The
first evolution built on this reuse was unstable — but, as section 13 records,
the cause was *not* the stencil width this probe examined. It was that Layer
3.3's `conformal_ricci` computes the generic metric Ricci and never reads
`Gammatilde^i`, so the system carried ADM's principal part. The reuse route
survives for everything except the Ricci tensor, which the grid now supplies in
genuine BSSN form. Section 13 is the most important section of this document.

This also answers a question Layer 3.3 could not. There, the conformal Ricci
decomposition agreed with the independently computed physical Ricci to `~1e-6`,
and that was correctly described as a *nested finite-difference floor* — a fixed
number with no error model. On a grid it is not a floor at all: it is truncation
error, and it converges at the discretization order.

**The consequence for callers is a hard requirement**: on the grid,
`spatial_step` and `metric_step` **must both equal `dx`**. Any other value makes
the internal difference sample coordinates that are not grid points, where the
provider can only return its nearest neighbour — first-order garbage. The grid
system sets these itself and does not take them from the caller.

## 3. Grid convention

A **half-open** uniform periodic domain:

```text
x_n = x_min + n * dx,    n = 0, ..., N-1,    dx = (x_max - x_min) / N
```

The upper endpoint `x_max` is **not** stored: it is the periodic image of
`x_min`. Storing both would duplicate a degree of freedom and make the
determinant of the discrete system singular.

Periodic indexing wraps with Euclidean remainder, never with unsigned integer
wraparound:

```text
wrap(i) = i.rem_euclid(N)
```

so the left neighbour of index `0` is `N-1` and the right neighbour of `N-1` is
`0`, with no underflow possible on any offset, positive or negative.

Rejected at construction: fewer points than the stencil requires, non-finite
bounds, non-positive domain length, and a non-finite or non-positive spacing.

## 4. Finite-difference operators

Second-order centred, periodic:

```text
D1 f_i = ( f_{i+1} - f_{i-1} ) / ( 2 dx )
D2 f_i = ( f_{i+1} - 2 f_i + f_{i-1} ) / ( dx^2 )
```

Validated against `f(x) = sin(kx)`, whose derivatives `k cos(kx)` and
`-k^2 sin(kx)` are exact, at several wave numbers below the Nyquist limit and
over at least four resolutions. Second-order convergence is **measured**, not
asserted from the stencil formula.

A fourth-order stencil is deliberately **not** implemented. It is not required
for a validated second-order pipeline and would compete for validation effort
with the parts of this increment that are actually new.

These operators are used for diagnostics and for the manufactured-state oracle.
They are **not** the differencing used inside the BSSN right-hand side — that is
Layer 3.3's nested difference, characterised in section 2. The documentation
says so plainly rather than implying a single unified stencil.

## 5. Field storage and the flat state

`BssnGridState` stores **17 scalar component arrays** in structure-of-arrays
order. Component `c` occupies the contiguous slice `[c*N, (c+1)*N)`:

| slot | component | slot | component |
| ---- | --------- | ---- | --------- |
| 0 | `phi` | 8..13 | `Atilde_xx, xy, xz, yy, yz, zz` |
| 1..6 | `gammatilde_xx, xy, xz, yy, yz, zz` | 14..16 | `Gammatilde^x, ^y, ^z` |
| 7 | `K` | | |

Symmetric rank-2 tensors store only their six independent components, so
symmetry is preserved **structurally** — it cannot be broken by a round trip,
because the redundant entries do not exist.

Structure-of-arrays is chosen because a component derivative reads `f[i-1]`,
`f[i]`, `f[i+1]` at stride 1, and because the flat layout `scirust_sim` requires
is then the storage itself: encoding and decoding are contiguous copies, not
scattered gathers. There is no separate "flattening" step to get wrong.

The flat method-of-lines state is exactly this array, so its length is `17 N`.

## 6. Method of lines

The semidiscrete system is `dY/dt = F(t, Y)` with `Y` the `17 N` array above.
Time integration reuses `scirust_sim::simulate` — the platform's existing
deterministic fixed-step RK4 — through its existing `System` trait, exactly as
Layer 3.2 and `GeodesicSystem` already do. **No integrator is written here.**

Per right-hand-side evaluation the grid system loops over points in ascending
index order (deterministic), builds the local ADM pair by reconstruction, calls
Layer 3.3's `bssn_evolution_rhs`, and writes the 17 components into the output
slice. Reductions accumulate in ascending index order, so they are
bit-reproducible.

## 7. Gauge

Prescribed only: `alpha = 1`, `beta^i = 0`. Gauge is **not evolved**. There is no
1+log slicing, no Gamma-driver shift, no gauge damping, and no evolved gauge
variable. This is a **major limitation**, not an implementation detail: live
gauge is what makes strong-field BSSN evolution work in practice, and none of it
is present.

## 8. Constraint monitoring

Free evolution. Constraints are **monitored and never enforced**. Projection
exists as an explicit opt-in experiment mode, disabled by default, and is never
applied between RK4 stages in the default evolution.

Monitored separately — never blended into a single score:

- BSSN algebraic: unit conformal determinant, trace-free `Atilde`, and
  `Gammatilde^i` consistency.
- ADM physical, from reconstructed fields: the Hamiltonian constraint and the
  three momentum-constraint components.

Each is reduced over the grid to a signed mean, `L1`, discrete `L2`, `Linf`, the
index attaining the maximum (lowest index wins ties, deterministically), and a
non-finite count.

## 9. Validation oracles

- **A — stationary Minkowski.** Identity conformal metric, `phi = 0`, `K = 0`,
  `Atilde = 0`, `Gammatilde^i = 0`, vacuum. Every right-hand side and every
  constraint must stay at machine level for many steps, with no drift.
- **B — manufactured periodic state.** A smooth periodic BSSN field with known
  derivatives. It does **not** satisfy the Einstein equations and is labelled a
  manufactured numerical oracle, not a physical solution. It validates packing,
  derivative extraction, and pointwise assembly, and it measures spatial
  convergence.
- **C — linearized transverse-traceless wave.** `h_yy = A sin(k(x-t))`,
  `h_zz = -A sin(k(x-t))`, with `K_ij = -(1/2) d_t gamma_ij` following the
  repository's sign convention. This is an exact solution of *linearized*
  general relativity in this gauge. The code evolves the **full nonlinear** BSSN
  system, so the measured difference contains both `O(A^2)` nonlinearity and
  `O(dx^2)` truncation; the amplitude is chosen so truncation dominates and the
  convergence measurement is meaningful. This is **not** nonlinear-wave
  validation.
- **D — off-constraint state.** A controlled periodic constraint violation,
  evolved freely, showing the expected initial residual and the absence of
  silent repair.
- **E — algebraic projection.** Injected determinant and trace violations,
  measured, explicitly projected, and compared against the unprojected
  evolution.

## 10. Numerical dissipation

Kreiss–Oliger dissipation is implemented but **disabled by default**
(`sigma = 0`), and is never enabled implicitly:

```text
Q f_i = -(sigma / 16 dx) ( f_{i+2} - 4 f_{i+1} + 6 f_i - 4 f_{i-1} + f_{i-2} )
```

The bracket's Fourier symbol is `4 (1 - cos theta)^2`: zero for a constant
field, maximal (`16`) at the Nyquist mode. For smooth `f` the bracket is
`O(dx^4)`, so `Q f = O(dx^3)` and second-order accuracy is preserved.

It is **numerical dissipation on the evolved variables**, not constraint
damping: it neither targets nor reduces the constraint residuals by
construction.

The undissipated scheme was measured first, as it must be — dissipation added
before measurement hides an instability rather than revealing one. That
discipline paid: dissipation demonstrably failed to cure the instability the
first revision had, which is what forced the real diagnosis in section 13. With
the BSSN principal part in place it is very nearly a no-op (amplitude ratio
`1.0066` at every `sigma` tested), which is the correct behaviour when there is
no high-frequency growth to damp.

## 11. Determinism

No RNG, no wall clock, no parallelism, no hidden global state. Point iteration
and reduction order are fixed and ascending. Identical inputs produce
byte-identical output, verified by running the experiment twice and comparing
bytes.

## 12. Measured results

All figures from `bssn_periodic_1d_evolution`, byte-identical across runs.

**Oracle A — stationary Minkowski.** At `N = 16, 32, 64, 128` the state change
after 1 unit of coordinate time is `0.000000e0` — *exactly* stationary, not
merely small. The right-hand side of a spatially constant field is exactly zero
because a centred difference subtracts identical values, so RK4 adds exactly
zero and there is no rounding to accumulate. Every constraint is `0.000000e0`.

**Oracle B — manufactured state, conformal Ricci reconstruction.**

| N | dx | mismatch L1 | L∞ | ‖R̃‖ | ‖R^φ‖ | order |
|---|---|---|---|---|---|---|
| 32 | 3.125e-2 | 6.224e-5 | 9.891e-5 | 1.961e-1 | 2.599e-3 | — |
| 64 | 1.563e-2 | 1.601e-5 | 2.521e-5 | 1.981e-1 | 2.624e-3 | **1.97** |
| 128 | 7.813e-3 | 4.029e-6 | 6.333e-6 | 1.986e-1 | 2.630e-3 | **1.99** |
| 256 | 3.906e-3 | 1.009e-6 | 1.585e-6 | 1.987e-1 | 2.632e-3 | **2.00** |

Layer 3.3 could only report this mismatch as a fixed `~1e-6` *floor* with no
error model. On a grid it is truncation error and converges at the
discretisation order. Both parts are genuinely nonzero, so the agreement is not
two zeros matching.

**Oracle C — linearized wave, short-time spatial convergence** (`A = 1e-6`,
`k = 2π`, `t = 0.1`, `C = 0.25`): metric `L∞` error `2.377e-9 → 6.063e-10 →
1.519e-10 → 3.815e-11` at `N = 16..128`, observed order **1.97 / 2.00 / 1.99**.

The wave's *constraint* residual behaves differently, and the difference is
physical rather than numerical. At fixed `A = 1e-6` the Hamiltonian residual is
flat in resolution (`7.74e-11 → 8.98e-11` across `N = 32..256`), but at fixed
`N = 64` it scales as `A^2` — measured exponents **3.96 / 4.01 / 4.00 per
amplitude doubling**. It is the quadratic nonlinearity the linearized oracle
neglects, not discretisation error. The truncation contribution cancels because
the linear-order Ricci *scalar* is proportional to `d^2(h_yy + h_zz)` and
`h_yy + h_zz = 0` holds exactly at every grid point. Asserting `dx`-convergence
of this quantity would have been asserting something false.

**Temporal convergence.** Measured against a finely-stepped run at the *same*
spatial resolution, so the spatial truncation error cancels exactly and what
remains is purely the RK4 error of the semidiscrete ODE system. Difference `L∞`
`7.945e-9 → 4.966e-10 → 3.101e-11 → 1.937e-12`, observed order **4.00 / 4.00 /
4.00**. Measuring against the analytic PDE solution instead would be dominated
by `dx^2` and would report a misleadingly low order.

**Benchmarks** (machine-dependent wall clock; the computation is deterministic):

| operation | N=32 | N=64 | N=128 |
|---|---|---|---|
| periodic first derivative | 73.0 ns | 142.0 ns | 282.6 ns |
| periodic second derivative | 71.6 ns | 143.7 ns | 297.3 ns |
| full BSSN grid RHS | 930 µs | 1.84 ms | 3.66 ms |
| constraint monitor | 1.07 ms | 2.13 ms | 4.23 ms |

Both the derivatives and the right-hand side scale linearly in `N` (RHS ratios
1.98 and 1.99 under doubling), as a one-dimensional grid should. Single-point
costs: conformal connection 3.55 µs, conformal Ricci 25.1 µs. Flat-state round
trip at `N = 128` is 181.6 ns — a memcpy, because storage *is* the flat layout.
Initial data from analytic ADM fields at `N = 64` costs 88.3 µs. Ten RK4 steps
at `N = 32` cost 36.7 ms, i.e. 3.67 ms per step — exactly four right-hand-side
evaluations, confirming no hidden work per stage.

Per grid point the RHS costs ~28.6 µs against Layer 3.3's ~10 µs local figure;
the difference is the `bssn_to_adm` reconstruction the provider performs on
every nested-difference sample. `bssn_grid_rhs` performs **zero heap
allocations** — the input is a borrowed view and the output is written in place,
and the providers are `Copy`. Storage is 17 f64 = **136 bytes per grid point**;
`simulate` adds six full-state buffers during integration.

## 13. Genuine BSSN form — the correction that made it stable

This is the increment's most important result, and it arrived as a correction to
an earlier wrong diagnosis. Both are recorded, because the wrong one is
instructive.

### What was broken

The first revision reused Layer 3.3's `conformal_ricci` unchanged. That computes
`Rtilde_ij` as the **generic Ricci tensor** of `gammatilde_ij` — correct as a
tensor identity, and it is why the two forms agree on the constraint surface.
But it is not the BSSN system. It also inherited Layer 3.3's
`d_t Gammatilde^i = 0`, which was correct there (every term of that equation
carries a spatial gradient, and Layer 3.3 had none) and wrong on a grid.

The result was BSSN *variables* carrying ADM's *principal part*. It behaved
exactly as weakly hyperbolic ADM does: `N >= 64` blew up before `t = 1` at every
Courant factor from `0.1` to `2.0`, the onset time roughly halved as the
resolution doubled, and Kreiss–Oliger dissipation could not cure it — at
`N = 128` it moved onset only from `t = 0.50` to `t = 0.60`, and at `N = 64` the
coefficient that averted the abort inflated the physical wave amplitude by
`2.3e3`.

### The wrong diagnosis, and the right one

That behaviour was first attributed to stencil width: composing an outer and an
inner difference of width `dx` spans `2 dx`, and a `2 dx`-spaced second
difference has symbol `2 cos(2 theta) - 2`, which vanishes at the Nyquist mode.
That observation is true, but it was **not the cause**. Two measurements settled
it:

- The connection constraint grew monotonically from machine zero
  (`1.36e-6 -> 3.06e-6 -> 9.6e-6 -> 6.3e-5`), and at early times it was nearly
  **resolution-independent** (`1.359e-6`, `1.374e-6`, `1.375e-6` at `t = 0.05`
  for `N = 32, 64, 128`). A discretisation artefact would scale with `dx`; a
  missing equation does not.
- Supplying `d_t Gammatilde^i` alone did **not** fix it, and made some cases
  worse.

The second measurement is what pointed at the real cause: `conformal_ricci`
never reads `Gammatilde^i` at all. Evolving a variable the principal part
ignores changes nothing about the principal part.

### The fix

Both changes are required, and neither suffices alone:

- `conformal_ricci_from_derivatives` writes `Rtilde_ij` in genuine BSSN form,

  ```text
  Rtilde_ij = -1/2 gammatilde^{lm} d_l d_m gammatilde_ij
            + gammatilde_{k(i} d_{j)} Gammatilde^k
            + Gammatilde^k Gammatilde_{(ij)k}
            + gammatilde^{lm} ( 2 Gammatilde^k_{l(i} Gammatilde_{j)km}
                              + Gammatilde^k_{im} Gammatilde_{klj} )
  ```

  where the second term carries the **evolved** `Gammatilde^k`. That term is
  what removes the mixed second derivatives `d_i d_k gammatilde_jl` from the
  principal part, leaving the manifestly elliptic `-1/2 gammatilde^{lm} d_l d_m`.
  This substitution is the entire reason BSSN exists.
- `bssn_connection_rhs` supplies the `d_t Gammatilde^i` equation, derived here
  from the definition `Gammatilde^i = -d_j gammatilde^{ij}` rather than
  transcribed, and constraint-substituted via the momentum constraint exactly as
  `d_t K` is.

The BSSN-form Ricci is validated against the generic one — they must agree, and
the difference converges at observed order **1.99 / 2.00 / 2.00**.

### Measured afterwards

| N | outcome |
|---|---|
| 32, 64, 128, 256 | all reach `t = 4`; error **decreases** under refinement |

Courant sweep: stable for `C <= 1` at every resolution; `C = 2` rejected at
**both** `N = 64` and `N = 128`. A resolution-**independent** boundary is what a
genuine CFL limit looks like, as opposed to the resolution-dependent onset the
broken version showed. (`N = 32` tolerates `C = 2` with a visibly degraded
error, so the boundary is not sharp at very coarse resolution.) **No rigorous
CFL bound is derived, and none is claimed.**

The connection constraint now grows only secularly — roughly doubling as the
time doubles — and is resolution-independent (`~9.9e-6` at `t = 0.5` for every
`N`).

Dissipation becomes very nearly a no-op: the amplitude ratio is `1.0066` at
every `sigma` from `0` to `0.5`, and the error is unchanged. That is the correct
behaviour when there is no high-frequency growth left to damp, and it is why
dissipation is not needed and stays off by default.

**None of this proves strong hyperbolicity.** That is an analytic property of
the continuum system. These are measurements on one discretisation of a
one-dimensional reduction, and they are reported as such.

## 14. Live slicing — 1+log

The lapse is now an **evolved field**, stored at every grid point (slot 17 of
18), driven by

```text
d_t alpha = -2 alpha K
```

with zero shift. Enabling it is always explicit: [`BssnSlicing::Prescribed`] is
the default, under which `d_t alpha = 0` exactly and every earlier result is
reproduced bit-for-bit.

Turning the lapse on activates three terms that were structurally zero while it
was constant, all of which are now supplied:

- `-D^i D_i alpha` in `d_t K`;
- `-e^{-4 phi} ( D_i D_j alpha )^TF` in `d_t Atilde_ij`;
- `-2 Atilde^{ij} d_j alpha` in `d_t Gammatilde^i`.

The covariant Hessian `D_i D_j alpha = d_i d_j alpha - Gamma^k_ij d_k alpha` uses
the **physical** Christoffel symbols, obtained from the conformal ones by

```text
Gamma^k_ij = Gammatilde^k_ij
           + 2 ( delta^k_i d_j phi + delta^k_j d_i phi
                 - gammatilde_ij gammatilde^{kl} d_l phi )
```

so the physical metric is never differenced a second time and the compact grid
stencils carry through.

### The oracle: gauge speed exactly `sqrt(2)`

Linearised about flat space, `d_t alpha = -2 alpha K` and `d_t K = -D^2 alpha`
combine to `d_t^2 alpha = 2 d_x^2 alpha` — a wave equation with characteristic
speed `sqrt(2)`. That is *faster than light*, and legitimately so: the lapse is
gauge and carries no physical signal.

Standing-wave data `alpha = 1 + A sin(kx)` with `K = 0` therefore evolves as
`alpha = 1 + A cos(sqrt(2) k t) sin(kx)` — a closed form with no free parameters
to fit. Measured at `A = 1e-6`, `k = 2 pi`, `t = 0.25`:

| N | `L∞(alpha - exact)` | order | numerical / exact amplitude |
| --- | --- | --- | --- |
| 32 | `2.842e-9` | — | `0.995309` |
| 64 | `7.103e-10` | **2.00** | `0.998828` |
| 128 | `1.778e-10` | **2.00** | `0.999708` |

Second order in `dx`, with the amplitude ratio converging to `1`. The gauge speed
is not fitted or asserted — it is the analytic prediction, and the numerics meet
it.

Minkowski with unit lapse has `K = 0`, so `d_t alpha = -2 alpha K` is *exactly*
zero and enabling 1+log leaves the vacuum solution bit-for-bit undisturbed.

A non-finite or non-positive lapse is rejected with a located typed error; the
lapse is never clamped.

## 15. The shift, and the Gamma-driver

The shift `beta^i` and the driver auxiliary `B^i` are now stored fields too
(slots 18–20 and 21–23 of 24), so every shift term in every equation is live:

```text
d_t phi          += beta^j d_j phi + (1/6) d_j beta^j
d_t gammatilde_ij += beta^k d_k gammatilde_ij + gammatilde_ik d_j beta^k
                   + gammatilde_jk d_i beta^k - (2/3) gammatilde_ij d_k beta^k
d_t K            += beta^j d_j K
d_t Atilde_ij    += beta^k d_k Atilde_ij + Atilde_ik d_j beta^k
                   + Atilde_jk d_i beta^k - (2/3) Atilde_ij d_k beta^k
d_t Gammatilde^i += beta^j d_j Gammatilde^i - Gammatilde^j d_j beta^i
                   + (2/3) Gammatilde^i d_j beta^j
                   + (1/3) gammatilde^{li} d_l d_j beta^j + gammatilde^{lj} d_j d_l beta^i
d_t alpha        += beta^j d_j alpha
```

### The sharpest oracle in this document

For a **spatially constant** shift every `d(beta)` term vanishes identically, so
the only surviving contribution is the advection `beta^j d_j`. The difference
between the shifted and unshifted right-hand sides must therefore be exactly
`v * d_x(field)` for every one of the seventeen evolved components — and since
both sides use the same compact stencil, it must hold to **rounding**, not to a
tolerance:

| v | residual | advection scale | relative |
| --- | --- | --- | --- |
| 0.1 | `1.735e-18` | `1.972e-3` | `8.80e-16` |
| 0.3 | `8.674e-19` | `5.916e-3` | `1.47e-16` |
| 0.7 | `6.776e-21` | `1.380e-2` | `4.91e-19` |

A sign error in any single advection term would show up here immediately. This
is why the constant-shift case was tested before anything was evolved.

### The Gamma-driver

```text
d_t beta^i = (3/4) B^i
d_t B^i    = d_t Gammatilde^i - eta B^i
```

The `3/4` sets the shift's characteristic speed to exactly `1` — the speed of
light — so it is not a free parameter. `eta >= 0` damps the long-wavelength
drift an undamped driver develops; a negative `eta` would amplify precisely what
the term exists to suppress, and is rejected.

The driver is fed the **full** connection rate, shift terms included. Feeding it
a partial rate would make the shift chase a quantity nothing else evolves.

On flat space with a constant shift, `d_t Gammatilde^i` is exactly zero and the
driver decouples into two exact ODEs with a closed-form solution
`B = B0 exp(-eta t)`, `beta = beta0 + (3/4)(B0/eta)(1 - exp(-eta t))`. Measured
against it at `t = 1`:

| eta | worst deviation | scale | relative |
| --- | --- | --- | --- |
| 1 | `7.698e-13` | `1.896e-1` | `4.06e-12` |
| 2 | `9.098e-12` | `1.297e-1` | `7.01e-11` |
| 4 | `3.973e-11` | `7.363e-2` | `5.40e-10` |

Prescribed shift is the **default**: `d_t beta^i = d_t B^i = 0`, freezing both
bit-for-bit. With `beta^i = 0` that reproduces every earlier result exactly.
Minkowski under the *full* moving-puncture gauge — 1+log slicing and the
Gamma-driver together — is bit-for-bit stationary, because `Gammatilde^i = 0`,
`K = 0` and `B^i = 0` make every gauge rate exactly zero.

### What is still not here

The gauge is now complete in form, but it has only been exercised on **weak,
smooth, periodic** data. Nothing here involves a puncture, a horizon, or a
strong field — which is what the moving-puncture gauge was invented for and the
only setting in which its behaviour is really tested.

## 16. Known limitations

Stated plainly, because the gap between this and a numerical-relativity code is
large:

- One spatially varying coordinate only. Transverse derivatives are exactly zero
  **by construction**, not by approximation.
- Periodic domain only. No outer boundary, radiative or otherwise.
- Prescribed gauge only. No live gauge of any kind.
- Weak, smooth fields only. No singular spacetimes.
- **Stability is measured, not proven.** The scheme is stable across every
  resolution tested for `C <= 1`, but that is an empirical statement about one
  discretisation of a 1D reduction, not a theorem.
- **Strong hyperbolicity is not proven by these tests.** Passing a weak-field
  convergence test is evidence, not proof; hyperbolicity is a property of the
  PDE system established analytically, and nothing here establishes it.
- No general three-dimensional validation.
- No black holes, no punctures, no excision.
- No adaptive mesh refinement.
- No waveform extraction.
- No observational validation.
