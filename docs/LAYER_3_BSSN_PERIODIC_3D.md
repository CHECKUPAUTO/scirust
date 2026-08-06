# Layer 3.7 — BSSN on a periodic three-dimensional grid

Layers 3.5 and 3.6 generalised the grid to `D` dimensions and validated two of
them. The third axis was carried by the type system but had **no evolution
validation of any kind**. This increment supplies it, and adds the one
configuration no lower-dimensional grid can represent.

## 1. What is new here, and what is not

**Not new**: the equations, and — this is worth stating plainly — the
dimension-generic code. `bssn_grid` has been generic over `D` since Layer 3.5,
and `UniformGrid<3>` already worked. Nothing in the physics changed.

**New**: evidence, and one generalisation that the evidence required.

- a body-diagonal **gauge** wave, and a body-diagonal **gravitational** wave;
- convergence with all three mixed-derivative pairs present;
- stability, benchmarks, and a deterministic 3D experiment;
- `TransverseTracelessWave` generalised from "along `x`" to an arbitrary
  direction and polarization.

**No 3D-specific defect was found.** Layers 3.5 and 3.6 each exposed a real bug;
this one did not. That is a result, not an omission, and it is reported as such.

## 2. The diagonal gravitational wave — the reason a third axis matters

The gauge wave of Layer 3.6 extends to the body diagonal without difficulty:
`grad^2 sin(k(x+y+z)) = -3 k^2`, so under 1+log slicing

```text
alpha(t, x, y, z) = 1 + A cos(k sqrt(6) t) sin(k (x + y + z))
```

exactly. But that only exercises the **lapse**. The gravitational degrees of
freedom live in `gammatilde_ij` and `Atilde_ij`, and the crate's only wave
oracle propagated along `x` with `P = diag(0, 1, -1)` — two non-zero components,
one varying axis.

A wave along `n = (1,1,1)/sqrt(3)` with

```text
P = e1 (x) e1 - e2 (x) e2,   e1 = (1,-1,0)/sqrt(2),   e2 = (1,1,-2)/sqrt(6)
```

has **all six independent components of `h_ij` non-zero while all three axes
vary**. Neither a one- nor a two-dimensional grid can arrange that. `P` is
exactly traceless and exactly transverse in binary floating point — `1/3 + 1/3`
is `2/3` with no rounding — so the test asserts those with `assert_eq!` against
zero rather than a tolerance.

| N | `L∞(gamma_ij − exact)` | relative | order |
| --- | --- | --- | --- |
| 8 | `9.347855e-9` | `9.35e-3` | — |
| 16 | `2.435925e-9` | `2.44e-3` | **1.94** |

The Hamiltonian constraint stays at `1.8e-10 → 3.1e-10`: the `O(A^2)` level the
linearized oracle neglects, not growth.

**Category: exact solution of the LINEARIZED vacuum field equations.** The code
solves the full nonlinear system, so the difference carries both `O(A^2)`
nonlinearity and `O(dx^2)` truncation; `A = 1e-6` makes truncation dominate.
This is *not* a nonlinear-wave oracle.

## 3. Generalising the oracle rather than copying it

`TransverseTracelessWave` now carries a direction and a polarization. Three
things made this safe to do rather than merely tidy:

- **`new` is bit-for-bit unchanged.** Its accessors are pinned by `assert_eq!`
  against the old closed form at 64 sample points, and both the 1D and 2D
  experiments reproduce byte-identical output (`10840` and `4388` bytes). The
  phase is computed as `k (n . x - t)` rather than `k . x - omega t` precisely
  so that `n = (1,0,0)` recovers `k (x - t)` exactly; the two forms are equal in
  real arithmetic and *not* equal in floating point.
- **The signature changed from `f64` to `&[f64; 3]`.** Keeping an `x`-only
  accessor would have silently dropped `y` and `z` for a diagonal wave. Forcing
  a compile error at every call site was the point.
- **`plane` validates.** Symmetry, tracelessness, transversality, and unit norm
  are checked, each with its own typed `InvalidPolarization` error. A
  polarization failing any of them is not a solution of the linearized field
  equations, so evolving it would compare the code against something that is not
  an oracle — and would look like a convergence failure.

## 4. The mixed derivatives, and a trap in the measurement

Two dimensions can only ever populate `d_x d_y`. The `(0,2)` and `(1,2)` pairs
are reached here for the first time. All three are bit-for-bit symmetric in
their axes, and the conformal Ricci reconstruction converges:

| N | `L∞(R_bssn − R_generic)` | `‖R‖` | `‖Γ̃^k‖` | order |
| --- | --- | --- | --- | --- |
| 8 | `7.462842e-2` | `8.327e-1` | `6.765e-2` | — |
| 16 | `2.024365e-2` | `9.257e-1` | `7.332e-2` | **1.88** |
| 32 | `5.165485e-3` | `9.508e-1` | `7.621e-2` | **1.97** |

The first version of this measurement reported order **0.95**, and it was
tempting to read that as a third-dimension bug. It was not. The probe metric
carried a `sin(k(y + 3z))` term — three wavelengths across the box, so `N = 8`
gives 2.67 points per wavelength, *below Nyquist*. With every phase reduced to
one wavelength the order recovered to 1.88 → 1.97. The oracle metric in both the
test and the experiment documents this, because the failure mode looks exactly
like a defect in the code under test.

## 5. Stability

| N | Courant | max state | `det gammatilde − 1` |
| --- | --- | --- | --- |
| 8 | 0.25 | `1.000755` | `4.19e-10` |
| 8 | 0.5 | `1.000667` | `1.26e-8` |
| 16 | 0.25 | `1.000915` | `2.07e-11` |
| 16 | 0.5 | `1.000910` | `6.98e-10` |

Bounded near unity — the lapse is `O(1)` and every other field is `O(A)` — and
the determinant constraint **falls by a factor ~20 under refinement at fixed
Courant**. That is what separates truncation from an instability, and it is the
property actually being claimed.

The test tolerance is derived from this model rather than from the number a run
happens to produce. It initially failed at `4.19e-10` against a `1e-10` bound
inherited from the 2D test — but the 2D experiment had itself measured
`3.6e-10` at `N = 8`, and only asserted at `N >= 16`. The bound is now `1e-9`
with the reasoning recorded at the assertion, and every sample is checked rather
than the final one, so an excursion that returns cannot pass.

The Hamiltonian residual under a live gauge is *not* small in absolute terms
(`1.7e-2` at `N = 8`, against a perturbation curvature scale of `A k^2 ≈ 4e-2`).
It converges at order ~1.74 between `N = 8` and `N = 16`, so it is truncation,
not drift — but free evolution applies no constraint damping, and at eight
points per wavelength that truncation is large. This is stated rather than
buried.

**Stability here is measured, not proven.** Strong hyperbolicity is an analytic
property of the continuum system, and nothing here establishes it.

## 6. Benchmarks

Machine-dependent wall clock; the computation is deterministic.

| operation | N=4 (64 pts) | N=8 (512 pts) | N=16 (4096 pts) |
| --- | --- | --- | --- |
| 3D BSSN right-hand side | 1.73 ms | 13.77 ms | 98.38 ms |

Ratios 7.96 and 7.14 against an 8× growth in point count — linear in the
*total* points. About 25 µs per grid point, roughly **twice** the 2D figure of
12.4 µs: each point now takes nine mixed second differences rather than four,
and three first-derivative axes rather than two. The cost of a third dimension
is therefore `N^3` growth **and** a constant factor of two, not `N^3` alone.

This is what bounds the test suite. A single `N = 16` evolution to `t = 1` costs
minutes in an unoptimised build, so the resolutions and step counts in the tests
are chosen against a stated error model — not for convenience.

## 7. Determinism

No RNG, no wall clock, no parallelism. Points are visited in ascending linear
index order and reductions accumulate in that order. The 3D experiment is
byte-identical across runs (5404 bytes), and the 1D and 2D experiments are
byte-identical to their pre-Layer-3.7 output.

## 8. Known limitations

- **Periodic domains only.** No outer boundary, radiative or otherwise.
- **Weak, smooth fields only.** No puncture, no horizon, no strong field — which
  is what the moving-puncture gauge exists for and the only setting in which its
  behaviour is really tested.
- **Coarse resolutions.** The `N^3` cost caps routine validation at `N = 16` for
  evolutions and `N = 32` for right-hand-side measurements.
- **Square cells only.** Anisotropic grids are refused with a typed error,
  because the Layer 3.3 conversion path differences with a single step.
- **Stability is measured, not proven**, and only over the times, resolutions,
  and Courant factors tabulated above.
- The `O(A^2)` limit of the wave oracle is a property of linearized gravity, not
  of this code; no nonlinear-wave oracle exists here.
- No black holes, no punctures, no excision, no constraint damping, no AMR, no
  waveform extraction, no observational validation.
