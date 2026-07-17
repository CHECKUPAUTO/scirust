# scirust-nonlocal-relativity

**EXPERIMENTAL.** This crate studies a fractional-memory modification of
test-particle worldline dynamics on a fixed general-relativistic background.

It does not implement fractional Einstein equations. It does not modify the
Einstein field equations, Einstein tensor, stress-energy tensor, matter-sourced
curvature, or established general relativity. No empirical validation is
claimed.

The current model evolves coordinates `x^rho` and contravariant velocity
`u^rho` on a background implementing `scirust_relativity::Metric<D>` and
`scirust_relativity::Connection<D>`. The ordinary geodesic acceleration is
augmented by a projected Caputo velocity-memory force:

```text
a_GR^rho = - Gamma^rho_(mu nu)(x) u^mu u^nu
m^rho(lambda_n) = CaputoDerivative_alpha[u^rho](lambda_n)
F_memory^rho = - kappa P^rho_sigma m^sigma
du^rho / d lambda = a_GR^rho + F_memory^rho
```

This is an ordinary first-order state equation with a fractional-history force
on the right-hand side. It is not a Caputo fractional differential equation for
the state variables themselves.

For non-null worldlines,

```text
P^rho_sigma = delta^rho_sigma - u^rho u_sigma / s
s = g_(mu nu) u^mu u^nu
u_sigma = g_(sigma nu) u^nu
```

The projection is checked through the diagnostic residual
`u_rho F_memory^rho`.

## Numerical contract

- The default backend retains complete velocity history and is the numerical
  memory oracle.
- The baseline Caputo evaluation uses `scirust_fractional::caputo_l1_uniform`.
- The first sample uses a zero memory vector because the Caputo history is
  insufficient.
- The baseline history cost is `O(D * N^2)` over `N` fixed steps.
- The explicit bounded short-memory backend retains only the most recent
  `W >= 2` samples. It is an approximation with `O(D * N * W)` history cost
  over `N` fixed steps and must be selected explicitly.
- The compatibility/default update is semi-implicit Euler:

```text
u_(n+1) = u_n + h a_n
x_(n+1) = x_n + h u_(n+1)
```

This is a deterministic reference integrator, not a precision integrator. An
additive explicit integrator API also provides `heun_pece`, a
predict-evaluate-correct-evaluate Heun method for the same ordinary state
equation:

```text
u*       = u_n + h a_n
x*       = x_n + h u*
a*       = a(x*, u*, provisional history including u*)
u_(n+1)  = u_n + h/2 (a_n + a*)
x_(n+1)  = x_n + h/2 (u_n + u_(n+1))
```

The provisional history is used only to evaluate the predicted acceleration;
accepted histories remain complete deterministic velocity histories.

## Phase 2 architecture

The advanced simulation API separates four responsibilities:

- `HistoryBackend<D>` stores accepted velocity samples and reports retained and
  used sample counts.
- `HistoryTransport<D>` maps retained samples into the current coordinate
  frame before memory evaluation. The production transport is coordinate
  identity/no-transport.
- `MemoryLaw<D>` evaluates the memory vector. The production law is the current
  coordinate Caputo L1 velocity-memory law.
- `WorldlineStepper<D>` advances the ordinary first-order state equation. The
  production steppers are semi-implicit Euler and Heun PECE.

Transport is separate from memory because future transported-history studies
should not change the Caputo stencil or history storage contract. The current
identity transport preserves the existing coordinate-memory model.

There is no RNG, no parallel reduction, no hidden global state, and no automatic
four-velocity renormalization. Metric-norm drift is measured and exposed.

All quantities use the coordinate system and geometric units of the supplied
background. The discretization is coordinate-dependent.

Positive `kappa` is a finite non-negative phenomenological damping-like
coupling. It is not a new fundamental constant.

## Phase 3: transported memory and proper time

Phase 3 extends the Phase 2 architecture additively. All Phase 1/2 public
items keep their original signatures and bit-for-bit behavior; every new
capability is opt-in.

### Typed history entries

`HistoryBackend::push_velocity` and `HistoryBackend::sample` are unchanged:
they store and return bare velocity components, which is all
`IdentityHistoryTransport` and the coordinate-memory pipeline ever needed.
That narrow shape is also exactly why Phase 2 transport could not be
geometric: a transport cannot carry a vector between two different tangent
spaces if it is only ever given the vector's components, with no source
point. `HistoryEntry<D>` (coordinates, contravariant velocity, accepted
parameter) is the typed accepted sample that supplies that source point.
`HistoryBackend::push_entry` and `HistoryBackend::entry` are new, additive
trait methods built on it; their default implementations fall back to the
original velocity-only behavior, so a backend that does not override them is
honestly limited to coordinate-identity transport.

### Discrete parallel transport

`DiscreteConnectionTransport` is a deterministic, explicitly discretized
approximation of parallel transport along the accepted worldline polyline.
Each time a new segment is accepted (including a Heun-PECE provisional
predictor evaluation), every currently retained history vector is advanced
by one Heun predict-evaluate-correct-evaluate step of the linear transport
equation `dV^mu/dlambda = -Gamma^mu_(alpha beta) u^alpha V^beta`:

```text
1. evaluate the transport derivative at the segment start;
2. predict the vector at the segment end;
3. evaluate the connection and velocity at the segment end;
4. correct with the average of the two derivatives.
```

Because this runs once per accepted segment for every retained vector,
transport accumulates along the actual accepted path rather than jumping
directly between a sample's original point and the current point, which
matters because parallel transport is path-dependent under a curved
connection. This is **not** an exact analytic bitensor propagator, **not** a
proof of covariance, and its discretization error grows with the segment
step and the number of transported segments. `IdentityHistoryTransport` is
unchanged and remains the production coordinate-memory transport; the
original coordinate memory model (raw components, no transport) is preserved
exactly.

**Complexity.** Transporting all retained vectors once per accepted step
costs `O(N)` transport evaluations per step (`O(N^2)` over `N` steps), each
evaluating a Christoffel contraction, i.e. `O(D^3)`. The discrete-transport
pipeline therefore costs `O(D^3 * N^2)` overall — more expensive than the
`O(D * N^2)` raw coordinate-memory baseline, exactly as expected for a
strategy that transports every retained vector directly instead of only
touching the newest sample.

### Affine parameter vs. proper time

`ParameterizationMode` makes the meaning of the fixed step explicit without
changing the numerical scheme: both modes advance the same uniform
`parameter_n = n * h`.

- `AffineParameter` is the default, unconstrained mode: no timelike
  assumption, bit-for-bit compatible with every other entry point in this
  crate.
- `NormalizedTimelikeProperTime { tolerance }` interprets the configured step
  as a proper-time step. It requires the initial state to be timelike with
  `g(u,u)` within `tolerance` of `-1` under a `(-,+,+,+)` signature, and
  requires every subsequently sampled step's metric norm to stay within
  `tolerance` of `-1`. Null and spacelike initial states, and states from an
  incompatible signature background, are rejected the same way: their norm is
  not close to `-1`. **No automatic four-velocity renormalization is ever
  performed** — a drift beyond `tolerance` is reported as a typed
  `ProperTimeNormDrift` error instead of being silently repaired.

`simulate_nonlocal_worldline_with_mode` wraps `simulate_nonlocal_worldline_with_policy`
with these checks; it does not change the underlying stepper.

**Proper-time diagnostics for affine trajectories.** `affine_trajectory_proper_time`
estimates how much proper time elapsed along an *affine*-parameter
trajectory of timelike states, using the left-endpoint quadrature
`Delta tau_n ~= h * sqrt(-g(u,u)_n)` evaluated at the accepted state at the
start of each step. This is a first-order-accurate diagnostic estimate, not
a resampling of the trajectory onto a uniform proper-time grid: the returned
increments are generally non-uniform and **must never** be passed to
`scirust_fractional::caputo_l1_uniform`, which requires uniform spacing.

### Coordinate-chart comparison

The Caputo velocity memory is evaluated componentwise in whatever chart the
background supplies, so it is coordinate-dependent by construction (Phase 1
already stated this). `CylindricalMinkowski` is the same flat spacetime as
`Minkowski`, expressed in cylindrical coordinates `(t, r, phi, z)`, whose
connection is not identically zero — unlike the Cartesian chart. Together
with `cartesian_to_cylindrical_coordinates`, `cartesian_to_cylindrical_velocity`,
`cylindrical_to_cartesian_coordinates`, and `cylindrical_to_cartesian_velocity`
(exact Jacobian transforms, not numerical approximations), it lets the same
physical motion be computed in two charts and compared.

`examples/coordinate_covariance.rs` runs a memory-coupled worldline in
Cartesian coordinates (the reference) and the same physical initial
condition in cylindrical coordinates, once with raw coordinate memory and
once with `DiscreteConnectionTransport`, at three refinement levels. On that
controlled experiment, transported memory's disagreement with the Cartesian
reference is consistently smaller than raw coordinate memory's (by roughly
three orders of magnitude in the shipped parameters) and shrinks under
refinement, while raw coordinate memory's disagreement stays roughly
constant — it is not a discretization artifact, it is the chart-dependence
itself. This is a controlled numerical demonstration, not a proof of exact
agreement between charts, and not a claim of covariance.

## Phase 4: curvature-modulated memory (research hook)

Phase 4 adds one more additive, opt-in `MemoryLaw`: a deterministic scalar
modulation of retained history vectors, applied before the Caputo
evaluation. It composes with every Phase 2/3 component (either backend,
either transport, either integrator, either parameterization mode) because
it only changes what number the Caputo stencil consumes at each retained
sample.

`HistoryModulator<D>` transforms one finite `HistoryEntry<D>` into a finite,
dimensionless scalar weight:

- `IdentityHistoryModulator` always returns `1.0`.
- `SchwarzschildKretschmannModulator` is an explicitly experimental,
  phenomenological instance: `q = 1 + beta * L^4 * K`, where
  `K = 48 M^2 / r^6` is the Schwarzschild Kretschmann scalar and `L` is a
  strictly positive reference length that makes the modulation
  dimensionless. It requires a strictly positive finite mass, a strictly
  positive finite reference length, a finite non-negative `beta`, and a
  finite radius strictly outside the horizon; it rejects a non-finite or
  non-positive resulting weight. **When `beta == 0.0`, evaluation bypasses
  the Kretschmann computation entirely and returns exactly `1.0`**, so a
  modulated pipeline reproduces the unmodulated baseline bit-for-bit
  whenever the rest of the numerical path is identical.

`ModulatedCaputoCoordinateMemory<M>` is the `MemoryLaw` that applies a
`HistoryModulator`'s weight to each retained (and, when a geometric
transport is used, already-transported) velocity sample, componentwise,
before the Caputo L1 stencil runs. The result is exactly a Caputo derivative
of a dimensionless *modulated* velocity history — nothing more. It must
**never** be described as a unique consequence of general relativity, a
quantum-gravity prediction, an experimentally derived law, or a modification
of the Einstein field equations. No structure resembling a modified field
equation, Einstein tensor, or stress-energy tensor is introduced anywhere in
this crate.

## Convergence studies

`run_convergence_study` compares the same final affine parameter at `h`,
`h/2`, and `h/4`. It reports endpoint coordinate and velocity differences,
observed self-convergence ratios, endpoint metric-norm drift, and endpoint
memory-force norm. The `h/4` result is a refinement reference, not an exact
oracle for the continuous model; self-convergence can reveal numerical
stability trends but cannot validate the physical model or prove the continuum
equation is correct.

## Example

```bash
cargo run -p scirust-nonlocal-relativity --example schwarzschild_memory
cargo run -p scirust-nonlocal-relativity --example convergence_study
cargo run -p scirust-nonlocal-relativity --example coordinate_covariance
cargo run -p scirust-nonlocal-relativity --example curvature_modulated_memory
```

The first example compares `kappa = 0` with a small positive coupling for an
exterior Schwarzschild worldline. The convergence study prints deterministic
CSV-like rows comparing Euler and Heun PECE on a short Schwarzschild exterior
experiment. `coordinate_covariance` prints deterministic CSV rows comparing
raw coordinate memory and `DiscreteConnectionTransport` memory across
Cartesian and cylindrical Minkowski charts, at three refinement levels.
`curvature_modulated_memory` prints deterministic CSV rows comparing
unmodulated and Schwarzschild-Kretschmann-modulated memory, at two
refinement levels and with both transport strategies.
