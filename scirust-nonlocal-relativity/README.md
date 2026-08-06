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

**Proper-time-based Caputo memory (non-uniform grid).** `proper_time_caputo_velocity_memory`
evaluates the Caputo velocity-memory vector of an already-computed
affine-parameter trajectory with respect to its own estimated, generally
non-uniform proper-time axis, using the new
`scirust_fractional::caputo_l1_nonuniform` operator — the general L1 scheme
without the uniform-grid assumption, validated independently in
`scirust-fractional` (exactness for linear functions on a non-uniform grid,
and numerical agreement with `caputo_l1_uniform` on a uniform one). This is
a pure post-hoc diagnostic: it does not feed back into the live integration
loop and is unrelated to `NormalizedTimelikeProperTime` mode, which advances
the state equation with a uniform proper-time step by construction. Because
the memory-force law is built to keep `g(u,u)` close to constant along an
accepted trajectory, the proper-time and affine-based memory values differ
mainly by a predictable rescaling (Caputo derivatives scale by `c^(-alpha)`
under a linear reparametrization at constant rate `c`), not by a large
physical effect — see `examples/proper_time_memory_comparison.rs`.

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

### Exact flat-spacetime transport (validation oracle)

Parallel transport on a flat (curvature-free) manifold is path-independent:
transporting along two different paths between the same two points differs
only by transport around the closed loop formed by one path followed by the
reverse of the other, and a flat connection has trivial holonomy around every
contractible loop. `exact_cylindrical_minkowski_transport` uses this directly,
with **no discretization**: it converts the vector to the Cartesian chart
(where its components are unchanged by transport along any path, since the
Cartesian connection vanishes identically), then converts back to the
cylindrical chart at the destination. This is exact only for this specific
flat-spacetime chart pair, along paths that stay within a simply connected
region (in particular, that do not wind around `r = 0`); it is not a general
bitensor propagator, and it does **not** extend to `Schwarzschild` or any
other curved background.

`transport_vector_along_polyline` exposes `DiscreteConnectionTransport`'s
per-segment mechanism directly over an explicit waypoint list, independent of
the full simulation/backend machinery. `examples/exact_transport_convergence.rs`
transports a fixed test vector along the same straight-line path at
increasing waypoint density and compares the numerical result against the
exact oracle: with the shipped parameters, the error shrinks by a factor of
essentially 4 each time the waypoint count doubles (second-order convergence,
consistent with the underlying Heun predictor-corrector scheme), reaching
`~3.5e-8` at 128 waypoints from `~3.5e-5` at 4. This directly validates
`DiscreteConnectionTransport`'s correctness against a known-exact answer,
rather than only against another discretization.

### Exact curved-background transport (circular orbit, follow-up)

The flat-spacetime oracle above is exact because curvature is zero, so
transport is path-independent. Schwarzschild has nonzero curvature, so that
argument does not apply — but along a **circular equatorial geodesic
orbit** (fixed `r`, `theta = pi/2`, constant four-velocity), Schwarzschild's
Christoffel symbols are constant (they depend only on `r` and `theta`), so
the transport equation reduces to a linear, constant-coefficient ODE with
the exact closed-form solution `V(lambda) = exp(-lambda A) V(0)` for the
fixed generator `A`. `exact_schwarzschild_circular_orbit_transport` (with
`schwarzschild_circular_orbit_four_velocity` and
`schwarzschild_circular_orbit_angular_velocity`) evaluates this via a
deterministic 4x4 matrix exponential, reusing the same already-validated
`Schwarzschild::christoffel` this crate uses everywhere else. Validated
against two exact conservation laws that hold for parallel transport under
*any* metric-compatible connection (`g(V,V)` and `g(V,u)` constant), and
against `DiscreteConnectionTransport`'s second-order convergence under path
refinement in `examples/schwarzschild_orbit_transport.rs`. **Exact only for
a circular equatorial geodesic orbit** at radius strictly exceeding `3 M`;
not a general bitensor propagator, and not valid for any other path.

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

## Reissner-Nordström field modulation (follow-up)

`SchwarzschildKretschmannModulator` is tied to one background and, since
Schwarzschild is vacuum (Ricci tensor identically zero), essentially one
available curvature invariant. This follow-up adds a second background,
`scirust_relativity::ReissnerNordstrom` (a static, charged, spherically
symmetric black hole, with exact analytic Christoffel symbols using the
same general lapse-function formula Schwarzschild already uses, reducing to
`Schwarzschild` exactly at `charge = 0`), and
`ReissnerNordstromFieldModulator`, a modulator built from the
electromagnetic field invariant `F_(mu nu) F^(mu nu) = 2 Q^2 / r^4` of the
background's radial Coulomb field — not a curvature invariant. The weight
is `q = 1 + beta * L^2 * |F^2|` (note `L^2`, not `L^4`: this invariant has a
different length-dimension than the Kretschmann scalar). Same `beta = 0.0`
bit-identical bypass pattern as `SchwarzschildKretschmannModulator`. This
crate uses the Reissner-Nordström metric exactly as it uses Schwarzschild's:
fixed, externally specified background data — nothing here solves the
Einstein or Maxwell equations, or computes the field's backreaction on the
metric.

## Adaptive-step worldline integration (follow-up)

Every integrator above advances a fixed affine-parameter step for a fixed
step count, and `proper_time_caputo_velocity_memory` only resamples an
already uniformly-stepped trajectory after the fact.
`simulate_nonlocal_worldline_adaptive` (with `AdaptiveNonlocalConfig`) closes
this gap: the live loop chooses its own non-uniform affine-parameter step,
using the classical embedded Heun-Euler pair (the same Euler predictor and
Heun corrector `HeunPeceStepper` already computes, reused as a
first-order/second-order error estimate — no extra acceleration evaluation
beyond one ordinary Heun step) for step-size control, and evaluating the
memory force with `scirust_fractional::caputo_l1_nonuniform` directly
against the resulting non-uniform history. Step acceptance uses the
componentwise scaled root-mean-square norm `scaled_local_error_norm`, shared
with the step-doubling controller: each coordinate and velocity component is
divided by `abs_tol_i + rel_tol * max(|low_i|, |high_i|)` (from an
`AdaptiveTolerance`), so the accept/reject decision is far less sensitive to
the coordinate chart and to differing component magnitudes than the earlier
unscaled L2 sum — an improvement in scaling robustness, **not** a covariant
error measure. A run that exceeds its accepted-step budget
(`AdaptiveStepBudgetExhausted`), a step whose retries exceed
`max_rejections_per_step` (`AdaptiveRejectionBudgetExhausted`), and a step
whose proposed shrink would cross `min_step` (the distinct
`AdaptiveMinimumStepExhausted`) are all typed errors — never a silently
out-of-tolerance or truncated trajectory. The returned trajectory samples a
non-uniform axis: it must **never** be passed to
`affine_trajectory_proper_time`, whose `step` argument assumes uniform
spacing.

Cross-validated against a very fine independent fixed-step `HeunPeceStepper`
run (agreement to `1.0e-4` or better with the shipped parameters);
`examples/adaptive_worldline.rs` shows it reaching comparable or better
accuracy than an 800-step fixed run with as few as 23 accepted steps at a
loose tolerance. This is a standard adaptive-Runge-Kutta technique, not a
new numerical method, and does not change the underlying state equation.

**Composing adaptive stepping with transport and modulation (follow-up).**
`simulate_nonlocal_worldline_adaptive` does not itself reuse `MemoryLaw` or
`WorldlineStepper` — both thread a single fixed `NonlocalConfig` step
through their signatures, which a variable step size cannot satisfy — but
[`crate::HistoryTransport`] and [`crate::HistoryModulator`] never depended
on a fixed step in the first place (`transport_segment` and `weight` both
take the step or the entry directly), so they compose cleanly.
`simulate_nonlocal_worldline_adaptive_with_policy` takes an
`AdaptiveSimulationPolicy<H, T, M>` (a history backend, a transport, and a
modulator, mirroring `NonlocalSimulationPolicy`'s role for the fixed-step
path) and reuses `HistoryBackend::push_entry` — the exact mechanism
`CompleteUniformHistory` and `BoundedShortMemoryHistory` already use to
transport every retained vector across each newly accepted segment — so
`DiscreteConnectionTransport`, `SchwarzschildKretschmannModulator`, and
`ReissnerNordstromFieldModulator` all compose with adaptive stepping exactly
as they compose with the fixed-step integrators, including together.
`simulate_nonlocal_worldline_adaptive` is now defined as the
`IdentityHistoryTransport` + `IdentityHistoryModulator` +
`CompleteUniformHistory` special case of this function (verified to
reproduce its pre-composition behavior bit-for-bit).
`examples/adaptive_transported_modulated.rs` runs all four combinations of
identity/discrete transport and unmodulated/Kretschmann-modulated memory on
the same trajectory and tolerance.

## Composing adaptive stepping with `MemoryLaw` and `WorldlineStepper` (follow-up)

`simulate_nonlocal_worldline_adaptive_with_policy`'s embedded Heun-Euler
controller still does not reuse `MemoryLaw` or `WorldlineStepper` themselves
(as opposed to `HistoryTransport`/`HistoryModulator`, composed above): both
traits thread a single fixed `NonlocalConfig` step through their signatures,
and `CaputoCoordinateMemory` specifically applies that one step to the
*entire* retained history via `caputo_l1_uniform`, which is only correct
when every accepted segment shares it.

`simulate_nonlocal_worldline_adaptive_with_stepper_policy` closes this for
`SemiImplicitEulerStepper` specifically. Two new `MemoryLaw` implementations,
`NonuniformCaputoCoordinateMemory` and
`NonuniformModulatedCaputoCoordinateMemory<M>`, read each retained sample's
own recorded parameter and evaluate `caputo_l1_nonuniform` directly instead
of assuming uniform spacing — they compose with the fixed-step architecture
too, not only the adaptive one, since `MemoryLaw` was already generic over
the history backend. Error control uses classical step-doubling (one full
step compared against two half steps, since semi-implicit Euler is
first-order and has no natural embedded higher-order partner) rather than an
embedded pair, with the same `1/(p+1) = 0.5` growth/shrink exponent as the
embedded controller above, since semi-implicit Euler is also a first-order
method.

`HeunPeceStepper` is deliberately **not** offered here — but no longer for a
parameter-formula reason. `StepperContext` now carries the true accumulated
affine parameter (`current_parameter`), and `HeunPeceStepper`'s predictor
computes its provisional point as `current_parameter + config.step`, so it is
sound under the non-uniform spacing adaptive stepping produces (this replaced
an earlier `step_index * config.step` reconstruction that was exact only when
every accepted step shared one size). The reason it is excluded here is that
this step-doubling controller's error estimate is specialised to a
first-order method, whereas Heun is second order, and adaptive Heun-PECE
already exists as `simulate_nonlocal_worldline_adaptive`'s embedded
Heun-Euler controller — which computes the same Heun corrector and also
exposes the Euler predictor the error estimate needs. Adding it here would be
a strictly inferior duplicate. See `adaptive_stepper.rs`'s module
documentation for the full reasoning.

`AdaptiveStepperPolicy<H, L, T>` (a history backend, a `MemoryLaw`, and a
transport) mirrors `NonlocalSimulationPolicy`'s role for this path, narrowed
to the components it actually varies — there is no stepper type parameter,
since `SemiImplicitEulerStepper` is the only sound choice.
`simulate_nonlocal_worldline_adaptive_with_stepper` is the
`NonuniformCaputoCoordinateMemory` + `IdentityHistoryTransport` +
`CompleteUniformHistory` special case. `examples/adaptive_worldline_stepper.rs`
runs several `MemoryLaw`/transport combinations on the same trajectory and
tolerance, plus a sanity-anchor row against the plain entry point.

## v2 hardening: scaled error control and history retention

The v2 hardening round makes the two adaptive controllers share one control
core and adds a research knob:

- `AdaptiveTolerance { relative, coordinate_absolute, velocity_absolute }` and
  `scaled_local_error_norm` define the shared componentwise scaled RMS error
  norm both controllers use. `AdaptiveNonlocalConfig::new` remains a
  compatibility constructor that seeds a uniform tolerance from one scalar;
  `AdaptiveNonlocalConfig::with_tolerance` sets the three fields independently.
- `HistoryRetention` (`EndpointOnly` default / `RefinedAcceptedHistory`) and
  `simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention` expose
  the persistent-history retention comparison of the step-doubling controller.
  The default is unchanged; `RefinedAcceptedHistory` additionally persists each
  accepted step's midpoint at its true affine parameter. On the shipped
  experiment (`experiments/nonlocal-relativity-v2`, binary `history_retention`)
  this does **not** improve endpoint accuracy while roughly doubling the memory
  cost, so it is research-only and `EndpointOnly` stays the default.

These are numerical-controller improvements. They do not change the state
equation, are coordinate-componentwise (not covariant), and carry no physical
claim.

## Kerr background (follow-up)

`scirust_relativity::Kerr` (mass `M`, spin `a = J/M`) is a third fixed
background: a stationary, axisymmetric, **rotating** black hole in standard
Boyer-Lindquist coordinates. Unlike `Schwarzschild` and `ReissnerNordstrom`,
its connection is evaluated by central finite differences
(`scirust_relativity::numerical_christoffel`), not an exact analytic
formula — Kerr's Christoffel symbols are algebraically far more complex (the
metric depends on both `r` and `theta` and has a nonzero off-diagonal
`t`-`phi` term), and hand-deriving them risked a transcription error with no
independent way to catch it. At `a = 0` the metric reduces to
`Schwarzschild`'s exactly, and the finite-difference Christoffel symbols
agree with `Schwarzschild`'s exact analytic ones to the finite-difference
tolerance — this crate's test suite checks both directly, alongside a
frame-dragging sign check (the ZAMO angular velocity `-g_tphi/g_phiphi` is
positive for positive spin, matching the Lense-Thirring precession
direction) and symmetric/hand-derived-value checks.

This crate's worldline and memory machinery runs on `Kerr` completely
unmodified — the only Kerr-specific code is its `Metric`/`Connection`
implementations. `examples/kerr_worldline.rs` runs a stationary observer
(deliberately not a Kerr circular orbit, which has no simple closed form
and is not derived or claimed anywhere in this crate) at increasing spin,
showing the coordinate `phi` stay exactly zero at `spin = 0` and pick up a
positive, spin-scaling drift at `spin > 0` — frame dragging emerging from
the geodesic equation and the finite-difference Christoffel symbols working
together, without being hand-coded anywhere.

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
cargo run -p scirust-nonlocal-relativity --example exact_transport_convergence
cargo run -p scirust-nonlocal-relativity --example proper_time_memory_comparison
cargo run -p scirust-nonlocal-relativity --example schwarzschild_orbit_transport
cargo run -p scirust-nonlocal-relativity --example reissner_nordstrom_field_modulation
cargo run -p scirust-nonlocal-relativity --example adaptive_worldline
cargo run -p scirust-nonlocal-relativity --example adaptive_transported_modulated
cargo run -p scirust-nonlocal-relativity --example kerr_worldline
cargo run -p scirust-nonlocal-relativity --example adaptive_worldline_stepper
```

The first example compares `kappa = 0` with a small positive coupling for an
exterior Schwarzschild worldline. The convergence study prints deterministic
CSV-like rows comparing Euler and Heun PECE on a short Schwarzschild exterior
experiment. `coordinate_covariance` prints deterministic CSV rows comparing
raw coordinate memory and `DiscreteConnectionTransport` memory across
Cartesian and cylindrical Minkowski charts, at three refinement levels.
`curvature_modulated_memory` prints deterministic CSV rows comparing
unmodulated and Schwarzschild-Kretschmann-modulated memory, at two
refinement levels and with both transport strategies. `exact_transport_convergence`
prints deterministic CSV rows showing `DiscreteConnectionTransport`'s
numerical error against the exact flat-spacetime transport oracle shrinking
under path refinement. `proper_time_memory_comparison` prints deterministic
CSV rows comparing affine-parameter and proper-time-based Caputo memory on
the same Schwarzschild exterior trajectory, at three refinement levels.
`schwarzschild_orbit_transport` prints deterministic CSV rows showing
`DiscreteConnectionTransport`'s numerical error against the exact
circular-orbit transport oracle in the curved Schwarzschild background
shrinking under path refinement. `reissner_nordstrom_field_modulation`
prints deterministic CSV rows comparing unmodulated and
electromagnetic-field-modulated memory on a Reissner-Nordström exterior
trajectory, at two refinement levels. `adaptive_worldline` prints
deterministic CSV rows comparing the adaptive integrator's accepted-step
count and accuracy against an independent fine fixed-step reference, across
tightening error tolerances. `adaptive_transported_modulated` prints
deterministic CSV rows comparing all four combinations of identity/discrete
transport and unmodulated/Kretschmann-modulated memory under adaptive
stepping, on the same trajectory and tolerance. `kerr_worldline` prints
deterministic CSV rows for a stationary observer in the Kerr background at
increasing spin, showing frame dragging emerge as a spin-scaling drift in
the `phi` coordinate. `adaptive_worldline_stepper` prints deterministic CSV
rows comparing several `MemoryLaw`/transport combinations under the
step-doubling adaptive integrator (`simulate_nonlocal_worldline_adaptive_with_stepper_policy`),
plus a sanity-anchor row against the plain entry point.
