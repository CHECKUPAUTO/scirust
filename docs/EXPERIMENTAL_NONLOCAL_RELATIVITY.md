# Experimental Nonlocal Relativity Layer

This document describes an **EXPERIMENTAL** SciRust research layer for
fractional-memory test-particle worldline dynamics on a fixed
general-relativistic background.

It is not a theory of fractional Einstein equations. It does not modify the
Einstein field equations, the Einstein tensor, the stress-energy tensor,
matter-generated curvature, or established general relativity. No empirical
validation is claimed.

## Index Conventions

- Coordinates are `x^rho`.
- Contravariant coordinate velocity is `u^rho = dx^rho / d lambda`.
- Greek indices such as `rho`, `mu`, `nu`, and `sigma` range over the supplied
  coordinate dimension `D`.
- Repeated indices are summed in the equations.
- The metric is covariant, `g_(mu nu)`.
- Christoffel symbols are indexed as `Gamma^rho_(mu nu)`.
- All quantities use the chart and geometric units of the supplied background.

## Equations

The fixed background implements both `Metric<D>` and `Connection<D>`. The
ordinary geodesic acceleration is

```text
a_GR^rho = - Gamma^rho_(mu nu)(x) u^mu u^nu .
```

The complete uniformly sampled velocity history is used to define a coordinate
Caputo memory vector:

```text
m^rho(lambda_n) = CaputoDerivative_alpha[u^rho](lambda_n) .
```

The implementation delegates this operation to
`scirust_fractional::caputo_l1_uniform`. It does not duplicate the fractional
operator.

The current velocity is lowered with the metric:

```text
u_sigma = g_(sigma nu) u^nu .
```

The metric norm is

```text
s = g_(mu nu) u^mu u^nu .
```

For non-null worldlines, the memory vector is projected orthogonally to the
current velocity:

```text
P^rho_sigma = delta^rho_sigma - u^rho u_sigma / s .
```

The experimental memory force is

```text
F_memory^rho = - kappa P^rho_sigma m^sigma .
```

The total trajectory-level equation is

```text
du^rho / d lambda = a_GR^rho + F_memory^rho .
```

The diagnostic residual

```text
u_rho F_memory^rho
```

is exposed so the projection can be audited numerically.

## Assumptions

- The spacetime geometry is fixed externally.
- The worldline is a test-particle trajectory and does not source curvature.
- The sampled worldline is non-null; `|s|` must exceed a configurable positive
  floor.
- The fractional order is in the first-release supported interval
  `0 < alpha < 1`.
- `kappa` is finite and non-negative.
- Positive `kappa` is a phenomenological damping-like coupling, not a new
  fundamental constant.
- The discretization is coordinate-dependent.
- The uniform affine-parameter step is finite and positive.
- Invalid non-finite numerical values are rejected rather than repaired.

## Numerical Algorithm

For the auditable baseline implementation:

1. Validate configuration, initial coordinates, and initial velocity.
2. Retain the complete velocity history.
3. At sample zero, use a zero memory vector because the Caputo L1 stencil has
   insufficient history.
4. For each later sample, evaluate each component of the Caputo memory vector
   from the complete uniform velocity history.
5. Evaluate the metric, metric norm, Christoffel symbols, ordinary geodesic
   acceleration, projected memory force, and diagnostics in fixed loop order.
6. Reject non-finite metric components, Christoffel symbols, memory values,
   forces, accelerations, generated states, and diagnostics.
7. Reject `|g_(mu nu) u^mu u^nu|` below the configured floor.
8. Advance with semi-implicit Euler:

```text
u_(n+1) = u_n + h a_n
x_(n+1) = x_n + h u_(n+1)
```

There is no RNG, no hidden global state, no parallel reduction, and no
automatic four-velocity renormalization. Metric-norm drift is measured and
reported instead.

The baseline history cost is `O(D * N^2)` over `N` fixed steps because each
step recomputes direct Caputo histories for all `D` velocity components.

Semi-implicit Euler is a reference integrator for reproducible experiments, not
a precision integrator.

## Falsifiable Observables

The current API exposes quantities that can be compared across `kappa = 0`,
small positive `kappa`, and independent implementations:

- coordinate trajectory samples `x^rho(lambda_n)`;
- velocity samples `u^rho(lambda_n)`;
- metric-norm drift from the initial sample;
- coordinate L2 norm of the Caputo memory vector;
- coordinate L2 norm of the projected memory force;
- orthogonality residual `u_rho F_memory^rho`;
- coordinate L2 norm of the ordinary geodesic acceleration;
- deviations from an uncoupled geodesic baseline in a fixed chart.

These are numerical observables of this discretized model. Agreement or
disagreement with physical data is not claimed.

## Known Limitations

- The memory kernel is applied componentwise in coordinates and is therefore
  coordinate-dependent.
- The current implementation is a trajectory-level constitutive experiment,
  not a covariant field theory.
- Complete-history direct evaluation has quadratic cost in the number of
  samples.
- The Euler update is low order and intended for auditability, not accuracy.
- The background connection and metric are assumed to be supplied consistently
  by the caller; the crate validates finiteness but does not prove geometric
  compatibility.
- Null and nearly null worldlines are outside this first implementation.
- No adaptive stepping, event handling, error estimation, or history
  compression is included.

## Roadmap

### 1. Current Worldline-Memory Model

The current crate implements a fixed-background, test-particle, coordinate
Caputo-memory modification of the worldline equation. It exposes deterministic
diagnostics and explicit failure modes.

### 2. Future Covariant Kernel Research

Future research may study kernels defined with bitensors, parallel transport,
proper-time history, or other covariant constructions. These are research
directions only and are not established physics in this crate.

### 3. Hypothetical Field-Equation Work

Any future field-equation investigation would require a separate mathematical
and numerical contract, independent validation, and clear distinction from
established general relativity. This crate does not implement such work, and
this roadmap item is not a claim that fractional field equations are
established physics.
