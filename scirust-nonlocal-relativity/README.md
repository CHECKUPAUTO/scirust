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

For non-null worldlines,

```text
P^rho_sigma = delta^rho_sigma - u^rho u_sigma / s
s = g_(mu nu) u^mu u^nu
u_sigma = g_(sigma nu) u^nu
```

The projection is checked through the diagnostic residual
`u_rho F_memory^rho`.

## Numerical contract

- Complete velocity history is retained.
- The baseline Caputo evaluation uses `scirust_fractional::caputo_l1_uniform`.
- The first sample uses a zero memory vector because the Caputo history is
  insufficient.
- The baseline history cost is `O(D * N^2)` over `N` fixed steps.
- The update is documented semi-implicit Euler:

```text
u_(n+1) = u_n + h a_n
x_(n+1) = x_n + h u_(n+1)
```

This is a deterministic reference integrator, not a precision integrator.
There is no RNG, no parallel reduction, no hidden global state, and no automatic
four-velocity renormalization. Metric-norm drift is measured and exposed.

All quantities use the coordinate system and geometric units of the supplied
background. The discretization is coordinate-dependent.

Positive `kappa` is a finite non-negative phenomenological damping-like
coupling. It is not a new fundamental constant.

## Example

```bash
cargo run -p scirust-nonlocal-relativity --example schwarzschild_memory
```

The example compares `kappa = 0` with a small positive coupling for an exterior
Schwarzschild worldline and prints compact CSV-like rows.
