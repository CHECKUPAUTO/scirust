# scirust-itd

A deterministic **2-D field-simulation core**, ported to pure Rust from the
*ITD* research simulator. It gives SciRust its first spatial-field / PDE-flavoured
kernel, alongside the ODE-style plants in [`scirust-sim`](../scirust-sim) and the
1-D quadrature/FEM in [`scirust-solvers`](../scirust-solvers).

Everything here was accepted only after matching the reference implementation
as an **oracle** — SciRust's standing discipline. The Rust operators, structural
signature and scenario indices are checked against values produced by the
original engine to a tight tolerance (`tests/oracle.rs`, `~1e-9` relative,
including the published 161×161 / 401-step configuration), and the analytic
invariants the reference asserts hold here too (`tests/analytic.rs`).

## What it provides

| Layer | API | Notes |
|---|---|---|
| **Field operators** | [`operators::gradient`] | second-order finite differences; reproduces NumPy `gradient(edge_order=2)` for uniform **and** non-uniform rectilinear axes, finite or periodic boundaries |
| | [`operators::vorticity`] | the 2-D curl `ω = ∂v_y/∂x − ∂v_x/∂y` |
| | [`operators::spatial_mean`] | domain integral / area (2-D trapezoidal quadrature; arithmetic mean in periodic mode) |
| | [`operators::bounded`] | saturating map `b(x) = x / (1 + x)` |
| **Structural signature** | [`signature::structural_metrics`] | heterogeneity, localization, roughness, sign-mixing, temporal deformation + weighted score |
| **Semi-Lagrangian transport** | [`transport::transport_previous_vorticity`] | periodic advection of a field: midpoint / RK4 back-tracing × four periodic interpolations (bilinear, 16-point cubic, convex-limited `cubic_local_bounded`, sum-preserving `cubic_local_sum_preserving`), with the exact-snap short circuit |
| **Simulation driver** | [`simulate`] / `simulate_canonical` | curvature-weighted rotational intensity `⟨ω²·e^{L²κ}⟩` and the structural signature, reduced to interval-integrated indices |
| | [`simulate_transport_compensated`] | the transport-compensated temporal-deformation mode (advect the previous vorticity before the temporal term; periodic boundary) |
| **Scenarios** | [`scenarios`] | calm (irrotational), coherent (rigid rotation), multi-vortex fields + the shared curvature weighting and reference `Config` |
| **Geometric transforms** | [`transforms::BilinearTransformPlan`] | rotation / reflection covariance of a sampled field (`f_Q(x)=f(Qᵀ(x−o)+o)`, `v_Q=Q·v`), with an **exact node-permutation** short circuit when the transform maps grid nodes onto grid nodes, and bilinear interpolation otherwise |
| **Covariance laws** | [`covariance`] | spatial-dilation (`x=o+(x'−o)/a`, `v_a=a·v`, `R_a=R/a²`) and moving-frame (Galilean `x=x'+c(t−t₀)`, `v'=v−c`; time-dependent translation `x=x'+b(t)`, `v'=v−ḃ`) coordinate and field laws |
| **Multi-scale profile** | [`multiscale::derive_multiscale_profile`] | a whole family of structural profiles derived from **one** `ℓ=1` reference run, exploiting the roughness component's exact linearity in the structural length (`roughness_raw(ℓ)=ℓ·roughness_raw(1)`) |
| **Material derivative** | [`material::material_vorticity_interval`] | splits a vorticity change over an interval into Eulerian `(ω₁−ω₀)/Δt`, advective `u·∇ω` and material tendencies, each reduced to an RMS-normalized rate |

## Quick start

```bash
cargo run -p scirust-itd --example itd_scenarios   # reproduce the published indices
cargo test -p scirust-itd                          # operator + scenario oracle checks
```

```rust
use scirust_itd::{simulate_canonical, Config, Scenario, SimConfig};

let r = simulate_canonical(Scenario::Coherent, &Config::default(), &SimConfig::default())?;
println!("intensity = {:.12}", r.intensity_index);   // 4.347614838944
# Ok::<(), scirust_itd::ItdError>(())
```

## Design

- **Pure Rust, zero dependencies**, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`.
- **Deterministic**: no randomness, no threads; identical inputs give identical
  numbers, and the finite-difference/quadrature summation order matches the
  reference so the cross-language agreement stays near machine precision.
- Both temporal-deformation modes are ported: the default *eulerian* mode and
  the semi-Lagrangian *transport-compensated* mode. All four of the reference's
  periodic interpolations are ported — the two exact schemes (`bilinear`,
  `cubic`) and both limiters (the convex-limited `cubic_local_bounded` and the
  discrete-sum-preserving `cubic_local_sum_preserving`) — for both the midpoint
  and RK4 trajectory methods.
- The reference's **field-geometry** machinery is ported as four modules —
  orthogonal transforms (`transforms`), spatial-dilation and moving-frame
  covariance laws (`covariance`), the multi-scale structural profile
  (`multiscale`), and the material-derivative interval diagnostic (`material`) —
  each validated against the same oracle. The higher-level `simulate_material_
  deformation` orchestration (a thin loop wiring `material_vorticity_interval`
  into a full run) is intentionally left in the reference; the reusable interval
  kernel is what transfers.

## Provenance & scope

This is a faithful **numerical** port, not a physical claim. As the reference
project states, its tests establish internal numerical and software
consistency; they do **not** establish the intensity index as a validated
physical observable, an entropy, or a universal measure of complexity. What
transfers to SciRust is the deterministic, oracle-validated machinery.

[`operators::gradient`]: https://docs.rs/scirust-itd
[`operators::vorticity`]: https://docs.rs/scirust-itd
[`operators::spatial_mean`]: https://docs.rs/scirust-itd
[`operators::bounded`]: https://docs.rs/scirust-itd
[`signature::structural_metrics`]: https://docs.rs/scirust-itd
[`transport::transport_previous_vorticity`]: https://docs.rs/scirust-itd
[`simulate`]: https://docs.rs/scirust-itd
[`simulate_transport_compensated`]: https://docs.rs/scirust-itd
[`scenarios`]: https://docs.rs/scirust-itd
[`transforms::BilinearTransformPlan`]: https://docs.rs/scirust-itd
[`covariance`]: https://docs.rs/scirust-itd
[`multiscale::derive_multiscale_profile`]: https://docs.rs/scirust-itd
[`material::material_vorticity_interval`]: https://docs.rs/scirust-itd
