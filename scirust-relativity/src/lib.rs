//! # `scirust-relativity` — deterministic differential geometry
//!
//! This crate provides a minimal, auditable foundation for spacetime geometry
//! and geodesic simulation:
//!
//! - covariant metric tensors;
//! - deterministic metric inversion;
//! - numerical Levi-Civita Christoffel symbols;
//! - four-dimensional Minkowski, Schwarzschild, Reissner-Nordström, Kerr,
//!   de Sitter, anti-de Sitter, and spatially flat FLRW spacetimes;
//! - Riemann, Ricci, Einstein, and Kretschmann curvature tensors from any
//!   metric-and-connection background, validated against exact analytic
//!   oracles (see [`CurvatureTensors`]);
//! - a reusable parallel-transport engine for vectors ([`transport_along_segment`],
//!   [`transport_along_polyline`], [`holonomy_defect`]), covectors, and rank-2
//!   covariant tensors ([`transport_covector_along_segment`],
//!   [`transport_covariant_tensor_along_segment`]), validated by metric
//!   compatibility and the holonomy/curvature identity;
//! - geodesic equations compatible with `scirust-sim`, geodesic-deviation
//!   (Jacobi) fields ([`integrate_geodesic_deviation`]) validated against the
//!   separation of nearby geodesics, and geodesic exponential/logarithm maps
//!   ([`geodesic_exponential`], [`geodesic_logarithm`]);
//! - local orthonormal frames (tetrads) for timelike observers
//!   ([`orthonormal_tetrad`]), built by metric Gram-Schmidt and validated by
//!   orthonormality, completeness, and preservation under parallel transport;
//! - Synge's world function and its bitensors — the biscalar and its
//!   first-derivative gradients ([`world_function`],
//!   [`world_function_with_gradients`]) and the van Vleck-Morette determinant
//!   ([`van_vleck_determinant`]) — built on the geodesic logarithm map and a
//!   deterministic [`determinant`], validated by flat exactness, the fundamental
//!   identity, base/field symmetry, and the known maximally-symmetric
//!   coincidence expansion;
//! - linearized gravity ([`LinearizedField`]) — the field equations to first
//!   order in a metric perturbation about Minkowski (the opening slice of the
//!   Layer 2 Covariant Gravity Workbench), validated by weak-field-Schwarzschild
//!   vacuum, the Newtonian Poisson limit, gauge invariance, and an `O(h^2)`
//!   cross-check against the nonlinear curvature;
//! - PPN parameter extraction ([`ppn`]) — the Eddington-Robertson `gamma` and
//!   `beta` from static isotropic weak-field metrics by deterministic asymptotic
//!   extrapolation, with an explicit isotropic-coordinate contract and validated
//!   against synthetic and isotropic-Schwarzschild oracles (`gamma = beta = 1`);
//! - the Einstein-Hilbert action and its **numerical** variation ([`action`]) —
//!   the vacuum field equations from a finite-difference functional derivative
//!   `delta S / delta g` for static, axisymmetric backgrounds, built on the
//!   metric-only [`ricci_scalar_from_metric`] and validated by vacuum
//!   stationarity (`G_(mu nu) + Lambda g_(mu nu) = 0`), a mismatched-`Lambda`
//!   nonzero cross-check, and grid convergence;
//! - 3+1 (ADM) kinematics ([`adm`]) — the lapse, shift, spatial metric, and
//!   extrinsic curvature of a foliation, with the Hamiltonian and momentum
//!   constraints as oracles (they vanish for exact solutions), reusing
//!   [`ricci_scalar_from_metric`] and [`numerical_christoffel`] at `D = 3` and
//!   validated on Schwarzschild, de Sitter, FLRW, and the horizon-penetrating
//!   [`PainleveGullstrand`] slicing. The bridge to Layer 3;
//! - the ADM constraint and evolution core ([`adm_evolution`]) — Layer 3.1: the
//!   Hamiltonian and momentum constraints and the evolution right-hand sides of
//!   `partial_t gamma_ij` and `partial_t K_ij`, evaluated from independently
//!   supplied spatial data (not extracted from an already-known 4-metric, which
//!   is what [`adm`] does), reusing [`ricci_tensor_from_metric`] and
//!   [`numerical_christoffel`] at `D = 3` and validated on Minkowski, a static
//!   Schwarzschild slice, flat FLRW, and deliberate constraint violations;
//! - homogeneous ADM time evolution ([`adm_homogeneous`]) — Layer 3.2: free
//!   evolution of spatially homogeneous ADM data with a barotropic perfect
//!   fluid, integrating the Layer 3.1 right-hand sides through the existing
//!   [`scirust_sim::simulate`] RK4 (no new integrator), with the Hamiltonian
//!   constraint monitored as a per-sample diagnostic and validated against the
//!   exact de Sitter, dust, and radiation solutions. Restricted to the
//!   homogeneous sector, where ADM's weak hyperbolicity cannot arise;
//! - the BSSN formulation core ([`bssn`]) — Layer 3.3: the conformal-traceless
//!   variable transformation and its inverse, the algebraic constraints,
//!   explicit projections, the conformal Ricci decomposition (cross-checked
//!   against [`ricci_tensor_from_metric`]), and the local evolution right-hand
//!   sides, validated against the Layer 3.1 ADM system. Implementing BSSN does
//!   **not** by itself demonstrate numerical stability;
//! - BSSN on a periodic grid ([`grid`], [`bssn_grid`]) —
//!   Layer 3.4: a half-open uniform periodic grid, centred finite-difference
//!   operators validated against `sin(kx)`, a grid-backed derivative provider
//!   that drives the *unmodified* Layer 3.3 evaluators, and a method-of-lines
//!   adapter integrating through [`scirust_sim::simulate`] (no new integrator).
//!   Validated in one, two, and three spatial dimensions (Layers 3.4 to 3.7),
//!   the last against a transverse-traceless gravitational wave propagating
//!   along the body diagonal. It remains a **periodic-box solver for weak,
//!   smooth fields** — no outer boundary, puncture, horizon, excision,
//!   constraint damping, or adaptive refinement. The pipeline is
//!   second-order accurate (Minkowski exactly stationary; conformal Ricci and
//!   wave propagation both converging at order 2; RK4 retaining order 4) and
//!   stable across every resolution tested, with a resolution-independent
//!   Courant boundary. Stability required writing `Rtilde_ij` in **genuine BSSN
//!   form** — carrying the evolved `Gammatilde^k`, which removes the mixed
//!   second derivatives from the principal part — and supplying the
//!   `d_t Gammatilde^i` equation Layer 3.3 deferred. Using the generic metric
//!   Ricci instead leaves ADM's weakly hyperbolic principal part wearing BSSN
//!   variables, and is measurably unstable. Carries the platform's first **live
//!   gauge**: 1+log slicing (`d_t alpha = -2 alpha K`, validated against the
//!   exact `sqrt(2)` gauge speed) and the Gamma-driver shift (validated by an
//!   exact constant-shift advection identity and a closed-form decay), together
//!   the moving-puncture gauge. Prescribed slicing and shift remain the
//!   defaults. Only weak-field tested, and stability here is **measured, not
//!   proven**.
//!
//! The crate does not assume that fractional calculus modifies general
//! relativity. Such models, if added later, must be exposed explicitly as
//! experimental constitutive or non-local extensions.
//!
//! ## Example
//!
//! ```
//! use scirust_relativity::{GeodesicSystem, Minkowski};
//! use scirust_sim::simulate;
//!
//! let system = GeodesicSystem::<_, 4>::new(Minkowski);
//! let initial = [0.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0];
//!
//! let trajectory = simulate(&system, &initial, 0.0, 2.0, 0.01)
//!     .expect("Minkowski geodesic integrates");
//!
//! let final_state = trajectory.last_state().expect("non-empty trajectory");
//! assert!((final_state[1] - 0.5).abs() < 1.0e-12);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod action;
pub mod adm;
pub mod adm_evolution;
pub mod adm_homogeneous;
pub mod bssn;
pub mod bssn_grid;
mod connection;
mod covariant_transport;
mod curvature;
mod de_sitter;
mod error;
mod exponential_map;
mod flrw;
mod geodesic;
mod geodesic_deviation;
pub mod grid;
mod isotropic_schwarzschild;
mod kerr;
mod linearized;
mod metric;
mod minkowski;
mod minkowski_spherical;
mod painleve_gullstrand;
mod parallel_transport;
pub mod ppn;
mod reissner_nordstrom;
mod schwarzschild;
mod static_spherical;
mod synge;
mod tetrad;

pub use connection::{Connection, christoffel_from_derivatives, numerical_christoffel};
pub use covariant_transport::{
    transport_covariant_tensor_along_polyline, transport_covariant_tensor_along_segment,
    transport_covector_along_polyline, transport_covector_along_segment,
};
pub use curvature::{CurvatureTensors, ricci_scalar_from_metric, ricci_tensor_from_metric};
pub use de_sitter::{AntiDeSitter, DeSitter};
pub use error::RelativityError;
pub use exponential_map::{geodesic_exponential, geodesic_logarithm};
pub use flrw::{ExponentialScaleFactor, Flrw, PowerLawScaleFactor, ScaleFactor};
pub use geodesic::GeodesicSystem;
pub use geodesic_deviation::{JacobiSample, integrate_geodesic_deviation};
pub use isotropic_schwarzschild::IsotropicSchwarzschild;
pub use kerr::Kerr;
pub use linearized::LinearizedField;
pub use metric::{Metric, determinant, invert_metric, metric_norm};
pub use minkowski::Minkowski;
pub use minkowski_spherical::MinkowskiSpherical;
pub use painleve_gullstrand::PainleveGullstrand;
pub use parallel_transport::{holonomy_defect, transport_along_polyline, transport_along_segment};
pub use reissner_nordstrom::ReissnerNordstrom;
pub use schwarzschild::Schwarzschild;
pub use synge::{
    WorldFunction, WorldFunctionSettings, van_vleck_determinant, world_function,
    world_function_with_gradients,
};
pub use tetrad::{OrthonormalTetrad, orthonormal_tetrad};
