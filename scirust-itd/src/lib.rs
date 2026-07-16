//! # `scirust-itd` — deterministic 2-D field-simulation core
//!
//! SciRust's [`scirust-sim`](../scirust_sim/index.html) crate covers ODE-style
//! plants (`y' = f(t, y)`) and gym-style environments, and the domain crates
//! ship physics *formulas*; what the platform lacked was a **spatial-field /
//! PDE-flavoured** simulation kernel. This crate fills that gap by porting the
//! numerical core of the *ITD* research simulator to pure Rust.
//!
//! Everything here is accepted only after matching the reference
//! implementation as an oracle — SciRust's standing acceptance discipline. The
//! Rust results are checked against values produced by the original engine to a
//! tight numerical tolerance (see `tests/oracle.rs`), and the analytic
//! sanity checks that the reference asserts (an irrotational field has zero
//! rotational intensity; a rigid rotation has constant vorticity) hold here too.
//!
//! ## What it provides
//!
//! 1. **Field operators** ([`operators`]) on a 2-D scalar field sampled on a
//!    uniform or non-uniform rectilinear grid, with *finite* (one-sided
//!    second-order edges, trapezoidal quadrature) or *periodic* (circular
//!    central differences) boundary conventions:
//!    - [`operators::gradient`] — second-order finite-difference gradient,
//!      reproducing NumPy's `gradient(..., edge_order=2)` for both uniform and
//!      non-uniform axes;
//!    - [`operators::vorticity`] — the 2-D curl `ω = ∂v_y/∂x − ∂v_x/∂y`;
//!    - [`operators::spatial_mean`] — the domain integral divided by the domain
//!      area (2-D trapezoidal quadrature on non-uniform grids);
//!    - [`operators::bounded`] — the saturating map `b(x) = x / (1 + x)`.
//! 2. **The structural signature** ([`signature`]) — five deterministic
//!    descriptors of a vorticity field (heterogeneity, localization, roughness,
//!    sign-mixing, temporal deformation) plus a normalized-weighted scalar
//!    score.
//! 3. **The simulation driver** ([`simulate`]) — steps a time-dependent
//!    velocity field, accumulating the **curvature-weighted rotational
//!    intensity** `⟨ω² · e^{L²κ}⟩` and the structural signature into
//!    interval-integrated indices (intensity / structure / coupled / the five
//!    component indices).
//! 4. **Canonical scenarios** ([`scenarios`]) — the calm (irrotational),
//!    coherent-vortex (rigid rotation) and multi-vortex velocity fields, the
//!    shared curvature weighting, and the reference [`scenarios::Config`], used
//!    both as examples and as validation fixtures.
//!
//! ## Provenance
//!
//! This is a faithful numerical port, not a physical claim. As the reference
//! project states, its tests establish internal numerical and software
//! consistency; they do not establish the intensity index as a validated
//! physical observable. What transfers to SciRust is the deterministic,
//! oracle-validated machinery.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// These are finite-difference / quadrature grid kernels: explicit `(i, j)`
// indexing mirrors the reference implementation and reads more clearly than
// zipped iterators when two coordinate axes and several fields are combined.
#![allow(clippy::needless_range_loop)]

mod error;
mod field;
pub mod geometry;
pub mod operators;
pub mod scenarios;
pub mod signature;
pub mod simulate;

pub use error::{ItdError, Result};
pub use field::Field2;
pub use geometry::{BoundaryMode, Geometry};
pub use scenarios::{Config, Scenario};
pub use signature::{StructuralMetrics, StructuralWeights, structural_metrics};
pub use simulate::{COMPONENT_NAMES, SimConfig, SimulationResult, simulate, simulate_canonical};
