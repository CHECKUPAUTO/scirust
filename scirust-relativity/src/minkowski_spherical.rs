//! Flat Minkowski spacetime in spherical spatial coordinates.
//!
//! This is the *same* flat geometry as [`crate::Minkowski`], written in the
//! chart `(t, r, theta, phi)` with metric `diag(-1, 1, r^2, r^2 sin^2 theta)`
//! (signature `(-,+,+,+)`). Unlike the Cartesian chart, its Christoffel
//! symbols are **non-zero** (e.g. `Gamma^r_(theta theta) = -r`,
//! `Gamma^theta_(r theta) = 1/r`), yet the Riemann tensor — and therefore
//! every curvature invariant — is still exactly zero. It is the lapse metric
//! of [`crate::static_spherical`] with `f(r) = 1`, so it shares that exact
//! analytic connection.
//!
//! It serves two purposes for the geometry core: a *strong* flatness test for
//! the curvature engine (the Cartesian chart has vanishing Christoffels and so
//! cannot exercise the connection-quadratic terms of the Riemann tensor, while
//! this chart can), and one half of a coordinate-independence check — the
//! scalar invariants computed here must match those of the Cartesian chart
//! (both zero) despite the very different Christoffel symbols.

use crate::static_spherical::{lapse_christoffel, lapse_metric};
use crate::{Connection, Metric};
use std::f64::consts::PI;

/// Four-dimensional flat Minkowski spacetime in spherical spatial coordinates
/// `(t, r, theta, phi)`, signature `(-,+,+,+)`, metric
/// `diag(-1, 1, r^2, r^2 sin^2 theta)`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MinkowskiSpherical;

impl MinkowskiSpherical {
    /// Determine whether coordinates lie in the regular spherical chart
    /// (`r > 0`, `0 < theta < pi`); the origin and the polar axis are chart
    /// singularities where `1/r` and `cot theta` diverge.
    #[must_use]
    pub fn is_in_regular_chart(&self, coordinates: &[f64; 4]) -> bool {
        if coordinates.iter().any(|coordinate| !coordinate.is_finite())
        {
            return false;
        }

        coordinates[1] > 0.0 && coordinates[2] > 0.0 && coordinates[2] < PI
    }
}

impl Metric<4> for MinkowskiSpherical {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        lapse_metric(1.0, coordinates[1], coordinates[2])
    }
}

impl Connection<4> for MinkowskiSpherical {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        // Lapse f = 1, f' = 0: the exact flat-space spherical Christoffels.
        lapse_christoffel(1.0, 0.0, coordinates[1], coordinates[2])
    }
}
