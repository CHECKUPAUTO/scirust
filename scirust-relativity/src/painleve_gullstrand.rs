//! Painlevé–Gullstrand Schwarzschild — a horizon-penetrating foliation.
//!
//! The Schwarzschild geometry in Painlevé–Gullstrand coordinates `(t, r, theta,
//! phi)`, where `t` is the proper time of radially infalling observers released
//! from rest at infinity. In signature `(-,+,+,+)` and geometric units,
//!
//! ```text
//! ds^2 = -(1 - 2M/r) dt^2 + 2 sqrt(2M/r) dt dr + dr^2 + r^2 dOmega^2 .
//! ```
//!
//! Unlike the standard Schwarzschild chart, this one is **regular at the horizon**
//! `r = 2M` and its constant-`t` slices are **flat** (`dr^2 + r^2 dOmega^2`), with
//! unit lapse and a non-zero radial shift `sqrt(2M/r)`. It is the same vacuum
//! geometry (`R_{mu nu} = 0`), so it is a precise oracle for the ADM constraints
//! with a non-trivial (non-zero-shift, spatially varying extrinsic curvature)
//! foliation — see `docs/LAYER_2_ADM.md`.
//!
//! This background provides the [`Metric`] only (no analytic [`crate::Connection`]):
//! its role is as an ADM foliation example, and its curvature is already the
//! standard Schwarzschild curvature validated in the geometry-core tests.

use crate::Metric;

/// Schwarzschild spacetime in Painlevé–Gullstrand (horizon-penetrating)
/// coordinates, parameterized by the mass `M > 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PainleveGullstrand {
    mass: f64,
}

impl PainleveGullstrand {
    /// Construct from a strictly positive, finite mass `M`.
    #[must_use]
    pub fn try_new(mass: f64) -> Option<Self> {
        if mass.is_finite() && mass > 0.0
        {
            Some(Self { mass })
        }
        else
        {
            None
        }
    }

    /// Return the mass `M`.
    #[must_use]
    pub const fn mass(&self) -> f64 {
        self.mass
    }
}

impl Metric<4> for PainleveGullstrand {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        let radius = coordinates[1];
        let polar_angle = coordinates[2];
        let shift = (2.0 * self.mass / radius).sqrt();
        let sin_polar = polar_angle.sin();
        [
            [-(1.0 - 2.0 * self.mass / radius), shift, 0.0, 0.0],
            [shift, 1.0, 0.0, 0.0],
            [0.0, 0.0, radius * radius, 0.0],
            [0.0, 0.0, 0.0, radius * radius * sin_polar * sin_polar],
        ]
    }
}
