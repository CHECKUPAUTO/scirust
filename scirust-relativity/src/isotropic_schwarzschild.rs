//! Schwarzschild spacetime in isotropic coordinates.
//!
//! This is the *same* physical geometry as [`crate::Schwarzschild`], written in
//! a different radial chart. In isotropic coordinates `(t, rho, theta, phi)`
//! the metric is
//!
//! ```text
//! ds^2 = -A(rho)^2 dt^2 + B(rho)^4 (drho^2 + rho^2 dtheta^2 + rho^2 sin^2 theta dphi^2)
//! A(rho) = (1 - M / 2 rho) / (1 + M / 2 rho),   B(rho) = 1 + M / 2 rho
//! ```
//!
//! with signature `(-,+,+,+)` and geometric units. The isotropic radius `rho`
//! relates to the areal (Schwarzschild) radius `r` by
//! `r = rho (1 + M / 2 rho)^2`, so the exterior `rho > M / 2` maps to
//! `r > 2 M`, and the horizon `rho = M / 2` maps to `r = 2 M`.
//!
//! Because every curvature scalar is coordinate independent, the Kretschmann
//! invariant here must equal `48 M^2 / r^6` with `r` the *areal* radius
//! [`IsotropicSchwarzschild::areal_radius`] — not `48 M^2 / rho^6`. That
//! identity is the coordinate-independence oracle exercised by this crate's
//! tests.
//!
//! Like [`crate::Kerr`], and for the same reason, this background's connection
//! is evaluated by central finite differences ([`crate::numerical_christoffel`])
//! rather than hand-derived analytic Christoffel symbols: the isotropic metric
//! functions `A` and `B` make the analytic symbols lengthy and error-prone to
//! transcribe, whereas `numerical_christoffel` is already validated against
//! every analytic-connection background in this crate. The curvature computed
//! from it is therefore a nested finite difference with a correspondingly
//! larger, honestly disclosed truncation error than the analytic-connection
//! backgrounds.

use crate::{Connection, Metric, numerical_christoffel};
use std::f64::consts::PI;

const CHRISTOFFEL_DIFFERENCE_STEP: f64 = 1.0e-6;

/// Schwarzschild spacetime in isotropic coordinates `(t, rho, theta, phi)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsotropicSchwarzschild {
    mass: f64,
}

impl IsotropicSchwarzschild {
    /// Construct from a strictly positive, finite geometric mass `M`.
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

    /// Return the geometric mass parameter `M`.
    #[must_use]
    pub const fn mass(&self) -> f64 {
        self.mass
    }

    /// Return the isotropic horizon radius `rho_h = M / 2` (where `A = 0`).
    #[must_use]
    pub fn horizon_radius(&self) -> f64 {
        self.mass / 2.0
    }

    /// Return the areal (Schwarzschild) radius `r = rho (1 + M / 2 rho)^2`
    /// corresponding to isotropic radius `rho`.
    #[must_use]
    pub fn areal_radius(&self, isotropic_radius: f64) -> f64 {
        let factor = 1.0 + self.mass / (2.0 * isotropic_radius);
        isotropic_radius * factor * factor
    }

    /// Lapse factor `A(rho) = (1 - M / 2 rho) / (1 + M / 2 rho)`.
    fn lapse(&self, isotropic_radius: f64) -> f64 {
        let half_mass_over_rho = self.mass / (2.0 * isotropic_radius);
        (1.0 - half_mass_over_rho) / (1.0 + half_mass_over_rho)
    }

    /// Conformal factor `B(rho) = 1 + M / 2 rho`.
    fn conformal(&self, isotropic_radius: f64) -> f64 {
        1.0 + self.mass / (2.0 * isotropic_radius)
    }

    /// Determine whether coordinates lie in the regular exterior isotropic
    /// chart (`rho > M / 2`, `0 < theta < pi`).
    #[must_use]
    pub fn is_in_exterior(&self, coordinates: &[f64; 4]) -> bool {
        if coordinates.iter().any(|coordinate| !coordinate.is_finite())
        {
            return false;
        }

        coordinates[1] > self.horizon_radius() && coordinates[2] > 0.0 && coordinates[2] < PI
    }
}

impl Metric<4> for IsotropicSchwarzschild {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        let isotropic_radius = coordinates[1];
        let polar_angle = coordinates[2];
        let sine = polar_angle.sin();

        let lapse = self.lapse(isotropic_radius);
        let conformal = self.conformal(isotropic_radius);
        let conformal_fourth = conformal * conformal * conformal * conformal;
        let radius_squared = isotropic_radius * isotropic_radius;

        [
            [-lapse * lapse, 0.0, 0.0, 0.0],
            [0.0, conformal_fourth, 0.0, 0.0],
            [0.0, 0.0, conformal_fourth * radius_squared, 0.0],
            [
                0.0,
                0.0,
                0.0,
                conformal_fourth * radius_squared * sine * sine,
            ],
        ]
    }
}

impl Connection<4> for IsotropicSchwarzschild {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        numerical_christoffel(self, coordinates, CHRISTOFFEL_DIFFERENCE_STEP)
            .unwrap_or([[[f64::NAN; 4]; 4]; 4])
    }
}
