//! Reissner-Nordström spacetime in standard exterior coordinates.

use crate::{Connection, Metric};
use std::f64::consts::PI;

/// Reissner-Nordström spacetime in geometric units with signature
/// `(-,+,+,+)`: a static, spherically symmetric, electrically charged black
/// hole.
///
/// Coordinates are ordered as `(t, r, theta, phi)`, exactly as
/// [`crate::Schwarzschild`]. The mass and charge parameters use geometric
/// units, so `G = c = 1`, and this crate uses the convention in which charge
/// carries the same dimension as mass (the metric's charge term is
/// `Q^2 / r^2`, dimensionless like `2 M / r`).
///
/// At `charge = 0` this is exactly [`crate::Schwarzschild`]: both the metric
/// and the Christoffel symbols reduce to the identical formulas (this crate's
/// test suite checks this directly).
///
/// This background is a fixed, externally specified geometry, used here
/// exactly like [`crate::Schwarzschild`]: as given closed-form data, not as
/// the result of solving the Einstein field equations from a matter or field
/// source. Nothing in this crate computes the electromagnetic field's
/// backreaction on the metric; the metric formula is simply taken as known.
///
/// The standard Reissner-Nordström chart is regular only in the exterior
/// domain:
///
/// - `r` greater than the outer horizon radius;
/// - `0 < theta < pi`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReissnerNordstrom {
    mass: f64,
    charge: f64,
}

impl ReissnerNordstrom {
    /// Construct a Reissner-Nordström spacetime from a positive finite mass
    /// and a finite charge satisfying the sub-extremal bound `charge^2 <
    /// mass^2`, which guarantees two distinct, real horizons.
    #[must_use]
    pub fn try_new(mass: f64, charge: f64) -> Option<Self> {
        if mass.is_finite() && mass > 0.0 && charge.is_finite() && charge * charge < mass * mass
        {
            Some(Self { mass, charge })
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

    /// Return the geometric charge parameter `Q`.
    #[must_use]
    pub const fn charge(&self) -> f64 {
        self.charge
    }

    /// Return the outer horizon radius `M + sqrt(M^2 - Q^2)`.
    #[must_use]
    pub fn outer_horizon_radius(&self) -> f64 {
        self.mass + (self.mass * self.mass - self.charge * self.charge).sqrt()
    }

    /// Determine whether coordinates lie in the regular exterior chart.
    #[must_use]
    pub fn is_in_exterior(&self, coordinates: &[f64; 4]) -> bool {
        if coordinates.iter().any(|coordinate| !coordinate.is_finite())
        {
            return false;
        }

        let radius = coordinates[1];
        let polar_angle = coordinates[2];

        radius > self.outer_horizon_radius() && polar_angle > 0.0 && polar_angle < PI
    }

    fn lapse(&self, radius: f64) -> f64 {
        1.0 - 2.0 * self.mass / radius + (self.charge * self.charge) / (radius * radius)
    }

    fn lapse_derivative(&self, radius: f64) -> f64 {
        2.0 * self.mass / (radius * radius)
            - 2.0 * self.charge * self.charge / (radius * radius * radius)
    }
}

impl Metric<4> for ReissnerNordstrom {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        let radius = coordinates[1];
        let polar_angle = coordinates[2];
        let lapse = self.lapse(radius);
        let radius_squared = radius * radius;
        let sine = polar_angle.sin();

        [
            [-lapse, 0.0, 0.0, 0.0],
            [0.0, 1.0 / lapse, 0.0, 0.0],
            [0.0, 0.0, radius_squared, 0.0],
            [0.0, 0.0, 0.0, radius_squared * sine * sine],
        ]
    }
}

impl Connection<4> for ReissnerNordstrom {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        let radius = coordinates[1];
        let polar_angle = coordinates[2];
        let lapse = self.lapse(radius);
        let lapse_derivative = self.lapse_derivative(radius);
        let half_lapse_derivative_over_lapse = lapse_derivative / (2.0 * lapse);
        let inverse_radius = 1.0 / radius;
        let sine = polar_angle.sin();
        let cosine = polar_angle.cos();
        let sine_squared = sine * sine;

        let mut symbols = [[[0.0_f64; 4]; 4]; 4];

        symbols[0][0][1] = half_lapse_derivative_over_lapse;
        symbols[0][1][0] = half_lapse_derivative_over_lapse;

        symbols[1][0][0] = lapse * lapse_derivative / 2.0;
        symbols[1][1][1] = -half_lapse_derivative_over_lapse;
        symbols[1][2][2] = -radius * lapse;
        symbols[1][3][3] = -radius * lapse * sine_squared;

        symbols[2][1][2] = inverse_radius;
        symbols[2][2][1] = inverse_radius;
        symbols[2][3][3] = -sine * cosine;

        symbols[3][1][3] = inverse_radius;
        symbols[3][3][1] = inverse_radius;
        symbols[3][2][3] = cosine / sine;
        symbols[3][3][2] = cosine / sine;

        symbols
    }
}
