//! Schwarzschild spacetime in standard exterior coordinates.

use crate::{Connection, Metric};
use std::f64::consts::PI;

/// Schwarzschild spacetime in geometric units with signature `(-,+,+,+)`.
///
/// Coordinates are ordered as `(t, r, theta, phi)`. The mass parameter and
/// coordinates use geometric units, so `G = c = 1`.
///
/// The standard Schwarzschild chart is regular only in the exterior domain:
///
/// - `r > 2 M`;
/// - `0 < theta < pi`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schwarzschild {
    mass: f64,
}

impl Schwarzschild {
    /// Construct a Schwarzschild spacetime from a positive finite mass.
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

    /// Return the Schwarzschild horizon radius `2 M`.
    #[must_use]
    pub const fn horizon_radius(&self) -> f64 {
        2.0 * self.mass
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

        radius > self.horizon_radius() && polar_angle > 0.0 && polar_angle < PI
    }

    fn lapse(&self, radius: f64) -> f64 {
        1.0 - self.horizon_radius() / radius
    }
}

impl Metric<4> for Schwarzschild {
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

impl Connection<4> for Schwarzschild {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        let radius = coordinates[1];
        let polar_angle = coordinates[2];
        let mass = self.mass;
        let radius_minus_horizon = radius - self.horizon_radius();
        let radial_common = mass / (radius * radius_minus_horizon);
        let inverse_radius = 1.0 / radius;
        let sine = polar_angle.sin();
        let cosine = polar_angle.cos();
        let sine_squared = sine * sine;

        let mut symbols = [[[0.0_f64; 4]; 4]; 4];

        symbols[0][0][1] = radial_common;
        symbols[0][1][0] = radial_common;

        symbols[1][0][0] = mass * radius_minus_horizon / (radius * radius * radius);
        symbols[1][1][1] = -radial_common;
        symbols[1][2][2] = -radius_minus_horizon;
        symbols[1][3][3] = -radius_minus_horizon * sine_squared;

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
