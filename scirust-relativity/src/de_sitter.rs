//! De Sitter and Anti-de Sitter spacetimes in static coordinates.
//!
//! Both are 4D maximally symmetric vacuum solutions of Einstein's equations
//! *with* a cosmological constant, `G_(mu nu) + Lambda g_(mu nu) = 0`. In
//! static coordinates `(t, r, theta, phi)` they share the lapse form
//! `f(r) = 1 - Lambda r^2 / 3` (signature `(-,+,+,+)`, geometric units), so
//! their metric and connection reuse [`crate::static_spherical`]. Their exact
//! curvature is a fixed function of `Lambda` (see `scirust-relativity`'s
//! curvature tests), which makes them precise analytic oracles for the
//! numerical curvature engine.

use crate::static_spherical::{lapse_christoffel, lapse_metric};
use crate::{Connection, Metric};
use std::f64::consts::PI;

/// De Sitter spacetime (positive cosmological constant) in static coordinates.
///
/// `f(r) = 1 - Lambda r^2 / 3` with `Lambda > 0`. The static chart is regular
/// in `0 < r < r_h` where `r_h = sqrt(3 / Lambda)` is the cosmological horizon
/// (`f(r_h) = 0`), and `0 < theta < pi`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeSitter {
    cosmological_constant: f64,
}

impl DeSitter {
    /// Construct de Sitter spacetime from a strictly positive, finite
    /// cosmological constant `Lambda`.
    #[must_use]
    pub fn try_new(cosmological_constant: f64) -> Option<Self> {
        if cosmological_constant.is_finite() && cosmological_constant > 0.0
        {
            Some(Self {
                cosmological_constant,
            })
        }
        else
        {
            None
        }
    }

    /// Return the (positive) cosmological constant `Lambda`.
    #[must_use]
    pub const fn cosmological_constant(&self) -> f64 {
        self.cosmological_constant
    }

    /// Return the cosmological horizon radius `r_h = sqrt(3 / Lambda)`.
    #[must_use]
    pub fn horizon_radius(&self) -> f64 {
        (3.0 / self.cosmological_constant).sqrt()
    }

    /// Lapse `f(r) = 1 - Lambda r^2 / 3`.
    fn lapse(&self, radius: f64) -> f64 {
        1.0 - self.cosmological_constant * radius * radius / 3.0
    }

    /// Radial derivative `f'(r) = -2 Lambda r / 3`.
    fn lapse_derivative(&self, radius: f64) -> f64 {
        -2.0 * self.cosmological_constant * radius / 3.0
    }

    /// Determine whether coordinates lie in the regular static interior chart
    /// (`0 < r < r_h`, `0 < theta < pi`).
    #[must_use]
    pub fn is_in_static_patch(&self, coordinates: &[f64; 4]) -> bool {
        if coordinates.iter().any(|coordinate| !coordinate.is_finite())
        {
            return false;
        }

        let radius = coordinates[1];
        let polar_angle = coordinates[2];

        radius > 0.0 && radius < self.horizon_radius() && polar_angle > 0.0 && polar_angle < PI
    }
}

impl Metric<4> for DeSitter {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        lapse_metric(self.lapse(coordinates[1]), coordinates[1], coordinates[2])
    }
}

impl Connection<4> for DeSitter {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        lapse_christoffel(
            self.lapse(coordinates[1]),
            self.lapse_derivative(coordinates[1]),
            coordinates[1],
            coordinates[2],
        )
    }
}

/// Anti-de Sitter spacetime (negative cosmological constant) in static
/// coordinates.
///
/// `f(r) = 1 - Lambda r^2 / 3 = 1 + |Lambda| r^2 / 3` with `Lambda < 0`, so
/// `f(r) > 0` for all `r > 0`: there is no horizon and the static chart is
/// regular for every `r > 0`, `0 < theta < pi`. The constructor takes the
/// magnitude `|Lambda|`; [`AntiDeSitter::cosmological_constant`] returns the
/// signed (negative) value used by the curvature oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AntiDeSitter {
    cosmological_constant_magnitude: f64,
}

impl AntiDeSitter {
    /// Construct anti-de Sitter spacetime from a strictly positive, finite
    /// cosmological-constant magnitude `|Lambda|` (the signed constant is
    /// `Lambda = -|Lambda|`).
    #[must_use]
    pub fn try_new(cosmological_constant_magnitude: f64) -> Option<Self> {
        if cosmological_constant_magnitude.is_finite() && cosmological_constant_magnitude > 0.0
        {
            Some(Self {
                cosmological_constant_magnitude,
            })
        }
        else
        {
            None
        }
    }

    /// Return the signed cosmological constant `Lambda = -|Lambda| < 0`.
    #[must_use]
    pub fn cosmological_constant(&self) -> f64 {
        -self.cosmological_constant_magnitude
    }

    /// Lapse `f(r) = 1 + |Lambda| r^2 / 3`.
    fn lapse(&self, radius: f64) -> f64 {
        1.0 + self.cosmological_constant_magnitude * radius * radius / 3.0
    }

    /// Radial derivative `f'(r) = 2 |Lambda| r / 3`.
    fn lapse_derivative(&self, radius: f64) -> f64 {
        2.0 * self.cosmological_constant_magnitude * radius / 3.0
    }

    /// Determine whether coordinates lie in the regular static chart
    /// (`r > 0`, `0 < theta < pi`).
    #[must_use]
    pub fn is_in_static_patch(&self, coordinates: &[f64; 4]) -> bool {
        if coordinates.iter().any(|coordinate| !coordinate.is_finite())
        {
            return false;
        }

        coordinates[1] > 0.0 && coordinates[2] > 0.0 && coordinates[2] < PI
    }
}

impl Metric<4> for AntiDeSitter {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        lapse_metric(self.lapse(coordinates[1]), coordinates[1], coordinates[2])
    }
}

impl Connection<4> for AntiDeSitter {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        lapse_christoffel(
            self.lapse(coordinates[1]),
            self.lapse_derivative(coordinates[1]),
            coordinates[1],
            coordinates[2],
        )
    }
}
