//! Kerr spacetime in standard Boyer-Lindquist exterior coordinates.

use crate::{Connection, Metric, numerical_christoffel};
use std::f64::consts::PI;

const CHRISTOFFEL_DIFFERENCE_STEP: f64 = 1.0e-6;

/// Kerr spacetime in geometric units with signature `(-,+,+,+)`: a
/// stationary, axisymmetric, rotating black hole.
///
/// Coordinates are ordered as `(t, r, theta, phi)`, standard Boyer-Lindquist
/// coordinates. The mass and spin parameters use geometric units, so
/// `G = c = 1`; `a = J / M` is the spin per unit mass, with dimension of
/// length like `M`.
///
/// Unlike [`crate::Schwarzschild`] and [`crate::ReissnerNordstrom`], this
/// background's connection is evaluated by **central finite differences**
/// ([`numerical_christoffel`]), not an exact analytic formula. The Kerr
/// Christoffel symbols are algebraically far more complex than either of
/// those (the metric depends on both `r` and `theta`, and has a nonzero
/// off-diagonal `t`-`phi` term, so many more components are nonzero and mix
/// all four coordinates); hand-deriving them carries a real risk of a
/// transcription error with no independent way to catch it in this
/// codebase. `numerical_christoffel` is itself already validated elsewhere
/// in this crate against every background with an exact analytic
/// connection; using it here trades exact analytic Christoffels for a
/// small, documented finite-difference truncation error — an explicit,
/// honestly disclosed engineering choice, not an oversight.
///
/// At `a = 0` this is exactly [`crate::Schwarzschild`]: the metric reduces
/// algebraically to the Schwarzschild metric (checked bit-for-bit in this
/// crate's test suite), and the finite-difference Christoffel symbols agree
/// with Schwarzschild's exact analytic ones to the finite-difference
/// tolerance.
///
/// The standard Kerr exterior chart is regular only for:
///
/// - `r` greater than the outer horizon radius `M + sqrt(M^2 - a^2)`;
/// - `0 < theta < pi`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kerr {
    mass: f64,
    spin: f64,
}

impl Kerr {
    /// Construct a Kerr spacetime from a positive finite mass and a finite
    /// spin satisfying the sub-extremal bound `spin^2 < mass^2`, which
    /// guarantees two distinct, real horizons.
    #[must_use]
    pub fn try_new(mass: f64, spin: f64) -> Option<Self> {
        if mass.is_finite() && mass > 0.0 && spin.is_finite() && spin * spin < mass * mass
        {
            Some(Self { mass, spin })
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

    /// Return the geometric spin parameter `a = J / M`.
    #[must_use]
    pub const fn spin(&self) -> f64 {
        self.spin
    }

    /// Return the outer horizon radius `M + sqrt(M^2 - a^2)`.
    #[must_use]
    pub fn outer_horizon_radius(&self) -> f64 {
        self.mass + (self.mass * self.mass - self.spin * self.spin).sqrt()
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

    /// Return `Sigma = r^2 + a^2 cos^2(theta)`.
    fn sigma(&self, radius: f64, polar_angle: f64) -> f64 {
        let cosine = polar_angle.cos();
        radius * radius + self.spin * self.spin * cosine * cosine
    }

    /// Return `Delta = r^2 - 2 M r + a^2`.
    fn delta(&self, radius: f64) -> f64 {
        radius * radius - 2.0 * self.mass * radius + self.spin * self.spin
    }
}

impl Metric<4> for Kerr {
    fn components(&self, coordinates: &[f64; 4]) -> [[f64; 4]; 4] {
        let radius = coordinates[1];
        let polar_angle = coordinates[2];
        let sine = polar_angle.sin();
        let sine_squared = sine * sine;
        let sigma = self.sigma(radius, polar_angle);
        let delta = self.delta(radius);
        let mass = self.mass;
        let spin = self.spin;

        let time_time = -(1.0 - 2.0 * mass * radius / sigma);
        let time_phi = -2.0 * mass * spin * radius * sine_squared / sigma;
        let radial_radial = sigma / delta;
        let polar_polar = sigma;
        let azimuthal_azimuthal = (radius * radius
            + spin * spin
            + 2.0 * mass * spin * spin * radius * sine_squared / sigma)
            * sine_squared;

        [
            [time_time, 0.0, 0.0, time_phi],
            [0.0, radial_radial, 0.0, 0.0],
            [0.0, 0.0, polar_polar, 0.0],
            [time_phi, 0.0, 0.0, azimuthal_azimuthal],
        ]
    }
}

impl Connection<4> for Kerr {
    fn christoffel(&self, coordinates: &[f64; 4]) -> [[[f64; 4]; 4]; 4] {
        numerical_christoffel(self, coordinates, CHRISTOFFEL_DIFFERENCE_STEP)
            .unwrap_or([[[f64::NAN; 4]; 4]; 4])
    }
}
