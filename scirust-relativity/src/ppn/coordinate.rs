//! The PPN-compatible coordinate contract.
//!
//! PPN coefficients are coordinate-sensitive, so the extractor never reads a raw
//! metric's radial coordinate as isotropic by assumption. Instead it operates on
//! [`StaticIsotropicMetric`], obtained either directly (the synthetic oracles) or
//! through [`IsotropicChartAdapter`], which validates conformal flatness of the
//! spatial metric at each radius and rejects non-isotropic charts.

use super::error::PpnError;
use crate::Metric;
use std::f64::consts::FRAC_PI_2;

/// A genuinely isotropic metric has an exactly conformally flat spatial block, so
/// its spatial conformal ratios agree to rounding. Areal-coordinate metrics
/// differ by `O(U)`, far above this floor, so this tolerance separates them.
const ISOTROPY_TOLERANCE: f64 = 1.0e-9;

/// A static, spherically symmetric metric in a PPN-compatible isotropic radial
/// coordinate `rho`: it exposes `g_00(rho)`, the spatial conformal factor
/// `A(rho)` (with `g_ij = A delta_ij`), and the mass scale `G M`.
///
/// Being a function of `rho` alone, an implementor is static and spherically
/// symmetric by construction.
pub trait StaticIsotropicMetric {
    /// The mass scale `G M` (finite, strictly positive).
    fn mass_scale(&self) -> f64;

    /// The time-time metric component `g_00(rho)`.
    fn g_tt(&self, radius: f64) -> Result<f64, PpnError>;

    /// The spatial conformal factor `A(rho)`, `g_ij = A delta_ij`.
    fn spatial_conformal_factor(&self, radius: f64) -> Result<f64, PpnError>;
}

/// Presents a spherical `Metric<4>` in `(t, r, theta, phi)` as a
/// [`StaticIsotropicMetric`], **checking conformal flatness** at each radius.
///
/// The spatial block of a genuine isotropic metric satisfies
/// `g_rr = g_(theta theta)/r^2 = g_(phi phi)/(r^2 sin^2 theta)`; the adapter
/// rejects any metric that violates this (for example areal-coordinate
/// Schwarzschild, whose `g_rr = 1/f` differs from `g_(theta theta)/r^2 = 1`).
/// This makes it impossible to silently apply the isotropic PPN formulas to a
/// non-isotropic chart.
pub struct IsotropicChartAdapter<'a, M> {
    metric: &'a M,
    mass_scale: f64,
}

impl<'a, M: Metric<4>> IsotropicChartAdapter<'a, M> {
    /// Wrap `metric` (interpreted in `(t, r, theta, phi)`) with mass scale
    /// `mass_scale`. Fails with [`PpnError::InvalidMassScale`] for a non-finite
    /// or non-positive mass.
    pub fn new(metric: &'a M, mass_scale: f64) -> Result<Self, PpnError> {
        if !mass_scale.is_finite() || mass_scale <= 0.0
        {
            return Err(PpnError::InvalidMassScale(mass_scale));
        }
        Ok(Self { metric, mass_scale })
    }

    /// Evaluate the metric on the equatorial ray `(0, radius, pi/2, 0)`,
    /// validating the radius and finiteness.
    fn equatorial_components(&self, radius: f64) -> Result<[[f64; 4]; 4], PpnError> {
        if !radius.is_finite() || radius <= 0.0
        {
            return Err(PpnError::InvalidMetricRadius(radius));
        }
        let components = self.metric.components(&[0.0, radius, FRAC_PI_2, 0.0]);
        if components.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(PpnError::NonFiniteMetricValue { radius });
        }
        Ok(components)
    }
}

impl<M: Metric<4>> StaticIsotropicMetric for IsotropicChartAdapter<'_, M> {
    fn mass_scale(&self) -> f64 {
        self.mass_scale
    }

    fn g_tt(&self, radius: f64) -> Result<f64, PpnError> {
        Ok(self.equatorial_components(radius)?[0][0])
    }

    fn spatial_conformal_factor(&self, radius: f64) -> Result<f64, PpnError> {
        let components = self.equatorial_components(radius)?;
        // On the equator sin(theta) = 1, so the three spatial conformal ratios
        // are g_rr, g_(theta theta)/r^2, g_(phi phi)/r^2 — equal iff isotropic.
        let radius_squared = radius * radius;
        let radial = components[1][1];
        let polar = components[2][2] / radius_squared;
        let azimuthal = components[3][3] / radius_squared;

        let scale = radial.abs().max(polar.abs()).max(azimuthal.abs()).max(1.0);
        let mismatch = (radial - polar).abs().max((radial - azimuthal).abs()) / scale;
        if mismatch > ISOTROPY_TOLERANCE
        {
            return Err(PpnError::NonIsotropicCoordinates {
                radius,
                relative_mismatch: mismatch,
            });
        }
        Ok(radial)
    }
}
