//! Synthetic isotropic PPN metrics with exactly controlled coefficients, for
//! validating the extractor against known answers.

use super::coordinate::StaticIsotropicMetric;
use super::error::PpnError;

/// A synthetic isotropic metric with controlled weak-field coefficients:
///
/// ```text
/// g_00(rho) = -1 + 2U - 2 beta_star U^2 + a3 U^3 + a4 U^4,
/// A(rho)    =  1 + 2 gamma_star U + b2 U^2 + b3 U^3,
/// U = mass / rho.
/// ```
///
/// With `a3 = a4 = b2 = b3 = 0` the effective estimators are *exactly* constant
/// (`gamma_eff == gamma_star`, `beta_eff == beta_star`), isolating extraction
/// correctness. Non-zero higher-order coefficients inject known contamination for
/// the convergence oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticPpnMetric {
    /// Mass scale `G M`.
    pub mass: f64,
    /// Injected `gamma`.
    pub gamma_star: f64,
    /// Injected `beta`.
    pub beta_star: f64,
    /// `U^3` coefficient of `g_00`.
    pub g_tt_cubic: f64,
    /// `U^4` coefficient of `g_00`.
    pub g_tt_quartic: f64,
    /// `U^2` coefficient of `A`.
    pub conformal_quadratic: f64,
    /// `U^3` coefficient of `A`.
    pub conformal_cubic: f64,
}

impl SyntheticPpnMetric {
    /// An exact synthetic metric (no contamination) with the given PPN pair.
    #[must_use]
    pub fn exact(mass: f64, gamma_star: f64, beta_star: f64) -> Self {
        Self {
            mass,
            gamma_star,
            beta_star,
            g_tt_cubic: 0.0,
            g_tt_quartic: 0.0,
            conformal_quadratic: 0.0,
            conformal_cubic: 0.0,
        }
    }

    /// A contaminated synthetic metric: the exact PPN pair plus the given
    /// higher-order `g_00` (`a3`, `a4`) and conformal (`b2`, `b3`) coefficients.
    #[must_use]
    pub fn contaminated(
        mass: f64,
        gamma_star: f64,
        beta_star: f64,
        g_tt_cubic: f64,
        g_tt_quartic: f64,
        conformal_quadratic: f64,
        conformal_cubic: f64,
    ) -> Self {
        Self {
            mass,
            gamma_star,
            beta_star,
            g_tt_cubic,
            g_tt_quartic,
            conformal_quadratic,
            conformal_cubic,
        }
    }

    fn compactness(&self, radius: f64) -> Result<f64, PpnError> {
        if !radius.is_finite() || radius <= 0.0
        {
            return Err(PpnError::InvalidMetricRadius(radius));
        }
        Ok(self.mass / radius)
    }
}

impl StaticIsotropicMetric for SyntheticPpnMetric {
    fn mass_scale(&self) -> f64 {
        self.mass
    }

    fn g_tt(&self, radius: f64) -> Result<f64, PpnError> {
        let u = self.compactness(radius)?;
        let value = -1.0 + 2.0 * u - 2.0 * self.beta_star * u * u
            + self.g_tt_cubic * u.powi(3)
            + self.g_tt_quartic * u.powi(4);
        if !value.is_finite()
        {
            return Err(PpnError::NonFiniteMetricValue { radius });
        }
        Ok(value)
    }

    fn spatial_conformal_factor(&self, radius: f64) -> Result<f64, PpnError> {
        let u = self.compactness(radius)?;
        let value = 1.0
            + 2.0 * self.gamma_star * u
            + self.conformal_quadratic * u * u
            + self.conformal_cubic * u.powi(3);
        if !value.is_finite()
        {
            return Err(PpnError::NonFiniteMetricValue { radius });
        }
        Ok(value)
    }
}
