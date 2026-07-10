//! Flat-plate boundary layers (zero pressure gradient).
//!
//! Laminar results are the exact Blasius similarity solution; turbulent
//! results are the classical 1/7-power-law correlations (smooth plate,
//! `5×10⁵ ≲ Re ≲ 10⁷`).

use crate::error::{FluidsError, positive};

/// Reynolds number commonly taken as the laminar→turbulent transition on
/// a smooth flat plate.
pub const RE_TRANSITION: f64 = 5.0e5;

/// Laminar (Blasius) boundary-layer thickness `δ = 5.0 x / √Re_x` \[m\].
pub fn blasius_thickness(x: f64, re_x: f64) -> Result<f64, FluidsError> {
    positive("x", x)?;
    positive("re_x", re_x)?;
    Ok(5.0 * x / re_x.sqrt())
}

/// Laminar displacement thickness `δ* = 1.7208 x / √Re_x` \[m\].
pub fn blasius_displacement_thickness(x: f64, re_x: f64) -> Result<f64, FluidsError> {
    positive("x", x)?;
    positive("re_x", re_x)?;
    Ok(1.7208 * x / re_x.sqrt())
}

/// Laminar momentum thickness `θ = 0.664 x / √Re_x` \[m\].
pub fn blasius_momentum_thickness(x: f64, re_x: f64) -> Result<f64, FluidsError> {
    positive("x", x)?;
    positive("re_x", re_x)?;
    Ok(0.664 * x / re_x.sqrt())
}

/// Laminar local skin-friction coefficient `c_f = 0.664 / √Re_x`.
pub fn blasius_cf_local(re_x: f64) -> Result<f64, FluidsError> {
    positive("re_x", re_x)?;
    Ok(0.664 / re_x.sqrt())
}

/// Laminar mean (drag) skin-friction coefficient over a plate of length L,
/// `C_f = 1.328 / √Re_L`.
pub fn blasius_cf_mean(re_l: f64) -> Result<f64, FluidsError> {
    positive("re_l", re_l)?;
    Ok(1.328 / re_l.sqrt())
}

/// Turbulent boundary-layer thickness `δ = 0.37 x / Re_x^{1/5}` \[m\]
/// (1/7-power law, smooth plate).
pub fn turbulent_thickness(x: f64, re_x: f64) -> Result<f64, FluidsError> {
    positive("x", x)?;
    positive("re_x", re_x)?;
    Ok(0.37 * x / re_x.powf(0.2))
}

/// Turbulent local skin-friction coefficient `c_f = 0.0592 / Re_x^{1/5}`.
pub fn turbulent_cf_local(re_x: f64) -> Result<f64, FluidsError> {
    positive("re_x", re_x)?;
    Ok(0.0592 / re_x.powf(0.2))
}

/// Turbulent mean skin-friction coefficient `C_f = 0.074 / Re_L^{1/5}`
/// (turbulent from the leading edge).
pub fn turbulent_cf_mean(re_l: f64) -> Result<f64, FluidsError> {
    positive("re_l", re_l)?;
    Ok(0.074 / re_l.powf(0.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blasius_at_re_1e6() {
        // δ/x = 5e-3, c_f = 6.64e-4, C_f = 1.328e-3 at Re = 1e6.
        assert!((blasius_thickness(1.0, 1.0e6).unwrap() - 5.0e-3).abs() < 1e-12);
        assert!((blasius_cf_local(1.0e6).unwrap() - 6.64e-4).abs() < 1e-12);
        assert!((blasius_cf_mean(1.0e6).unwrap() - 1.328e-3).abs() < 1e-12);
    }

    #[test]
    fn mean_is_twice_local_for_laminar() {
        // C_f(L) = 2 c_f(L) is exact for the Blasius √x law.
        let re = 3.7e5;
        let mean = blasius_cf_mean(re).unwrap();
        let local = blasius_cf_local(re).unwrap();
        assert!((mean - 2.0 * local).abs() < 1e-15);
    }

    #[test]
    fn shape_factor_is_blasius_value() {
        // H = δ*/θ = 1.7208/0.664 ≈ 2.59 for Blasius.
        let h = blasius_displacement_thickness(1.0, 1e6).unwrap()
            / blasius_momentum_thickness(1.0, 1e6).unwrap();
        assert!((h - 2.5916).abs() < 1e-3, "H = {h}");
    }

    #[test]
    fn turbulent_layer_thicker_than_laminar() {
        // At Re_x = 1e6 the turbulent layer is much thicker: 0.37/10^{6/5}
        // ≈ 0.0233 x vs 0.005 x.
        let dt = turbulent_thickness(1.0, 1.0e6).unwrap();
        let dl = blasius_thickness(1.0, 1.0e6).unwrap();
        assert!((dt - 0.0233).abs() < 2e-4, "δ_t = {dt}");
        assert!(dt > 4.0 * dl);
    }

    #[test]
    fn rejects_non_positive() {
        assert!(blasius_cf_local(0.0).is_err());
        assert!(turbulent_thickness(-1.0, 1e6).is_err());
    }
}
