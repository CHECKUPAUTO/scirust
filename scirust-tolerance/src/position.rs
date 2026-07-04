//! Positional tolerancing (GD&T / ISO GPS position) and its inertial form.
//!
//! A position tolerance controls where a feature axis lies relative to true
//! position with a **diametral** zone `Ø`. The verifiable, numeric core of
//! ASME Y14.5 / ISO 1101 position:
//!
//! - [`true_position`] — the diametral positional deviation `2·√(Δx² + Δy²)`.
//! - [`mmc_bonus`] / [`total_position_tolerance`] — the bonus tolerance a
//!   feature earns as its size departs from the maximum-material condition.
//! - [`coord_to_position`] / [`position_to_coord`] — conversion between a
//!   `±` coordinate zone and the equivalent diametral position zone.
//! - [`positional_inertia`] — the **inertial** view: because the expected
//!   squared radial deviation is `E[Δx² + Δy²] = Iₓ² + I_y²`, the positional
//!   inertia is `√(Iₓ² + I_y²)` (the [`crate::inertia::vector_inertia`] of the
//!   two axes), tying position tolerancing into the inertial framework.
//!
//! Full GD&T (feature control frames, datum precedence, envelope/independency)
//! is a rules language beyond a numeric crate; this covers the computable part.

use crate::inertia::Inertia;
use serde::{Deserialize, Serialize};

/// Diametral positional deviation `2·√(Δx² + Δy²)` of a feature whose axis is
/// offset by `(dx, dy)` from true position — directly comparable to a `Ø`
/// position tolerance.
pub fn true_position(dx: f64, dy: f64) -> f64 {
    2.0 * (dx * dx + dy * dy).sqrt()
}

/// Whether an axis offset `(dx, dy)` lies within a diametral position
/// tolerance `diametral_tol` (`2·√(Δx²+Δy²) ≤ Ø`).
pub fn conforms(dx: f64, dy: f64, diametral_tol: f64) -> bool {
    true_position(dx, dy) <= diametral_tol
}

/// Whether a feature is internal (a hole — MMC is the smallest size) or
/// external (a pin/boss — MMC is the largest size).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    /// Internal feature (hole): maximum-material condition is the *smallest*
    /// size.
    Internal,
    /// External feature (shaft/pin): maximum-material condition is the
    /// *largest* size.
    External,
}

/// Bonus tolerance earned under a maximum-material-condition (MMC) modifier:
/// the amount the actual size has departed from MMC toward least material.
///
/// For an [`FeatureType::Internal`] feature bonus `= actual − mmc_size`
/// (a larger hole earns bonus); for [`FeatureType::External`] bonus
/// `= mmc_size − actual` (a smaller pin earns bonus). Clamped at 0 — a feature
/// at or beyond MMC earns none (a value beyond MMC is a size violation handled
/// separately).
pub fn mmc_bonus(actual_size: f64, mmc_size: f64, feature: FeatureType) -> f64 {
    let departure = match feature
    {
        FeatureType::Internal => actual_size - mmc_size,
        FeatureType::External => mmc_size - actual_size,
    };
    departure.max(0.0)
}

/// Total available position tolerance at MMC: the stated `Ø` plus the
/// [`mmc_bonus`]. The feature conforms when its [`true_position`] is within
/// this total.
pub fn total_position_tolerance(
    stated: f64,
    actual_size: f64,
    mmc_size: f64,
    feature: FeatureType,
) -> f64 {
    stated + mmc_bonus(actual_size, mmc_size, feature)
}

/// Convert a `±` coordinate tolerance zone (`±tx` on X, `±ty` on Y) to the
/// equivalent diametral position tolerance that *contains* it:
/// `Ø = 2·√(tx² + ty²)` (the circle circumscribing the rectangular zone). For a
/// symmetric `±t` on both axes this is the familiar `2·√2·t`.
pub fn coord_to_position(tx: f64, ty: f64) -> f64 {
    2.0 * (tx * tx + ty * ty).sqrt()
}

/// Convert a diametral position tolerance `Ø` to the equivalent *symmetric*
/// per-axis `±` coordinate tolerance whose square zone is inscribed in it:
/// `±t` with `t = Ø / (2·√2)`. Inverse of [`coord_to_position`] on a symmetric
/// zone.
pub fn position_to_coord(diametral_tol: f64) -> f64 {
    diametral_tol / (2.0 * std::f64::consts::SQRT_2)
}

/// Positional inertia `√(Iₓ² + I_y²)` from the per-axis inertias about true
/// position — the root of the expected squared radial deviation
/// `E[Δx² + Δy²] = Iₓ² + I_y²`, and exactly the
/// [`crate::inertia::vector_inertia`] of the two axes.
pub fn positional_inertia(ix: f64, iy: f64) -> f64 {
    (ix * ix + iy * iy).sqrt()
}

/// Positional inertia from paired coordinate-deviation samples `(Δx, Δy)`
/// measured about true position (the target for each axis is 0). Estimates each
/// axis's inertia and combines them radially. Returns 0 for empty input.
pub fn positional_inertia_from_samples(dx: &[f64], dy: &[f64]) -> f64 {
    if dx.is_empty() || dy.is_empty()
    {
        return 0.0;
    }
    let ix = Inertia::from_sample(dx, 0.0).value();
    let iy = Inertia::from_sample(dy, 0.0).value();
    positional_inertia(ix, iy)
}

/// The `Cp = 1` maximum positional inertia for a diametral position tolerance
/// `diametral_tol`: `I_pos,max = Ø / (6·target_cp)` — the radial generalisation
/// of `I_max = IT/6`, with the diametral tolerance playing the role of the
/// tolerance interval. `target_cp = 2` gives the `Ø/12` "6σ" budget.
pub fn i_max_position(diametral_tol: f64, target_cp: f64) -> f64 {
    if target_cp <= 0.0
    {
        return f64::INFINITY;
    }
    diametral_tol / (6.0 * target_cp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inertia::vector_inertia;
    use approx::assert_relative_eq;

    #[test]
    fn true_position_is_diametral() {
        // (3, 4) offset ⇒ radius 5 ⇒ Ø 10.
        assert_relative_eq!(true_position(3.0, 4.0), 10.0, epsilon = 1e-12);
        assert!(conforms(3.0, 4.0, 10.0));
        assert!(!conforms(3.0, 4.0, 9.999));
    }

    #[test]
    fn coord_position_conversions_round_trip() {
        // Symmetric ±0.1 ⇒ Ø = 2√2·0.1 ≈ 0.2828.
        let phi = coord_to_position(0.1, 0.1);
        assert_relative_eq!(phi, 2.0 * std::f64::consts::SQRT_2 * 0.1, epsilon = 1e-12);
        assert_relative_eq!(position_to_coord(phi), 0.1, epsilon = 1e-12);
        // Rectangular zone: Ø = 2√(0.05²+0.12²) = 2·0.13 = 0.26.
        assert_relative_eq!(coord_to_position(0.05, 0.12), 0.26, epsilon = 1e-12);
    }

    #[test]
    fn mmc_bonus_and_total() {
        // Hole: MMC 10.0, actual 10.2 ⇒ bonus 0.2; total 0.1 + 0.2 = 0.3.
        assert_relative_eq!(
            mmc_bonus(10.2, 10.0, FeatureType::Internal),
            0.2,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            total_position_tolerance(0.1, 10.2, 10.0, FeatureType::Internal),
            0.3,
            epsilon = 1e-12
        );
        // Pin: MMC 5.0 (largest), actual 4.9 ⇒ bonus 0.1.
        assert_relative_eq!(
            mmc_bonus(4.9, 5.0, FeatureType::External),
            0.1,
            epsilon = 1e-12
        );
        // At/beyond MMC ⇒ no bonus.
        assert_relative_eq!(
            mmc_bonus(10.0, 10.0, FeatureType::Internal),
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            mmc_bonus(9.8, 10.0, FeatureType::Internal),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn positional_inertia_matches_vector_inertia() {
        let ix = 3.0;
        let iy = 4.0;
        assert_relative_eq!(positional_inertia(ix, iy), 5.0, epsilon = 1e-12);
        let v = vector_inertia(&[Inertia::new(ix, 0.0), Inertia::new(iy, 0.0)]);
        assert_relative_eq!(positional_inertia(ix, iy), v, epsilon = 1e-12);
    }

    #[test]
    fn positional_inertia_from_samples_combines_axes() {
        // δx = 0.03, δy = 0.04, no spread ⇒ I_pos = 0.05.
        let dx = [0.03, 0.03, 0.03];
        let dy = [0.04, 0.04, 0.04];
        assert_relative_eq!(
            positional_inertia_from_samples(&dx, &dy),
            0.05,
            epsilon = 1e-12
        );
    }

    #[test]
    fn i_max_position_conventions() {
        assert_relative_eq!(i_max_position(0.6, 1.0), 0.1, epsilon = 1e-12);
        assert_relative_eq!(i_max_position(0.6, 2.0), 0.05, epsilon = 1e-12);
    }
}
