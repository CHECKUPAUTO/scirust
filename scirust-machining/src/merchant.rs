//! Usinage — **coupe orthogonale** (modèle de **Merchant**) : rapport de coupe,
//! angle de cisaillement, déformation de cisaillement.
//!
//! ```text
//! rapport de coupe    r = t/tc          (épaisseur avant/copeau, r < 1)
//! angle de cisaillement tan φ = r·cosα / (1 − r·sinα)
//! déformation         γ = cot φ + tan(φ − α)
//! relation de Merchant φ = 45° + α/2 − β/2   (β angle de frottement)
//! ```
//!
//! `t` épaisseur du copeau **non déformé** (avance), `tc` épaisseur du copeau
//! **formé**, `r` rapport de coupe, `α` angle de coupe (rake, rad), `φ` angle de
//! plan de cisaillement (rad), `β` angle de frottement copeau-outil, `γ`
//! déformation de cisaillement. Le copeau étant plus épais que la couche coupée,
//! `r < 1`.
//!
//! **Convention** : angles en rad. **Limite honnête** : coupe **orthogonale**
//! idéalisée (plan de cisaillement unique, régime continu) ; l'angle de coupe
//! `α` et le frottement `β` sont fournis par l'appelant. Ne modélise pas
//! l'arête rapportée ni la coupe oblique.

use core::f64::consts::FRAC_PI_4;

/// Rapport de coupe `r = t/tc` (< 1).
///
/// Panique si `chip_thickness <= 0`.
pub fn chip_thickness_ratio(uncut_thickness: f64, chip_thickness: f64) -> f64 {
    assert!(
        chip_thickness > 0.0,
        "l'épaisseur du copeau doit être strictement positive"
    );
    uncut_thickness / chip_thickness
}

/// Angle de cisaillement `tan φ = r·cosα/(1 − r·sinα)` → `φ` (rad).
pub fn shear_angle(cutting_ratio: f64, rake_angle: f64) -> f64 {
    let (s, c) = (rake_angle.sin(), rake_angle.cos());
    (cutting_ratio * c).atan2(1.0 - cutting_ratio * s)
}

/// Déformation de cisaillement `γ = cot φ + tan(φ − α)`.
///
/// Panique si `tan φ = 0`.
pub fn shear_strain(shear_angle: f64, rake_angle: f64) -> f64 {
    let t = shear_angle.tan();
    assert!(t != 0.0, "l'angle de cisaillement ne doit pas être nul");
    1.0 / t + (shear_angle - rake_angle).tan()
}

/// Angle de cisaillement par la **relation de Merchant** `φ = π/4 + α/2 − β/2`.
pub fn merchant_shear_angle(rake_angle: f64, friction_angle: f64) -> f64 {
    FRAC_PI_4 + rake_angle / 2.0 - friction_angle / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cutting_ratio_below_one() {
        // Copeau plus épais que la couche coupée → r < 1.
        let r = chip_thickness_ratio(0.1, 0.25);
        assert_relative_eq!(r, 0.4, epsilon = 1e-12);
        assert!(r < 1.0);
    }

    #[test]
    fn shear_angle_zero_rake() {
        // α=0 : tan φ = r → φ = atan(r).
        let r = 0.4;
        assert_relative_eq!(shear_angle(r, 0.0), r.atan(), epsilon = 1e-12);
    }

    #[test]
    fn thicker_chip_lowers_shear_angle() {
        // Un copeau plus épais (r plus petit) → angle de cisaillement plus faible.
        assert!(shear_angle(0.3, 0.0) < shear_angle(0.6, 0.0));
    }

    #[test]
    fn merchant_relation_value() {
        // α=10°, β=30° → φ = 45 + 5 − 15 = 35°.
        let phi = merchant_shear_angle(10.0_f64.to_radians(), 30.0_f64.to_radians());
        assert_relative_eq!(phi, 35.0_f64.to_radians(), epsilon = 1e-12);
    }

    #[test]
    fn shear_strain_is_large_at_small_angles() {
        // La déformation de cisaillement croît fortement quand φ diminue.
        assert!(shear_strain(0.2, 0.0) > shear_strain(0.6, 0.0));
    }

    #[test]
    #[should_panic(expected = "épaisseur du copeau")]
    fn zero_chip_thickness_panics() {
        chip_thickness_ratio(0.1, 0.0);
    }
}
