//! Fonderie — **solidification** : module thermique, règle de **Chvorinov** et
//! dimensionnement de masselotte.
//!
//! ```text
//! module thermique   M = V/A
//! Chvorinov          t = B·(V/A)ⁿ = B·Mⁿ    (n ≈ 2)
//! masselotte saine   M_masselotte ≥ k·M_pièce   (k ≈ 1,2)
//! ```
//!
//! `V` volume de la pièce (m³), `A` surface d'échange thermique (m²), `M` module
//! (m), `B` constante du moule (s/m^{2n}), `n` exposant de Chvorinov (~2), `t`
//! temps de solidification (s). La masselotte doit solidifier **après** la pièce
//! (module supérieur) pour l'alimenter jusqu'au bout.
//!
//! **Convention** : SI cohérent. **Limite honnête** : règle **empirique** de
//! Chvorinov (moule semi-infini, surchauffe modérée) ; `B` et `n` dépendent du
//! matériau/moule et sont fournis par l'appelant. Ne calcule ni la retassure
//! résiduelle, ni le réseau d'alimentation complet.

/// Module thermique `M = V/A` (m).
///
/// Panique si `area <= 0`.
pub fn casting_modulus(volume: f64, area: f64) -> f64 {
    assert!(area > 0.0, "la surface doit être strictement positive");
    volume / area
}

/// Temps de solidification (Chvorinov) `t = B·Mⁿ` (s).
///
/// Panique si `modulus < 0`.
pub fn solidification_time(mold_constant: f64, modulus: f64, exponent: f64) -> f64 {
    assert!(modulus >= 0.0, "le module doit être positif");
    mold_constant * modulus.powf(exponent)
}

/// Module minimal de masselotte pour une alimentation saine
/// `M_masselotte = k·M_pièce`.
pub fn riser_modulus(casting_modulus: f64, factor: f64) -> f64 {
    factor * casting_modulus
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn modulus_of_a_cube() {
        // Cube d'arête a : V=a³, A=6a² → M = a/6.
        let a = 0.06_f64;
        assert_relative_eq!(
            casting_modulus(a.powi(3), 6.0 * a * a),
            a / 6.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn solidification_scales_with_modulus_squared() {
        // n=2 : doubler le module quadruple le temps.
        let t1 = solidification_time(1500.0, 0.01, 2.0);
        let t2 = solidification_time(1500.0, 0.02, 2.0);
        assert_relative_eq!(t2 / t1, 4.0, epsilon = 1e-9);
        assert_relative_eq!(t1, 1500.0 * 0.01f64.powi(2), epsilon = 1e-9);
    }

    #[test]
    fn riser_must_solidify_later() {
        // Masselotte de module 1,2× → solidifie après la pièce (temps plus long).
        let mc = casting_modulus(0.06f64.powi(3), 6.0 * 0.06 * 0.06);
        let mr = riser_modulus(mc, 1.2);
        assert!(mr > mc);
        assert!(solidification_time(1500.0, mr, 2.0) > solidification_time(1500.0, mc, 2.0));
    }

    #[test]
    fn bigger_modulus_takes_longer() {
        // Une pièce plus massive (module plus grand) met plus de temps à solidifier.
        assert!(solidification_time(1500.0, 0.02, 2.0) > solidification_time(1500.0, 0.005, 2.0));
    }

    #[test]
    #[should_panic(expected = "surface")]
    fn zero_area_panics() {
        casting_modulus(1e-3, 0.0);
    }
}
