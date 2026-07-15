//! Plaque circulaire mince sous pression uniforme — rigidité flexionnelle et
//! flèche maximale (théorie des plaques, cas de Roark) ; complète
//! [`crate::beams`].
//!
//! ```text
//! rigidité flexionnelle   D = E·t³ / (12·(1 - ν²))
//! bord encastré           w_max = p·R⁴ / (64·D)
//! bord simplement appuyé  w_max = (5 + ν)/(1 + ν) · p·R⁴ / (64·D)
//! ```
//!
//! `E` module de Young (Pa), `t` épaisseur de la plaque (m), `ν` coefficient de
//! Poisson (sans dimension), `D` rigidité flexionnelle (N·m), `p` pression
//! uniforme (Pa), `R` rayon de la plaque (m), `w_max` flèche maximale au centre
//! (m).
//!
//! **Convention** : unités SI cohérentes (N, m, Pa). **Limite honnête** :
//! plaque circulaire mince, petites flèches (w ≪ t), charge uniformément
//! répartie, matériau homogène isotrope élastique linéaire ; le cisaillement
//! transverse est négligé. Les propriétés matériau (`E`, `ν`) et la géométrie
//! sont FOURNIES par l'appelant — aucune valeur « par défaut » n'est inventée.

/// Rigidité flexionnelle d'une plaque `D = E·t³ / (12·(1 - ν²))` (N·m),
/// module de Young `youngs_modulus`, épaisseur `thickness`, coefficient de
/// Poisson `poisson_ratio`.
///
/// Panique si `thickness <= 0`, si `youngs_modulus <= 0`, ou si
/// `poisson_ratio` n'est pas dans `]-1, 0.5]`.
pub fn plate_flexural_rigidity(youngs_modulus: f64, thickness: f64, poisson_ratio: f64) -> f64 {
    assert!(
        youngs_modulus > 0.0,
        "le module de Young doit être strictement positif"
    );
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    assert!(
        poisson_ratio > -1.0 && poisson_ratio <= 0.5,
        "le coefficient de Poisson doit être dans l'intervalle ]-1, 0.5]"
    );
    youngs_modulus * thickness.powi(3) / (12.0 * (1.0 - poisson_ratio * poisson_ratio))
}

/// Flèche maximale au centre d'une plaque circulaire **à bord encastré** sous
/// pression uniforme `w_max = p·R⁴ / (64·D)` (m), pression `pressure`, rayon
/// `radius`, rigidité flexionnelle `flexural_rigidity`.
///
/// Panique si `radius <= 0` ou si `flexural_rigidity <= 0`.
pub fn clamped_plate_max_deflection(pressure: f64, radius: f64, flexural_rigidity: f64) -> f64 {
    assert!(radius > 0.0, "le rayon doit être strictement positif");
    assert!(
        flexural_rigidity > 0.0,
        "la rigidité flexionnelle doit être strictement positive"
    );
    pressure * radius.powi(4) / (64.0 * flexural_rigidity)
}

/// Flèche maximale au centre d'une plaque circulaire **à bord simplement
/// appuyé** sous pression uniforme
/// `w_max = (5 + ν)/(1 + ν) · p·R⁴ / (64·D)` (m), pression `pressure`, rayon
/// `radius`, rigidité flexionnelle `flexural_rigidity`, coefficient de Poisson
/// `poisson`.
///
/// Panique si `radius <= 0`, si `flexural_rigidity <= 0`, ou si `poisson`
/// n'est pas dans `]-1, 0.5]`.
pub fn simply_supported_max_deflection(
    pressure: f64,
    radius: f64,
    flexural_rigidity: f64,
    poisson: f64,
) -> f64 {
    assert!(radius > 0.0, "le rayon doit être strictement positif");
    assert!(
        flexural_rigidity > 0.0,
        "la rigidité flexionnelle doit être strictement positive"
    );
    assert!(
        poisson > -1.0 && poisson <= 0.5,
        "le coefficient de Poisson doit être dans l'intervalle ]-1, 0.5]"
    );
    (5.0 + poisson) / (1.0 + poisson) * pressure * radius.powi(4) / (64.0 * flexural_rigidity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn flexural_rigidity_known_value() {
        // Acier : E = 200 GPa, t = 10 mm, ν = 0,3.
        // D = 200e9·(0,01)³ / (12·(1-0,09)) = 200000 / 10,92 ≈ 18315,018 N·m.
        let d = plate_flexural_rigidity(200.0e9, 0.01, 0.3);
        assert_relative_eq!(d, 200_000.0 / 10.92, epsilon = 1e-6);
        assert_relative_eq!(d, 18_315.018_315_018_316, epsilon = 1e-3);
    }

    #[test]
    fn flexural_rigidity_scales_with_thickness_cubed() {
        // Doubler l'épaisseur multiplie D par 8 (t³).
        let d1 = plate_flexural_rigidity(200.0e9, 0.01, 0.3);
        let d2 = plate_flexural_rigidity(200.0e9, 0.02, 0.3);
        assert_relative_eq!(d2, 8.0 * d1, epsilon = 1e-6);
    }

    #[test]
    fn clamped_deflection_known_value() {
        // p = 1 bar = 1e5 Pa, R = 0,5 m, D ci-dessus.
        // w = 1e5·0,5⁴ / (64·D) = 6250 / (64·18315,018) ≈ 5,3320e-3 m.
        let d = plate_flexural_rigidity(200.0e9, 0.01, 0.3);
        let w = clamped_plate_max_deflection(1.0e5, 0.5, d);
        assert_relative_eq!(w, 1.0e5 * 0.5_f64.powi(4) / (64.0 * d), epsilon = 1e-12);
        assert_relative_eq!(w, 5.331_982e-3, epsilon = 1e-6);
    }

    #[test]
    fn clamped_deflection_linear_in_pressure() {
        // Proportionnalité : doubler la pression double la flèche.
        let d = plate_flexural_rigidity(70.0e9, 0.008, 0.33);
        let w1 = clamped_plate_max_deflection(5.0e4, 0.3, d);
        let w2 = clamped_plate_max_deflection(1.0e5, 0.3, d);
        assert_relative_eq!(w2, 2.0 * w1, epsilon = 1e-12);
    }

    #[test]
    fn simply_supported_ratio_to_clamped() {
        // Pour p, R, D identiques : w_ss / w_clamped = (5+ν)/(1+ν).
        // ν = 0,3 → 5,3 / 1,3 ≈ 4,076923.
        let d = plate_flexural_rigidity(200.0e9, 0.01, 0.3);
        let w_clamped = clamped_plate_max_deflection(1.0e5, 0.5, d);
        let w_ss = simply_supported_max_deflection(1.0e5, 0.5, d, 0.3);
        assert_relative_eq!(w_ss / w_clamped, 5.3 / 1.3, epsilon = 1e-12);
        assert_relative_eq!(w_ss / w_clamped, 4.076_923_076_923_077, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la rigidité flexionnelle doit être strictement positive")]
    fn clamped_deflection_rejects_nonpositive_rigidity() {
        let _ = clamped_plate_max_deflection(1.0e5, 0.5, 0.0);
    }
}
