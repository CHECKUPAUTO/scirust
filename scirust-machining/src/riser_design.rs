//! Fonderie — **dimensionnement de masselotte** par la méthode des modules
//! (module de refroidissement `M = V/A`, critère de solidification différée).
//!
//! ```text
//! module de refroidissement   M = V / A
//! critère de masselotte       M_masselotte = k · M_pièce      (k ≈ 1,2)
//! masselotte cylindrique      M = r·D / (2 + 4r)   avec r = H/D
//!   ⇒ diamètre                D = M·(2 + 4r) / r
//!   ⇒ volume                  V = (π/4)·r·D³
//! rendement d'alimentation    η = V_retrait / V_masselotte
//! ```
//!
//! `V` volume (m³), `A` surface d'échange thermique (m²), `M` module (m),
//! `k` facteur de sécurité (sans dimension, usuellement ~1,2), `D` diamètre de
//! la masselotte (m), `H` hauteur (m), `r = H/D` élancement (sans dimension),
//! `V_retrait` volume de retassure à compenser (m³), `η` rendement (sans
//! dimension). La masselotte doit posséder un module **supérieur** à celui de
//! la pièce pour solidifier en dernier et l'alimenter jusqu'au bout.
//!
//! **Convention** : SI cohérent. La masselotte cylindrique est idéalisée avec
//! **toutes** ses surfaces exposées (deux bases + surface latérale). **Limite
//! honnête** : méthode des modules de Chvorinov (`M = V/A`) ; le facteur de
//! sécurité `k` et le volume de retrait sont **fournis** par l'appelant (jamais
//! de valeur « par défaut » inventée). Ne dimensionne ni le col d'attaque, ni
//! le réseau d'alimentation, et ne modélise pas la retassure résiduelle.

use core::f64::consts::PI;

/// Module de refroidissement `M = V/A` (m).
///
/// Panique si `cooling_surface_area <= 0` ou si `volume < 0`.
pub fn riser_cooling_modulus(volume: f64, cooling_surface_area: f64) -> f64 {
    assert!(volume >= 0.0, "le volume doit être positif");
    assert!(
        cooling_surface_area > 0.0,
        "la surface de refroidissement doit être strictement positive"
    );
    volume / cooling_surface_area
}

/// Module minimal exigé pour la masselotte `M_masselotte = k · M_pièce` (m).
///
/// Panique si `casting_modulus < 0` ou si `safety_factor <= 0`.
pub fn riser_modulus_criterion(casting_modulus: f64, safety_factor: f64) -> f64 {
    assert!(
        casting_modulus >= 0.0,
        "le module de la pièce doit être positif"
    );
    assert!(
        safety_factor > 0.0,
        "le facteur de sécurité doit être strictement positif"
    );
    safety_factor * casting_modulus
}

/// Volume d'une masselotte cylindrique atteignant un module donné,
/// `V = (π/4)·r·D³` avec `D = M·(2 + 4r)/r` et `r = H/D` (m³).
///
/// Toutes les surfaces du cylindre sont supposées échanger la chaleur.
///
/// Panique si `target_modulus <= 0` ou si `height_to_diameter_ratio <= 0`.
pub fn riser_cylinder_volume_for_modulus(
    target_modulus: f64,
    height_to_diameter_ratio: f64,
) -> f64 {
    assert!(
        target_modulus > 0.0,
        "le module visé doit être strictement positif"
    );
    assert!(
        height_to_diameter_ratio > 0.0,
        "l'élancement H/D doit être strictement positif"
    );
    let r = height_to_diameter_ratio;
    let diameter = target_modulus * (2.0 + 4.0 * r) / r;
    (PI / 4.0) * r * diameter.powi(3)
}

/// Rendement d'alimentation `η = V_retrait / V_masselotte` (sans dimension).
///
/// Panique si `riser_volume <= 0` ou si `casting_shrinkage_volume < 0`.
pub fn riser_feeding_efficiency(riser_volume: f64, casting_shrinkage_volume: f64) -> f64 {
    assert!(
        riser_volume > 0.0,
        "le volume de la masselotte doit être strictement positif"
    );
    assert!(
        casting_shrinkage_volume >= 0.0,
        "le volume de retrait doit être positif"
    );
    casting_shrinkage_volume / riser_volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cooling_modulus_of_a_sphere() {
        // Sphère de rayon R : V=(4/3)πR³, A=4πR² → M = R/3.
        let radius = 0.03_f64;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let area = 4.0 * PI * radius * radius;
        assert_relative_eq!(
            riser_cooling_modulus(volume, area),
            radius / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn criterion_is_scaled_casting_modulus() {
        // η du critère : le module exigé vaut k fois celui de la pièce.
        let mc = 0.015_f64;
        let k = 1.2_f64;
        assert_relative_eq!(riser_modulus_criterion(mc, k), k * mc, epsilon = 1e-12);
        // Avec k>1 la masselotte a bien un module supérieur à la pièce.
        assert!(riser_modulus_criterion(mc, k) > mc);
    }

    #[test]
    fn cylinder_volume_reproduces_target_modulus() {
        // Réciprocité : le volume calculé, ramené à son module V/A, redonne la cible.
        let target = 0.02_f64;
        let r = 1.0_f64;
        let volume = riser_cylinder_volume_for_modulus(target, r);
        // Reconstruction de la géométrie idéalisée.
        let diameter = target * (2.0 + 4.0 * r) / r;
        let area = PI * diameter * diameter * (0.5 + r);
        assert_relative_eq!(riser_cooling_modulus(volume, area), target, epsilon = 1e-12);
        // Cas chiffré : M=0,02 m, r=1 → D=0,12 m, V=(π/4)·0,12³.
        assert_relative_eq!(diameter, 0.12, epsilon = 1e-12);
        assert_relative_eq!(volume, (PI / 4.0) * 0.12_f64.powi(3), epsilon = 1e-12);
    }

    #[test]
    fn cylinder_volume_scales_with_modulus_cubed() {
        // À élancement fixe, D ∝ M donc V ∝ M³ : doubler M multiplie V par 8.
        let r = 0.8_f64;
        let v1 = riser_cylinder_volume_for_modulus(0.01, r);
        let v2 = riser_cylinder_volume_for_modulus(0.02, r);
        assert_relative_eq!(v2 / v1, 8.0, epsilon = 1e-9);
    }

    #[test]
    fn feeding_efficiency_ratio() {
        // Rendement = retrait / masselotte : 2,8e-5 / 2,0e-4 = 0,14.
        assert_relative_eq!(
            riser_feeding_efficiency(2.0e-4, 2.8e-5),
            0.14,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "surface")]
    fn zero_area_panics() {
        riser_cooling_modulus(1e-3, 0.0);
    }
}
