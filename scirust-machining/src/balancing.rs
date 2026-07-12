//! Équilibrage des rotors — force centrifuge de balourd, correction en un plan et
//! balourd résiduel admissible selon la qualité **ISO 1940-1** (grades `G`).
//!
//! ```text
//! balourd (statique)      U = m·e            (g·mm ou kg·m)
//! force centrifuge        F = m·e·ω² = U·ω²
//! excentricité admissible e_per = 1000·G/ω   (µm, avec G en mm/s, ω en rad/s)
//! balourd admissible      U_per = e_per·M_rotor
//! masse de correction     m_c = U/r_c        (à un rayon de correction r_c)
//! ```
//!
//! `m` masse du balourd, `e` excentricité (décalage centre de masse/axe), `ω`
//! vitesse de rotation (rad/s), `G` grade de qualité ISO 1940 (`e·ω` en mm/s,
//! p. ex. `G6.3` pour de la machine-outil courante, `G2.5` pour des broches),
//! `M_rotor` masse du rotor.
//!
//! **Convention** : SI cohérent sauf l'excentricité admissible rendue en **µm**
//! (usage ISO 1940). **Limite honnête** : rotor **rigide**, équilibrage en un ou
//! deux plans idéaux ; pas de rotor flexible, ni de couplage de plans réel — la
//! qualité G est une donnée normative fournie par l'appelant.

/// Force centrifuge d'un balourd `F = m·e·ω²` (N).
pub fn centrifugal_force(mass_kg: f64, eccentricity_m: f64, omega_rad_s: f64) -> f64 {
    mass_kg * eccentricity_m * omega_rad_s * omega_rad_s
}

/// Balourd `U = m·e` (kg·m dans les unités SI).
pub fn unbalance(mass_kg: f64, eccentricity_m: f64) -> f64 {
    mass_kg * eccentricity_m
}

/// Excentricité résiduelle **admissible** `e_per = 1000·G/ω` (µm),
/// `G` grade ISO 1940 en mm/s, `ω` vitesse de service en rad/s.
///
/// Panique si `omega <= 0`.
pub fn permissible_eccentricity_um(grade_g_mm_s: f64, omega_rad_s: f64) -> f64 {
    assert!(
        omega_rad_s > 0.0,
        "la vitesse de rotation doit être strictement positive"
    );
    1000.0 * grade_g_mm_s / omega_rad_s
}

/// Balourd résiduel **admissible** `U_per = e_per·M_rotor` (g·mm),
/// `e_per` en µm et `M_rotor` en kg (produit homogène à des g·mm).
pub fn permissible_unbalance_g_mm(permissible_eccentricity_um: f64, rotor_mass_kg: f64) -> f64 {
    permissible_eccentricity_um * rotor_mass_kg
}

/// Masse de correction `m_c = U/r_c` à placer au rayon `r_c` pour annuler un
/// balourd `U` (unités cohérentes : `U` en kg·m et `r_c` en m → `m_c` en kg).
///
/// Panique si `correction_radius <= 0`.
pub fn correction_mass(unbalance: f64, correction_radius: f64) -> f64 {
    assert!(
        correction_radius > 0.0,
        "le rayon de correction doit être strictement positif"
    );
    unbalance / correction_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn centrifugal_force_scales_with_square_of_speed() {
        // m=0,01 kg, e=0,05 m, ω=100 rad/s → F = 0,01·0,05·10000 = 5 N.
        assert_relative_eq!(centrifugal_force(0.01, 0.05, 100.0), 5.0, epsilon = 1e-9);
        // Doubler ω quadruple la force.
        let f1 = centrifugal_force(0.01, 0.05, 100.0);
        let f2 = centrifugal_force(0.01, 0.05, 200.0);
        assert_relative_eq!(f2 / f1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn iso1940_permissible_eccentricity() {
        // Grade G6.3 à 3000 tr/min (ω=100π rad/s) → e_per = 1000·6,3/(100π) ≈ 20 µm.
        let omega = 100.0 * core::f64::consts::PI;
        let e_per = permissible_eccentricity_um(6.3, omega);
        assert_relative_eq!(e_per, 1000.0 * 6.3 / omega, epsilon = 1e-9);
        assert!(e_per > 19.0 && e_per < 21.0);
    }

    #[test]
    fn permissible_unbalance_from_eccentricity_and_mass() {
        // e_per=20 µm, rotor 15 kg → U_per = 300 g·mm.
        assert_relative_eq!(
            permissible_unbalance_g_mm(20.0, 15.0),
            300.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn correction_mass_cancels_unbalance() {
        // U = 0,01·0,05 = 5e-4 kg·m ; à r_c=0,1 m → m_c = 5e-3 kg = 5 g.
        let u = unbalance(0.01, 0.05);
        assert_relative_eq!(correction_mass(u, 0.1), 5e-3, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "rayon de correction")]
    fn zero_radius_panics() {
        correction_mass(5e-4, 0.0);
    }
}
