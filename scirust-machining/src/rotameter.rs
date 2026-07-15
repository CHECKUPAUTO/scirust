//! Débitmètre à **section variable** (rotamètre à flotteur) : débit volumique à
//! l'équilibre du flotteur dans un tube conique.
//!
//! ```text
//! aire annulaire      A_a = π·(R_h² − r_f²)
//! chute de pression   ΔP  = V_f·(ρ_f − ρ)·g / A_c
//! débit volumique     Q   = Cd·A_a·√(2·g·V_f·(ρ_f − ρ)/(ρ·A_c))
//!                         = Cd·A_a·√(2·ΔP/ρ)
//! ```
//!
//! `R_h` rayon interne du tube à la hauteur du flotteur (m), `r_f` rayon du
//! flotteur (m), `A_a` aire annulaire de passage (m²), `V_f` volume du flotteur
//! (m³), `A_c` aire de la section droite maximale du flotteur (m²), `ρ_f` masse
//! volumique du flotteur (kg/m³), `ρ` masse volumique du fluide (kg/m³), `g`
//! accélération de la pesanteur (m/s²), `Cd` coefficient de décharge (sans
//! dimension), `ΔP` chute de pression au passage (Pa), `Q` débit volumique
//! (m³/s).
//!
//! **Convention** : SI cohérent. **Limite honnête** : modèle d'équilibre du
//! flotteur (poids − poussée = traînée) donnant une chute de pression constante.
//! Le coefficient de décharge `Cd` (dépendant du nombre de Reynolds et de la
//! forme du flotteur), les masses volumiques et la géométrie du tube conique
//! (l'aire annulaire varie avec la hauteur) sont **fournis** par l'appelant.
//! Aucune constante physique, matériau ou procédé n'est inventée ici.

use core::f64::consts::PI;

/// Aire annulaire de passage `A_a = π·(R_h² − r_f²)` (m²).
///
/// Section libre entre le flotteur et la paroi du tube à la hauteur d'équilibre.
///
/// Panique si `float_radius < 0` ou `tube_radius_at_height < float_radius`.
pub fn rotameter_annular_area(tube_radius_at_height: f64, float_radius: f64) -> f64 {
    assert!(
        float_radius >= 0.0,
        "le rayon du flotteur doit être positif"
    );
    assert!(
        tube_radius_at_height >= float_radius,
        "le rayon du tube doit être supérieur ou égal à celui du flotteur"
    );
    PI * (tube_radius_at_height.powi(2) - float_radius.powi(2))
}

/// Chute de pression à l'équilibre `ΔP = V_f·(ρ_f − ρ)·g / A_c` (Pa).
///
/// À l'équilibre, cette chute de pression au passage du flotteur est constante,
/// indépendante du débit (c'est l'aire annulaire qui s'ajuste avec la hauteur).
///
/// Panique si `float_cross_section <= 0`, `float_volume < 0`, `gravity <= 0`, ou
/// `float_density < fluid_density`.
pub fn rotameter_float_equilibrium_pressure_drop(
    float_volume: f64,
    float_density: f64,
    fluid_density: f64,
    gravity: f64,
    float_cross_section: f64,
) -> f64 {
    assert!(
        float_cross_section > 0.0,
        "l'aire de section du flotteur doit être strictement positive"
    );
    assert!(
        float_volume >= 0.0,
        "le volume du flotteur doit être positif"
    );
    assert!(gravity > 0.0, "la pesanteur doit être strictement positive");
    assert!(
        float_density >= fluid_density,
        "le flotteur doit être plus dense que le fluide (ρ_f ≥ ρ)"
    );
    float_volume * (float_density - fluid_density) * gravity / float_cross_section
}

/// Débit volumique à l'équilibre
/// `Q = Cd·A_a·√(2·g·V_f·(ρ_f − ρ)/(ρ·A_c))` (m³/s).
///
/// Équivaut à `Cd·A_a·√(2·ΔP/ρ)` avec `ΔP` la chute de pression d'équilibre.
///
/// Panique si `discharge_coefficient < 0`, `annular_area < 0`,
/// `float_volume < 0`, `float_density < fluid_density`, `fluid_density <= 0`,
/// `gravity <= 0`, ou `float_cross_section <= 0`.
#[allow(clippy::too_many_arguments)]
pub fn rotameter_volumetric_flow(
    discharge_coefficient: f64,
    annular_area: f64,
    float_volume: f64,
    float_density: f64,
    fluid_density: f64,
    gravity: f64,
    float_cross_section: f64,
) -> f64 {
    assert!(
        discharge_coefficient >= 0.0,
        "le coefficient de décharge doit être positif"
    );
    assert!(annular_area >= 0.0, "l'aire annulaire doit être positive");
    assert!(
        float_volume >= 0.0,
        "le volume du flotteur doit être positif"
    );
    assert!(
        fluid_density > 0.0,
        "la masse volumique du fluide doit être strictement positive"
    );
    assert!(gravity > 0.0, "la pesanteur doit être strictement positive");
    assert!(
        float_cross_section > 0.0,
        "l'aire de section du flotteur doit être strictement positive"
    );
    assert!(
        float_density >= fluid_density,
        "le flotteur doit être plus dense que le fluide (ρ_f ≥ ρ)"
    );
    let radicand = 2.0 * gravity * float_volume * (float_density - fluid_density)
        / (fluid_density * float_cross_section);
    discharge_coefficient * annular_area * radicand.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn annular_area_vanishes_when_float_fills_tube() {
        // Flotteur au diamètre du tube : aucun passage, A_a = 0.
        assert_relative_eq!(rotameter_annular_area(0.010, 0.010), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn annular_area_realistic_value() {
        // R_h = 11 mm, r_f = 10 mm → A_a = π·(0,011² − 0,010²) = π·2,1e-5.
        let a = rotameter_annular_area(0.011, 0.010);
        assert_relative_eq!(a, 6.597_344_57e-5, max_relative = 1e-6);
    }

    #[test]
    fn pressure_drop_realistic_value() {
        // V_f = 1e-6 m³, ρ_f = 8000, ρ = 1000, g = 9,81, A_c = 1e-4.
        // ΔP = 1e-6·7000·9,81 / 1e-4 = 0,068670 / 1e-4 = 686,70 Pa.
        let dp = rotameter_float_equilibrium_pressure_drop(1.0e-6, 8000.0, 1000.0, 9.81, 1.0e-4);
        assert_relative_eq!(dp, 686.70, max_relative = 1e-9);
    }

    #[test]
    fn flow_equals_pressure_drop_form() {
        // Identité Q = Cd·A_a·√(2·ΔP/ρ) : la forme « chute de pression » et la
        // forme développée doivent coïncider.
        let (cd, a_a, v_f, rho_f, rho, g, a_c) =
            (0.7, 6.5e-5, 1.0e-6, 8000.0, 1000.0, 9.81, 1.0e-4);
        let dp = rotameter_float_equilibrium_pressure_drop(v_f, rho_f, rho, g, a_c);
        let expected = cd * a_a * (2.0_f64 * dp / rho).sqrt();
        let q = rotameter_volumetric_flow(cd, a_a, v_f, rho_f, rho, g, a_c);
        assert_relative_eq!(q, expected, max_relative = 1e-12);
    }

    #[test]
    fn flow_is_proportional_to_annular_area() {
        // Q ∝ A_a : doubler l'aire annulaire double le débit.
        let q1 = rotameter_volumetric_flow(0.7, 6.5e-5, 1.0e-6, 8000.0, 1000.0, 9.81, 1.0e-4);
        let q2 = rotameter_volumetric_flow(0.7, 13.0e-5, 1.0e-6, 8000.0, 1000.0, 9.81, 1.0e-4);
        assert_relative_eq!(q2 / q1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn flow_realistic_case() {
        // Cd = 0,7, A_a = π·2,1e-5 = 6,5973446e-5, V_f = 1e-6, ρ_f = 8000,
        // ρ = 1000, g = 9,81, A_c = 1e-4.
        // Radicande = 2·9,81·1e-6·7000/(1000·1e-4) = 0,13734/0,1 = 1,3734.
        // Q = 0,7·6,5973446e-5·√1,3734 = 5,4120990e-5 m³/s.
        let a_a = rotameter_annular_area(0.011, 0.010);
        let q = rotameter_volumetric_flow(0.7, a_a, 1.0e-6, 8000.0, 1000.0, 9.81, 1.0e-4);
        assert_relative_eq!(q, 5.412_099e-5, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le flotteur doit être plus dense que le fluide")]
    fn lighter_float_panics() {
        rotameter_float_equilibrium_pressure_drop(1.0e-6, 900.0, 1000.0, 9.81, 1.0e-4);
    }
}
