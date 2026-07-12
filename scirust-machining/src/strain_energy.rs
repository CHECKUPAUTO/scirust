//! Énergie de déformation élastique — densités d'énergie en traction et en
//! cisaillement, module de résilience et énergie totale.
//!
//! ```text
//! densité (traction)     u = σ²/(2·E)
//! densité (cisaillement) u = τ²/(2·G)
//! module de résilience   u_r = σe²/(2·E)      (à la limite élastique)
//! énergie totale         U = u·V
//! ```
//!
//! `σ`/`τ` contrainte normale/de cisaillement (Pa), `E`/`G` modules de Young/de
//! cisaillement (Pa), `σe` limite élastique (Pa), `V` volume (m³), `u` densité
//! d'énergie (J/m³), `U` énergie (J). Le module de résilience est l'énergie
//! élastique maximale stockée par unité de volume avant plastification.
//!
//! **Convention** : SI cohérent. **Limite honnête** : domaine **élastique
//! linéaire** (contrainte uniforme) ; la ténacité (aire totale sous la courbe
//! jusqu'à rupture) demande l'intégration de la courbe réelle et n'est pas
//! calculée ici.

/// Densité d'énergie de déformation en traction `u = σ²/(2·E)` (J/m³).
///
/// Panique si `youngs_modulus <= 0`.
pub fn axial_strain_energy_density(stress: f64, youngs_modulus: f64) -> f64 {
    assert!(youngs_modulus > 0.0, "E doit être strictement positif");
    stress * stress / (2.0 * youngs_modulus)
}

/// Densité d'énergie de déformation en cisaillement `u = τ²/(2·G)` (J/m³).
///
/// Panique si `shear_modulus <= 0`.
pub fn shear_strain_energy_density(tau: f64, shear_modulus: f64) -> f64 {
    assert!(shear_modulus > 0.0, "G doit être strictement positif");
    tau * tau / (2.0 * shear_modulus)
}

/// Module de résilience `u_r = σe²/(2·E)` (J/m³).
///
/// Panique si `youngs_modulus <= 0`.
pub fn modulus_of_resilience(yield_stress: f64, youngs_modulus: f64) -> f64 {
    assert!(youngs_modulus > 0.0, "E doit être strictement positif");
    yield_stress * yield_stress / (2.0 * youngs_modulus)
}

/// Énergie de déformation totale `U = u·V` (J).
pub fn total_strain_energy(energy_density: f64, volume: f64) -> f64 {
    energy_density * volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn axial_density_scales_with_stress_squared() {
        // Doubler la contrainte quadruple la densité d'énergie.
        let u1 = axial_strain_energy_density(100e6, 210e9);
        let u2 = axial_strain_energy_density(200e6, 210e9);
        assert_relative_eq!(u2 / u1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn resilience_of_spring_steel() {
        // σe=1200 MPa, E=210 GPa → u_r = (1,2e9)²/(2·210e9) ≈ 3,43 MJ/m³.
        let ur = modulus_of_resilience(1200e6, 210e9);
        assert_relative_eq!(ur, (1200e6f64).powi(2) / (2.0 * 210e9), epsilon = 1.0);
        assert!(ur > 3.4e6 && ur < 3.5e6);
    }

    #[test]
    fn total_energy_is_density_times_volume() {
        // u=1e5 J/m³, V=2e-3 m³ → U = 200 J.
        assert_relative_eq!(total_strain_energy(1e5, 2e-3), 200.0, epsilon = 1e-9);
    }

    #[test]
    fn shear_density_definition() {
        assert_relative_eq!(
            shear_strain_energy_density(50e6, 80e9),
            (50e6f64).powi(2) / (2.0 * 80e9),
            epsilon = 1e-3
        );
    }

    #[test]
    #[should_panic(expected = "E doit être")]
    fn zero_modulus_panics() {
        axial_strain_energy_density(100e6, 0.0);
    }
}
