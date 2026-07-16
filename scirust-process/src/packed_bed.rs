//! Perte de charge en lit fixe garni — équation d'Ergun (somme d'un terme
//! visqueux de Kozeny-Carman et d'un terme inertiel de Burke-Plummer) et perte
//! de charge intégrée sur la hauteur du lit.
//!
//! ```text
//! équation d'Ergun (gradient de pression)
//!   dP/L = 150 · (1 − ε)² · μ · U / (ε³ · d_p²)
//!        + 1.75 · (1 − ε) · ρ · U² / (ε³ · d_p)                  [Pa·m⁻¹]
//! terme visqueux seul (Kozeny-Carman)
//!   (dP/L)_visq = 150 · (1 − ε)² · μ · U / (ε³ · d_p²)           [Pa·m⁻¹]
//! terme inertiel seul (Burke-Plummer)
//!   (dP/L)_inert = 1.75 · (1 − ε) · ρ · U² / (ε³ · d_p)          [Pa·m⁻¹]
//! perte de charge sur la hauteur
//!   ΔP = (dP/L) · L                                              [Pa]
//! ```
//!
//! `ε` porosité (fraction de vide) du lit [sans dimension, 0 < ε < 1], `μ`
//! viscosité dynamique du fluide [Pa·s], `U` vitesse **superficielle** (en fût
//! vide) [m·s⁻¹], `d_p` diamètre de particule [m], `ρ` masse volumique du fluide
//! [kg·m⁻³], `L` hauteur (épaisseur) du lit [m] ; `dP/L` gradient de pression
//! [Pa·m⁻¹] ; `ΔP` perte de charge à travers le lit [Pa].
//!
//! **Limite honnête** : modèle à l'échelle des **opérations unitaires**, valable
//! pour un lit de particules approximativement **sphériques et uniformes**. La
//! vitesse employée est la vitesse **superficielle** (débit volumique divisé par
//! la section du fût vide), pas la vitesse interstitielle. La **porosité** `ε`
//! du lit et le **diamètre de particule** `d_p` sont **FOURNIS** par l'appelant
//! (ils dépendent du garnissage et de sa mise en place, jamais supposés « par
//! défaut »). Toutes les **propriétés du fluide** (masse volumique, viscosité)
//! sont **FOURNIES** : aucune enthalpie, volatilité, coefficient de partage,
//! constante cinétique ou diffusivité n'est calculé ni inventé par ce module.

/// Gradient de pression d'Ergun en lit fixe
/// `dP/L = 150·(1 − ε)²·μ·U/(ε³·d_p²) + 1.75·(1 − ε)·ρ·U²/(ε³·d_p)` (Pa·m⁻¹),
/// somme du terme visqueux (Kozeny-Carman) et du terme inertiel (Burke-Plummer).
///
/// `voidage` (ε) porosité du lit [sans dimension], `fluid_viscosity` (μ) [Pa·s],
/// `superficial_velocity` (U) vitesse superficielle [m·s⁻¹], `particle_diameter`
/// (d_p) [m], `fluid_density` (ρ) [kg·m⁻³].
///
/// Panique si `ε` hors de `]0, 1[`, si `μ < 0`, si `U < 0`, si `d_p ≤ 0`, ou si
/// `ρ < 0`.
pub fn packedbed_ergun_pressure_gradient(
    voidage: f64,
    fluid_viscosity: f64,
    superficial_velocity: f64,
    particle_diameter: f64,
    fluid_density: f64,
) -> f64 {
    assert!(
        voidage > 0.0 && voidage < 1.0,
        "0 < ε < 1 requis (porosité du lit)"
    );
    assert!(fluid_viscosity >= 0.0, "μ ≥ 0 requis (viscosité du fluide)");
    assert!(
        superficial_velocity >= 0.0,
        "U ≥ 0 requis (vitesse superficielle)"
    );
    assert!(
        particle_diameter > 0.0,
        "d_p > 0 requis (diamètre de particule)"
    );
    assert!(
        fluid_density >= 0.0,
        "ρ ≥ 0 requis (masse volumique du fluide)"
    );
    150.0 * (1.0 - voidage).powi(2) * fluid_viscosity * superficial_velocity
        / (voidage.powi(3) * particle_diameter * particle_diameter)
        + 1.75 * (1.0 - voidage) * fluid_density * superficial_velocity * superficial_velocity
            / (voidage.powi(3) * particle_diameter)
}

/// Terme **visqueux** seul du gradient d'Ergun (loi de Kozeny-Carman)
/// `(dP/L)_visq = 150·(1 − ε)²·μ·U/(ε³·d_p²)` (Pa·m⁻¹), dominant en écoulement
/// rampant (faible nombre de Reynolds de particule).
///
/// `voidage` (ε) porosité du lit [sans dimension], `fluid_viscosity` (μ) [Pa·s],
/// `superficial_velocity` (U) vitesse superficielle [m·s⁻¹], `particle_diameter`
/// (d_p) [m].
///
/// Panique si `ε` hors de `]0, 1[`, si `μ < 0`, si `U < 0`, ou si `d_p ≤ 0`.
pub fn packedbed_kozeny_carman_gradient(
    voidage: f64,
    fluid_viscosity: f64,
    superficial_velocity: f64,
    particle_diameter: f64,
) -> f64 {
    assert!(
        voidage > 0.0 && voidage < 1.0,
        "0 < ε < 1 requis (porosité du lit)"
    );
    assert!(fluid_viscosity >= 0.0, "μ ≥ 0 requis (viscosité du fluide)");
    assert!(
        superficial_velocity >= 0.0,
        "U ≥ 0 requis (vitesse superficielle)"
    );
    assert!(
        particle_diameter > 0.0,
        "d_p > 0 requis (diamètre de particule)"
    );
    150.0 * (1.0 - voidage).powi(2) * fluid_viscosity * superficial_velocity
        / (voidage.powi(3) * particle_diameter * particle_diameter)
}

/// Terme **inertiel** seul du gradient d'Ergun (loi de Burke-Plummer)
/// `(dP/L)_inert = 1.75·(1 − ε)·ρ·U²/(ε³·d_p)` (Pa·m⁻¹), dominant en écoulement
/// turbulent (grand nombre de Reynolds de particule).
///
/// `voidage` (ε) porosité du lit [sans dimension], `superficial_velocity` (U)
/// vitesse superficielle [m·s⁻¹], `particle_diameter` (d_p) [m], `fluid_density`
/// (ρ) [kg·m⁻³].
///
/// Panique si `ε` hors de `]0, 1[`, si `U < 0`, si `d_p ≤ 0`, ou si `ρ < 0`.
pub fn packedbed_burke_plummer_gradient(
    voidage: f64,
    superficial_velocity: f64,
    particle_diameter: f64,
    fluid_density: f64,
) -> f64 {
    assert!(
        voidage > 0.0 && voidage < 1.0,
        "0 < ε < 1 requis (porosité du lit)"
    );
    assert!(
        superficial_velocity >= 0.0,
        "U ≥ 0 requis (vitesse superficielle)"
    );
    assert!(
        particle_diameter > 0.0,
        "d_p > 0 requis (diamètre de particule)"
    );
    assert!(
        fluid_density >= 0.0,
        "ρ ≥ 0 requis (masse volumique du fluide)"
    );
    1.75 * (1.0 - voidage) * fluid_density * superficial_velocity * superficial_velocity
        / (voidage.powi(3) * particle_diameter)
}

/// Perte de charge à travers le lit `ΔP = (dP/L)·L` (Pa), gradient de pression
/// intégré sur la hauteur du lit (gradient supposé uniforme).
///
/// `pressure_gradient` (dP/L) gradient de pression [Pa·m⁻¹], `bed_height` (L)
/// hauteur du lit [m].
///
/// Panique si `dP/L < 0` ou si `L < 0`.
pub fn packedbed_pressure_drop(pressure_gradient: f64, bed_height: f64) -> f64 {
    assert!(
        pressure_gradient >= 0.0,
        "dP/L ≥ 0 requis (gradient de pression)"
    );
    assert!(bed_height >= 0.0, "L ≥ 0 requis (hauteur du lit)");
    pressure_gradient * bed_height
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ergun_is_sum_of_the_two_branches() {
        // dP/L = (dP/L)_visq + (dP/L)_inert : identité de décomposition d'Ergun.
        let (voidage, mu, u, dp, rho) = (0.4_f64, 1.8e-5_f64, 0.1_f64, 5.0e-3_f64, 1.2_f64);
        let total = packedbed_ergun_pressure_gradient(voidage, mu, u, dp, rho);
        let visc = packedbed_kozeny_carman_gradient(voidage, mu, u, dp);
        let inert = packedbed_burke_plummer_gradient(voidage, u, dp, rho);
        assert_relative_eq!(total, visc + inert, max_relative = 1e-12);
    }

    #[test]
    fn ergun_realistic_case() {
        // ε = 0.4, μ = 1.8e-5 Pa·s, U = 0.1 m/s, d_p = 5 mm, ρ = 1.2 kg/m³.
        //   visqueux = 150·0.6²·1.8e-5·0.1 / (0.4³·(5e-3)²)
        //            = 9.72e-5 / 1.6e-6 = 60.75 Pa/m
        //   inertiel = 1.75·0.6·1.2·0.1² / (0.4³·5e-3)
        //            = 0.0126 / 3.2e-4 = 39.375 Pa/m
        //   total    = 60.75 + 39.375 = 100.125 Pa/m.
        let total =
            packedbed_ergun_pressure_gradient(0.4_f64, 1.8e-5_f64, 0.1_f64, 5.0e-3_f64, 1.2_f64);
        assert_relative_eq!(total, 100.125, max_relative = 1e-3);
        let visc = packedbed_kozeny_carman_gradient(0.4_f64, 1.8e-5_f64, 0.1_f64, 5.0e-3_f64);
        assert_relative_eq!(visc, 60.75, max_relative = 1e-3);
        let inert = packedbed_burke_plummer_gradient(0.4_f64, 0.1_f64, 5.0e-3_f64, 1.2_f64);
        assert_relative_eq!(inert, 39.375, max_relative = 1e-3);
    }

    #[test]
    fn viscous_branch_scales_linearly_with_velocity() {
        // (dP/L)_visq ∝ U : doubler la vitesse double le terme de Kozeny-Carman.
        let single = packedbed_kozeny_carman_gradient(0.45_f64, 1.0e-3_f64, 0.02_f64, 3.0e-3_f64);
        let double = packedbed_kozeny_carman_gradient(0.45_f64, 1.0e-3_f64, 0.04_f64, 3.0e-3_f64);
        assert_relative_eq!(double, 2.0 * single, max_relative = 1e-12);
    }

    #[test]
    fn inertial_branch_scales_quadratically_with_velocity() {
        // (dP/L)_inert ∝ U² : doubler la vitesse quadruple le terme de
        // Burke-Plummer.
        let single = packedbed_burke_plummer_gradient(0.42_f64, 0.05_f64, 4.0e-3_f64, 1000.0_f64);
        let double = packedbed_burke_plummer_gradient(0.42_f64, 0.10_f64, 4.0e-3_f64, 1000.0_f64);
        assert_relative_eq!(double, 4.0 * single, max_relative = 1e-12);
    }

    #[test]
    fn pressure_drop_is_gradient_times_height() {
        // ΔP = (dP/L)·L : cas simple 100.125 Pa/m sur 2 m ⇒ 200.25 Pa, et
        // linéarité en hauteur.
        let grad =
            packedbed_ergun_pressure_gradient(0.4_f64, 1.8e-5_f64, 0.1_f64, 5.0e-3_f64, 1.2_f64);
        assert_relative_eq!(
            packedbed_pressure_drop(grad, 2.0_f64),
            200.25,
            max_relative = 1e-3
        );
        let one = packedbed_pressure_drop(grad, 1.0_f64);
        let three = packedbed_pressure_drop(grad, 3.0_f64);
        assert_relative_eq!(three, 3.0 * one, max_relative = 1e-12);
    }

    #[test]
    fn zero_velocity_gives_zero_gradient() {
        // U = 0 ⇒ aucun écoulement ⇒ gradient nul (cas limite).
        let grad =
            packedbed_ergun_pressure_gradient(0.4_f64, 1.8e-5_f64, 0.0_f64, 5.0e-3_f64, 1.2_f64);
        assert_relative_eq!(grad, 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "0 < ε < 1 requis")]
    fn ergun_panics_on_invalid_voidage() {
        // ε = 0 ⇒ division par ε³ = 0 ⇒ entrée rejetée.
        let _ =
            packedbed_ergun_pressure_gradient(0.0_f64, 1.8e-5_f64, 0.1_f64, 5.0e-3_f64, 1.2_f64);
    }
}
