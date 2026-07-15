//! Tension superficielle — surpression de **Laplace** aux interfaces sphériques
//! et longueur capillaire.
//!
//! ```text
//! goutte / interface unique   Δp = 2·γ/r
//! bulle de savon (2 films)    Δp = 4·γ/r
//! longueur capillaire         l_c = √(γ / (ρ·g))
//! ```
//!
//! `γ` (gamma) tension superficielle (N/m), `r` rayon de courbure de l'interface
//! (m), `ρ` (rho) masse volumique du liquide (kg/m³), `g` accélération de la
//! pesanteur (m/s²), `Δp` surpression capillaire (Pa), `l_c` longueur capillaire
//! (m). Une bulle de savon possède **deux** interfaces liquide/air, d'où le
//! facteur 4 au lieu de 2 pour une goutte à interface unique.
//!
//! **Convention** : SI cohérent. **Limite honnête** : interfaces **sphériques**
//! (rayon de courbure unique, tensioactifs négligés). La tension superficielle,
//! la masse volumique et l'accélération de la pesanteur sont **fournies par
//! l'appelant** ; aucune valeur « par défaut » de matériau n'est inventée.

/// Accélération de la pesanteur standard (m/s²) utilisée par
/// [`laplace_capillary_length`].
pub const SURFACE_TENSION_STANDARD_GRAVITY: f64 = 9.81;

/// Surpression de Laplace d'une goutte ou interface sphérique **unique**
/// `Δp = 2·γ/r` (Pa).
///
/// Panique si `surface_tension < 0` ou `radius <= 0`.
pub fn laplace_droplet_pressure(surface_tension: f64, radius: f64) -> f64 {
    assert!(
        surface_tension >= 0.0,
        "la tension superficielle doit être positive"
    );
    assert!(radius > 0.0, "le rayon doit être strictement positif");
    2.0 * surface_tension / radius
}

/// Surpression de Laplace d'une bulle de savon à **deux** interfaces
/// `Δp = 4·γ/r` (Pa).
///
/// Panique si `surface_tension < 0` ou `radius <= 0`.
pub fn laplace_bubble_pressure(surface_tension: f64, radius: f64) -> f64 {
    assert!(
        surface_tension >= 0.0,
        "la tension superficielle doit être positive"
    );
    assert!(radius > 0.0, "le rayon doit être strictement positif");
    4.0 * surface_tension / radius
}

/// Longueur capillaire `l_c = √(γ / (ρ·g))` (m), avec
/// `g = `[`SURFACE_TENSION_STANDARD_GRAVITY`].
///
/// Panique si `surface_tension < 0` ou `density <= 0`.
pub fn laplace_capillary_length(surface_tension: f64, density: f64) -> f64 {
    assert!(
        surface_tension >= 0.0,
        "la tension superficielle doit être positive"
    );
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    (surface_tension / (density * SURFACE_TENSION_STANDARD_GRAVITY)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bubble_is_twice_the_droplet_pressure() {
        // Deux interfaces → surpression double à γ et r identiques.
        let gamma = 0.0728_f64;
        let r = 1e-3_f64;
        assert_relative_eq!(
            laplace_bubble_pressure(gamma, r),
            2.0 * laplace_droplet_pressure(gamma, r),
            epsilon = 1e-12
        );
    }

    #[test]
    fn droplet_pressure_realistic_value() {
        // Goutte d'eau, γ = 0,0728 N/m, r = 1 mm :
        // Δp = 2·0,0728 / 1e-3 = 145,6 Pa.
        assert_relative_eq!(
            laplace_droplet_pressure(0.0728, 1e-3),
            145.6,
            epsilon = 1e-9
        );
    }

    #[test]
    fn pressure_inversely_proportional_to_radius() {
        // Diviser le rayon par deux double la surpression.
        let gamma = 0.05_f64;
        let p1 = laplace_droplet_pressure(gamma, 2e-3);
        let p2 = laplace_droplet_pressure(gamma, 1e-3);
        assert_relative_eq!(p2 / p1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn capillary_length_realistic_value() {
        // Eau : γ = 0,0728 N/m, ρ = 1000 kg/m³, g = 9,81 m/s².
        // l_c = √(0,0728 / (1000·9,81)) ≈ 2,72415e-3 m (~2,7 mm).
        let expected = (0.0728_f64 / (1000.0 * 9.81)).sqrt();
        assert_relative_eq!(
            laplace_capillary_length(0.0728, 1000.0),
            expected,
            epsilon = 1e-15
        );
        // Contrôle chiffré indépendant.
        assert_relative_eq!(
            laplace_capillary_length(0.0728, 1000.0),
            2.724_151e-3,
            epsilon = 1e-8
        );
    }

    #[test]
    fn capillary_length_scales_as_sqrt_gamma() {
        // Quadrupler γ double la longueur capillaire (dépendance en √γ).
        let l1 = laplace_capillary_length(0.02, 1000.0);
        let l2 = laplace_capillary_length(0.08, 1000.0);
        assert_relative_eq!(l2 / l1, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "rayon")]
    fn zero_radius_panics() {
        laplace_droplet_pressure(0.0728, 0.0);
    }
}
