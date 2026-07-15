//! **Ascension capillaire** (loi de **Jurin**) — remontée d'un liquide mouillant
//! dans un tube cylindrique étroit sous l'effet de la tension superficielle.
//!
//! ```text
//! hauteur de Jurin       h = 2·γ·cos(θ) / (ρ·g·r)
//! saut de pression       Δp = 2·γ·cos(θ) / r
//! relation hauteur/saut  h = Δp / (ρ·g)
//! ```
//!
//! `h` hauteur d'ascension capillaire (m), `Δp` saut de pression de Laplace au
//! travers du ménisque (Pa), `γ` tension superficielle du liquide (N/m), `θ`
//! angle de contact liquide/paroi (rad), `ρ` masse volumique du liquide (kg/m³),
//! `g` accélération de la pesanteur (m/s²), `r` rayon intérieur du tube (m).
//! Une hauteur `h` **positive** correspond à un liquide **mouillant**
//! (`θ < π/2`, cos θ > 0) qui monte ; `h` négative à un liquide **non mouillant**
//! (`θ > π/2`, ex. mercure) qui redescend.
//!
//! **Convention** : SI ; angle de contact en radians. **Limite honnête** : tube
//! **cylindrique droit et étroit** (rayon très inférieur à la longueur
//! capillaire, ménisque supposé sphérique), régime **statique** à l'équilibre,
//! **évaporation et effets dynamiques négligés**. La mouillabilité est entièrement
//! portée par l'angle de contact `θ` fourni ; la tension superficielle `γ` et la
//! masse volumique `ρ` sont des **données de l'appelant** (elles dépendent du
//! couple liquide/solide et de la température) et ne sont jamais supposées.
//! L'accélération de la pesanteur est prise égale à [`CAPILLARY_GRAVITY`]
//! = 9,81 m/s² ; sous une autre gravité, adapter les données en conséquence.

/// Accélération de la pesanteur retenue par la loi de Jurin (m/s²).
pub const CAPILLARY_GRAVITY: f64 = 9.81;

/// Hauteur d'ascension capillaire de **Jurin** `h = 2·γ·cos(θ)/(ρ·g·r)` (m).
///
/// Hauteur d'équilibre atteinte par un liquide dans un tube cylindrique étroit.
/// Le signe suit `cos(θ)` : positif pour un liquide mouillant (montée), négatif
/// pour un liquide non mouillant (descente sous le niveau libre).
///
/// Panique si `surface_tension < 0`, `density <= 0` ou `tube_radius <= 0`.
pub fn capillary_height(
    surface_tension: f64,
    contact_angle_rad: f64,
    density: f64,
    tube_radius: f64,
) -> f64 {
    assert!(
        surface_tension >= 0.0,
        "la tension superficielle γ doit être ≥ 0"
    );
    assert!(density > 0.0, "la masse volumique ρ doit être > 0");
    assert!(tube_radius > 0.0, "le rayon du tube r doit être > 0");
    2.0 * surface_tension * contact_angle_rad.cos() / (density * CAPILLARY_GRAVITY * tube_radius)
}

/// Saut de pression capillaire de **Laplace** `Δp = 2·γ·cos(θ)/r` (Pa).
///
/// Différence de pression au travers du ménisque hémisphérique dans le tube ;
/// c'est le moteur de l'ascension. Positif pour un liquide mouillant.
///
/// Panique si `surface_tension < 0` ou `tube_radius <= 0`.
pub fn capillary_pressure(surface_tension: f64, contact_angle_rad: f64, tube_radius: f64) -> f64 {
    assert!(
        surface_tension >= 0.0,
        "la tension superficielle γ doit être ≥ 0"
    );
    assert!(tube_radius > 0.0, "le rayon du tube r doit être > 0");
    2.0 * surface_tension * contact_angle_rad.cos() / tube_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn height_equals_pressure_over_rho_g() {
        // Identité structurelle : h = Δp/(ρ·g) puisque les deux partagent 2·γ·cos θ.
        let gamma = 0.0728_f64;
        let theta = 0.3_f64;
        let rho = 998.0_f64;
        let r = 4.0e-4_f64;
        let h = capillary_height(gamma, theta, rho, r);
        let dp = capillary_pressure(gamma, theta, r);
        assert_relative_eq!(h, dp / (rho * CAPILLARY_GRAVITY), epsilon = 1e-12);
    }

    #[test]
    fn height_is_inversely_proportional_to_radius() {
        // h ∝ 1/r : doubler le rayon divise la hauteur par deux.
        let gamma = 0.05_f64;
        let theta = 0.2_f64;
        let rho = 1000.0_f64;
        let h1 = capillary_height(gamma, theta, rho, 1.0e-3);
        let h2 = capillary_height(gamma, theta, rho, 2.0e-3);
        assert_relative_eq!(h1, 2.0 * h2, epsilon = 1e-12);
    }

    #[test]
    fn non_wetting_liquid_gives_negative_height() {
        // Angle obtus (θ > π/2, ex. mercure) → cos θ < 0 → dépression capillaire.
        let theta = 2.0_f64; // ≈ 114,6°, cos < 0
        let h = capillary_height(0.485, theta, 13534.0, 5.0e-4);
        assert!(h < 0.0, "un liquide non mouillant doit redescendre");
        // À θ = π/2 exactement (cos = 0), aucune ascension.
        assert_relative_eq!(
            capillary_height(0.485, core::f64::consts::FRAC_PI_2, 13534.0, 5.0e-4),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn water_in_capillary_realistic_case() {
        // Eau à 20 °C : γ = 0,0728 N/m, θ = 0 (mouillage total), ρ = 998 kg/m³,
        // tube r = 0,5 mm.
        // h = 2·0,0728·1 / (998·9,81·5e-4)
        //   = 0,1456 / 4,895190 = 0,0297434 m ≈ 29,7 mm.
        let h = capillary_height(0.0728, 0.0, 998.0, 5.0e-4);
        assert_relative_eq!(h, 0.029_743_4, epsilon = 1e-6);
        // Saut de pression associé : Δp = 2·0,0728/5e-4 = 291,2 Pa.
        let dp = capillary_pressure(0.0728, 0.0, 5.0e-4);
        assert_relative_eq!(dp, 291.2, epsilon = 1e-9);
    }

    #[test]
    fn pressure_scales_linearly_with_surface_tension() {
        // Δp ∝ γ à géométrie et mouillage fixés.
        let theta = 0.4_f64;
        let r = 3.0e-4_f64;
        let dp1 = capillary_pressure(0.02, theta, r);
        let dp3 = capillary_pressure(0.06, theta, r);
        assert_relative_eq!(dp3, 3.0 * dp1, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le rayon du tube r doit être > 0")]
    fn zero_radius_panics() {
        capillary_height(0.0728, 0.0, 998.0, 0.0);
    }
}
