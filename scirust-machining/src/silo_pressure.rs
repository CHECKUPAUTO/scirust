//! **Pression dans un silo** — équation de Janssen pour matériaux en vrac : le
//! frottement du produit sur les parois reporte une part croissante du poids sur
//! celles-ci, si bien que la pression verticale **sature** avec la profondeur au
//! lieu de croître linéairement comme dans un liquide.
//!
//! ```text
//! rayon hydraulique      R = A / U
//! pression verticale     p_v(z) = (ρ·g·R / (μ·K)) · (1 − exp(−μ·K·z / R))
//! pression pariétale      p_h(z) = K · p_v(z)
//! pression asymptotique  p_∞ = ρ·g·R / (μ·K)   (limite z → ∞)
//! ```
//!
//! `A` aire de la section droite du produit (m²), `U` périmètre en contact avec la
//! paroi (m), `R` rayon hydraulique (m), `ρ` masse volumique apparente du produit
//! en vrac (kg/m³), `g` accélération de la pesanteur (m/s²), `μ` coefficient de
//! frottement produit/paroi (sans unité), `K` rapport de pression latérale
//! p_h/p_v (sans unité), `z` profondeur mesurée depuis la surface libre (m),
//! `p_v`/`p_h`/`p_∞` pressions (Pa).
//!
//! **Convention** : unités SI (m, m², kg/m³, m/s², Pa) ; profondeur `z ≥ 0`
//! comptée vers le bas depuis la surface du produit ; silo à section constante.
//! **Limite honnête** : équation de Janssen (contrainte verticale supposée
//! uniforme sur la section, `μ`, `K` et `ρ` constants avec la profondeur). La masse
//! volumique apparente `ρ`, le coefficient de frottement pariétal `μ` et le rapport
//! de pression latérale `K` sont des **données du produit et du procédé fournies
//! par l'appelant** ; aucune valeur « par défaut » de matériau, de frottement ou de
//! remplissage n'est supposée.

/// Rayon hydraulique `R = A / U` (aire de section droite sur périmètre mouillé), en m.
///
/// Panique si `cross_section_area <= 0` ou `perimeter <= 0`.
pub fn silo_hydraulic_radius(cross_section_area: f64, perimeter: f64) -> f64 {
    assert!(
        cross_section_area > 0.0,
        "l'aire de section A doit être strictement positive"
    );
    assert!(
        perimeter > 0.0,
        "le périmètre U doit être strictement positif"
    );
    cross_section_area / perimeter
}

/// Pression verticale de Janssen à la profondeur `z` :
/// `p_v = (ρ·g·R / (μ·K)) · (1 − exp(−μ·K·z / R))`, en Pa.
///
/// Panique si `bulk_density <= 0`, `gravity <= 0`, `hydraulic_radius <= 0`,
/// `wall_friction_coefficient <= 0`, `lateral_ratio <= 0` ou `depth < 0`.
pub fn silo_janssen_vertical_pressure(
    bulk_density: f64,
    gravity: f64,
    hydraulic_radius: f64,
    wall_friction_coefficient: f64,
    lateral_ratio: f64,
    depth: f64,
) -> f64 {
    assert!(
        bulk_density > 0.0,
        "la masse volumique apparente ρ doit être strictement positive"
    );
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    assert!(
        hydraulic_radius > 0.0,
        "le rayon hydraulique R doit être strictement positif"
    );
    assert!(
        wall_friction_coefficient > 0.0,
        "le coefficient de frottement pariétal μ doit être strictement positif"
    );
    assert!(
        lateral_ratio > 0.0,
        "le rapport de pression latérale K doit être strictement positif"
    );
    assert!(depth >= 0.0, "la profondeur z doit être positive ou nulle");
    let asymptotic =
        bulk_density * gravity * hydraulic_radius / (wall_friction_coefficient * lateral_ratio);
    asymptotic
        * (1.0 - (-wall_friction_coefficient * lateral_ratio * depth / hydraulic_radius).exp())
}

/// Pression horizontale sur la paroi `p_h = K · p_v` (déduite de la pression
/// verticale par le rapport de pression latérale), en Pa.
///
/// Panique si `vertical_pressure < 0` ou `lateral_ratio <= 0`.
pub fn silo_janssen_wall_pressure(vertical_pressure: f64, lateral_ratio: f64) -> f64 {
    assert!(
        vertical_pressure >= 0.0,
        "la pression verticale p_v doit être positive ou nulle"
    );
    assert!(
        lateral_ratio > 0.0,
        "le rapport de pression latérale K doit être strictement positif"
    );
    lateral_ratio * vertical_pressure
}

/// Pression verticale limite en grande profondeur `p_∞ = ρ·g·R / (μ·K)`
/// (valeur asymptotique de [`silo_janssen_vertical_pressure`] quand `z → ∞`), en Pa.
///
/// Panique si `bulk_density <= 0`, `gravity <= 0`, `hydraulic_radius <= 0`,
/// `wall_friction_coefficient <= 0` ou `lateral_ratio <= 0`.
pub fn silo_asymptotic_pressure(
    bulk_density: f64,
    gravity: f64,
    hydraulic_radius: f64,
    wall_friction_coefficient: f64,
    lateral_ratio: f64,
) -> f64 {
    assert!(
        bulk_density > 0.0,
        "la masse volumique apparente ρ doit être strictement positive"
    );
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    assert!(
        hydraulic_radius > 0.0,
        "le rayon hydraulique R doit être strictement positif"
    );
    assert!(
        wall_friction_coefficient > 0.0,
        "le coefficient de frottement pariétal μ doit être strictement positif"
    );
    assert!(
        lateral_ratio > 0.0,
        "le rapport de pression latérale K doit être strictement positif"
    );
    bulk_density * gravity * hydraulic_radius / (wall_friction_coefficient * lateral_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hydraulic_radius_of_circular_section_is_diameter_over_four() {
        // Section circulaire de diamètre D : A = π·D²/4, U = π·D, donc R = D/4.
        use core::f64::consts::PI;
        let diameter = 4.0_f64;
        let area = PI * diameter * diameter / 4.0;
        let perimeter = PI * diameter;
        assert_relative_eq!(
            silo_hydraulic_radius(area, perimeter),
            diameter / 4.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn vertical_pressure_tends_to_asymptote() {
        // À grande profondeur, la pression verticale sature vers p_∞ = ρgR/(μK).
        let (rho, g, r, mu, k) = (800.0_f64, 9.81_f64, 1.0_f64, 0.4_f64, 0.5_f64);
        let asymptote = silo_asymptotic_pressure(rho, g, r, mu, k);
        // z très grand -> exp(−μKz/R) ≈ 0.
        let p_v = silo_janssen_vertical_pressure(rho, g, r, mu, k, 1.0e3_f64);
        assert_relative_eq!(p_v, asymptote, epsilon = 1e-6);
    }

    #[test]
    fn shallow_depth_recovers_hydrostatic_slope() {
        // Pour μKz/R ≪ 1 : 1 − exp(−x) ≈ x, donc p_v ≈ ρ·g·z (comportement liquide).
        let (rho, g, r, mu, k) = (750.0_f64, 9.81_f64, 1.2_f64, 0.45_f64, 0.55_f64);
        let z = 1.0e-4_f64;
        let p_v = silo_janssen_vertical_pressure(rho, g, r, mu, k, z);
        assert_relative_eq!(p_v, rho * g * z, epsilon = 1e-2);
    }

    #[test]
    fn wall_pressure_is_lateral_ratio_times_vertical() {
        // Identité p_h = K·p_v reliant les fonctions verticale et pariétale.
        let (rho, g, r, mu, k) = (900.0_f64, 9.81_f64, 0.8_f64, 0.5_f64, 0.6_f64);
        let z = 3.0_f64;
        let p_v = silo_janssen_vertical_pressure(rho, g, r, mu, k, z);
        let p_h = silo_janssen_wall_pressure(p_v, k);
        assert_relative_eq!(p_h, k * p_v, epsilon = 1e-9);
    }

    #[test]
    fn realistic_grain_silo_case() {
        // Silo circulaire D = 4 m -> R = 1 m ; produit ρ = 800 kg/m³, μ = 0,4,
        // K = 0,5, g = 9,81 m/s². p_∞ = 800·9,81·1/(0,4·0,5) = 39 240 Pa.
        let (rho, g, r, mu, k) = (800.0_f64, 9.81_f64, 1.0_f64, 0.4_f64, 0.5_f64);
        assert_relative_eq!(
            silo_asymptotic_pressure(rho, g, r, mu, k),
            39_240.0_f64,
            epsilon = 1e-6
        );
        // À z = 5 m : μKz/R = 0,2·5 = 1 ; 1 − e^(−1) = 0,632120558…
        // p_v = 39 240 · 0,632120558 = 24 804,41 Pa.
        let p_v = silo_janssen_vertical_pressure(rho, g, r, mu, k, 5.0_f64);
        assert_relative_eq!(p_v, 39_240.0_f64 * (1.0 - (-1.0_f64).exp()), epsilon = 1e-9);
        assert_relative_eq!(p_v, 24_804.41_f64, epsilon = 1e-1);
        // Pression pariétale correspondante p_h = K·p_v = 12 402,2 Pa.
        assert_relative_eq!(
            silo_janssen_wall_pressure(p_v, k),
            0.5 * p_v,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "strictement positif")]
    fn zero_perimeter_panics() {
        let _ = silo_hydraulic_radius(1.0_f64, 0.0_f64);
    }
}
