//! Clavette Woodruff (demi-lune) — vérification statique d'une liaison
//! arbre-moyeu : effort tangentiel, cisaillement de la clavette et pression de
//! matage sur la partie saillante.
//!
//! Le couple `T` (N·m) transmis par un arbre de diamètre `d` (mm) engendre à sa
//! surface un effort tangentiel, d'où les sollicitations de la clavette de
//! largeur `w` (épaisseur), de longueur portante `L` (corde) et de hauteur
//! saillante `h` (partie émergeant de l'arbre, en appui sur le moyeu) :
//!
//! ```text
//! F = 2·T / d                       (effort tangentiel)
//! τ = F / (w·L)   = 2·T / (d·w·L)   (cisaillement, section w·L)
//! p = F / (h·L)   = 2·T / (d·h·L)   (matage sur la hauteur saillante h)
//! ```
//!
//! Légende / unités :
//! - `torque` : couple transmis `T` (N·m)
//! - `shaft_diameter` : diamètre d'arbre `d` (mm)
//! - `key_width` : largeur (épaisseur) de la clavette `w` (mm)
//! - `key_length` : longueur portante (corde) `L` (mm)
//! - `key_height_above_shaft` : hauteur saillante en appui `h` (mm)
//! - `F` en N, `τ` et `p` en MPa (N/mm²) — les conversions N·m → N·mm sont
//!   internes.
//!
//! **Limite honnête** : modèle usuel de dimensionnement statique à répartition
//! uniforme. Il néglige la concentration de contrainte en fond de rainure (plus
//! sévère pour une rainure Woodruff, profonde), la fatigue et le partage réel de
//! charge. Les contraintes/pressions admissibles dépendent des matériaux et du
//! régime : elles sont FOURNIES par l'appelant, aucune valeur « par défaut »
//! n'est inventée ici. Complète [`crate::keys`] (clavette parallèle).

/// Effort tangentiel `F = 2·T/d` (N) à la surface d'un arbre de diamètre
/// `shaft_diameter` (mm) transmettant un couple `torque` (N·m).
///
/// Panique si `shaft_diameter <= 0`.
pub fn woodruff_tangential_force(torque_nm: f64, shaft_diameter_mm: f64) -> f64 {
    assert!(
        shaft_diameter_mm > 0.0,
        "le diamètre d'arbre doit être strictement positif"
    );
    2.0 * torque_nm * 1000.0 / shaft_diameter_mm
}

/// Contrainte de cisaillement de la clavette Woodruff
/// `τ = 2·T/(d·w·L) = F/(w·L)` (MPa), pour un couple `torque` (N·m), un
/// diamètre `shaft_diameter` (mm), une largeur `key_width` (mm) et une longueur
/// portante `key_length` (mm).
///
/// Panique si une dimension est non strictement positive.
pub fn woodruff_shear_stress(
    torque_nm: f64,
    shaft_diameter_mm: f64,
    key_width_mm: f64,
    key_length_mm: f64,
) -> f64 {
    assert!(
        key_width_mm > 0.0 && key_length_mm > 0.0,
        "largeur et longueur de clavette doivent être strictement positives"
    );
    woodruff_tangential_force(torque_nm, shaft_diameter_mm) / (key_width_mm * key_length_mm)
}

/// Pression de matage sur la hauteur saillante de la clavette Woodruff
/// `p = 2·T/(d·h·L) = F/(h·L)` (MPa), pour un couple `torque` (N·m), un
/// diamètre `shaft_diameter` (mm), une hauteur saillante `key_height_above_shaft`
/// (mm) et une longueur portante `key_length` (mm).
///
/// Panique si une dimension est non strictement positive.
pub fn woodruff_bearing_stress(
    torque_nm: f64,
    shaft_diameter_mm: f64,
    key_height_above_shaft_mm: f64,
    key_length_mm: f64,
) -> f64 {
    assert!(
        key_height_above_shaft_mm > 0.0 && key_length_mm > 0.0,
        "hauteur saillante et longueur de clavette doivent être strictement positives"
    );
    woodruff_tangential_force(torque_nm, shaft_diameter_mm)
        / (key_height_above_shaft_mm * key_length_mm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tangential_force_matches_closed_form() {
        // T=100 N·m, d=25 mm → F = 2·100·1000/25 = 8000 N.
        assert_relative_eq!(
            woodruff_tangential_force(100.0, 25.0),
            8_000.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn shear_stress_realistic_case() {
        // F=8000 N, w=6 mm, L=20 mm → τ = 8000/(6·20) = 66,666… MPa.
        // Vérifie aussi la forme fermée 2·T/(d·w·L).
        assert_relative_eq!(
            woodruff_shear_stress(100.0, 25.0, 6.0, 20.0),
            8_000.0 / 120.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            woodruff_shear_stress(100.0, 25.0, 6.0, 20.0),
            2.0 * 100.0 * 1000.0 / (25.0 * 6.0 * 20.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn bearing_stress_realistic_case() {
        // F=8000 N, h=3 mm, L=20 mm → p = 8000/(3·20) = 133,333… MPa.
        assert_relative_eq!(
            woodruff_bearing_stress(100.0, 25.0, 3.0, 20.0),
            8_000.0 / 60.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn bearing_over_shear_equals_width_over_height() {
        // τ = F/(w·L), p = F/(h·L) ⇒ p/τ = w/h, indépendant du chargement.
        let tau = woodruff_shear_stress(180.0, 32.0, 8.0, 24.0);
        let p = woodruff_bearing_stress(180.0, 32.0, 4.5, 24.0);
        assert_relative_eq!(p / tau, 8.0 / 4.5, epsilon = 1e-9);
    }

    #[test]
    fn stress_is_linear_in_torque() {
        // Doubler le couple double les contraintes (dépendance linéaire en T).
        let base = woodruff_shear_stress(90.0, 30.0, 5.0, 18.0);
        let doubled = woodruff_shear_stress(180.0, 30.0, 5.0, 18.0);
        assert_relative_eq!(doubled, 2.0 * base, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "largeur et longueur de clavette doivent être strictement positives")]
    fn zero_width_panics() {
        let _ = woodruff_shear_stress(100.0, 25.0, 0.0, 20.0);
    }
}
