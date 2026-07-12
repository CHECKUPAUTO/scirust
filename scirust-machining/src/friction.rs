//! Frottement sec de **Coulomb** — lois d'adhérence/glissement, angle et cône
//! d'adhérence, et arc-boutement (auto-blocage) sur plan incliné.
//!
//! Entre deux solides en contact, l'effort tangentiel `T` transmissible est
//! borné par la réaction normale `N` :
//!
//! ```text
//! adhérence (repos) :  |T| ≤ μs·N
//! glissement         :  T = μc·N   (opposé au mouvement)
//! ```
//!
//! `μs` coefficient d'adhérence (statique), `μc` coefficient de frottement de
//! glissement (dynamique), en général `μc ≤ μs`. L'**angle d'adhérence** (ou de
//! frottement) `φ = arctan(μ)` définit le **cône d'adhérence** : à l'équilibre,
//! la réaction de contact reste à l'intérieur d'un cône de demi-angle `φ` autour
//! de la normale. Un solide sur un plan incliné d'angle `α` reste immobile
//! (arc-boutement) tant que `α ≤ φ`, c.-à-d. `tan α ≤ μs` ; l'**angle de repos**
//! (glissement imminent) vaut donc `arctan(μs)`.
//!
//! **Convention** : efforts en N (ou toute unité cohérente), angles en degrés,
//! coefficients sans dimension.
//!
//! **Limite honnête** : modèle de Coulomb idéalisé — coefficient constant,
//! indépendant de la vitesse, de la surface de contact et de l'état de surface.
//! Il ne rend compte ni du frottement visqueux/de Stribeck, ni du frottement de
//! roulement, ni de l'adhérence dépendant du temps ; `μs` et `μc` sont des
//! données du couple de matériaux fournies par l'appelant.

/// Angle d'adhérence (ou de frottement) `φ = arctan(μ)` en **degrés**.
pub fn friction_angle_deg(mu: f64) -> f64 {
    mu.atan().to_degrees()
}

/// Effort tangentiel maximal d'adhérence `T_max = μs·N` (N).
///
/// Panique si `normal < 0`.
pub fn max_static_friction(mu_s: f64, normal_n: f64) -> f64 {
    assert!(
        normal_n >= 0.0,
        "la réaction normale doit être positive ou nulle"
    );
    mu_s * normal_n
}

/// Effort de frottement de glissement `T = μc·N` (N), opposé au mouvement.
///
/// Panique si `normal < 0`.
pub fn kinetic_friction(mu_c: f64, normal_n: f64) -> f64 {
    assert!(
        normal_n >= 0.0,
        "la réaction normale doit être positive ou nulle"
    );
    mu_c * normal_n
}

/// `true` si le contact **glisse** : l'effort tangentiel demandé `tangential`
/// dépasse l'adhérence disponible `μs·N`.
///
/// Panique si `normal < 0`.
pub fn is_sliding(tangential_n: f64, normal_n: f64, mu_s: f64) -> bool {
    tangential_n.abs() > max_static_friction(mu_s, normal_n)
}

/// `true` si la direction de la réaction (inclinée de `force_angle_from_normal`
/// degrés par rapport à la normale) reste **dans le cône d'adhérence**, donc
/// l'équilibre est possible sans glissement : `tan(angle) ≤ μs`.
///
/// Panique si l'angle n'est pas dans `[0°, 90°[`.
pub fn within_adhesion_cone(force_angle_from_normal_deg: f64, mu_s: f64) -> bool {
    assert!(
        (0.0..90.0).contains(&force_angle_from_normal_deg),
        "l'angle par rapport à la normale doit être dans [0°, 90°["
    );
    force_angle_from_normal_deg.to_radians().tan() <= mu_s
}

/// Angle de repos (glissement imminent sur plan incliné) `= arctan(μs)` en
/// **degrés** — angle d'inclinaison maximal avant glissement. Égal à l'angle
/// d'adhérence.
pub fn angle_of_repose_deg(mu_s: f64) -> f64 {
    friction_angle_deg(mu_s)
}

/// Arc-boutement : `true` si un solide posé sur un plan incliné d'angle
/// `incline_angle` (degrés) **reste immobile** (auto-blocage), c.-à-d.
/// `tan α ≤ μs`.
///
/// Panique si l'angle n'est pas dans `[0°, 90°[`.
pub fn incline_self_locking(incline_angle_deg: f64, mu_s: f64) -> bool {
    assert!(
        (0.0..90.0).contains(&incline_angle_deg),
        "l'angle d'inclinaison doit être dans [0°, 90°["
    );
    incline_angle_deg.to_radians().tan() <= mu_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn friction_angle_matches_arctan() {
        // μ=0,3 → φ ≈ 16,699°.
        assert_relative_eq!(friction_angle_deg(0.3), 16.699, epsilon = 1e-3);
    }

    #[test]
    fn max_static_and_kinetic_scale_with_normal() {
        assert_relative_eq!(max_static_friction(0.3, 100.0), 30.0, epsilon = 1e-9);
        assert_relative_eq!(kinetic_friction(0.25, 100.0), 25.0, epsilon = 1e-9);
    }

    #[test]
    fn sliding_starts_above_the_adhesion_limit() {
        // N=100, μs=0,3 → limite 30 N.
        assert!(!is_sliding(25.0, 100.0, 0.3)); // en-deçà : adhérence
        assert!(is_sliding(35.0, 100.0, 0.3)); // au-delà : glissement
    }

    #[test]
    fn adhesion_cone_holds_below_the_friction_angle() {
        // μs=0,3 → φ≈16,7°. À 10° on tient, à 20° on glisse.
        assert!(within_adhesion_cone(10.0, 0.3));
        assert!(!within_adhesion_cone(20.0, 0.3));
        // exactement à l'angle d'adhérence : à la limite (tan φ = μs) → tient.
        assert!(within_adhesion_cone(friction_angle_deg(0.3) - 1e-6, 0.3));
    }

    #[test]
    fn angle_of_repose_equals_friction_angle() {
        assert_relative_eq!(
            angle_of_repose_deg(0.5),
            friction_angle_deg(0.5),
            epsilon = 1e-12
        );
        // arctan(1) = 45°.
        assert_relative_eq!(angle_of_repose_deg(1.0), 45.0, epsilon = 1e-9);
    }

    #[test]
    fn incline_self_locks_below_the_repose_angle() {
        // μs=0,3 (repos ≈16,7°). À 15° ça tient, à 20° ça glisse.
        assert!(incline_self_locking(15.0, 0.3));
        assert!(!incline_self_locking(20.0, 0.3));
    }

    #[test]
    #[should_panic(expected = "réaction normale")]
    fn negative_normal_panics() {
        max_static_friction(0.3, -1.0);
    }
}
