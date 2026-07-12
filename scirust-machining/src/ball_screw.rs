//! **Vis à billes** — conversion rotation ↔ translation et couple ↔ effort axial
//! d'un axe de machine.
//!
//! ```text
//! vitesse linéaire  v = N·p/60                         (N tr/min, p pas)
//! régime            N = 60·v/p
//! couple d'entraîn. C = F·p/(2π·η)                      (F effort axial à vaincre)
//! effort récupéré   F = 2π·η·C/p                        (couple → effort axial)
//! ```
//!
//! `p` pas (course par tour, m), `N` vitesse de rotation (tr/min), `v` vitesse
//! linéaire (m/s), `F` effort axial (N), `C` couple d'entraînement (N·m), `η`
//! rendement direct (vis à billes ≈ 0,9). `2π·η/p` est le rapport de
//! transformation couple → effort.
//!
//! **Convention** : SI (pas et course en m, vitesses en m/s et tr/min).
//! **Limite honnête** : régime **établi** (couple d'entraînement, hors
//! accélération) ; le rendement `η` est une donnée du couple vis/écrou fournie
//! par l'appelant. À la différence de [`crate::power_screws`] (modèle de
//! frottement par angle d'hélice, filets carré/trapézoïdal), ce module raisonne
//! directement en rendement catalogue. Voir [`crate::reflected_inertia`] pour
//! l'inertie ramenée et [`crate::motor_torque`] pour l'accélération.

use core::f64::consts::PI;

/// Vitesse linéaire `v = N·p/60` (m/s).
///
/// Panique si `lead <= 0`.
pub fn linear_speed(rotational_speed_rpm: f64, lead: f64) -> f64 {
    assert!(lead > 0.0, "le pas doit être strictement positif");
    rotational_speed_rpm * lead / 60.0
}

/// Vitesse de rotation `N = 60·v/p` (tr/min).
///
/// Panique si `lead <= 0`.
pub fn rotational_speed_rpm(linear_speed: f64, lead: f64) -> f64 {
    assert!(lead > 0.0, "le pas doit être strictement positif");
    60.0 * linear_speed / lead
}

/// Couple d'entraînement `C = F·p/(2π·η)` pour vaincre un effort axial.
///
/// Panique si `lead <= 0` ou `efficiency` hors `]0, 1]`.
pub fn drive_torque(axial_force: f64, lead: f64, efficiency: f64) -> f64 {
    assert!(lead > 0.0, "le pas doit être strictement positif");
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement doit être dans ]0, 1]"
    );
    axial_force * lead / (2.0 * PI * efficiency)
}

/// Effort axial `F = 2π·η·C/p` produit par un couple.
///
/// Panique si `lead <= 0` ou `efficiency` hors `]0, 1]`.
pub fn axial_force_from_torque(torque: f64, lead: f64, efficiency: f64) -> f64 {
    assert!(lead > 0.0, "le pas doit être strictement positif");
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement doit être dans ]0, 1]"
    );
    2.0 * PI * efficiency * torque / lead
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn speed_conversion_round_trips() {
        // Pas 10 mm à 1500 tr/min → 0,25 m/s ; conversion inverse redonne 1500.
        let v = linear_speed(1500.0, 0.010);
        assert_relative_eq!(v, 0.25, epsilon = 1e-12);
        assert_relative_eq!(rotational_speed_rpm(v, 0.010), 1500.0, epsilon = 1e-9);
    }

    #[test]
    fn torque_and_force_are_inverse_at_full_efficiency() {
        // À η=1, F(C(F)) redonne F.
        let f = axial_force_from_torque(drive_torque(5000.0, 0.010, 1.0), 0.010, 1.0);
        assert_relative_eq!(f, 5000.0, epsilon = 1e-9);
    }

    #[test]
    fn efficiency_increases_required_torque() {
        // Un rendement < 1 exige plus de couple pour le même effort.
        assert!(drive_torque(5000.0, 0.010, 0.9) > drive_torque(5000.0, 0.010, 1.0));
    }

    #[test]
    fn drive_torque_matches_formula() {
        // C = F·p/(2πη). F=5000 N, p=10 mm, η=0,9 → ≈ 8,84 N·m.
        let c = drive_torque(5000.0, 0.010, 0.9);
        assert_relative_eq!(c, 5000.0 * 0.010 / (2.0 * PI * 0.9), epsilon = 1e-9);
        assert!(c > 8.8 && c < 8.9);
    }

    #[test]
    #[should_panic(expected = "le pas")]
    fn zero_lead_panics() {
        linear_speed(1500.0, 0.0);
    }
}
