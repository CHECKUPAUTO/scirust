//! Ressort **spiral plan** (ressort moteur / de rappel) — ruban plat élastique
//! enroulé en spirale, sollicité en flexion pure : couple de rappel, contrainte de
//! flexion du ruban, énergie emmagasinée et longueur active pour une raideur visée.
//!
//! ```text
//! couple de rappel      M = E·b·t³·θ / (12·L)
//! contrainte de flexion σ = 6·M / (b·t²)
//! énergie emmagasinée   U = ½·M·θ           (ressort linéaire)
//! longueur active       L = E·b·t³ / (12·k) (k = M/θ, raideur angulaire visée)
//! ```
//!
//! `E` module de Young (Pa), `b` largeur du ruban (m), `t` épaisseur du ruban (m),
//! `L` longueur active (développée) du ruban (m), `θ` déflexion angulaire totale de
//! l'arbre (rad), `M` couple de rappel (N·m), `σ` contrainte de flexion maximale
//! dans le ruban (Pa), `k` raideur angulaire (N·m/rad).
//!
//! **Convention** : SI cohérent (m, rad, N·m, Pa). **Limite honnête** : ruban plat
//! élastique enroulé en spirale, **flexion pure**, comportement **linéaire** tant que
//! les spires ne se touchent pas (pas de tassement des spires). Le module de Young,
//! les dimensions du ruban et la déflexion sont **fournis par l'appelant** : ce module
//! ne compose que ces primitives et n'invente aucune constante de matériau ni de
//! procédé. Distinct de [`crate::torsion_springs`] (barre de torsion) et de
//! [`crate::constant_force_spring`] (effort de déroulement quasi constant).

/// Couple de rappel d'un ressort spiral plan :
/// `M = E·b·t³·θ / (12·L)` (N·m).
///
/// Panique si `youngs_modulus < 0`, si `thickness < 0`, si `width < 0`,
/// si `active_length <= 0` ou si `angular_deflection < 0`.
pub fn spiral_spring_torque(
    youngs_modulus: f64,
    thickness: f64,
    width: f64,
    active_length: f64,
    angular_deflection: f64,
) -> f64 {
    assert!(youngs_modulus >= 0.0, "youngs_modulus doit être ≥ 0");
    assert!(thickness >= 0.0, "thickness doit être ≥ 0");
    assert!(width >= 0.0, "width doit être ≥ 0");
    assert!(active_length > 0.0, "active_length doit être > 0");
    assert!(
        angular_deflection >= 0.0,
        "angular_deflection doit être ≥ 0"
    );
    youngs_modulus * width * thickness.powi(3) * angular_deflection / (12.0 * active_length)
}

/// Contrainte de flexion maximale dans le ruban d'un ressort spiral plan :
/// `σ = 6·M / (b·t²)` (Pa).
///
/// Panique si `torque < 0`, si `width <= 0` ou si `thickness <= 0`.
pub fn spiral_spring_bending_stress(torque: f64, width: f64, thickness: f64) -> f64 {
    assert!(torque >= 0.0, "torque doit être ≥ 0");
    assert!(width > 0.0, "width doit être > 0");
    assert!(thickness > 0.0, "thickness doit être > 0");
    6.0 * torque / (width * thickness.powi(2))
}

/// Énergie emmagasinée par un ressort spiral plan **linéaire** :
/// `U = ½·M·θ` (J).
///
/// Panique si `torque < 0` ou si `angular_deflection < 0`.
pub fn spiral_spring_stored_energy(torque: f64, angular_deflection: f64) -> f64 {
    assert!(torque >= 0.0, "torque doit être ≥ 0");
    assert!(
        angular_deflection >= 0.0,
        "angular_deflection doit être ≥ 0"
    );
    0.5 * torque * angular_deflection
}

/// Longueur active (développée) du ruban donnant une raideur angulaire visée :
/// `L = E·b·t³ / (12·k)` (m), où `k = M/θ` (N·m/rad).
///
/// Réciproque de [`spiral_spring_torque`] : à `L` et `θ` fixés,
/// `k = M/θ = E·b·t³ / (12·L)`, donc réinjecter cette raideur redonne `L`.
///
/// Panique si `youngs_modulus < 0`, si `thickness < 0`, si `width < 0`
/// ou si `torsional_rate <= 0`.
pub fn spiral_spring_active_length(
    youngs_modulus: f64,
    thickness: f64,
    width: f64,
    torsional_rate: f64,
) -> f64 {
    assert!(youngs_modulus >= 0.0, "youngs_modulus doit être ≥ 0");
    assert!(thickness >= 0.0, "thickness doit être ≥ 0");
    assert!(width >= 0.0, "width doit être ≥ 0");
    assert!(torsional_rate > 0.0, "torsional_rate doit être > 0");
    youngs_modulus * width * thickness.powi(3) / (12.0 * torsional_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn torque_matches_hand_computed_case() {
        // Cas chiffré réaliste (acier à ressort) calculé à la main :
        // E = 200e9, b = 0.02, t = 5e-4, L = 0.5, θ = 1.0 rad.
        // num = 200e9·0.02·(5e-4)³·1.0 = 4e9·1.25e-10 = 0.5
        // den = 12·0.5 = 6
        // M = 0.5 / 6 = 0.083333… N·m
        let m = spiral_spring_torque(200.0e9, 5.0e-4, 0.02, 0.5, 1.0);
        assert_relative_eq!(m, 0.5 / 6.0, max_relative = 1e-12);
    }

    #[test]
    fn torque_and_active_length_are_reciprocal() {
        // L → M → k = M/θ → L doit redonner la longueur de départ.
        let (e, t, b) = (210.0e9, 3.0e-4_f64, 0.015);
        let length = 0.8_f64;
        let theta = 2.5_f64;
        let m = spiral_spring_torque(e, t, b, length, theta);
        let rate = m / theta;
        assert_relative_eq!(
            spiral_spring_active_length(e, t, b, rate),
            length,
            max_relative = 1e-12
        );
    }

    #[test]
    fn torque_is_linear_in_deflection() {
        // M ∝ θ : doubler la déflexion double le couple.
        let base = spiral_spring_torque(200.0e9, 5.0e-4, 0.02, 0.5, 1.0);
        let doubled = spiral_spring_torque(200.0e9, 5.0e-4, 0.02, 0.5, 2.0);
        assert_relative_eq!(doubled, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn bending_stress_matches_hand_computed_case() {
        // σ = 6·M / (b·t²) avec M = 10, b = 0.02, t = 5e-4.
        // den = 0.02·(5e-4)² = 0.02·2.5e-7 = 5e-9
        // σ = 60 / 5e-9 = 1.2e10 Pa
        let sigma = spiral_spring_bending_stress(10.0, 0.02, 5.0e-4);
        assert_relative_eq!(sigma, 1.2e10, max_relative = 1e-12);
        // σ ∝ 1/t² : moitié de l'épaisseur quadruple la contrainte.
        let sigma_thin = spiral_spring_bending_stress(10.0, 0.02, 2.5e-4);
        assert_relative_eq!(sigma_thin, 4.0 * sigma, max_relative = 1e-12);
    }

    #[test]
    fn stored_energy_matches_half_torque_times_deflection() {
        // U = ½·M·θ avec M = 8, θ = 2.5 rad → U = 10 J.
        let u = spiral_spring_stored_energy(8.0, 2.5);
        assert_relative_eq!(u, 10.0, max_relative = 1e-12);
        // Cohérence : pour un ressort linéaire, U = ½·k·θ² = ½·(M/θ)·θ².
        let (e, t, b, length, theta) = (200.0e9, 5.0e-4, 0.02, 0.5, 1.3_f64);
        let m = spiral_spring_torque(e, t, b, length, theta);
        let rate = m / theta;
        assert_relative_eq!(
            spiral_spring_stored_energy(m, theta),
            0.5 * rate * theta.powi(2),
            max_relative = 1e-12
        );
    }

    #[test]
    fn torque_scales_with_thickness_cubed() {
        // M ∝ t³ : doubler l'épaisseur multiplie le couple par 8.
        let base = spiral_spring_torque(200.0e9, 5.0e-4, 0.02, 0.5, 1.0);
        let thick = spiral_spring_torque(200.0e9, 1.0e-3, 0.02, 0.5, 1.0);
        assert_relative_eq!(thick, 8.0 * base, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "active_length doit être > 0")]
    fn zero_active_length_panics() {
        spiral_spring_torque(200.0e9, 5.0e-4, 0.02, 0.0, 1.0);
    }
}
