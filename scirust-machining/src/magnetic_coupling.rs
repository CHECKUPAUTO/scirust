//! **Accouplement magnétique synchrone** (transmission sans contact) — couple
//! transmis en fonction de l'angle de décalage entre aimants, pas polaire
//! angulaire, condition de synchronisme et puissance transmise en régime
//! permanent, à partir du couple maximal de décrochage fourni.
//!
//! ```text
//! couple transmis     T   = T_max·sin(θ)
//! pas polaire         θ_p = π / p
//! synchronisme        |θ| ≤ π/2
//! puissance transmise P   = T·ω
//! ```
//!
//! `T_max` couple maximal de décrochage (« pull-out », N·m), `θ` angle de charge
//! entre le rotor menant et le rotor mené (rad, décalage des aimants), `T` couple
//! effectivement transmis (N·m), `p` nombre de paires de pôles (sans unité,
//! entier > 0), `θ_p` pas polaire angulaire (rad), `ω` vitesse angulaire de
//! rotation (rad/s), `P` puissance transmise (W).
//!
//! **Convention** : unités SI cohérentes. Le couple est maximal à `θ = π/2`
//! (position de décrochage) ; en deçà l'accouplement est synchrone et transmet
//! `T = T_max·sin(θ)`. Au-delà de `π/2` un pôle « glisse » : le couple chute et
//! l'accouplement patine (perte de synchronisme).
//!
//! **Limite honnête** : accouplement **synchrone** à aimants permanents, en
//! **régime permanent**. Le couple maximal de décrochage `T_max` est une donnée
//! **fournie par l'appelant** (il dépend des aimants, de l'entrefer et de la
//! barrière de séparation) ; aucune valeur « par défaut » n'est supposée. Ce
//! modèle **ne décrit pas** les courants de Foucault d'un accouplement
//! **asynchrone** (à hystérésis ou à conducteur), ni le transitoire de
//! décrochage lui-même.

use core::f64::consts::PI;

/// Couple transmis `T = T_max·sin(θ)` par un accouplement magnétique synchrone.
///
/// Le couple est maximal à `θ = π/2` (décrochage) ; au-delà l'accouplement
/// patine et le modèle synchrone n'est plus valable.
///
/// Panique si `max_torque < 0` ou si `load_angle_rad` n'est pas fini.
pub fn magcoup_torque(max_torque: f64, load_angle_rad: f64) -> f64 {
    assert!(
        max_torque >= 0.0,
        "le couple maximal de décrochage T_max ne peut pas être négatif"
    );
    assert!(
        load_angle_rad.is_finite(),
        "l'angle de charge θ doit être un nombre fini"
    );
    max_torque * load_angle_rad.sin()
}

/// Pas polaire angulaire `θ_p = π / p` (décrochage atteint au demi-pas).
///
/// Panique si `pole_pairs <= 0` ou si `pole_pairs` n'est pas fini.
pub fn magcoup_pole_pitch_angle(pole_pairs: f64) -> f64 {
    assert!(
        pole_pairs > 0.0,
        "le nombre de paires de pôles p doit être strictement positif"
    );
    assert!(
        pole_pairs.is_finite(),
        "le nombre de paires de pôles p doit être un nombre fini"
    );
    PI / pole_pairs
}

/// Indique si l'accouplement reste synchrone `|θ| ≤ π/2` (couple transmissible
/// sans décrochage).
///
/// Panique si `load_angle_rad` n'est pas fini.
pub fn magcoup_is_synchronized(load_angle_rad: f64) -> bool {
    assert!(
        load_angle_rad.is_finite(),
        "l'angle de charge θ doit être un nombre fini"
    );
    load_angle_rad.abs() <= PI / 2.0
}

/// Puissance transmise `P = T·ω` en régime permanent.
///
/// Panique si `torque < 0` ou si `angular_speed_rad < 0`.
pub fn magcoup_transmitted_power(torque: f64, angular_speed_rad: f64) -> f64 {
    assert!(
        torque >= 0.0,
        "le couple transmis T ne peut pas être négatif"
    );
    assert!(
        angular_speed_rad >= 0.0,
        "la vitesse angulaire ω ne peut pas être négative"
    );
    torque * angular_speed_rad
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn torque_is_maximal_at_pull_out_angle() {
        // Cas limite θ = π/2 : sin(π/2) = 1 donc T = T_max (décrochage).
        let max_torque = 40.0;
        assert_relative_eq!(
            magcoup_torque(max_torque, PI / 2.0),
            max_torque,
            epsilon = 1e-12
        );
    }

    #[test]
    fn torque_is_zero_at_aligned_magnets() {
        // Cas limite θ = 0 : aimants alignés, aucun couple transmis.
        assert_relative_eq!(magcoup_torque(40.0, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn torque_is_proportional_to_max_torque() {
        // T ∝ T_max à angle fixé : doubler T_max double le couple transmis.
        let angle = PI / 5.0;
        let t1 = magcoup_torque(30.0, angle);
        let t2 = magcoup_torque(60.0, angle);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-12);
    }

    #[test]
    fn pole_pitch_times_pole_pairs_equals_pi() {
        // Identité θ_p·p = π : le pas polaire couvre π/p radians.
        let p = 4.0;
        assert_relative_eq!(magcoup_pole_pitch_angle(p) * p, PI, epsilon = 1e-12);
    }

    #[test]
    fn synchronism_boundary_is_half_pi() {
        // Synchronisme conservé jusqu'à π/2 inclus, perdu au-delà.
        assert!(magcoup_is_synchronized(PI / 3.0));
        assert!(magcoup_is_synchronized(-PI / 3.0));
        assert!(magcoup_is_synchronized(PI / 2.0));
        assert!(!magcoup_is_synchronized(2.0 * PI / 3.0));
    }

    #[test]
    fn realistic_torque_and_power_case() {
        // T_max = 40 N·m ; θ = π/6 (sin = 0,5) ⇒ T = 20 N·m.
        // ω = 100 rad/s ⇒ P = T·ω = 20·100 = 2000 W.
        let torque = magcoup_torque(40.0, PI / 6.0);
        assert_relative_eq!(torque, 20.0, epsilon = 1e-9);
        let power = magcoup_transmitted_power(torque, 100.0);
        assert_relative_eq!(power, 2000.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le couple maximal de décrochage T_max ne peut pas être négatif")]
    fn negative_max_torque_panics() {
        magcoup_torque(-1.0, PI / 4.0);
    }
}
