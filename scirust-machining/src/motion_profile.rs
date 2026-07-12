//! **Profil de mouvement** trapézoïdal (ou triangulaire) d'un axe — phases
//! d'accélération, de croisière et de décélération d'un déplacement point à point.
//!
//! ```text
//! temps d'accél.   t_a = v/a
//! distance d'accél. d_a = v²/(2a)                       (idem en décélération)
//! triangulaire si  d < v²/a          (pas de palier de croisière)
//! crête triangul.  v_pk = √(a·d)                        (profil symétrique)
//! temps total trap. T = d/v + v/a                       (si d ≥ v²/a)
//! ```
//!
//! `d` déplacement (m), `v` vitesse de croisière visée (m/s), `a` accélération
//! (m/s²), `t_a` durée d'une rampe, `v_pk` vitesse de crête d'un profil
//! triangulaire (croisière jamais atteinte), `T` durée totale.
//!
//! **Convention** : SI ; accélération et décélération **symétriques** (même `a`).
//! **Limite honnête** : profil **trapézoïdal** (jerk infini aux transitions) ;
//! pour un profil en S (jerk limité), ces temps sont des bornes basses. Si le
//! déplacement est trop court pour atteindre `v`, le profil devient
//! **triangulaire** : utiliser [`triangular_peak_velocity`] et
//! [`is_triangular`].

/// Durée d'une rampe d'accélération `t_a = v/a`.
///
/// Panique si `acceleration <= 0` ou `velocity < 0`.
pub fn accel_time(velocity: f64, acceleration: f64) -> f64 {
    assert!(
        acceleration > 0.0 && velocity >= 0.0,
        "a > 0 et v ≥ 0 requis"
    );
    velocity / acceleration
}

/// Distance parcourue pendant une rampe `d_a = v²/(2a)`.
///
/// Panique si `acceleration <= 0` ou `velocity < 0`.
pub fn accel_distance(velocity: f64, acceleration: f64) -> f64 {
    assert!(
        acceleration > 0.0 && velocity >= 0.0,
        "a > 0 et v ≥ 0 requis"
    );
    velocity * velocity / (2.0 * acceleration)
}

/// Vrai si le déplacement est **triangulaire** (croisière jamais atteinte) :
/// `d < v²/a`.
///
/// Panique si `acceleration <= 0`, `velocity <= 0` ou `distance < 0`.
pub fn is_triangular(distance: f64, velocity: f64, acceleration: f64) -> bool {
    assert!(
        acceleration > 0.0 && velocity > 0.0 && distance >= 0.0,
        "a, v > 0 et d ≥ 0 requis"
    );
    distance < velocity * velocity / acceleration
}

/// Vitesse de crête d'un profil **triangulaire** `v_pk = √(a·d)`.
///
/// Panique si `acceleration <= 0` ou `distance < 0`.
pub fn triangular_peak_velocity(distance: f64, acceleration: f64) -> f64 {
    assert!(
        acceleration > 0.0 && distance >= 0.0,
        "a > 0 et d ≥ 0 requis"
    );
    (acceleration * distance).sqrt()
}

/// Durée totale d'un profil **trapézoïdal** `T = d/v + v/a`.
///
/// Panique si `acceleration <= 0`, `velocity <= 0`, `distance < 0`, ou si le
/// déplacement est en réalité triangulaire (`d < v²/a`).
pub fn trapezoidal_total_time(distance: f64, velocity: f64, acceleration: f64) -> f64 {
    assert!(
        acceleration > 0.0 && velocity > 0.0 && distance >= 0.0,
        "a, v > 0 et d ≥ 0 requis"
    );
    assert!(
        distance >= velocity * velocity / acceleration,
        "déplacement triangulaire : la vitesse de croisière n'est pas atteinte"
    );
    distance / velocity + velocity / acceleration
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ramp_time_and_distance() {
        // v=0,5 m/s, a=5 m/s² → t_a=0,1 s, d_a=0,025 m.
        assert_relative_eq!(accel_time(0.5, 5.0), 0.1, epsilon = 1e-12);
        assert_relative_eq!(accel_distance(0.5, 5.0), 0.025, epsilon = 1e-12);
    }

    #[test]
    fn long_move_is_trapezoidal() {
        // d=1 m ≥ v²/a = 0,05 m → trapézoïdal.
        assert!(!is_triangular(1.0, 0.5, 5.0));
        // Durée = 1/0,5 + 0,5/5 = 2,1 s.
        assert_relative_eq!(trapezoidal_total_time(1.0, 0.5, 5.0), 2.1, epsilon = 1e-12);
    }

    #[test]
    fn short_move_is_triangular() {
        // d=0,02 m < v²/a = 0,05 m → triangulaire.
        assert!(is_triangular(0.02, 0.5, 5.0));
        // v_pk = √(5·0,02) = √0,1 ≈ 0,316 m/s < 0,5 visé.
        let vpk = triangular_peak_velocity(0.02, 5.0);
        assert_relative_eq!(vpk, (5.0_f64 * 0.02).sqrt(), epsilon = 1e-12);
        assert!(vpk < 0.5);
    }

    #[test]
    fn triangular_peak_reaches_exactly_v_at_threshold() {
        // Au seuil d = v²/a, la crête triangulaire égale la vitesse de croisière.
        let (v, a) = (0.5, 5.0);
        let d = v * v / a;
        assert_relative_eq!(triangular_peak_velocity(d, a), v, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "triangulaire")]
    fn trapezoidal_on_short_move_panics() {
        trapezoidal_total_time(0.02, 0.5, 5.0);
    }
}
