//! **Mouvement d'un projectile** (balistique du vide) — portée, hauteur maximale,
//! temps de vol et angle optimal, sur sol horizontal et sans traînée.
//!
//! ```text
//! portée        R = v²·sin(2θ)/g
//! hauteur max.  H = (v·sin θ)²/(2·g)
//! temps de vol  t = 2·v·sin θ/g
//! angle optimal θ* = π/4              (portée maximale, altitude de départ = d'arrivée)
//! ```
//!
//! `v` vitesse initiale (m/s), `θ` angle de lancer par rapport à l'horizontale
//! (rad), `g` accélération de la pesanteur (m/s²), `R` portée horizontale (m),
//! `H` hauteur maximale au-dessus du point de lancer (m), `t` durée totale du vol
//! (s).
//!
//! **Convention** : SI ; angle en radians, mesuré depuis l'horizontale.
//! **Limite honnête** : traînée aérodynamique **négligée**, sol **horizontal**,
//! lancer et réception à la **même altitude** ; balistique idéale du vide. La
//! valeur de `g` (locale ou planétaire) est **fournie par l'appelant** : aucune
//! constante de pesanteur n'est supposée par défaut.

/// Portée horizontale `R = v²·sin(2θ)/g` (sol horizontal, altitudes égales).
///
/// Panique si `initial_speed < 0` ou `gravity <= 0`.
pub fn projectile_range(initial_speed: f64, launch_angle_rad: f64, gravity: f64) -> f64 {
    assert!(initial_speed >= 0.0, "v ≥ 0 requis");
    assert!(gravity > 0.0, "g > 0 requis");
    initial_speed * initial_speed * (2.0_f64 * launch_angle_rad).sin() / gravity
}

/// Hauteur maximale `H = (v·sin θ)²/(2·g)` au-dessus du point de lancer.
///
/// Panique si `initial_speed < 0` ou `gravity <= 0`.
pub fn projectile_max_height(initial_speed: f64, launch_angle_rad: f64, gravity: f64) -> f64 {
    assert!(initial_speed >= 0.0, "v ≥ 0 requis");
    assert!(gravity > 0.0, "g > 0 requis");
    let vertical = initial_speed * launch_angle_rad.sin();
    vertical * vertical / (2.0_f64 * gravity)
}

/// Temps de vol total `t = 2·v·sin θ/g` (retour à l'altitude de lancer).
///
/// Panique si `initial_speed < 0` ou `gravity <= 0`.
pub fn projectile_time_of_flight(initial_speed: f64, launch_angle_rad: f64, gravity: f64) -> f64 {
    assert!(initial_speed >= 0.0, "v ≥ 0 requis");
    assert!(gravity > 0.0, "g > 0 requis");
    2.0_f64 * initial_speed * launch_angle_rad.sin() / gravity
}

/// Angle de lancer maximisant la portée `θ* = π/4` (documentaire, sol horizontal).
///
/// Ne panique jamais.
pub fn projectile_optimal_angle() -> f64 {
    core::f64::consts::FRAC_PI_4
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_3, FRAC_PI_4, FRAC_PI_6};

    #[test]
    fn realistic_case() {
        // v = 10 m/s, θ = 30° (π/6), g = 10 m/s² (fourni).
        // R = 100·sin(60°)/10 = 10·sin(60°) = 5·√3 ≈ 8,660254 m.
        // H = (10·sin30°)²/(2·10) = 25/20 = 1,25 m.
        // t = 2·10·sin30°/10 = 1,0 s.
        assert_relative_eq!(
            projectile_range(10.0, FRAC_PI_6, 10.0),
            5.0 * 3.0_f64.sqrt(),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            projectile_max_height(10.0, FRAC_PI_6, 10.0),
            1.25,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            projectile_time_of_flight(10.0, FRAC_PI_6, 10.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn optimal_angle_maximizes_range() {
        // La portée à π/4 domine celle des autres angles (même v, même g).
        let r_opt = projectile_range(20.0, projectile_optimal_angle(), 9.81);
        assert_relative_eq!(projectile_optimal_angle(), FRAC_PI_4, epsilon = 1e-15);
        assert!(r_opt >= projectile_range(20.0, FRAC_PI_6, 9.81));
        assert!(r_opt >= projectile_range(20.0, FRAC_PI_3, 9.81));
    }

    #[test]
    fn complementary_angles_equal_range() {
        // Réciprocité : θ et (π/2 − θ) donnent la même portée (sin(2θ) = sin(π−2θ)).
        let a = 0.4_f64;
        let complement = core::f64::consts::FRAC_PI_2 - a;
        assert_relative_eq!(
            projectile_range(15.0, a, 9.81),
            projectile_range(15.0, complement, 9.81),
            epsilon = 1e-12
        );
    }

    #[test]
    fn range_equals_horizontal_speed_times_flight_time() {
        // Identité : R = (v·cos θ)·t, la portée est la distance horizontale parcourue.
        let (v, angle, g) = (18.0_f64, 0.7_f64, 9.81_f64);
        let horizontal = v * angle.cos();
        assert_relative_eq!(
            projectile_range(v, angle, g),
            horizontal * projectile_time_of_flight(v, angle, g),
            epsilon = 1e-12
        );
    }

    #[test]
    fn height_relates_to_flight_time() {
        // Identité : H = g·t²/8 (t = 2·v·sin θ/g ⇒ H = v²·sin²θ/(2g)).
        let (v, angle, g) = (22.0_f64, 0.9_f64, 9.81_f64);
        let t = projectile_time_of_flight(v, angle, g);
        assert_relative_eq!(
            projectile_max_height(v, angle, g),
            g * t * t / 8.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn horizontal_launch_has_zero_height_and_range() {
        // θ = 0 : ni hauteur ni portée (pas de composante verticale).
        assert_relative_eq!(projectile_max_height(30.0, 0.0, 9.81), 0.0, epsilon = 1e-12);
        assert_relative_eq!(projectile_range(30.0, 0.0, 9.81), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "g > 0")]
    fn zero_gravity_panics() {
        projectile_range(20.0, FRAC_PI_4, 0.0);
    }
}
