//! Mécanisme à **coulisse (Scotch yoke)** — cinématique de la coulisse en
//! mouvement harmonique simple (MHS) pur en fonction de l'angle de manivelle.
//!
//! ```text
//! déplacement    x = r·cosθ
//! vitesse        v = −r·ω·sinθ
//! accélération   a = −r·ω²·cosθ = −ω²·x
//! ```
//!
//! `r` rayon de manivelle (m), `θ` angle de manivelle mesuré depuis l'axe de la
//! coulisse (rad), `ω` vitesse angulaire de la manivelle (rad/s, supposée
//! constante). Le déplacement `x` est compté depuis le centre de la course ; la
//! course totale vaut `2r`. Résultats : `x` en m, `v` en m/s, `a` en m/s².
//!
//! **Convention** : unités SI cohérentes, angles en radians.
//!
//! **Limite honnête** : contrairement à la bielle-manivelle (cf.
//! [`crate::slider_crank`]), le mouvement de la coulisse est un MHS **exact**,
//! sans obliquité ni harmoniques supérieures ; liaison glissière parfaite,
//! `ω` constante, ni jeu ni flexibilité. Aucune constante physique ou de
//! matériau n'est fournie par défaut : `r` et `ω` sont donnés par l'appelant.

/// Déplacement de la coulisse `x = r·cosθ` (m), compté depuis le centre de course.
///
/// Panique si `crank_radius < 0`.
pub fn scotch_displacement(crank_radius: f64, crank_angle_rad: f64) -> f64 {
    assert!(
        crank_radius >= 0.0,
        "le rayon de manivelle doit être positif ou nul"
    );
    crank_radius * crank_angle_rad.cos()
}

/// Vitesse de la coulisse `v = −r·ω·sinθ` (m/s).
///
/// Panique si `crank_radius < 0`.
pub fn scotch_velocity(crank_radius: f64, omega_rad_s: f64, crank_angle_rad: f64) -> f64 {
    assert!(
        crank_radius >= 0.0,
        "le rayon de manivelle doit être positif ou nul"
    );
    -crank_radius * omega_rad_s * crank_angle_rad.sin()
}

/// Accélération de la coulisse `a = −r·ω²·cosθ` (m/s²), soit `a = −ω²·x`.
///
/// Panique si `crank_radius < 0`.
pub fn scotch_acceleration(crank_radius: f64, omega_rad_s: f64, crank_angle_rad: f64) -> f64 {
    assert!(
        crank_radius >= 0.0,
        "le rayon de manivelle doit être positif ou nul"
    );
    -crank_radius * omega_rad_s * omega_rad_s * crank_angle_rad.cos()
}

/// Course totale de la coulisse `2r` (m).
///
/// Panique si `crank_radius < 0`.
pub fn scotch_stroke(crank_radius: f64) -> f64 {
    assert!(
        crank_radius >= 0.0,
        "le rayon de manivelle doit être positif ou nul"
    );
    2.0 * crank_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn displacement_extremes_and_center() {
        // θ=0 → x=+r ; θ=π/2 → x=0 ; θ=π → x=−r.
        assert_relative_eq!(scotch_displacement(0.05, 0.0), 0.05, epsilon = 1e-12);
        assert_relative_eq!(scotch_displacement(0.05, FRAC_PI_2), 0.0, epsilon = 1e-12);
        assert_relative_eq!(scotch_displacement(0.05, PI), -0.05, epsilon = 1e-12);
    }

    #[test]
    fn velocity_zero_at_dead_centers_and_extreme_at_center() {
        // v s'annule aux points morts (θ=0, π) et vaut −rω à θ=π/2.
        assert_relative_eq!(scotch_velocity(0.05, 100.0, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(scotch_velocity(0.05, 100.0, PI), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            scotch_velocity(0.05, 100.0, FRAC_PI_2),
            -0.05 * 100.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn acceleration_is_minus_omega_squared_times_displacement() {
        // Identité du MHS : a = −ω²·x, quel que soit θ.
        let (r, w) = (0.05_f64, 100.0_f64);
        for &theta in &[0.3_f64, 1.1, 2.7, 4.9]
        {
            let x = scotch_displacement(r, theta);
            let a = scotch_acceleration(r, w, theta);
            assert_relative_eq!(a, -w * w * x, epsilon = 1e-9);
        }
    }

    #[test]
    fn energy_identity_holds() {
        // Identité MHS : v² + ω²·x² = (r·ω)² (amplitude de vitesse constante).
        let (r, w) = (0.08_f64, 30.0_f64);
        for &theta in &[0.0_f64, 0.7, 1.9, 3.5, 5.2]
        {
            let x = scotch_displacement(r, theta);
            let v = scotch_velocity(r, w, theta);
            assert_relative_eq!(v * v + w * w * x * x, (r * w).powi(2), epsilon = 1e-9);
        }
    }

    #[test]
    fn realistic_numeric_case() {
        // r=0,05 m ; ω=20 rad/s ; θ=60° (π/3).
        // cos60°=0,5 → x=0,05·0,5=0,025 m.
        // sin60°=√3/2≈0,8660254 → v=−0,05·20·0,8660254=−0,8660254 m/s.
        // a=−0,05·400·0,5=−10,0 m/s².
        let (r, w, theta) = (0.05_f64, 20.0_f64, PI / 3.0);
        assert_relative_eq!(scotch_displacement(r, theta), 0.025, epsilon = 1e-9);
        assert_relative_eq!(
            scotch_velocity(r, w, theta),
            -0.05 * 20.0 * (3.0_f64.sqrt() / 2.0),
            epsilon = 1e-9
        );
        assert_relative_eq!(scotch_acceleration(r, w, theta), -10.0, epsilon = 1e-9);
    }

    #[test]
    fn stroke_is_twice_radius() {
        assert_relative_eq!(scotch_stroke(0.05), 0.10, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le rayon de manivelle doit être positif ou nul")]
    fn negative_radius_panics() {
        scotch_displacement(-1.0, 0.0);
    }
}
