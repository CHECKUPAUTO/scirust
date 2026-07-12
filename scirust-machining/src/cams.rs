//! Cames — lois de mouvement du suiveur pour une came à disque, en montée d'une
//! course `h` sur un angle de came `β` : mouvement harmonique simple (MHS) et
//! cycloïdal, avec déplacement `s`, vitesse `v` et accélération `a`.
//!
//! ```text
//! MHS       : s = (h/2)·(1 − cos(π·θ/β))
//! Cycloïdal : s = h·(θ/β − sin(2π·θ/β)/(2π))
//! ```
//!
//! `θ` angle de came instantané (rad), `β` angle de montée (rad), `ω` vitesse de
//! rotation de la came (rad/s). La loi cycloïdale annule `v` **et** `a` aux
//! extrémités (démarrage/arrêt sans à-coup), au prix d'une accélération de
//! pointe plus élevée ; la loi MHS annule `v` aux extrémités mais pas `a`.
//!
//! **Convention** : `h` et `s` dans la même unité de longueur ; angles en rad ;
//! `v` en (longueur/s), `a` en (longueur/s²). **Limite honnête** : cinématique
//! du suiveur pour ces lois idéales, sans dynamique de contact, angle de
//! pression, ni rayon de courbure du profil — calculs de tracé distincts.

use core::f64::consts::PI;

fn check(beta_rad: f64) {
    assert!(
        beta_rad > 0.0,
        "l'angle de montée doit être strictement positif"
    );
}

/// Déplacement du suiveur — loi **MHS** : `s = (h/2)(1 − cos(π·θ/β))`.
pub fn shm_displacement(rise: f64, theta_rad: f64, beta_rad: f64) -> f64 {
    check(beta_rad);
    rise / 2.0 * (1.0 - (PI * theta_rad / beta_rad).cos())
}

/// Vitesse du suiveur — loi **MHS** : `v = (π·h·ω)/(2β)·sin(π·θ/β)`.
pub fn shm_velocity(rise: f64, theta_rad: f64, beta_rad: f64, omega_rad_s: f64) -> f64 {
    check(beta_rad);
    PI * rise * omega_rad_s / (2.0 * beta_rad) * (PI * theta_rad / beta_rad).sin()
}

/// Accélération du suiveur — loi **MHS** : `a = (π²·h·ω²)/(2β²)·cos(π·θ/β)`.
pub fn shm_acceleration(rise: f64, theta_rad: f64, beta_rad: f64, omega_rad_s: f64) -> f64 {
    check(beta_rad);
    PI * PI * rise * omega_rad_s * omega_rad_s / (2.0 * beta_rad * beta_rad)
        * (PI * theta_rad / beta_rad).cos()
}

/// Déplacement du suiveur — loi **cycloïdale** :
/// `s = h·(θ/β − sin(2π·θ/β)/(2π))`.
pub fn cycloidal_displacement(rise: f64, theta_rad: f64, beta_rad: f64) -> f64 {
    check(beta_rad);
    let r = theta_rad / beta_rad;
    rise * (r - (2.0 * PI * r).sin() / (2.0 * PI))
}

/// Vitesse du suiveur — loi **cycloïdale** :
/// `v = (h·ω/β)·(1 − cos(2π·θ/β))`.
pub fn cycloidal_velocity(rise: f64, theta_rad: f64, beta_rad: f64, omega_rad_s: f64) -> f64 {
    check(beta_rad);
    let r = theta_rad / beta_rad;
    rise * omega_rad_s / beta_rad * (1.0 - (2.0 * PI * r).cos())
}

/// Accélération du suiveur — loi **cycloïdale** :
/// `a = (2π·h·ω²/β²)·sin(2π·θ/β)`.
pub fn cycloidal_acceleration(rise: f64, theta_rad: f64, beta_rad: f64, omega_rad_s: f64) -> f64 {
    check(beta_rad);
    let r = theta_rad / beta_rad;
    2.0 * PI * rise * omega_rad_s * omega_rad_s / (beta_rad * beta_rad) * (2.0 * PI * r).sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shm_reaches_full_rise_at_beta() {
        let beta = PI; // 180°
        assert_relative_eq!(shm_displacement(10.0, 0.0, beta), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            shm_displacement(10.0, beta / 2.0, beta),
            5.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(shm_displacement(10.0, beta, beta), 10.0, epsilon = 1e-12);
        // vitesse nulle aux extrémités.
        assert_relative_eq!(shm_velocity(10.0, 0.0, beta, 5.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(shm_velocity(10.0, beta, beta, 5.0), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn cycloidal_zeroes_velocity_and_acceleration_at_ends() {
        let beta = PI;
        // course complète.
        assert_relative_eq!(
            cycloidal_displacement(10.0, beta, beta),
            10.0,
            epsilon = 1e-12
        );
        // v et a nuls aux deux extrémités (démarrage/arrêt doux).
        for &theta in &[0.0, beta]
        {
            assert_relative_eq!(
                cycloidal_velocity(10.0, theta, beta, 5.0),
                0.0,
                epsilon = 1e-9
            );
            assert_relative_eq!(
                cycloidal_acceleration(10.0, theta, beta, 5.0),
                0.0,
                epsilon = 1e-9
            );
        }
    }

    #[test]
    #[should_panic(expected = "angle de montée")]
    fn zero_beta_panics() {
        shm_displacement(10.0, 0.0, 0.0);
    }
}
