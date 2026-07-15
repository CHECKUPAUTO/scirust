//! **Dimensionnement d'arbre en torsion pure** — puissance transmise et
//! diamètre d'arbre plein circulaire pour un couple donné.
//!
//! ```text
//! puissance transmise    P = T·omega                     (arbre en rotation)
//! couple transmis        T = P / omega                   (réciproque)
//! diamètre requis        d = (16·T / (PI·tau))^(1/3)      (arbre plein, torsion pure)
//! couple admissible      T = PI·tau·d^3 / 16              (réciproque du diamètre)
//! ```
//!
//! `P` puissance mécanique (W), `T` couple de torsion (N·m), `omega` vitesse
//! angulaire (rad/s), `d` diamètre de l'arbre plein (m), `tau` contrainte de
//! cisaillement admissible du matériau (Pa). La relation diamètre/couple découle
//! de la torsion d'une section circulaire pleine : `tau_max = 16·T / (PI·d^3)`.
//!
//! **Convention** : SI cohérent (W, N·m, rad/s, m, Pa). **Limite honnête** :
//! l'arbre est supposé **plein, circulaire, en torsion pure**, matériau élastique
//! homogène ; la contrainte de cisaillement admissible `tau` est une **donnée
//! fournie par l'appelant** (limite du matériau divisée par le coefficient de
//! sécurité choisi), aucune valeur « par défaut » n'est inventée. On ne combine
//! **pas** flexion et torsion (dimensionnement ASME/Soderberg à la charge de
//! l'appelant) et on néglige les concentrations de contrainte (gorges, épaulements,
//! rainures de clavette). Voir [`crate::torque`] et [`crate::stress`].

use core::f64::consts::PI;

/// Puissance transmise `P = T·omega` (W), arbre tournant à `omega` (rad/s).
///
/// Panique si `torque < 0` ou `angular_speed_rad < 0`.
pub fn shaft_power_from_torque(torque: f64, angular_speed_rad: f64) -> f64 {
    assert!(torque >= 0.0, "le couple doit être positif ou nul");
    assert!(
        angular_speed_rad >= 0.0,
        "la vitesse angulaire doit être positive ou nulle"
    );
    torque * angular_speed_rad
}

/// Couple transmis `T = P / omega` (N·m) pour une puissance et une vitesse données.
///
/// Panique si `power < 0` ou `angular_speed_rad <= 0`.
pub fn shaft_torque_from_power(power: f64, angular_speed_rad: f64) -> f64 {
    assert!(power >= 0.0, "la puissance doit être positive ou nulle");
    assert!(
        angular_speed_rad > 0.0,
        "la vitesse angulaire doit être strictement positive"
    );
    power / angular_speed_rad
}

/// Diamètre requis `d = (16·T / (PI·tau))^(1/3)` (m), arbre plein en torsion pure.
///
/// Panique si `torque < 0` ou `allowable_shear <= 0`.
pub fn shaft_diameter_from_torque(torque: f64, allowable_shear: f64) -> f64 {
    assert!(torque >= 0.0, "le couple doit être positif ou nul");
    assert!(
        allowable_shear > 0.0,
        "la contrainte de cisaillement admissible doit être strictement positive"
    );
    (16.0_f64 * torque / (PI * allowable_shear)).cbrt()
}

/// Couple admissible `T = PI·tau·d^3 / 16` (N·m), arbre plein en torsion pure.
///
/// Panique si `diameter < 0` ou `allowable_shear < 0`.
pub fn shaft_torque_from_diameter(diameter: f64, allowable_shear: f64) -> f64 {
    assert!(diameter >= 0.0, "le diamètre doit être positif ou nul");
    assert!(
        allowable_shear >= 0.0,
        "la contrainte de cisaillement admissible doit être positive ou nulle"
    );
    PI * allowable_shear * diameter.powi(3) / 16.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn power_and_torque_are_reciprocal() {
        // T(P(T, omega), omega) doit rendre T : (T·omega) / omega = T.
        let torque = 250.0;
        let omega = 157.0;
        let power = shaft_power_from_torque(torque, omega);
        assert_relative_eq!(
            shaft_torque_from_power(power, omega),
            torque,
            epsilon = 1e-9
        );
    }

    #[test]
    fn diameter_and_torque_are_reciprocal() {
        // d(T(d, tau), tau) doit rendre d : la torsion pure est bijective en d^3.
        let diameter = 0.04;
        let tau = 55.0e6;
        let torque = shaft_torque_from_diameter(diameter, tau);
        assert_relative_eq!(
            shaft_diameter_from_torque(torque, tau),
            diameter,
            epsilon = 1e-12
        );
    }

    #[test]
    fn power_is_linear_in_torque() {
        // À vitesse fixe, doubler le couple double la puissance.
        let p1 = shaft_power_from_torque(100.0, 50.0 * PI);
        let p2 = shaft_power_from_torque(200.0, 50.0 * PI);
        assert_relative_eq!(p2 / p1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn diameter_scales_as_torque_cube_root() {
        // À tau fixe, multiplier le couple par 8 multiplie le diamètre par 2.
        let tau = 40.0e6;
        let d1 = shaft_diameter_from_torque(500.0, tau);
        let d2 = shaft_diameter_from_torque(8.0 * 500.0, tau);
        assert_relative_eq!(d2 / d1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_torque_from_diameter() {
        // Arbre plein d = 50 mm, tau_adm = 40 MPa (torsion pure) :
        // T = PI·40e6·(0,05)^3 / 16 = PI·40e6·1,25e-4 / 16 = PI·312,5 = 981,748 N·m.
        let torque = shaft_torque_from_diameter(0.05, 40.0e6);
        assert_relative_eq!(torque, 312.5 * PI, epsilon = 1e-9);
        // Réciproque : ce couple redonne bien d = 50 mm.
        assert_relative_eq!(
            shaft_diameter_from_torque(torque, 40.0e6),
            0.05,
            epsilon = 1e-12
        );
    }

    #[test]
    fn realistic_power_at_1500_rpm() {
        // T = 100 N·m à N = 1500 tr/min → omega = 1500·2·PI/60 = 50·PI rad/s.
        // P = 100·50·PI = 5000·PI = 15 708 W ≈ 15,7 kW.
        let omega = 50.0 * PI;
        assert_relative_eq!(
            shaft_power_from_torque(100.0, omega),
            5000.0 * PI,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(
        expected = "la contrainte de cisaillement admissible doit être strictement positive"
    )]
    fn zero_allowable_shear_panics() {
        shaft_diameter_from_torque(500.0, 0.0);
    }
}
