//! Usinage — **perçage** : couple, puissance, vitesse de pénétration et effort
//! de poussée d'un foret hélicoïdal.
//!
//! ```text
//! couple           M = kc·f·D²/8
//! puissance (kW)   P = M·2π·N/60 / 1000
//! pénétration      vf = f·N              (mm/min)
//! poussée          Ff = kf·f·D/2
//! ```
//!
//! `kc` effort spécifique de coupe (N/mm²), `f` avance par tour (mm/tr), `D`
//! diamètre du foret (mm), `N` fréquence de rotation (tr/min), `M` couple (N·mm si
//! `kc` en N/mm² et dimensions en mm), `kf` effort spécifique d'avance (N/mm²)
//! pour la poussée.
//!
//! **Convention** : unités de fiche outil (mm, tr/min, N/mm²) ; le couple sort
//! en **N·mm**. **Limite honnête** : modèle d'ingénieur (couple depuis l'effort
//! spécifique) ; `kc`/`kf` sont des données du couple outil/matière fournies par
//! l'appelant. Foret standard à deux lèvres ; ne modélise pas l'âme (chisel edge)
//! en détail. Le temps de perçage est dans [`crate::time`].

use core::f64::consts::PI;

/// Couple de perçage `M = kc·f·D²/8` (N·mm avec `kc` en N/mm², `f`,`D` en mm).
pub fn drilling_torque(specific_cutting_force: f64, feed_per_rev: f64, diameter: f64) -> f64 {
    specific_cutting_force * feed_per_rev * diameter * diameter / 8.0
}

/// Puissance de perçage `P = M·2π·N/60` (W si `M` en N·m), ici rendue en **kW**
/// à partir d'un couple en **N·m**.
pub fn drilling_power_kw(torque_nm: f64, rpm: f64) -> f64 {
    torque_nm * 2.0 * PI * rpm / 60.0 / 1000.0
}

/// Vitesse de pénétration `vf = f·N` (mm/min).
pub fn penetration_rate(feed_per_rev: f64, rpm: f64) -> f64 {
    feed_per_rev * rpm
}

/// Effort de poussée `Ff = kf·f·D/2` (N avec `kf` en N/mm², `f`,`D` en mm).
pub fn drilling_thrust(specific_feed_force: f64, feed_per_rev: f64, diameter: f64) -> f64 {
    specific_feed_force * feed_per_rev * diameter / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn torque_scales_with_diameter_squared() {
        // M ∝ D² : doubler le diamètre quadruple le couple.
        let m1 = drilling_torque(2000.0, 0.2, 10.0);
        let m2 = drilling_torque(2000.0, 0.2, 20.0);
        assert_relative_eq!(m2 / m1, 4.0, epsilon = 1e-9);
        assert_relative_eq!(m1, 2000.0 * 0.2 * 100.0 / 8.0, epsilon = 1e-6);
    }

    #[test]
    fn power_from_torque_and_speed() {
        // M=5 N·m, N=1000 tr/min → P = 5·2π·1000/60/1000 ≈ 0,524 kW.
        assert_relative_eq!(
            drilling_power_kw(5.0, 1000.0),
            5.0 * 2.0 * PI * 1000.0 / 60.0 / 1000.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn penetration_rate_definition() {
        // f=0,2 mm/tr, N=800 tr/min → vf = 160 mm/min.
        assert_relative_eq!(penetration_rate(0.2, 800.0), 160.0, epsilon = 1e-9);
    }

    #[test]
    fn thrust_grows_with_feed() {
        // Ff ∝ f : doubler l'avance double la poussée.
        let f1 = drilling_thrust(1500.0, 0.1, 12.0);
        let f2 = drilling_thrust(1500.0, 0.2, 12.0);
        assert_relative_eq!(f2 / f1, 2.0, epsilon = 1e-9);
    }
}
