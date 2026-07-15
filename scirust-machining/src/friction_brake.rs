//! **Frein à sabot/bloc** — couple de freinage d'un ou deux sabots serrés sur un
//! tambour, puissance dissipée par frottement et temps d'arrêt d'un rotor.
//!
//! ```text
//! couple (un bloc)      T  = μ·N·r
//! couple (deux sabots)  T2 = 2·μ·N·r
//! puissance dissipée    P  = T·ω
//! temps d'arrêt         t  = I·ω/T
//! ```
//!
//! `μ` coefficient de frottement sabot/tambour (sans dimension), `N` effort
//! normal presseur du sabot sur le tambour (N), `r` rayon du tambour (m), `T`
//! couple de freinage (N·m), `ω` vitesse angulaire du tambour (rad/s), `P`
//! puissance dissipée par frottement (W), `I` moment d'inertie du rotor à freiner
//! (kg·m²), `t` temps d'arrêt sous couple constant (s).
//!
//! **Convention** : SI. **Limite honnête** : le coefficient de frottement `μ` est
//! **fourni par l'appelant** et supposé **constant** (indépendant de la vitesse et
//! de la température) ; on suppose une **pression de contact uniforme** sur le
//! sabot, un couple de freinage **constant** pendant l'arrêt, et l'effet
//! d'**auto-serrage** (moment de l'effort de frottement sur l'articulation du
//! sabot) est **négligé** — la formule à deux sabots suppose deux blocs opposés
//! identiques sans distinction serrant/desserrant. Aucune valeur de `μ`, de
//! matériau ou de procédé n'est inventée. Distinct de [`crate::brake_thermal`]
//! (échauffement du disque) et de [`crate::brakes`] (couple de freinage général).

/// Couple de freinage d'un sabot unique `T = μ·N·r`.
///
/// Panique si `friction_coefficient < 0`, `normal_force < 0` ou `drum_radius < 0`.
pub fn friction_brake_torque(
    friction_coefficient: f64,
    normal_force: f64,
    drum_radius: f64,
) -> f64 {
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement μ doit être positif"
    );
    assert!(normal_force >= 0.0, "l'effort normal N doit être positif");
    assert!(
        drum_radius >= 0.0,
        "le rayon du tambour r doit être positif"
    );
    friction_coefficient * normal_force * drum_radius
}

/// Couple de freinage de deux sabots opposés identiques `T2 = 2·μ·N·r`.
///
/// Panique si `friction_coefficient < 0`, `normal_force < 0` ou `drum_radius < 0`.
pub fn friction_brake_double_shoe_torque(
    friction_coefficient: f64,
    normal_force: f64,
    drum_radius: f64,
) -> f64 {
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement μ doit être positif"
    );
    assert!(normal_force >= 0.0, "l'effort normal N doit être positif");
    assert!(
        drum_radius >= 0.0,
        "le rayon du tambour r doit être positif"
    );
    2.0 * friction_coefficient * normal_force * drum_radius
}

/// Puissance dissipée par frottement `P = T·ω`.
///
/// Panique si `braking_torque < 0` ou `angular_speed_rad < 0`.
pub fn friction_brake_heat_power(braking_torque: f64, angular_speed_rad: f64) -> f64 {
    assert!(
        braking_torque >= 0.0,
        "le couple de freinage T doit être positif"
    );
    assert!(
        angular_speed_rad >= 0.0,
        "la vitesse angulaire ω doit être positive"
    );
    braking_torque * angular_speed_rad
}

/// Temps d'arrêt d'un rotor sous couple de freinage constant `t = I·ω/T`.
///
/// Panique si `inertia < 0`, `angular_speed_rad < 0` ou `braking_torque <= 0`.
pub fn friction_brake_stopping_time(
    inertia: f64,
    angular_speed_rad: f64,
    braking_torque: f64,
) -> f64 {
    assert!(inertia >= 0.0, "le moment d'inertie I doit être positif");
    assert!(
        angular_speed_rad >= 0.0,
        "la vitesse angulaire ω doit être positive"
    );
    assert!(
        braking_torque > 0.0,
        "le couple de freinage T doit être strictement positif (dénominateur non nul)"
    );
    inertia * angular_speed_rad / braking_torque
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn double_shoe_is_twice_single_shoe() {
        // Deux sabots opposés identiques doublent le couple d'un sabot seul.
        let single = friction_brake_torque(0.35, 800.0, 0.15);
        let double = friction_brake_double_shoe_torque(0.35, 800.0, 0.15);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-12);
    }

    #[test]
    fn torque_realistic_case() {
        // μ=0,35 ; N=800 N ; r=0,15 m → T = 0,35·800·0,15 = 42 N·m.
        let t = friction_brake_torque(0.35, 800.0, 0.15);
        assert_relative_eq!(t, 42.0, epsilon = 1e-9);
    }

    #[test]
    fn torque_scales_linearly_with_normal_force() {
        // T ∝ N : doubler l'effort presseur double le couple de freinage.
        let t1 = friction_brake_torque(0.4, 500.0, 0.2);
        let t2 = friction_brake_torque(0.4, 1000.0, 0.2);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-12);
    }

    #[test]
    fn heat_power_realistic_case() {
        // Couple T=42 N·m à ω=50 rad/s → P = 42·50 = 2100 W.
        let p = friction_brake_heat_power(42.0, 50.0);
        assert_relative_eq!(p, 2100.0, epsilon = 1e-9);
    }

    #[test]
    fn stopping_time_realistic_case_and_energy_identity() {
        // I=2 kg·m², ω=50 rad/s, T=42 N·m → t = 2·50/42 = 100/42 ≈ 2,380952 s.
        let t = friction_brake_stopping_time(2.0, 50.0, 42.0);
        assert_relative_eq!(t, 100.0 / 42.0, epsilon = 1e-12);
        assert_relative_eq!(t, 2.380_952_380_952_381, epsilon = 1e-9);
        // Identité cinétique : ½·I·ω² = ½·T·ω·t (énergie = puissance moyenne · temps).
        let kinetic = 0.5 * 2.0 * 50.0 * 50.0;
        let dissipated = 0.5 * friction_brake_heat_power(42.0, 50.0) * t;
        assert_relative_eq!(kinetic, dissipated, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "dénominateur non nul")]
    fn zero_braking_torque_panics() {
        friction_brake_stopping_time(2.0, 50.0, 0.0);
    }
}
