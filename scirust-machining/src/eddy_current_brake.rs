//! Frein à **courants de Foucault** (freinage sans contact) — couple résistant
//! proportionnel à la vitesse en régime linéaire, puissance dissipée par effet
//! Joule dans le disque et décélération exponentielle du rotor.
//!
//! ```text
//! couple résistant       T = k·ω
//! puissance dissipée      P = T·ω
//! constante de temps      τ = J / k
//! vitesse à l'instant t   ω(t) = ω₀·exp(-t/τ)
//! ```
//!
//! `k` coefficient de traînée magnétique (N·m·s/rad), `ω` vitesse angulaire
//! (rad/s), `T` couple résistant (N·m), `P` puissance dissipée dans le disque
//! (W), `J` moment d'inertie du rotor (kg·m²), `τ` constante de temps de la
//! décélération (s), `ω₀` vitesse angulaire initiale (rad/s), `t` temps écoulé
//! (s). En régime linéaire le couple s'oppose à la rotation proportionnellement
//! à la vitesse ; la puissance électrique dissipée par effet Joule dans le
//! disque vaut `P = T·ω = k·ω²`. Une inertie `J` freinée par un couple `T = k·ω`
//! obéit à `J·dω/dt = -k·ω`, d'où une décroissance exponentielle de constante de
//! temps `τ = J/k` : à `t = τ` la vitesse est tombée à `1/e ≈ 37 %` de `ω₀`.
//!
//! **Convention** : SI cohérent (N·m, rad/s, W, kg·m², s). **Limite honnête** :
//! régime **linéaire** bas/moyen où le couple est proportionnel à la vitesse ; à
//! haute vitesse le couple sature puis décroît (effet de peau, démagnétisation
//! apparente) — non modélisé ici. Le coefficient de traînée magnétique `k` est
//! **fourni par l'appelant** (il dépend de la conductivité du disque, de
//! l'entrefer et du champ magnétique) de même que l'inertie `J` : aucune valeur
//! matériau, géométrie ni champ « par défaut » n'est inventée. Freinage sans
//! usure mais ne maintient pas le rotor à l'arrêt (couple nul à vitesse nulle).

/// Couple résistant `T = k·ω` (N·m) en régime linéaire.
///
/// Couple s'opposant à la rotation, proportionnel à la vitesse angulaire via le
/// coefficient de traînée magnétique fourni.
///
/// Panique si `drag_coefficient < 0` ou `angular_speed < 0`.
pub fn eddybrake_torque(drag_coefficient: f64, angular_speed: f64) -> f64 {
    assert!(
        drag_coefficient >= 0.0,
        "le coefficient de traînée doit être positif ou nul"
    );
    assert!(
        angular_speed >= 0.0,
        "la vitesse angulaire doit être positive ou nulle"
    );
    drag_coefficient * angular_speed
}

/// Puissance dissipée `P = T·ω` (W) par effet Joule dans le disque.
///
/// Puissance mécanique prélevée sur le rotor et convertie en chaleur dans le
/// disque conducteur.
///
/// Panique si `torque < 0` ou `angular_speed < 0`.
pub fn eddybrake_power_dissipated(torque: f64, angular_speed: f64) -> f64 {
    assert!(torque >= 0.0, "le couple doit être positif ou nul");
    assert!(
        angular_speed >= 0.0,
        "la vitesse angulaire doit être positive ou nulle"
    );
    torque * angular_speed
}

/// Constante de temps `τ = J / k` (s) de la décélération exponentielle.
///
/// Temps au bout duquel la vitesse tombe à `1/e ≈ 37 %` de sa valeur initiale
/// pour un rotor d'inertie `inertia` freiné par le coefficient
/// `drag_coefficient`.
///
/// Panique si `inertia <= 0` ou `drag_coefficient <= 0`.
pub fn eddybrake_time_constant(inertia: f64, drag_coefficient: f64) -> f64 {
    assert!(inertia > 0.0, "l'inertie doit être strictement positive");
    assert!(
        drag_coefficient > 0.0,
        "le coefficient de traînée doit être strictement positif"
    );
    inertia / drag_coefficient
}

/// Vitesse angulaire à l'instant `t` : `ω(t) = ω₀·exp(-t/τ)` (rad/s).
///
/// Décroissance exponentielle de la vitesse pour un freinage à couple
/// proportionnel à la vitesse, de constante de temps `time_constant`.
///
/// Panique si `initial_speed < 0`, `elapsed_time < 0` ou `time_constant <= 0`.
pub fn eddybrake_speed_after_time(
    initial_speed: f64,
    elapsed_time: f64,
    time_constant: f64,
) -> f64 {
    assert!(
        initial_speed >= 0.0,
        "la vitesse initiale doit être positive ou nulle"
    );
    assert!(
        elapsed_time >= 0.0,
        "le temps écoulé doit être positif ou nul"
    );
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive"
    );
    initial_speed * (-elapsed_time / time_constant).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::E;

    #[test]
    fn torque_is_proportional_to_speed() {
        // T = k·ω : doubler la vitesse double le couple à coefficient constant.
        let k = 0.5_f64;
        let t1 = eddybrake_torque(k, 100.0);
        let t2 = eddybrake_torque(k, 200.0);
        assert_relative_eq!(t1, 50.0, epsilon = 1e-12);
        assert_relative_eq!(t2 / t1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn power_equals_coefficient_times_speed_squared() {
        // P = T·ω = k·ω² : cohérence entre couple et puissance.
        // k = 0,5 ; ω = 100 → T = 50 N·m, P = 50·100 = 5000 W = 0,5·100².
        let k = 0.5_f64;
        let omega = 100.0_f64;
        let torque = eddybrake_torque(k, omega);
        let power = eddybrake_power_dissipated(torque, omega);
        assert_relative_eq!(power, 5000.0, epsilon = 1e-9);
        assert_relative_eq!(power, k * omega.powi(2), epsilon = 1e-9);
    }

    #[test]
    fn time_constant_is_inertia_over_coefficient() {
        // τ = J/k : J = 2 kg·m², k = 0,5 → τ = 4 s.
        let tau = eddybrake_time_constant(2.0, 0.5);
        assert_relative_eq!(tau, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn speed_falls_to_one_over_e_at_one_time_constant() {
        // ω(τ) = ω₀·exp(-1) = ω₀/e.
        let omega0 = 100.0_f64;
        let tau = 4.0_f64;
        let speed = eddybrake_speed_after_time(omega0, tau, tau);
        assert_relative_eq!(speed, omega0 / E, epsilon = 1e-9);
        assert_relative_eq!(speed, 36.787_944_117_144_23, epsilon = 1e-6);
    }

    #[test]
    fn speed_halves_after_tau_times_ln_two() {
        // ω(τ·ln2) = ω₀·exp(-ln2) = ω₀/2 (demi-vie de la décélération).
        let omega0 = 250.0_f64;
        let tau = 3.0_f64;
        let half_life = tau * core::f64::consts::LN_2;
        let speed = eddybrake_speed_after_time(omega0, half_life, tau);
        assert_relative_eq!(speed, 125.0, epsilon = 1e-9);
    }

    #[test]
    fn speed_at_zero_time_is_initial_speed() {
        // ω(0) = ω₀ : à l'instant initial la vitesse est inchangée.
        let omega0 = 42.0_f64;
        assert_relative_eq!(
            eddybrake_speed_after_time(omega0, 0.0, 5.0),
            omega0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "constante de temps doit être strictement positive")]
    fn zero_time_constant_panics() {
        eddybrake_speed_after_time(100.0, 1.0, 0.0);
    }
}
